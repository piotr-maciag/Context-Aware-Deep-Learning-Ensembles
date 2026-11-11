"""Reproducible sampled spatial splits for Town Neck, independent of Svalbard."""

from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

DEFAULT_SPLIT_SIZES = {"train": 5386, "validation": 1243, "test": 1493}
COLUMNS = ["date", "cell_id", "class", "split"]


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _allocate(total, weights):
    """Largest-remainder allocation keeps the requested total exact."""
    exact = np.asarray(weights, dtype=float)
    exact = total * exact / exact.sum()
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact-counts), kind="stable")
    counts[order[:total-counts.sum()]] += 1
    return counts


def context_footprint(frame, rows, columns, radius):
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Context radius must be a non-negative integer")
    if rows <= 0 or columns <= 0 or not frame.cell_id.between(1, rows*columns).all():
        raise ValueError("Invalid grid dimensions or out-of-bounds cell IDs")
    ids = frame.cell_id.to_numpy(dtype=np.int64)-1
    rr, cc = ids % rows, ids // rows
    footprint = np.zeros((rows, columns), dtype=bool)
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            r, c = rr+dr, cc+dc
            valid = (r >= 0) & (r < rows) & (c >= 0) & (c < columns)
            footprint[r[valid], c[valid]] = True
    return footprint


def validate_splits(frames, source, rows, columns, radius, split_sizes):
    if set(frames) != {"train", "validation", "test"} or set(split_sizes) != set(frames):
        raise ValueError("Expected all three train/validation/test partitions")
    if source.date.nunique() != 1 or source.cell_id.duplicated().any():
        raise ValueError("Expected unique raw cell IDs from one survey")
    source_labels = source.set_index("cell_id")["class"]
    classes = set(source["class"])
    footprints = {}
    for name, frame in frames.items():
        if list(frame.columns) != COLUMNS or len(frame) != split_sizes[name]:
            raise ValueError(f"Unexpected schema or size for {name}")
        if frame.duplicated(["date", "cell_id"]).any() or not frame.split.eq(name).all():
            raise ValueError(f"Duplicate or invalid records in {name}")
        if set(frame["class"]) != classes or set(frame.date) != set(source.date):
            raise ValueError(f"Missing classes or unexpected date in {name}")
        if not np.array_equal(source_labels.reindex(frame.cell_id), frame["class"]):
            raise ValueError(f"Target labels differ from raw descriptor in {name}")
        footprints[name] = context_footprint(frame, rows, columns, radius)
    names = list(frames)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i+1:]:
            overlap = int(np.count_nonzero(footprints[left] & footprints[right]))
            overlaps[f"{left}/{right}"] = overlap
            if overlap:
                raise ValueError(f"Context overlap between {left} and {right}: {overlap}")
    return footprints, overlaps


def create_sampled_splits(raw_root, output_root, split_sizes=None, random_seed=42,
                          block_size=32, max_context_radius=3):
    raw_root, output_root = Path(raw_root), Path(output_root)
    split_sizes = dict(DEFAULT_SPLIT_SIZES if split_sizes is None else split_sizes)
    if list(split_sizes) != ["train", "validation", "test"] or any(
        not isinstance(n, int) or n <= 0 for n in split_sizes.values()
    ):
        raise ValueError("Specify positive integer train/validation/test sizes, in that order")
    if (not isinstance(max_context_radius, int) or not isinstance(block_size, int)
            or max_context_radius < 0 or block_size <= 2*max_context_radius):
        raise ValueError("Block size must exceed twice the maximum context radius")
    descriptor_path = raw_root / "Descriptor_bs_town_neck.csv"
    metadata_path = raw_root / "bs_town_neck_grid_metadata.json"
    summary_path = raw_root / "bs_town_neck_preprocessing_summary.json"
    if (raw_root / "bs_town_neck_preprocessing_in_progress.json").exists() or not summary_path.exists():
        raise ValueError("Raw preprocessing must finish before creating splits")
    summary = json.loads(summary_path.read_text())
    if _hash(descriptor_path) != summary["descriptor_sha256"]:
        raise ValueError("Raw descriptor changed since preprocessing; verify the prepared dataset")
    source = pd.read_csv(descriptor_path, dtype={"date": str, "cell_id": np.int64, "class": np.int64})
    if source.duplicated(["date", "cell_id"]).any() or source.date.nunique() != 1:
        raise ValueError("Expected unique labelled cells from one Town Neck survey")
    metadata = json.loads(metadata_path.read_text())
    rows, columns = metadata["grid_rows"], metadata["grid_columns"]
    if not source.cell_id.between(1, rows*columns).all():
        raise ValueError("Raw descriptor IDs exceed the declared grid")
    classes = np.sort(source["class"].unique())
    frequencies = source["class"].value_counts().reindex(classes).to_numpy()
    quotas = {name: _allocate(size, frequencies) for name, size in split_sizes.items()}
    if any(np.any(q == 0) for q in quotas.values()):
        raise ValueError("Requested sample too small to represent every class proportionally")
    candidates = source.copy()
    candidates["row"] = (candidates.cell_id-1) % rows
    candidates["column"] = (candidates.cell_id-1) // rows
    br, bc = candidates.row//block_size, candidates.column//block_size
    block_columns = (columns+block_size-1)//block_size
    candidates["block_id"] = br*block_columns+bc
    # Erode each block by r cells. Context reaches the block edge but never crosses it.
    r = max_context_radius
    safe = ((candidates.row >= br*block_size+r)
            & (candidates.row < np.minimum((br+1)*block_size, rows)-r)
            & (candidates.column >= bc*block_size+r)
            & (candidates.column < np.minimum((bc+1)*block_size, columns)-r))
    candidates = candidates.loc[safe].copy()
    blocks = np.sort(candidates.block_id.unique())
    block_counts = _allocate(len(blocks), list(split_sizes.values()))
    rng = np.random.default_rng(random_seed)
    for attempt in range(1000):
        assigned = np.split(rng.permutation(blocks), np.cumsum(block_counts)[:-1])
        assignments = dict(zip(split_sizes, assigned))
        pools = {name: candidates.loc[candidates.block_id.isin(ids)]
                 for name, ids in assignments.items()}
        if all(np.all(pool["class"].value_counts().reindex(classes, fill_value=0).to_numpy()
                      >= quotas[name]) for name, pool in pools.items()):
            break
    else:
        raise ValueError("No block assignment supplies all class quotas; adjust block size or sample sizes")
    frames, details = {}, []
    # Separate RNG keeps target sampling independent of the number of partition attempts.
    sample_rng = np.random.default_rng(random_seed)
    for name, pool in pools.items():
        pieces = []
        for code, count in zip(classes, quotas[name]):
            group = pool.loc[pool["class"].eq(code)].sort_values("cell_id")
            indices = sample_rng.choice(len(group), size=int(count), replace=False)
            pieces.append(group.iloc[indices])
        sampled = pd.concat(pieces).sort_values("cell_id").assign(split=name)
        details.append(sampled)
        frames[name] = sampled[COLUMNS].reset_index(drop=True)
    footprints, overlaps = validate_splits(frames, source, rows, columns, r, split_sizes)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in frames.items():
        files[name] = output_root / f"bs_{name}_split.csv"
        frame.to_csv(files[name], index=False)
    files["all"] = output_root / "bs_dataset_splits.csv"
    pd.concat(frames.values(), ignore_index=True).to_csv(files["all"], index=False)
    pd.concat(details, ignore_index=True).to_csv(output_root / "bs_town_neck_sampled_cells.csv", index=False)
    assignment_frame = pd.DataFrame([(int(b), name) for name, ids in assignments.items() for b in ids],
                                    columns=["block_id", "split"]).sort_values("block_id")
    assignment_frame["block_row"] = assignment_frame.block_id//block_columns
    assignment_frame["block_column"] = assignment_frame.block_id%block_columns
    assignment_frame.to_csv(output_root / "bs_town_neck_spatial_blocks.csv", index=False)
    distribution = pd.DataFrame({name: frame["class"].value_counts().reindex(classes)
                                 for name, frame in frames.items()}).astype(int)
    distribution.index.name = "class"
    distribution.insert(0, "class_name", [metadata["target_classes"][str(c)] for c in classes])
    distribution["total"] = distribution[list(split_sizes)].sum(axis=1)
    distribution.to_csv(output_root / "bs_town_neck_class_counts.csv")
    provenance = {
        "policy": "town_neck_stratified_sample_spatial_blocks_v1", "random_seed":random_seed,
        "raw_dataset": "Data/Raw/bs_town_neck", "channels":metadata["feature_suffixes"],
        "grid_rows":rows, "grid_columns":columns, "max_context_radius":r,
        "block_size_cells":block_size, "block_size_m":block_size,
        "block_edge_exclusion_cells":r, "block_assignment_attempt":attempt+1,
        "block_assignment_criterion":"Class quota availability only, never model performance",
        "sampling":"Without replacement, proportional to the full eligible descriptor's class counts",
        "requested_split_sizes":split_sizes, "total_sampled_targets":sum(split_sizes.values()),
        "eligible_raw_targets":len(source), "targets_after_block_buffers":len(candidates),
        "context_overlap_cells":overlaps,
        "context_footprint_cells":{name:int(mask.sum()) for name,mask in footprints.items()},
        "class_counts":{name:{str(c):int(n) for c,n in zip(classes,quotas[name])} for name in frames},
        "raw_descriptor_sha256":_hash(descriptor_path), "raw_grid_metadata_sha256":_hash(metadata_path),
        "csv_sha256":{name:_hash(path) for name,path in files.items()},
        "imputation":"Not applied; future normalization and missing-value means must use training only",
        "notes":["Block-wise holdout within a single survey, not survey-date or cross-beach holdout",
                 "All radii 0, 1, 2, 3 share these exact targets and partitions",
                 "Read available RGB context from the full raw dataset; missing context needs training means",
                 "Existing Svalbard training/split validation helpers require separate Town Neck configuration"],
    }
    (output_root / "bs_split_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(distribution.to_string(), flush=True)
    print(f"Selected {sum(split_sizes.values()):,} targets. Radius-{r} cross-split context overlaps: {overlaps}", flush=True)
    return frames, distribution, provenance
