"""Prepare independent Town Neck RGB cells without modifying Svalbard data."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import json
import time

import numpy as np
from PIL import Image

DATE = "20160921"
CLASS_NAMES = {1: "Water", 2: "Sand", 3: "Gravel", 4: "Cobble/boulder",
               5: "Vegetated", 6: "Development"}
SOURCE_CLASSES = ["water", "sand", "gravel", "cobble_boulder", "vegetated",
                  "development", "nodata", "unusual"]
# RGB display palette, checked against the supplied integer doodle annotations.
PALETTE = {(51, 102, 204): 1, (220, 57, 18): 2, (255, 153, 0): 3,
           (16, 150, 24): 4, (153, 0, 153): 5, (0, 153, 198): 6,
           (221, 68, 119): 7, (102, 170, 0): 8, (0, 0, 0): 0}


def _digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _geometry(path):
    with Image.open(path) as image:
        tags = image.tag_v2
        scale, tie = tags[33550], tags[33922]
        if image.mode != "RGB" or not np.allclose(scale[:2], [.05, .05]):
            raise ValueError(f"Expected 5 cm RGB GeoTIFF: {path}")
        keys = tags[34735]
        key_values = {keys[i]: keys[i+3] for i in range(4, len(keys), 4)}
        if key_values.get(3072) != 26919 or key_values.get(1025) != 1:
            raise ValueError(f"Expected EPSG:26919 pixel-area georeferencing: {path}")
        return image.height, image.width, tie[3], tie[4]


def prepare_dataset(source_root, output_root, purity_threshold=.9, progress_every=10000):
    """Export 20x20 RGB matrices and a descriptor; do not create dataset splits.

    Cells use column-major IDs with the local grid row count as the stride.
    Existing files from an interrupted run are rewritten deterministically.
    A completed output is not overwritten, to protect later experiments.
    """
    source_root, output_root = Path(source_root), Path(output_root)
    if not isinstance(progress_every, int) or progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")
    if not 0.5 < purity_threshold <= 1:
        raise ValueError("Purity must be greater than 0.5 and at most 1")
    if (output_root / "bs_town_neck_preprocessing_summary.json").exists():
        raise FileExistsError(f"Completed dataset already exists: {output_root}")
    pending = output_root / "bs_town_neck_preprocessing_in_progress.json"
    configuration = {"source": str(source_root.resolve()), "purity_threshold": purity_threshold}
    if output_root.exists() and any(output_root.iterdir()):
        if not pending.exists():
            raise FileExistsError("Non-empty output without an interrupted-run marker; use a new directory")
        previous = json.loads(pending.read_text())
        if any(previous.get(key) != value for key, value in configuration.items()):
            raise ValueError("Interrupted-run settings differ; use a new output directory")
    if source_root.joinpath("classes.txt").read_text().splitlines() != SOURCE_CLASSES:
        raise ValueError("Source class list differs from the verified Dataset A")
    images = sorted(source_root.joinpath("images/cm5").glob("*.tif"))
    if not images:
        raise FileNotFoundError("No 5 cm source GeoTIFFs found")
    geometry = {f.stem: _geometry(f) for f in images}
    xmin = min(g[2] for g in geometry.values())
    ymax = max(g[3] for g in geometry.values())
    offsets = {}
    for key, (h, w, x, y) in geometry.items():
        rr, cc = (ymax-y)/.05, (x-xmin)/.05
        if not np.allclose([rr, cc], np.round([rr, cc]), rtol=0, atol=1e-5):
            raise ValueError("Source tiles do not share the same pixel grid")
        offsets[key] = round(rr), round(cc)
    height = max(offsets[k][0]+g[0] for k,g in geometry.items())
    width = max(offsets[k][1]+g[1] for k,g in geometry.items())
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    palette = {r*65536+g*256+b: code for (r,g,b), code in PALETTE.items()}
    manifest = []
    print(f"Reading {len(images)} aligned RGB/label tiles...", flush=True)
    for image_path in images:
        label_path = source_root / "label_images/cm5" / image_path.name
        if _geometry(label_path) != geometry[image_path.stem]:
            raise ValueError(f"Image/label georeferencing mismatch: {image_path.name}")
        with Image.open(image_path) as im:
            image = np.array(im)
        with Image.open(label_path) as im:
            color = np.array(im).astype(np.uint32)
        packed = color[:,:,0]*65536+color[:,:,1]*256+color[:,:,2]
        unknown = set(map(int, np.unique(packed))) - palette.keys()
        if unknown:
            raise ValueError(f"Unknown label colors {unknown}: {label_path}")
        decoded = np.zeros(packed.shape, dtype=np.uint8)
        for value, code in palette.items():
            decoded[packed == value] = code
        r,c = offsets[image_path.stem]
        h,w = decoded.shape
        region = np.s_[r:r+h,c:c+w]
        if coverage[region].any():
            raise ValueError("Overlapping source tiles must be resolved explicitly")
        rgb[region], labels[region], coverage[region] = image, decoded, True
        for path in (image_path, label_path):
            manifest.append({"path":str(path.relative_to(source_root)), "sha256":_digest(path)})
    # All-black source pixels represent unavailable RGB; labels cannot make them valid.
    valid_rgb = coverage & np.any(rgb != 0, axis=2)
    rows, columns = height//20, width//20
    output_root.mkdir(parents=True, exist_ok=True)
    cell_root = output_root / "beach_all" / DATE
    cell_root.mkdir(parents=True, exist_ok=True)
    configuration["source_manifest_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if pending.exists() and json.loads(pending.read_text()).get(
        "source_manifest_sha256"
    ) != configuration["source_manifest_sha256"]:
        raise ValueError("Interrupted-run source files changed; use a new output directory")
    pending.write_text(json.dumps(configuration)+"\n")
    stats = Counter()
    by_class = Counter()
    started = time.perf_counter()
    descriptor_path = output_root / "Descriptor_bs_town_neck.csv"
    audit_path = output_root / "bs_town_neck_cell_inventory.csv"
    numbers = [str(i).encode("ascii") for i in range(256)]
    with descriptor_path.open("w", newline="") as df, audit_path.open("w", newline="") as af:
        descriptor, audit = csv.writer(df), csv.writer(af)
        descriptor.writerow(["date","cell_id","class"])
        audit.writerow(["date","cell_id","row","column","easting_center","northing_center",
                        "valid_rgb_pixels","valid_label_pixels","dominant_class","dominant_pixels",
                        "purity","selected","exclusion_reason"])
        for column in range(columns):
            for row in range(rows):
                cell_id = column*rows+row+1
                region = np.s_[row*20:(row+1)*20,column*20:(column+1)*20]
                cell_rgb, cell_label, available = rgb[region], labels[region], valid_rgb[region]
                histogram = np.bincount(cell_label.ravel(), minlength=9)
                valid_count = int(histogram[1:7].sum())
                best = int(histogram[1:7].argmax())+1
                best_count = int(histogram[best])
                available_count = int(available.sum())
                if available_count < 400:
                    reason = "incomplete_rgb"
                elif valid_count < 400:
                    reason = "invalid_or_ambiguous_labels"
                elif best_count < int(np.ceil(400*purity_threshold)):
                    reason = "mixed_classes"
                else:
                    reason = ""
                selected = not reason
                if available_count:
                    # Preserve every available cell as context, including excluded targets.
                    for channel, suffix in enumerate("rgb"):
                        matrix = cell_rgb[:,:,channel]
                        lines = []
                        for values, good in zip(matrix,available):
                            lines.append(b",".join(numbers[int(v)] if ok else b"nan"
                                                  for v,ok in zip(values,good)))
                        (cell_root / f"{cell_id}{suffix}.txt").write_bytes(b"\n".join(lines)+b"\n")
                    stats["context_cells"] += 1
                if selected:
                    descriptor.writerow([DATE,cell_id,best])
                    stats["selected_cells"] += 1
                    by_class[best] += 1
                else:
                    stats[reason] += 1
                audit.writerow([DATE,cell_id,row,column,xmin+column+.5,ymax-row-.5,
                                available_count,valid_count,best,best_count,best_count/400,
                                int(selected),reason])
                if cell_id % progress_every == 0:
                    print(f"Prepared {cell_id:,}/{rows*columns:,} grid cells; "
                          f"selected {stats['selected_cells']:,}", flush=True)
    metadata = {
        "dataset":"Town Neck Beach Dataset A", "date":DATE, "channel_setup":"RGB_only",
        "feature_suffixes":["r","g","b"], "cell_matrix_size":20, "pixel_size_m":.05,
        "cell_size_m":1, "crs":"EPSG:26919", "grid_rows":rows,"grid_columns":columns,
        "cell_id_formula":"column * grid_rows + row + 1", "row_direction":"south",
        "column_direction":"east", "row_column_indexing":"zero-based",
        "mosaic_origin_upper_left":[xmin,ymax], "mosaic_shape_pixels":[height,width],
        "discarded_bottom_rows_pixels":height%20,"discarded_right_columns_pixels":width%20,
        "purity_threshold":purity_threshold,"require_valid_label_pixels":400,
        "require_valid_rgb_pixels":400, "rgb_missing_rule":"all three channels zero",
        "missing_value":"nan", "normalization":"None; calculate using future training partition only",
        "target_classes":CLASS_NAMES,"excluded_label_codes":[0,7,8],
        "label_rgb_to_code":{','.join(map(str,k)):v for k,v in PALETTE.items()},
        "splits_created":False,
        "split_requirement":"Spatial blocks with disjoint context footprints; no survey-date split",
        "loader_requirement":"Use this grid geometry, not hardcoded Svalbard GRID_ROWS/GRID_COLUMNS",
        "source_dataset_doi":"10.5061/dryad.2fqz612ps",
        "label_provenance":"Doodler dense labels generated from sparse human annotations",
        "source_manifest":manifest,
    }
    (output_root / "bs_town_neck_grid_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    # Match the existing class-name reader's named-column Excel convention.
    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Classes"
    sheet.append(["class","name"])
    for code,name in CLASS_NAMES.items():
        sheet.append([code,name])
    workbook.save(output_root / "Classes.xlsx")
    actual_files = sum(1 for p in cell_root.iterdir() if p.suffix == ".txt")
    if actual_files != stats["context_cells"]*3:
        raise ValueError("Output matrix file count differs from expected count")
    summary = {"grid_cells":rows*columns, **dict(stats), "matrix_files":actual_files,
               "selected_cells_by_class":{CLASS_NAMES[k]:by_class[k] for k in CLASS_NAMES},
               "elapsed_seconds":time.perf_counter()-started,
               "descriptor_sha256":_digest(descriptor_path), "inventory_sha256":_digest(audit_path)}
    (output_root / "bs_town_neck_preprocessing_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    pending.unlink()
    print(json.dumps(summary,indent=2),flush=True)
    return summary
