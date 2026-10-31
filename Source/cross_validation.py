"""Five spatial folds, single validation splits, and all-data deployment fitting."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader

from Source.beach_dataset import (
    BeachCellDataset, FEATURE_SUFFIXES, class_weights, compute_channel_stats,
    make_class_mapping, cell_id_to_grid_coordinates,
)
from Source.grid_search import _run_single_grid_config
from Source.timing import elapsed_seconds
from Source.training import classification_metrics, predict, train_one_epoch


PROTOCOL = "spatial_five_fold_single_validation_v3"
ENSEMBLE_METHODS = (
    "hard_voting", "probability_voting", "weighted_probability_per_model",
    "weighted_probability_per_model_class",
)


def _channel_configuration(feature_suffixes, channel_setup=None):
    from Source.channel_setups import CHANNEL_SETUPS, get_channel_setup

    if feature_suffixes is None:
        setup_name, feature_suffixes = get_channel_setup(channel_setup)
    else:
        if isinstance(feature_suffixes, str):
            raise ValueError("feature_suffixes must be a sequence of channel suffixes, not a string")
        feature_suffixes = tuple(feature_suffixes)
        setup_name = next((name for name, suffixes in CHANNEL_SETUPS.items()
                           if tuple(suffixes) == feature_suffixes), None)
        if channel_setup is not None:
            expected_name, expected_suffixes = get_channel_setup(channel_setup)
            if feature_suffixes != tuple(expected_suffixes):
                raise ValueError(f"feature_suffixes do not match channel setup {expected_name}")
            setup_name = expected_name
    feature_suffixes = tuple(feature_suffixes)
    if (not feature_suffixes or len(set(feature_suffixes)) != len(feature_suffixes)
            or not set(feature_suffixes).issubset(FEATURE_SUFFIXES)):
        raise ValueError("feature_suffixes must contain unique, supported channel suffixes")
    return {"channel_setup": setup_name, "feature_suffixes": list(feature_suffixes),
            "in_channels": len(feature_suffixes)}


def _check_channel_configuration(metadata, channels, source, *, legacy=False):
    # Pre-ablation completed manifests did not record channels; they used all five.
    suffixes = metadata.get("feature_suffixes", FEATURE_SUFFIXES if legacy else ())
    if (tuple(suffixes) != tuple(channels["feature_suffixes"])
            or metadata.get("in_channels", len(suffixes)) != channels["in_channels"]
            or (metadata.get("channel_setup") is not None
                and metadata["channel_setup"] != channels["channel_setup"])):
        raise ValueError(f"{source}: incompatible channel configuration; select matching model result directories")


def fingerprint(frame, columns=None):
    columns = columns or [c for c in ("date", "cell_id", "class", "spatial_group", "max_context_radius", "outer_fold", "split") if c in frame]
    records = frame[columns].astype(str).sort_values(columns).to_dict("records")
    return sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _validate_radius(radius):
    if int(radius) != radius or radius < 0:
        raise ValueError("Context radius must be a non-negative integer")
    return int(radius)


def spatial_groups(frame, max_context_radius):
    """Connect same-survey centres whose maximum-radius mosaics intersect."""
    radius = _validate_radius(max_context_radius)
    result = pd.Series(index=frame.index, dtype="string")
    for date, part in frame.groupby("date", sort=True):
        coordinates = np.array([cell_id_to_grid_coordinates(cell) for cell in part.cell_id])
        # DBSCAN with min_samples=1 yields connected components, including singletons.
        labels = (DBSCAN(eps=2 * radius, min_samples=1, metric="chebyshev").fit_predict(coordinates)
                  if radius else np.arange(len(part)))
        result.loc[part.index] = [f"{date}:{label}" for label in labels]
    return result


def buffered_training_partition(candidates, heldout, max_context_radius):
    """Remove training centres at Chebyshev distance <= 2r from held-out centres."""
    radius = _validate_radius(max_context_radius)
    excluded = pd.Series(False, index=candidates.index)
    for date, part in candidates.groupby("date", sort=False):
        evaluation = heldout.loc[heldout.date == date]
        if evaluation.empty:
            continue
        a = np.array([cell_id_to_grid_coordinates(cell) for cell in part.cell_id])
        b = np.array([cell_id_to_grid_coordinates(cell) for cell in evaluation.cell_id])
        distances = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
        excluded.loc[part.index] = (distances <= 2 * radius).any(axis=1)
    return candidates.loc[~excluded].copy(), candidates.loc[excluded].copy()


def _check_partition(training, evaluation, expected_classes, radius):
    if set(training["class"]) != expected_classes or set(evaluation["class"]) != expected_classes:
        raise ValueError("Every training and evaluation partition must contain all classes")
    if set(training.spatial_group) & set(evaluation.spatial_group):
        raise ValueError("Spatial groups cross a training/evaluation boundary")
    _, overlapping = buffered_training_partition(training, evaluation, radius)
    if not overlapping.empty:
        raise ValueError("Training and evaluation contexts overlap within a survey")


def _grouped_partitions(frame, n_splits, seed, radius):
    classes = set(frame["class"])
    if n_splits < 2 or frame.groupby("class").spatial_group.nunique().min() < n_splits:
        raise ValueError("Insufficient independent spatial groups per class for requested folds")
    for attempt in range(100):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + attempt)
        partitions = list(splitter.split(frame, frame["class"], frame.spatial_group))
        try:
            for training, evaluation in partitions:
                fit, _ = buffered_training_partition(frame.iloc[training], frame.iloc[evaluation], radius)
                _check_partition(fit, frame.iloc[evaluation], classes, radius)
        except ValueError:
            continue
        return partitions
    raise ValueError("Cannot construct class-complete spatial folds; reduce fold count or add annotations")


def outer_training_frame(samples, outer, max_context_radius):
    if outer == 0:
        return samples.copy()
    heldout = samples.loc[samples.outer_fold == outer]
    training, _ = buffered_training_partition(samples.loc[samples.outer_fold != outer], heldout, max_context_radius)
    _check_partition(training, heldout, set(samples["class"]), max_context_radius)
    return training


def create_cv_plan(frame, outer_folds=5, random_seed=42, max_context_radius=3,
                   validation_fraction=0.2):
    radius = _validate_radius(max_context_radius)
    samples = frame.drop(columns=["split", "outer_fold", "inner_fold", "spatial_group", "max_context_radius"], errors="ignore").copy()
    for column in ("date", "cell_id", "class"):
        samples[column] = samples[column].astype(str).str.strip()
        if samples[column].isin(["", "nan", "<NA>"]).any():
            raise ValueError(f"Invalid {column} in CV data")
    if samples.duplicated(["date", "cell_id"]).any():
        raise ValueError("Duplicate date/cell_id pairs in CV data")
    samples = samples.sort_values(["date", "cell_id"]).reset_index(drop=True)
    samples["spatial_group"] = spatial_groups(samples, radius)
    samples["max_context_radius"] = radius
    samples["outer_fold"] = 0
    for fold, (_, heldout) in enumerate(_grouped_partitions(samples, outer_folds, random_seed, radius), 1):
        samples.loc[heldout, "outer_fold"] = fold
    if not 0 < validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5]")
    validation_rows = []
    for outer in range(1, outer_folds + 1):
        training = outer_training_frame(samples, outer, radius)
        # Choose one class-complete grouped holdout, not a rotating inner CV.
        partitions = _grouped_partitions(training, max(2, round(1 / validation_fraction)),
                                         random_seed + outer + 1, radius)
        fit_indices, validation_indices = min(
            partitions, key=lambda pair: abs(len(pair[1]) / len(training) - validation_fraction)
        )
        for split, indices in (("training", fit_indices), ("validation", validation_indices)):
            part = training.iloc[indices].copy()
            part["outer_fold"] = outer
            part["split"] = split
            validation_rows.append(part)
    return samples, pd.concat(validation_rows, ignore_index=True)


def save_cv_plan(samples, inner, output_dir, random_seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "gs_cv_samples.csv", index=False)
    inner.to_csv(output_dir / "gs_cv_validation_splits.csv", index=False)
    manifest = {
        "protocol": PROTOCOL, "random_seed": random_seed,
        # Retain the shared artifact-schema key; it now hashes validation splits, not inner folds.
        "sample_hash": fingerprint(samples), "inner_hash": fingerprint(inner),
        "outer_folds": int(samples.outer_fold.max()),
        "validation_splits_per_outer_fold": 1,
        "selection_metric": "mean_validation_macro_f1_across_outer_runs",
        "evaluation_note": "Global configuration selection across folds is not independent nested CV.",
        "max_context_radius": int(samples.max_context_radius.iloc[0]),
        "buffer_distance_cells": 2 * int(samples.max_context_radius.iloc[0]),
        "grouping": "same_survey_connected_context_components",
        "distance_metric": "chebyshev",
    }
    write_json(output_dir / "gs_cv_manifest.json", manifest)
    return manifest


def read_cv_plan(directory):
    directory = Path(directory)
    samples = pd.read_csv(directory / "gs_cv_samples.csv", dtype={"date": str, "cell_id": str, "class": str})
    manifest = json.loads((directory / "gs_cv_manifest.json").read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Old CV protocol; regenerate the plan with gs_create_dataset_splits.ipynb")
    inner = pd.read_csv(directory / "gs_cv_validation_splits.csv", dtype={"date": str, "cell_id": str, "class": str})
    if manifest["protocol"] != PROTOCOL or fingerprint(samples) != manifest["sample_hash"] or fingerprint(inner) != manifest["inner_hash"]:
        raise ValueError("CV manifest does not match samples/folds; regenerate the CV plan")
    outer_count = manifest["outer_folds"]
    if manifest.get("validation_splits_per_outer_fold") != 1:
        raise ValueError("Expected one training/validation split per outer fold")
    radius = _validate_radius(manifest["max_context_radius"])
    if (manifest.get("buffer_distance_cells") != 2 * radius
            or manifest.get("grouping") != "same_survey_connected_context_components"
            or manifest.get("distance_metric") != "chebyshev"
            or set(samples.max_context_radius) != {radius}
            or set(inner.max_context_radius) != {radius}
            or not spatial_groups(samples, radius).astype(str).equals(samples.spatial_group.astype(str))):
        raise ValueError("CV spatial groups or buffer policy do not match the saved plan")
    if set(samples.outer_fold) != set(range(1, outer_count + 1)) or samples.duplicated(["date", "cell_id"]).any():
        raise ValueError("Invalid outer-fold assignments")
    if not samples.groupby("spatial_group").outer_fold.nunique().eq(1).all():
        raise ValueError("A spatial group crosses outer folds")
    if set(inner.outer_fold) != set(range(1, outer_count + 1)):
        raise ValueError("Missing validation split assignments")
    for outer in range(1, outer_count + 1):
        expected = outer_training_frame(samples, outer, radius)
        current = inner.loc[inner.outer_fold == outer]
        if not current.groupby("spatial_group").split.nunique().eq(1).all():
            raise ValueError(f"A spatial group crosses training/validation for outer fold {outer}")
        if (set(current.split) != {"training", "validation"}
                or current.duplicated(["date", "cell_id"]).any()
                or fingerprint(current, ["date", "cell_id", "class", "spatial_group", "max_context_radius"]) != fingerprint(expected, ["date", "cell_id", "class", "spatial_group", "max_context_radius"])):
            raise ValueError(f"Invalid validation partition for outer fold {outer}")
        evaluation = current.loc[current.split == "validation"]
        training, _ = buffered_training_partition(current.loc[current.split == "training"], evaluation, radius)
        _check_partition(training, evaluation, set(samples["class"]), radius)
    return samples, inner, manifest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def metric_row(truth, prediction, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(truth, prediction, labels=labels, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(truth, prediction), "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1, "mcc": matthews_corrcoef(truth, prediction)}


def summarize_folds(frame):
    metrics = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc"}
    columns = [c for c in frame if (c in metrics or c.endswith("_seconds")) and pd.api.types.is_numeric_dtype(frame[c])]
    return frame.groupby("method")[columns].agg(["mean", "std"]).reset_index()


def _class_metrics(report, names, outer, method):
    rows = []
    for index, name in names.items():
        values = report.loc[name]
        rows.append({"method": method, "outer_fold": outer, "class_index": index,
                     "class_name": name, "precision": values["precision"],
                     "recall": values["recall"], "f1": values["f1-score"]})
    return rows


def _save_class_summaries(rows, destination):
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "fold_per_class_metrics.csv", index=False)
    frame.groupby(["method", "class_index", "class_name"])[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(destination / "cv_per_class_summary.csv")
    return frame


def _prediction_frame(frame, truth, probabilities, mapping):
    result = frame.reset_index(drop=True).copy()
    reverse = {v: k for k, v in mapping.items()}
    result["true_class_code"] = [reverse[int(x)] for x in truth]
    result["predicted_class_code"] = [reverse[int(x)] for x in probabilities.argmax(1)]
    for code, index in mapping.items():
        result[f"probability_class_{code}"] = probabilities[:, index]
    return result


def run_spatial_cv(
    *, model_factory, parameter_grid, cv_dir, cell_data_root, output_dir,
    model_type, class_name_by_code, image_size=224, model_name=None,
    batch_size=32, num_workers=8, epochs=50, early_stopping_patience=5,
    early_stopping_min_delta=1e-4, augment=True, device=None, random_seed=42,
    feature_suffixes=None, channel_setup=None,
):
    """Evaluate each configuration in five single-split runs, then refit once.

    Omitted channels use channel_setup or BEACH_CHANNEL_SETUP. The caller's
    model_factory(config) must use the selected input-channel count. output_dir
    is used as supplied.
    """
    from Source.training import get_device

    channels = _channel_configuration(feature_suffixes, channel_setup)
    feature_suffixes = tuple(channels["feature_suffixes"])
    device = device or get_device()
    output_dir, cell_data_root = Path(output_dir), Path(cell_data_root)
    cv_root = output_dir / "cross_validation"
    cv_root.mkdir(parents=True, exist_ok=True)
    # A failed rerun cannot leave an apparently completed previous experiment.
    (cv_root / "completed.json").unlink(missing_ok=True)
    samples, inner, manifest = read_cv_plan(cv_dir)
    if int(random_seed) != int(manifest["random_seed"]):
        raise ValueError("Training seed must match the saved CV plan")
    grid = list(parameter_grid)
    if not grid:
        raise ValueError("Parameter grid is empty")
    maximum_radius = manifest["max_context_radius"]
    if any(_validate_radius(config["context_radius"]) > maximum_radius for config in grid):
        raise ValueError("Grid context radius exceeds the saved spatial buffer; regenerate the CV plan")
    experiment = {**manifest, **channels, "run_id": uuid4().hex, "model_type": model_type, "parameter_grid": grid,
                  "image_size": image_size, "epochs": epochs, "batch_size": batch_size,
                  "augmentation": augment, "early_stopping_patience": early_stopping_patience,
                  "early_stopping_min_delta": early_stopping_min_delta}
    write_json(cv_root / "experiment.json", experiment)
    mapping = make_class_mapping(samples)
    names = {index: f"{code}: {class_name_by_code.get(code, code)}" for code, index in mapping.items()}
    fold_rows, heldout_frames, class_rows = [], [], []

    def dataset(frame, config, mean, std, training=False):
        return BeachCellDataset(frame, cell_data_root, mapping, image_size=image_size,
                                feature_suffixes=feature_suffixes,
                                mean=mean, std=std, augment=training and augment,
                                context_radius=int(config["context_radius"]))

    def loader(frame, config, mean, std, training=False):
        return DataLoader(dataset(frame, config, mean, std, training), batch_size=batch_size,
                          shuffle=training, num_workers=num_workers)

    def search(outer, destination):
        destination.mkdir(parents=True, exist_ok=True)
        partition = inner.loc[inner.outer_fold == outer]
        validation = partition.loc[partition.split == "validation"].reset_index(drop=True)
        fit, excluded = buffered_training_partition(
            partition.loc[partition.split == "training"], validation, maximum_radius
        )
        fit = fit.reset_index(drop=True)
        heldout = samples.loc[samples.outer_fold == outer].reset_index(drop=True)
        _check_partition(fit, validation, set(mapping), maximum_radius)
        _check_partition(fit, heldout, set(mapping), maximum_radius)
        _check_partition(validation, heldout, set(mapping), maximum_radius)
        fit.to_csv(destination / "training_cells.csv", index=False)
        validation.to_csv(destination / "validation_cells.csv", index=False)
        heldout.to_csv(destination / "evaluation_cells.csv", index=False)
        with elapsed_seconds(device) as preparation:
            mean, std = compute_channel_stats(fit, cell_data_root,
                                               feature_suffixes=feature_suffixes, context_radius=0)
        pd.DataFrame({"feature": feature_suffixes, "mean": mean, "std": std}).to_csv(
            destination / "normalization_stats.csv", index=False)
        pd.DataFrame([{"training_cells": len(fit), "validation_cells": len(validation),
                       "test_cells": len(heldout), "buffer_excluded_cells": len(excluded)}]).to_csv(
            destination / "partition_summary.csv", index=False)
        gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
        devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)] or [device]
        queue = Queue()
        for item in devices:
            queue.put(item)

        def candidate(config_index):
            assigned_device = queue.get()
            try:
                config = grid[config_index]
                target = destination / f"config_{config_index + 1}"
                target.mkdir(parents=True, exist_ok=True)
                print(f"{model_type}: fold {outer}, grid {config_index + 1}/{len(grid)} on {assigned_device}", flush=True)
                result = _run_single_grid_config(
                    run_index=config_index + 1, config=config, model_factory=model_factory,
                    train_loader=loader(fit, config, mean, std, True),
                    validation_loader=loader(validation, config, mean, std), loader_factory=None,
                    class_weights=class_weights(fit, mapping), idx_to_class=names,
                    device=assigned_device, epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    random_seed=random_seed + 100 * outer, quiet=True,
                )
                _prediction_frame(validation, result["y_true"], result["y_probability"], mapping).to_csv(
                    target / "validation_predictions.csv", index=False)
                model = model_factory(config).to(assigned_device)
                model.load_state_dict(result["state_dict"])
                with elapsed_seconds(assigned_device) as test_time:
                    truth, prediction, probabilities = predict(
                        model, loader(heldout, config, mean, std), assigned_device, show_progress=False)
                _prediction_frame(heldout, truth, probabilities, mapping).to_csv(
                    target / "test_predictions.csv", index=False)
                report, confusion = classification_metrics(truth, prediction, names)
                report.to_csv(target / "test_classification_report.csv")
                confusion.to_csv(target / "test_confusion_matrix.csv")
                result["report"].to_csv(target / "validation_classification_report.csv")
                history = result["history"].assign(config_index=config_index, outer_fold=outer)
                history.to_csv(target / "training_history.csv", index=False)
                row = {**result["summary_row"], "config_index": config_index, "outer_fold": outer,
                       "method": model_type, "prediction_seconds": test_time["seconds"],
                       "preprocessing_seconds": preparation["seconds"],
                       **{f"validation_{k}": v for k, v in metric_row(
                           result["y_true"], result["y_probability"].argmax(1), list(range(len(mapping)))).items()},
                       **{f"test_{k}": v for k, v in metric_row(truth, prediction, list(range(len(mapping)))).items()}}
                pd.DataFrame([row]).to_csv(target / "metrics_and_timing.csv", index=False)
                per_class = []
                for split, current_report in (("validation", result["report"]), ("test", report)):
                    per_class.extend({**item, "split": split, "config_index": config_index}
                                     for item in _class_metrics(current_report, names, outer, model_type))
                del model
                del result
                print(f"Finished {model_type} fold {outer}, grid {config_index + 1}: validation F1={row['validation_macro_f1']:.4f}", flush=True)
                return row, history, per_class
            finally:
                queue.put(assigned_device)

        with elapsed_seconds(device) as search_time:
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                results = list(pool.map(candidate, range(len(grid))))
        trace = pd.DataFrame([r[0] for r in results])
        trace.to_csv(destination / "grid_search_trace.csv", index=False)
        pd.concat([r[1] for r in results], ignore_index=True).to_csv(destination / "epoch_trace.csv", index=False)
        pd.DataFrame([item for r in results for item in r[2]]).to_csv(destination / "per_class_configuration_metrics.csv", index=False)
        write_json(destination / "search_timing.json", {"search_wall_seconds": search_time["seconds"],
                   "training_work_seconds": float(trace.training_seconds.sum()),
                   "preprocessing_seconds": preparation["seconds"]})
        return trace

    def refit(training, config, refit_epochs, destination, outer):
        destination.mkdir(parents=True, exist_ok=True)
        with elapsed_seconds(device) as preparation:
            mean, std = compute_channel_stats(training, cell_data_root,
                                              feature_suffixes=feature_suffixes, context_radius=0)
        torch.manual_seed(random_seed + outer)
        model = model_factory(config).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights(training, mapping).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        train_loader = loader(training, config, mean, std, True)
        history = []
        with elapsed_seconds(device) as fitting:
            for epoch in range(1, refit_epochs + 1):
                scores = train_one_epoch(model, train_loader, criterion, optimizer, device, show_progress=False)
                history.append({"epoch": epoch, **scores})
        checkpoint = destination / f"{model_type}_best.pt"
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint)
        from Source.bs_splits import file_sha256
        metadata = {
            "model_type": model_type, "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": file_sha256(checkpoint),
            "image_size": image_size if image_size is not None else 20 * (2 * int(config["context_radius"]) + 1),
            "context_radius": int(config["context_radius"]),
            "context_window_size": 2 * int(config["context_radius"]) + 1,
            **channels,
            "num_classes": len(mapping), "class_to_idx": mapping,
            "idx_to_class_code": {str(v): k for k, v in mapping.items()},
            "class_name_by_code": class_name_by_code, "best_config": config,
            "channel_mean": mean.tolist(), "channel_std": std.tolist(),
            "protocol": PROTOCOL, "outer_fold": outer, "cv_sample_hash": manifest["sample_hash"],
            "max_context_radius": maximum_radius,
            "buffer_distance_cells": manifest["buffer_distance_cells"],
            "run_id": experiment["run_id"],
            "cv_inner_hash": manifest["inner_hash"], "refit_epochs": refit_epochs,
            "training_sample_hash": fingerprint(training, ["date", "cell_id", "class"]),
            "purpose": "deployment" if outer == 0 else "outer_fold_evaluation",
        }
        if model_name:
            metadata["model_name"] = model_name
        write_json(destination / "inference_metadata.json", metadata)
        pd.DataFrame(history).to_csv(destination / "training_history.csv", index=False)
        pd.DataFrame({"feature": feature_suffixes, "mean": mean, "std": std}).to_csv(destination / "normalization_stats.csv", index=False)
        return model, mean, std, fitting["seconds"], preparation["seconds"]

    traces = [search(outer, cv_root / f"fold_{outer}")
              for outer in range(1, manifest["outer_folds"] + 1)]
    trace = pd.concat(traces, ignore_index=True)
    trace.to_csv(cv_root / "configuration_fold_metrics.csv", index=False)
    metric_columns = [c for c in trace if c.startswith(("validation_", "test_"))
                      or c.endswith("_seconds")]
    averages = trace.groupby("config_index", sort=True)[metric_columns].agg(["mean", "std"])
    averages.columns = [f"{metric}_{stat}" for metric, stat in averages.columns]
    summary = pd.DataFrame([{**config, "config_index": i} for i, config in enumerate(grid)]).merge(
        averages.reset_index(), on="config_index", validate="one_to_one")
    summary = summary.sort_values("validation_macro_f1_mean", ascending=False, kind="stable")
    selected = int(summary.iloc[0].config_index)
    summary["validation_macro_f1"] = summary["validation_macro_f1_mean"]
    summary["selected_for_deployment"] = summary.config_index == selected
    summary.to_csv(cv_root / "configuration_summary.csv", index=False)
    summary.to_csv(output_dir / "grid_search_summary.csv", index=False)
    summary.head(1).to_csv(output_dir / "best_grid_search_result.csv", index=False)
    config = grid[selected].copy()
    count = max(1, int(np.rint(trace.loc[trace.config_index == selected, "best_epoch"].median())))
    write_json(output_dir / "best_config.json", config)
    write_json(cv_root / "selection.json", {"config_index": selected, "best_config": config,
               "selection_metric": "mean_validation_macro_f1", "refit_epochs": count,
               "evaluation_note": manifest["evaluation_note"]})
    all_class_metrics = pd.concat([
        pd.read_csv(cv_root / f"fold_{outer}" / "per_class_configuration_metrics.csv")
        for outer in range(1, manifest["outer_folds"] + 1)], ignore_index=True)
    all_class_metrics.to_csv(cv_root / "configuration_fold_per_class_metrics.csv", index=False)
    all_class_metrics.groupby(["config_index", "split", "class_index", "class_name"])[
        ["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(
            cv_root / "configuration_per_class_summary.csv")
    # No fold checkpoint is retained: reuse cached predictions for the global winner.
    for outer in range(1, manifest["outer_folds"] + 1):
        destination = cv_root / f"fold_{outer}"
        candidate_dir = destination / f"config_{selected + 1}"
        result = pd.read_csv(candidate_dir / "test_predictions.csv", dtype={"date": str, "cell_id": str,
                              "class": str, "true_class_code": str, "predicted_class_code": str})
        result.to_csv(destination / "outer_predictions.csv", index=False)
        heldout_frames.append(result)
        validation_predictions = pd.read_csv(candidate_dir / "validation_predictions.csv", dtype=str)
        validation_predictions.to_csv(destination / "validation_predictions.csv", index=False)
        row = trace.loc[(trace.outer_fold == outer) & (trace.config_index == selected)].iloc[0].to_dict()
        row["selected_configuration_training_seconds"] = row["training_seconds"]
        row["training_seconds"] = float(trace.loc[trace.outer_fold == outer, "training_seconds"].sum())
        row.update({k: row[f"test_{k}"] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "mcc")})
        fold_rows.append(row)
        pd.DataFrame([row]).to_csv(destination / "metrics_and_timing.csv", index=False)
        report, confusion = classification_metrics(result.true_class_code.map(mapping).to_numpy(),
            result.predicted_class_code.map(mapping).to_numpy(), names)
        report.to_csv(destination / "outer_classification_report.csv")
        confusion.to_csv(destination / "outer_confusion_matrix.csv")
        class_rows.extend(_class_metrics(report, names, outer, model_type))

    oof = pd.concat(heldout_frames, ignore_index=True).sort_values(["date", "cell_id"]).reset_index(drop=True)
    oof.to_csv(cv_root / "oof_predictions.csv", index=False)
    truth = oof.true_class_code.astype(str).map(mapping).to_numpy()
    prediction = oof.predicted_class_code.astype(str).map(mapping).to_numpy()
    report, confusion = classification_metrics(truth, prediction, names)
    report.to_csv(cv_root / "oof_classification_report.csv")
    confusion.to_csv(cv_root / "oof_confusion_matrix.csv")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(cv_root / "fold_metrics_and_timing.csv", index=False)
    class_folds = _save_class_summaries(class_rows, cv_root)
    summarize_folds(folds).to_csv(cv_root / "cv_summary.csv", index=False)
    pd.DataFrame([metric_row(truth, prediction, list(range(len(mapping))))]).to_csv(cv_root / "pooled_oof_metrics.csv", index=False)

    print(f"{model_type}: fitting selected mean-validation winner on all {len(samples)} cells; no deployment search", flush=True)
    model, _, _, fitting, preparation = refit(samples, config, count, output_dir, 0)
    del model
    write_json(output_dir / "deployment_timing.json", {"refit_training_seconds": fitting,
               "preprocessing_seconds": preparation, "deployment_search_seconds": 0.0})
    write_json(cv_root / "completed.json", experiment)
    return {"fold_metrics": folds, "summary": summarize_folds(folds), "oof_predictions": oof,
            "per_class_fold_metrics": class_folds,
            "classification_report": report, "confusion_matrix": confusion,
            "deployment_configuration": summary.head(1), "configuration_summary": summary,
            "configuration_fold_metrics": trace}


def _aligned_probabilities(path, expected, class_codes):
    frame = pd.read_csv(path, dtype={"date": str, "cell_id": str, "class": str, "true_class_code": str})
    keys = ["date", "cell_id", "class"]
    if frame.duplicated(["date", "cell_id"]).any() or fingerprint(frame, keys) != fingerprint(expected, keys):
        raise ValueError(f"Prediction samples differ from the CV plan: {path}")
    frame = expected[keys].merge(frame.drop(columns="class"), on=["date", "cell_id"], validate="one_to_one")
    if not frame["class"].equals(frame.true_class_code):
        raise ValueError(f"True labels do not match the CV plan: {path}")
    probabilities = frame[[f"probability_class_{code}" for code in class_codes]].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError(f"Invalid class probabilities: {path}")
    return probabilities


def evaluate_spatial_ensembles(*, cv_dir, model_result_dirs, output_dir, class_name_by_code,
                              feature_suffixes=None, channel_setup=None):
    """Combine matching-channel winners; weights use validation predictions only.

    Omitted channels/setup are inferred from the first deployment artifact.
    All directory arguments are used as supplied.
    """
    from Source.ensemble import macro_f1_weights, per_class_f1_weights

    channels = (_channel_configuration(feature_suffixes, channel_setup)
                if feature_suffixes is not None or channel_setup is not None else None)
    if len(model_result_dirs) < 2 or "small_cnn" in model_result_dirs:
        raise ValueError("Ensembles require at least two pretrained backbones; Small CNN is a baseline")
    samples, validation_splits, manifest = read_cv_plan(cv_dir)
    mapping = make_class_mapping(samples)
    codes = list(mapping)
    reverse = {v: k for k, v in mapping.items()}
    names = {index: f"{code}: {class_name_by_code.get(code, code)}" for code, index in mapping.items()}
    output_dir = Path(output_dir)
    cv_root = output_dir / "cross_validation"
    cv_root.mkdir(parents=True, exist_ok=True)
    (cv_root / "completed.json").unlink(missing_ok=True)
    all_rows, all_predictions, weight_rows, class_rows = [], [], [], []
    experiments, metadata_hashes = {}, {}
    for name, directory in model_result_dirs.items():
        directory = Path(directory)
        completed = json.loads((directory / "cross_validation" / "completed.json").read_text())
        for key in ("protocol", "sample_hash", "inner_hash", "random_seed", "max_context_radius"):
            if completed[key] != manifest[key]:
                raise ValueError(f"{name}: incompatible {key}; rerun training with the current CV plan")
        metadata = json.loads((directory / "inference_metadata.json").read_text())
        if channels is None:
            channels = _channel_configuration(metadata.get("feature_suffixes", ()),
                                               metadata.get("channel_setup"))
        _check_channel_configuration(completed, channels, name, legacy=True)
        _check_channel_configuration(metadata, channels, name)
        if (metadata.get("purpose") != "deployment"
                or metadata.get("cv_sample_hash") != manifest["sample_hash"]
                or metadata.get("cv_inner_hash") != manifest["inner_hash"]
                or metadata.get("run_id") != completed["run_id"]):
            raise ValueError(f"{name}: deployment model does not match the CV experiment")
        experiments[name] = completed
        metadata_hashes[name] = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def weights(outer):
        folds = range(1, manifest["outer_folds"] + 1) if outer == 0 else [outer]
        probability_parts = {name: [] for name in model_result_dirs}
        truths = []
        for fold in folds:
            validation = validation_splits.loc[(validation_splits.outer_fold == fold)
                                               & (validation_splits.split == "validation")]
            truths.append(validation["class"].map(mapping).to_numpy())
            for name, directory in model_result_dirs.items():
                parent = Path(directory) / "cross_validation" / f"fold_{fold}"
                probability_parts[name].append(_aligned_probabilities(
                    parent / "validation_predictions.csv", validation, codes))
        probabilities = {name: np.concatenate(parts) for name, parts in probability_parts.items()}
        truth = np.concatenate(truths)
        started = perf_counter()
        scores, global_weights = macro_f1_weights(truth, probabilities)
        global_seconds = perf_counter() - started
        started = perf_counter()
        class_scores, by_class = per_class_f1_weights(truth, probabilities)
        class_seconds = perf_counter() - started
        seconds = {"weighted_probability_per_model": global_seconds,
                   "weighted_probability_per_model_class": class_seconds}
        return scores, global_weights, class_scores, by_class, seconds

    for outer in range(1, manifest["outer_folds"] + 1):
        destination = cv_root / f"fold_{outer}"
        destination.mkdir(parents=True, exist_ok=True)
        scores, global_weights, class_scores, by_class, fitting = weights(outer)
        write_json(destination / "weights.json", {
            "global_model_weights": global_weights,
            "weights_by_model_and_class": {name: dict(zip(codes, values.tolist())) for name, values in by_class.items()},
            "source": "single_validation_split_predictions_only",
        })
        heldout = samples.loc[samples.outer_fold == outer].reset_index(drop=True)
        probabilities, component_times = {}, []
        for name, directory in model_result_dirs.items():
            parent = Path(directory) / "cross_validation" / f"fold_{outer}"
            probabilities[name] = _aligned_probabilities(parent / "outer_predictions.csv", heldout, codes)
            component_times.append(pd.read_csv(parent / "metrics_and_timing.csv").iloc[0])
            weight_rows.append({"outer_fold": outer, "model": name, "inner_macro_f1": scores[name], "global_weight": global_weights[name]})
        truth = heldout["class"].map(mapping).to_numpy()
        result_frame = heldout.copy()
        result_frame["true_class_code"] = heldout["class"]
        for method in ENSEMBLE_METHODS:
            started = perf_counter()
            # Time only this method's combination, not the other three candidates.
            prediction, combined = combine_ensemble(method, probabilities, global_weights, by_class)
            combining = perf_counter() - started
            for index, code in reverse.items():
                result_frame[f"{method}_probability_class_{code}"] = combined[:, index]
            result_frame[f"{method}_predicted_class_code"] = [reverse[int(x)] for x in prediction]
            weighted = method.startswith("weighted_")
            row = {"method": method, "outer_fold": outer,
                   **metric_row(truth, prediction, list(range(len(mapping)))),
                   "component_training_seconds": sum(t.training_seconds for t in component_times),
                   "weight_fitting_seconds": fitting[method] if weighted else 0.0,
                   "training_seconds": sum(t.training_seconds for t in component_times) + (fitting[method] if weighted else 0.0),
                   "component_prediction_seconds": sum(t.prediction_seconds for t in component_times),
                   "combination_seconds": combining,
                   "prediction_seconds": sum(t.prediction_seconds for t in component_times) + combining}
            all_rows.append(row)
            report, confusion = classification_metrics(truth, prediction, names)
            report.to_csv(destination / f"{method}_classification_report.csv")
            confusion.to_csv(destination / f"{method}_confusion_matrix.csv")
            class_rows.extend(_class_metrics(report, names, outer, method))
        result_frame.to_csv(destination / "outer_predictions.csv", index=False)
        all_predictions.append(result_frame)
        print(f"Completed all four grain-size ensembles for outer fold {outer}", flush=True)

    oof = pd.concat(all_predictions, ignore_index=True).sort_values(["date", "cell_id"]).reset_index(drop=True)
    oof.to_csv(cv_root / "oof_predictions.csv", index=False)
    folds = pd.DataFrame(all_rows)
    folds.to_csv(cv_root / "fold_metrics_and_timing.csv", index=False)
    class_folds = _save_class_summaries(class_rows, cv_root)
    summarize_folds(folds).to_csv(cv_root / "cv_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(cv_root / "fold_model_weights.csv", index=False)
    truth = oof.true_class_code.map(mapping).to_numpy()
    pooled = []
    for method in ENSEMBLE_METHODS:
        prediction = oof[f"{method}_predicted_class_code"].map(mapping).to_numpy()
        report, confusion = classification_metrics(truth, prediction, names)
        report.to_csv(cv_root / f"oof_{method}_classification_report.csv")
        confusion.to_csv(cv_root / f"oof_{method}_confusion_matrix.csv")
        pooled.append({"method": method, **metric_row(truth, prediction, list(range(len(mapping))))})
    pd.DataFrame(pooled).to_csv(cv_root / "pooled_oof_metrics.csv", index=False)

    scores, global_weights, class_scores, by_class, fitting = weights(0)
    definition = {
        "protocol": PROTOCOL, **channels, "cv_sample_hash": manifest["sample_hash"],
        "cv_inner_hash": manifest["inner_hash"],
        "max_context_radius": manifest["max_context_radius"],
        "buffer_distance_cells": manifest["buffer_distance_cells"],
        "selection_metric": "mean_validation_macro_f1_across_outer_runs",
        "evaluation_note": manifest["evaluation_note"],
        "candidate_methods": list(ENSEMBLE_METHODS),
        "global_model_weights": global_weights, "validation_model_macro_f1": scores,
        "weights_by_model_and_class": {name: dict(zip(codes, values.tolist())) for name, values in by_class.items()},
        "validation_f1_by_model_and_class": {name: dict(zip(codes, values.tolist())) for name, values in class_scores.items()},
        "class_to_idx": mapping, "idx_to_class_code": {str(v): k for k, v in mapping.items()},
        "models": {name: {"result_dir": str(directory),
                           "checkpoint_path": str(Path(directory) / f"{name}_best.pt"),
                           "metadata_path": str(Path(directory) / "inference_metadata.json")}
                   for name, directory in model_result_dirs.items()},
        "weight_fitting_seconds": fitting,
        "deployment_model_metadata_hashes": metadata_hashes,
        "weight_source": "pooled_single_split_validation_predictions_for_global_winners_not_test_labels",
    }
    write_json(output_dir / "ensemble_definition.json", definition)
    write_json(cv_root / "completed.json", {**manifest, **channels, "models": experiments})
    return {"fold_metrics": folds, "summary": summarize_folds(folds), "oof_predictions": oof,
            "per_class_fold_metrics": class_folds,
            "pooled_metrics": pd.DataFrame(pooled), "deployment_weights": pd.DataFrame([
                {"model": name, "inner_macro_f1": scores[name], "weight": global_weights[name]} for name in global_weights])}


def combine_ensemble(method, probabilities, global_weights, by_class):
    from Source.ensemble import (
        average_probabilities, class_weighted_average_probabilities,
        hard_voting_predictions, weighted_average_probabilities,
    )
    if method == "hard_voting":
        return hard_voting_predictions(probabilities)
    if method == "probability_voting":
        combined = average_probabilities(probabilities)
    elif method == "weighted_probability_per_model":
        combined = weighted_average_probabilities(probabilities, global_weights)
    elif method == "weighted_probability_per_model_class":
        combined = class_weighted_average_probabilities(probabilities, by_class)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    return combined.argmax(1), combined
