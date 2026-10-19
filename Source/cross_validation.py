"""Nested, stratified cell-level evaluation and separate deployment fitting."""

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
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader

from Source.beach_dataset import (
    BeachCellDataset, FEATURE_SUFFIXES, class_weights, compute_channel_stats,
    make_class_mapping,
)
from Source.grid_search import _run_single_grid_config
from Source.timing import elapsed_seconds
from Source.training import classification_metrics, predict, train_one_epoch


PROTOCOL = "nested_stratified_cell_cv_v1"
ENSEMBLE_METHODS = (
    "hard_voting", "probability_voting", "weighted_probability_per_model",
    "weighted_probability_per_model_class",
)


def fingerprint(frame, columns=None):
    columns = columns or [c for c in ("date", "cell_id", "class", "outer_fold", "inner_fold") if c in frame]
    records = frame[columns].astype(str).sort_values(columns).to_dict("records")
    return sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def create_cv_plan(frame, outer_folds=5, inner_folds=3, random_seed=42):
    samples = frame.drop(columns=["split", "outer_fold", "inner_fold"], errors="ignore").copy()
    for column in ("date", "cell_id", "class"):
        samples[column] = samples[column].astype(str).str.strip()
        if samples[column].isin(["", "nan", "<NA>"]).any():
            raise ValueError(f"Invalid {column} in CV data")
    if samples.duplicated(["date", "cell_id"]).any():
        raise ValueError("Duplicate date/cell_id pairs in CV data")
    samples = samples.sort_values(["date", "cell_id"]).reset_index(drop=True)
    if outer_folds < 2 or inner_folds < 2 or samples["class"].value_counts().min() < outer_folds:
        raise ValueError("Insufficient examples per class for requested outer folds")
    samples["outer_fold"] = 0
    splitter = StratifiedKFold(outer_folds, shuffle=True, random_state=random_seed)
    for fold, (_, heldout) in enumerate(splitter.split(samples, samples["class"]), 1):
        samples.loc[heldout, "outer_fold"] = fold
    inner_rows = []
    # Fold zero describes the independent, all-labelled-data deployment search.
    for outer in range(outer_folds + 1):
        training = samples if outer == 0 else samples.loc[samples.outer_fold != outer]
        if training["class"].value_counts().min() < inner_folds:
            raise ValueError("Insufficient training examples per class for inner folds")
        inner_splitter = StratifiedKFold(inner_folds, shuffle=True, random_state=random_seed + outer + 1)
        for inner, (_, validation) in enumerate(inner_splitter.split(training, training["class"]), 1):
            part = training.iloc[validation][["date", "cell_id", "class"]].copy()
            part["outer_fold"] = outer
            part["inner_fold"] = inner
            inner_rows.append(part)
    return samples, pd.concat(inner_rows, ignore_index=True)


def save_cv_plan(samples, inner, output_dir, random_seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "gs_cv_samples.csv", index=False)
    inner.to_csv(output_dir / "gs_cv_inner_folds.csv", index=False)
    manifest = {
        "protocol": PROTOCOL, "random_seed": random_seed,
        "sample_hash": fingerprint(samples), "inner_hash": fingerprint(inner),
        "outer_folds": int(samples.outer_fold.max()),
        "inner_folds": int(inner.inner_fold.max()),
    }
    write_json(output_dir / "gs_cv_manifest.json", manifest)
    return manifest


def read_cv_plan(directory):
    directory = Path(directory)
    samples = pd.read_csv(directory / "gs_cv_samples.csv", dtype={"date": str, "cell_id": str, "class": str})
    inner = pd.read_csv(directory / "gs_cv_inner_folds.csv", dtype={"date": str, "cell_id": str, "class": str})
    manifest = json.loads((directory / "gs_cv_manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL or fingerprint(samples) != manifest["sample_hash"] or fingerprint(inner) != manifest["inner_hash"]:
        raise ValueError("CV manifest does not match samples/folds; regenerate the CV plan")
    outer_count, inner_count = manifest["outer_folds"], manifest["inner_folds"]
    if set(samples.outer_fold) != set(range(1, outer_count + 1)) or samples.duplicated(["date", "cell_id"]).any():
        raise ValueError("Invalid outer-fold assignments")
    if set(inner.outer_fold) != set(range(outer_count + 1)):
        raise ValueError("Missing inner-fold or deployment assignments")
    for outer in range(outer_count + 1):
        expected = samples if outer == 0 else samples.loc[samples.outer_fold != outer]
        current = inner.loc[inner.outer_fold == outer]
        if (set(current.inner_fold) != set(range(1, inner_count + 1))
                or current.duplicated(["date", "cell_id"]).any()
                or fingerprint(current, ["date", "cell_id", "class"]) != fingerprint(expected, ["date", "cell_id", "class"])):
            raise ValueError(f"Invalid inner partition for outer fold {outer}")
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


def run_nested_cv(
    *, model_factory, parameter_grid, cv_dir, cell_data_root, output_dir,
    model_type, class_name_by_code, image_size=224, model_name=None,
    batch_size=32, num_workers=8, epochs=50, early_stopping_patience=5,
    early_stopping_min_delta=1e-4, augment=True, device=None, random_seed=42,
):
    from Source.training import get_device

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
    experiment = {**manifest, "run_id": uuid4().hex, "model_type": model_type, "parameter_grid": grid,
                  "image_size": image_size, "epochs": epochs, "batch_size": batch_size,
                  "augmentation": augment, "early_stopping_patience": early_stopping_patience,
                  "early_stopping_min_delta": early_stopping_min_delta}
    write_json(cv_root / "experiment.json", experiment)
    mapping = make_class_mapping(samples)
    names = {index: f"{code}: {class_name_by_code.get(code, code)}" for code, index in mapping.items()}
    fold_rows, heldout_frames, class_rows = [], [], []

    def dataset(frame, config, mean, std, training=False):
        return BeachCellDataset(frame, cell_data_root, mapping, image_size=image_size,
                                mean=mean, std=std, augment=training and augment,
                                context_radius=int(config["context_radius"]))

    def loader(frame, config, mean, std, training=False):
        return DataLoader(dataset(frame, config, mean, std, training), batch_size=batch_size,
                          shuffle=training, num_workers=num_workers)

    def search(outer, training, destination):
        destination.mkdir(parents=True, exist_ok=True)
        assignments = inner.loc[inner.outer_fold == outer, ["date", "cell_id", "inner_fold"]]
        partition = training.merge(assignments, on=["date", "cell_id"], validate="one_to_one")
        stats, parts = {}, {}
        with elapsed_seconds(device) as preparation:
            for fold in sorted(partition.inner_fold.unique()):
                fit = partition.loc[partition.inner_fold != fold].reset_index(drop=True)
                validation = partition.loc[partition.inner_fold == fold].reset_index(drop=True)
                stats[fold] = compute_channel_stats(fit, cell_data_root, context_radius=0)
                parts[fold] = (fit, validation)
        gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
        devices = [torch.device(f"cuda:{i}") for i in range(gpu_count)] or [device]
        queue = Queue()
        for item in devices:
            queue.put(item)

        def candidate(task):
            config_index, fold = task
            assigned_device = queue.get()
            try:
                fit, validation = parts[fold]
                mean, std = stats[fold]
                config = grid[config_index]
                print(f"{model_type}: outer {outer}, inner {fold}, grid {config_index + 1}/{len(grid)} on {assigned_device}", flush=True)
                result = _run_single_grid_config(
                    run_index=config_index + 1, config=config, model_factory=model_factory,
                    train_loader=loader(fit, config, mean, std, True),
                    validation_loader=loader(validation, config, mean, std), loader_factory=None,
                    class_weights=class_weights(fit, mapping), idx_to_class=names,
                    device=assigned_device, epochs=epochs,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    random_seed=random_seed + 100 * outer + fold, quiet=True,
                )
                predicted = _prediction_frame(validation, result["y_true"], result["y_probability"], mapping)
                history = result["history"].assign(config_index=config_index, inner_fold=fold)
                row = {**result["summary_row"], "config_index": config_index, "inner_fold": int(fold)}
                per_class = result["report"].rename_axis("label").reset_index().assign(config_index=config_index, inner_fold=fold)
                # Inner states are not deployment or evaluation checkpoints.
                del result
                print(f"Finished {model_type} outer {outer}, inner {fold}, grid {config_index + 1}: F1={row['validation_macro_f1']:.4f}", flush=True)
                return row, predicted, history, per_class
            finally:
                queue.put(assigned_device)

        tasks = [(i, fold) for i in range(len(grid)) for fold in parts]
        with elapsed_seconds(device) as search_time:
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                results = list(pool.map(candidate, tasks))
        trace = pd.DataFrame([r[0] for r in results])
        trace.to_csv(destination / "inner_grid_search_trace.csv", index=False)
        pd.concat([r[2] for r in results], ignore_index=True).to_csv(destination / "inner_epoch_trace.csv", index=False)
        pd.concat([r[3] for r in results], ignore_index=True).to_csv(destination / "inner_classification_metric_trace.csv", index=False)
        means = trace.groupby("config_index", sort=True).validation_macro_f1.mean()
        selected = int(means.idxmax())
        best_config = grid[selected].copy()
        selected_trace = trace.loc[trace.config_index == selected]
        refit_epochs = max(1, int(np.rint(selected_trace.best_epoch.median())))
        oof = pd.concat([r[1] for r in results if r[0]["config_index"] == selected], ignore_index=True)
        oof = oof.sort_values(["date", "cell_id"]).reset_index(drop=True)
        oof.to_csv(destination / "inner_oof_predictions.csv", index=False)
        summary = pd.DataFrame([{**config, "config_index": i,
                                "validation_macro_f1": float(means.loc[i]),
                                "training_seconds": float(trace.loc[trace.config_index == i, "training_seconds"].sum())}
                               for i, config in enumerate(grid)]).sort_values("validation_macro_f1", ascending=False, kind="stable")
        summary.to_csv(destination / "grid_search_summary.csv", index=False)
        summary.head(1).to_csv(destination / "best_grid_search_result.csv", index=False)
        write_json(destination / "best_config.json", best_config)
        return best_config, refit_epochs, oof, summary, {
            "search_wall_seconds": search_time["seconds"],
            "inner_training_seconds": float(trace.training_seconds.sum()),
            "inner_prediction_seconds": float(trace.validation_prediction_seconds.sum()),
            "inner_preprocessing_seconds": preparation["seconds"],
        }

    def refit(training, config, refit_epochs, destination, outer):
        destination.mkdir(parents=True, exist_ok=True)
        with elapsed_seconds(device) as preparation:
            mean, std = compute_channel_stats(training, cell_data_root, context_radius=0)
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
        metadata = {
            "model_type": model_type, "checkpoint_filename": checkpoint.name,
            "image_size": image_size if image_size is not None else 20 * (2 * int(config["context_radius"]) + 1),
            "context_radius": int(config["context_radius"]),
            "context_window_size": 2 * int(config["context_radius"]) + 1,
            "feature_suffixes": list(FEATURE_SUFFIXES), "in_channels": len(FEATURE_SUFFIXES),
            "num_classes": len(mapping), "class_to_idx": mapping,
            "idx_to_class_code": {str(v): k for k, v in mapping.items()},
            "class_name_by_code": class_name_by_code, "best_config": config,
            "channel_mean": mean.tolist(), "channel_std": std.tolist(),
            "protocol": PROTOCOL, "outer_fold": outer, "cv_sample_hash": manifest["sample_hash"],
            "run_id": experiment["run_id"],
            "cv_inner_hash": manifest["inner_hash"], "refit_epochs": refit_epochs,
            "training_sample_hash": fingerprint(training, ["date", "cell_id", "class"]),
            "purpose": "deployment" if outer == 0 else "outer_fold_evaluation",
        }
        if model_name:
            metadata["model_name"] = model_name
        write_json(destination / "inference_metadata.json", metadata)
        pd.DataFrame(history).to_csv(destination / "training_history.csv", index=False)
        pd.DataFrame({"feature": FEATURE_SUFFIXES, "mean": mean, "std": std}).to_csv(destination / "normalization_stats.csv", index=False)
        return model, mean, std, fitting["seconds"], preparation["seconds"]

    for outer in range(1, manifest["outer_folds"] + 1):
        destination = cv_root / f"fold_{outer}"
        training = samples.loc[samples.outer_fold != outer].reset_index(drop=True)
        heldout = samples.loc[samples.outer_fold == outer].reset_index(drop=True)
        config, count, _, _, times = search(outer, training, destination)
        model, mean, std, fitting, preparation = refit(training, config, count, destination, outer)
        with elapsed_seconds(device) as prediction_time:
            truth, prediction, probabilities = predict(model, loader(heldout, config, mean, std), device, show_progress=False)
        result = _prediction_frame(heldout, truth, probabilities, mapping)
        result.to_csv(destination / "outer_predictions.csv", index=False)
        report, confusion = classification_metrics(truth, prediction, names)
        report.to_csv(destination / "outer_classification_report.csv")
        confusion.to_csv(destination / "outer_confusion_matrix.csv")
        class_rows.extend(_class_metrics(report, names, outer, model_type))
        heldout_frames.append(result)
        row = {"method": model_type, "outer_fold": outer, **metric_row(truth, prediction, list(range(len(mapping)))),
               **times, "refit_training_seconds": fitting,
               "training_seconds": times["inner_training_seconds"] + fitting,
               "preprocessing_seconds": times["inner_preprocessing_seconds"] + preparation,
               "prediction_seconds": prediction_time["seconds"], "refit_epochs": count,
               "context_radius": config["context_radius"]}
        fold_rows.append(row)
        pd.DataFrame([row]).to_csv(destination / "metrics_and_timing.csv", index=False)
        print(f"Completed {model_type} outer fold {outer}: F1={row['macro_f1']:.4f}, prediction={row['prediction_seconds']:.2f}s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

    print(f"{model_type}: selecting and fitting separate all-data deployment model", flush=True)
    config, count, _, summary, times = search(0, samples, output_dir)
    model, _, _, fitting, preparation = refit(samples, config, count, output_dir, 0)
    del model
    write_json(output_dir / "deployment_timing.json", {**times, "refit_training_seconds": fitting,
               "preprocessing_seconds": times["inner_preprocessing_seconds"] + preparation})
    write_json(cv_root / "completed.json", experiment)
    return {"fold_metrics": folds, "summary": summarize_folds(folds), "oof_predictions": oof,
            "per_class_fold_metrics": class_folds,
            "classification_report": report, "confusion_matrix": confusion,
            "deployment_configuration": summary.head(1)}


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


def evaluate_nested_ensembles(*, cv_dir, model_result_dirs, output_dir, class_name_by_code):
    """Combine saved outer-fold predictions; weights use only inner OOF labels."""
    from Source.ensemble import macro_f1_weights, per_class_f1_weights

    if len(model_result_dirs) < 2 or "small_cnn" in model_result_dirs:
        raise ValueError("Ensembles require at least two pretrained backbones; Small CNN is a baseline")
    samples, _, manifest = read_cv_plan(cv_dir)
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
        for key in ("protocol", "sample_hash", "inner_hash", "random_seed"):
            if completed[key] != manifest[key]:
                raise ValueError(f"{name}: incompatible {key}; rerun training with the current CV plan")
        metadata = json.loads((directory / "inference_metadata.json").read_text())
        if (metadata.get("purpose") != "deployment"
                or metadata.get("cv_sample_hash") != manifest["sample_hash"]
                or metadata.get("cv_inner_hash") != manifest["inner_hash"]
                or metadata.get("run_id") != completed["run_id"]):
            raise ValueError(f"{name}: deployment model does not match the CV experiment")
        experiments[name] = completed
        metadata_hashes[name] = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def weights(outer):
        training = samples if outer == 0 else samples.loc[samples.outer_fold != outer]
        probabilities = {}
        for name, directory in model_result_dirs.items():
            parent = Path(directory) if outer == 0 else Path(directory) / "cross_validation" / f"fold_{outer}"
            probabilities[name] = _aligned_probabilities(parent / "inner_oof_predictions.csv", training, codes)
        truth = training["class"].map(mapping).to_numpy()
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
            "source": "inner_out_of_fold_predictions_only",
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
        "protocol": PROTOCOL, "cv_sample_hash": manifest["sample_hash"],
        "cv_inner_hash": manifest["inner_hash"],
        "selection_metric": "deployment_inner_oof_macro_f1",
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
        "weight_source": "all_data_deployment_inner_oof_not_outer_evaluation_predictions",
    }
    write_json(output_dir / "ensemble_definition.json", definition)
    write_json(cv_root / "completed.json", {**manifest, "models": experiments})
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
