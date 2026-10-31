from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from Source.beach_dataset import BeachCellDataset
from Source.inference import (
    CellInferenceDataset,
    discover_complete_cells_from_base_suffix,
    load_model_for_inference,
)
from Source.training import predict


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    result_dir: Path
    checkpoint_path: Path
    metadata: dict


def load_model_artifact(
    name: str,
    result_dir: Path,
    checkpoint_filename: str | None = None,
) -> ModelArtifact:
    result_dir = Path(result_dir)
    from Source.channel_setups import require_completed_result
    require_completed_result(result_dir)
    metadata_path = result_dir / "inference_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing inference metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata.setdefault("model_type", name)
    if metadata.get("protocol") in ("nested_stratified_cell_cv_v1", "nested_spatial_group_cv_v2", "spatial_five_fold_single_validation_v3"):
        completion_path = result_dir / "cross_validation" / "completed.json"
        if not completion_path.exists():
            raise ValueError(f"{name}: grain-size experiment is incomplete; finish CV and deployment fitting")
        completion = json.loads(completion_path.read_text())
        if metadata.get("purpose") != "deployment" or metadata.get("run_id") != completion.get("run_id"):
            raise ValueError(f"{name}: deployment model does not match its completed grain-size experiment")

    if checkpoint_filename is None:
        checkpoint_filename = metadata.get("checkpoint_filename")
    if not checkpoint_filename:
        checkpoint_filename = _default_checkpoint_filename(metadata.get("model_type", name))

    checkpoint_path = result_dir / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")
    if metadata.get("checkpoint_sha256") is not None:
        from Source.bs_splits import file_sha256
        if file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint does not match its saved metadata; retrain or restore the matching checkpoint")

    return ModelArtifact(
        name=name,
        result_dir=result_dir,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )


def validate_artifact_compatibility(artifacts: Sequence[ModelArtifact]) -> None:
    if not artifacts:
        raise ValueError("At least one model artifact is required.")

    reference = artifacts[0].metadata
    required_keys = ("class_to_idx", "idx_to_class_code", "feature_suffixes")
    for artifact in artifacts:
        missing = [key for key in required_keys if key not in artifact.metadata]
        if missing:
            raise ValueError(f"{artifact.name} metadata is missing keys: {missing}")
        for key in required_keys:
            if artifact.metadata[key] != reference[key]:
                raise ValueError(
                    f"Incompatible {key} for {artifact.name}; all ensemble models "
                    "must use the same class mapping and feature order."
                )


def predict_artifact_on_split(
    artifact: ModelArtifact,
    split_df: pd.DataFrame,
    cell_data_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = artifact.metadata
    dataset = BeachCellDataset(
        split_df,
        cell_data_root,
        {str(key): int(value) for key, value in metadata["class_to_idx"].items()},
        image_size=int(metadata["image_size"]),
        mean=metadata["channel_mean"],
        std=metadata["channel_std"],
        feature_suffixes=metadata["feature_suffixes"],
        augment=False,
        context_radius=int(metadata["context_radius"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = load_model_for_inference(
        artifact.checkpoint_path,
        metadata,
        device=device,
    )
    y_true, _, probabilities = predict(
        model,
        loader,
        device,
        show_progress=show_progress,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_true, probabilities


def macro_f1_weights(
    y_true: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    if not probabilities_by_model:
        raise ValueError("No model probabilities were provided.")

    scores = {
        name: float(
            f1_score(
                y_true,
                probabilities.argmax(axis=1),
                average="macro",
                zero_division=0,
            )
        )
        for name, probabilities in probabilities_by_model.items()
    }
    score_sum = sum(scores.values())
    if score_sum <= 0:
        equal_weight = 1.0 / len(scores)
        weights = {name: equal_weight for name in scores}
    else:
        weights = {name: score / score_sum for name, score in scores.items()}
    return scores, weights


def weighted_average_probabilities(
    probabilities_by_model: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    if set(probabilities_by_model) != set(weights):
        raise ValueError("Probability and weight model names do not match.")

    shapes = {probabilities.shape for probabilities in probabilities_by_model.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model probability arrays have incompatible shapes: {shapes}")

    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")

    combined = np.zeros(next(iter(shapes)), dtype=np.float64)
    for name, probabilities in probabilities_by_model.items():
        combined += (float(weights[name]) / weight_sum) * probabilities
    return combined


def _load_ensemble_definition(artifacts, definition_path):
    validate_artifact_compatibility(artifacts)
    definition_path = Path(definition_path)
    from Source.channel_setups import require_completed_result
    require_completed_result(definition_path.parent)
    if not definition_path.exists():
        raise FileNotFoundError(f"Missing ensemble definition: {definition_path}")

    with definition_path.open("r", encoding="utf-8") as file:
        definition = json.load(file)

    for key in ("class_to_idx", "idx_to_class_code"):
        if definition.get(key) != artifacts[0].metadata[key]:
            raise ValueError(f"Ensemble definition and model artifacts have different {key}")
    if ("feature_suffixes" in definition
            and definition["feature_suffixes"] != artifacts[0].metadata["feature_suffixes"]):
        raise ValueError("Ensemble definition and model artifacts use different input channels")
    if definition.get("channel_setup") is not None:
        from Source.channel_setups import get_channel_setup
        _, expected_suffixes = get_channel_setup(definition["channel_setup"])
        if tuple(artifacts[0].metadata["feature_suffixes"]) != expected_suffixes:
            raise ValueError("Ensemble definition and model artifacts use different channel setups")
    if "split_provenance" in definition:
        for artifact in artifacts:
            if artifact.metadata.get("split_provenance") != definition["split_provenance"]:
                raise ValueError(f"{artifact.name}: model and ensemble use different dataset splits")

    grain_size_cv = definition.get("protocol") in ("nested_stratified_cell_cv_v1", "nested_spatial_group_cv_v2", "spatial_five_fold_single_validation_v3")
    if grain_size_cv and not (definition_path.parent / "cross_validation" / "completed.json").exists():
        raise ValueError("Grain-size ensemble evaluation is incomplete; rerun it to completion")
    expected_hashes = definition.get("deployment_model_metadata_hashes")
    if grain_size_cv or expected_hashes is not None:
        from hashlib import sha256
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != {a.name for a in artifacts}:
            raise ValueError("Ensemble definition must fingerprint exactly the loaded model artifacts")
        for artifact in artifacts:
            metadata_hash = sha256(json.dumps(artifact.metadata, sort_keys=True).encode()).hexdigest()
            if ((grain_size_cv and artifact.metadata.get("purpose") != "deployment")
                    or metadata_hash != expected_hashes.get(artifact.name)):
                raise ValueError(f"{artifact.name}: model changed; rerun the matching ensemble evaluation")
    else:
        import warnings
        warnings.warn(
            "Legacy ensemble definition lacks model fingerprints. Rerun ensemble evaluation "
            "before deployment to verify that saved weights match the current backbones.",
            RuntimeWarning,
            stacklevel=2,
        )
    return definition


def load_global_weighted_ensemble(
    artifacts: Sequence[ModelArtifact],
    definition_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, float], dict]:
    """Load the best models and validation-derived global ensemble weights."""
    definition = _load_ensemble_definition(artifacts, definition_path)

    raw_weights = definition.get("global_model_weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(
            f"Ensemble definition does not contain global_model_weights: {definition_path}"
        )

    artifact_names = {artifact.name for artifact in artifacts}
    if set(raw_weights) != artifact_names:
        raise ValueError(
            "Ensemble definition and model artifacts have different model names: "
            f"weights={sorted(raw_weights)}, artifacts={sorted(artifact_names)}"
        )

    weights = {name: float(raw_weights[name]) for name in raw_weights}
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Global ensemble weights must be finite and non-negative.")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Global ensemble weights must sum to a positive value.")
    weights = {name: weight / weight_sum for name, weight in weights.items()}

    models = {}
    for artifact in artifacts:
        models[artifact.name] = load_model_for_inference(
            artifact.checkpoint_path,
            artifact.metadata,
            device=device,
        )
    return models, weights, definition


def load_class_weighted_ensemble(
    artifacts: Sequence[ModelArtifact],
    definition_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, np.ndarray], dict]:
    """Load per-model/class weights in output-index order, never JSON key order."""
    definition = _load_ensemble_definition(artifacts, definition_path)
    raw_weights = definition.get("weights_by_model_and_class")
    if not isinstance(raw_weights, dict) or set(raw_weights) != {a.name for a in artifacts}:
        raise ValueError("Class-weighted ensemble must contain weights for exactly the loaded models")
    mapping = {int(index): str(code) for index, code in definition["idx_to_class_code"].items()}
    if set(mapping) != set(range(len(mapping))):
        raise ValueError("Model output indices must be contiguous from zero")
    codes = [mapping[index] for index in range(len(mapping))]
    weights = {}
    for artifact in artifacts:
        values = raw_weights[artifact.name]
        if not isinstance(values, dict) or set(values) != set(codes):
            raise ValueError(f"{artifact.name}: class weights must match every encoded class")
        weights[artifact.name] = np.array([values[code] for code in codes], dtype=np.float64)
    matrix = np.stack(list(weights.values()))
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix.sum(axis=0) <= 0).any():
        raise ValueError("Class weights must be finite, non-negative, and have a positive total for every class")
    totals = matrix.sum(axis=0)
    weights = {name: values / totals for name, values in weights.items()}
    models = {artifact.name: load_model_for_inference(artifact.checkpoint_path, artifact.metadata, device=device)
              for artifact in artifacts}
    return models, weights, definition


@torch.no_grad()
def predict_model_probabilities_for_cells(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    cell_ids: list[str] = []
    probability_batches: list[np.ndarray] = []
    for x, batch_cell_ids in loader:
        logits = model(x.to(device, non_blocking=device.type == "cuda"))
        probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        cell_ids.extend(str(cell_id) for cell_id in batch_cell_ids)

    if not probability_batches:
        return cell_ids, np.empty((0, 0), dtype=np.float32)
    return cell_ids, np.concatenate(probability_batches, axis=0)


def _classify_date_directory_with_weighted_ensemble(
    date_dir: Path,
    source_root: Path,
    output_root: Path,
    artifacts: Sequence[ModelArtifact],
    models: dict[str, torch.nn.Module],
    weights: dict[str, float | np.ndarray],
    idx_to_class_code: dict[int, str],
    batch_size: int,
    num_workers: int,
    base_suffix: str,
    device: torch.device,
    candidate_cell_ids: Sequence[str] | None = None,
    class_weighted: bool = False,
) -> dict:
    """Keep one running probability sum per date rather than all model outputs."""
    survey_started = perf_counter()
    timings = {f"{artifact.name}_prediction_seconds": 0.0 for artifact in artifacts}
    timings["ensemble_calculation_seconds"] = 0.0

    def finish(row):
        timings["survey_total_seconds"] = perf_counter() - survey_started
        row.update(timings)
        timing_dir = Path(output_root) / date_dir.name
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = timing_dir / "prediction_timing.csv"
        pd.DataFrame([
            {"date": date_dir.name, "stage": key.removesuffix("_seconds"), "seconds": value}
            for key, value in timings.items()
        ]).to_csv(timing_path, index=False)
        row["timing_path"] = timing_path
        print(
            f"  {date_dir.name}: ensemble calculation {timings['ensemble_calculation_seconds']:.3f}s; "
            f"survey total {timings['survey_total_seconds']:.2f}s",
            flush=True,
        )
        return row

    date = date_dir.name
    reference_metadata = artifacts[0].metadata
    if set(weights) != {artifact.name for artifact in artifacts}:
        raise ValueError("Probability and weight model names do not match")
    if class_weighted:
        saved_mapping = {int(index): str(code) for index, code in reference_metadata["idx_to_class_code"].items()}
        if saved_mapping != idx_to_class_code:
            raise ValueError("Prediction class mapping differs from the model output mapping")
        matrix = np.stack([np.asarray(weights[artifact.name], dtype=np.float64) for artifact in artifacts])
        if matrix.shape != (len(artifacts), len(idx_to_class_code)):
            raise ValueError("Each model must have one weight per output class")
        if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix.sum(0) <= 0).any():
            raise ValueError("Invalid class weights")
        matrix /= matrix.sum(axis=0)
        weights = {artifact.name: matrix[index] for index, artifact in enumerate(artifacts)}
    discovery_started = perf_counter()
    if candidate_cell_ids is None:
        cell_ids, incomplete_count = discover_complete_cells_from_base_suffix(
            date_dir,
            feature_suffixes=reference_metadata["feature_suffixes"],
            base_suffix=base_suffix,
        )
    else:
        cell_ids = []
        incomplete_count = 0
        seen_cell_ids = set()
        for candidate_cell_id in candidate_cell_ids:
            cell_id = str(candidate_cell_id).strip()
            if not cell_id or cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(cell_id)
            if all(
                (date_dir / f"{cell_id}{suffix}.txt").is_file()
                for suffix in reference_metadata["feature_suffixes"]
            ):
                cell_ids.append(cell_id)
            else:
                incomplete_count += 1

    timings["cell_discovery_seconds"] = perf_counter() - discovery_started
    timings["prediction_csv_write_seconds"] = 0.0
    if not cell_ids:
        output_path = None
        if candidate_cell_ids is not None:
            write_started = perf_counter()
            output_dir = Path(output_root) / date
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "classification.csv"
            pd.DataFrame(columns=["date", "cell_id", "predicted_class"]).to_csv(
                output_path,
                index=False,
            )
            timings["prediction_csv_write_seconds"] = perf_counter() - write_started
        return finish({
            "date": date,
            "classified_cells": 0,
            "incomplete_candidates": incomplete_count,
            "device": str(device),
            "output_path": output_path,
        })

    combined_probabilities = None
    expected_cell_ids = [str(cell_id) for cell_id in cell_ids]
    for artifact in artifacts:
        metadata = artifact.metadata
        weight_description = ("per-class weights" if class_weighted
                              else f"weight {weights[artifact.name]:.6f}")
        print(
            f"  {date}: running {artifact.name} "
            f"(context radius {metadata['context_radius']}, {weight_description})",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_started = perf_counter()
        dataset = CellInferenceDataset(
            source_root=source_root,
            date=date,
            cell_ids=cell_ids,
            image_size=int(metadata["image_size"]),
            mean=metadata["channel_mean"],
            std=metadata["channel_std"],
            feature_suffixes=metadata["feature_suffixes"],
            context_radius=int(metadata["context_radius"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        predicted_cell_ids, probabilities = predict_model_probabilities_for_cells(
            models[artifact.name],
            loader,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds = perf_counter() - prediction_started
        timings[f"{artifact.name}_prediction_seconds"] = prediction_seconds
        print(
            f"  {date}: finished {artifact.name} for {len(predicted_cell_ids):,} cells "
            f"in {prediction_seconds:.2f}s",
            flush=True,
        )
        if predicted_cell_ids != expected_cell_ids:
            raise RuntimeError(f"Cell order changed during inference for {artifact.name}.")

        ensemble_started = perf_counter()
        if class_weighted:
            weighted_probabilities = probabilities.astype(np.float64, copy=False) * weights[artifact.name][None, :]
        else:
            weighted_probabilities = probabilities.astype(np.float32, copy=False) * np.float32(weights[artifact.name])
        if combined_probabilities is None:
            combined_probabilities = weighted_probabilities
        else:
            combined_probabilities += weighted_probabilities
        del probabilities, weighted_probabilities
        timings["ensemble_calculation_seconds"] += perf_counter() - ensemble_started

    if combined_probabilities is None:
        raise RuntimeError(f"No ensemble probabilities were generated for {date}.")

    ensemble_started = perf_counter()
    if class_weighted:
        totals = combined_probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(combined_probabilities).all() or (totals <= 0).any():
            raise ValueError("Combined class-weighted probabilities must be finite and positive per cell")
        combined_probabilities /= totals

    predicted_indices = combined_probabilities.argmax(axis=1)
    predictions = pd.DataFrame(
        {
            "date": date,
            "cell_id": expected_cell_ids,
            "predicted_class": [
                idx_to_class_code[int(index)] for index in predicted_indices
            ],
        }
    )
    timings["ensemble_calculation_seconds"] += perf_counter() - ensemble_started

    write_started = perf_counter()
    output_dir = Path(output_root) / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification.csv"
    predictions.to_csv(output_path, index=False)
    timings["prediction_csv_write_seconds"] = perf_counter() - write_started
    return finish({
        "date": date,
        "classified_cells": len(predictions),
        "incomplete_candidates": incomplete_count,
        "device": str(device),
        "output_path": output_path,
    })


def classify_date_directory_with_global_ensemble(
    date_dir: Path, source_root: Path, output_root: Path,
    artifacts: Sequence[ModelArtifact], models: dict[str, torch.nn.Module],
    weights: dict[str, float], idx_to_class_code: dict[int, str],
    batch_size: int, num_workers: int, base_suffix: str, device: torch.device,
    candidate_cell_ids: Sequence[str] | None = None,
) -> dict:
    """Classify one date using a single global weight per model."""
    return _classify_date_directory_with_weighted_ensemble(
        date_dir, source_root, output_root, artifacts, models, weights,
        idx_to_class_code, batch_size, num_workers, base_suffix, device,
        candidate_cell_ids=candidate_cell_ids,
    )


def classify_date_directory_with_class_ensemble(
    date_dir: Path, source_root: Path, output_root: Path,
    artifacts: Sequence[ModelArtifact], models: dict[str, torch.nn.Module],
    weights: dict[str, np.ndarray], idx_to_class_code: dict[int, str],
    batch_size: int, num_workers: int, base_suffix: str, device: torch.device,
    candidate_cell_ids: Sequence[str] | None = None,
) -> dict:
    """Classify one date using saved per-model/class probability weights."""
    return _classify_date_directory_with_weighted_ensemble(
        date_dir, source_root, output_root, artifacts, models, weights,
        idx_to_class_code, batch_size, num_workers, base_suffix, device,
        candidate_cell_ids=candidate_cell_ids, class_weighted=True,
    )


def _classify_date_directories_with_weighted_ensemble(
    date_dirs: Sequence[Path],
    source_root: Path,
    output_root: Path,
    artifacts: Sequence[ModelArtifact],
    models: dict[str, torch.nn.Module],
    weights: dict[str, float | np.ndarray],
    idx_to_class_code: dict[int, str],
    batch_size: int,
    num_workers: int,
    base_suffix: str,
    device: torch.device,
    *,
    class_weighted: bool = False,
) -> pd.DataFrame:
    """Classify date directories sequentially and report durable nohup progress."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    date_dirs = list(date_dirs)
    rows = []
    for date_index, date_dir in enumerate(date_dirs, start=1):
        print(
            f"Starting date {date_index}/{len(date_dirs)}: {date_dir.name}",
            flush=True,
        )
        classify_date = (classify_date_directory_with_class_ensemble if class_weighted
                         else classify_date_directory_with_global_ensemble)
        row = classify_date(
            date_dir=date_dir,
            source_root=source_root,
            output_root=output_root,
            artifacts=artifacts,
            models=models,
            weights=weights,
            idx_to_class_code=idx_to_class_code,
            batch_size=batch_size,
            num_workers=num_workers,
            base_suffix=base_suffix,
            device=device,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "classification_summary.csv", index=False)
        print(
            f"Finished {row['date']}: classified {row['classified_cells']:,}, "
            f"skipped {row['incomplete_candidates']:,}, saved {row['output_path']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def classify_date_directories_with_global_ensemble(
    date_dirs: Sequence[Path], source_root: Path, output_root: Path,
    artifacts: Sequence[ModelArtifact], models: dict[str, torch.nn.Module],
    weights: dict[str, float], idx_to_class_code: dict[int, str],
    batch_size: int, num_workers: int, base_suffix: str, device: torch.device,
) -> pd.DataFrame:
    """Global-weighted inference with the existing public API."""
    return _classify_date_directories_with_weighted_ensemble(
        date_dirs, source_root, output_root, artifacts, models, weights,
        idx_to_class_code, batch_size, num_workers, base_suffix, device,
    )


def classify_date_directories_with_class_ensemble(
    date_dirs: Sequence[Path], source_root: Path, output_root: Path,
    artifacts: Sequence[ModelArtifact], models: dict[str, torch.nn.Module],
    weights: dict[str, np.ndarray], idx_to_class_code: dict[int, str],
    batch_size: int, num_workers: int, base_suffix: str, device: torch.device,
) -> pd.DataFrame:
    """Class-weighted inference with sequential dates and flushed nohup progress."""
    return _classify_date_directories_with_weighted_ensemble(
        date_dirs, source_root, output_root, artifacts, models, weights,
        idx_to_class_code, batch_size, num_workers, base_suffix, device,
        class_weighted=True,
    )


def average_probabilities(
    probabilities_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    if not probabilities_by_model:
        raise ValueError("No model probabilities were provided.")
    shapes = {probabilities.shape for probabilities in probabilities_by_model.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model probability arrays have incompatible shapes: {shapes}")
    return np.mean(np.stack(list(probabilities_by_model.values()), axis=0), axis=0)


def hard_voting_predictions(
    probabilities_by_model: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not probabilities_by_model:
        raise ValueError("No model probabilities were provided.")
    shapes = {probabilities.shape for probabilities in probabilities_by_model.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model probability arrays have incompatible shapes: {shapes}")

    sample_count, num_classes = next(iter(shapes))
    model_predictions = np.stack(
        [probabilities.argmax(axis=1) for probabilities in probabilities_by_model.values()],
        axis=0,
    )
    vote_counts = np.zeros((sample_count, num_classes), dtype=np.float64)
    sample_indices = np.arange(sample_count)
    for predictions in model_predictions:
        vote_counts[sample_indices, predictions] += 1.0
    vote_fractions = vote_counts / len(probabilities_by_model)

    mean_probabilities = average_probabilities(probabilities_by_model)
    predictions = np.empty(sample_count, dtype=np.int64)
    for sample_index in range(sample_count):
        tied_classes = np.flatnonzero(
            vote_counts[sample_index] == vote_counts[sample_index].max()
        )
        predictions[sample_index] = tied_classes[
            mean_probabilities[sample_index, tied_classes].argmax()
        ]
    return predictions, vote_fractions


def per_class_f1_weights(
    y_true: np.ndarray,
    probabilities_by_model: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not probabilities_by_model:
        raise ValueError("No model probabilities were provided.")

    shapes = {probabilities.shape for probabilities in probabilities_by_model.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model probability arrays have incompatible shapes: {shapes}")

    _, num_classes = next(iter(shapes))
    labels = np.arange(num_classes)
    scores = {
        name: f1_score(
            y_true,
            probabilities.argmax(axis=1),
            labels=labels,
            average=None,
            zero_division=0,
        ).astype(np.float64)
        for name, probabilities in probabilities_by_model.items()
    }

    score_matrix = np.stack(list(scores.values()), axis=0)
    score_sums = score_matrix.sum(axis=0)
    weight_matrix = np.empty_like(score_matrix)
    nonzero_classes = score_sums > 0
    weight_matrix[:, nonzero_classes] = (
        score_matrix[:, nonzero_classes] / score_sums[nonzero_classes]
    )
    weight_matrix[:, ~nonzero_classes] = 1.0 / len(scores)

    weights = {
        name: weight_matrix[index]
        for index, name in enumerate(scores)
    }
    return scores, weights


def class_weighted_average_probabilities(
    probabilities_by_model: dict[str, np.ndarray],
    weights_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    if set(probabilities_by_model) != set(weights_by_model):
        raise ValueError("Probability and weight model names do not match.")

    shapes = {probabilities.shape for probabilities in probabilities_by_model.values()}
    if len(shapes) != 1:
        raise ValueError(f"Model probability arrays have incompatible shapes: {shapes}")

    sample_count, num_classes = next(iter(shapes))
    weight_matrix = np.stack(
        [np.asarray(weights_by_model[name], dtype=np.float64) for name in probabilities_by_model],
        axis=0,
    )
    if weight_matrix.shape != (len(probabilities_by_model), num_classes):
        raise ValueError(
            "Each model must provide exactly one weight per output class; "
            f"expected {(len(probabilities_by_model), num_classes)}, got {weight_matrix.shape}."
        )

    class_weight_sums = weight_matrix.sum(axis=0)
    if np.any(class_weight_sums <= 0):
        raise ValueError("Ensemble weights must have a positive sum for every class.")
    normalized_weights = weight_matrix / class_weight_sums

    combined = np.zeros((sample_count, num_classes), dtype=np.float64)
    for model_index, (name, probabilities) in enumerate(probabilities_by_model.items()):
        combined += probabilities * normalized_weights[model_index][None, :]

    row_sums = combined.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Combined ensemble scores must have a positive sum per sample.")
    return combined / row_sums


def _default_checkpoint_filename(model_type: str) -> str:
    filenames = {
        "convnext_tiny": "convnext_tiny_best.pt",
        "vit_tiny": "vit_tiny_best.pt",
        "swin_tiny": "swin_tiny_best.pt",
        "efficientnetv2_s": "efficientnetv2_s_best.pt",
        "small_cnn": "small_cnn_best.pt",
    }
    try:
        return filenames[model_type]
    except KeyError as exc:
        raise ValueError(
            f"No default checkpoint filename is known for model_type={model_type!r}."
        ) from exc
