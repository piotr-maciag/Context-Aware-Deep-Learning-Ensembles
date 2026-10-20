from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
    metadata_path = result_dir / "inference_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing inference metadata: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata.setdefault("model_type", name)

    if checkpoint_filename is None:
        checkpoint_filename = metadata.get("checkpoint_filename")
    if not checkpoint_filename:
        checkpoint_filename = _default_checkpoint_filename(metadata.get("model_type", name))

    checkpoint_path = result_dir / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

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


def load_global_weighted_ensemble(
    artifacts: Sequence[ModelArtifact],
    definition_path: Path,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, float], dict]:
    """Load the best models and validation-derived global ensemble weights."""
    definition_path = Path(definition_path)
    if not definition_path.exists():
        raise FileNotFoundError(f"Missing ensemble definition: {definition_path}")

    with definition_path.open("r", encoding="utf-8") as file:
        definition = json.load(file)

    if definition.get("protocol") in ("nested_stratified_cell_cv_v1", "nested_spatial_group_cv_v2"):
        from hashlib import sha256
        expected_hashes = definition["deployment_model_metadata_hashes"]
        for artifact in artifacts:
            metadata_hash = sha256(json.dumps(artifact.metadata, sort_keys=True).encode()).hexdigest()
            if (artifact.metadata.get("purpose") != "deployment"
                    or metadata_hash != expected_hashes.get(artifact.name)):
                raise ValueError(f"{artifact.name}: deployment model changed; rerun grain-size ensemble evaluation")

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


def classify_date_directory_with_global_ensemble(
    date_dir: Path,
    source_root: Path,
    output_root: Path,
    artifacts: Sequence[ModelArtifact],
    models: dict[str, torch.nn.Module],
    weights: dict[str, float],
    idx_to_class_code: dict[int, str],
    batch_size: int,
    num_workers: int,
    base_suffix: str,
    device: torch.device,
    candidate_cell_ids: Sequence[str] | None = None,
) -> dict:
    """Classify one date with globally weighted probability averaging."""
    date = date_dir.name
    reference_metadata = artifacts[0].metadata
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

    if not cell_ids:
        output_path = None
        if candidate_cell_ids is not None:
            output_dir = Path(output_root) / date
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "classification.csv"
            pd.DataFrame(columns=["date", "cell_id", "predicted_class"]).to_csv(
                output_path,
                index=False,
            )
        return {
            "date": date,
            "classified_cells": 0,
            "incomplete_candidates": incomplete_count,
            "device": str(device),
            "output_path": output_path,
        }

    combined_probabilities = None
    expected_cell_ids = [str(cell_id) for cell_id in cell_ids]
    for artifact in artifacts:
        metadata = artifact.metadata
        print(
            f"  {date}: running {artifact.name} "
            f"(context radius {metadata['context_radius']}, weight {weights[artifact.name]:.6f})",
            flush=True,
        )
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
        print(
            f"  {date}: finished {artifact.name} for {len(predicted_cell_ids):,} cells",
            flush=True,
        )
        if predicted_cell_ids != expected_cell_ids:
            raise RuntimeError(f"Cell order changed during inference for {artifact.name}.")

        weighted_probabilities = probabilities.astype(np.float32, copy=False) * np.float32(
            weights[artifact.name]
        )
        if combined_probabilities is None:
            combined_probabilities = weighted_probabilities
        else:
            combined_probabilities += weighted_probabilities

    if combined_probabilities is None:
        raise RuntimeError(f"No ensemble probabilities were generated for {date}.")

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

    output_dir = Path(output_root) / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification.csv"
    predictions.to_csv(output_path, index=False)
    return {
        "date": date,
        "classified_cells": len(predictions),
        "incomplete_candidates": incomplete_count,
        "device": str(device),
        "output_path": output_path,
    }


def classify_date_directories_with_global_ensemble(
    date_dirs: Sequence[Path],
    source_root: Path,
    output_root: Path,
    artifacts: Sequence[ModelArtifact],
    models: dict[str, torch.nn.Module],
    weights: dict[str, float],
    idx_to_class_code: dict[int, str],
    batch_size: int,
    num_workers: int,
    base_suffix: str,
    device: torch.device,
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
        row = classify_date_directory_with_global_ensemble(
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
        print(
            f"Finished {row['date']}: classified {row['classified_cells']:,}, "
            f"skipped {row['incomplete_candidates']:,}, saved {row['output_path']}",
            flush=True,
        )
    return pd.DataFrame(rows)


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
