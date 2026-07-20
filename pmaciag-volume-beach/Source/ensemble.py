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
from Source.inference import load_model_for_inference
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
