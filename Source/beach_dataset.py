from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


FEATURE_SUFFIXES: tuple[str, ...] = ("r", "g", "b", "s", "d")
MISSING_VALUE_SENTINELS: tuple[float, ...] = (-999.0,)


@dataclass(frozen=True)
class DatasetPaths:
    annotated_root: Path
    split_csv: Path


def read_split_csv(split_csv: Path) -> pd.DataFrame:
    split = pd.read_csv(
        split_csv,
        dtype={"date": "string", "cell_id": "string", "class": "string", "split": "string"},
    )
    required = {"date", "cell_id", "class", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Missing columns in {split_csv}: {sorted(missing)}")

    split = split.dropna(subset=["date", "cell_id", "class"]).copy()
    for column in ["date", "cell_id", "class", "split"]:
        split[column] = split[column].str.strip()
    return split


def make_class_mapping(*frames: pd.DataFrame) -> dict[str, int]:
    labels = pd.concat([frame["class"] for frame in frames], ignore_index=True).dropna().astype(str)
    return {label: idx for idx, label in enumerate(sorted(labels.unique(), key=_natural_label_key))}


def _natural_label_key(label: str) -> tuple[int, int | str]:
    return (0, int(label)) if label.isdigit() else (1, label)


def cell_file_paths(
    annotated_root: Path,
    date: str,
    cell_id: str,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> list[Path]:
    cell_dir = annotated_root / str(date)
    return [cell_dir / f"{cell_id}{suffix}.txt" for suffix in feature_suffixes]


def find_missing_files(
    frame: pd.DataFrame,
    annotated_root: Path,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        for path in cell_file_paths(annotated_root, row.date, row.cell_id, feature_suffixes):
            if not path.exists():
                rows.append(
                    {
                        "date": row.date,
                        "cell_id": row.cell_id,
                        "class": row[2],
                        "missing_path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def summarize_nonfinite_values(
    frame: pd.DataFrame,
    annotated_root: Path,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        for suffix, path in zip(
            feature_suffixes,
            cell_file_paths(annotated_root, row.date, row.cell_id, feature_suffixes),
        ):
            array = _load_matrix(path)
            nan_count = int(np.isnan(array).sum())
            inf_count = int(np.isinf(array).sum())
            if nan_count or inf_count:
                result = {
                    "date": row.date,
                    "cell_id": row.cell_id,
                    "class": row[2],
                    "feature": suffix,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "path": str(path),
                }
                if hasattr(row, "split"):
                    result["split"] = row.split
                rows.append(result)
    return pd.DataFrame(rows)


def load_cell_array(
    annotated_root: Path,
    date: str,
    cell_id: str,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> np.ndarray:
    channels = []
    for path in cell_file_paths(annotated_root, date, cell_id, feature_suffixes):
        if not path.exists():
            raise FileNotFoundError(path)
        matrix = _load_matrix(path)
        if matrix.shape != (20, 20):
            raise ValueError(f"Expected 20x20 matrix in {path}, got {matrix.shape}")
        channels.append(matrix)
    return np.stack(channels, axis=0)


def _load_matrix(path: Path) -> np.ndarray:
    try:
        matrix = np.loadtxt(path, dtype=np.float32, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(path, dtype=np.float32)
    return _replace_missing_sentinels(matrix)


def _replace_missing_sentinels(matrix: np.ndarray) -> np.ndarray:
    matrix = np.array(matrix, dtype=np.float32, copy=True)
    for sentinel in MISSING_VALUE_SENTINELS:
        matrix[matrix == sentinel] = np.nan
    return matrix


class BeachCellDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        annotated_root: Path,
        class_to_idx: dict[str, int],
        image_size: int = 64,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.annotated_root = Path(annotated_root)
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.feature_suffixes = tuple(feature_suffixes)
        self.mean = _as_channel_tensor(mean) if mean is not None else None
        self.std = _as_channel_tensor(std) if std is not None else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        array = load_cell_array(
            self.annotated_root,
            str(row["date"]),
            str(row["cell_id"]),
            self.feature_suffixes,
        )
        x = torch.from_numpy(array)
        if self.augment:
            x = _augment_spatial(x)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.mean is not None and self.std is not None:
            x = torch.where(torch.isfinite(x), x, self.mean)
            x = (x - self.mean) / self.std.clamp_min(1e-6)
        else:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        y = torch.tensor(self.class_to_idx[str(row["class"])], dtype=torch.long)
        return x, y


def _augment_spatial(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(1,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(2,))
    rotations = int(torch.randint(0, 4, ()).item())
    if rotations:
        x = torch.rot90(x, k=rotations, dims=(1, 2))
    return x.contiguous()


def _as_channel_tensor(values: Sequence[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)


def compute_channel_stats(
    frame: pd.DataFrame,
    annotated_root: Path,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(len(feature_suffixes), dtype=np.float64)
    channel_sq_sum = np.zeros(len(feature_suffixes), dtype=np.float64)
    channel_count = np.zeros(len(feature_suffixes), dtype=np.float64)

    for row in frame.itertuples(index=False):
        array = load_cell_array(annotated_root, row.date, row.cell_id, feature_suffixes)
        finite = np.isfinite(array)
        cleaned = np.where(finite, array, 0.0)
        channel_sum += cleaned.sum(axis=(1, 2))
        channel_sq_sum += np.square(cleaned, dtype=np.float64).sum(axis=(1, 2))
        channel_count += finite.sum(axis=(1, 2))

    if np.any(channel_count == 0):
        empty_features = [
            feature for feature, count in zip(feature_suffixes, channel_count) if count == 0
        ]
        raise ValueError(f"No finite values found for feature(s): {empty_features}")

    mean = channel_sum / channel_count
    variance = channel_sq_sum / channel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(frame: pd.DataFrame, class_to_idx: dict[str, int]) -> torch.Tensor:
    counts = frame["class"].map(class_to_idx).value_counts().sort_index()
    weights = len(frame) / (len(class_to_idx) * counts)
    return torch.tensor(weights.reindex(range(len(class_to_idx))).to_numpy(), dtype=torch.float32)


def read_class_name_mapping(path: Path, known_classes: Iterable[str]) -> dict[str, str]:
    """Read encoded-class to class-name mapping from Excel when available.

    The expected layout is either named columns such as class/code/id and name/label,
    or a two-column sheet with encoded class in one column and class name in another.
    If the workbook instead contains descriptor-like rows, a numeric fallback is used.
    """
    known = {str(value) for value in known_classes}
    fallback = {label: f"class_{label}" for label in sorted(known, key=_natural_label_key)}

    if not path.exists():
        return fallback

    raw = pd.read_excel(path, dtype=object)
    if raw.empty:
        raw = pd.read_excel(path, header=None, dtype=object)
    if raw.empty:
        return fallback

    named = _mapping_from_named_columns(raw, known)
    if named:
        return {label: named.get(label, fallback[label]) for label in fallback}

    positional = _mapping_from_positional_columns(raw, known)
    if positional:
        return {label: positional.get(label, fallback[label]) for label in fallback}

    return fallback


def _mapping_from_named_columns(frame: pd.DataFrame, known_classes: set[str]) -> dict[str, str]:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    code_names = ["class", "class_id", "class code", "code", "id", "encoding", "encoded"]
    label_names = ["name", "class_name", "class name", "label", "description"]

    code_column = next((columns[name] for name in code_names if name in columns), None)
    label_column = next((columns[name] for name in label_names if name in columns), None)
    if code_column is None or label_column is None:
        return {}

    return _mapping_from_columns(frame, code_column, label_column, known_classes)


def _mapping_from_positional_columns(frame: pd.DataFrame, known_classes: set[str]) -> dict[str, str]:
    best_mapping: dict[str, str] = {}
    columns = list(frame.columns)
    for code_column in columns:
        codes = frame[code_column].dropna().map(lambda value: str(value).strip())
        code_overlap = set(codes) & known_classes
        if not code_overlap:
            continue
        for label_column in columns:
            if label_column == code_column:
                continue
            mapping = _mapping_from_columns(frame, code_column, label_column, known_classes)
            distinct_names = set(mapping.values())
            if len(mapping) > len(best_mapping) and len(distinct_names) > 1:
                best_mapping = mapping
    return best_mapping


def _mapping_from_columns(
    frame: pd.DataFrame,
    code_column: str | int,
    label_column: str | int,
    known_classes: set[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for code, label in frame[[code_column, label_column]].dropna().itertuples(index=False):
        code_text = str(code).strip()
        label_text = str(label).strip()
        if code_text in known_classes and label_text and label_text != code_text and not label_text.isdigit():
            mapping[code_text] = label_text
    return mapping
