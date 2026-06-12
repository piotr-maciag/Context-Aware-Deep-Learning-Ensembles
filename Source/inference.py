from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from Source.beach_dataset import FEATURE_SUFFIXES, _as_channel_tensor, load_cell_array
from Source.models import build_convnext_tiny
from Source.training import get_device


def load_inference_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_complete_cells(
    date_dir: Path,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
) -> tuple[list[str], pd.DataFrame]:
    feature_to_ids: dict[str, set[str]] = {}
    for suffix in feature_suffixes:
        feature_to_ids[suffix] = {
            path.name[: -len(f"{suffix}.txt")]
            for path in date_dir.glob(f"*{suffix}.txt")
            if path.is_file()
        }

    complete_ids = set.intersection(*feature_to_ids.values()) if feature_to_ids else set()
    all_ids = set.union(*feature_to_ids.values()) if feature_to_ids else set()

    incomplete_rows = []
    for cell_id in sorted(all_ids - complete_ids, key=_cell_sort_key):
        missing = [suffix for suffix in feature_suffixes if cell_id not in feature_to_ids[suffix]]
        incomplete_rows.append(
            {
                "date": date_dir.name,
                "cell_id": cell_id,
                "missing_features": ",".join(missing),
            }
        )

    return sorted(complete_ids, key=_cell_sort_key), pd.DataFrame(incomplete_rows)


def discover_complete_cells_from_base_suffix(
    date_dir: Path,
    feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
    base_suffix: str = "r",
) -> tuple[list[str], int]:
    if base_suffix not in feature_suffixes:
        raise ValueError(f"base_suffix must be one of {feature_suffixes}, got {base_suffix}")

    complete_ids = []
    incomplete_count = 0
    for base_path in date_dir.glob(f"*{base_suffix}.txt"):
        if not base_path.is_file():
            continue
        cell_id = base_path.name[: -len(f"{base_suffix}.txt")]
        if all((date_dir / f"{cell_id}{suffix}.txt").exists() for suffix in feature_suffixes):
            complete_ids.append(cell_id)
        else:
            incomplete_count += 1

    return sorted(complete_ids, key=_cell_sort_key), incomplete_count


def discover_date_directories(source_root: Path) -> list[Path]:
    return sorted(
        [path for path in source_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )


class CellInferenceDataset(Dataset):
    def __init__(
        self,
        source_root: Path,
        date: str,
        cell_ids: Sequence[str],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        feature_suffixes: Sequence[str] = FEATURE_SUFFIXES,
    ) -> None:
        self.source_root = Path(source_root)
        self.date = str(date)
        self.cell_ids = [str(cell_id) for cell_id in cell_ids]
        self.image_size = int(image_size)
        self.feature_suffixes = tuple(feature_suffixes)
        self.mean = _as_channel_tensor(mean)
        self.std = _as_channel_tensor(std)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        cell_id = self.cell_ids[index]
        array = load_cell_array(
            self.source_root,
            self.date,
            cell_id,
            self.feature_suffixes,
        )
        x = torch.from_numpy(array)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        x = torch.where(torch.isfinite(x), x, self.mean)
        x = (x - self.mean) / self.std.clamp_min(1e-6)
        return x, cell_id


def load_convnext_for_inference(
    checkpoint_path: Path,
    metadata: dict,
    device: torch.device | None = None,
) -> torch.nn.Module:
    if device is None:
        device = get_device()

    idx_to_class_code = {
        int(index): str(class_code)
        for index, class_code in metadata["idx_to_class_code"].items()
    }
    model = build_convnext_tiny(
        num_classes=len(idx_to_class_code),
        in_channels=len(metadata["feature_suffixes"]),
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_cells(
    model: torch.nn.Module,
    loader: DataLoader,
    idx_to_class_code: dict[int, str],
    device: torch.device,
    progress_description: str = "Classifying cells",
) -> pd.DataFrame:
    rows = []
    for x, cell_ids in tqdm(loader, desc=progress_description, leave=False):
        x = x.to(device)
        predicted_indices = model(x).argmax(dim=1).cpu().tolist()
        for cell_id, predicted_index in zip(cell_ids, predicted_indices):
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "predicted_class": idx_to_class_code[int(predicted_index)],
                }
            )
    return pd.DataFrame(rows)


def _cell_sort_key(cell_id: str) -> tuple[int, int | str]:
    return (0, int(cell_id)) if str(cell_id).isdigit() else (1, str(cell_id))
