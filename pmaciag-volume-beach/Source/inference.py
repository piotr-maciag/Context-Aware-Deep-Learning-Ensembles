from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from queue import Queue
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from Source.beach_dataset import FEATURE_SUFFIXES, _as_channel_tensor, load_cell_context_array
from Source.models import (
    build_convnext_tiny,
    build_efficientnetv2_s,
    build_small_cnn,
    build_swin_tiny,
    build_vit_tiny,
)
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
        context_radius: int = 0,
    ) -> None:
        self.source_root = Path(source_root)
        self.date = str(date)
        self.cell_ids = [str(cell_id) for cell_id in cell_ids]
        self.image_size = int(image_size)
        self.feature_suffixes = tuple(feature_suffixes)
        self.context_radius = int(context_radius)
        self.mean = _as_channel_tensor(mean)
        self.std = _as_channel_tensor(std)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        cell_id = self.cell_ids[index]
        array = load_cell_context_array(
            self.source_root,
            self.date,
            cell_id,
            self.feature_suffixes,
            context_radius=self.context_radius,
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


def load_model_for_inference(
    checkpoint_path: Path,
    metadata: dict,
    device: torch.device | None = None,
) -> torch.nn.Module:
    if device is None:
        device = get_device()

    model_type = metadata.get("model_type", "convnext_tiny")
    idx_to_class_code = {
        int(index): str(class_code)
        for index, class_code in metadata["idx_to_class_code"].items()
    }
    feature_suffixes = metadata["feature_suffixes"]
    best_config = metadata.get("best_config", {})

    if model_type == "convnext_tiny":
        model = build_convnext_tiny(
            num_classes=len(idx_to_class_code),
            in_channels=len(feature_suffixes),
            pretrained=False,
        )
    elif model_type == "vit_tiny":
        model = build_vit_tiny(
            num_classes=len(idx_to_class_code),
            in_channels=len(feature_suffixes),
            pretrained=False,
            model_name=metadata.get("model_name", "vit_tiny_patch16_224"),
            dropout=float(best_config.get("dropout", 0.0)),
        )
    elif model_type == "small_cnn":
        model = build_small_cnn(
            num_classes=len(idx_to_class_code),
            in_channels=len(feature_suffixes),
            dropout=float(best_config.get("dropout", 0.25)),
        )
    elif model_type == "swin_tiny":
        model = build_swin_tiny(
            num_classes=len(idx_to_class_code),
            in_channels=len(feature_suffixes),
            pretrained=False,
            model_name=metadata.get("model_name", "swin_tiny_patch4_window7_224"),
            dropout=float(best_config.get("dropout", 0.0)),
        )
    elif model_type == "efficientnetv2_s":
        model = build_efficientnetv2_s(
            num_classes=len(idx_to_class_code),
            in_channels=len(feature_suffixes),
            pretrained=False,
            model_name=metadata.get("model_name", "tf_efficientnetv2_s"),
            dropout=float(best_config.get("dropout", 0.2)),
        )
    else:
        raise ValueError(f"Unsupported model_type in inference metadata: {model_type}")

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
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    for x, cell_ids in tqdm(
        loader,
        desc=progress_description,
        leave=False,
        disable=not show_progress,
    ):
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


def classify_date_directory(
    date_dir: Path,
    source_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    metadata: dict,
    idx_to_class_code: dict[int, str],
    context_radius: int,
    batch_size: int,
    num_workers: int,
    base_suffix: str,
    device: torch.device,
    show_progress: bool = True,
) -> dict:
    date = date_dir.name
    cell_ids, incomplete_count = discover_complete_cells_from_base_suffix(
        date_dir,
        feature_suffixes=metadata["feature_suffixes"],
        base_suffix=base_suffix,
    )
    if not cell_ids:
        return {
            "date": date,
            "classified_cells": 0,
            "incomplete_candidates": incomplete_count,
            "device": str(device),
            "output_path": None,
        }

    model = load_model_for_inference(checkpoint_path, metadata, device=device)
    dataset = CellInferenceDataset(
        source_root=source_root,
        date=date,
        cell_ids=cell_ids,
        image_size=metadata["image_size"],
        mean=metadata["channel_mean"],
        std=metadata["channel_std"],
        feature_suffixes=metadata["feature_suffixes"],
        context_radius=context_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    predictions = predict_cells(
        model,
        loader,
        idx_to_class_code,
        device,
        progress_description=f"Classifying {date} ({len(cell_ids):,} cells)",
        show_progress=show_progress,
    )
    predictions.insert(0, "date", date)
    predictions = predictions[["date", "cell_id", "predicted_class"]]

    output_dir = output_root / date
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification.csv"
    predictions.to_csv(output_path, index=False)

    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    return {
        "date": date,
        "classified_cells": len(predictions),
        "incomplete_candidates": incomplete_count,
        "device": str(device),
        "output_path": output_path,
    }


def classify_date_directories(
    date_dirs: Sequence[Path],
    source_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    metadata: dict,
    idx_to_class_code: dict[int, str],
    context_radius: int,
    batch_size: int,
    num_workers: int,
    base_suffix: str = "r",
    device: torch.device | None = None,
) -> pd.DataFrame:
    if device is None:
        device = get_device()

    output_root.mkdir(parents=True, exist_ok=True)
    date_dirs = list(date_dirs)
    devices = _parallel_inference_devices(device, len(date_dirs))
    if len(devices) <= 1:
        rows = []
        for date_dir in tqdm(date_dirs, desc="Classifying date folders"):
            row = classify_date_directory(
                date_dir=date_dir,
                source_root=source_root,
                output_root=output_root,
                checkpoint_path=checkpoint_path,
                metadata=metadata,
                idx_to_class_code=idx_to_class_code,
                context_radius=context_radius,
                batch_size=batch_size,
                num_workers=num_workers,
                base_suffix=base_suffix,
                device=device,
                show_progress=True,
            )
            rows.append(row)
            _print_classification_summary(row)
        return pd.DataFrame(rows)

    print(
        f"Classifying {len(date_dirs)} date folders across "
        f"{len(devices)} CUDA devices.",
        flush=True,
    )
    device_queue: Queue[torch.device] = Queue()
    for worker_device in devices:
        device_queue.put(worker_device)

    futures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for date_dir in date_dirs:
            futures.append(
                executor.submit(
                    _classify_date_directory_from_queue,
                    device_queue,
                    date_dir,
                    source_root,
                    output_root,
                    checkpoint_path,
                    metadata,
                    idx_to_class_code,
                    context_radius,
                    batch_size,
                    num_workers,
                    base_suffix,
                )
            )

        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            _print_classification_summary(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _parallel_inference_devices(device: torch.device, work_count: int) -> list[torch.device]:
    if work_count <= 1 or device.type != "cuda" or not torch.cuda.is_available():
        return []
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return []
    return [torch.device(f"cuda:{index}") for index in range(device_count)]


def _classify_date_directory_from_queue(
    device_queue: Queue,
    date_dir: Path,
    source_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    metadata: dict,
    idx_to_class_code: dict[int, str],
    context_radius: int,
    batch_size: int,
    num_workers: int,
    base_suffix: str,
) -> dict:
    device = device_queue.get()
    try:
        return classify_date_directory(
            date_dir=date_dir,
            source_root=source_root,
            output_root=output_root,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            idx_to_class_code=idx_to_class_code,
            context_radius=context_radius,
            batch_size=batch_size,
            num_workers=num_workers,
            base_suffix=base_suffix,
            device=device,
            show_progress=False,
        )
    finally:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        device_queue.put(device)


def _print_classification_summary(row: dict) -> None:
    print(
        f"Finished {row['date']} on {row['device']}: "
        f"classified {row['classified_cells']:,}, "
        f"skipped {row['incomplete_candidates']:,}, "
        f"saved {row['output_path']}",
        flush=True,
    )


def _cell_sort_key(cell_id: str) -> tuple[int, int | str]:
    return (0, int(cell_id)) if str(cell_id).isdigit() else (1, str(cell_id))
