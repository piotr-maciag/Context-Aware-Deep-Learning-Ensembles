from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from queue import Queue
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from Source.training import classification_metrics, fit_model, predict


@dataclass
class GridSearchResult:
    model: nn.Module
    config: dict[str, Any]
    history: pd.DataFrame
    validation_report: pd.DataFrame
    validation_score: float
    summary: pd.DataFrame
    metric_trace: pd.DataFrame


def run_grid_search(
    model_factory: Callable[[dict[str, Any]], nn.Module],
    parameter_grid: Iterable[dict[str, Any]],
    train_loader: DataLoader | None,
    validation_loader: DataLoader | None,
    class_weights: torch.Tensor,
    idx_to_class: dict[int, str],
    device: torch.device,
    epochs: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    random_seed: int,
    loader_factory: Callable[[dict[str, Any]], tuple[DataLoader, DataLoader]] | None = None,
) -> GridSearchResult:
    parameter_grid = list(parameter_grid)
    parallel_devices = _parallel_grid_devices(device, len(parameter_grid))
    if len(parallel_devices) > 1:
        return _run_grid_search_parallel(
            model_factory=model_factory,
            parameter_grid=parameter_grid,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loader_factory=loader_factory,
            class_weights=class_weights,
            idx_to_class=idx_to_class,
            return_device=device,
            devices=parallel_devices,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            random_seed=random_seed,
        )

    best_score = float("-inf")
    best_state = None
    best_config: dict[str, Any] | None = None
    best_history: pd.DataFrame | None = None
    best_report: pd.DataFrame | None = None
    summary_rows: list[dict[str, Any]] = []
    metric_trace_rows: list[dict[str, Any]] = []

    for run_index, config in enumerate(parameter_grid, start=1):
        result = _run_single_grid_config(
            run_index=run_index,
            config=config,
            model_factory=model_factory,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loader_factory=loader_factory,
            class_weights=class_weights,
            idx_to_class=idx_to_class,
            device=device,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            random_seed=random_seed,
            quiet=False,
        )

        summary_rows.append(result["summary_row"])
        metric_trace_rows.extend(result["metric_trace_rows"])

        if result["validation_score"] > best_score:
            best_score = result["validation_score"]
            best_config = dict(config)
            best_history = result["history"].copy()
            best_report = result["report"].copy()
            best_state = result["state_dict"]

    if best_state is None or best_config is None:
        raise RuntimeError("Grid search did not run any configurations.")

    best_model = model_factory(best_config).to(device)
    best_model.load_state_dict(best_state)
    summary = pd.DataFrame(summary_rows).sort_values(
        "validation_macro_f1", ascending=False
    )
    metric_trace = pd.DataFrame(metric_trace_rows)
    return GridSearchResult(
        model=best_model,
        config=best_config,
        history=best_history if best_history is not None else pd.DataFrame(),
        validation_report=best_report if best_report is not None else pd.DataFrame(),
        validation_score=best_score,
        summary=summary,
        metric_trace=metric_trace,
    )


def _parallel_grid_devices(device: torch.device, run_count: int) -> list[torch.device]:
    if run_count <= 1 or device.type != "cuda" or not torch.cuda.is_available():
        return []
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return []
    return [torch.device(f"cuda:{index}") for index in range(device_count)]


def _run_grid_search_parallel(
    model_factory: Callable[[dict[str, Any]], nn.Module],
    parameter_grid: list[dict[str, Any]],
    train_loader: DataLoader | None,
    validation_loader: DataLoader | None,
    loader_factory: Callable[[dict[str, Any]], tuple[DataLoader, DataLoader]] | None,
    class_weights: torch.Tensor,
    idx_to_class: dict[int, str],
    return_device: torch.device,
    devices: list[torch.device],
    epochs: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    random_seed: int,
) -> GridSearchResult:
    print(
        f"Running {len(parameter_grid)} grid configurations across "
        f"{len(devices)} CUDA devices.",
        flush=True,
    )
    device_queue: Queue[torch.device] = Queue()
    for device in devices:
        device_queue.put(device)

    futures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        for run_index, config in enumerate(parameter_grid, start=1):
            futures.append(
                executor.submit(
                    _run_single_grid_config_from_queue,
                    device_queue,
                    run_index,
                    config,
                    model_factory,
                    train_loader,
                    validation_loader,
                    loader_factory,
                    class_weights,
                    idx_to_class,
                    epochs,
                    early_stopping_patience,
                    early_stopping_min_delta,
                    random_seed,
                )
            )

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["summary_row"]
            print(
                f"Finished grid run {row['run_index']} on {row['device']}: "
                f"macro F1={row['validation_macro_f1']:.4f}, "
                f"epochs={row['epochs_trained']}, config={result['config']}",
                flush=True,
            )

    results = sorted(results, key=lambda result: result["summary_row"]["run_index"])
    summary_rows = [result["summary_row"] for result in results]
    metric_trace_rows = [
        row
        for result in results
        for row in result["metric_trace_rows"]
    ]
    best_result = max(results, key=lambda result: result["validation_score"])
    best_config = dict(best_result["config"])
    best_model = model_factory(best_config).to(return_device)
    best_model.load_state_dict(best_result["state_dict"])

    summary = pd.DataFrame(summary_rows).sort_values(
        "validation_macro_f1", ascending=False
    )
    metric_trace = pd.DataFrame(metric_trace_rows)
    return GridSearchResult(
        model=best_model,
        config=best_config,
        history=best_result["history"].copy(),
        validation_report=best_result["report"].copy(),
        validation_score=float(best_result["validation_score"]),
        summary=summary,
        metric_trace=metric_trace,
    )


def _run_single_grid_config_from_queue(
    device_queue: Queue,
    run_index: int,
    config: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], nn.Module],
    train_loader: DataLoader | None,
    validation_loader: DataLoader | None,
    loader_factory: Callable[[dict[str, Any]], tuple[DataLoader, DataLoader]] | None,
    class_weights: torch.Tensor,
    idx_to_class: dict[int, str],
    epochs: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    random_seed: int,
) -> dict[str, Any]:
    device = device_queue.get()
    try:
        print(
            f"Starting grid run {run_index} on {device}: config={config}",
            flush=True,
        )
        return _run_single_grid_config(
            run_index=run_index,
            config=config,
            model_factory=model_factory,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loader_factory=loader_factory,
            class_weights=class_weights,
            idx_to_class=idx_to_class,
            device=device,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            random_seed=random_seed,
            quiet=True,
        )
    finally:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        device_queue.put(device)


def _run_single_grid_config(
    run_index: int,
    config: dict[str, Any],
    model_factory: Callable[[dict[str, Any]], nn.Module],
    train_loader: DataLoader | None,
    validation_loader: DataLoader | None,
    loader_factory: Callable[[dict[str, Any]], tuple[DataLoader, DataLoader]] | None,
    class_weights: torch.Tensor,
    idx_to_class: dict[int, str],
    device: torch.device,
    epochs: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    random_seed: int,
    quiet: bool,
) -> dict[str, Any]:
    if not quiet:
        print(f"\nGrid search run {run_index}: {config}", flush=True)
    torch.manual_seed(random_seed + run_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed + run_index)

    if loader_factory is not None:
        current_train_loader, current_validation_loader = loader_factory(config)
    elif train_loader is not None and validation_loader is not None:
        current_train_loader = train_loader
        current_validation_loader = validation_loader
    else:
        raise ValueError(
            "Either train_loader and validation_loader, or loader_factory, must be provided."
        )

    model = model_factory(config).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = fit_model(
        model,
        current_train_loader,
        current_validation_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        checkpoint_path=None,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        verbose=not quiet,
        show_progress=not quiet,
    )

    y_true, y_pred, _ = predict(model, current_validation_loader, device, show_progress=not quiet)
    report, _ = classification_metrics(y_true, y_pred, idx_to_class)
    validation_score = float(report.loc["macro avg", "f1-score"])

    summary_row = dict(config)
    summary_row.update(
        {
            "run_index": run_index,
            "device": str(device),
            "epochs_trained": int(history["epoch"].max()),
            "best_validation_loss": float(history["validation_loss"].min()),
            "best_validation_accuracy": float(history["validation_accuracy"].max()),
            "validation_macro_f1": validation_score,
        }
    )
    metric_trace_rows = []
    for label, metrics in report.iterrows():
        metric_row = dict(config)
        metric_row["run_index"] = run_index
        metric_row["device"] = str(device)
        metric_row["label"] = label
        for metric_name, metric_value in metrics.items():
            metric_row[metric_name] = metric_value
        metric_trace_rows.append(metric_row)

    if not quiet:
        print(f"Validation macro F1: {validation_score:.4f}", flush=True)

    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    del model
    return {
        "config": dict(config),
        "history": history.copy(),
        "report": report.copy(),
        "validation_score": validation_score,
        "summary_row": summary_row,
        "metric_trace_rows": metric_trace_rows,
        "state_dict": state_dict,
    }
