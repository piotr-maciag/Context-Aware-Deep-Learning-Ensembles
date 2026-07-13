from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
    best_score = float("-inf")
    best_state = None
    best_model = None
    best_config: dict[str, Any] | None = None
    best_history: pd.DataFrame | None = None
    best_report: pd.DataFrame | None = None
    summary_rows: list[dict[str, Any]] = []
    metric_trace_rows: list[dict[str, Any]] = []

    for run_index, config in enumerate(parameter_grid, start=1):
        print(f"\nGrid search run {run_index}: {config}")
        torch.manual_seed(random_seed + run_index)

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
        )

        y_true, y_pred, _ = predict(model, current_validation_loader, device)
        report, _ = classification_metrics(y_true, y_pred, idx_to_class)
        validation_score = float(report.loc["macro avg", "f1-score"])

        row = dict(config)
        row.update(
            {
                "run_index": run_index,
                "epochs_trained": int(history["epoch"].max()),
                "best_validation_loss": float(history["validation_loss"].min()),
                "best_validation_accuracy": float(history["validation_accuracy"].max()),
                "validation_macro_f1": validation_score,
            }
        )
        summary_rows.append(row)
        for label, metrics in report.iterrows():
            metric_row = dict(config)
            metric_row["run_index"] = run_index
            metric_row["label"] = label
            for metric_name, metric_value in metrics.items():
                metric_row[metric_name] = metric_value
            metric_trace_rows.append(metric_row)
        print(f"Validation macro F1: {validation_score:.4f}")

        if validation_score > best_score:
            best_score = validation_score
            best_config = dict(config)
            best_history = history.copy()
            best_report = report.copy()
            best_model = model
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            del model

    if best_model is None or best_state is None or best_config is None:
        raise RuntimeError("Grid search did not run any configurations.")

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
