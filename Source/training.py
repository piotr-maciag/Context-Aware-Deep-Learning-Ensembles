from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training", leave=False, disable=not show_progress):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=show_progress,
        )
        validation_metrics = evaluate_loss_accuracy(
            model,
            validation_loader,
            criterion,
            device,
            desc="Validation",
            show_progress=show_progress,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "is_best_epoch": False,
        }
        history.append(row)
        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.4f} | "
                f"val loss={row['validation_loss']:.4f}, acc={row['validation_accuracy']:.4f}",
                flush=True,
            )

        improved = validation_metrics["loss"] < (best_validation_loss - early_stopping_min_delta)
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            for previous_row in history:
                previous_row["is_best_epoch"] = False
            history[-1]["is_best_epoch"] = True
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(
                    f"Early stopping after epoch {epoch:03d}. "
                    f"Best validation loss was {best_validation_loss:.4f} at epoch {best_epoch:03d}.",
                    flush=True,
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return pd.DataFrame(history)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    y_probability = []

    for x, y in tqdm(loader, desc="Predicting", leave=False, disable=not show_progress):
        x = x.to(device)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)

        y_true.append(y.cpu().numpy())
        y_pred.append(probabilities.argmax(dim=1).cpu().numpy())
        y_probability.append(probabilities.cpu().numpy())

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_probability),
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[idx] for idx in labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
    return report_df, confusion_df
