from __future__ import annotations

from math import ceil
from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def load_method_predictions(
    method_specs: Sequence[Mapping[str, object]],
    class_codes: Sequence[int],
) -> dict[str, dict[str, object]]:
    expected_classes = set(map(int, class_codes))
    loaded: dict[str, dict[str, object]] = {}
    reference_truth: pd.Series | None = None

    for spec in method_specs:
        name = str(spec["name"])
        result_type = str(spec["type"])
        path = Path(spec["path"])
        prediction_column = str(spec["prediction_column"])
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path, dtype={"date": "string", "cell_id": "string"})
        required = {"date", "cell_id", "true_class_code", prediction_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"Missing test identifiers or classes in {path}")
        if frame.duplicated(["date", "cell_id"]).any():
            raise ValueError(f"Duplicate date/cell_id pairs in {path}")

        frame = frame.sort_values(["date", "cell_id"]).reset_index(drop=True)
        y_true = pd.to_numeric(frame["true_class_code"], errors="raise").astype(int)
        y_pred = pd.to_numeric(frame[prediction_column], errors="raise").astype(int)
        observed = set(y_true) | set(y_pred)
        if not observed.issubset(expected_classes):
            raise ValueError(f"Unexpected classes in {path}: {sorted(observed - expected_classes)}")

        keyed_truth = frame.set_index(["date", "cell_id"])["true_class_code"].astype(int).sort_index()
        if reference_truth is None:
            reference_truth = keyed_truth
        elif not keyed_truth.equals(reference_truth):
            raise ValueError(f"Test samples or true labels in {path} differ from the reference")

        loaded[name] = {
            "type": result_type,
            "path": path,
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred.to_numpy(),
        }

    if not loaded:
        raise ValueError("No prediction methods were configured")
    return loaded


def calculate_method_metrics(
    predictions: Mapping[str, Mapping[str, object]],
    class_codes: Sequence[int],
    class_names: Mapping[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    labels = list(map(int, class_codes))
    aggregate_rows = []
    per_class_rows = []
    confusion_matrices: dict[str, np.ndarray] = {}

    for method, result in predictions.items():
        result_type = str(result["type"])
        y_true = np.asarray(result["y_true"], dtype=int)
        y_pred = np.asarray(result["y_pred"], dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )

        aggregate_rows.append(
            {
                "Method": method,
                "Type": result_type,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Macro precision": precision.mean(),
                "Macro recall": recall.mean(),
                "Macro F1": f1.mean(),
                "MCC": matthews_corrcoef(y_true, y_pred),
            }
        )

        for index, class_code in enumerate(labels):
            per_class_rows.append(
                {
                    "Method": method,
                    "Type": result_type,
                    "Class": class_code,
                    "Class name": class_names.get(class_code, f"class_{class_code}"),
                    "Precision": precision[index],
                    "Recall": recall[index],
                    "F1-score": f1[index],
                }
            )

        confusion_matrices[method] = confusion_matrix(y_true, y_pred, labels=labels)

    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(per_class_rows),
        confusion_matrices,
    )


def plot_exact_confusion_matrices(
    confusion_matrices: Mapping[str, np.ndarray],
    class_codes: Sequence[int],
    figure_title: str,
    columns: int = 3,
    panel_size: float | None = None,
    font_scale: float = 1.0,
) -> plt.Figure:
    rows = ceil(len(confusion_matrices) / columns)
    class_count = len(class_codes)
    effective_panel_size = panel_size or max(4.2, 0.50 * class_count)
    figure = plt.figure(
        figsize=(
            columns * effective_panel_size,
            rows * (effective_panel_size + 0.8),
        )
    )
    grid = figure.add_gridspec(rows, columns * 2)
    axes = []
    method_items = list(confusion_matrices.items())
    green_map = plt.get_cmap("Greens")
    red_map = plt.get_cmap("Reds")

    for index, (method, matrix) in enumerate(method_items):
        row = index // columns
        index_in_row = index % columns
        methods_in_row = min(columns, len(method_items) - row * columns)
        centering_offset = columns - methods_in_row
        grid_column = centering_offset + index_in_row * 2
        axis = figure.add_subplot(grid[row, grid_column : grid_column + 2])
        axes.append(axis)

        maximum = max(int(matrix.max()), 1)
        intensity = np.log1p(matrix) / np.log1p(maximum)
        colors = np.ones((*matrix.shape, 4), dtype=float)
        for true_index in range(class_count):
            for predicted_index in range(class_count):
                color_map = green_map if true_index == predicted_index else red_map
                colors[true_index, predicted_index] = color_map(
                    0.08 + 0.84 * intensity[true_index, predicted_index]
                )
        axis.imshow(colors, interpolation="nearest", aspect="equal")

        threshold = 0.58
        for true_index in range(class_count):
            for predicted_index in range(class_count):
                value = int(matrix[true_index, predicted_index])
                text_color = "white" if intensity[true_index, predicted_index] > threshold else "black"
                axis.text(
                    predicted_index,
                    true_index,
                    f"{value:d}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=(7 if class_count > 5 else 9) * font_scale,
                )

        display_title = {
            "Globally weighted probability voting": (
                "Globally weighted\nprobability voting"
            ),
            "Model-and-class weighted probability voting": (
                "Model-and-class weighted\nprobability voting"
            ),
        }.get(method, method)
        axis.set_title(
            display_title,
            fontsize=11 * font_scale,
            fontweight="bold",
        )
        axis.set_xlabel("Predicted class", fontsize=10 * font_scale)
        axis.set_ylabel("True class", fontsize=10 * font_scale)
        axis.set_xticks(range(class_count), labels=list(class_codes))
        axis.set_yticks(range(class_count), labels=list(class_codes))
        axis.tick_params(labelsize=9 * font_scale)

    legend_handles = [
        Patch(facecolor=green_map(0.65), edgecolor="black", label="Correct classification"),
        Patch(facecolor=red_map(0.65), edgecolor="black", label="Misclassification"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10 * font_scale,
    )
    figure.suptitle(
        figure_title,
        fontsize=14 * font_scale,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.025,
        "Color intensity is log-scaled; annotations show exact cell counts.",
        ha="center",
        fontsize=10 * font_scale,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure


def print_latex_metric_tables(
    aggregate_metrics: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    confusion_matrices: Mapping[str, np.ndarray],
    class_codes: Sequence[int],
    dataset_label: str,
    label_prefix: str,
    per_class_method_label: str,
) -> None:
    print("% Aggregate metrics")
    print(r"\begin{longtable}{lrrrrr}")
    print(
        rf"\caption{{Aggregated test-set metrics for the {dataset_label}.}} "
        rf"\label{{tab:{label_prefix}-aggregate-metrics}} \\"
    )
    print(r"\toprule")
    print(
        r"\textbf{Method} & \textbf{Acc.} & \textbf{Prec.} & "
        r"\textbf{Rec.} & \textbf{F1-score} & \textbf{MCC} \\"
    )
    print(r"\midrule")
    print(r"\endfirsthead")
    print(
        rf"\caption[]{{Aggregated test-set metrics for the {dataset_label}.}} \\"
    )
    print(r"\toprule")
    print(
        r"\textbf{Method} & \textbf{Acc.} & \textbf{Prec.} & "
        r"\textbf{Rec.} & \textbf{F1-score} & \textbf{MCC} \\"
    )
    print(r"\midrule")
    print(r"\endhead")
    print(r"\midrule")
    print(r"\multicolumn{6}{r}{Continued on next page} \\")
    print(r"\midrule")
    print(r"\endfoot")
    print(r"\bottomrule")
    print(r"\endlastfoot")
    wrapped_method_names = {
        "Globally weighted probability voting": (
            r"\shortstack[l]{Globally weighted\\probability voting}"
        ),
        "Model-and-class weighted probability voting": (
            r"\shortstack[l]{Model-and-class\\weighted probability\\voting}"
        ),
    }
    metric_columns = [
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "MCC",
    ]
    rounded_metrics = aggregate_metrics[metric_columns].round(3)
    best_values = rounded_metrics.max()
    previous_type = None
    for row_index, row in enumerate(aggregate_metrics.itertuples(index=False)):
        result_type = str(row[1])
        if previous_type is not None and result_type != previous_type:
            print(r"\midrule")
        previous_type = result_type
        method = wrapped_method_names.get(str(row[0]), str(row[0]))
        formatted_metrics = []
        for column_index, column in enumerate(metric_columns):
            value = rounded_metrics.iloc[row_index, column_index]
            formatted = f"{value:.3f}"
            if value == best_values[column]:
                formatted = rf"\cellcolor{{gray!25}}{formatted}"
            formatted_metrics.append(formatted)
        print(f"{method} & {' & '.join(formatted_metrics)} \\\\")
    print(r"\end{longtable}")

    print("% Per-class metrics")
    print(r"\begin{longtable}{rlrrr}")
    print(
        rf"\caption{{Per-class test-set metrics for \textbf{{{per_class_method_label}}} "
        rf"ensemble on the {dataset_label}.}} "
        rf"\label{{tab:{label_prefix}-per-class-metrics}} \\"
    )
    print(r"\toprule")
    print(
        r"\textbf{Class} & \textbf{Class name} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{F1-score} \\"
    )
    print(r"\midrule")
    print(r"\endfirsthead")
    print(
        rf"\caption[]{{Per-class test-set metrics for "
        rf"\textbf{{{per_class_method_label}}} ensemble on the {dataset_label}.}} \\"
    )
    print(r"\toprule")
    print(
        r"\textbf{Class} & \textbf{Class name} & \textbf{Precision} & "
        r"\textbf{Recall} & \textbf{F1-score} \\"
    )
    print(r"\midrule")
    print(r"\endhead")
    print(r"\midrule")
    print(r"\multicolumn{5}{r}{Continued on next page} \\")
    print(r"\midrule")
    print(r"\endfoot")
    print(r"\bottomrule")
    print(r"\endlastfoot")
    for row in per_class_metrics.itertuples(index=False):
        class_name = (
            str(row[1])
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("<", r"$<$")
            .replace(">", r"$>$")
        )
        print(
            f"{int(row[0])} & {class_name} & {row[2]:.3f} & "
            f"{row[3]:.3f} & {row[4]:.3f} \\\\"
        )
    print(r"\end{longtable}")

    print("% Exact confusion matrices")
    for method, matrix in confusion_matrices.items():
        slug = re.sub(r"[^a-z0-9]+", "-", method.lower()).strip("-")
        matrix_frame = pd.DataFrame(
            matrix,
            index=[f"True {code}" for code in class_codes],
            columns=[f"Pred. {code}" for code in class_codes],
        )
        print(f"% {method}")
        print(
            matrix_frame.to_latex(
                escape=True,
                caption=f"Exact test-set confusion matrix for {method} on the {dataset_label}.",
                label=f"tab:{label_prefix}-{slug}-confusion-matrix",
            )
        )


def print_latex_best_backbones_by_class(
    per_class_metrics: pd.DataFrame,
    dataset_label: str,
    label_prefix: str,
) -> None:
    backbone_metrics = per_class_metrics.loc[
        per_class_metrics["Type"] == "Model"
    ].copy()
    if backbone_metrics.empty:
        raise ValueError("No standalone model metrics are available")

    def escape_latex(value: object) -> str:
        return (
            str(value)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("<", r"$<$")
            .replace(">", r"$>$")
        )

    backbone_names = backbone_metrics["Method"].drop_duplicates().tolist()
    column_specification = "@{}rl" + ("c" * len(backbone_names)) + "@{}"
    total_columns = 2 + len(backbone_names)
    backbone_headers = " & ".join(
        {
            "Small CNN": r"\shortstack{\textbf{Small}\\\textbf{CNN}}",
            "EfficientNetV2-S": (
                r"\shortstack{\textbf{EfficientNet}\\\textbf{V2-S}}"
            ),
            "ViT-Tiny": r"\shortstack{\textbf{ViT}\\\textbf{Tiny}}",
            "Swin-Tiny": r"\shortstack{\textbf{Swin}\\\textbf{Tiny}}",
            "ConvNeXt-Tiny": (
                r"\shortstack{\textbf{ConvNeXt}\\\textbf{Tiny}}"
            ),
        }.get(name, rf"\textbf{{{escape_latex(name)}}}")
        for name in backbone_names
    )
    header = (
        r"\textbf{Class} & \textbf{Class name} & "
        f"{backbone_headers} \\\\"
    )

    print("% Best standalone backbone by class and metric")
    print(rf"\begin{{longtable}}{{{column_specification}}}")
    print(
        rf"\caption{{Standalone backbones achieving the highest class-specific "
        rf"F1-score in the {dataset_label}. An X identifies the best result; "
        rf"exact ties are retained.}} "
        rf"\label{{tab:{label_prefix}-best-backbone-per-class}} \\"
    )
    print(r"\toprule")
    print(header)
    print(r"\midrule")
    print(r"\endfirsthead")
    print(
        rf"\caption[]{{Standalone backbones achieving the highest "
        rf"class-specific F1-score in the {dataset_label}.}} \\"
    )
    print(r"\toprule")
    print(header)
    print(r"\midrule")
    print(r"\endhead")
    print(r"\midrule")
    print(
        rf"\multicolumn{{{total_columns}}}{{r}}{{Continued on next page}} \\"
    )
    print(r"\midrule")
    print(r"\endfoot")
    print(r"\bottomrule")
    print(r"\endlastfoot")

    class_rows = backbone_metrics[["Class", "Class name"]].drop_duplicates()
    for class_code, class_name in class_rows.itertuples(index=False, name=None):
        current_class = backbone_metrics.loc[
            backbone_metrics["Class"] == class_code
        ].set_index("Method")
        metric_values = current_class["F1-score"].astype(float)
        best_value = metric_values.max()
        winner_mask = np.isclose(
            metric_values.reindex(backbone_names).to_numpy(),
            best_value,
            rtol=0.0,
            atol=1e-12,
        )
        markers = ["\\textbf{X}" if winner else "" for winner in winner_mask]
        print(
            f"{int(class_code)} & {escape_latex(class_name)} & "
            f"{' & '.join(markers)} \\\\"
        )

    print(r"\end{longtable}")
