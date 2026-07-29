from __future__ import annotations

from math import ceil
from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
)

CLASS_LINE_COLORS = {
    0: "#666666",   # Snow: darkened from white for visibility
    1: "#FDBF6F",   # Dry sediment
    2: "#FF7F0E",   # Wet sediment
    3: "#1F78B4",   # Water
    4: "#A6A6A6",   # Bedrock
    5: "#D9D9D9",   # Boulder
    6: "#000000",   # Driftwood
    7: "#22A447",   # Algae
    8: "#9ECAE1",   # Ice
    9: "#C44EAA",   # Anthropogenic
    11: "#FDBF6F",  # Fine grain size
    12: "#FF7F0E",  # Medium grain size
    13: "#E31A1C",  # Coarse grain size
}


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
        if prediction_column == "predicted_class_code":
            probability_prefix = ""
        elif prediction_column.endswith("_predicted_class_code"):
            probability_prefix = prediction_column.removesuffix(
                "predicted_class_code"
            )
        else:
            raise ValueError(
                f"Cannot infer probability columns from {prediction_column}"
            )
        probability_columns = [
            f"{probability_prefix}probability_class_{class_code}"
            for class_code in class_codes
        ]
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path, dtype={"date": "string", "cell_id": "string"})
        required = {
            "date",
            "cell_id",
            "true_class_code",
            prediction_column,
            *probability_columns,
        }
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
        y_probability = frame[probability_columns].apply(
            pd.to_numeric,
            errors="raise",
        ).to_numpy(dtype=float)
        if not np.isfinite(y_probability).all():
            raise ValueError(f"Non-finite class probabilities in {path}")
        if ((y_probability < 0) | (y_probability > 1)).any():
            raise ValueError(f"Class probabilities outside [0, 1] in {path}")
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
            "y_probability": y_probability,
        }

    if not loaded:
        raise ValueError("No prediction methods were configured")
    return loaded


def load_best_grid_search_configurations(
    method_specs: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for spec in method_specs:
        if str(spec["type"]) != "Model":
            continue
        result_path = Path(spec["path"]).parent / "best_grid_search_result.csv"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result = pd.read_csv(result_path)
        if len(result) != 1:
            raise ValueError(
                f"Expected one best grid-search result in {result_path}, "
                f"found {len(result)}"
            )
        selected = result.iloc[0]
        rows.append(
            {
                "Model": str(spec["name"]),
                "Learning rate": float(selected["learning_rate"]),
                "Weight decay": float(selected["weight_decay"]),
                "Dropout": (
                    float(selected["dropout"])
                    if "dropout" in result.columns
                    else np.nan
                ),
                "Context radius": int(selected["context_radius"]),
                "Validation macro F1": float(selected["validation_macro_f1"]),
            }
        )
    if not rows:
        raise ValueError("No standalone model grid-search results are available")
    return pd.DataFrame(rows)


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
        bbox_to_anchor=(0.5, 0.005),
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
        0.042,
        "Color intensity is log-scaled; annotations show exact cell counts.",
        ha="center",
        fontsize=10 * font_scale,
    )
    figure.tight_layout(rect=(0, 0.085, 1, 0.97))
    return figure


def plot_multiclass_precision_recall_curves(
    predictions: Mapping[str, Mapping[str, object]],
    class_codes: Sequence[int],
    class_names: Mapping[int, str],
    figure_title: str,
    columns: int = 3,
    panel_size: float = 4.2,
    font_scale: float = 1.0,
) -> plt.Figure:
    method_items = list(predictions.items())
    rows = ceil(len(method_items) / columns)
    labels = list(map(int, class_codes))
    class_count = len(labels)
    legend_item_count = class_count + 1
    legend_columns = min(4, legend_item_count)
    legend_rows = ceil(legend_item_count / legend_columns)
    horizontal_gap_ratio = 0.22
    grid_width_ratios = []
    for column_index in range(columns):
        grid_width_ratios.extend([1.0, 1.0])
        if column_index < columns - 1:
            grid_width_ratios.append(horizontal_gap_ratio)
    figure = plt.figure(
        figsize=(
            (
                columns
                + (columns - 1) * horizontal_gap_ratio / 2
            )
            * panel_size,
            rows * (panel_size + 1.15) + legend_rows * 0.30,
        )
    )
    grid = figure.add_gridspec(
        rows,
        len(grid_width_ratios),
        width_ratios=grid_width_ratios,
        hspace=0.48,
        wspace=0.0,
    )
    fallback_color_map = plt.get_cmap("tab10")
    class_colors = [
        CLASS_LINE_COLORS.get(class_code, fallback_color_map(index % 10))
        for index, class_code in enumerate(labels)
    ]

    for index, (method, result) in enumerate(method_items):
        row = index // columns
        index_in_row = index % columns
        methods_in_row = min(columns, len(method_items) - row * columns)
        if methods_in_row == 1:
            panel_slots = np.asarray([columns // 2])
        else:
            panel_slots = np.linspace(
                0,
                columns - 1,
                methods_in_row,
                dtype=int,
            )
        grid_column = int(panel_slots[index_in_row]) * 3
        axis = figure.add_subplot(grid[row, grid_column : grid_column + 2])

        y_true = np.asarray(result["y_true"], dtype=int)
        y_probability = np.asarray(result["y_probability"], dtype=float)
        y_binary = np.column_stack([y_true == class_code for class_code in labels])
        recall_grid = np.linspace(0.0, 1.0, 501)
        interpolated_precisions = []
        class_average_precisions = []
        for class_index, color in enumerate(class_colors):
            binary_truth = y_binary[:, class_index].astype(int)
            if np.unique(binary_truth).size < 2:
                continue
            precision, recall, _ = precision_recall_curve(
                binary_truth,
                y_probability[:, class_index],
            )
            class_average_precisions.append(
                average_precision_score(
                    binary_truth,
                    y_probability[:, class_index],
                )
            )
            interpolated_precisions.append(
                np.interp(
                    recall_grid,
                    recall[::-1],
                    precision[::-1],
                )
            )
            axis.plot(
                recall,
                precision,
                color=color,
                linewidth=1.35 * font_scale,
                alpha=0.9,
            )

        if not interpolated_precisions:
            raise ValueError(
                f"Precision-recall curves cannot be calculated for {method}"
            )
        mean_precision = np.mean(interpolated_precisions, axis=0)
        macro_average_precision = float(np.mean(class_average_precisions))
        axis.plot(
            recall_grid,
            mean_precision,
            color="black",
            linewidth=2.2 * font_scale,
            linestyle="--",
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
            f"{display_title}\nMacro AP = {macro_average_precision:.3f}",
            fontsize=11 * font_scale,
            fontweight="bold",
        )
        axis.set_xlabel("Recall", fontsize=10 * font_scale)
        axis.set_ylabel("Precision", fontsize=10 * font_scale)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="0.88", linewidth=0.7, alpha=0.8)
        axis.tick_params(labelsize=9 * font_scale)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2,
            label=f"{class_code}: {class_names.get(class_code, class_code)}",
        )
        for class_code, color in zip(labels, class_colors)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.2,
            linestyle="--",
            label="Macro-average PR",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=legend_columns,
        frameon=False,
        fontsize=10 * font_scale,
    )
    figure.suptitle(
        figure_title,
        fontsize=14 * font_scale,
        fontweight="bold",
        y=0.995,
    )
    bottom_margin = 0.07 + legend_rows * 0.018
    figure.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=bottom_margin,
        top=0.93,
    )
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


def print_latex_best_grid_search_configurations(
    configurations: pd.DataFrame,
    dataset_label: str,
    label_prefix: str,
) -> None:
    header = (
        r"\textbf{Model} & "
        r"\shortstack{\textbf{Learning}\\\textbf{rate}} & "
        r"\shortstack{\textbf{Weight}\\\textbf{decay}} & "
        r"\textbf{Dropout} & "
        r"\shortstack{\textbf{Context}\\\textbf{radius}} & "
        r"\shortstack{\textbf{Validation}\\\textbf{F1-score}} \\"
    )
    print("% Best-performing grid-search configurations")
    print(r"\begin{longtable}{@{}lrrrrr@{}}")
    print(
        rf"\caption{{Best-performing grid-search configurations for the "
        rf"{dataset_label}, selected according to macro-averaged F1-score "
        rf"on the validation set.}} "
        rf"\label{{tab:{label_prefix}-best-grid-search-configurations}} \\"
    )
    print(r"\toprule")
    print(header)
    print(r"\midrule")
    print(r"\endfirsthead")
    print(
        rf"\caption[]{{Best-performing grid-search configurations for the "
        rf"{dataset_label}.}} \\"
    )
    print(r"\toprule")
    print(header)
    print(r"\midrule")
    print(r"\endhead")
    print(r"\midrule")
    print(r"\multicolumn{6}{r}{Continued on next page} \\")
    print(r"\midrule")
    print(r"\endfoot")
    print(r"\bottomrule")
    print(r"\endlastfoot")

    for row in configurations.itertuples(index=False):
        dropout = "--" if pd.isna(row[3]) else f"{row[3]:g}"
        print(
            f"{row[0]} & {row[1]:.0e} & {row[2]:.0e} & {dropout} & "
            f"{int(row[4])} & {row[5]:.3f} \\\\"
        )
    print(r"\end{longtable}")
