"""Figure generation for the study.

All figures share one validated palette and one set of mark specifications, so
the report reads as a single system. Categorical hues are assigned in a fixed
slot order and never cycled; magnitude (the confusion matrices) uses a single
blue ramp light-to-dark; every multi-series chart carries a legend so identity
is never colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
#: Categorical slots, in fixed assignment order.
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Single-hue sequential ramp (blue, steps 100 -> 700) for magnitude encoding.
SEQUENTIAL: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Class colours, used consistently wherever benign and metastatic are compared.
CLASS_COLORS: dict[str, str] = {"benign": SERIES[0], "metastatic": SERIES[1]}

LINE_WIDTH = 2.0
MARKER_SIZE = 4.0  # points; 8px at the 2x figure DPI used here
_SEQ_CMAP = LinearSegmentedColormap.from_list("hp_blue", SEQUENTIAL)


def apply_style() -> None:
    """Install the shared Matplotlib style. Safe to call more than once."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _despine(axes: plt.Axes) -> None:
    """Drop the top and right spines so the data carries the frame."""
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def _save(figure: Figure, path: Path) -> Path:
    """Write ``figure`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    path: Path,
    metric_key: str | None = "auc",
    metric_label: str = "ROC AUC",
    title: str = "Training curves",
) -> Path:
    """Plot per-epoch loss and, optionally, a second metric for each run.

    Train and validation are two series on one axis; loss and the metric get
    separate panels rather than a second y-scale, so no panel ever carries two
    different units.

    Args:
        histories: Mapping of run name to a mapping with ``train_loss``,
            ``val_loss`` and optionally ``train_<metric>``/``val_<metric>``.
        path: Output PNG path.
        metric_key: Suffix of the second metric, or ``None`` for loss only.
        metric_label: Axis label for the second metric.
        title: Figure title.

    Returns:
        The written path.
    """
    apply_style()
    runs = list(histories)
    rows = 2 if metric_key else 1
    figure, grid = plt.subplots(
        rows, len(runs), figsize=(3.2 * len(runs), 2.7 * rows), squeeze=False
    )

    for column, run in enumerate(runs):
        record = histories[run]
        epochs = np.arange(1, len(record["train_loss"]) + 1)

        axes = grid[0][column]
        axes.plot(
            epochs, record["train_loss"], color=SERIES[0], marker="o", label="train"
        )
        axes.plot(
            epochs, record["val_loss"], color=SERIES[1], marker="o", label="validation"
        )
        axes.set_title(run)
        axes.set_xlabel("epoch")
        if column == 0:
            axes.set_ylabel("loss")
        axes.set_xticks(epochs)
        _despine(axes)
        if column == 0:
            axes.legend(loc="best")

        if metric_key:
            axes = grid[1][column]
            train_key, val_key = f"train_{metric_key}", f"val_{metric_key}"
            if train_key in record:
                axes.plot(
                    epochs, record[train_key], color=SERIES[0], marker="o", label="train"
                )
            axes.plot(
                epochs, record[val_key], color=SERIES[1], marker="o", label="validation"
            )
            axes.set_xlabel("epoch")
            if column == 0:
                axes.set_ylabel(metric_label)
                axes.legend(loc="best")
            axes.set_xticks(epochs)
            _despine(axes)

    figure.suptitle(title, color=INK_PRIMARY, fontsize=11, fontweight="semibold")
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
def plot_roc_overlay(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    path: Path,
    title: str = "ROC - metastasis detection on held-out test patches",
) -> Path:
    """Overlay ROC curves, labelling each series with its AUC.

    Args:
        curves: Mapping of model name to ``(fpr, tpr, auc)``. At most three
            series; the palette's all-pairs guarantee covers three slots.
        path: Output PNG path.
        title: Figure title.

    Returns:
        The written path.
    """
    apply_style()
    if len(curves) > len(SERIES):
        raise ValueError(
            f"{len(curves)} series exceeds the {len(SERIES)}-slot palette; "
            "facet instead of adding hues"
        )

    figure, axes = plt.subplots(figsize=(4.6, 4.4))
    axes.plot(
        [0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle=(0, (4, 3)),
        label="chance",
    )
    for slot, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        axes.plot(fpr, tpr, color=SERIES[slot], label=f"{name} - AUC {auc:.3f}")

    axes.set_xlabel("false positive rate (1 - specificity)")
    axes.set_ylabel("true positive rate (sensitivity)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.002)
    axes.set_title(title, fontsize=9.5)
    axes.legend(loc="lower right")
    axes.set_aspect("equal")
    _despine(axes)
    figure.tight_layout()
    return _save(figure, path)


def plot_single_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    path: Path,
    label: str,
    title: str,
) -> Path:
    """Plot one ROC curve. A single series needs no legend box, so the title
    names it and the AUC is annotated directly on the plot."""
    apply_style()
    figure, axes = plt.subplots(figsize=(4.2, 4.0))
    axes.plot([0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle=(0, (4, 3)))
    axes.plot(fpr, tpr, color=SERIES[0])
    axes.annotate(
        f"{label}\nAUC {auc:.3f}",
        xy=(0.62, 0.18), xycoords="axes fraction",
        color=INK_SECONDARY, fontsize=9, ha="left",
    )
    axes.annotate(
        "chance", xy=(0.70, 0.66), xycoords="axes fraction",
        color=INK_MUTED, fontsize=8, rotation=38,
    )
    axes.set_xlabel("false positive rate")
    axes.set_ylabel("true positive rate")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.002)
    axes.set_title(title, fontsize=9.5)
    axes.set_aspect("equal")
    _despine(axes)
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(
    matrices: Mapping[str, np.ndarray],
    path: Path,
    class_names: Sequence[str] = ("benign", "metastatic"),
    title: str = "Confusion matrices (test set)",
) -> Path:
    """Draw one confusion matrix per model as a row-normalised heatmap.

    Cells are shaded by row-normalised rate on a single blue ramp, and every
    cell carries its count and rate as a visible label, so the numbers never
    depend on reading a colour.

    Args:
        matrices: Mapping of model name to a 2x2 matrix, truth by prediction.
        path: Output PNG path.
        class_names: Display names of the negative and positive class.
        title: Figure title.

    Returns:
        The written path.
    """
    apply_style()
    names = list(matrices)
    columns = min(len(names), 2)
    rows = int(np.ceil(len(names) / columns))
    figure, grid = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.0 * rows), squeeze=False
    )

    for index, name in enumerate(names):
        axes = grid[index // columns][index % columns]
        matrix = np.asarray(matrices[name], dtype=np.float64)
        totals = matrix.sum(axis=1, keepdims=True)
        rates = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

        axes.imshow(rates, cmap=_SEQ_CMAP, vmin=0.0, vmax=1.0)
        axes.set_xticks([0, 1], labels=class_names)
        axes.set_yticks([0, 1], labels=class_names)
        axes.set_xlabel("predicted")
        axes.set_ylabel("actual")
        axes.set_title(name)
        axes.grid(False)
        for row in range(2):
            for column in range(2):
                axes.text(
                    column, row,
                    f"{int(matrix[row, column]):,}\n{rates[row, column]:.1%}",
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if rates[row, column] > 0.55 else INK_PRIMARY,
                )

    for empty in range(len(names), rows * columns):
        grid[empty // columns][empty % columns].axis("off")

    figure.suptitle(title, color=INK_PRIMARY, fontsize=11, fontweight="semibold")
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def plot_calibration(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    title: str = "Calibration - predicted vs observed tumour rate",
) -> Path:
    """Plot reliability curves against the perfect-calibration diagonal."""
    apply_style()
    figure, axes = plt.subplots(figsize=(4.6, 4.2))
    axes.plot(
        [0, 1], [0, 1], color=BASELINE, linewidth=1.0, linestyle=(0, (4, 3)),
        label="perfectly calibrated",
    )
    for slot, (name, (predicted, observed)) in enumerate(curves.items()):
        axes.plot(predicted, observed, color=SERIES[slot], marker="o", label=name)

    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed fraction metastatic")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title(title, fontsize=9.5)
    axes.legend(loc="upper left")
    axes.set_aspect("equal")
    _despine(axes)
    figure.tight_layout()
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# Image grids
# --------------------------------------------------------------------------- #
def plot_reconstruction_grid(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    path: Path,
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] = ("benign", "metastatic"),
    title: str = "Autoencoder reconstructions",
) -> Path:
    """Show input patches above their reconstructions.

    Args:
        inputs: ``(n, h, w, 3)`` float array in ``[0, 1]``.
        reconstructions: Same shape as ``inputs``.
        path: Output PNG path.
        labels: Optional per-patch class labels, shown above each column.
        class_names: Display names used for ``labels``.
        title: Figure title.

    Returns:
        The written path.
    """
    apply_style()
    count = len(inputs)
    figure, grid = plt.subplots(2, count, figsize=(1.05 * count, 2.5), squeeze=False)

    for column in range(count):
        for row, image in enumerate((inputs[column], reconstructions[column])):
            axes = grid[row][column]
            axes.imshow(np.clip(image, 0.0, 1.0))
            axes.set_xticks([])
            axes.set_yticks([])
            axes.grid(False)
            for spine in axes.spines.values():
                spine.set_color(GRIDLINE)
        if labels is not None:
            grid[0][column].set_title(
                class_names[int(labels[column])],
                fontsize=7,
                color=CLASS_COLORS[class_names[int(labels[column])]],
            )

    grid[0][0].set_ylabel("input", fontsize=8, color=INK_SECONDARY)
    grid[1][0].set_ylabel("output", fontsize=8, color=INK_SECONDARY)
    figure.suptitle(title, color=INK_PRIMARY, fontsize=11, fontweight="semibold")
    figure.tight_layout()
    return _save(figure, path)


def plot_patch_grid(
    images: np.ndarray,
    path: Path,
    labels: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    class_names: Sequence[str] = ("benign", "metastatic"),
    columns: int = 8,
    title: str = "Highest reconstruction error",
) -> Path:
    """Show a grid of patches annotated with class and score.

    The border of each tile is drawn in the class colour and the class name is
    printed, so the label never rests on colour alone.
    """
    apply_style()
    count = len(images)
    rows = int(np.ceil(count / columns))
    # Each tile needs headroom for a two-line caption, otherwise the captions of
    # one row overprint the images of the row above.
    figure, grid = plt.subplots(
        rows, columns, figsize=(1.15 * columns, 1.55 * rows), squeeze=False
    )

    for index in range(rows * columns):
        axes = grid[index // columns][index % columns]
        axes.set_xticks([])
        axes.set_yticks([])
        axes.grid(False)
        if index >= count:
            axes.axis("off")
            continue

        axes.imshow(np.clip(images[index], 0.0, 1.0))
        color = GRIDLINE
        caption = ""
        if labels is not None:
            name = class_names[int(labels[index])]
            color = CLASS_COLORS[name]
            caption = name
        if scores is not None:
            caption = f"{caption}\n{scores[index]:.4f}".strip()
        if caption:
            axes.set_title(caption, fontsize=6.5, color=INK_SECONDARY, pad=3)
        for spine in axes.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.6)

    figure.suptitle(title, color=INK_PRIMARY, fontsize=11, fontweight="semibold")
    figure.tight_layout()
    figure.subplots_adjust(hspace=0.42)
    return _save(figure, path)


# --------------------------------------------------------------------------- #
# Reconstruction-error distributions
# --------------------------------------------------------------------------- #
def plot_error_histograms(
    errors: np.ndarray,
    labels: np.ndarray,
    path: Path,
    threshold: float | None = None,
    class_names: Sequence[str] = ("benign", "metastatic"),
    bins: int = 60,
    title: str = "Reconstruction error by class",
) -> Path:
    """Plot per-class reconstruction-error distributions.

    Each class is drawn as a translucent fill with a 2px outline of the same hue
    so the overlap stays readable, is named in the legend and labelled again in
    the annotation, and the chosen threshold is marked.

    Args:
        errors: Per-patch reconstruction errors.
        labels: Ground-truth labels in ``{0, 1}``.
        path: Output PNG path.
        threshold: Optional decision threshold to mark.
        class_names: Display names of the two classes.
        bins: Histogram bin count.
        title: Figure title.

    Returns:
        The written path.
    """
    apply_style()
    score = np.asarray(errors, dtype=np.float64).ravel()
    truth = np.asarray(labels).astype(np.int64).ravel()
    edges = np.histogram_bin_edges(score, bins=bins)

    figure, axes = plt.subplots(figsize=(5.6, 3.6))
    for class_value, name in enumerate(class_names):
        subset = score[truth == class_value]
        if subset.size == 0:
            continue
        color = CLASS_COLORS[name]
        axes.hist(
            subset, bins=edges, density=True, color=color, alpha=0.35,
            label=f"{name} (n={subset.size:,}, mean {subset.mean():.4f})",
        )
        axes.hist(
            subset, bins=edges, density=True, histtype="step",
            color=color, linewidth=LINE_WIDTH,
        )

    if threshold is not None:
        axes.axvline(threshold, color=INK_SECONDARY, linewidth=1.2, linestyle=(0, (3, 2)))
        axes.annotate(
            f"threshold {threshold:.4f}",
            xy=(threshold, axes.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords="offset points",
            color=INK_SECONDARY, fontsize=8,
        )

    axes.set_xlabel("mean squared reconstruction error per patch")
    axes.set_ylabel("density")
    axes.set_title(title, fontsize=9.5)
    axes.legend(loc="upper right")
    _despine(axes)
    figure.tight_layout()
    return _save(figure, path)
