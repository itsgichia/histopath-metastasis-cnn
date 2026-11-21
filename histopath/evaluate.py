"""Evaluation metrics for binary metastasis detection.

Everything here operates on plain NumPy arrays of true labels and predicted
positive-class probabilities, so the same functions score the supervised
classifiers and the anomaly detector. Metrics are reported as a flat dictionary
that maps directly onto a CSV row.

Sensitivity is reported alongside specificity throughout: in sentinel lymph node
assessment a missed metastasis is the costly error, so a model's recall at a
usable specificity matters more than its headline accuracy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

#: Metric keys written to ``results/metrics.csv``, in display order.
METRIC_COLUMNS: tuple[str, ...] = (
    "accuracy",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
    "brier",
    "ece",
    "threshold",
    "n_samples",
    "tn",
    "fp",
    "fn",
    "tp",
)


@dataclass(frozen=True)
class CalibrationCurve:
    """Binned reliability data for a probability forecast.

    Attributes:
        bin_centers: Centre of each occupied probability bin.
        mean_predicted: Mean predicted probability inside each bin.
        fraction_positive: Observed positive rate inside each bin.
        counts: Number of samples in each bin.
    """

    bin_centers: np.ndarray
    mean_predicted: np.ndarray
    fraction_positive: np.ndarray
    counts: np.ndarray


def _as_arrays(
    y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D arrays and check their shapes agree."""
    true = np.asarray(y_true).astype(np.int64).ravel()
    prob = np.asarray(y_prob).astype(np.float64).ravel()
    if true.shape != prob.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {prob.shape}")
    if true.size == 0:
        raise ValueError("empty input")
    return true, prob


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return the equal-width expected calibration error.

    ECE is the sample-weighted mean absolute gap between confidence and observed
    accuracy across probability bins. Zero means perfectly calibrated.
    """
    true, prob = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    error = 0.0
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        weight = mask.mean()
        error += weight * abs(prob[mask].mean() - true[mask].mean())
    return float(error)


def calibration_curve_points(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Bin predictions into a reliability curve, dropping empty bins."""
    true, prob = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    centers: list[float] = []
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        centers.append(float((edges[index] + edges[index + 1]) / 2))
        predicted.append(float(prob[mask].mean()))
        observed.append(float(true[mask].mean()))
        counts.append(int(mask.sum()))

    return CalibrationCurve(
        bin_centers=np.asarray(centers),
        mean_predicted=np.asarray(predicted),
        fraction_positive=np.asarray(observed),
        counts=np.asarray(counts),
    )


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float]:
    """Score a probabilistic binary classifier.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        y_prob: Predicted probability of the positive (metastatic) class.
        threshold: Decision threshold applied to ``y_prob``.
        n_bins: Bin count for the calibration error.

    Returns:
        Dictionary keyed by :data:`METRIC_COLUMNS`. Threshold-free metrics
        (``roc_auc``, ``average_precision``) are ``nan`` when only one class is
        present, because they are undefined there.
    """
    true, prob = _as_arrays(y_true, y_prob)
    predicted = (prob >= threshold).astype(np.int64)
    both_classes = np.unique(true).size > 1

    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "roc_auc": float(roc_auc_score(true, prob)) if both_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(true, prob)) if both_classes else float("nan")
        ),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "brier": float(brier_score_loss(true, prob)) if both_classes else float("nan"),
        "ece": expected_calibration_error(true, prob, n_bins),
        "threshold": float(threshold),
        "n_samples": float(true.size),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def confusion(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return the 2x2 confusion matrix with rows = truth, columns = prediction."""
    true, prob = _as_arrays(y_true, y_prob)
    return confusion_matrix(true, (prob >= threshold).astype(np.int64), labels=[0, 1])


def roc_points(
    y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(fpr, tpr, auc)`` for a probability forecast."""
    true, prob = _as_arrays(y_true, y_prob)
    fpr, tpr, _ = roc_curve(true, prob)
    return fpr, tpr, float(roc_auc_score(true, prob))


def youden_threshold(
    y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray
) -> float:
    """Return the threshold maximising Youden's J (sensitivity + specificity - 1).

    This is a balanced, prevalence-independent operating point. It is selected on
    validation data only and then applied unchanged to the test set.
    """
    true, prob = _as_arrays(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(true, prob)
    best = int(np.argmax(tpr - fpr))
    chosen = float(thresholds[best])
    # roc_curve prepends an infinite threshold; fall back to the next point.
    if not np.isfinite(chosen):
        chosen = float(thresholds[min(best + 1, thresholds.size - 1)])
    return chosen


def write_metrics_csv(rows: Iterable[dict[str, object]], path: Path) -> Path:
    """Write metric rows to ``path`` as CSV and return the path.

    The ``model`` key is written first, then :data:`METRIC_COLUMNS`, then any
    remaining keys in insertion order.
    """
    materialised = list(rows)
    if not materialised:
        raise ValueError("no metric rows to write")

    ordered = ["model", *METRIC_COLUMNS]
    extras = [
        key
        for row in materialised
        for key in row
        if key not in ordered
    ]
    seen: set[str] = set()
    fieldnames = ordered + [k for k in extras if not (k in seen or seen.add(k))]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path
