"""Unsupervised anomaly detection on reconstruction error.

The premise: train an autoencoder on benign lymph node tissue only, so it
learns to reconstruct healthy morphology well and nothing else. At test time
every patch is scored by how badly it reconstructs. If metastatic tissue is
genuinely "unlike healthy tissue", tumour patches should carry higher error, and
the error alone should separate the classes without the model ever having seen a
tumour label.

The scorer is a pure ranking, so it is evaluated with ROC AUC first; a threshold
is only chosen afterwards, on validation data, and then applied unchanged to the
test set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .evaluate import classification_metrics, youden_threshold

LOGGER = logging.getLogger(__name__)

ThresholdMethod = Literal["youden", "percentile"]


@dataclass(frozen=True)
class ThresholdChoice:
    """A selected anomaly threshold and how it was obtained.

    Attributes:
        value: The error value above which a patch is flagged.
        method: Selection rule used.
        percentile: Percentile of benign validation errors, when applicable.
    """

    value: float
    method: ThresholdMethod
    percentile: float | None = None


def reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every patch in ``loader`` by its mean squared reconstruction error.

    Args:
        model: Trained autoencoder.
        loader: Loader yielding ``(image, target)``. The target is used only as
            the label when it is scalar per sample; when the loader is an
            autoencoder loader (target is the image) the returned labels are
            all-zero and should be ignored.
        device: Compute device; defaults to CPU.

    Returns:
        ``(errors, labels)``: per-patch mean squared error over all pixels and
        channels, and the accompanying labels.
    """
    target_device = device or torch.device("cpu")
    model.to(target_device).eval()

    errors: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(target_device)
            reconstruction = model(inputs)
            per_patch = ((reconstruction - inputs) ** 2).flatten(1).mean(dim=1)
            errors.append(per_patch.float().cpu().numpy())
            as_array = np.asarray(target.detach().cpu())
            labels.append(
                as_array.astype(np.float64).ravel()
                if as_array.ndim == 1
                else np.zeros(inputs.shape[0], dtype=np.float64)
            )

    return np.concatenate(errors), np.concatenate(labels)


def error_auc(
    errors: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
) -> float:
    """Return the ROC AUC of reconstruction error used directly as a score.

    No threshold is involved: this measures only whether tumour patches rank
    above benign ones by error. An AUC near 0.5 means the error carries no class
    signal; below 0.5 means tumour tissue reconstructs *better* than benign,
    which is itself informative.
    """
    score = np.asarray(errors, dtype=np.float64).ravel()
    truth = np.asarray(labels).astype(np.int64).ravel()
    if np.unique(truth).size < 2:
        return float("nan")
    return float(roc_auc_score(truth, score))


def select_threshold(
    errors: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray | None = None,
    method: ThresholdMethod = "youden",
    percentile: float = 95.0,
) -> ThresholdChoice:
    """Choose an error threshold on validation data.

    Args:
        errors: Per-patch reconstruction errors.
        labels: Labels, required for ``"youden"`` and ignored otherwise.
        method: ``"youden"`` maximises sensitivity + specificity - 1 and needs
            labels; ``"percentile"`` takes a quantile of the *benign* errors
            when labels are supplied, or of all errors when they are not, which
            keeps the rule usable in a fully unlabelled deployment.
        percentile: Quantile used by the ``"percentile"`` method.

    Returns:
        The selected :class:`ThresholdChoice`.

    Raises:
        ValueError: If ``method`` is ``"youden"`` without labels, or unknown.
    """
    score = np.asarray(errors, dtype=np.float64).ravel()

    if method == "youden":
        if labels is None:
            raise ValueError("the 'youden' method requires labels")
        truth = np.asarray(labels).astype(np.int64).ravel()
        return ThresholdChoice(value=youden_threshold(truth, score), method="youden")

    if method == "percentile":
        pool = score
        if labels is not None:
            truth = np.asarray(labels).astype(np.int64).ravel()
            benign = score[truth == 0]
            if benign.size:
                pool = benign
        return ThresholdChoice(
            value=float(np.percentile(pool, percentile)),
            method="percentile",
            percentile=float(percentile),
        )

    raise ValueError(f"unknown threshold method: {method!r}")


def anomaly_metrics(
    errors: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Score the anomaly detector at a fixed error threshold.

    Reconstruction error is not a probability, so it is min-max scaled to
    ``[0, 1]`` before being handed to :func:`classification_metrics`. That
    rescaling is monotone, which leaves ROC AUC, precision, recall and the
    confusion matrix untouched; the calibration columns (``brier``, ``ece``) are
    therefore meaningless for this model and are overwritten with ``nan``.

    Args:
        errors: Per-patch reconstruction errors.
        labels: Ground-truth labels in ``{0, 1}``.
        threshold: Error value above which a patch is flagged as anomalous.

    Returns:
        The usual metric dictionary, with calibration entries blanked.
    """
    score = np.asarray(errors, dtype=np.float64).ravel()
    truth = np.asarray(labels).astype(np.int64).ravel()

    low, high = float(score.min()), float(score.max())
    span = high - low
    scaled = (score - low) / span if span > 0 else np.zeros_like(score)
    scaled_threshold = (threshold - low) / span if span > 0 else 0.5

    metrics = classification_metrics(truth, scaled, threshold=scaled_threshold)
    metrics["brier"] = float("nan")
    metrics["ece"] = float("nan")
    metrics["threshold"] = float(threshold)
    metrics["threshold_scaled"] = float(scaled_threshold)
    return metrics


def error_summary(
    errors: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray
) -> dict[str, float]:
    """Return per-class reconstruction-error summary statistics."""
    score = np.asarray(errors, dtype=np.float64).ravel()
    truth = np.asarray(labels).astype(np.int64).ravel()
    summary: dict[str, float] = {}
    for class_value, name in ((0, "benign"), (1, "metastatic")):
        subset = score[truth == class_value]
        if subset.size == 0:
            continue
        summary[f"{name}_n"] = float(subset.size)
        summary[f"{name}_mean_error"] = float(subset.mean())
        summary[f"{name}_median_error"] = float(np.median(subset))
        summary[f"{name}_p95_error"] = float(np.percentile(subset, 95))
    if "benign_mean_error" in summary and "metastatic_mean_error" in summary:
        summary["mean_error_ratio"] = (
            summary["metastatic_mean_error"] / summary["benign_mean_error"]
        )
    return summary


def top_error_indices(
    errors: Sequence[float] | np.ndarray, k: int = 16
) -> np.ndarray:
    """Return the indices of the ``k`` highest-error patches, worst first."""
    score = np.asarray(errors, dtype=np.float64).ravel()
    count = min(k, score.size)
    top = np.argpartition(-score, count - 1)[:count]
    return top[np.argsort(-score[top])]
