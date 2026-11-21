"""Tests for reconstruction-error scoring, thresholds and anomaly metrics.

The central behavioural claim is checked twice: a deliberately corrupted patch
must score above a clean one, both against a stub model that reconstructs the
healthy prototype exactly, and against a real autoencoder trained briefly on
clean patches only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from histopath import anomaly
from histopath import data as hdata
from histopath.models import ConvAutoencoder
from histopath.train import TrainConfig, train_autoencoder

SIZE = 32


def clean_patch(seed: int, size: int = SIZE) -> np.ndarray:
    """A smooth, low-variance patch standing in for benign tissue."""
    rng = np.random.default_rng(seed)
    return np.clip(
        150 + rng.normal(0, 4, size=(size, size, 3)), 0, 255
    ).astype(np.uint8)


def corrupt(patch: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return ``patch`` with a bright salt-and-pepper block stamped into it."""
    rng = np.random.default_rng(seed)
    damaged = patch.copy()
    damaged[8:24, 8:24] = rng.integers(0, 2, size=(16, 16, 3), dtype=np.uint8) * 255
    return damaged


class PrototypeAutoencoder(nn.Module):
    """Stub model that always outputs the healthy prototype.

    This is what a perfectly trained benign-only autoencoder approximates: it
    reproduces healthy morphology and nothing else. Using it makes the scorer's
    behaviour deterministic and independent of optimisation luck.
    """

    def __init__(self, prototype: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prototype", prototype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the prototype, broadcast to the batch size of ``x``."""
        return self.prototype.unsqueeze(0).expand(x.shape[0], -1, -1, -1).to(x.device)


def make_loader(images: np.ndarray, labels: np.ndarray) -> torch.utils.data.DataLoader:
    """Wrap raw patches in a labelled autoencoder-transform loader."""
    dataset = hdata.PatchDataset(
        images, labels, hdata.autoencoder_transform(), return_label=True
    )
    return hdata.make_loader(dataset, batch_size=4)


class TestScorerRanksCorruption:
    """A corrupted patch must out-score a clean one."""

    def test_stub_model_ranks_the_corrupted_patch_higher(self) -> None:
        base = clean_patch(0)
        prototype = torch.from_numpy(base).permute(2, 0, 1).float() / 255.0
        model = PrototypeAutoencoder(prototype)

        images = np.stack([base, corrupt(base)])
        errors, _ = anomaly.reconstruction_errors(
            model, make_loader(images, np.array([0, 1])), torch.device("cpu")
        )
        assert errors[0] == pytest.approx(0.0, abs=1e-12)
        assert errors[1] > errors[0]

    def test_trained_autoencoder_ranks_the_corrupted_patch_higher(self) -> None:
        """Train on clean patches only, then score one clean and one corrupted."""
        torch.manual_seed(0)
        clean = np.stack([clean_patch(seed) for seed in range(32)])
        dataset = hdata.PatchDataset(
            clean,
            np.zeros(len(clean), dtype=np.int64),
            hdata.autoencoder_transform(),
            return_label=False,
        )
        loader = hdata.make_loader(dataset, batch_size=8)
        model, _ = train_autoencoder(
            ConvAutoencoder(channels=(8, 16, 24)),
            loader,
            loader,
            TrainConfig(epochs=12, patience=99, lr=5e-3),
            name="benign-only",
        )

        held_out = clean_patch(999)
        images = np.stack([held_out, corrupt(held_out, seed=1)])
        errors, _ = anomaly.reconstruction_errors(
            model, make_loader(images, np.array([0, 1])), torch.device("cpu")
        )
        assert errors[1] > errors[0]

    def test_error_grows_with_corruption_severity(self) -> None:
        base = clean_patch(1)
        prototype = torch.from_numpy(base).permute(2, 0, 1).float() / 255.0
        model = PrototypeAutoencoder(prototype)

        images = [base]
        for extent in (4, 10, 16):
            damaged = base.copy()
            damaged[:extent, :extent] = 255
            images.append(damaged)

        errors, _ = anomaly.reconstruction_errors(
            model,
            make_loader(np.stack(images), np.zeros(len(images), dtype=np.int64)),
            torch.device("cpu"),
        )
        assert np.all(np.diff(errors) > 0)


class TestErrorAuc:
    """Error-as-score AUC must behave like any ranking metric."""

    def test_perfect_ranking_scores_one(self) -> None:
        errors = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        assert anomaly.error_auc(errors, labels) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self) -> None:
        errors = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])
        assert anomaly.error_auc(errors, labels) == pytest.approx(0.0)

    def test_uninformative_error_scores_half(self) -> None:
        errors = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 1, 0, 1])
        assert anomaly.error_auc(errors, labels) == pytest.approx(0.5)

    def test_single_class_input_is_undefined(self) -> None:
        assert np.isnan(anomaly.error_auc(np.array([0.1, 0.2]), np.array([0, 0])))


class TestSelectThreshold:
    """Threshold selection must be explicit about what it uses."""

    @pytest.fixture
    def separable(self) -> tuple[np.ndarray, np.ndarray]:
        errors = np.array([0.01, 0.02, 0.03, 0.30, 0.31, 0.32])
        labels = np.array([0, 0, 0, 1, 1, 1])
        return errors, labels

    def test_youden_splits_a_clean_gap(self, separable) -> None:
        errors, labels = separable
        choice = anomaly.select_threshold(errors, labels, method="youden")
        assert choice.method == "youden"
        assert 0.03 < choice.value <= 0.30
        np.testing.assert_array_equal((errors >= choice.value).astype(int), labels)

    def test_youden_requires_labels(self, separable) -> None:
        errors, _ = separable
        with pytest.raises(ValueError, match="requires labels"):
            anomaly.select_threshold(errors, None, method="youden")

    def test_percentile_uses_only_benign_errors_when_labelled(self, separable) -> None:
        errors, labels = separable
        choice = anomaly.select_threshold(
            errors, labels, method="percentile", percentile=100.0
        )
        assert choice.value == pytest.approx(0.03)
        assert choice.percentile == 100.0

    def test_percentile_falls_back_to_all_errors_when_unlabelled(
        self, separable
    ) -> None:
        errors, _ = separable
        choice = anomaly.select_threshold(
            errors, None, method="percentile", percentile=100.0
        )
        assert choice.value == pytest.approx(0.32)

    def test_rejects_an_unknown_method(self, separable) -> None:
        errors, labels = separable
        with pytest.raises(ValueError, match="unknown threshold method"):
            anomaly.select_threshold(
                errors, labels, method="magic"  # type: ignore[arg-type]
            )


class TestAnomalyMetrics:
    """Rescaling must not disturb the ranking or the confusion counts."""

    def test_recovers_a_perfect_split(self) -> None:
        errors = np.array([0.01, 0.02, 0.30, 0.31])
        labels = np.array([0, 0, 1, 1])
        metrics = anomaly.anomaly_metrics(errors, labels, threshold=0.1)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
            2.0, 0.0, 0.0, 2.0,
        )

    def test_reports_the_raw_threshold_alongside_the_scaled_one(self) -> None:
        errors = np.array([0.0, 0.5, 1.0, 1.5])
        labels = np.array([0, 0, 1, 1])
        metrics = anomaly.anomaly_metrics(errors, labels, threshold=0.75)
        assert metrics["threshold"] == pytest.approx(0.75)
        assert metrics["threshold_scaled"] == pytest.approx(0.5)

    def test_calibration_columns_are_blanked(self) -> None:
        metrics = anomaly.anomaly_metrics(
            np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1]), threshold=0.25
        )
        assert np.isnan(metrics["brier"])
        assert np.isnan(metrics["ece"])

    def test_handles_a_degenerate_constant_score(self) -> None:
        metrics = anomaly.anomaly_metrics(
            np.full(4, 0.2), np.array([0, 0, 1, 1]), threshold=0.2
        )
        assert np.isfinite(metrics["accuracy"])


class TestErrorSummaryAndTopK:
    """Reporting helpers must be correct and ordered."""

    def test_summary_separates_the_classes(self) -> None:
        errors = np.array([0.1, 0.2, 0.6, 0.8])
        labels = np.array([0, 0, 1, 1])
        summary = anomaly.error_summary(errors, labels)
        assert summary["benign_n"] == 2.0
        assert summary["metastatic_n"] == 2.0
        assert summary["benign_mean_error"] == pytest.approx(0.15)
        assert summary["metastatic_mean_error"] == pytest.approx(0.7)
        assert summary["mean_error_ratio"] == pytest.approx(0.7 / 0.15)

    def test_summary_skips_an_absent_class(self) -> None:
        summary = anomaly.error_summary(np.array([0.1, 0.2]), np.array([0, 0]))
        assert "benign_n" in summary
        assert "metastatic_n" not in summary
        assert "mean_error_ratio" not in summary

    def test_top_error_indices_are_ordered_worst_first(self) -> None:
        errors = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        np.testing.assert_array_equal(
            anomaly.top_error_indices(errors, k=3), [1, 3, 4]
        )

    def test_top_error_indices_clip_to_the_available_count(self) -> None:
        errors = np.array([0.1, 0.9])
        assert anomaly.top_error_indices(errors, k=10).size == 2


class TestReconstructionErrors:
    """The scorer must handle both loader flavours."""

    def test_returns_one_error_per_patch_with_labels(self) -> None:
        base = clean_patch(5)
        prototype = torch.from_numpy(base).permute(2, 0, 1).float() / 255.0
        images = np.stack([base] * 6)
        labels = np.array([0, 1, 0, 1, 0, 1])
        errors, returned = anomaly.reconstruction_errors(
            PrototypeAutoencoder(prototype), make_loader(images, labels)
        )
        assert errors.shape == (6,)
        np.testing.assert_array_equal(returned.astype(int), labels)

    def test_ignores_labels_for_an_unlabelled_loader(self) -> None:
        base = clean_patch(6)
        prototype = torch.from_numpy(base).permute(2, 0, 1).float() / 255.0
        dataset = hdata.PatchDataset(
            np.stack([base] * 4),
            np.ones(4, dtype=np.int64),
            hdata.autoencoder_transform(),
            return_label=False,
        )
        errors, returned = anomaly.reconstruction_errors(
            PrototypeAutoencoder(prototype), hdata.make_loader(dataset, batch_size=2)
        )
        assert errors.shape == (4,)
        np.testing.assert_array_equal(returned, np.zeros(4))
