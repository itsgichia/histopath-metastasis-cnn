"""Tests for metric correctness against hand-computed fixtures."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from histopath import evaluate as heval


class TestClassificationMetrics:
    """Each metric is checked against a value worked out by hand."""

    def test_perfect_separation_scores_one(self) -> None:
        truth = np.array([0, 0, 1, 1])
        probability = np.array([0.01, 0.2, 0.8, 0.99])
        metrics = heval.classification_metrics(truth, probability)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["specificity"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero_auc(self) -> None:
        truth = np.array([0, 0, 1, 1])
        probability = np.array([0.9, 0.8, 0.2, 0.1])
        assert heval.classification_metrics(truth, probability)["roc_auc"] == (
            pytest.approx(0.0)
        )

    def test_random_ranking_scores_half_auc(self) -> None:
        truth = np.array([0, 1, 0, 1])
        probability = np.array([0.5, 0.5, 0.5, 0.5])
        assert heval.classification_metrics(truth, probability)["roc_auc"] == (
            pytest.approx(0.5)
        )

    def test_confusion_counts_match_a_worked_example(self) -> None:
        # Six benign, four metastatic. At threshold 0.5 the predictions give
        # tn=4, fp=2, fn=1, tp=3.
        truth = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        probability = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.2, 0.8, 0.9, 0.95])
        metrics = heval.classification_metrics(truth, probability)
        assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (
            4.0, 2.0, 1.0, 3.0,
        )
        assert metrics["accuracy"] == pytest.approx(7 / 10)
        assert metrics["precision"] == pytest.approx(3 / 5)
        assert metrics["recall"] == pytest.approx(3 / 4)
        assert metrics["specificity"] == pytest.approx(4 / 6)
        assert metrics["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
        assert metrics["n_samples"] == 10.0

    def test_threshold_changes_the_operating_point(self) -> None:
        truth = np.array([0, 0, 1, 1])
        probability = np.array([0.3, 0.6, 0.4, 0.9])
        strict = heval.classification_metrics(truth, probability, threshold=0.8)
        lenient = heval.classification_metrics(truth, probability, threshold=0.35)
        assert strict["recall"] == pytest.approx(0.5)
        assert strict["specificity"] == pytest.approx(1.0)
        assert lenient["recall"] == pytest.approx(1.0)
        assert lenient["specificity"] == pytest.approx(0.5)
        # The ranking is unchanged, so AUC is threshold free.
        assert strict["roc_auc"] == pytest.approx(lenient["roc_auc"])

    def test_brier_score_matches_the_definition(self) -> None:
        truth = np.array([0, 1])
        probability = np.array([0.25, 0.75])
        expected = ((0.25 - 0) ** 2 + (0.75 - 1) ** 2) / 2
        assert heval.classification_metrics(truth, probability)["brier"] == (
            pytest.approx(expected)
        )

    def test_single_class_input_yields_undefined_ranking_metrics(self) -> None:
        metrics = heval.classification_metrics(
            np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4])
        )
        assert np.isnan(metrics["roc_auc"])
        assert np.isnan(metrics["average_precision"])
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            heval.classification_metrics(np.array([0, 1]), np.array([0.5]))

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            heval.classification_metrics(np.array([]), np.array([]))

    def test_returns_every_declared_column(self) -> None:
        metrics = heval.classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
        assert set(heval.METRIC_COLUMNS) <= set(metrics)


class TestCalibration:
    """Calibration must be zero for a perfectly calibrated forecast."""

    def test_perfectly_calibrated_forecast_has_zero_error(self) -> None:
        # In each bin the predicted probability equals the observed rate.
        probability = np.repeat([0.25, 0.75], 400)
        truth = np.concatenate(
            [
                np.array([1] * 100 + [0] * 300),  # 25% positive at p=0.25
                np.array([1] * 300 + [0] * 100),  # 75% positive at p=0.75
            ]
        )
        assert heval.expected_calibration_error(truth, probability) == (
            pytest.approx(0.0, abs=1e-12)
        )

    def test_maximally_overconfident_forecast_has_error_one(self) -> None:
        truth = np.array([0] * 50 + [1] * 50)
        probability = np.array([1.0] * 50 + [0.0] * 50)
        assert heval.expected_calibration_error(truth, probability) == (
            pytest.approx(1.0)
        )

    def test_error_matches_a_hand_computed_two_bin_case(self) -> None:
        # Bin [0.0,0.1): p=0.0, observed 0.5 over 2 samples -> gap 0.5, weight 0.5
        # Bin [0.9,1.0]: p=1.0, observed 1.0 over 2 samples -> gap 0.0, weight 0.5
        truth = np.array([0, 1, 1, 1])
        probability = np.array([0.0, 0.0, 1.0, 1.0])
        assert heval.expected_calibration_error(truth, probability) == (
            pytest.approx(0.25)
        )

    def test_curve_drops_empty_bins(self) -> None:
        curve = heval.calibration_curve_points(
            np.array([0, 1, 0, 1]), np.array([0.05, 0.95, 0.05, 0.95]), n_bins=10
        )
        assert curve.bin_centers.size == 2
        np.testing.assert_allclose(curve.fraction_positive, [0.0, 1.0])
        np.testing.assert_array_equal(curve.counts, [2, 2])

    def test_curve_counts_sum_to_the_sample_size(self) -> None:
        rng = np.random.default_rng(0)
        probability = rng.random(500)
        truth = (rng.random(500) < probability).astype(int)
        curve = heval.calibration_curve_points(truth, probability, n_bins=10)
        assert int(curve.counts.sum()) == 500


class TestConfusionAndRoc:
    """Matrix orientation and threshold selection must be unambiguous."""

    def test_matrix_rows_are_truth_and_columns_are_prediction(self) -> None:
        truth = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.9, 0.1, 0.9])
        matrix = heval.confusion(truth, probability)
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])

    def test_matrix_total_equals_the_sample_count(self) -> None:
        rng = np.random.default_rng(2)
        truth = rng.integers(0, 2, 200)
        probability = rng.random(200)
        assert int(heval.confusion(truth, probability).sum()) == 200

    def test_roc_points_start_and_end_at_the_corners(self) -> None:
        truth = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.6, 0.9])
        fpr, tpr, auc = heval.roc_points(truth, probability)
        assert (fpr[0], tpr[0]) == (0.0, 0.0)
        assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert auc == pytest.approx(1.0)

    def test_youden_threshold_separates_a_clean_gap(self) -> None:
        truth = np.array([0, 0, 0, 1, 1, 1])
        probability = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
        threshold = heval.youden_threshold(truth, probability)
        assert 0.15 < threshold <= 0.85
        predicted = (probability >= threshold).astype(int)
        np.testing.assert_array_equal(predicted, truth)

    def test_youden_threshold_is_finite(self) -> None:
        rng = np.random.default_rng(3)
        truth = rng.integers(0, 2, 100)
        probability = rng.random(100)
        assert np.isfinite(heval.youden_threshold(truth, probability))


class TestMetricsCsv:
    """The exported table must be stable and complete."""

    def test_writes_model_first_then_the_declared_columns(self, tmp_path) -> None:
        rows = [
            {"model": "A", **heval.classification_metrics([0, 1], [0.2, 0.8])},
            {"model": "B", **heval.classification_metrics([0, 1], [0.8, 0.2])},
        ]
        path = heval.write_metrics_csv(rows, tmp_path / "metrics.csv")
        with path.open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert reader.fieldnames[0] == "model"
            assert reader.fieldnames[1 : 1 + len(heval.METRIC_COLUMNS)] == list(
                heval.METRIC_COLUMNS
            )
            assert [row["model"] for row in reader] == ["A", "B"]

    def test_keeps_extra_columns(self, tmp_path) -> None:
        rows = [
            {
                "model": "A",
                "threshold_source": "val_youden",
                **heval.classification_metrics([0, 1], [0.2, 0.8]),
            }
        ]
        path = heval.write_metrics_csv(rows, tmp_path / "metrics.csv")
        with path.open() as handle:
            assert "threshold_source" in (csv.DictReader(handle).fieldnames or [])

    def test_rejects_an_empty_table(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="no metric rows"):
            heval.write_metrics_csv([], tmp_path / "metrics.csv")
