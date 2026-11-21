"""Tests for the training loop, early stopping and best-weight restoration.

Everything here runs on two tiny batches of synthetic patches, so the suite
stays fast while still exercising the real loop end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from histopath import data as hdata
from histopath.models import BaselineCNN, ConvAutoencoder
from histopath.train import (
    History,
    TrainConfig,
    forward_logits,
    logits_to_probabilities,
    save_checkpoint,
    train_autoencoder,
    train_classifier,
)

SIZE = 32
BATCH = 4


@pytest.fixture
def separable_loaders() -> tuple[torch.utils.data.DataLoader, ...]:
    """Two batches whose classes are trivially separable by mean brightness."""
    rng = np.random.default_rng(0)
    images = np.empty((2 * BATCH, SIZE, SIZE, 3), dtype=np.uint8)
    labels = np.empty(2 * BATCH, dtype=np.int64)
    for index in range(2 * BATCH):
        label = index % 2
        base = 200 if label else 40
        images[index] = np.clip(
            base + rng.normal(0, 8, size=(SIZE, SIZE, 3)), 0, 255
        ).astype(np.uint8)
        labels[index] = label

    dataset = hdata.PatchDataset(
        images, labels, hdata.classifier_transform(stats=((0.5,) * 3, (0.25,) * 3))
    )
    loader = hdata.make_loader(dataset, batch_size=BATCH)
    return loader, loader


@pytest.fixture
def reconstruction_loaders() -> tuple[torch.utils.data.DataLoader, ...]:
    """Two batches of smooth patches served as autoencoder input/target pairs."""
    rng = np.random.default_rng(1)
    images = np.clip(
        128 + rng.normal(0, 10, size=(2 * BATCH, SIZE, SIZE, 3)), 0, 255
    ).astype(np.uint8)
    dataset = hdata.PatchDataset(
        images,
        np.zeros(2 * BATCH, dtype=np.int64),
        hdata.autoencoder_transform(),
        return_label=False,
    )
    loader = hdata.make_loader(dataset, batch_size=BATCH)
    return loader, loader


class TestTrainClassifier:
    """The classifier loop must run, record history and improve."""

    def test_runs_the_requested_number_of_epochs(self, separable_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            BaselineCNN(channels=(8, 16)),
            *separable_loaders,
            TrainConfig(epochs=3, patience=99),
            name="tiny",
        )
        assert len(history.records) == 3
        assert [record["epoch"] for record in history.records] == [1.0, 2.0, 3.0]

    def test_records_the_expected_metric_keys(self, separable_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            BaselineCNN(channels=(8, 16)),
            *separable_loaders,
            TrainConfig(epochs=2, patience=99),
        )
        for key in ("train_loss", "val_loss", "train_auc", "val_auc", "lr", "seconds"):
            assert key in history.records[0], key

    def test_reduces_training_loss_on_a_separable_problem(
        self, separable_loaders
    ) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            BaselineCNN(channels=(8, 16)),
            *separable_loaders,
            TrainConfig(epochs=8, patience=99, lr=3e-3),
        )
        losses = history.series("train_loss")
        assert losses[-1] < losses[0]

    def test_tracks_the_best_epoch(self, separable_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            BaselineCNN(channels=(8, 16)),
            *separable_loaders,
            TrainConfig(epochs=4, patience=99),
        )
        assert 1 <= history.best_epoch <= 4
        assert history.best_score == max(history.series("val_auc"))

    def test_early_stops_when_nothing_improves(self, separable_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            nn.Sequential(nn.Flatten(), nn.Linear(3 * SIZE * SIZE, 1), nn.Flatten(0)),
            *separable_loaders,
            TrainConfig(epochs=30, patience=1, lr=0.0, min_delta=1e9),
        )
        assert len(history.records) < 30

    def test_accepts_a_positive_class_weight(self, separable_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_classifier(
            BaselineCNN(channels=(8, 16)),
            *separable_loaders,
            TrainConfig(epochs=1),
            pos_weight=2.0,
        )
        assert len(history.records) == 1


class TestTrainAutoencoder:
    """The autoencoder loop must minimise reconstruction error."""

    def test_reduces_reconstruction_loss(self, reconstruction_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_autoencoder(
            ConvAutoencoder(channels=(8, 16, 24)),
            *reconstruction_loaders,
            TrainConfig(epochs=6, patience=99, lr=3e-3),
        )
        losses = history.series("train_loss")
        assert losses[-1] < losses[0]

    def test_monitors_validation_loss_downward(self, reconstruction_loaders) -> None:
        torch.manual_seed(0)
        _, history = train_autoencoder(
            ConvAutoencoder(channels=(8, 16, 24)),
            *reconstruction_loaders,
            TrainConfig(epochs=3, patience=99),
        )
        assert history.monitor == "val_loss"
        assert history.mode == "min"
        assert history.best_score == min(history.series("val_loss"))

    def test_restores_the_best_weights(self, reconstruction_loaders) -> None:
        """After training, the returned model must score the recorded best."""
        torch.manual_seed(0)
        model, history = train_autoencoder(
            ConvAutoencoder(channels=(8, 16, 24)),
            *reconstruction_loaders,
            TrainConfig(epochs=5, patience=99, lr=3e-3),
        )
        model.eval()
        criterion = nn.MSELoss()
        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, target in reconstruction_loaders[1]:
                total += float(criterion(model(inputs), target)) * inputs.shape[0]
                count += inputs.shape[0]
        assert total / count == pytest.approx(history.best_score, rel=1e-3)


class TestHistory:
    """History export must be CSV ready."""

    def test_rows_carry_the_run_name(self) -> None:
        history = History(name="demo", monitor="val_loss", mode="min")
        history.records.append({"epoch": 1.0, "val_loss": 0.5})
        rows = history.to_rows()
        assert rows == [{"run": "demo", "epoch": 1.0, "val_loss": 0.5}]

    def test_series_skips_absent_keys(self) -> None:
        history = History(name="demo", monitor="val_loss", mode="min")
        history.records.append({"epoch": 1.0, "val_loss": 0.5})
        history.records.append({"epoch": 2.0})
        assert history.series("val_loss") == [0.5]
        assert history.series("val_auc") == []

    def test_write_csv_appends_a_single_header(self, tmp_path) -> None:
        path = tmp_path / "history.csv"
        for name in ("a", "b"):
            history = History(name=name, monitor="val_loss", mode="min")
            history.records.append({"epoch": 1.0, "val_loss": 0.5})
            history.write_csv(path)
        lines = path.read_text().strip().splitlines()
        assert lines[0].startswith("run,")
        assert len(lines) == 3


class TestHelpers:
    """Inference helpers must be shape correct and numerically safe."""

    def test_forward_logits_returns_one_score_per_sample(
        self, separable_loaders
    ) -> None:
        logits, labels = forward_logits(
            BaselineCNN(channels=(8, 16)), separable_loaders[1], torch.device("cpu")
        )
        assert logits.shape == labels.shape == (2 * BATCH,)

    def test_sigmoid_is_stable_at_extreme_logits(self) -> None:
        probabilities = logits_to_probabilities(
            np.array([-1e4, -50.0, 0.0, 50.0, 1e4])
        )
        assert np.isfinite(probabilities).all()
        assert probabilities[2] == pytest.approx(0.5)
        assert probabilities[0] == pytest.approx(0.0, abs=1e-12)
        assert probabilities[-1] == pytest.approx(1.0, abs=1e-12)

    def test_sigmoid_matches_torch(self) -> None:
        logits = np.linspace(-8, 8, 33)
        np.testing.assert_allclose(
            logits_to_probabilities(logits),
            torch.sigmoid(torch.from_numpy(logits)).numpy(),
            atol=1e-12,
        )

    def test_save_checkpoint_writes_a_loadable_state_dict(self, tmp_path) -> None:
        model = BaselineCNN(channels=(8, 16))
        path = save_checkpoint(model, tmp_path / "nested" / "model.pt")
        assert path.exists()
        reloaded = BaselineCNN(channels=(8, 16))
        reloaded.load_state_dict(torch.load(path, weights_only=True))
