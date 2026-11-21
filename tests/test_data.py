"""Tests for subset selection, transforms and dataset wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from histopath import data as hdata


@pytest.fixture
def labels() -> np.ndarray:
    """A deterministic, mildly imbalanced label vector."""
    rng = np.random.default_rng(7)
    return (rng.random(1000) < 0.4).astype(np.int64)


class TestStratifiedSubsetIndices:
    """The subset draw must be reproducible, balanced and sorted."""

    def test_is_deterministic_across_calls(self, labels: np.ndarray) -> None:
        first = hdata.stratified_subset_indices(labels, 200, seed=123)
        second = hdata.stratified_subset_indices(labels, 200, seed=123)
        np.testing.assert_array_equal(first, second)

    def test_is_independent_of_call_order(self, labels: np.ndarray) -> None:
        expected = hdata.stratified_subset_indices(labels, 200, seed=123)
        hdata.stratified_subset_indices(labels, 50, seed=999)
        hdata.stratified_subset_indices(labels, 77, seed=5)
        np.testing.assert_array_equal(
            hdata.stratified_subset_indices(labels, 200, seed=123), expected
        )

    def test_different_seeds_select_different_rows(self, labels: np.ndarray) -> None:
        first = hdata.stratified_subset_indices(labels, 200, seed=1)
        second = hdata.stratified_subset_indices(labels, 200, seed=2)
        assert not np.array_equal(first, second)

    def test_returns_sorted_unique_indices(self, labels: np.ndarray) -> None:
        indices = hdata.stratified_subset_indices(labels, 200, seed=3)
        assert np.all(np.diff(indices) > 0)
        assert indices.size == np.unique(indices).size

    def test_is_class_balanced(self, labels: np.ndarray) -> None:
        indices = hdata.stratified_subset_indices(labels, 200, seed=4)
        selected = labels[indices]
        assert selected.size == 200
        assert int((selected == 0).sum()) == int((selected == 1).sum()) == 100

    def test_indices_stay_in_range(self, labels: np.ndarray) -> None:
        indices = hdata.stratified_subset_indices(labels, 300, seed=5)
        assert indices.min() >= 0
        assert indices.max() < labels.size

    def test_clips_when_a_class_is_too_small(self) -> None:
        labels = np.array([0] * 90 + [1] * 10, dtype=np.int64)
        indices = hdata.stratified_subset_indices(labels, 100, seed=6)
        selected = labels[indices]
        assert int((selected == 1).sum()) == 10
        assert int((selected == 0).sum()) == 50

    @pytest.mark.parametrize("n_total", [0, -5])
    def test_rejects_non_positive_sizes(self, labels: np.ndarray, n_total: int) -> None:
        with pytest.raises(ValueError, match="n_total must be positive"):
            hdata.stratified_subset_indices(labels, n_total)

    def test_rejects_empty_labels(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            hdata.stratified_subset_indices(np.array([], dtype=np.int64), 10)


class TestTransforms:
    """The three pipelines must differ in exactly the documented way."""

    @pytest.fixture
    def patch(self) -> np.ndarray:
        rng = np.random.default_rng(11)
        return rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)

    def test_produces_chw_float_tensor(self, patch: np.ndarray) -> None:
        tensor = hdata.autoencoder_transform()(patch)
        assert tensor.shape == (3, 96, 96)
        assert tensor.dtype == torch.float32

    def test_autoencoder_pipeline_stays_in_unit_interval(self, patch: np.ndarray) -> None:
        tensor = hdata.autoencoder_transform()(patch)
        assert float(tensor.min()) >= 0.0
        assert float(tensor.max()) <= 1.0

    def test_classifier_pipeline_normalises(self, patch: np.ndarray) -> None:
        mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        raw = hdata.autoencoder_transform()(patch)
        normalised = hdata.classifier_transform(stats=(mean, std))(patch)
        expected = (raw - 0.5) / 0.25
        torch.testing.assert_close(normalised, expected)

    def test_imagenet_pipeline_uses_imagenet_statistics(self, patch: np.ndarray) -> None:
        transform = hdata.imagenet_transform()
        torch.testing.assert_close(
            transform.mean.flatten(), torch.tensor(hdata.IMAGENET_MEAN)
        )
        torch.testing.assert_close(
            transform.std.flatten(), torch.tensor(hdata.IMAGENET_STD)
        )

    def test_augmentation_is_label_preserving_and_shape_stable(
        self, patch: np.ndarray
    ) -> None:
        torch.manual_seed(0)
        transform = hdata.autoencoder_transform(train=True)
        outputs = [transform(patch) for _ in range(12)]
        assert all(tensor.shape == (3, 96, 96) for tensor in outputs)
        # Dihedral transforms permute pixels, so the sorted values never change.
        reference = torch.sort(outputs[0].flatten()).values
        for tensor in outputs[1:]:
            torch.testing.assert_close(torch.sort(tensor.flatten()).values, reference)

    def test_augmentation_actually_changes_something(self, patch: np.ndarray) -> None:
        torch.manual_seed(1)
        transform = hdata.autoencoder_transform(train=True)
        outputs = [transform(patch) for _ in range(12)]
        assert any(not torch.equal(outputs[0], tensor) for tensor in outputs[1:])

    def test_rejects_half_specified_statistics(self) -> None:
        with pytest.raises(ValueError, match="together"):
            hdata.PatchTransform(mean=(0.5, 0.5, 0.5), std=None)


class TestChannelStats:
    """Normalisation statistics must be measured, not assumed."""

    def test_matches_a_hand_computed_value(self) -> None:
        images = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        images[..., 0] = 255
        images[..., 1] = 0
        images[..., 2] = 128
        mean, std = hdata.compute_channel_stats(images)
        assert mean == pytest.approx((1.0, 0.0, 128 / 255), abs=1e-9)
        assert std == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    def test_tracks_a_known_distribution(self) -> None:
        rng = np.random.default_rng(3)
        images = rng.integers(0, 256, size=(200, 8, 8, 3), dtype=np.uint8)
        mean, std = hdata.compute_channel_stats(images)
        flat = images.astype(np.float64).reshape(-1, 3) / 255.0
        assert mean == pytest.approx(tuple(flat.mean(axis=0)), abs=1e-10)
        assert std == pytest.approx(tuple(flat.std(axis=0)), abs=1e-10)

    def test_is_independent_of_the_chunk_size(self) -> None:
        rng = np.random.default_rng(4)
        images = rng.integers(0, 256, size=(97, 4, 4, 3), dtype=np.uint8)
        small = hdata.compute_channel_stats(images, chunk_size=7)
        large = hdata.compute_channel_stats(images, chunk_size=1000)
        assert small[0] == pytest.approx(large[0], abs=1e-12)
        assert small[1] == pytest.approx(large[1], abs=1e-12)

    def test_rejects_an_empty_array(self) -> None:
        with pytest.raises(ValueError, match="empty array"):
            hdata.compute_channel_stats(np.zeros((0, 4, 4, 3), dtype=np.uint8))


class TestPatchDataset:
    """The dataset must serve classifier and autoencoder targets correctly."""

    @pytest.fixture
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(13)
        images = rng.integers(0, 256, size=(10, 96, 96, 3), dtype=np.uint8)
        labels = np.array([0, 1] * 5, dtype=np.int64)
        return images, labels

    def test_returns_image_and_label(self, arrays) -> None:
        images, labels = arrays
        dataset = hdata.PatchDataset(images, labels, hdata.autoencoder_transform())
        image, label = dataset[3]
        assert image.shape == (3, 96, 96)
        assert float(label) == float(labels[3])
        assert len(dataset) == 10

    def test_returns_the_image_as_its_own_target_for_autoencoders(self, arrays) -> None:
        images, labels = arrays
        dataset = hdata.PatchDataset(
            images, labels, hdata.autoencoder_transform(), return_label=False
        )
        image, target = dataset[0]
        torch.testing.assert_close(image, target)

    def test_rejects_mismatched_lengths(self, arrays) -> None:
        images, labels = arrays
        with pytest.raises(ValueError, match="disagree"):
            hdata.PatchDataset(images, labels[:5], hdata.autoencoder_transform())

    def test_subset_selects_the_requested_rows(self, arrays) -> None:
        images, labels = arrays
        dataset = hdata.PatchDataset(images, labels, hdata.autoencoder_transform())
        subset = dataset.subset(np.array([1, 3, 5]))
        assert len(subset) == 3
        np.testing.assert_array_equal(subset.labels, labels[[1, 3, 5]])

    def test_loader_yields_batches_of_the_right_shape(self, arrays) -> None:
        images, labels = arrays
        dataset = hdata.PatchDataset(images, labels, hdata.autoencoder_transform())
        batch, batch_labels = next(iter(hdata.make_loader(dataset, batch_size=4)))
        assert batch.shape == (4, 3, 96, 96)
        assert batch_labels.shape == (4,)


class TestSourceAndHelpers:
    """Provenance helpers must point at the pinned revision."""

    def test_url_contains_repository_revision_and_path(self) -> None:
        url = hdata.PCAM.url("train")
        assert hdata.PCAM.repo_id in url
        assert hdata.PCAM.revision in url
        assert url.endswith(hdata.PCAM.files["train"])

    def test_every_split_is_pinned_and_checksummed(self) -> None:
        for split in hdata.SPLITS:
            assert split in hdata.PCAM.files
            assert len(hdata.PCAM.sha256[split]) == 64

    def test_licence_is_recorded(self) -> None:
        assert hdata.PCAM.license_id == "CC0-1.0"

    def test_negative_indices_selects_only_benign_rows(self) -> None:
        labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(hdata.negative_indices(labels), [0, 3, 4])

    def test_resolve_device_honours_an_explicit_choice(self) -> None:
        assert hdata.resolve_device("cpu").type == "cpu"

    def test_seed_everything_makes_torch_reproducible(self) -> None:
        hdata.seed_everything(42)
        first = torch.rand(5)
        hdata.seed_everything(42)
        torch.testing.assert_close(torch.rand(5), first)
