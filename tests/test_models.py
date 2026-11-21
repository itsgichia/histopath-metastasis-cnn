"""Tests for model output shapes, ranges and the freezing contract."""

from __future__ import annotations

import pytest
import torch

from histopath.models import (
    BaselineCNN,
    ConvAutoencoder,
    EncoderClassifier,
    count_parameters,
)

BATCH = 4


@pytest.fixture
def patches() -> torch.Tensor:
    """A small batch of 96x96 RGB patches."""
    torch.manual_seed(0)
    return torch.rand(BATCH, 3, 96, 96)


class TestBaselineCNN:
    """The supervised classifier emits one logit per patch."""

    def test_output_shape(self, patches: torch.Tensor) -> None:
        assert BaselineCNN()(patches).shape == (BATCH,)

    def test_output_is_finite_and_unbounded(self, patches: torch.Tensor) -> None:
        logits = BaselineCNN().eval()(patches)
        assert torch.isfinite(logits).all()
        assert logits.dtype == torch.float32

    @pytest.mark.parametrize("size", [32, 64, 96])
    def test_accepts_other_divisible_sizes(self, size: int) -> None:
        model = BaselineCNN()
        assert model(torch.rand(2, 3, size, size)).shape == (2,)

    def test_gradients_reach_the_first_layer(self, patches: torch.Tensor) -> None:
        model = BaselineCNN()
        model(patches).sum().backward()
        first = next(model.features[0].parameters())
        assert first.grad is not None
        assert torch.isfinite(first.grad).all()


class TestConvAutoencoder:
    """Reconstructions must match the input shape and stay in [0, 1]."""

    def test_reconstruction_shape_matches_input(self, patches: torch.Tensor) -> None:
        assert ConvAutoencoder()(patches).shape == patches.shape

    def test_latent_shape_is_eight_times_smaller_per_side(
        self, patches: torch.Tensor
    ) -> None:
        model = ConvAutoencoder()
        latent = model.encode(patches)
        assert latent.shape == (BATCH, model.latent_channels, 12, 12)

    def test_output_is_bounded_to_the_unit_interval(self, patches: torch.Tensor) -> None:
        with torch.no_grad():
            output = ConvAutoencoder().eval()(patches)
        assert float(output.min()) >= 0.0
        assert float(output.max()) <= 1.0

    def test_latent_width_follows_the_channel_argument(self) -> None:
        model = ConvAutoencoder(channels=(8, 16, 24))
        assert model.latent_channels == 24
        assert model.encode(torch.rand(2, 3, 32, 32)).shape == (2, 24, 4, 4)

    def test_rejects_a_wrong_number_of_stages(self) -> None:
        with pytest.raises(ValueError, match="three encoder stages"):
            ConvAutoencoder(channels=(8, 16))


class TestEncoderClassifier:
    """Transfer learning must honour the freeze flag in both directions."""

    @pytest.fixture
    def autoencoder(self) -> ConvAutoencoder:
        torch.manual_seed(1)
        return ConvAutoencoder(channels=(8, 16, 24))

    def test_output_shape(self, autoencoder: ConvAutoencoder) -> None:
        model = EncoderClassifier(autoencoder.encoder, autoencoder.latent_channels)
        assert model(torch.rand(BATCH, 3, 96, 96)).shape == (BATCH,)

    def test_frozen_encoder_has_no_trainable_parameters(
        self, autoencoder: ConvAutoencoder
    ) -> None:
        model = EncoderClassifier(
            autoencoder.encoder, autoencoder.latent_channels, freeze_encoder=True
        )
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

    def test_frozen_encoder_stays_in_eval_mode_after_train(
        self, autoencoder: ConvAutoencoder
    ) -> None:
        model = EncoderClassifier(
            autoencoder.encoder, autoencoder.latent_channels, freeze_encoder=True
        )
        model.train()
        assert not model.encoder.training
        assert model.head.training

    def test_frozen_encoder_weights_do_not_move(
        self, autoencoder: ConvAutoencoder
    ) -> None:
        model = EncoderClassifier(
            autoencoder.encoder, autoencoder.latent_channels, freeze_encoder=True
        )
        before = next(model.encoder.parameters()).detach().clone()
        optimiser = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        for _ in range(3):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.rand(BATCH, 3, 96, 96)), torch.ones(BATCH)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        torch.testing.assert_close(next(model.encoder.parameters()), before)

    def test_unfrozen_encoder_weights_do_move(
        self, autoencoder: ConvAutoencoder
    ) -> None:
        model = EncoderClassifier(
            autoencoder.encoder, autoencoder.latent_channels, freeze_encoder=False
        )
        before = next(model.encoder.parameters()).detach().clone()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(3):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(torch.rand(BATCH, 3, 96, 96)), torch.ones(BATCH)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        assert not torch.allclose(next(model.encoder.parameters()), before)

    def test_unfrozen_encoder_trains_in_train_mode(
        self, autoencoder: ConvAutoencoder
    ) -> None:
        model = EncoderClassifier(
            autoencoder.encoder, autoencoder.latent_channels, freeze_encoder=False
        )
        model.train()
        assert model.encoder.training


class TestParameterCount:
    """Parameter counting must distinguish trainable from total."""

    def test_counts_every_parameter(self) -> None:
        model = torch.nn.Linear(10, 4)
        assert count_parameters(model) == 10 * 4 + 4

    def test_trainable_only_excludes_frozen_parameters(self) -> None:
        model = torch.nn.Linear(10, 4)
        model.weight.requires_grad_(False)
        assert count_parameters(model, trainable_only=True) == 4
        assert count_parameters(model) == 44

    def test_models_stay_small_enough_for_a_laptop(self) -> None:
        assert count_parameters(BaselineCNN()) < 1_000_000
        assert count_parameters(ConvAutoencoder()) < 1_000_000
