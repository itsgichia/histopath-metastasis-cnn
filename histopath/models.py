"""Model definitions for the PCam metastasis-detection study.

Three architectures, deliberately small so the whole grid trains on a laptop:

* :class:`BaselineCNN` - supervised convolutional classifier trained from
  scratch.
* :class:`ConvAutoencoder` - symmetric encoder/decoder trained unsupervised on
  the reconstruction loss. Also serves as the anomaly-detection model when it
  is fitted on benign patches only.
* :class:`EncoderClassifier` - the autoencoder's encoder with a fresh
  classification head, used both frozen and fine-tuned to test whether
  unsupervised pretraining transfers.

All classifiers emit a single raw logit per patch, so they pair with
``BCEWithLogitsLoss``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _conv_block(in_channels: int, out_channels: int, dropout: float) -> nn.Sequential:
    """Build one ``Conv-BN-ReLU`` pair followed by pooling and dropout.

    Two 3x3 convolutions give a 5x5 receptive field per block at lower
    parameter cost than a single 5x5 kernel, and batch norm keeps the
    from-scratch model trainable at a comparatively high learning rate.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class BaselineCNN(nn.Module):
    """Supervised CNN trained from scratch on normalised PCam patches.

    Three convolutional blocks take a 96x96 patch down to 12x12, global average
    pooling removes the remaining spatial extent, and a small fully connected
    head produces one logit.

    Args:
        channels: Output channel count of each convolutional block.
        conv_dropout: Spatial dropout rate applied at the end of each block.
        head_dropout: Dropout rate applied before the final linear layer.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        conv_dropout: float = 0.1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(_conv_block(in_channels, out_channels, conv_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return one logit per patch, shaped ``(batch,)``."""
        features = self.pool(self.features(x))
        return self.classifier(features).squeeze(-1)


class ConvAutoencoder(nn.Module):
    """Fully convolutional autoencoder for 96x96 RGB patches.

    The encoder applies three stride-2 convolutions, so a 96x96x3 patch becomes
    a ``latent_channels``x12x12 feature map; the decoder mirrors that with
    transposed convolutions and a ``Sigmoid`` output. Inputs must therefore be
    scaled to ``[0, 1]`` and not channel-normalised, which is why
    :func:`histopath.data.autoencoder_transform` skips normalisation.

    Args:
        channels: Encoder channel widths. The last entry is the latent width.
    """

    def __init__(self, channels: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"expected three encoder stages, got {len(channels)}")
        c1, c2, c3 = channels

        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                c1, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )
        self.latent_channels = c3

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent feature map of ``x``."""
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """Return the reconstruction of ``x``, same shape as the input."""
        return self.decoder(self.encoder(x))


class EncoderClassifier(nn.Module):
    """A pretrained autoencoder encoder plus a fresh classification head.

    Args:
        encoder: Encoder module, normally taken from a trained
            :class:`ConvAutoencoder`.
        latent_channels: Channel count of the encoder's output feature map.
        freeze_encoder: When ``True`` the encoder's parameters are frozen and it
            is held in evaluation mode, so only the head learns. This separates
            "are the unsupervised features linearly useful?" from "does the
            pretrained initialisation help after fine-tuning?".
        dropout: Dropout rate inside the head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_channels: int,
        freeze_encoder: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(latent_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "EncoderClassifier":
        """Keep a frozen encoder in eval mode so its batch-norm stays fixed."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Return one logit per patch, shaped ``(batch,)``."""
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.head(self.pool(features)).squeeze(-1)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of (trainable) parameters in ``model``."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )
