"""Detecting metastatic tissue in lymph node histopathology patches.

A compact comparison of four deep learning approaches on the PatchCamelyon
benchmark: a supervised CNN trained from scratch, an unsupervised convolutional
autoencoder, transfer of that autoencoder's encoder into a classifier (frozen
and fine-tuned), and an anomaly detector trained on benign tissue alone.

Author:
    Gichia Muiruri
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Gichia Muiruri"

__all__ = ["anomaly", "data", "evaluate", "models", "plots", "train"]
