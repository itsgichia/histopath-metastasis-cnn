"""Training loops with early stopping, LR scheduling and history tracking.

Two entry points share one generic loop:

* :func:`train_classifier` optimises ``BCEWithLogitsLoss`` and early-stops on
  validation ROC AUC.
* :func:`train_autoencoder` optimises reconstruction MSE and early-stops on
  validation loss.

Both restore the best-scoring weights before returning, so the object handed
back is the model that should be evaluated rather than the last epoch's.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)

Mode = Literal["min", "max"]


@dataclass
class TrainConfig:
    """Hyperparameters shared by every training run.

    Attributes:
        epochs: Maximum number of epochs.
        lr: Initial Adam learning rate.
        weight_decay: L2 penalty applied by Adam.
        patience: Epochs without improvement tolerated before stopping early.
        min_delta: Smallest change in the monitored metric that counts as an
            improvement.
        scheduler_factor: Multiplier applied to the LR on plateau.
        scheduler_patience: Plateau length, in epochs, before the LR is cut.
        grad_clip: Global gradient-norm clip, or ``None`` to disable.
        log_every: Print a progress line every N batches; ``0`` disables.
    """

    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    grad_clip: float | None = 5.0
    log_every: int = 0


@dataclass
class History:
    """Per-epoch training record.

    Attributes:
        name: Run name, used as a column in the exported table.
        monitor: Name of the early-stopping metric.
        mode: Whether the monitored metric is minimised or maximised.
        records: One dictionary per completed epoch.
        best_epoch: 1-based index of the best epoch.
        best_score: Value of the monitored metric at ``best_epoch``.
        wall_seconds: Total training wall time.
    """

    name: str
    monitor: str
    mode: Mode
    records: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = float("nan")
    wall_seconds: float = 0.0

    def to_rows(self) -> list[dict[str, float | str]]:
        """Return the history as flat rows ready for CSV export."""
        return [{"run": self.name, **record} for record in self.records]

    def series(self, key: str) -> list[float]:
        """Return the per-epoch values of ``key``, skipping missing entries."""
        return [record[key] for record in self.records if key in record]

    def write_csv(self, path: Path) -> None:
        """Append the history to a CSV file, writing a header when new."""
        import csv

        rows = self.to_rows()
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)


def _improved(current: float, best: float, mode: Mode, min_delta: float) -> bool:
    """Return whether ``current`` beats ``best`` by at least ``min_delta``."""
    if np.isnan(best):
        return True
    return current > best + min_delta if mode == "max" else current < best - min_delta


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    collect_outputs: bool,
    log_every: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one pass over ``loader``.

    When ``optimizer`` is ``None`` the pass is evaluation only. Returns the mean
    loss plus, if requested, the concatenated model outputs and targets.

    Running totals are kept as device tensors and copied to the host once, at
    the end of the pass. Reading a loss scalar inside the loop would force a
    synchronisation on every batch, which on an Apple ``mps`` device drains the
    command queue and costs several times the compute itself.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_items = 0
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch_index, (inputs, target) in enumerate(loader):
        # Blocking transfers: the loader's CPU tensors are not in pinned memory,
        # so an asynchronous copy could race the next iteration freeing them.
        inputs = inputs.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(training):
            prediction = model(inputs)
            loss = criterion(prediction, target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_items = inputs.shape[0]
        total_loss += loss.detach().float() * batch_items
        total_items += batch_items

        if collect_outputs:
            outputs.append(prediction.detach().float())
            targets.append(target.detach().float())

        if log_every and batch_index % log_every == 0:
            LOGGER.info(
                "  batch %d/%d loss %.4f", batch_index, len(loader), float(loss)
            )

    mean_loss = float(total_loss.cpu()) / max(total_items, 1)
    stacked_outputs = (
        torch.cat(outputs).cpu().numpy() if outputs else np.empty(0)
    )
    stacked_targets = (
        torch.cat(targets).cpu().numpy() if targets else np.empty(0)
    )
    return mean_loss, stacked_outputs, stacked_targets


def _fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
    device: torch.device,
    name: str,
    monitor: str,
    mode: Mode,
    extra_metrics: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
) -> tuple[nn.Module, History]:
    """Generic training loop with early stopping and LR scheduling."""
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    history = History(name=name, monitor=monitor, mode=mode)
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        collect = extra_metrics is not None

        train_loss, train_out, train_tgt = _run_epoch(
            model, train_loader, criterion, device, optimizer,
            config.grad_clip, collect, config.log_every,
        )
        val_loss, val_out, val_tgt = _run_epoch(
            model, val_loader, criterion, device, None, None, collect, 0,
        )

        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_start,
        }
        if extra_metrics is not None:
            for key, value in extra_metrics(train_out, train_tgt).items():
                record[f"train_{key}"] = value
            for key, value in extra_metrics(val_out, val_tgt).items():
                record[f"val_{key}"] = value

        score = record[monitor]
        history.records.append(record)
        # Only name the monitored metric when it is not already shown above.
        extra = "" if monitor == "val_loss" else f" {monitor} {score:.4f}"
        LOGGER.info(
            "[%s] epoch %d/%d train_loss %.4f val_loss %.4f%s (%.1fs)",
            name, epoch, config.epochs, train_loss, val_loss, extra,
            record["seconds"],
        )

        if _improved(score, history.best_score, mode, config.min_delta):
            history.best_score = score
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                LOGGER.info("[%s] early stop at epoch %d", name, epoch)
                break

        scheduler.step(score)

    model.load_state_dict(best_state)
    history.wall_seconds = time.perf_counter() - start
    LOGGER.info(
        "[%s] best %s %.4f at epoch %d (%.1fs total)",
        name, monitor, history.best_score, history.best_epoch, history.wall_seconds,
    )
    return model, history


def _auc_and_accuracy(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return ROC AUC and accuracy at a 0.5 probability threshold."""
    if logits.size == 0:
        return {}
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    accuracy = float(((probabilities >= 0.5).astype(int) == targets.astype(int)).mean())
    unique = np.unique(targets)
    auc = (
        float(roc_auc_score(targets, probabilities))
        if unique.size > 1
        else float("nan")
    )
    return {"auc": auc, "accuracy": accuracy}


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    name: str = "classifier",
    pos_weight: float | None = None,
) -> tuple[nn.Module, History]:
    """Train a binary classifier, early-stopping on validation ROC AUC.

    Args:
        model: Module emitting one logit per patch.
        train_loader: Loader yielding ``(image, label)`` pairs.
        val_loader: Validation loader.
        config: Hyperparameters; defaults to :class:`TrainConfig`.
        device: Compute device; defaults to CPU.
        name: Run name recorded in the history.
        pos_weight: Optional positive-class weight for class imbalance.

    Returns:
        ``(model, history)`` with the best-AUC weights loaded.
    """
    settings = config or TrainConfig()
    target_device = device or torch.device("cpu")
    weight = (
        None if pos_weight is None
        else torch.tensor([pos_weight], device=target_device)
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    return _fit(
        model, train_loader, val_loader, criterion, settings, target_device,
        name=name, monitor="val_auc", mode="max", extra_metrics=_auc_and_accuracy,
    )


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    name: str = "autoencoder",
) -> tuple[nn.Module, History]:
    """Train an autoencoder on reconstruction MSE, early-stopping on val loss.

    Args:
        model: Module mapping an image batch to a reconstruction of equal shape.
        train_loader: Loader yielding ``(image, image)`` pairs.
        val_loader: Validation loader of the same form.
        config: Hyperparameters; defaults to :class:`TrainConfig`.
        device: Compute device; defaults to CPU.
        name: Run name recorded in the history.

    Returns:
        ``(model, history)`` with the lowest-val-loss weights loaded.
    """
    settings = config or TrainConfig()
    return _fit(
        model, train_loader, val_loader, nn.MSELoss(), settings,
        device or torch.device("cpu"),
        name=name, monitor="val_loss", mode="min", extra_metrics=None,
    )


def save_checkpoint(model: nn.Module, path: Path) -> Path:
    """Write ``model``'s ``state_dict`` to ``path`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def forward_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and return ``(logits, targets)``."""
    model.to(device).eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, target in loader:
            output = model(inputs.to(device))
            logits.append(output.float().cpu().numpy())
            targets.append(np.asarray(target).astype(np.float64))
    return np.concatenate(logits), np.concatenate(targets)


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Map raw logits to probabilities with a numerically stable sigmoid."""
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))),
        np.exp(np.clip(logits, -60, 60)) / (1.0 + np.exp(np.clip(logits, -60, 60))),
    )
