#!/usr/bin/env python3
"""Run the whole study: train every model, write result tables and figures.

    python scripts/run_all.py                    # default configuration
    python scripts/run_all.py --epochs 4 --subset 4000   # quick smoke run

Five training stages run in order, because the later ones consume the earlier
ones' weights:

1. ``baseline-cnn``        - supervised CNN from scratch.
2. ``autoencoder``         - unsupervised reconstruction on all training patches.
3. ``encoder-frozen``      - stage 2's encoder, frozen, with a new head.
4. ``encoder-finetuned``   - stage 2's encoder, unfrozen, with a new head.
5. ``anomaly-autoencoder`` - reconstruction on benign training patches only.

Outputs land in ``results/*.csv`` and ``figures/*.png``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from histopath import anomaly, plots  # noqa: E402
from histopath import data as hdata  # noqa: E402
from histopath import evaluate as heval  # noqa: E402
from histopath.models import (  # noqa: E402
    BaselineCNN,
    ConvAutoencoder,
    EncoderClassifier,
    count_parameters,
)
from histopath.train import (  # noqa: E402
    History,
    TrainConfig,
    forward_logits,
    logits_to_probabilities,
    train_autoencoder,
    train_classifier,
)

LOGGER = logging.getLogger("run_all")

DISPLAY_NAMES: dict[str, str] = {
    "baseline-cnn": "Baseline CNN",
    "encoder-frozen": "AE encoder (frozen)",
    "encoder-finetuned": "AE encoder (fine-tuned)",
    "anomaly-detector": "Anomaly detector",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--epochs", type=int, default=10, help="maximum epochs per training stage"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=hdata.DEFAULT_SUBSET["train"],
        help="training subset size; validation and test take a quarter each",
    )
    parser.add_argument("--seed", type=int, default=hdata.DEFAULT_SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="minibatch size; 64 keeps peak memory modest on an 8 GB laptop",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="directory for model weights (git-ignored); omit to skip saving",
    )
    return parser.parse_args(argv)


def subset_sizes(train_size: int) -> dict[str, int]:
    """Return per-split subset sizes derived from the training size."""
    eval_size = max(2, train_size // 4)
    return {"train": train_size, "valid": eval_size, "test": eval_size}


def history_series(history: History) -> dict[str, list[float]]:
    """Extract the plottable series of a training history."""
    keys = ("train_loss", "val_loss", "train_auc", "val_auc")
    return {key: history.series(key) for key in keys if history.series(key)}


def write_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write ``rows`` to ``path`` as CSV using the union of their keys."""
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    """Train every model, score it and write all artefacts."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )

    root = hdata.project_root()
    results_dir = args.results or root / "results"
    figures_dir = args.figures or root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hdata.seed_everything(args.seed)
    device = hdata.resolve_device(args.device)
    sizes = subset_sizes(args.subset)
    started = time.perf_counter()
    LOGGER.info("device %s | subset %s | epochs %d", device, sizes, args.epochs)

    # ----------------------------------------------------------------- data
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {
        split: hdata.load_subset(split, sizes[split], args.seed)
        for split in hdata.SPLITS
    }
    stats = hdata.channel_stats("train", sizes["train"], args.seed)
    LOGGER.info("channel mean %s std %s", stats[0], stats[1])

    def classifier_dataset(split: str, train: bool) -> hdata.PatchDataset:
        images, labels = raw[split]
        return hdata.PatchDataset(
            images, labels, hdata.classifier_transform(train=train, stats=stats)
        )

    def autoencoder_dataset(
        split: str, train: bool, labelled: bool = False, mask: np.ndarray | None = None
    ) -> hdata.PatchDataset:
        images, labels = raw[split]
        if mask is not None:
            images, labels = np.asarray(images)[mask], labels[mask]
        return hdata.PatchDataset(
            images, labels, hdata.autoencoder_transform(train=train),
            return_label=labelled,
        )

    generator = torch.Generator().manual_seed(args.seed)

    # Models trained from scratch see channel-normalised patches.
    clf_loaders = {
        "train": hdata.make_loader(
            classifier_dataset("train", True), args.batch_size, True, generator
        ),
        "valid": hdata.make_loader(classifier_dataset("valid", False), args.batch_size),
        "test": hdata.make_loader(classifier_dataset("test", False), args.batch_size),
    }
    # Unlabelled pairs for the autoencoder's own reconstruction objective.
    ae_loaders = {
        "train": hdata.make_loader(
            autoencoder_dataset("train", True), args.batch_size, True, generator
        ),
        "valid": hdata.make_loader(autoencoder_dataset("valid", False), args.batch_size),
    }
    # Labelled patches on the *autoencoder's* scale. The transfer classifiers
    # reuse an encoder that was pretrained on [0, 1] inputs, so they must be
    # trained and scored on the same scale; feeding them standardised patches
    # would shift the input distribution out from under the pretrained weights
    # and make the comparison with the baseline meaningless.
    transfer_loaders = {
        "train": hdata.make_loader(
            autoencoder_dataset("train", True, labelled=True),
            args.batch_size, True, generator,
        ),
        "valid": hdata.make_loader(
            autoencoder_dataset("valid", False, labelled=True), args.batch_size
        ),
        "test": hdata.make_loader(
            autoencoder_dataset("test", False, labelled=True), args.batch_size
        ),
    }

    histories: dict[str, History] = {}
    metric_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def score_classifier(
        key: str, model: torch.nn.Module, loaders: dict[str, Any]
    ) -> None:
        """Pick a threshold on validation data, then score the test set."""
        val_logits, val_labels = forward_logits(model, loaders["valid"], device)
        test_logits, test_labels = forward_logits(model, loaders["test"], device)
        val_prob = logits_to_probabilities(val_logits)
        test_prob = logits_to_probabilities(test_logits)
        tuned = heval.youden_threshold(val_labels, val_prob)
        predictions[key] = (test_labels, test_prob, val_prob)

        for source, threshold in (("default_0.5", 0.5), ("val_youden", tuned)):
            metric_rows.append(
                {
                    "model": DISPLAY_NAMES[key],
                    "model_key": key,
                    "threshold_source": source,
                    **heval.classification_metrics(test_labels, test_prob, threshold),
                }
            )

    # --------------------------------------------------- 1. baseline CNN
    LOGGER.info("stage 1/5: baseline CNN")
    baseline = BaselineCNN()
    baseline, history = train_classifier(
        baseline, clf_loaders["train"], clf_loaders["valid"],
        TrainConfig(epochs=args.epochs, lr=1e-3), device, name="baseline-cnn",
    )
    histories["baseline-cnn"] = history
    score_classifier("baseline-cnn", baseline, clf_loaders)
    model_rows.append(
        {
            "model": DISPLAY_NAMES["baseline-cnn"],
            "model_key": "baseline-cnn",
            "parameters": count_parameters(baseline),
            "trainable_parameters": count_parameters(baseline, trainable_only=True),
            "epochs_run": len(history.records),
            "best_epoch": history.best_epoch,
            "best_val_auc": history.best_score,
            "train_seconds": round(history.wall_seconds, 1),
        }
    )

    # ------------------------------------------- 2. general autoencoder
    LOGGER.info("stage 2/5: convolutional autoencoder (unsupervised)")
    autoencoder = ConvAutoencoder()
    autoencoder, ae_history = train_autoencoder(
        autoencoder, ae_loaders["train"], ae_loaders["valid"],
        TrainConfig(epochs=args.epochs, lr=1e-3), device, name="autoencoder",
    )
    histories["autoencoder"] = ae_history
    model_rows.append(
        {
            "model": "Convolutional autoencoder",
            "model_key": "autoencoder",
            "parameters": count_parameters(autoencoder),
            "trainable_parameters": count_parameters(autoencoder, trainable_only=True),
            "epochs_run": len(ae_history.records),
            "best_epoch": ae_history.best_epoch,
            "best_val_mse": ae_history.best_score,
            "train_seconds": round(ae_history.wall_seconds, 1),
        }
    )

    # ---------------------------------------- 3 & 4. transfer learning
    for stage, (key, freeze, lr) in enumerate(
        (("encoder-frozen", True, 1e-3), ("encoder-finetuned", False, 3e-4)), start=3
    ):
        LOGGER.info("stage %d/5: %s", stage, key)
        transfer = EncoderClassifier(
            encoder=copy.deepcopy(autoencoder.encoder),
            latent_channels=autoencoder.latent_channels,
            freeze_encoder=freeze,
        )
        transfer, transfer_history = train_classifier(
            transfer, transfer_loaders["train"], transfer_loaders["valid"],
            TrainConfig(epochs=args.epochs, lr=lr), device, name=key,
        )
        histories[key] = transfer_history
        score_classifier(key, transfer, transfer_loaders)
        model_rows.append(
            {
                "model": DISPLAY_NAMES[key],
                "model_key": key,
                "parameters": count_parameters(transfer),
                "trainable_parameters": count_parameters(transfer, trainable_only=True),
                "epochs_run": len(transfer_history.records),
                "best_epoch": transfer_history.best_epoch,
                "best_val_auc": transfer_history.best_score,
                "train_seconds": round(transfer_history.wall_seconds, 1),
            }
        )
        if args.checkpoints:
            torch.save(transfer.state_dict(), Path(args.checkpoints) / f"{key}.pt")

    # ------------------------------------- 5. anomaly detection on benign
    LOGGER.info("stage 5/5: anomaly autoencoder (benign patches only)")
    benign_train = hdata.negative_indices(raw["train"][1])
    benign_valid = hdata.negative_indices(raw["valid"][1])
    LOGGER.info(
        "benign-only training set: %d train / %d valid patches",
        benign_train.size, benign_valid.size,
    )
    anomaly_model = ConvAutoencoder()
    anomaly_model, anomaly_history = train_autoencoder(
        anomaly_model,
        hdata.make_loader(
            autoencoder_dataset("train", True, mask=benign_train),
            args.batch_size, True, generator,
        ),
        hdata.make_loader(
            autoencoder_dataset("valid", False, mask=benign_valid), args.batch_size
        ),
        TrainConfig(epochs=args.epochs, lr=1e-3), device, name="anomaly-autoencoder",
    )
    histories["anomaly-autoencoder"] = anomaly_history

    val_errors, val_labels = anomaly.reconstruction_errors(
        anomaly_model, transfer_loaders["valid"], device
    )
    test_errors, test_error_labels = anomaly.reconstruction_errors(
        anomaly_model, transfer_loaders["test"], device
    )
    test_error_auc = anomaly.error_auc(test_errors, test_error_labels)
    LOGGER.info("reconstruction-error ROC AUC on test: %.4f", test_error_auc)

    thresholds = {
        "val_youden": anomaly.select_threshold(val_errors, val_labels, "youden"),
        "val_benign_p95": anomaly.select_threshold(
            val_errors, val_labels, "percentile", percentile=95.0
        ),
    }
    for source, choice in thresholds.items():
        metric_rows.append(
            {
                "model": DISPLAY_NAMES["anomaly-detector"],
                "model_key": "anomaly-detector",
                "threshold_source": source,
                **anomaly.anomaly_metrics(test_errors, test_error_labels, choice.value),
            }
        )

    model_rows.append(
        {
            "model": "Anomaly autoencoder (benign only)",
            "model_key": "anomaly-autoencoder",
            "parameters": count_parameters(anomaly_model),
            "trainable_parameters": count_parameters(anomaly_model, trainable_only=True),
            "epochs_run": len(anomaly_history.records),
            "best_epoch": anomaly_history.best_epoch,
            "best_val_mse": anomaly_history.best_score,
            "train_seconds": round(anomaly_history.wall_seconds, 1),
        }
    )

    error_rows = [
        {
            "split": "test",
            "error_roc_auc": test_error_auc,
            **anomaly.error_summary(test_errors, test_error_labels),
            "threshold_youden": thresholds["val_youden"].value,
            "threshold_benign_p95": thresholds["val_benign_p95"].value,
        },
        {
            "split": "valid",
            "error_roc_auc": anomaly.error_auc(val_errors, val_labels),
            **anomaly.error_summary(val_errors, val_labels),
        },
    ]

    # ------------------------------------------------------------ tables
    heval.write_metrics_csv(metric_rows, results_dir / "metrics.csv")

    # Per-patch anomaly scores, so the anomaly figures can be rebuilt from the
    # cached images without retraining anything.
    write_rows(
        [
            {
                "test_index": index,
                "label": int(test_error_labels[index]),
                "reconstruction_error": float(test_errors[index]),
            }
            for index in range(test_errors.size)
        ],
        results_dir / "anomaly_test_scores.csv",
    )
    write_rows(model_rows, results_dir / "model_summary.csv")
    write_rows(error_rows, results_dir / "anomaly_error_summary.csv")

    history_path = results_dir / "training_history.csv"
    history_path.unlink(missing_ok=True)
    for history in histories.values():
        history.write_csv(history_path)

    write_rows(
        [
            {
                "split": split,
                "n_patches": int(raw[split][1].size),
                "n_benign": int((raw[split][1] == 0).sum()),
                "n_metastatic": int((raw[split][1] == 1).sum()),
                "positive_rate": float(raw[split][1].mean()),
            }
            for split in hdata.SPLITS
        ],
        results_dir / "dataset_summary.csv",
    )

    write_rows(
        [
            {
                "dataset": hdata.PCAM.name,
                "source": f"huggingface.co/datasets/{hdata.PCAM.repo_id}",
                "revision": hdata.PCAM.revision,
                "licence": hdata.PCAM.license_id,
                "seed": args.seed,
                "train_patches": sizes["train"],
                "valid_patches": sizes["valid"],
                "test_patches": sizes["test"],
                "max_epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "channel_mean": ";".join(f"{v:.5f}" for v in stats[0]),
                "channel_std": ";".join(f"{v:.5f}" for v in stats[1]),
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
                "total_seconds": round(time.perf_counter() - started, 1),
            }
        ],
        results_dir / "run_metadata.csv",
    )

    # ----------------------------------------------------------- figures
    LOGGER.info("writing figures")
    plots.plot_training_curves(
        {
            DISPLAY_NAMES[key]: history_series(histories[key])
            for key in ("baseline-cnn", "encoder-frozen", "encoder-finetuned")
        },
        figures_dir / "training_curves_classifiers.png",
        metric_key="auc",
        metric_label="ROC AUC",
        title="Classifier training - loss and validation ROC AUC per epoch",
    )
    plots.plot_training_curves(
        {
            "Autoencoder (all patches)": history_series(histories["autoencoder"]),
            "Autoencoder (benign only)": history_series(histories["anomaly-autoencoder"]),
        },
        figures_dir / "training_curves_autoencoders.png",
        metric_key=None,
        title="Autoencoder training - reconstruction MSE per epoch",
    )

    plots.plot_roc_overlay(
        {
            DISPLAY_NAMES[key]: heval.roc_points(*predictions[key][:2])
            for key in ("baseline-cnn", "encoder-frozen", "encoder-finetuned")
        },
        figures_dir / "roc_overlay.png",
    )

    tuned_rows = {
        row["model_key"]: row
        for row in metric_rows
        if row["threshold_source"] in {"val_youden"}
    }
    matrices = {
        DISPLAY_NAMES[key]: heval.confusion(
            *predictions[key][:2], threshold=tuned_rows[key]["threshold"]
        )
        for key in ("baseline-cnn", "encoder-frozen", "encoder-finetuned")
    }
    anomaly_matrix = np.array(
        [
            [tuned_rows["anomaly-detector"]["tn"], tuned_rows["anomaly-detector"]["fp"]],
            [tuned_rows["anomaly-detector"]["fn"], tuned_rows["anomaly-detector"]["tp"]],
        ]
    )
    matrices[DISPLAY_NAMES["anomaly-detector"]] = anomaly_matrix
    plots.plot_confusion_matrices(
        matrices,
        figures_dir / "confusion_matrices.png",
        title="Confusion matrices at the validation-selected threshold (test set)",
    )

    plots.plot_calibration(
        {
            DISPLAY_NAMES[key]: (
                lambda curve: (curve.mean_predicted, curve.fraction_positive)
            )(heval.calibration_curve_points(*predictions[key][:2]))
            for key in ("baseline-cnn", "encoder-frozen", "encoder-finetuned")
        },
        figures_dir / "calibration.png",
    )

    # reconstruction grid: alternating benign and metastatic test patches
    test_images, test_labels_raw = raw["test"]
    chosen: list[int] = []
    for class_value in (0, 1):
        pool = np.flatnonzero(test_labels_raw == class_value)[:4]
        chosen.extend(pool.tolist())
    order = [chosen[i // 2 + 4 * (i % 2)] for i in range(8)]
    grid_inputs = np.asarray(test_images)[order].astype(np.float32) / 255.0
    with torch.no_grad():
        batch = torch.from_numpy(grid_inputs).permute(0, 3, 1, 2).to(device)
        reconstructed = autoencoder.to(device).eval()(batch)
    plots.plot_reconstruction_grid(
        grid_inputs,
        reconstructed.permute(0, 2, 3, 1).cpu().numpy(),
        figures_dir / "autoencoder_reconstructions.png",
        labels=test_labels_raw[order],
        title="Autoencoder reconstructions of held-out test patches",
    )

    plots.plot_error_histograms(
        test_errors,
        test_error_labels,
        figures_dir / "reconstruction_error_hist.png",
        threshold=thresholds["val_youden"].value,
        title="Benign-only autoencoder: reconstruction error by true class (test set)",
    )
    plots.plot_single_roc(
        *heval.roc_points(test_error_labels, test_errors),
        path=figures_dir / "anomaly_roc.png",
        label="reconstruction error as score",
        title="Anomaly detection without tumour labels (test set)",
    )

    worst = anomaly.top_error_indices(test_errors, k=16)
    plots.plot_patch_grid(
        np.asarray(test_images)[worst].astype(np.float32) / 255.0,
        figures_dir / "top_error_patches.png",
        labels=test_error_labels[worst].astype(int),
        scores=test_errors[worst],
        title="Highest-error test patches under the benign-only autoencoder",
    )

    if args.checkpoints:
        Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
        torch.save(baseline.state_dict(), Path(args.checkpoints) / "baseline-cnn.pt")
        torch.save(autoencoder.state_dict(), Path(args.checkpoints) / "autoencoder.pt")
        torch.save(
            anomaly_model.state_dict(), Path(args.checkpoints) / "anomaly-autoencoder.pt"
        )

    elapsed = time.perf_counter() - started
    LOGGER.info("done in %.1f s (%.1f min)", elapsed, elapsed / 60)
    print(f"\n{'model':<26}{'threshold':>14}{'ROC AUC':>10}{'acc':>8}"
          f"{'recall':>8}{'spec':>8}{'F1':>8}")
    for row in metric_rows:
        print(
            f"{row['model']:<26}{row['threshold_source']:>14}"
            f"{row['roc_auc']:>10.4f}{row['accuracy']:>8.4f}"
            f"{row['recall']:>8.4f}{row['specificity']:>8.4f}{row['f1']:>8.4f}"
        )
    print(f"\nresults -> {results_dir}\nfigures -> {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
