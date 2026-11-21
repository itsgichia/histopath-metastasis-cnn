#!/usr/bin/env python3
"""Download the PatchCamelyon slice used by this study and build its caches.

One command gets everything the experiments need:

    python scripts/download_data.py

Downloads land in ``data/pcam/`` and decoded subsets in ``data/cache/``; both
are git-ignored. Re-running is cheap: existing files are verified, not refetched.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from histopath import data as hdata  # noqa: E402

LOGGER = logging.getLogger("download_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subset",
        type=int,
        default=hdata.DEFAULT_SUBSET["train"],
        help="training subset size; validation and test are scaled to a quarter of it",
    )
    parser.add_argument(
        "--seed", type=int, default=hdata.DEFAULT_SEED, help="subset draw seed"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip SHA-256 verification"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild the decoded caches"
    )
    return parser.parse_args(argv)


def subset_sizes(train_size: int) -> dict[str, int]:
    """Return per-split subset sizes derived from the training size."""
    eval_size = max(2, train_size // 4)
    return {"train": train_size, "valid": eval_size, "test": eval_size}


def main(argv: list[str] | None = None) -> int:
    """Download shards, build caches and report what was obtained."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = hdata.PCAM
    print(f"dataset : {source.name}")
    print(f"source  : huggingface.co/datasets/{source.repo_id}")
    print(f"revision: {source.revision}")
    print(f"licence : {source.license_id}\n")

    paths = hdata.download_shards(source=source, verify=not args.no_verify)
    sizes = subset_sizes(args.subset)

    print(f"\n{'split':<8}{'shard rows':>12}{'shard pos':>12}"
          f"{'subset':>10}{'subset pos':>12}")
    for split in hdata.SPLITS:
        shard_labels = hdata.read_split_labels(paths[split])
        hdata.build_subset_cache(
            split, sizes[split], args.seed, source, overwrite=args.overwrite
        )
        _, subset_labels = hdata.load_subset(split, sizes[split], args.seed, source)
        print(
            f"{split:<8}{shard_labels.size:>12,}{shard_labels.mean():>12.4f}"
            f"{subset_labels.size:>10,}{subset_labels.mean():>12.4f}"
        )

    mean, std = hdata.channel_stats("train", sizes["train"], args.seed)
    print(
        "\ntrain channel mean: "
        + ", ".join(f"{value:.4f}" for value in mean)
        + "\ntrain channel std : "
        + ", ".join(f"{value:.4f}" for value in std)
    )

    images, _ = hdata.load_subset("train", sizes["train"], args.seed, source)
    print(f"patch shape       : {tuple(np.asarray(images[0]).shape)}")
    print(f"\ncache directory   : {hdata.cache_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
