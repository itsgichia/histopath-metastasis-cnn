"""Data access for the PatchCamelyon (PCam) lymph node histopathology benchmark.

The module downloads a small, reproducible slice of PCam from the Hugging Face
mirror ``1aurent/PatchCamelyon`` (CC0-1.0), decodes it once into ``.npy`` caches
and exposes PyTorch datasets plus the three transform pipelines used by the
study:

* :func:`classifier_transform` - dataset-specific channel normalisation, for the
  supervised models that are trained from scratch on PCam.
* :func:`autoencoder_transform` - no normalisation, because the autoencoder
  reconstructs into a ``Sigmoid`` output bounded to ``[0, 1]``.
* :func:`imagenet_transform` - ImageNet channel statistics, provided for any
  pretrained backbone. The default experiment grid uses no pretrained weights,
  so this pipeline is unused unless a backbone is added.

Nothing in this module writes into the repository tree apart from
``data/`` (git-ignored) and the small JSON of normalisation statistics.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

Split = Literal["train", "valid", "test"]
SPLITS: tuple[Split, ...] = ("train", "valid", "test")

#: Patch side length in pixels. PCam patches are fixed at 96x96 RGB.
PATCH_SIZE = 96

#: ImageNet channel statistics, for use with a pretrained backbone.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Fallback PCam statistics, as measured on the default 20,000-patch training
#: subset. The live values are recomputed from the cache by
#: :func:`compute_channel_stats` and memoised in ``data/cache/norm_stats.json``;
#: these constants only serve code paths that run before any data has been
#: downloaded (for example the unit tests).
PCAM_FALLBACK_MEAN: tuple[float, float, float] = (0.70051, 0.53881, 0.69137)
PCAM_FALLBACK_STD: tuple[float, float, float] = (0.23507, 0.27767, 0.21339)

#: Default seed for every deterministic subset draw in the study.
DEFAULT_SEED = 20260214


@dataclass(frozen=True)
class DatasetSource:
    """A pinned, downloadable dataset revision.

    Attributes:
        name: Human readable dataset name.
        repo_id: Hugging Face dataset repository identifier.
        revision: Commit SHA the files are pulled from, so a rerun months later
            resolves to byte-identical shards.
        license_id: SPDX-style licence identifier of the hosted copy.
        files: Mapping of split name to the repository-relative shard path.
        sha256: Mapping of split name to the SHA-256 of the downloaded shard.
    """

    name: str
    repo_id: str
    revision: str
    license_id: str
    files: dict[str, str] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)

    def url(self, split: str) -> str:
        """Return the resolve URL of the shard backing ``split``."""
        return (
            f"https://huggingface.co/datasets/{self.repo_id}/resolve/"
            f"{self.revision}/{self.files[split]}"
        )

    def local_path(self, split: str, root: Path) -> Path:
        """Return the local cache path of the shard backing ``split``."""
        return root / Path(self.files[split]).name


#: The dataset actually used by this study.
#:
#: Only the first shard of each PCam split is fetched. Those shards are
#: internally shuffled and close to class balanced, so a stratified draw from
#: one shard is a sound small-scale stand-in for the full split; see the README
#: for the verification numbers.
PCAM = DatasetSource(
    name="PatchCamelyon (PCam)",
    repo_id="1aurent/PatchCamelyon",
    revision="e4bd149e7a868a9d811fdd9f9a9fb78c05c104ab",
    license_id="CC0-1.0",
    files={
        "train": "data/train-00000-of-00013-4717c3cf92578c96.parquet",
        "valid": "data/valid-00000-of-00002-0e1a29e0620125c6.parquet",
        "test": "data/test-00000-of-00002-bb04e6313f58efa0.parquet",
    },
    sha256={
        "train": "99ed88ef674a4462d1cabb418699d210b288b27b51486cea5fa10e1edd2817dd",
        "valid": "167e61e4f2bda552feb0de6460566ac75f862927d132f7e14e4534ecbd816b4a",
        "test": "2843436b6f9d3925645f7888e8de04125d86d7a65682c401e679ab2951e7e781",
    },
)

#: Default subset sizes, chosen so the full study finishes on a laptop CPU.
DEFAULT_SUBSET: dict[str, int] = {"train": 20_000, "valid": 5_000, "test": 5_000}

CLASS_NAMES: tuple[str, str] = ("benign", "metastatic")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Return the repository root, inferred from this file's location."""
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return the git-ignored directory holding raw shards and caches."""
    return project_root() / "data"


def raw_dir() -> Path:
    """Return the directory holding the downloaded parquet shards."""
    return data_root() / "pcam"


def cache_dir() -> Path:
    """Return the directory holding decoded ``.npy`` subset caches."""
    return data_root() / "cache"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_shards(
    source: DatasetSource = PCAM,
    splits: Iterable[str] = SPLITS,
    dest: Path | None = None,
    verify: bool = True,
) -> dict[str, Path]:
    """Download the parquet shard of each requested split.

    Existing files are left alone, so the function is safe to re-run.

    Args:
        source: Pinned dataset revision to pull from.
        splits: Split names to fetch.
        dest: Target directory. Defaults to :func:`raw_dir`.
        verify: Whether to check the SHA-256 of each file against ``source``.

    Returns:
        Mapping of split name to the local shard path.

    Raises:
        ValueError: If ``verify`` is set and a digest does not match.
    """
    target_dir = dest or raw_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split in splits:
        path = source.local_path(split, target_dir)
        if not path.exists():
            url = source.url(split)
            LOGGER.info("downloading %s -> %s", url, path)
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url) as response:  # noqa: S310
                with tmp.open("wb") as handle:
                    while block := response.read(1 << 20):
                        handle.write(block)
            tmp.replace(path)
        else:
            LOGGER.info("already present: %s", path)

        expected = source.sha256.get(split)
        if verify and expected:
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"checksum mismatch for {path.name}: "
                    f"expected {expected}, got {actual}"
                )
        paths[split] = path

    return paths


# --------------------------------------------------------------------------- #
# Deterministic subset selection
# --------------------------------------------------------------------------- #
def stratified_subset_indices(
    labels: Sequence[int] | np.ndarray,
    n_total: int,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Draw a class-balanced subset of row indices, reproducibly.

    The draw depends only on ``labels``, ``n_total`` and ``seed``, so the same
    arguments always select the same rows regardless of platform or of how many
    other draws happened first. Returned indices are sorted ascending, which
    also makes the subsequent parquet read sequential.

    Args:
        labels: Per-row class labels of the full split.
        n_total: Number of rows to select. Split evenly across the classes; if a
            class has too few rows, every available row of it is taken.
        seed: Seed of the generator used for the within-class draw.

    Returns:
        Sorted array of selected row indices.

    Raises:
        ValueError: If ``n_total`` is not positive or ``labels`` is empty.
    """
    label_array = np.asarray(labels).astype(np.int64).ravel()
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")

    classes = np.unique(label_array)
    per_class = n_total // len(classes)
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []

    for class_value in classes:
        candidates = np.flatnonzero(label_array == class_value)
        take = min(per_class, candidates.size)
        if take < per_class:
            LOGGER.warning(
                "class %s has only %d rows, requested %d",
                class_value,
                candidates.size,
                per_class,
            )
        chosen.append(rng.choice(candidates, size=take, replace=False))

    return np.sort(np.concatenate(chosen))


# --------------------------------------------------------------------------- #
# Decoding and caching
# --------------------------------------------------------------------------- #
def read_split_labels(path: Path) -> np.ndarray:
    """Read the full label column of a PCam parquet shard as ``int64``."""
    import pyarrow.parquet as pq

    column = pq.ParquetFile(path).read(columns=["label"])["label"].to_numpy()
    return np.asarray(column).astype(np.int64)


def _decode_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    """Decode the images at ``indices`` into a ``uint8`` NHWC array."""
    import pyarrow.parquet as pq
    from PIL import Image

    parquet = pq.ParquetFile(path)
    images = np.empty((indices.size, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    wanted = set(indices.tolist())
    position = {index: slot for slot, index in enumerate(indices.tolist())}

    row_offset = 0
    for group in range(parquet.metadata.num_row_groups):
        n_rows = parquet.metadata.row_group(group).num_rows
        group_rows = range(row_offset, row_offset + n_rows)
        if wanted.isdisjoint(group_rows):
            row_offset += n_rows
            continue

        table = parquet.read_row_group(group, columns=["image"])
        payloads = table["image"].to_pylist()
        for local, payload in enumerate(payloads):
            absolute = row_offset + local
            if absolute not in wanted:
                continue
            raw = payload["bytes"] if isinstance(payload, dict) else payload
            with Image.open(io.BytesIO(raw)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if array.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                raise ValueError(
                    f"unexpected patch shape {array.shape} at row {absolute}"
                )
            images[position[absolute]] = array
        row_offset += n_rows

    return images


def build_subset_cache(
    split: str,
    n_total: int,
    seed: int = DEFAULT_SEED,
    source: DatasetSource = PCAM,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Decode a deterministic subset of ``split`` into ``.npy`` caches.

    Args:
        split: One of ``"train"``, ``"valid"``, ``"test"``.
        n_total: Requested subset size (class balanced).
        seed: Seed of the subset draw.
        source: Dataset revision the shard came from.
        overwrite: Rebuild even if the cache already exists.

    Returns:
        ``(images_path, labels_path)`` of the cached arrays.
    """
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_n{n_total}_s{seed}"
    images_path = cache / f"{stem}_images.npy"
    labels_path = cache / f"{stem}_labels.npy"

    if images_path.exists() and labels_path.exists() and not overwrite:
        LOGGER.info("cache hit: %s", images_path.name)
        return images_path, labels_path

    shard = source.local_path(split, raw_dir())
    if not shard.exists():
        raise FileNotFoundError(
            f"missing shard {shard}; run scripts/download_data.py first"
        )

    all_labels = read_split_labels(shard)
    indices = stratified_subset_indices(all_labels, n_total, seed)
    LOGGER.info(
        "decoding %d/%d rows of %s (positive rate %.4f)",
        indices.size,
        all_labels.size,
        split,
        float(all_labels[indices].mean()),
    )
    images = _decode_rows(shard, indices)
    np.save(images_path, images)
    np.save(labels_path, all_labels[indices].astype(np.int64))
    return images_path, labels_path


def load_subset(
    split: str,
    n_total: int,
    seed: int = DEFAULT_SEED,
    source: DatasetSource = PCAM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for a cached subset, building it if needed."""
    images_path, labels_path = build_subset_cache(split, n_total, seed, source)
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)
    return images, labels


# --------------------------------------------------------------------------- #
# Normalisation statistics
# --------------------------------------------------------------------------- #
def compute_channel_stats(
    images: np.ndarray, chunk_size: int = 512
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute per-channel mean and standard deviation in ``[0, 1]`` units.

    Accumulated in chunks from running sums, so a memory-mapped array of tens of
    thousands of patches is never materialised as float64 all at once.

    Args:
        images: ``uint8`` array shaped ``(n, h, w, 3)``.
        chunk_size: Number of patches accumulated per pass.

    Returns:
        ``(mean, std)`` as three-tuples of floats.
    """
    channels = images.shape[-1]
    total = np.zeros(channels, dtype=np.float64)
    total_squares = np.zeros(channels, dtype=np.float64)
    count = 0

    for start in range(0, len(images), chunk_size):
        block = np.asarray(images[start : start + chunk_size], dtype=np.float64)
        block = block.reshape(-1, channels) / 255.0
        total += block.sum(axis=0)
        total_squares += np.square(block).sum(axis=0)
        count += block.shape[0]

    if count == 0:
        raise ValueError("cannot compute statistics of an empty array")

    mean = total / count
    variance = np.maximum(total_squares / count - np.square(mean), 0.0)
    return tuple(mean.tolist()), tuple(np.sqrt(variance).tolist())


def channel_stats(
    split: str = "train",
    n_total: int | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the normalisation statistics of the training subset.

    The value is measured once from the cached training images and memoised in
    ``data/cache/norm_stats.json``. If no cache exists yet the module-level
    fallback constants are returned so that import-time code paths and tests do
    not require a download.
    """
    size = DEFAULT_SUBSET["train"] if n_total is None else n_total
    stats_path = cache_dir() / "norm_stats.json"

    if stats_path.exists():
        payload = json.loads(stats_path.read_text())
        entry = payload.get(f"{split}_n{size}_s{seed}")
        if entry:
            return tuple(entry["mean"]), tuple(entry["std"])

    images_path = cache_dir() / f"{split}_n{size}_s{seed}_images.npy"
    if not images_path.exists():
        LOGGER.warning("no cache for %s; using fallback statistics", images_path.name)
        return PCAM_FALLBACK_MEAN, PCAM_FALLBACK_STD

    mean, std = compute_channel_stats(np.load(images_path, mmap_mode="r"))
    payload = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    payload[f"{split}_n{size}_s{seed}"] = {"mean": list(mean), "std": list(std)}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2) + "\n")
    return mean, std


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
class PatchTransform:
    """Convert a ``uint8`` HWC patch to a normalised CHW float tensor.

    Augmentation is restricted to the dihedral group (flips and quarter turns).
    Histopathology patches carry no canonical orientation, so those operations
    are label preserving, unlike crops or shears which would change the amount
    of tissue in view.

    Args:
        mean: Per-channel means to subtract, or ``None`` to skip normalisation.
        std: Per-channel standard deviations to divide by, or ``None``.
        augment: Whether to apply random flips and quarter turns.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        augment: bool = False,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be given together")
        self.mean = None if mean is None else torch.tensor(mean).view(-1, 1, 1).float()
        self.std = None if std is None else torch.tensor(std).view(-1, 1, 1).float()
        self.augment = augment

    def __call__(self, patch: np.ndarray) -> torch.Tensor:
        """Transform one patch."""
        # copy=True: cached subsets are memory mapped and therefore read-only,
        # which torch.from_numpy refuses to wrap without warning.
        array = np.array(patch, dtype=np.uint8, copy=True, order="C")
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        tensor = tensor.div_(255.0)

        if self.augment:
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(2,))
            if torch.rand(()) < 0.5:
                tensor = torch.flip(tensor, dims=(1,))
            turns = int(torch.randint(0, 4, ()))
            if turns:
                tensor = torch.rot90(tensor, turns, dims=(1, 2))
            tensor = tensor.contiguous()

        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
        return tensor


def classifier_transform(
    train: bool = False,
    stats: tuple[Sequence[float], Sequence[float]] | None = None,
) -> PatchTransform:
    """Pipeline for the supervised models trained from scratch on PCam."""
    mean, std = stats if stats is not None else channel_stats()
    return PatchTransform(mean=mean, std=std, augment=train)


def autoencoder_transform(train: bool = False) -> PatchTransform:
    """Pipeline for the autoencoders: scaled to ``[0, 1]``, not normalised.

    The decoder ends in a ``Sigmoid``, so targets must live in ``[0, 1]`` for the
    reconstruction loss to be well posed.
    """
    return PatchTransform(mean=None, std=None, augment=train)


def imagenet_transform(train: bool = False) -> PatchTransform:
    """Pipeline for a pretrained ImageNet backbone (unused by default)."""
    return PatchTransform(mean=IMAGENET_MEAN, std=IMAGENET_STD, augment=train)


# --------------------------------------------------------------------------- #
# Datasets and loaders
# --------------------------------------------------------------------------- #
class PatchDataset(Dataset):
    """In-memory PCam patch dataset.

    Args:
        images: ``uint8`` array shaped ``(n, 96, 96, 3)``.
        labels: Integer labels shaped ``(n,)``.
        transform: Callable mapping a patch to a tensor.
        return_label: When ``False``, ``__getitem__`` returns the image twice,
            which is the ``(input, target)`` pairing an autoencoder needs.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Callable[[np.ndarray], torch.Tensor],
        return_label: bool = True,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError(
                f"images and labels disagree: {len(images)} vs {len(labels)}"
            )
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)
        self.transform = transform
        self.return_label = return_label

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = np.asarray(self.images[index])
        tensor = self.transform(patch)
        if self.return_label:
            return tensor, torch.tensor(float(self.labels[index]))
        return tensor, tensor

    def subset(self, indices: np.ndarray) -> "PatchDataset":
        """Return a view-like copy restricted to ``indices``."""
        selected = np.asarray(indices)
        return PatchDataset(
            images=np.asarray(self.images)[selected],
            labels=self.labels[selected],
            transform=self.transform,
            return_label=self.return_label,
        )


def make_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a ``DataLoader`` with settings that behave well on macOS.

    Worker processes are deliberately disabled: the subsets are small enough to
    sit in RAM, and single-process loading avoids the fork-safety warnings that
    ``mps`` plus multiprocessing produce on macOS.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def negative_indices(labels: np.ndarray) -> np.ndarray:
    """Return the indices of the benign (non-tumour) patches."""
    return np.flatnonzero(np.asarray(labels).astype(np.int64) == 0)


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a compute device.

    Args:
        preference: ``"auto"``, ``"mps"``, ``"cuda"`` or ``"cpu"``. ``"auto"``
            prefers Apple ``mps``, then ``cuda``, then ``cpu``.

    Returns:
        The selected device.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs for repeatable runs."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
