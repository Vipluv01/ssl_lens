"""Dataset wrappers for pretraining and for the label-efficiency evaluation.

STL-10 (Coates, Ng & Lee 2011) is used specifically because it was built for
this kind of study: 100,000 unlabeled images for pretraining, plus a much
smaller labeled set (5,000 train / 8,000 test, 10 classes) for evaluating
what pretraining bought you. Unlike CIFAR-10, its unlabeled split is drawn
from a broader distribution than the labeled classes -- closer to the real
situation "lots of raw data, little labeled data" this project is about.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import STL10

from ssl_lens.augmentations import TwoViews, eval_transform, simsiam_augmentation

IMAGE_SIZE = 96  # STL-10's native resolution -- no resizing needed for it,
                  # only for eval_transform's normalization step


_UNLABELED_IMAGE_BYTES = 3 * IMAGE_SIZE * IMAGE_SIZE  # STL-10 stores each
                                                        # image as a fixed-
                                                        # size flat byte block


class PretrainDataset(Dataset):
    """Yields (view1, view2) pairs from STL-10's unlabeled split. No labels
    are touched anywhere in this class -- that's the entire point of the
    pretraining phase, and keeping it structurally impossible for a label to
    leak in here (rather than just "not using" one that's available) is
    worth the small amount of extra code.

    Deliberately does NOT use torchvision's STL10 loader for this split.
    That loader reads the entire unlabeled binary (100,000 x 3 x 96 x 96
    bytes, ~2.5GB) into a single in-memory numpy array in __init__ -- and on
    this machine, that combined with training state pushed the process to
    10GB+ resident memory and a literal "stuck" state (confirmed directly
    via `top`, not inferred). A memmap over the same file lets the OS page
    image data in from disk on demand as __getitem__ actually accesses it,
    instead of materializing all 100,000 images in RAM before training even
    starts. The byte layout matches torchvision's own STL10 loader exactly
    (channel-major, i.e. R plane then G then B, each 96x96) so this reads
    the identical file format, just without the eager full load.
    """

    def __init__(self, root: str = "data", max_images: int | None = None) -> None:
        bin_path = Path(root) / "stl10_binary" / "unlabeled_X.bin"
        if not bin_path.exists():
            raise FileNotFoundError(
                f"{bin_path} not found -- download STL-10 first "
                "(torchvision.datasets.STL10(root=root, split='unlabeled', download=True))"
            )
        total_images = bin_path.stat().st_size // _UNLABELED_IMAGE_BYTES
        n = total_images if max_images is None else min(max_images, total_images)

        # mode="r": read-only memmap. Backed by the file itself, not RAM --
        # the OS's page cache handles what's actually resident, and it can
        # evict pages under memory pressure instead of the process holding
        # everything hostage the way an eager np.fromfile load does.
        self._data = np.memmap(bin_path, dtype=np.uint8, mode="r",
                                shape=(total_images, 3, IMAGE_SIZE, IMAGE_SIZE))[:n]
        self._two_views = TwoViews(simsiam_augmentation(IMAGE_SIZE))

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # STL-10's on-disk byte layout is channel-major but WIDTH-major
        # within each channel plane (column-major / Fortran-style storage,
        # a known quirk of this dataset's binary format) -- torchvision's
        # own loader corrects this with an explicit axis swap
        # (np.transpose(images, (0, 1, 3, 2))) after reshaping, and skipping
        # that step produces images that are wrong for every single sample
        # (verified directly: comparing this loader's raw output against
        # torchvision's before this fix showed a mismatch on all 5 images
        # checked). `.swapaxes(1, 2)` here is the single-image equivalent of
        # that same correction, applied to the (C, H, W) slice out of the
        # memmap before handing it to PIL, which wants (H, W, C).
        chw = np.asarray(self._data[idx]).swapaxes(1, 2)
        img = Image.fromarray(chw.transpose(1, 2, 0))
        return self._two_views(img)


class LabeledDataset(Dataset):
    """STL-10's train or test split with a single (non-augmented)
    eval_transform applied -- used for the linear probe, k-NN, and
    fine-tune evaluation stages, and for computing the encoder's actual
    downstream features."""

    def __init__(self, root: str = "data", split: str = "train") -> None:
        assert split in ("train", "test")
        self._base = STL10(root=root, split=split, download=False)
        self._transform = eval_transform(IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self._base[idx]
        return self._transform(img), label

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self._base.labels)


def stratified_label_subset(dataset: LabeledDataset, fraction: float, *, seed: int) -> Subset:
    """A class-balanced fraction of a labeled dataset.

    The whole point of the label-efficiency curve is comparing accuracy AT a
    given label budget across methods -- if a plain random slice at 1%
    (STL-10 train = 5,000 images / 10 classes = 500/class; 1% is only 50
    images total) happened to drop a class to zero by chance, the resulting
    number would reflect that accident, not the method's actual label
    efficiency. Stratifying removes that as a confound, the same reasoning
    as adapt.task.subsample_train on the fine-tuning project.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return Subset(dataset, list(range(len(dataset))))

    labels = dataset.labels
    by_class: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        by_class.setdefault(int(lbl), []).append(i)

    rng = random.Random(seed)
    keep: list[int] = []
    for lbl, idxs in by_class.items():
        n = max(1, round(len(idxs) * fraction))
        keep.extend(rng.sample(idxs, n))
    rng.shuffle(keep)
    return Subset(dataset, keep)
