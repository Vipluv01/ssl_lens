import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from ssl_lens.data import stratified_label_subset


class _FakeLabeledDataset:
    """Mimics LabeledDataset's interface without touching STL-10 on disk."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        return (None, int(self._labels[idx]))

    @property
    def labels(self) -> np.ndarray:
        return self._labels


def _make_dataset(n_per_class: int = 500, n_classes: int = 10) -> _FakeLabeledDataset:
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return _FakeLabeledDataset(labels)


def test_1_percent_keeps_every_class():
    """STL-10 train at 1% is only ~50 images across 10 classes -- exactly
    the regime where a plain random slice risks losing a class entirely."""
    ds = _make_dataset()
    sub = stratified_label_subset(ds, 0.01, seed=0)
    kept_labels = {ds.labels[i] for i in sub.indices}
    assert kept_labels == set(range(10)), f"lost classes: only {sorted(kept_labels)} survived"
    assert 40 <= len(sub) <= 60


def test_full_fraction_is_identity():
    ds = _make_dataset()
    sub = stratified_label_subset(ds, 1.0, seed=0)
    assert len(sub) == len(ds)


def test_fraction_is_balanced_across_classes():
    ds = _make_dataset(n_per_class=200, n_classes=5)
    sub = stratified_label_subset(ds, 0.1, seed=1)
    import collections
    counts = collections.Counter(ds.labels[i] for i in sub.indices)
    assert len(counts) == 5
    # Each class should be close to 10% of 200 = 20, not wildly skewed.
    for c, n in counts.items():
        assert 15 <= n <= 25, f"class {c} has {n} examples, expected ~20"


def test_rejects_bad_fraction():
    ds = _make_dataset()
    with pytest.raises(ValueError):
        stratified_label_subset(ds, 0.0, seed=0)
    with pytest.raises(ValueError):
        stratified_label_subset(ds, 1.5, seed=0)


def test_pretrain_dataset_matches_torchvision_byte_for_byte():
    """Regression test for a real bug: an earlier version of the memmap
    loader omitted STL-10's height/width axis swap (its on-disk format is
    column-major within each channel plane, a documented quirk of the
    dataset), which silently corrupted every single training image with no
    error anywhere -- caught only by comparing raw output against
    torchvision's reference loader directly, not by any shape or dtype
    check. Runs against the real downloaded STL-10 binary.
    """
    from pathlib import Path

    import numpy as np

    from ssl_lens.data import PretrainDataset

    root = Path(__file__).resolve().parents[1] / "data"
    if not (root / "stl10_binary" / "unlabeled_X.bin").exists():
        import pytest
        pytest.skip("STL-10 not downloaded in this environment")

    from torchvision.datasets import STL10
    ref = STL10(root=str(root), split="unlabeled", download=False)

    mine = PretrainDataset(root=str(root), max_images=10)
    for i in range(5):
        got = np.asarray(mine._data[i]).swapaxes(1, 2)
        want = ref.data[i]
        assert np.array_equal(got, want), f"image {i} does not match torchvision's reference loader"
