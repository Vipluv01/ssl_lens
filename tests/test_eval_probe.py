"""Sanity checks for the linear-probe and k-NN accuracy functions against
synthetic features with KNOWN separability -- validates the eval MACHINERY
itself before it's ever pointed at a real encoder's embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ssl_lens.eval_probe import knn_accuracy, linear_probe_accuracy


def _linearly_separable_data(n_per_class=50, n_classes=4, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, size=(n_classes, dim))  # well-separated centers
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.normal(0, 0.5, size=(n_per_class, dim)))
        y.append(np.full(n_per_class, c))
    return np.concatenate(X), np.concatenate(y)


def test_linear_probe_recovers_easy_separation():
    X, y = _linearly_separable_data()
    # Split into train/test halves.
    n = len(X)
    idx = np.random.default_rng(1).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]
    acc = linear_probe_accuracy(X[tr], y[tr], X[te], y[te])
    assert acc > 0.95, f"trivially separable classes should give near-perfect accuracy, got {acc}"


def test_knn_recovers_easy_separation():
    X, y = _linearly_separable_data()
    n = len(X)
    idx = np.random.default_rng(1).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]
    acc = knn_accuracy(X[tr], y[tr], X[te], y[te], k=5)
    assert acc > 0.95, f"trivially separable classes should give near-perfect k-NN accuracy, got {acc}"


def test_linear_probe_fails_on_random_labels():
    """A representation with NO signal (random labels) should score near
    chance -- confirms the probe isn't somehow always reporting high
    accuracy regardless of input."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(200, 16))
    y = rng.integers(0, 4, size=200)
    n = len(X)
    tr, te = slice(0, 100), slice(100, 200)
    acc = linear_probe_accuracy(X[tr], y[tr], X[te], y[te])
    assert acc < 0.45, f"random labels should score near chance (0.25), got {acc}"


def test_knn_handles_k_larger_than_available_neighbors():
    """At a 1% label fraction, some classes may have fewer than k examples
    -- must clamp gracefully rather than raise."""
    X, y = _linearly_separable_data(n_per_class=3, n_classes=2)
    acc = knn_accuracy(X, y, X, y, k=50)  # k=50 >> 6 available training points
    assert 0.0 <= acc <= 1.0
