"""Toy 2D classification datasets, numpy-only (no sklearn dependency).

Two dimensions so every dataset — and every decision boundary a model learns on it — can
be *plotted*. Both are deliberately not linearly separable: a single neuron fails on
them, which is the whole pedagogical point of the DNN refresher chapters.
"""

from __future__ import annotations

import numpy as np


def make_moons(n: int = 200, noise: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles ("moons"), the classic binary toy problem.

    Returns ``X`` of shape ``(n, 2)`` (float64) and labels ``y`` of shape ``(n,)`` with
    values in {0, 1}. Class 0 is the upper arc, class 1 the lower arc shifted right/down
    so the tips interleave — separable by a curve, never by a line.
    """
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = np.linspace(0.0, np.pi, n0)
    t1 = np.linspace(0.0, np.pi, n1)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    x = np.concatenate([x0, x1]) + rng.normal(scale=noise, size=(n, 2))
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def make_spiral(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved spiral arms, one per class (the CS231n multi-class toy problem).

    Returns ``X`` of shape ``(n_per_class * n_classes, 2)`` and integer labels ``y`` in
    ``{0, ..., n_classes - 1}``. Radius grows with angle so the arms wrap around each
    other — a good stress test for softmax + cross-entropy training, and impossible for
    any linear classifier.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    x = np.zeros((n, 2))
    y = np.zeros(n, dtype=np.int64)
    for c in range(n_classes):
        idx = slice(c * n_per_class, (c + 1) * n_per_class)
        radius = np.linspace(0.05, 1.0, n_per_class)
        theta = (
            np.linspace(c * 2 * np.pi / n_classes, c * 2 * np.pi / n_classes + 1.5 * np.pi,
                        n_per_class)
            + rng.normal(scale=noise, size=n_per_class) * radius
        )
        x[idx] = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
        y[idx] = c
    perm = rng.permutation(n)
    return x[perm], y[perm]
