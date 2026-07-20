"""Shared plotting helper for the DNN refresher chapters."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray,
                           y: np.ndarray, ax: plt.Axes | None = None,
                           resolution: int = 200) -> plt.Axes:
    """Shade a model's decision regions over the 2D plane and scatter the data on top.

    ``predict`` maps an ``(n, 2)`` array of points to ``(n,)`` scores or class labels —
    continuous scores show the boundary as a gradient, integer labels as flat regions.
    Evaluates ``predict`` on a ``resolution x resolution`` grid covering the data with a
    margin, so a slow scalar-engine model is best passed with a modest ``resolution``
    (e.g. 60: 3,600 forward passes instead of 40,000).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    margin = 0.4
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = np.asarray(predict(grid), dtype=np.float64).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=25, zorder=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return ax
