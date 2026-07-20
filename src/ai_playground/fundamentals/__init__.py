"""DNN fundamentals: a scalar autograd engine and MLP, built for the refresher chapters.

See notebooks/00_dnn_refresher/ for the cell-by-cell construction of everything here.
"""

from ai_playground.fundamentals.autograd import Value
from ai_playground.fundamentals.datasets import make_moons, make_spiral
from ai_playground.fundamentals.nn import MLP, Layer, Module, Neuron
from ai_playground.fundamentals.viz import plot_decision_boundary

__all__ = [
    "MLP",
    "Layer",
    "Module",
    "Neuron",
    "Value",
    "make_moons",
    "make_spiral",
    "plot_decision_boundary",
]
