"""A multi-layer perceptron built on the scalar autograd engine.

Deliberately tiny and readable: a Neuron is a dot product plus bias through a
nonlinearity, a Layer is a list of Neurons, an MLP is a list of Layers. A [2, 8, 1] MLP
has 33 parameters and trains on a few hundred points in seconds — the point is to *see*
the machinery, not to be fast. The repo's transformer FFN
(``ai_playground.models.layers.SwiGLU``) is this same two-Linear-layer shape, vectorized,
with a multiplicative gate.

Why biases here when the rest of the repo uses ``bias=False``? The LLM convention works
because normalization layers (RMSNorm) absorb the shift and at ``dim=4096`` the bias
vector is noise. This MLP has no normalization and 2-dimensional inputs: without biases
every neuron's decision boundary must pass through the origin, and the toy datasets in
``datasets.py`` become unlearnable. Same math, different regime — see CLAUDE.md's
convention note.

Weight init scales by 1/sqrt(n_in) so pre-activations start with ~unit variance
regardless of fan-in — the idea behind Glorot & Bengio (2010),
"Understanding the difficulty of training deep feedforward neural networks"
(https://proceedings.mlr.press/v9/glorot10a.html).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ai_playground.fundamentals.autograd import Value

_ACTIVATIONS = ("tanh", "relu", "linear")


class Module:
    """Minimal parameter container — the shape of ``torch.nn.Module`` without the rest."""

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        """Reset accumulated gradients before the next backward pass.

        Gradients accumulate by design (see ``Value``); forgetting this call mixes the
        previous step's gradients into the current one — the classic training-loop bug.
        """
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """w · x + b through a nonlinearity: one unit, (n_in + 1) parameters."""

    def __init__(self, n_in: int, activation: str = "tanh",
                 rng: random.Random | None = None) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        rng = rng if rng is not None else random.Random(0)
        scale = 1.0 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-1.0, 1.0) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: Sequence[Value | float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        return act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    """n_out independent Neurons over the same input — one row of a weight matrix each.

    Calling a Layer is exactly the matmul ``x @ W + b`` from the notebooks, computed as
    n_out explicit dot products.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "tanh",
                 rng: random.Random | None = None) -> None:
        rng = rng if rng is not None else random.Random(0)
        self.neurons = [Neuron(n_in, activation=activation, rng=rng) for _ in range(n_out)]

    def __call__(self, x: Sequence[Value | float]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A stack of Layers; the last layer is linear so it emits raw scores (logits).

    Keeping the output linear is the same convention as the transformer's output head:
    losses (MSE, cross-entropy) and samplers want unsquashed scores, and softmax/tanh are
    applied by the caller when needed.

    ``MLP(2, [8, 1])`` → 2 inputs, one hidden tanh layer of 8, one linear output;
    33 parameters ((2+1)*8 + (8+1)*1).
    """

    def __init__(self, n_in: int, layer_sizes: Sequence[int], activation: str = "tanh",
                 seed: int = 0) -> None:
        rng = random.Random(seed)
        sizes = [n_in, *layer_sizes]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=activation if i < len(layer_sizes) - 1 else "linear",
                rng=rng,
            )
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        out: Sequence[Value | float] = x
        for layer in self.layers:
            out = layer(out)
        return out[0] if len(out) == 1 else list(out)

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
