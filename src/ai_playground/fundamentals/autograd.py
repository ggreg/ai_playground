"""Scalar reverse-mode automatic differentiation — a minimal autograd engine.

This is the smallest useful answer to "what does ``loss.backward()`` actually do?".
Every value is a single float, so every chain-rule application is one visible line of
code — no broadcasting, no batching, no tensor layout to obscure the math. PyTorch's
autograd is this exact algorithm (reverse-mode differentiation over a dynamically
recorded computation graph), vectorized over tensors.

Reverse mode matters for deep learning because one backward sweep yields the gradient of
a *scalar* loss with respect to *all* N parameters at roughly the cost of one forward
pass — for a 7B-parameter model that's one sweep, not 7 billion forward differences.

The design follows Karpathy's micrograd (https://github.com/karpathy/micrograd); the
algorithm is backpropagation as popularized by Rumelhart, Hinton & Williams (1986),
"Learning representations by back-propagating errors" (https://doi.org/10.1038/323533a0).
See notebooks/00_dnn_refresher/01_backprop_micrograd.ipynb for the cell-by-cell build.
"""

from __future__ import annotations

import math


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate.

    Each arithmetic operation returns a new ``Value`` holding (a) the result, (b) the
    operands (``_children``), and (c) a ``_backward`` closure that pushes the output's
    gradient onto the operands via the chain rule. ``backward()`` replays those closures
    in reverse topological order.

    Gradients *accumulate* (``+=``): if a value feeds into two places (``b = a + a``, or
    a weight reused across a batch), the total derivative is the sum of the path
    contributions. Assigning instead of adding is the classic from-scratch backprop bug.
    """

    def __init__(self, data: float, _children: tuple["Value", ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = _children
        self._op = _op

    # --- arithmetic -----------------------------------------------------------------

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1 and d(a+b)/db = 1: addition routes the gradient unchanged.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b: each operand's gradient is scaled by the *other* operand.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: int | float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only int/float exponents are supported (keeps the rule d/dx x^n)")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __rsub__(self, other: float) -> "Value":
        return Value(other) + (-self)

    def __rtruediv__(self, other: float) -> "Value":
        return Value(other) * self**-1

    # --- nonlinearities and transcendentals -----------------------------------------

    def tanh(self) -> "Value":
        """Hyperbolic tangent, the classic MLP activation (zero-centered, bounded).

        Local derivative 1 - tanh(x)^2 lives in (0, 1] and vanishes for |x| > ~3 —
        the "saturation" that motivated ReLU for deep nets.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        """max(0, x): gradient passes untouched where x > 0, is cut to zero elsewhere."""
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        """e^x — its own derivative, which is why it appears all over softmax gradients."""
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        """Natural log; with exp(), enough to build softmax + cross-entropy from scratch."""
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # --- backpropagation ------------------------------------------------------------

    def backward(self) -> None:
        """Backpropagate d(self)/d(node) into ``.grad`` of every node in the graph.

        Topologically sorts the graph so each node's ``_backward`` runs only after every
        path *from the output* to that node has deposited its gradient — the same
        ordering guarantee PyTorch's engine enforces with reference counts.
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build(v: "Value") -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0  # d(self)/d(self)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
