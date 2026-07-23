"""p0 — your scalar autograd engine. See README.md for the brief and the import rules
(short version: no autograd library and no peeking at the repo's engine). Everything
below the SCAFFOLDING line is yours to replace.
"""

from ai_playground.fundamentals.datasets import make_moons

# ---------------------------------------------------------------- SCAFFOLDING (given) --
# 100 points, 2 features in [-1.5, 1.5]-ish, labels 0/1. Deterministic.
MOONS_X, MOONS_Y = make_moons(n=100, noise=0.1, seed=0)
MOONS = [([float(x1), float(x2)], int(y)) for (x1, x2), y in zip(MOONS_X, MOONS_Y)]

# ------------------------------------------------------------------- YOUR CODE (build) --


class Value:
    """A scalar that remembers how it was computed, so backward() can apply the chain rule.

    Required surface (the tests use exactly this): .data, .grad, +, *, -, ** (int),
    tanh(), backward(). Design everything else yourself.
    """

    def __init__(self, data: float):
        raise NotImplementedError


def train_moons() -> float:
    """Train an MLP built from your Values on MOONS; return final train accuracy (0..1).

    Architecture, loss, learning rate, and step count are your design decisions.
    Budget hint: a 2-8-1 tanh net and a few hundred full-batch steps separate the moons;
    keep it small enough to finish in about a minute of pure Python.
    """
    raise NotImplementedError


if __name__ == "__main__":
    acc = train_moons()
    print(f"moons train accuracy: {acc:.3f} (target >= 0.90)")
