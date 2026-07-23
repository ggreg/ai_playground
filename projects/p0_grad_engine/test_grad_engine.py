"""Acceptance tests for p0 — autograd from a blank file.

Gradients are checked against central finite differences (no autograd library as
referee), plus the classic shared-subexpression accumulation case, then the engine has
to actually train something.
"""

from pathlib import Path

import pytest

import grad_engine
from grad_engine import Value

H = 1e-5
TOL = 1e-4


def _finite_diff(f, at: list[float], i: int) -> float:
    lo, hi = list(at), list(at)
    lo[i] -= H
    hi[i] += H
    return (f(hi) - f(lo)) / (2 * H)


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "grad_engine.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    for banned in ("import torch", "from torch", "fundamentals.autograd", "fundamentals.nn"):
        assert banned not in code, f"grad_engine.py must not use '{banned}' — see README rules"


def test_shared_subexpression_accumulates(attempt):
    # y = x*x + x*z  =>  dy/dx = 2x + z, dy/dz = x. Only correct if grads accumulate.
    x = attempt(Value, 3.0)
    z = attempt(Value, -4.0)
    y = x * x + x * z
    attempt(y.backward)
    assert y.data == pytest.approx(3.0 * 3.0 + 3.0 * -4.0)
    assert x.grad == pytest.approx(2 * 3.0 + -4.0), "dy/dx wrong — is your grad '+=' or '='?"
    assert z.grad == pytest.approx(3.0)


def test_gradients_match_finite_differences(attempt):
    def expr(vals: list[float]) -> float:
        a, b, c = (attempt(Value, v) for v in vals)
        out = (a * b + c) ** 2 + (a - c).tanh() * b
        return out.data

    at = [0.7, -1.3, 0.4]
    a, b, c = (attempt(Value, v) for v in at)
    out = (a * b + c) ** 2 + (a - c).tanh() * b
    attempt(out.backward)
    for i, v in enumerate((a, b, c)):
        want = _finite_diff(expr, at, i)
        assert v.grad == pytest.approx(want, abs=TOL), (
            f"grad of input {i}: engine says {v.grad}, finite differences say {want}"
        )


def test_moons_training(attempt):
    acc = attempt(grad_engine.train_moons)
    assert isinstance(acc, float)
    assert acc >= 0.90, (
        f"moons train accuracy {acc:.3f} < 0.90 — the engine's gradients may be subtly "
        "wrong (loss decreasing but slowly is the classic symptom), or train longer."
    )
