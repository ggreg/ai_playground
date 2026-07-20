"""Tests for the fundamentals package (scalar autograd engine, MLP, toy datasets).

The gradient tests all follow one pattern: build the identical expression with `Value`
and with `torch.autograd`, backprop both, and require the gradients to match. That makes
PyTorch the oracle — exactly the claim the refresher chapters make ("this engine is what
`loss.backward()` does, vectorized").
"""

import numpy as np
import pytest
import torch

from ai_playground.fundamentals import MLP, Value, make_moons, make_spiral


def torch_grad(fn, *inputs: float) -> tuple[float, list[float]]:
    """Evaluate fn on scalar tensors with requires_grad, return (output, grads)."""
    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    out = fn(*ts)
    out.backward()
    return out.item(), [t.grad.item() for t in ts]


class TestValueArithmetic:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            (lambda: Value(2.0) + Value(3.0), 5.0),
            (lambda: Value(2.0) * Value(3.0), 6.0),
            (lambda: Value(2.0) - Value(3.0), -1.0),
            (lambda: Value(3.0) / Value(2.0), 1.5),
            (lambda: Value(2.0) ** 3, 8.0),
            (lambda: -Value(2.0), -2.0),
            (lambda: 1.0 + Value(2.0), 3.0),
            (lambda: 2.0 * Value(3.0), 6.0),
            (lambda: 1.0 - Value(2.0), -1.0),
            (lambda: 3.0 / Value(2.0), 1.5),
        ],
    )
    def test_forward_values(self, expr, expected):
        assert expr().data == pytest.approx(expected)

    def test_pow_rejects_value_exponent(self):
        with pytest.raises(TypeError):
            Value(2.0) ** Value(3.0)


class TestValueGradients:
    @pytest.mark.parametrize(
        "ours,theirs,inputs",
        [
            (lambda a, b: a + b, lambda a, b: a + b, (2.0, 3.0)),
            (lambda a, b: a * b, lambda a, b: a * b, (2.0, 3.0)),
            (lambda a, b: a / b, lambda a, b: a / b, (2.0, 3.0)),
            (lambda a: a**3, lambda a: a**3, (2.0,)),
            (lambda a: a.tanh(), lambda a: torch.tanh(a), (0.7,)),
            (lambda a: a.relu(), lambda a: torch.relu(a), (0.7,)),
            (lambda a: a.relu(), lambda a: torch.relu(a), (-0.7,)),
            (lambda a: a.exp(), lambda a: torch.exp(a), (0.5,)),
            (lambda a: a.log(), lambda a: torch.log(a), (2.5,)),
        ],
    )
    def test_single_op_matches_torch(self, ours, theirs, inputs):
        vals = [Value(x) for x in inputs]
        out = ours(*vals)
        out.backward()
        expected_out, expected_grads = torch_grad(theirs, *inputs)
        assert out.data == pytest.approx(expected_out, abs=1e-8)
        for v, g in zip(vals, expected_grads):
            assert v.grad == pytest.approx(g, abs=1e-8)

    def test_composite_expression_matches_torch(self):
        # The classic micrograd sanity expression: shared inputs, mixed ops.
        def f(a, b):
            c = a * b + b**3
            d = c + c * a + (1.0 - b)
            return d + d * 2.0 + (b + a).tanh()

        a, b = Value(-2.0), Value(3.0)
        out = f(a, b)
        out.backward()
        def ft(a, b):
            c = a * b + b**3
            d = c + c * a + (1.0 - b)
            return d + d * 2.0 + torch.tanh(b + a)

        expected_out, (ga, gb) = torch_grad(ft, -2.0, 3.0)
        assert out.data == pytest.approx(expected_out, abs=1e-8)
        assert a.grad == pytest.approx(ga, abs=1e-8)
        assert b.grad == pytest.approx(gb, abs=1e-8)

    def test_diamond_graph_accumulates(self):
        # b = a + a: both paths must deposit gradient (grad +=, not =).
        a = Value(3.0)
        b = a + a
        b.backward()
        assert a.grad == pytest.approx(2.0)

    def test_reused_subgraph_matches_torch(self):
        def f(a):
            s = a.tanh() if isinstance(a, Value) else torch.tanh(a)
            return s * s + s

        a = Value(0.4)
        out = f(a)
        out.backward()
        _, (ga,) = torch_grad(f, 0.4)
        assert a.grad == pytest.approx(ga, abs=1e-8)

    def test_finite_difference_gradcheck(self):
        def f(a, b, c):
            return ((a * b + c).tanh() * a + (b * 0.5).exp()) / c

        xs = (0.6, -0.8, 1.7)
        vals = [Value(x) for x in xs]
        f(*vals).backward()
        eps = 1e-6
        for i in range(3):
            hi = [x + eps if j == i else x for j, x in enumerate(xs)]
            lo = [x - eps if j == i else x for j, x in enumerate(xs)]
            numeric = (f(*[Value(x) for x in hi]).data - f(*[Value(x) for x in lo]).data) / (
                2 * eps
            )
            assert vals[i].grad == pytest.approx(numeric, abs=1e-5)


class TestMLP:
    @pytest.mark.parametrize(
        "n_in,sizes",
        [(2, [8, 1]), (2, [4, 4, 3]), (3, [5, 2])],
    )
    def test_parameter_count(self, n_in, sizes):
        mlp = MLP(n_in, sizes)
        dims = [n_in, *sizes]
        expected = sum((dims[i] + 1) * dims[i + 1] for i in range(len(sizes)))
        assert len(mlp.parameters()) == expected

    def test_output_arity(self):
        x = [0.5, -0.5]
        assert isinstance(MLP(2, [4, 1])(x), Value)
        out = MLP(2, [4, 3])(x)
        assert isinstance(out, list) and len(out) == 3

    def test_zero_grad(self):
        mlp = MLP(2, [4, 1])
        mlp([0.5, -0.5]).backward()
        assert any(p.grad != 0.0 for p in mlp.parameters())
        mlp.zero_grad()
        assert all(p.grad == 0.0 for p in mlp.parameters())

    def test_seed_determinism(self):
        p1 = [p.data for p in MLP(2, [4, 1], seed=7).parameters()]
        p2 = [p.data for p in MLP(2, [4, 1], seed=7).parameters()]
        p3 = [p.data for p in MLP(2, [4, 1], seed=8).parameters()]
        assert p1 == p2
        assert p1 != p3

    def test_grad_parity_with_torch(self):
        """Copy an MLP's weights into an equivalent torch model; grads must match."""
        mlp = MLP(2, [4, 1], seed=3)
        tmodel = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
        ).double()
        with torch.no_grad():
            for layer, tlin in zip(mlp.layers, [tmodel[0], tmodel[2]]):
                tlin.weight.copy_(
                    torch.tensor(
                        [[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64
                    )
                )
                tlin.bias.copy_(
                    torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64)
                )

        X, y = make_moons(n=16, seed=0)
        targets = 2.0 * y - 1.0  # {-1, +1} for a tanh-friendly MSE

        loss = sum(
            (mlp([*xi]) - float(ti)) ** 2 for xi, ti in zip(X, targets)
        ) / len(X)
        loss.backward()

        tX = torch.tensor(X, dtype=torch.float64)
        tt = torch.tensor(targets, dtype=torch.float64)
        tloss = ((tmodel(tX).squeeze(1) - tt) ** 2).mean()
        tloss.backward()

        assert loss.data == pytest.approx(tloss.item(), abs=1e-10)
        ours = np.array([p.grad for p in mlp.parameters()])
        theirs = np.concatenate(
            [
                torch.cat([tmodel[0].weight.grad, tmodel[0].bias.grad.unsqueeze(1)], dim=1)
                .flatten()
                .numpy(),
                torch.cat([tmodel[2].weight.grad, tmodel[2].bias.grad.unsqueeze(1)], dim=1)
                .flatten()
                .numpy(),
            ]
        )
        assert np.abs(ours - theirs).max() < 1e-6


class TestDatasets:
    def test_moons_shapes_and_labels(self):
        X, y = make_moons(n=101, seed=1)
        assert X.shape == (101, 2) and X.dtype == np.float64
        assert y.shape == (101,) and set(np.unique(y)) == {0, 1}

    def test_spiral_shapes_and_balance(self):
        X, y = make_spiral(n_per_class=50, n_classes=3, seed=1)
        assert X.shape == (150, 2)
        assert set(np.unique(y)) == {0, 1, 2}
        assert all((y == c).sum() == 50 for c in range(3))

    def test_seed_reproducibility(self):
        Xa, ya = make_moons(n=40, seed=5)
        Xb, yb = make_moons(n=40, seed=5)
        Xc, _ = make_moons(n=40, seed=6)
        assert np.array_equal(Xa, Xb) and np.array_equal(ya, yb)
        assert not np.array_equal(Xa, Xc)


class TestTrainingSmoke:
    def test_loss_decreases(self):
        """30 full-batch GD steps on 40 moons points must reduce the loss."""
        X, y = make_moons(n=40, noise=0.05, seed=2)
        targets = 2.0 * y - 1.0
        mlp = MLP(2, [4, 1], seed=1)

        def batch_loss() -> Value:
            return sum((mlp([*xi]).tanh() - float(ti)) ** 2 for xi, ti in zip(X, targets)) / len(X)

        initial = batch_loss().data
        for _ in range(30):
            mlp.zero_grad()
            loss = batch_loss()
            loss.backward()
            for p in mlp.parameters():
                p.data -= 0.5 * p.grad
        assert batch_loss().data < initial
