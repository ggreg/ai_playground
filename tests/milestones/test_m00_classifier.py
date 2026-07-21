"""M0 (session S0.5) — train an MLP classifier with your own autograd engine.

Certifies: the reader's from-scratch reverse-mode gradients agree with PyTorch to 1e-6
on a real softmax + cross-entropy batch, and the training loop those gradients power
actually learns (>= 95% train accuracy on the 3-class spiral). Together these prove the
two halves of the refresher: correct derivatives, and a loop that uses them.
"""

from conftest import metric


def test_gradient_parity_with_torch(metrics):
    diff = metric(metrics, "m0_grad_max_abs_diff", "S0.5")
    assert diff < 1e-6, (
        f"Max abs gradient difference vs PyTorch is {diff:.3e} (need < 1e-6). "
        "The weight-copy recipe is in tests/test_fundamentals.py::TestMLP."
    )


def test_spiral_train_accuracy(metrics):
    acc = metric(metrics, "m0_spiral_train_acc", "S0.5")
    assert acc >= 0.95, f"Spiral train accuracy is {acc:.3f} (need >= 0.95)."
