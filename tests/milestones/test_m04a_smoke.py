"""M4a (session S2.2) — your config wired into the training loop, smoke-tested.

Certifies: the reader connected their own config.json to the chapter's training loop
and ran it — the recorded 20-step smoke curve must end lower than it starts, proving
data, loss, backward, and step are actually wired together.
"""

from conftest import load_loss_curve


def test_smoke_run_loss_decreases(ws):
    losses = load_loss_curve(ws / "loss_smoke.json", "S2.2")
    assert losses[-1] < losses[0], (
        f"Smoke-run loss went {losses[0]:.4f} -> {losses[-1]:.4f}; it must decrease. "
        "Revisit S0.3's failure-mode gallery: sign of the step, zero_grad, LR."
    )
