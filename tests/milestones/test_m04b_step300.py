"""M4b (session S2.3) — the first real training run, verified.

Certifies: the 300-step run of the reader's own model produced a checkpoint that
reloads into a fresh model without key errors, and a loss curve showing real learning
(final < 60% of initial) — the first durable artifact of the reader's LLM.
"""

from conftest import load_checkpoint, load_loss_curve, rebuild_model, require


def test_checkpoint_reloads(ws):
    ckpt = load_checkpoint(require(ws / "step300.pt", "S2.2", "First checkpoint"))
    model, config = rebuild_model(ckpt)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_loss_curve_shows_learning(ws):
    losses = load_loss_curve(ws / "loss_step300.json", "S2.3")
    assert losses[-1] < 0.6 * losses[0], (
        f"Final loss {losses[-1]:.4f} is not below 60% of initial {losses[0]:.4f}. "
        "300 steps should clearly move a 2M-param model — check LR and batch size."
    )
