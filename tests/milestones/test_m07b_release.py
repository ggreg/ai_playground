"""M7b (session S2.7) — the release checkpoint.

Certifies: the 1,000-step release run (adopted schedule + precision, effective batch
64) produced the checkpoint the finale will serve — it reloads cleanly and its final
loss beats the 300-step checkpoint's, so the extra compute demonstrably bought a
better model.
"""

from conftest import load_checkpoint, load_loss_curve, rebuild_model, require


def test_release_checkpoint_reloads(ws):
    ckpt = load_checkpoint(require(ws / "final.pt", "S2.7", "Release checkpoint"))
    rebuild_model(ckpt)


def test_release_beats_step300(ws):
    final_losses = load_loss_curve(ws / "loss_final.json", "S2.7")
    step300_losses = load_loss_curve(ws / "loss_step300.json", "S2.3")
    assert final_losses[-1] < step300_losses[-1], (
        f"Release final loss {final_losses[-1]:.4f} does not beat step300's "
        f"{step300_losses[-1]:.4f} — 700 more steps with a tuned recipe should win."
    )
