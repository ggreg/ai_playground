"""M8 (session S2.8) — trust your optimizer.

Certifies: 20 identical training steps of the reader's model under AdamWFromScratch
and torch.optim.AdamW leave every parameter tensor within 1e-5 — the reader now
understands, update by update, what the release run's optimizer did.
"""

from conftest import metric


def test_from_scratch_adamw_matches_torch(metrics):
    worst = metric(metrics, "m8_optimizer_max_diff", "S2.8")
    assert worst < 1e-5, (
        f"Worst per-parameter deviation is {worst:.3e} (need < 1e-5). Usual suspects: "
        "bias correction, decoupled weight decay applied to the wrong term, or eps placement."
    )
