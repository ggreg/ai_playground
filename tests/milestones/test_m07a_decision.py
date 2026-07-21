"""M7a (session S2.6) — the effective-batch decision, with numbers.

Certifies: the reader measured the largest micro-batch their memory allows, derived
accum_steps from it, and wrote the decision down before burning compute on the release
run — the order (decide, then train) is the habit being certified.
"""

import re

from conftest import decisions_entry


def test_effective_batch_decision_recorded(ws):
    entry = decisions_entry(ws, "M7", "S2.6")
    assert "batch" in entry.lower(), "The M7 entry must state the micro/effective batch choice."
    numbers = re.findall(r"\d+", entry)
    assert len(numbers) >= 2, (
        "The M7 entry needs at least the micro-batch and accum_steps as numbers "
        "(effective batch 64 = micro_batch x accum_steps)."
    )
