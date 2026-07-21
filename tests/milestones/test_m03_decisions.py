"""M3 (session S1.4) — dense or MoE, decided with numbers.

Certifies: the reader computed total vs ACTIVE parameters for a 4-expert top-1 MoE
variant of their own config and wrote the decision down with the serving cost spelled
out — the total/active distinction is the one MoE idea the finale's memory math needs.
"""

import re

from conftest import decisions_entry


def test_moe_decision_recorded(ws):
    entry = decisions_entry(ws, "M3", "S1.4")
    lower = entry.lower()
    assert "total" in lower and "active" in lower, (
        "The M3 entry must state both TOTAL and ACTIVE parameter counts — that "
        "distinction is the whole point of the milestone."
    )
    numbers = re.findall(r"\d[\d,_]*", entry)
    assert len(numbers) >= 2, "The M3 entry needs the two parameter counts, as numbers."
    assert len(entry.split()) >= 40, (
        "The M3 entry needs a paragraph of reasoning (what would MoE cost at M13?), "
        "not just the numbers."
    )
