"""M11 (session S5.6) — predicted decode-step time on the virtual T4.

Certifies: the reader turned their model's decode-step DRAM traffic into a
tokens/sec prediction via the virtual GPU's trace — and wrote the assumptions down,
because a prediction without assumptions can't be falsified at M13d.
"""

from conftest import metric


def test_prediction_recorded(metrics):
    pred = metric(metrics, "m11_predicted", "S5.6")
    tps = pred.get("tokens_per_sec")
    assert isinstance(tps, (int, float)) and tps > 0, (
        "m11_predicted needs 'tokens_per_sec' (> 0)."
    )
    assert isinstance(pred.get("assumptions"), str) and pred["assumptions"].strip(), (
        "m11_predicted needs 'assumptions' — bytes moved per step, batch, context length."
    )
