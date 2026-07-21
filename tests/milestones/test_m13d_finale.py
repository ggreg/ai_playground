"""M13d (session S4.6) — the finale: your model's text through your engine.

Certifies: the reader decoded real text from their own trained weights served through
their own paged engine, measured the rate against M11's prediction, and the metrics
paper trail runs unbroken from M0's gradient parity to the finale.
"""

from conftest import metric

JOURNEY_KEYS = (
    "m0_grad_max_abs_diff",  # the refresher: gradients proven correct
    "m5_schedule_sweep",     # training: recipe chosen from measurements
    "m9_roofline",           # performance: decode located on the roofline
    "m11_predicted",         # prediction the finale's measurement answers
    "m12_gemms",             # the GEMMs the engine actually runs
)


def test_finale_text(metrics):
    finale = metric(metrics, "m13_finale", "S4.6")
    text = finale.get("text")
    assert isinstance(text, str) and text.strip(), (
        "m13_finale needs 'text' — the decoded output of the first request. It will "
        "read like a 2M-parameter model wrote it; that's the point."
    )
    tps = finale.get("tokens_per_sec")
    assert isinstance(tps, (int, float)) and tps > 0, (
        "m13_finale needs 'tokens_per_sec' — the measured rate to hold against "
        "M11's prediction."
    )


def test_journey_is_complete(metrics):
    metric(metrics, "m13_finale", "S4.6")  # not at the finale yet -> skip, not fail
    missing = [k for k in JOURNEY_KEYS if k not in metrics]
    assert not missing, (
        f"metrics.json is missing the paper trail for: {missing}. metrics.json "
        "accumulates — never overwrite earlier keys; rerun the sessions that "
        "produced them if they were lost."
    )
