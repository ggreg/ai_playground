"""M5 (session S2.4) — the schedule sweep, and a winner adopted.

Certifies: the reader ran constant / cosine / warmup+cosine from the same seed on
their own model, recorded all three outcomes, and committed to one — a decision made
from measurements, not folklore.
"""

from conftest import metric

SCHEDULES = ("constant", "cosine", "warmup_cosine")


def test_sweep_recorded_and_adopted(metrics, reader_config_raw):
    sweep = metric(metrics, "m5_schedule_sweep", "S2.4")
    for name in SCHEDULES:
        assert name in sweep and isinstance(sweep[name], (int, float)), (
            f"m5_schedule_sweep needs a final loss for '{name}'."
        )
    chosen = sweep.get("chosen")
    assert chosen in SCHEDULES, f"m5_schedule_sweep['chosen'] must be one of {SCHEDULES}."
    assert reader_config_raw.get("schedule") == chosen, (
        f"config.json must record the adopted schedule (\"schedule\": \"{chosen}\") — "
        "the release run at M7 reads it from there."
    )
