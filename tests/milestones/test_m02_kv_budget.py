"""M2 (session S1.3) — choose your attention shape against a KV-cache budget.

Certifies: the reader chose n_kv_heads deliberately (MHA/GQA/MQA) and the resulting
fp32 KV cache for batch 8 x 512 tokens fits under 8 MB — the serving-memory constraint
the mini-vLLM engine will page against at the finale — without breaking M1's budget.
"""

import pytest


def test_n_kv_heads_chosen(reader_config_raw):
    if reader_config_raw.get("n_kv_heads") is None:
        pytest.skip("n_kv_heads not chosen yet (null = MHA default) — run session S1.3.")


def test_kv_cache_fits_budget(reader_config_raw, reader_config):
    if reader_config_raw.get("n_kv_heads") is None:
        pytest.skip("n_kv_heads not chosen yet — run session S1.3.")
    # Per token: K and V, one vector per kv-head per layer, fp32 (4 bytes).
    bytes_per_token = 2 * reader_config.n_layers * reader_config.kv_heads * reader_config.head_dim * 4
    total = 8 * 512 * bytes_per_token
    assert total < 8 * 1024 * 1024, (
        f"KV cache at batch 8 x 512 tokens is {total / 1024 / 1024:.2f} MB (budget 8 MB). "
        f"With n_kv_heads={reader_config.kv_heads} that's {bytes_per_token} bytes/token — "
        "fewer kv-heads (toward MQA) shrinks it linearly."
    )


def test_still_under_parameter_budget(reader_config_raw, reader_config):
    if reader_config_raw.get("n_kv_heads") is None:
        pytest.skip("n_kv_heads not chosen yet — run session S1.3.")
    n = reader_config.num_params()
    assert n <= 2_000_000, f"n_kv_heads change pushed the config to {n:,} params (> 2M)."
