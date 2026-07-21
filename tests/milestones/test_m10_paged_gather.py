"""M10 (session S5.4) — the paged gather, verified on the SIMT simulator.

Certifies: the reader wrote the finale's KV access pattern (reading token rows through
a block table) as a simulator kernel, verified it against numpy, and measured its
coalescing — so "paging is nearly free at block size 16" is something they measured,
not something they were told.
"""

from conftest import metric


def test_coalescing_measured(metrics):
    result = metric(metrics, "m10_coalescing", "S5.4")
    block_size = result.get("block_size")
    ratio = result.get("transactions_ratio")
    assert isinstance(block_size, int) and block_size > 0, (
        "m10_coalescing needs 'block_size' (tokens per block, int > 0)."
    )
    assert isinstance(ratio, (int, float)) and ratio >= 1.0, (
        "m10_coalescing needs 'transactions_ratio' (measured/ideal, >= 1.0)."
    )
    assert ratio <= 2.0, (
        f"transactions_ratio {ratio} is far from ideal — tokens are contiguous within "
        "a block, so a 16-token-block gather should stay near 1.0. Check the kernel's "
        "thread-to-element mapping."
    )
