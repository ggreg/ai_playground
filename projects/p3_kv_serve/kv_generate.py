"""p3 — your inference engine over the given model. See README.md for the brief and the
rules (short version: the repo's inference package is off-limits; given_model.py is the
spec of the model you're serving).
"""

import torch

from given_model import GivenGPT


def generate_full(model: GivenGPT, prompt_ids: list[int], n_new: int) -> list[int]:
    """Greedy decoding by full recompute: each step runs model(prompt+generated so far)
    and takes the argmax of the last position. Returns prompt_ids + n_new new tokens."""
    raise NotImplementedError


def generate_cached(model: GivenGPT, prompt_ids: list[int], n_new: int) -> list[int]:
    """Greedy decoding through your own KV cache — token-for-token identical output to
    generate_full. Prefill the prompt once, then each decode step feeds ONE token and
    attends over cached K/V. The model's forward() has no cache hooks: rebuild the pass
    from its exposed pieces (blocks[i].wq/wk/wv/wo, ln1/ln2, w_up/w_down, embeddings,
    ln_f, head) following the math documented in given_model.py."""
    raise NotImplementedError


def top_p_sample(logits: torch.Tensor, p: float, generator: torch.Generator) -> int:
    """Nucleus sampling: from 1-D logits, keep the smallest set of tokens (in descending
    probability order) whose cumulative probability is >= p, renormalize over that set,
    and sample one token id with the given generator."""
    raise NotImplementedError
