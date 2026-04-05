"""Text generation with KV cache and various sampling strategies."""

import torch
import torch.nn.functional as F

from ..models.transformer import Transformer


def top_p_sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    1. Scale logits by temperature
    2. Sort probabilities descending
    3. Compute cumulative probability
    4. Zero out tokens beyond the top-p threshold
    5. Sample from the remaining distribution

    Lower temperature = more deterministic
    Lower top_p = fewer tokens considered = more focused
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    # Shift right so that the token that crosses the threshold is kept
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[sorted_mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Sample
    idx = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, idx)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Autoregressive generation with KV cache.

    The KV cache avoids recomputing attention for all previous tokens
    at each generation step. Without it, generation is O(n²) in sequence
    length; with it, each step is O(n) — a massive speedup.

    Args:
        model: Transformer model
        prompt_tokens: (1, prompt_len) token IDs
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold

    Returns:
        (1, prompt_len + generated_len) full sequence of token IDs
    """
    model.eval()
    model.reset_caches()

    device = next(model.parameters()).device
    tokens = prompt_tokens.to(device)

    # Prefill: process entire prompt at once (populates KV cache)
    logits = model(tokens, use_cache=True)

    # Decode: generate one token at a time
    for _ in range(max_new_tokens):
        # Only need logits for the last position
        next_logits = logits[:, -1, :]
        next_token = top_p_sample(next_logits, temperature, top_p)
        tokens = torch.cat([tokens, next_token], dim=1)

        # Feed only the new token (KV cache has everything else)
        logits = model(next_token, use_cache=True)

    model.reset_caches()
    return tokens
