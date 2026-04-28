# Concepts FAQ

Conceptual questions about LLM internals, with pointers to the specific code in this repo. This doc grows as questions come up — it's meant to complement [LEARNING_PATH.md](LEARNING_PATH.md) and [PAPERS.md](PAPERS.md).

## Table of Contents

- [How are token embeddings calculated?](#how-are-token-embeddings-calculated)
- [What's the loss function used for training?](#whats-the-loss-function-used-for-training)
- [Does cross-entropy training reduce bias between tokens?](#does-cross-entropy-training-reduce-bias-between-tokens)

---

## How are token embeddings calculated?

Token embeddings in this codebase are a **learned lookup table** — not a calculation in the traditional sense, but a parameter matrix trained alongside everything else.

### The Mechanism

In [src/ai_playground/models/transformer.py](../src/ai_playground/models/transformer.py):

```python
self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
```

This creates a matrix `E` of shape `(vocab_size, dim)` — for the tiny config, that's `(32000, 256)`. Each row is the embedding vector for one token ID.

The "calculation" is just an indexed lookup:

```python
h = self.tok_embeddings(tokens)   # (batch, seq_len) → (batch, seq_len, dim)
```

Internally, `nn.Embedding(V, D).forward(ids)` is equivalent to `E[ids]` — gather rows from `E` based on integer token IDs. There's no matmul, no nonlinearity. Just a memory lookup.

### How Embeddings Get Their Values

1. **Initialization** — random Gaussian, std=0.02 (see [transformer.py](../src/ai_playground/models/transformer.py) `_init_weights`):
   ```python
   nn.init.normal_(module.weight, mean=0.0, std=0.02)
   ```
   At step 0, the embedding for "cat" is just random noise.

2. **Training** — gradients flow back through every operation that touched a token's embedding. If token ID 42 was used in a sequence, the loss has a gradient w.r.t. row 42 of `E`. AdamW updates that row. Tokens that appear together in similar contexts end up with similar vectors — this is what gives embeddings their famous semantic structure.

3. **Math view** — equivalently, you can think of it as a one-hot matmul: `E @ one_hot(token_id)`. PyTorch implements it as a gather for efficiency, but the gradient is the same.

### Why It's Just a Lookup (Not a Function)

The model has to map a discrete vocabulary (32000 distinct integers) into continuous space. There's no smooth function from "token ID" to "vector" — token IDs are arbitrary labels. So the model learns a separate vector for each one. The embedding table is parameter-heavy: for the small config (`vocab_size=32000, dim=768`), it's `32000 × 768 ≈ 24.6M` parameters — about 20% of the total.

### Position Comes Later

Important detail: token embeddings have **no positional information**. "cat" gets the same vector whether it's at position 0 or position 100. Position is injected later inside attention, by rotating Q and K vectors via RoPE — see `precompute_rope_frequencies` in [layers.py](../src/ai_playground/models/layers.py).

### Optional Weight Tying

The output projection (`dim → vocab_size`) and the input embedding (`vocab_size → dim`) are transposes of each other in shape. Setting `tie_word_embeddings=True` in the config makes them share the same parameter matrix:

```python
if config.tie_word_embeddings:
    self.output.weight = self.tok_embeddings.weight
```

This halves the embedding parameter count and often improves quality — the model uses the same representation for "understanding" and "producing" tokens.

### Initialization in Modern LLMs

Random Gaussian with small std is dominant, but the exact std matters:

- **GPT-2 / nanoGPT / this repo**: `N(0, 0.02)`. The 0.02 number is essentially tradition — Radford picked it for GPT-2 and most projects copied it.
- **LLaMA / LLaMA 2/3**: also `N(0, 0.02)` for embeddings.
- **GPT-NeoX, BLOOM, PaLM**: scaled init like `N(0, sqrt(2 / (5*dim)))` — derived from a stability analysis to keep activations roughly unit-variance through the network.
- **Megatron-LM**: similar scaled init, with extra `1/sqrt(2 * n_layers)` scaling on residual-stream output projections to control variance buildup.

The general principle: **small, zero-mean, roughly Gaussian, with std chosen so initial activations don't blow up or vanish across layers.** The exact distribution isn't load-bearing — uniform with the same variance works similarly. What matters is the variance.

---

## What's the loss function used for training?

The loss function is **cross-entropy** — the standard for next-token prediction in language models. From [src/ai_playground/training/trainer.py](../src/ai_playground/training/trainer.py):

```python
logits = self.model(x)
loss = nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)), y.view(-1)
)
```

### What's Being Computed

Cross-entropy measures the distance between two probability distributions: the model's predicted distribution over the vocabulary, and the "true" distribution (a one-hot vector at the correct next token).

For a single token prediction:

```
L = -log P(y_true | x) = -log( exp(z_y_true) / Σ_v exp(z_v) )
```

Where `z` are the logits over the vocabulary. Equivalently:

```
L = -z_y_true + log Σ_v exp(z_v)
```

The training loss for a batch is the mean over all token positions.

### How the Tensors Flow

The data shapes tell the story:

- `x`: `(batch, seq_len)` — input token IDs
- `y`: `(batch, seq_len)` — target tokens (input shifted by 1, see [data.py](../src/ai_playground/training/data.py) `TextDataset`)
- `logits`: `(batch, seq_len, vocab_size)` — the model's score for every vocab token at every position

The reshape:

```python
logits.view(-1, logits.size(-1))   # (batch * seq_len, vocab_size)
y.view(-1)                          # (batch * seq_len,)
```

Treats every position in every sequence as an independent classification problem with `vocab_size` classes. PyTorch's `cross_entropy` then:

1. Applies log-softmax across the vocab dimension (numerically stable — never explicitly computes softmax then log)
2. Picks the log-probability at the true class index
3. Negates and averages

So for a tiny model with `vocab_size=32000` and a batch of `(8, 512)`, you're computing 4096 simultaneous 32000-way classifications and averaging their negative log-likelihoods.

### Why This Loss for Language Models

- **Probabilistic interpretation**: minimizing cross-entropy is equivalent to maximum likelihood estimation — you're directly maximizing `P(y | x)` under the model.
- **Perplexity** (the standard LLM eval metric) is just `exp(cross_entropy_loss)`. A loss of 2.3 → perplexity ~10 → the model is "as confused as if choosing uniformly among 10 words" at each position.
- **Random baseline**: with `vocab_size=32000`, untrained loss is ≈ `log(32000) ≈ 10.4`. You'll see this value in the first few training steps before it drops.

### Two Details Worth Noting

1. **Loss scaling for grad accumulation** — when accumulating gradients over N micro-batches, dividing by N keeps the effective gradient magnitude equivalent to a single batch of size `batch_size * grad_accum_steps`:
   ```python
   loss = loss / cfg.grad_accum_steps
   ```

2. **Mixed precision interaction** — the forward pass and loss are computed under `torch.amp.autocast` in BF16/FP16, but the loss reduction itself is done in FP32 internally by `cross_entropy` for numerical stability. The log-sum-exp trick can underflow in FP16 otherwise.

### What's Not Used (and Why)

- **No label smoothing** — common in machine translation, rarely used in LLM pretraining since it slightly hurts perplexity.
- **No padding mask** — this codebase uses fixed-length packed sequences from `TextDataset`, so every position is a real prediction. Real production pretraining usually uses an `ignore_index` for padding tokens.
- **No auxiliary losses** — pure next-token prediction. MoE models would add a load-balancing loss; some methods add z-loss (regularizing `log Z`); this implementation keeps it minimal.

### Cross-Entropy Universality

This loss is genuinely close to universal across modern LLMs:

- Every GPT family (1, 2, 3, 4), LLaMA, Mistral, Gemma, Qwen, DeepSeek, Claude, etc. uses cross-entropy for pretraining.
- It's just maximum likelihood estimation — you can't really beat it as the primary objective when you want a model that assigns probabilities to text.

What varies in practice is what's added on top:

- **Z-loss** (`λ * (log Z)²`) — used in PaLM, LLaMA 2 — regularizes the partition function for stability.
- **MoE load-balancing loss** — only for mixture-of-experts models, encourages even expert usage.

---

## Does cross-entropy training reduce bias between tokens?

**No — cross-entropy actively rewards bias between tokens.** Reducing bias is not its job, and in some ways the opposite is true.

The word "bias" is overloaded, so let's separate three distinct things:

### 1. Frequency Bias: Cross-Entropy Reinforces It

Cross-entropy is **maximum likelihood estimation**. Its only job is to make the model's predicted distribution match the empirical token distribution in the data. If "the" appears 100× more often than "antidisestablishmentarianism" in context, the loss is *minimized* when the model assigns roughly that frequency ratio to those tokens.

So the loss directly encodes frequency. It doesn't reduce frequency bias — it *creates* it, on purpose. That's the entire point of language modeling: learn the actual distribution.

You can see this in cross-entropy's gradient w.r.t. logits:

```
∂L/∂z_v = p_v - 1[v = y_true]
```

Where `p_v` is the predicted probability for token `v`. Common tokens get the "is it the true token" signal often. Rare tokens almost never. The whole training process is biased toward learning common patterns first.

### 2. Representational Bias: Cross-Entropy Creates Geometric Imbalance

Cross-entropy with softmax produces measurable geometric biases in embeddings:

- **Frequent tokens get larger embedding norms.** They're updated more often, in more directions, and tend to grow. Rare tokens stay closer to their initialization.
- **Embeddings cluster anisotropically.** Trained token embeddings concentrate in a narrow cone of the embedding space rather than filling it uniformly. This is sometimes called the **"representation degeneration problem"** (Gao et al., 2019, [arxiv.org/abs/1907.12009](https://arxiv.org/abs/1907.12009)).
- **Why anisotropy happens**: the softmax cross-entropy gradient pushes the true token's embedding in the direction of the hidden state and pulls all other tokens away. The "pull away" force is partial information (signal averaged over many losses), while the "push toward" force is concentrated. The cumulative effect is geometric collapse.

Mitigations have been explored but aren't standard:

- **Cosine similarity-based softmax** (replace dot product with cosine).
- **Contrastive auxiliary losses** (e.g., SimCSE) to spread embeddings out.
- **Tokenizer choices** (BPE with merges balanced for frequency) — softens the worst rare-token cases.

These are not part of standard LLM pretraining. The default is: live with the anisotropy.

### 3. Social/Dataset Bias: Cross-Entropy Faithfully Reproduces It

If the training corpus says "doctor → he" more often than "doctor → she", cross-entropy will train the model to predict "he" with that empirical probability. The loss has no notion of fairness, no preference for balanced outcomes, and no awareness of demographic categories.

In fact, cross-entropy is *too good* at picking up these patterns — it minimizes loss by faithfully encoding co-occurrence statistics, including the ones we'd rather it didn't. This is why every production LLM relies on **post-pretraining steps** to address this:

- **Data curation** — filter or rebalance the training corpus.
- **Instruction tuning / SFT** — fine-tune on examples that demonstrate desired behavior.
- **RLHF / RLAIF / DPO** — preference optimization explicitly steers the model away from harmful or biased outputs.
- **Constitutional AI** — train against principles rather than raw human preferences.

None of these change the pretraining loss. They all assume cross-entropy on raw web text will produce a biased model and add additional training stages to correct it.

### The Underlying Reason

Cross-entropy is a **distribution-matching** loss. It cannot reduce a bias that exists in the data, because the data is its ground truth. By construction:

```
argmin_θ E_{x ~ D}[ -log p_θ(x) ]  =  p_data
```

A loss whose optimum is the data distribution cannot, on its own, debias that distribution.

### What *Could* Reduce Bias

If you wanted to reduce bias during pretraining (rather than fix it afterward), you'd need to change either the data or the loss:

- **Data side**: oversample underrepresented tokens/contexts; rebalance demographic representations; curate training corpora.
- **Loss side**: add regularization terms — fairness constraints, contrastive losses on protected attributes, KL penalties to a "neutral" reference distribution. None of these are mainstream.
- **Architecture side**: weight tying ([transformer.py](../src/ai_playground/models/transformer.py)) modestly helps the rare-token representational bias because the dense softmax gradients flow back into the embedding table for *every* vocab item on every step, not just the few sampled in the batch.

### TL;DR

| Type of bias | Does cross-entropy reduce it? |
|---|---|
| Frequency bias | No — it learns it (that's the whole point) |
| Representational/geometric bias in embeddings | No — softmax cross-entropy actively *creates* anisotropy |
| Social/dataset bias | No — it faithfully encodes whatever's in the data |

Cross-entropy is a measurement-of-fit loss, not a fairness loss. Bias reduction in modern LLMs happens through **data choices, post-training (RLHF/DPO), and inference-time interventions** — never through the pretraining objective itself.
