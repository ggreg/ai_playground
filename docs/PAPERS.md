# Key Papers

Papers referenced throughout this codebase, organized by topic.

## Transformer Architecture

- **Attention Is All You Need** (Vaswani et al., 2017) — The original transformer
- **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023) — Architecture we implement: pre-norm, RoPE, SwiGLU, GQA
- **LLaMA 2** (Touvron et al., 2023) — Introduces Grouped-Query Attention

## Attention Variants

- **Multi-Query Attention** (Shazeer, 2019) — Single KV head shared across all Q heads → `attention.py`
- **GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023) — Groups of Q heads share KV → `attention.py`
- **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022) — IO-aware attention algorithm
- **FlashAttention-2** (Dao, 2023) — Better parallelism and work partitioning

## Positional Encodings

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021) — RoPE → `layers.py`
- **ALiBi: Train Short, Test Long** (Press et al., 2022) — Attention with Linear Biases

## Normalization & Activations

- **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019) — RMSNorm → `layers.py`
- **GLU Variants Improve Transformer** (Shazeer, 2020) — SwiGLU → `layers.py`

## Training Optimization

- **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019) — AdamW → `optimizers.py`
- **Mixed Precision Training** (Micikevicius et al., 2018) — FP16 training with loss scaling
- **Scaling Language Models** (Hoffmann et al., 2022) — Chinchilla scaling laws, FLOP counting

## Inference Optimization

- **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019) — MQA for fast inference
- **Efficient Memory Management for Large Language Model Serving** (Kwon et al., 2023) — PagedAttention (vLLM)
- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023)

## Distributed Training

- **Megatron-LM** (Shoeybi et al., 2019) — Tensor and pipeline parallelism
- **ZeRO: Memory Optimizations** (Rajbhandari et al., 2020) — Partitioned optimizer states/gradients/parameters
- **PyTorch FSDP** (Zhao et al., 2023) — Fully Sharded Data Parallel

## Quantization

- **GPTQ: Accurate Post-Training Quantization** (Frantar et al., 2023)
- **AWQ: Activation-aware Weight Quantization** (Lin et al., 2024)
- **LLM.int8(): 8-bit Matrix Multiplication** (Dettmers et al., 2022)
