# Learning Path

A suggested order for working through the AI playground, from LLM internals to GPU optimization.

## Phase 1: Transformer Internals

Understand every component of a modern LLM before optimizing anything.

1. **Attention Mechanisms** — `notebooks/01_transformer_internals/01_attention_mechanisms.ipynb`
   - Multi-Head, Grouped-Query, Multi-Query attention
   - KV cache memory implications
   - Code: `src/ai_playground/models/attention.py`

2. **Positional Encodings** — `notebooks/01_transformer_internals/02_positional_encodings.ipynb`
   - Sinusoidal, RoPE, ALiBi
   - Why RoPE won (relative positions, length extrapolation)
   - Code: `src/ai_playground/models/layers.py` (`precompute_rope_frequencies`, `apply_rope`)

3. **Normalization & Activations** — `notebooks/01_transformer_internals/03_activation_functions.ipynb`, `04_normalization.ipynb`
   - LayerNorm vs RMSNorm (why skip the mean?)
   - GELU vs SwiGLU (gated activations)
   - Code: `src/ai_playground/models/layers.py` (`RMSNorm`, `SwiGLU`)

4. **Tokenizers** — `notebooks/01_transformer_internals/05_tokenizers_deep_dive.ipynb`
   - BPE algorithm internals
   - tiktoken vs SentencePiece

**Milestone**: You can read any LLM architecture paper and understand every component.

## Phase 2: Training Optimization

Make training fast and stable on a single GPU.

5. **Mixed Precision** — `notebooks/02_training_optimization/01_mixed_precision.ipynb`
   - FP32 vs FP16 vs BF16 vs FP8
   - Loss scaling and why BF16 doesn't need it
   - Code: `src/ai_playground/training/trainer.py` (AMP setup)

6. **Gradient Accumulation** — `notebooks/02_training_optimization/02_gradient_accumulation.ipynb`
   - Simulating large batches on small GPUs
   - Code: `src/ai_playground/training/trainer.py` (accumulation loop)

7. **Optimizers** — `notebooks/02_training_optimization/04_optimizer_internals.ipynb`
   - AdamW from scratch (every step explained)
   - Code: `src/ai_playground/training/optimizers.py`

8. **Learning Rate Schedules** — `notebooks/02_training_optimization/03_lr_schedules.ipynb`
   - Cosine decay with warmup
   - WSD (Warmup-Stable-Decay)
   - Code: `src/ai_playground/training/trainer.py` (`get_lr`)

**Milestone**: You can train a model efficiently on a single GPU and understand every training hyperparameter.

## Phase 3: Inference Optimization

Make inference fast and memory-efficient.

9. **KV Cache** — `notebooks/04_inference_optimization/01_kv_cache.ipynb`
   - Why generation without cache is O(n^2)
   - Prefill vs decode phases
   - Code: `src/ai_playground/inference/generate.py`, `models/attention.py` (cache logic)

10. **Quantization** — `notebooks/04_inference_optimization/02_quantization.ipynb`
    - INT8 absmax, GPTQ, AWQ, GGUF
    - Code: `src/ai_playground/inference/quantize.py`

11. **Flash Attention** — `notebooks/04_inference_optimization/05_flash_attention.ipynb`
    - IO-aware attention (memory hierarchy)
    - Code: `models/attention.py` (SDPA path)

12. **Speculative Decoding** — `notebooks/04_inference_optimization/03_speculative_decoding.ipynb`
    - Draft + verify for faster generation

**Milestone**: You understand every optimization used in vLLM, TGI, and other serving frameworks.

## Phase 4: Distributed Training

Scale to multiple GPUs and nodes.

13. **Data Parallelism** — `notebooks/03_distributed_training/01_data_parallel.ipynb`
    - DDP: replicate model, split data, all-reduce gradients
    - FSDP: shard everything (ZeRO Stage 3)
    - Code: `src/ai_playground/training/distributed.py`

14. **Tensor Parallelism** — `notebooks/03_distributed_training/02_tensor_parallel.ipynb`
    - Megatron-style column/row parallel Linear layers

15. **Pipeline Parallelism** — `notebooks/03_distributed_training/03_pipeline_parallel.ipynb`
    - GPipe, 1F1B schedules, micro-batching

**Milestone**: You can design a parallelism strategy for training any model on any cluster.

## Phase 5: GPU & NVIDIA Tools

Squeeze maximum performance from hardware.

16. **CUDA Basics** — `notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb`
    - Memory hierarchy, warps, occupancy
    - Writing kernels with Numba/CuPy

17. **Triton Kernels** — `notebooks/05_gpu_nvidia_tools/02_triton_kernels.ipynb`
    - Fused operations (why fusing matters for memory bandwidth)
    - Writing custom kernels in Triton

18. **Profiling** — `notebooks/05_gpu_nvidia_tools/03_nsight_profiling.ipynb`
    - Nsight Systems (timeline), Nsight Compute (kernel-level)
    - Code: `src/ai_playground/profiling/nsight.py`

19. **MFU** — Use `profiling/flops.py` to measure and optimize Model FLOP Utilization
    - Target: 40-60% on modern GPUs

**Milestone**: You can profile, diagnose, and fix GPU performance bottlenecks.
