# Key Papers

Papers referenced throughout this codebase, organized by topic. Each entry includes a summary, links to the paper, official code, and documentation where available.

## Foundations

- **Learning representations by back-propagating errors** (Rumelhart, Hinton & Williams, 1986)
  The paper that made deep learning trainable: reverse-mode differentiation through a computation graph delivers the gradient of one scalar loss with respect to every weight at ~the cost of a forward pass. `src/ai_playground/fundamentals/autograd.py` implements it at scalar scale; `loss.backward()` is the same algorithm over tensors.
  [Paper](https://doi.org/10.1038/323533a0)

- **Approximation by Superpositions of a Sigmoidal Function** (Cybenko, 1989)
  Universal approximation: one hidden layer of sigmoidal units can approximate any continuous function on a compact set. Explains why capacity is never the question for MLPs — trainability and efficiency are (depth composes features more cheaply than width buys them).
  [Paper](https://doi.org/10.1007/BF02551274)

- **Understanding the difficulty of training deep feedforward neural networks** (Glorot & Bengio, 2010)
  Why initialization scale matters: keep activation and gradient variance roughly constant across layers, giving the 1/√n_in weight scaling used in `fundamentals/nn.py`. The reasoning previews every later normalization trick (LayerNorm, RMSNorm).
  [Paper](https://proceedings.mlr.press/v9/glorot10a.html)

## Tokenization

- **Neural Machine Translation of Rare Words with Subword Units** (Sennrich et al., 2016)
  Introduces Byte Pair Encoding (BPE) for NLP: start with characters, repeatedly merge the most frequent adjacent pair until you reach the desired vocabulary size. This determines `vocab_size` in the model config. Larger vocab = shorter sequences (faster attention) but bigger embedding table. Used by GPT-2, LLaMA (via sentencepiece), and most modern LLMs.
  [Paper](https://arxiv.org/abs/1508.07909)

## Transformer Architecture

- **Attention Is All You Need** (Vaswani et al., 2017)
  The original transformer architecture introducing multi-head self-attention, positional encodings, and the encoder-decoder structure that replaced RNNs for sequence modeling. Everything in this repo descends from this paper.
  [Paper](https://arxiv.org/abs/1706.03762) · [Code](https://github.com/tensorflow/tensor2tensor)

- **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
  The architecture we implement: decoder-only transformer with pre-norm (RMSNorm), RoPE positional embeddings, SwiGLU activations, and no bias terms. Showed that smaller models trained on more data can match larger models trained on less data.
  [Paper](https://arxiv.org/abs/2302.13971) · [Code](https://github.com/meta-llama/llama)

- **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023)
  Introduces Grouped-Query Attention (GQA) to the LLaMA architecture for faster inference. Also covers RLHF for chat fine-tuning. Our `n_kv_heads` parameter comes directly from this work.
  [Paper](https://arxiv.org/abs/2307.09288) · [Code](https://github.com/meta-llama/llama) · [Docs](https://llama.meta.com/)

## Mixture of Experts

- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (Shazeer et al., 2017)
  The original MoE paper for deep learning: replaces the FFN with N expert FFNs and a learned gating network that selects top-k experts per token. Showed that MoE can scale to 137B parameters with sublinear compute cost.
  [Paper](https://arxiv.org/abs/1701.06538)

- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (Fedus et al., 2022)
  Simplifies MoE by routing to a single expert (top-1) per token. Introduces the auxiliary load balancing loss to prevent expert collapse. Scaled to 1.6T parameters. Our MoE notebook implements this balancing loss.
  [Paper](https://arxiv.org/abs/2101.03961) · [Code](https://github.com/google-research/t5x)

- **Mixtral of Experts** (Jiang et al., 2024)
  Mixtral 8x7B: 8 expert FFNs with top-2 routing per layer. 47B total params but only 13B active per token. Matches or beats LLaMA 2 70B while being 6x faster at inference. The most influential open-source MoE model.
  [Paper](https://arxiv.org/abs/2401.04088) · [Code](https://github.com/mistralai/mistral-inference)

- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** (DeepSeek-AI, 2024)
  Pushes MoE further with 160 fine-grained experts and top-6 routing. 236B total params, 21B active. Introduces DeepSeekMoE architecture with shared experts alongside routed experts.
  [Paper](https://arxiv.org/abs/2405.04434) · [Code](https://github.com/deepseek-ai/DeepSeek-V2)

## Attention Variants

- **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019)
  Multi-Query Attention (MQA): all query heads share a single KV head. Dramatically reduces KV cache memory and decode latency at a small quality cost. Set `n_kv_heads=1` in our config to use this.
  [Paper](https://arxiv.org/abs/1911.02150)

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023)
  Grouped-Query Attention: a middle ground between MHA and MQA where groups of query heads share KV heads. Gets most of MQA's speed benefit with almost no quality loss. Our `attention.py` implements this via configurable `n_kv_heads`.
  [Paper](https://arxiv.org/abs/2305.13245)

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022)
  Rewrites attention to be IO-aware: tiles the computation to keep data in SRAM instead of reading/writing HBM repeatedly. Gives 2-4x speedup and reduces memory from O(n^2) to O(n). The key insight is that memory bandwidth, not compute, is the bottleneck for attention.
  [Paper](https://arxiv.org/abs/2205.14135) · [Code](https://github.com/Dao-AILab/flash-attention)

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** (Dao, 2023)
  Improves on FlashAttention with better work partitioning across thread blocks and warps, achieving 50-73% of A100 theoretical max FLOPS (vs 25-40% for FlashAttention-1). Also adds support for head dim up to 256 and cross-attention.
  [Paper](https://arxiv.org/abs/2307.08691) · [Code](https://github.com/Dao-AILab/flash-attention)

## Positional Encodings

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
  RoPE: encodes position by rotating query and key vectors in 2D subspaces. Key advantages over learned/sinusoidal: relative position awareness, decays with distance, and enables length extrapolation. Implemented in `layers.py` (`precompute_rope_frequencies`, `apply_rope`).
  [Paper](https://arxiv.org/abs/2104.09864) · [Code](https://github.com/ZhuiyiTechnology/roformer)

- **Train Short, Test Long: Attention with Linear Biases** (Press et al., 2022)
  ALiBi: adds a linear bias to attention scores based on query-key distance instead of using positional embeddings. Enables extrapolation to longer sequences at inference time without retraining. Simpler than RoPE but less widely adopted.
  [Paper](https://arxiv.org/abs/2108.12409) · [Code](https://github.com/ofirpress/attention_with_linear_biases)

## Normalization & Activations

- **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)
  RMSNorm: drops the mean-centering step from LayerNorm, keeping only the RMS scaling. ~10-15% faster than LayerNorm with equivalent quality. Used in LLaMA, Mistral, Gemma. Implemented in `layers.py`.
  [Paper](https://arxiv.org/abs/1910.07467)

- **GLU Variants Improve Transformer** (Shazeer, 2020)
  SwiGLU: a gated linear unit variant using SiLU (Swish) activation. Outperforms ReLU and GELU in transformers at the cost of an extra linear projection (3 matrices instead of 2 in FFN). Implemented in `layers.py`.
  [Paper](https://arxiv.org/abs/2002.05202)

## Training Optimization

- **Adam: A Method for Stochastic Optimization** (Kingma & Ba, 2015)
  Adaptive optimizer that tracks per-parameter first and second moments of gradients. The foundation for AdamW. Bias correction handles the zero-initialization of moment estimates. Our `optimizers.py` builds this step by step.
  [Paper](https://arxiv.org/abs/1412.6980)

- **Decoupled Weight Decay Regularization** (Loshchilov & Hutter, 2019)
  AdamW: fixes Adam's weight decay by decoupling it from the gradient update. Standard optimizer for transformer training. Our `optimizers.py` implements this from scratch for educational purposes.
  [Paper](https://arxiv.org/abs/1711.05101) · [Docs](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

- **SGDR: Stochastic Gradient Descent with Warm Restarts** (Loshchilov & Hutter, 2017)
  Introduces cosine annealing for learning rate scheduling. The cosine schedule spends more time at moderate LRs and decays smoothly, which consistently outperforms step decay. Our `trainer.py` uses cosine annealing with linear warmup.
  [Paper](https://arxiv.org/abs/1608.03983)

- **Mixed Precision Training** (Micikevicius et al., 2018)
  Train with FP16 computation and FP32 master weights to get ~2x speedup with minimal quality loss. Introduces loss scaling to prevent gradient underflow in FP16. BF16 (used in this repo) simplifies this by having the same exponent range as FP32, eliminating the need for loss scaling.
  [Paper](https://arxiv.org/abs/1710.03740) · [Docs](https://pytorch.org/docs/stable/amp.html)

- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022)
  The "Chinchilla" paper: derives scaling laws showing that models and datasets should be scaled equally. A 70B model trained on 1.4T tokens outperforms a 280B model trained on 300B tokens. Our `flops.py` FLOP counting formula comes from this paper.
  [Paper](https://arxiv.org/abs/2203.15556)

## Sampling & Decoding

- **The Curious Case of Neural Text Degeneration** (Holtzman et al., 2020)
  Introduces nucleus (top-p) sampling: instead of picking top-k tokens, sample from the smallest set whose cumulative probability exceeds p. Produces more diverse and natural text than top-k or greedy decoding. Implemented in `inference/generate.py`.
  [Paper](https://arxiv.org/abs/1904.09751)

## Inference Optimization

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023)
  vLLM's PagedAttention: manages KV cache memory like an OS manages virtual memory — using non-contiguous pages instead of pre-allocated contiguous blocks. Reduces KV cache waste by 60-80% and enables much higher batch sizes for serving.
  [Paper](https://arxiv.org/abs/2309.06180) · [Code](https://github.com/vllm-project/vllm) · [Docs](https://docs.vllm.ai/)

- **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., 2023)
  Uses a small "draft" model to propose N tokens, then verifies them all at once with the large model in a single forward pass. Provides 2-3x decode speedup without changing output distribution, because rejected tokens are resampled correctly.
  [Paper](https://arxiv.org/abs/2211.17192)

## Distributed Training

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (Shoeybi et al., 2019)
  Introduces tensor parallelism for transformers: splits attention heads and FFN matrices across GPUs within a layer. Enables training models too large for a single GPU's memory while maintaining high GPU utilization. Column-parallel + row-parallel linear layers with a single all-reduce per layer.
  [Paper](https://arxiv.org/abs/1909.08053) · [Code](https://github.com/NVIDIA/Megatron-LM)

- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020)
  Partitions optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across GPUs to eliminate memory redundancy in data parallelism. Stage 1 alone reduces memory by 4x. Stage 3 is equivalent to FSDP. Implemented in DeepSpeed.
  [Paper](https://arxiv.org/abs/1910.02054) · [Code](https://github.com/microsoft/DeepSpeed) · [Docs](https://www.deepspeed.ai/)

- **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel** (Zhao et al., 2023)
  PyTorch's native implementation of ZeRO Stage 3. Shards model parameters, gradients, and optimizer states across GPUs, all-gathering parameters just-in-time for each layer's forward/backward. Our `distributed.py` wraps this.
  [Paper](https://arxiv.org/abs/2304.11277) · [Docs](https://pytorch.org/docs/stable/fsdp.html)

## Quantization

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., 2023)
  One-shot weight quantization using approximate second-order information (inverse Hessian). Quantizes to 3-4 bits with minimal quality loss by compensating quantization error across columns. Much better than naive round-to-nearest at low bit widths.
  [Paper](https://arxiv.org/abs/2210.17323) · [Code](https://github.com/IST-DASLab/gptq) · [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

- **AWQ: Activation-aware Weight Quantization** (Lin et al., 2024)
  Observes that 1% of weight channels are disproportionately important (determined by activation magnitudes, not weight magnitudes). Protects these salient channels by per-channel scaling before quantization. Simpler than GPTQ and often better quality at 4-bit.
  [Paper](https://arxiv.org/abs/2306.00978) · [Code](https://github.com/mit-han-lab/llm-awq)

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Dettmers et al., 2022)
  Discovers that transformer weights have emergent outlier features (a few dimensions with values 10-100x larger than the rest). Handles them via mixed decomposition: outlier dimensions stay in FP16, the rest use INT8. Enables inference of 175B models on a single server with no quality loss.
  [Paper](https://arxiv.org/abs/2208.07339) · [Code](https://github.com/TimDettmers/bitsandbytes) · [Docs](https://huggingface.co/docs/bitsandbytes/)

## GPU Kernels & Performance

- **Beating cuBLAS in Single-Precision General Matrix Multiplication** (Salykova, 2024)
  Step-by-step walkthrough of writing an SGEMM kernel for the RTX 3090 (Ampere) that matches or beats cuBLAS. Develops two kernels — `128x128x8` and `128x256x8` — using hierarchical tiling (block / warp / thread), 8x8 register accumulators, double-buffered shared memory padded to 132 floats to avoid bank conflicts, vectorized `ld.shared.v4.f32` / `st.shared.v4.f32`, and `cp.async.ca.shared.global` for direct global→shared transfers on Ampere+. The companion notebook in `notebooks/05_gpu_nvidia_tools/02_sgemm_optimization.ipynb` reproduces this progression.
  [Article](https://salykova.github.io/sgemm-gpu) · [Code](https://github.com/salykova/sgemm.cu)

- **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance** (Boehm, 2022)
  The other classic SGEMM optimization tutorial. Walks from a naive kernel (~250 GFLOPS) through 10 progressively faster versions on an A6000, ending at ~93% of cuBLAS. Covers global memory coalescing, shared memory blocking, 1D and 2D thread tiling, vectorized loads, and warp tiling. The complementary read to Salykova for understanding *why* each step matters before Salykova's article shows how to push further with `cp.async`.
  [Article](https://siboehm.com/articles/22/CUDA-MMM) · [Code](https://github.com/siboehm/SGEMM_CUDA)

- **CUTLASS: CUDA Templates for Linear Algebra Subroutines** (NVIDIA, 2017–)
  NVIDIA's open-source template library implementing the same hierarchical-decomposition GEMM strategy used inside cuBLAS, parameterized over data types, tile sizes, layouts, and instruction sets (FMA, mma.sync tensor cores, Hopper WGMMA, TMA). Reading CUTLASS is the natural next step after writing SGEMM by hand — it shows the production-grade form of the same ideas plus tensor cores, software pipelining, and asynchronous warp-specialized kernels.
  [Code](https://github.com/NVIDIA/cutlass) · [Docs](https://docs.nvidia.com/cutlass/) · [GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)

- **Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking** (Jia et al., 2018)
  Reverse-engineers SM partitioning, shared memory bank structure, register file bandwidth, and instruction latencies on Volta. The numbers carry over with adjustments to Turing/Ampere/Ada. Useful when reasoning about why a kernel achieves the throughput it does — most performance ceilings come from one of the limits this paper measures.
  [Paper](https://arxiv.org/abs/1804.06826)

- **Programmatic Dependent Launch and Async Copy on Ampere** (NVIDIA Programming Guide)
  The `cp.async.ca.shared.global` instruction (Ampere SM_80+) lets a thread asynchronously copy from global memory directly into shared memory, bypassing the register file and freeing the SM to run independent compute while the load completes. Combined with `cp.async.commit_group` / `cp.async.wait_group`, this is what makes software pipelining of GEMM tiles practical without occupying registers as a load buffer. Salykova's `128x256x8` kernel relies on this.
  [Async copy docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async) · [PTX guide](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-copy)

## Agents

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
  Introduces the alternating reason → act → observe loop that nearly every modern agent uses. Showed that interleaving free-form reasoning ("thoughts") with tool calls outperforms both pure chain-of-thought and pure tool calling on reasoning-heavy benchmarks. The core loop in `agents/agent.py` is a direct implementation.
  [Paper](https://arxiv.org/abs/2210.03629) · [Code](https://github.com/ysymyth/ReAct)

- **Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
  First end-to-end demonstration that LMs can be fine-tuned to call APIs by self-supervised generation of tool-use traces. Pre-dates the structured tool-use APIs from OpenAI/Anthropic but shows the same idea: tool calling is a learnable text-generation pattern, not a special architectural feature.
  [Paper](https://arxiv.org/abs/2302.04761)

- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023)
  Adds a "reflect on failure, write notes, try again" loop on top of a base agent. The reflections accumulate as natural-language hints fed back into subsequent attempts. Gets dramatic improvements on coding/reasoning benchmarks at the cost of more LLM calls per task. Our `planner.py` supports an optional reflection step in the same spirit.
  [Paper](https://arxiv.org/abs/2303.11366) · [Code](https://github.com/noahshinn/reflexion)

- **MemGPT: Towards LLMs as Operating Systems** (Packer et al., 2023)
  Frames context-window management as virtual memory with explicit page-in/page-out function calls. The agent itself decides what to keep in working memory vs offload to recall storage. More principled than naive truncation; our `memory.py` implements a simpler version of the same two-tier idea.
  [Paper](https://arxiv.org/abs/2310.08560) · [Code](https://github.com/cpacker/MemGPT)

- **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023)
  The "Smallville" demo: 25 agents with rich memory streams, reflection, and planning living in a simulated town. Notable for the memory architecture (importance-weighted retrieval + periodic high-level reflections that summarize recent memories into beliefs).
  [Paper](https://arxiv.org/abs/2304.03442) · [Code](https://github.com/joonspk-research/generative_agents)

- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** (Wu et al., 2023)
  Microsoft's framework for multi-agent setups, formalising the "agents talking to agents" pattern. Our `multi_agent.py` is a much-stripped-down version: a supervisor that routes work to specialized workers based on their descriptions.
  [Paper](https://arxiv.org/abs/2308.08155) · [Code](https://github.com/microsoft/autogen)

- **Voyager: An Open-Ended Embodied Agent with Large Language Models** (Wang et al., 2023)
  LLM agent playing Minecraft. Notable for the "skill library" idea: every successfully written code snippet (the agent acts via code generation) gets stored, indexed, and reused later. A nice example of long-term *procedural* memory, distinct from the factual long-term memory in our `VectorMemory`.
  [Paper](https://arxiv.org/abs/2305.16291) · [Code](https://github.com/MineDojo/Voyager)

- **Building Effective Agents** (Anthropic, 2024)
  Engineering-focused blog post rather than a paper, but the most useful single read on agent design. Argues for simple, composable patterns over heavyweight frameworks. Covers the building blocks: augmented LLMs, prompt chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer, agents.
  [Post](https://www.anthropic.com/research/building-effective-agents)
