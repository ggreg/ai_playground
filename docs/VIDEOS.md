# Video Resources

Curated videos matched to this repo's modules. Watching one before working through the corresponding notebook gives you the intuition; the notebook then makes it concrete in code. All links verified.

## Transformer Internals (`notebooks/01_transformer_internals/`)

- **Transformers, the tech behind LLMs | Deep Learning Chapter 5** (3Blue1Brown)
  The best visual intuition for what a transformer actually computes: embeddings as directions in space, the residual stream, and next-token prediction. Watch before `00_transformer_overview.ipynb`.
  [Watch](https://www.youtube.com/watch?v=wjZofJX0v4M)

- **Attention in transformers, step-by-step | Deep Learning Chapter 6** (3Blue1Brown)
  Queries, keys, and values animated — why attention is a soft, differentiable lookup. Pairs directly with `01_attention_mechanisms.ipynb` and `src/ai_playground/models/attention.py`.
  [Watch](https://www.youtube.com/watch?v=eMlx5fFNoYc)

- **How might LLMs store facts | Deep Learning Chapter 7** (3Blue1Brown)
  What the FFN/MLP blocks are for — the part of the architecture attention videos usually skip. Relevant to the SwiGLU implementation in `layers.py`.
  [Watch](https://www.youtube.com/watch?v=9-Jl0dxWQs8)

- **Let's build GPT: from scratch, in code, spelled out** (Andrej Karpathy)
  Builds a GPT in ~2 hours of live coding, from bigram counts to multi-head attention. The single most effective transformer tutorial; our from-scratch model follows the same spirit with LLaMA-style components.
  [Watch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

- **Let's build the GPT Tokenizer** (Andrej Karpathy)
  BPE from scratch — merges, regex splitting, and the weird failure modes tokenization causes (spelling, arithmetic, trailing whitespace). Background for the `vocab_size` config choice; see also [PAPERS.md § Tokenization](PAPERS.md).
  [Watch](https://www.youtube.com/watch?v=zduSFxRajkE)

- **A Visual Guide to Mixture of Experts (MoE) in LLMs** (Maarten Grootendorst)
  Dense vs sparse MoE, top-k routing, expert capacity, and load balancing — the concepts implemented in `02_mixture_of_experts.ipynb`.
  [Watch](https://www.youtube.com/watch?v=sOPDGQjFcuM)

## Training Optimization (`notebooks/02_training_optimization/`)

- **Let's reproduce GPT-2 (124M)** (Andrej Karpathy)
  A ~4-hour end-to-end training run that hits every topic in module 02: mixed precision, gradient accumulation, fused AdamW, LR warmup + cosine decay, torch.compile, and DDP. The closest thing to this repo in video form.
  [Watch](https://www.youtube.com/watch?v=l8pRSuU81PU)

- **Neural Networks: Zero to Hero** (Andrej Karpathy, playlist)
  The prerequisite series: backprop by hand, micrograd, makemore. If gradients feel shaky, start here before the optimizer notebooks (`04_adamw_from_scratch.ipynb`).
  [Watch](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

- **Deep Dive into LLMs like ChatGPT** (Andrej Karpathy)
  The full lifecycle — pretraining, SFT, RLHF — at a conceptual level. Good for placing this repo's pretraining focus in the bigger picture.
  [Watch](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Inference Optimization (`notebooks/04_inference_optimization/`)

- **Fast LLM Serving with vLLM and PagedAttention** (Anyscale)
  The vLLM authors present PagedAttention and continuous batching — the two ideas built from scratch in `00_mini_vllm.ipynb`. See [PAPERS.md](PAPERS.md) for the SOSP 2023 paper.
  [Watch](https://www.youtube.com/watch?v=5ZlavKF_98U)

- **The KV Cache: Memory Usage in Transformers** (Efficient NLP)
  The memory math that makes serving a cache-management problem — the right primer before the mini-vLLM chapter.
  [Watch](https://www.youtube.com/watch?v=80bIUggRJf4)

## Distributed Training (`notebooks/03_distributed_training/`)

- **Distributed Training with PyTorch: complete tutorial with cloud infrastructure and code** (Umar Jamil)
  DDP from the ground up: gradient synchronization, all-reduce, torchrun, and a real multi-node cloud setup. Directly applicable to `src/ai_playground/training/distributed.py`.
  [Watch](https://www.youtube.com/watch?v=toUSzwR0EV8)

- **Distributed Data Parallel in PyTorch — official video series** (PyTorch)
  Short official tutorials going from single-GPU to multi-node DDP, including a minGPT training example.
  [Watch](https://docs.pytorch.org/tutorials/beginner/ddp_series_intro.html)

## GPU & NVIDIA Tools (`notebooks/05_gpu_nvidia_tools/`)

- **How CUDA Programming Works** (Stephen Jones, GTC 2022)
  A CUDA architect explains the hardware truths behind the programming model: memory bandwidth as the real constraint, warps, occupancy. Pairs with `01_cuda_basics.ipynb`, and is the mental model behind the SGEMM notebook's roofline analysis.
  [Watch](https://www.youtube.com/watch?v=QQceTDjA4f4)

- **Flash Attention derived and coded from first principles with Triton** (Umar Jamil)
  Derives online softmax and tiling from scratch, then implements Flash Attention in Triton. Long but self-contained — no prior CUDA/Triton assumed. See [PAPERS.md](PAPERS.md) for the Flash Attention papers.
  [Watch](https://www.youtube.com/watch?v=zy8ChVd_oTM)

- **GPU MODE lecture series** (GPU MODE)
  Ongoing lecture series on CUDA, Triton, and GPU performance by practitioners. [Lecture 50](https://www.youtube.com/watch?v=4jQTb6sRGLg) recounts a CUDA → Triton → Flash Attention learning journey similar to this repo's module 05.
  [Channel](https://www.youtube.com/@GPUMODE)
