# Cloud Setup Guide

How to train and run inference on cloud GPUs. Organized by provider, with cost estimates and recommended configurations for each model size.

## GPU Requirements by Model Size

| Config | Params | Training GPU | VRAM Needed | Est. Cost/hr |
|--------|--------|-------------|-------------|--------------|
| `tiny.yaml` | ~10M | Any (CPU/MPS ok) | <1 GB | Free |
| `small.yaml` | ~125M | 1x A10G or T4 | ~8 GB | $0.50–1.00 |
| `medium.yaml` | ~350M | 1x A100 40GB | ~20 GB | $1.50–3.00 |
| `medium.yaml` (multi-GPU) | ~350M | 2–4x A100 | ~10 GB/GPU | $3.00–12.00 |

## Option 1: Lambda Cloud (Recommended for Learning)

Cheapest on-demand A100/H100 instances. No complex setup.

### Setup

1. Create an account at [lambdalabs.com](https://lambdalabs.com)
2. Add an SSH key in the dashboard
3. Launch an instance (recommended: 1x A100 40GB for `small.yaml`, or 4x A100 for `medium.yaml`)
4. SSH in and run:

```bash
# Clone the repo
git clone https://github.com/ggreg/ai_playground.git
cd ai_playground

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install dependencies
uv sync --extra dev

# Verify GPU access
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}')"
```

### Training

```bash
# Single GPU — small model
uv run python scripts/train.py --config configs/small.yaml --dtype bfloat16

# Multi-GPU — medium model with DDP
uv run python scripts/launch_distributed.py --nproc 4 --mode ddp --config configs/medium.yaml

# Multi-GPU — medium model with FSDP (lower memory per GPU)
uv run python scripts/launch_distributed.py --nproc 4 --mode fsdp --config configs/medium.yaml
```

### Inference benchmark

```bash
uv run python scripts/benchmark.py --config configs/small.yaml --dtype bfloat16
```

### Cost tips
- Lambda charges by the hour, so **stop instances when idle**
- Use `tmux` or `screen` so training survives SSH disconnects
- Estimated cost for full `small.yaml` training (5K steps): ~$2–4 on 1x A100

---

## Option 2: Vast.ai (Cheapest GPUs)

Marketplace for renting GPUs from independent hosts. Lowest prices but less reliable.

### Setup

1. Create an account at [vast.ai](https://vast.ai)
2. Install the CLI: `pip install vastai`
3. Search for instances:

```bash
# Find cheap A100 instances
vastai search offers 'gpu_name=A100 num_gpus=1 dph<2.0 inet_down>200' -o 'dph'
```

4. Rent an instance and SSH in
5. Follow the same clone/install steps as Lambda above

### Cost tips
- Interruptible instances are 2–5x cheaper but can be preempted
- Check reliability ratings before renting — prefer hosts with 99%+ uptime

---

## Option 3: Google Colab Pro (Easiest Start)

Best for running notebooks interactively. Limited for long training runs.

### Setup

1. Subscribe to [Colab Pro](https://colab.research.google.com/signup) ($10/mo) or Pro+ ($50/mo)
2. Open a notebook, set runtime to **GPU** (T4 or A100 if available)
3. In the first cell:

```python
# Clone and install
!git clone https://github.com/ggreg/ai_playground.git
%cd ai_playground
!pip install -e ".[dev]"

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

4. Run training from a cell:

```python
!python scripts/train.py --config configs/small.yaml --dtype bfloat16 --max-steps 1000
```

### Limitations
- Sessions timeout after idle periods (90 min free, longer on Pro)
- Max ~12 hours continuous runtime on Pro+
- Single GPU only (no distributed training)
- For long training, prefer Lambda or Vast.ai

---

## Option 4: AWS EC2 (Most Flexible)

Best GPU selection and reliability, but more setup overhead.

### Instance types

| Instance | GPU | VRAM | On-Demand $/hr | Use case |
|----------|-----|------|-----------------|----------|
| g5.xlarge | 1x A10G | 24 GB | ~$1.00 | `small.yaml` training |
| p4d.24xlarge | 8x A100 40GB | 320 GB | ~$32.00 | `medium.yaml` multi-GPU |
| p5.48xlarge | 8x H100 80GB | 640 GB | ~$98.00 | Large-scale experiments |

### Setup

1. Launch an instance with the **Deep Learning AMI** (Ubuntu, includes CUDA + PyTorch)
2. SSH in:

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

3. Clone and install:

```bash
git clone https://github.com/ggreg/ai_playground.git
cd ai_playground
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --extra dev
```

### Cost tips
- Use **Spot Instances** for 60–90% savings (add checkpointing — save every N steps)
- Set a billing alarm to avoid surprise charges
- Use `aws ec2 stop-instances` when done (don't just close SSH)

---

## Option 5: Modal (Serverless Inference)

Pay-per-second GPU compute. Great for inference and short experiments — no idle costs.

### Setup

1. Install: `pip install modal`
2. Authenticate: `modal setup`
3. Create `modal_serve.py`:

```python
import modal

app = modal.App("ai-playground")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "pyyaml", "safetensors"
).run_commands("git clone https://github.com/ggreg/ai_playground.git /app")

@app.function(gpu="A100", image=image, timeout=3600)
def train(config: str = "configs/small.yaml", max_steps: int = 1000):
    import subprocess
    subprocess.run(
        ["python", "scripts/train.py", "--config", config,
         "--dtype", "bfloat16", "--max-steps", str(max_steps)],
        cwd="/app", check=True,
    )

@app.function(gpu="A100", image=image)
def benchmark(config: str = "configs/small.yaml"):
    import subprocess
    subprocess.run(
        ["python", "scripts/benchmark.py", "--config", config, "--dtype", "bfloat16"],
        cwd="/app", check=True,
    )
```

4. Run: `modal run modal_serve.py::train --max-steps 500`

### Cost tips
- You only pay while the function is running (billed per second)
- A100: ~$0.001/sec (~$3.60/hr effective)
- Great for benchmarking and inference; for long training, Lambda/AWS is cheaper

---

## General Tips

### Always use tmux for training

```bash
# Start a session
tmux new -s train

# Run training inside tmux, then detach with Ctrl+B, D
# Reattach later with:
tmux attach -t train
```

### Monitor GPU utilization

```bash
# Real-time GPU stats (update every 1s)
watch -n 1 nvidia-smi

# Compact view
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv -l 1
```

### Save checkpoints to persistent storage

Cloud instances are ephemeral. After training, copy checkpoints off the machine:

```bash
# To your local machine
scp -r user@instance:/path/to/ai_playground/checkpoints/ ./checkpoints/

# Or to S3
aws s3 sync checkpoints/ s3://your-bucket/ai-playground/checkpoints/
```

### Estimate training cost before starting

Use the FLOP estimator to predict training time:

```bash
uv run python -c "
from ai_playground.profiling.flops import estimate_flops
from ai_playground.models.config import TransformerConfig
cfg = TransformerConfig.SMALL
flops_per_step = estimate_flops(cfg, seq_len=2048, batch_size=16)
total_flops = flops_per_step * 5000  # max_steps from small.yaml
# A100 does ~312 TFLOPS BF16, assume 50% MFU
time_seconds = total_flops / (312e12 * 0.5)
print(f'Estimated training time: {time_seconds / 3600:.1f} hours')
print(f'Estimated cost at \$3/hr: \${time_seconds / 3600 * 3:.0f}')
"
```
