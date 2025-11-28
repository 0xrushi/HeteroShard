# Quick Start Guide

This quick start uses the unified LLaMA 8B 4‑bit distributed example.

## 1) Install

```bash
pip install -r requirements.txt
```

For PNG diagrams, install Graphviz system package (optional). Mermaid .mmd works without it.

## 2) Create config with explicit shard_plan

```bash
python init_config.py
```

When prompted, define the shard plan, for example (2 stages):
- local: start 0, end 16
- worker 1 (id=1): start 16, end 32, is_last=true

The script saves `hetero_config.json`.

Copy it to all machines:
```bash
scp hetero_config.json user@<worker-ip>:~/hetrogpu/
```

## 3) Start workers (one process per GPU/stage)

On each worker machine (run one process per GPU/stage listed in shard_plan):
```bash
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json
```

Workers load their shard, wait for connection, and begin training when the coordinator starts.

## 4) Start coordinator (same script)

On the coordinator machine:
```bash
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json
```

The script auto‑detects role from the config and starts the multi‑stage trainer. It generates `architecture.png` with layer ranges per device.

## Checklist

Before running:

- [ ] PyTorch installed on all machines
- [ ] Workers and coordinator reachable over network
- [ ] Port open (default 9999) on worker machines
- [ ] hetero_config.json includes valid shard_plan matching model layers
- [ ] One worker process per GPU/stage is running before starting coordinator

## Verify Setup

### Check Network Connectivity

From coordinator machine:
```bash
ping <worker-ip>
```

### Check Port Accessibility

From coordinator machine:
```bash
nc -zv <worker-ip> 9999
```

### Check PyTorch Installation

On both machines:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Expected Output (high level)

- Workers print losses or gradient steps per micro-batch and acknowledge optimizer steps.
- Coordinator prints per‑step loss and LR and generates `architecture.png`.

## Need Help?

1. Check [README.md](README.md) Troubleshooting section
2. Verify network connectivity, ports, and PyTorch install
3. Ensure workers are running before coordinator
