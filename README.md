# Heterogeneous GPU Training Framework

A distributed deep learning framework for training models across heterogeneous hardware (e.g., Mac MPS + Linux NVIDIA/AMD GPU).

## Overview

This framework enables you to split neural network training across multiple machines with different hardware accelerators, connected via TCP/IP. It provides:

- **Device Abstraction Layer (DAL)**: Unified interface for CUDA, MPS, and CPU
- **Efficient Tensor Transport**: Optimized TCP/IP communication for PyTorch tensors
- **Flexible Architecture**: Easy to extend and customize for different model architectures
- **Visualization**: Automatic generation of architecture diagrams (Graphviz/Mermaid)

## Intro Video

[![Intro Video](https://img.youtube.com/vi/49jfIW3-Iy0/0.jpg)](https://www.youtube.com/watch?v=49jfIW3-Iy0)

## Features

- Explicit multi-stage pipeline with shard_plan
  - You specify exact layer ranges per stage as contiguous [start, end) indices.
  - One worker process per GPU, ordered as stages after the local stage.
- Forward and backward across stages
  - Forward: coordinator local shard → worker stages in order; last stage computes loss.
  - Backward: last stage returns dL/dx; coordinator drives upstream gradients hop-by-hop.
- Gradient synchronization and worker updates
  - Coordinator signals optimizer boundaries; workers step and zero_grads in lock-step.
  - Worker LoRA adapters are trained and saved locally by default (latest state).
- Architecture diagrams
  - Graphviz PNG/SVG/PDF and Mermaid .mmd, generated offline from config.
- Unified distributed entrypoint
  - One script auto-detects role from hetero_config.json.

## Limitations

- Do not split inside MoE blocks; expert-parallel MoE is not implemented.
- No aux-loss pass-through and no extra payload tensors at boundaries.
- Requires an explicit shard_plan with contiguous slices; tested on decoder-only stacks.
- Star-orchestrated over TCP sockets; no NCCL collectives or all-to-all patterns.
- Worker LoRA saves overwrite the latest by default; merging script assumes non-overlapping keys.

## Project Structure

```
hetero_framework/
├── core/
│   ├── __init__.py
│   ├── dal.py              # Device Abstraction Layer
│   ├── transport.py        # TCP/IP Tensor Transfer
│   ├── protocol.py         # Message serialization
├── visualizer/
│   ├── __init__.py
│   ├── graph_gen.py        # Graphviz/Mermaid generator
├── trainer/
│   ├── __init__.py
│   ├── worker.py           # Legacy simple worker
│   ├── remote_pipeline_worker.py  # Last-stage gradient worker
│   ├── relay_worker.py     # Multi-stage relay gradient worker
│   └── pipeline_multistage.py     # Coordinator multi-stage trainer
└── examples/
    ├── demo_llama8b4bit_distributed.py  # Unified (worker/coordinator) training script
    └── demo_real.py                      # Diagram-only renderer from config
```

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hetrogpu
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Graphviz for visualization:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install graphviz
   
   # macOS
   brew install graphviz
   
   # Arch Linux
   sudo pacman -S graphviz
   
   # Then install Python package (included in requirements.txt)
   pip install graphviz
   ```

### Configuration-Based Setup (Recommended)

1) Generate configuration (explicit shard_plan):
```bash
python init_config.py
```

2) Copy config to all machines:
```bash
scp hetero_config.json user@worker-ip:~/hetrogpu/
```

3) Start workers (one process per GPU/stage):
```bash
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json
```

4) Start coordinator (same command on the coordinator machine):
```bash
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json
```

Notes:
- The script auto-detects role from hetero_config.json.
- The trainer uses your shard_plan to split layers and orchestrate multi-stage forward/backward.
- Architecture diagram architecture.png is generated automatically when training starts.

**Benefits:**
- One config file for all machines
- Auto-detects worker identity
- Supports multiple GPUs per machine

## Usage Examples

### Unified distributed example (training)

Run the same script on workers and coordinator (auto-detects role):

```bash
# Workers
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json

# Coordinator (same command)
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json
```

### Checkpoint / Resume

- Coordinator periodically checkpoints and instructs workers to save in lock-step.
- Default directory is the example’s `output_dir`; override via CLI or config.
- Resume from a specific global step present on coordinator and all workers.

Examples:

```bash
python examples/demo_llama8b4bit_distributed.py --config hetero_config.json

# Coordinator: resume at step 20 and use custom dir
python examples/demo_llama8b4bit_distributed.py \
  --config hetero_config.json \
  --resume_step 20 \
  --checkpoint_dir ./lora_unsloth_sft_distributed
```

## Architecture

### System Topology

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   Localhost (Device A)  │         │  Remote Worker (Device B)│
│   ┌─────────────────┐   │         │   ┌─────────────────┐   │
│   │   Layer 1-2     │   │         │   │   Layer 3-4     │   │
│   │   (Local GPU)   │   │         │   │  (Remote GPU)   │   │
│   └─────────────────┘   │         │   └─────────────────┘   │
│          ↓              │         │          ↓              │
│   Coordinator Logic     │ ──TCP──→│   Worker Process        │
└─────────────────────────┘         └─────────────────────────┘
```

### Communication Flow

1. **Forward Pass:**
   - Coordinator executes local model shard
   - Serializes intermediate tensors
   - Sends via TCP to worker
   - Worker processes and returns result

2. **Data Transfer Protocol:**
   - 8-byte header with data size (big-endian)
   - Serialized PyTorch tensor data
   - Automatic CPU transfer for network compatibility

## Configuration

### Network Configuration

- **Default Port:** 9999
- **Protocol:** TCP/IP
- **Firewall:** Ensure port is open on worker machines

### Device Selection

Devices are picked by PyTorch (CUDA if available, else CPU/MPS). Run one worker process per GPU; the coordinator uses CUDA if available.

## Visualization

The framework generates architecture diagrams showing:
- Node distribution (local vs remote)
- Layer placement
- Network connections
- Device information

Supported formats:
- **PNG/SVG/PDF** (via Graphviz)
- **Mermaid** (.mmd files for documentation)

## Troubleshooting

### Connection Refused

**Problem:** Coordinator can't connect to worker

**Solutions:**
- Verify worker is running
- Check IP address is correct
- Verify port is open: `sudo ufw allow 9999` (Linux)
- Test connectivity: `ping <worker-ip>`

### CUDA Not Available

**Problem:** Worker falls back to CPU

**Solutions:**
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch with CUDA support

## Performance Tips

1. **Batch Size:** Use larger batches to amortize network overhead
2. **Model Partitioning:** Place compute-intensive layers on faster GPUs
3. **Network:** Use wired Gigabit Ethernet for best performance
4. **Tensor Size:** Minimize intermediate tensor dimensions between devices

## Advanced Topics

### Adding More Workers

Workers are configured exclusively via `hetero_config.json` now. To add more workers:
- Append entries to the `workers` list (same IP allowed multiple times with different ports/device indices for multi‑GPU hosts).
- Expand the `shard_plan` with additional contiguous layer ranges, one per worker, and set `is_last: true` on exactly one final stage.

### Custom Message Protocol

Extend the protocol for custom communication:

```python
from hetero_framework.core.protocol import Message, MessageType

msg = Message(
    type=MessageType.CONTROL,
    payload={"command": "checkpoint"},
    metadata={"step": 100}
)
```

## Constraints (explicit splits)

- Keep splits at block boundaries; do not split inside a MoE block.
- No aux-loss pass-through and no extra payloads are used in the pipeline protocol.
- Expert-parallel MoE (token all-to-all inside a layer) is not implemented in this pipeline trainer. A stub exists for future work.

For the distributed LLaMA example with multiple stages, define an explicit `shard_plan` in `hetero_config.json` and run one worker process per remote stage (per GPU/port). The coordinator orchestrates forward/backward across all stages and auto-generates `architecture.png`.


## License

This project is licensed under the Apache License, Version 2.0.

You may obtain a copy of the License at:

- `http://www.apache.org/licenses/LICENSE-2.0`

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the `LICENSE` file in this repository for the specific language
governing permissions and limitations under the License.

## Acknowledgments

Built for heterogeneous GPU training across different hardware platforms.

