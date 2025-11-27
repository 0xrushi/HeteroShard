"""
Example Worker Script

Run this script on remote GPU machines (Device B, C, etc.).
Edit the model shard and configuration as needed.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hetero_framework.trainer.worker import run_worker


# Define the model shard for this worker
class RemoteShard(nn.Module):
    """Model shard to run on this worker."""
    def __init__(self, hidden_dim=10, output_dim=10):
        super().__init__()
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def main():
    """Main worker script."""
    # Configuration - EDIT THESE VALUES
    LISTEN_IP = "0.0.0.0"  # Listen on all interfaces
    LISTEN_PORT = 9999      # Port to listen on
    DEVICE = "auto"         # "auto", "cuda", "mps", or "cpu"
    
    print("="*70)
    print("REMOTE WORKER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Listen IP:   {LISTEN_IP}")
    print(f"  Listen Port: {LISTEN_PORT}")
    print(f"  Device:      {DEVICE}")
    print()
    
    # Create model shard
    model_shard = RemoteShard()
    
    # Run worker
    print("Starting worker...")
    print("Press Ctrl+C to stop.\n")
    
    run_worker(
        model_shard=model_shard,
        listen_ip=LISTEN_IP,
        listen_port=LISTEN_PORT,
        device=DEVICE
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorker stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

