"""
Demo: Single GPU / CPU Simulation

This demo simulates a heterogeneous setup on a single machine.
Useful for testing and development without multiple machines.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hetero_framework.trainer.coordinator import HeterogeneousTrainer
from hetero_framework.trainer.worker import RemoteWorker
import threading
import time


# Define model shards
class LocalShard(nn.Module):
    """First part of the model (runs on coordinator)."""
    def __init__(self, input_dim=10, hidden_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x


class RemoteShard(nn.Module):
    """Second part of the model (runs on worker)."""
    def __init__(self, hidden_dim=10, output_dim=10):
        super().__init__()
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def run_simulated_worker():
    """Run a simulated worker in a separate thread."""
    print("\n[WORKER THREAD] Starting simulated worker...")
    time.sleep(1)  # Give coordinator time to initialize
    
    remote_shard = RemoteShard()
    worker = RemoteWorker(
        model_shard=remote_shard,
        listen_ip="127.0.0.1",
        listen_port=9999,
        device="cpu"  # Use CPU for simulation
    )
    
    try:
        worker.start()
    except Exception as e:
        print(f"[WORKER THREAD] Error: {e}")
    finally:
        worker.stop()


def main():
    """Main simulation demo."""
    print("="*70)
    print("HETEROGENEOUS TRAINING - SIMULATION MODE")
    print("="*70)
    print("\nThis demo simulates a 2-device setup on a single machine.")
    print("Worker will run in a background thread.\n")
    
    # Start worker in background thread
    worker_thread = threading.Thread(target=run_simulated_worker, daemon=True)
    worker_thread.start()
    
    # Give worker time to start
    time.sleep(2)
    
    # Create coordinator
    local_shard = LocalShard()
    
    trainer = HeterogeneousTrainer(
        local_model=local_shard,
        remote_workers=[{"ip": "127.0.0.1", "port": 9999}],
        device="cpu",  # Use CPU for simulation
        visualize=True
    )
    
    # Connect to worker
    if not trainer.connect_workers():
        print("Failed to connect to simulated worker.")
        return
    
    # Generate visualization
    print("Generating architecture visualization...")
    trainer.generate_visualization("simulation_architecture", format="png")
    
    # Run a few training steps
    print("\n" + "="*70)
    print("RUNNING TRAINING STEPS")
    print("="*70 + "\n")
    
    for step in range(3):
        print(f"\n{'─'*70}")
        print(f"STEP {step + 1}")
        print(f"{'─'*70}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 10)
        
        # Run training step
        output = trainer.train_step(dummy_input)
        
        print(f"\n✓ Step {step + 1} complete!")
        print(f"  Input shape:  {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Output sample: {output[0, :5].tolist()}")
    
    # Cleanup
    print("\n" + "="*70)
    trainer.close()
    
    print("\n✓ Simulation complete!")
    print("\nCheck the generated 'simulation_architecture.png' file to see the")
    print("architecture diagram.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

