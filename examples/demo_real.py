"""
Demo: Real Multi-Node Setup

This demo is for actual multi-machine heterogeneous training.
Run this on the coordinator machine (Device A).
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hetero_framework.trainer.coordinator import HeterogeneousTrainer


# Define local model shard
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


def main():
    """Main coordinator script for real multi-node setup."""
    print("="*70)
    print("HETEROGENEOUS TRAINING - MULTI-NODE MODE")
    print("="*70)
    print("\nThis coordinator will connect to remote workers.")
    print("Make sure workers are running before starting!\n")
    
    # Configuration - EDIT THESE VALUES
    REMOTE_WORKERS = [
        {"ip": "192.168.1.50", "port": 9999},  # Worker 1
        # Add more workers here if needed:
        # {"ip": "192.168.1.51", "port": 9999},  # Worker 2
    ]
    
    print("Configuration:")
    print(f"  Local device: auto-detect")
    print(f"  Remote workers: {len(REMOTE_WORKERS)}")
    for i, worker in enumerate(REMOTE_WORKERS):
        print(f"    Worker {i+1}: {worker['ip']}:{worker['port']}")
    print()
    
    # Create local model shard
    local_shard = LocalShard()
    
    # Create trainer
    trainer = HeterogeneousTrainer(
        local_model=local_shard,
        remote_workers=REMOTE_WORKERS,
        device="auto",  # Auto-detect best device
        visualize=True
    )
    
    # Connect to workers
    if not trainer.connect_workers():
        print("\n❌ Failed to connect to workers.")
        print("\nTroubleshooting:")
        print("  1. Ensure worker script is running on remote machine(s)")
        print("  2. Check IP addresses and ports in configuration")
        print("  3. Verify network connectivity (ping the remote machine)")
        print("  4. Check firewall settings on remote machine(s)")
        return
    
    # Generate visualization
    print("Generating architecture visualization...")
    trainer.generate_visualization("real_architecture", format="png")
    trainer.generate_visualization("real_architecture", format="mmd")
    
    # Run training steps
    print("\n" + "="*70)
    print("RUNNING TRAINING STEPS")
    print("="*70 + "\n")
    
    num_steps = 5
    
    for step in range(num_steps):
        print(f"\n{'─'*70}")
        print(f"STEP {step + 1}/{num_steps}")
        print(f"{'─'*70}")
        
        # Create dummy input (replace with real data)
        dummy_input = torch.randn(1, 10)
        
        # Run training step
        try:
            output = trainer.train_step(dummy_input)
            
            print(f"\n✓ Step {step + 1} complete!")
            print(f"  Input shape:  {tuple(dummy_input.shape)}")
            print(f"  Output shape: {tuple(output.shape)}")
            print(f"  Output sample: {output[0, :5].tolist()}")
            
        except Exception as e:
            print(f"\n❌ Error in step {step + 1}: {e}")
            break
    
    # Cleanup
    print("\n" + "="*70)
    trainer.close()
    
    print("\n✓ Training complete!")
    print("\nGenerated files:")
    print("  - real_architecture.png (Graphviz diagram)")
    print("  - real_architecture.mmd (Mermaid diagram)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

