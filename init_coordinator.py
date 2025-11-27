#!/usr/bin/env python3
"""
Coordinator Initialization Script

Reads configuration and starts the coordinator with the appropriate settings.
"""

import json
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from hetero_framework.trainer import HeterogeneousTrainer
import torch
import torch.nn as nn


# Define a default local model shard (users should customize this)
class DefaultLocalShard(nn.Module):
    """Default local model shard - customize for your use case."""
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
    """Main coordinator initialization."""
    parser = argparse.ArgumentParser(description="Initialize coordinator from config file")
    parser.add_argument("--config", type=str, default="hetero_config.json",
                       help="Path to configuration file")
    parser.add_argument("--steps", type=int, default=5,
                       help="Number of training steps to run")
    args = parser.parse_args()
    
    print("=" * 70)
    print("COORDINATOR INITIALIZATION")
    print("=" * 70)
    print()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print(f"   Run 'python init_config.py' to create a configuration")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)
    
    # Extract worker configurations
    remote_workers = [
        {"ip": w["ip"], "port": w["port"]}
        for w in config["workers"]
    ]
    
    print("Configuration loaded:")
    print(f"  Coordinator: {config['coordinator']['hostname']} ({config['coordinator']['ip']})")
    print(f"  Device: {config['coordinator']['device']}")
    print(f"  Workers: {len(remote_workers)}")
    for i, worker_full in enumerate(config["workers"]):
        print(f"    Worker {i+1}: {worker_full['hostname']} ({worker_full['ip']}:{worker_full['port']}) - {worker_full['num_gpus']} GPU(s)")
    print(f"  Visualization: {'enabled' if config['training']['visualize'] else 'disabled'}")
    print()
    
    # Create local model shard
    local_shard = DefaultLocalShard()
    
    # Create trainer
    trainer = HeterogeneousTrainer(
        local_model=local_shard,
        remote_workers=remote_workers,
        device=config['coordinator']['device'],
        visualize=config['training']['visualize']
    )
    
    # Connect to workers
    if not trainer.connect_workers():
        print("\n❌ Failed to connect to workers.")
        print("\nTroubleshooting:")
        print("  1. Ensure worker scripts are running on remote machines")
        print("  2. Check IP addresses and ports in configuration")
        print("  3. Verify network connectivity (ping the remote machines)")
        print("  4. Check firewall settings on remote machines")
        return
    
    # Generate visualization
    if config['training']['visualize']:
        try:
            print("Generating architecture visualization...")
            viz_format = config['training'].get('visualization_format', 'png')
            trainer.generate_visualization("architecture", format=viz_format, raise_on_error=True)
            
            # Also generate mermaid for documentation
            if viz_format != 'mmd':
                trainer.generate_visualization("architecture", format="mmd", raise_on_error=False)
        except ImportError as e:
            print(f"\n⚠️  Visualization skipped: {e}")
            print("   Continuing without diagrams...")
        except Exception as e:
            print(f"\n⚠️  Visualization error: {e}")
            print("   Continuing without diagrams...")
    
    # Run training steps
    print("\n" + "=" * 70)
    print("RUNNING TRAINING STEPS")
    print("=" * 70 + "\n")
    
    num_steps = args.steps
    
    for step in range(num_steps):
        print(f"\n{'─' * 70}")
        print(f"STEP {step + 1}/{num_steps}")
        print(f"{'─' * 70}")
        
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
    print("\n" + "=" * 70)
    trainer.close()
    
    print("\n✓ Training complete!")
    if config['training']['visualize']:
        print("\nGenerated files:")
        print(f"  - architecture.{config['training'].get('visualization_format', 'png')}")
        print("  - architecture.mmd (Mermaid diagram)")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

