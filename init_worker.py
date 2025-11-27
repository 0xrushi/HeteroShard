#!/usr/bin/env python3
"""
Worker Initialization Script

Reads configuration and starts a worker with the appropriate settings.
"""

import json
import sys
import os
import argparse
import socket

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from hetero_framework.trainer.worker import run_worker
import torch.nn as nn


def find_worker_config(config_file: str) -> dict:
    """Find this machine's configuration in the config file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)
    
    # Try to identify this machine
    hostname = socket.gethostname()
    
    # Try to get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "unknown"
    
    print(f"📍 Current machine: {hostname} ({local_ip})")
    print()
    
    # Find matching worker config
    matching_worker = None
    for worker in config.get("workers", []):
        if worker["hostname"] == hostname or worker["ip"] == local_ip:
            matching_worker = worker
            break
    
    if not matching_worker:
        print("❌ Could not find configuration for this machine!")
        print(f"   Current hostname: {hostname}")
        print(f"   Current IP: {local_ip}")
        print()
        print("Available workers in config:")
        for worker in config.get("workers", []):
            print(f"  - Worker {worker['id']}: {worker['hostname']} ({worker['ip']})")
        print()
        print("You can manually specify worker ID with --worker-id flag")
        sys.exit(1)
    
    return matching_worker


# Define a default model shard (users should customize this)
class DefaultRemoteShard(nn.Module):
    """Default model shard - customize for your use case."""
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
    """Main worker initialization."""
    parser = argparse.ArgumentParser(description="Initialize worker from config file")
    parser.add_argument("--config", type=str, default="hetero_config.json",
                       help="Path to configuration file")
    parser.add_argument("--worker-id", type=int, default=None,
                       help="Manually specify worker ID (1-based)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("WORKER INITIALIZATION")
    print("=" * 70)
    print()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Find worker config
    if args.worker_id:
        worker_config = next(
            (w for w in config["workers"] if w["id"] == args.worker_id),
            None
        )
        if not worker_config:
            print(f"❌ Worker ID {args.worker_id} not found in config")
            sys.exit(1)
    else:
        worker_config = find_worker_config(args.config)
    
    print(f"✓ Using configuration for Worker {worker_config['id']}")
    print(f"  Hostname: {worker_config['hostname']}")
    print(f"  IP: {worker_config['ip']}")
    print(f"  Port: {worker_config['port']}")
    print(f"  Device: {worker_config['device']}")
    print(f"  GPUs: {worker_config['num_gpus']}")
    print()
    
    # Create model shard
    model_shard = DefaultRemoteShard()
    
    # Run worker
    print("Starting worker...")
    print("Press Ctrl+C to stop.")
    print()
    
    run_worker(
        model_shard=model_shard,
        listen_ip="0.0.0.0",  # Listen on all interfaces
        listen_port=worker_config["port"],
        device=worker_config["device"]
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorker stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

