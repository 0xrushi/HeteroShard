#!/usr/bin/env python3
"""
Interactive Configuration Generator for Heterogeneous Training

This script creates a configuration file that can be used across all machines
without needing to modify individual files.
"""

import json
import os
import sys
from typing import List, Dict, Any


def print_header():
    """Print welcome header."""
    print("=" * 70)
    print("HETEROGENEOUS TRAINING - CONFIGURATION GENERATOR")
    print("=" * 70)
    print()
    print("This script will help you configure your distributed training setup.")
    print("The generated config can be copied to all machines.")
    print()


def get_int_input(prompt: str, default: int = 1, min_val: int = 1) -> int:
    """Get integer input from user with validation."""
    while True:
        try:
            value = input(f"{prompt} [default: {default}]: ").strip()
            if not value:
                return default
            num = int(value)
            if num < min_val:
                print(f"❌ Value must be at least {min_val}")
                continue
            return num
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n❌ Configuration cancelled by user")
            sys.exit(1)


def get_string_input(prompt: str, default: str = "") -> str:
    """Get string input from user."""
    try:
        value = input(f"{prompt} [default: {default}]: ").strip()
        return value if value else default
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user")
        sys.exit(1)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    try:
        while True:
            value = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not value:
                return default
            if value in ['y', 'yes']:
                return True
            if value in ['n', 'no']:
                return False
            print("❌ Please enter 'y' or 'n'")
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user")
        sys.exit(1)


def detect_current_machine_info() -> Dict[str, Any]:
    """Try to detect current machine information."""
    import socket
    
    info = {
        "hostname": socket.gethostname(),
        "ip": "unknown"
    }
    
    try:
        # Try to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["ip"] = s.getsockname()[0]
        s.close()
    except:
        pass
    
    return info


def main():
    """Main configuration generator."""
    print_header()
    
    # Detect current machine
    current_machine = detect_current_machine_info()
    print(f"📍 Current machine detected:")
    print(f"   Hostname: {current_machine['hostname']}")
    print(f"   IP: {current_machine['ip']}")
    print()
    
    # Initialize configuration
    config = {
        "version": "0.1.0",
        "coordinator": {},
        "workers": [],
        "training": {}
    }
    
    # ============== COORDINATOR CONFIGURATION ==============
    print("=" * 70)
    print("COORDINATOR CONFIGURATION")
    print("=" * 70)
    print()
    
    is_coordinator_current = get_yes_no(
        "Is the current machine the coordinator?",
        default=True
    )
    
    if is_coordinator_current:
        config["coordinator"]["hostname"] = current_machine["hostname"]
        config["coordinator"]["ip"] = current_machine["ip"]
    else:
        config["coordinator"]["hostname"] = get_string_input(
            "Enter coordinator hostname",
            default="coordinator"
        )
        config["coordinator"]["ip"] = get_string_input(
            "Enter coordinator IP address",
            default="192.168.1.100"
        )
    
    config["coordinator"]["device"] = get_string_input(
        "Coordinator device (auto/cuda/mps/cpu)",
        default="auto"
    )
    
    print()
    
    # ============== WORKER CONFIGURATION ==============
    print("=" * 70)
    print("REMOTE WORKER CONFIGURATION")
    print("=" * 70)
    print()
    
    num_workers = get_int_input(
        "How many remote worker machines (excluding coordinator)?",
        default=1,
        min_val=1
    )
    
    print()
    
    for i in range(num_workers):
        print(f"\n--- Worker {i + 1} Configuration ---")
        
        worker = {
            "id": i + 1,
            "hostname": get_string_input(
                f"  Worker {i + 1} hostname",
                default=f"worker{i + 1}"
            ),
            "ip": get_string_input(
                f"  Worker {i + 1} IP address",
                default=f"192.168.1.{150 + i}"
            ),
            "port": get_int_input(
                f"  Worker {i + 1} port",
                default=9999,
                min_val=1024
            ),
            "device": get_string_input(
                f"  Worker {i + 1} device (auto/cuda/mps/cpu)",
                default="auto"
            ),
            "num_gpus": get_int_input(
                f"  Number of GPUs on Worker {i + 1}",
                default=1,
                min_val=1
            )
        }
        
        config["workers"].append(worker)
    
    print()
    
    # ============== TRAINING CONFIGURATION ==============
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print()
    
    config["training"]["visualize"] = get_yes_no(
        "Generate architecture diagrams?",
        default=True
    )
    
    config["training"]["visualization_format"] = get_string_input(
        "Diagram format (png/svg/pdf/mmd)",
        default="png"
    )
    
    config["training"]["log_level"] = get_string_input(
        "Log level (DEBUG/INFO/WARNING/ERROR)",
        default="INFO"
    )
    
    print()
    
    # ============== SAVE CONFIGURATION ==============
    print("=" * 70)
    print("SAVE CONFIGURATION")
    print("=" * 70)
    print()
    
    config_filename = get_string_input(
        "Configuration filename",
        default="hetero_config.json"
    )
    
    # Create config directory if it doesn't exist
    config_dir = os.path.dirname(config_filename) if os.path.dirname(config_filename) else "."
    if config_dir != "." and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    # Save configuration
    try:
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_filename}")
        print()
        
        # Print summary
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print()
        print(f"Coordinator: {config['coordinator']['hostname']} ({config['coordinator']['ip']})")
        print(f"Workers: {len(config['workers'])}")
        for worker in config["workers"]:
            print(f"  - Worker {worker['id']}: {worker['hostname']} ({worker['ip']}:{worker['port']}) - {worker['num_gpus']} GPU(s)")
        print(f"Visualization: {'enabled' if config['training']['visualize'] else 'disabled'}")
        print()
        
        # Print next steps
        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("1. Copy this config file to all machines:")
        print(f"   scp {config_filename} user@worker-ip:~/hetrogpu/")
        print()
        print("2. On each worker machine, run:")
        print(f"   python init_worker.py --config {config_filename}")
        print()
        print("3. On the coordinator machine, run:")
        print(f"   python init_coordinator.py --config {config_filename}")
        print()
        print("Or use the examples with --config flag:")
        print(f"   python examples/demo_real.py --config {config_filename}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Configuration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

