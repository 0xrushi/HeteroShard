#!/usr/bin/env python3
"""
Combine multiple LoRA adapter directories into a single adapter for single-GPU use.

Each input directory must contain adapter_model.safetensors and adapter_config.json
as saved by PeftModel.save_pretrained(...).

Usage:
  python scripts/combine_lora_adapters.py \
      --adapters path/to/worker1 path/to/worker2 [...] \
      --output merged_adapter

Then, load on a single GPU:
  from peft import PeftModel
  model = AutoModelForCausalLM.from_pretrained(base)
  model = PeftModel.from_pretrained(model, 'merged_adapter')
"""

import argparse
import json
import os
import sys
from pathlib import Path
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


def main():
    parser = argparse.ArgumentParser(description="Combine multiple LoRA adapters into one")
    parser.add_argument(
        "--adapters", nargs="+", help="List of adapter directories to merge (ordered)"
    )
    parser.add_argument("--output", required=True, help="Output directory for merged adapter")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: dict = {}
    adapter_config = None

    for ad in args.adapters:
        ad_path = Path(ad)
        st_path = ad_path / "adapter_model.safetensors"
        cfg_path = ad_path / "adapter_config.json"
        if not st_path.exists() or not cfg_path.exists():
            print(
                f"Missing adapter files in {ad_path}: expected adapter_model.safetensors and adapter_config.json"
            )
            sys.exit(1)
        state = load_safetensors(str(st_path))
        overlaps = set(merged.keys()).intersection(state.keys())
        if overlaps:
            print(f"Overlapping adapter keys between adapters: {sorted(list(overlaps))[:5]} ...")
            print("   Ensure shard ranges do not overlap or merge logic is adjusted.")
            sys.exit(1)
        merged.update(state)
        # Load config (reuse the first; assumes same LoRA hyperparams)
        if adapter_config is None:
            adapter_config = json.loads(Path(cfg_path).read_text())

    if adapter_config is None:
        print("No adapters loaded.")
        sys.exit(1)

    save_safetensors(merged, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    print(f"✓ Merged {len(args.adapters)} adapters into: {out_dir}")


if __name__ == "__main__":
    main()
