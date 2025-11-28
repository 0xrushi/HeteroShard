"""
Unified Distributed Llama 8B 4-bit Training Script

This single script can run as either coordinator or worker.
It auto-detects its role based on the hetero_config.json file.

Usage:
    # On worker machine:
    python demo_llama8b4bit_distributed.py --config hetero_config.json

    # On coordinator machine:
    python demo_llama8b4bit_distributed.py --config hetero_config.json

The script automatically determines if it's a coordinator or worker
based on hostname/IP matching in the config file.
"""

import argparse
import json
import math
import os
import socket as socket_module
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.models.llama.modeling_llama import create_causal_mask

from hetero_framework.trainer import (
    MultiStagePipelineTrainer,
    run_relay_worker,
)


class Config:
    base_model = "unsloth/llama-3.1-8b-bnb-4bit"
    output_dir = "./lora_unsloth_sft_distributed"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    bias = "none"
    task_type = "CAUSAL_LM"

    # Training params
    max_length = 512
    batch_size = 1
    grad_accum_steps = 4
    lr = 2e-4
    weight_decay = 0.0
    adam_eps = 1e-8
    adam_betas = (0.9, 0.95)
    num_epochs = 1
    warmup_steps = 10
    clip_grad_norm = 1.0
    use_fp16 = False

    system_prompt = "You are a helpful assistant."
    instruction_template = "<|user|>\n{instruction}\n"
    response_template = "<|assistant|>\n{response}\n"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    save_every_n_steps = 20

    # Distributed
    split_layer = 16


class AlpacaSFTDataset(Dataset):
    """Stanford Alpaca dataset for SFT."""

    def __init__(
        self,
        tokenizer,
        max_length: int,
        split: str = "train[:100]",
        add_bos_eos: bool = True,
        mask_user_tokens: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos_eos = add_bos_eos
        self.mask_user_tokens = mask_user_tokens
        self.dataset = load_dataset("tatsu-lab/alpaca", split=split)

        self.formatted_texts: list[str] = []
        for row in self.dataset:
            instruction = (row.get("instruction") or "").strip()
            input_text = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()

            user_block = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction

            text = (
                (Config.bos_token if self.add_bos_eos else "")
                + f"{Config.system_prompt}\n"
                + Config.instruction_template.format(instruction=user_block)
                + Config.response_template.format(response=output)
                + (Config.eos_token if self.add_bos_eos else "")
            )
            self.formatted_texts.append(text)

    def __len__(self) -> int:
        return len(self.formatted_texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        text = self.formatted_texts[idx]
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        labels = input_ids.clone()

        if self.mask_user_tokens:
            decoded = self.tokenizer.decode(input_ids)
            assistant_tag = "<|assistant|>"
            pos = decoded.find(assistant_tag)
            if pos != -1:
                prefix_text = decoded[: pos + len(assistant_tag)]
                prefix_ids = self.tokenizer(
                    prefix_text, truncation=True, max_length=self.max_length, return_tensors="pt"
                )["input_ids"][0]
                cutoff = len(prefix_ids)
                labels[:cutoff] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


class CosineScheduler:
    """Simple cosine decay with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_count <= self.warmup_steps:
                lr = base_lr * (self.step_count / max(1, self.warmup_steps))
            else:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            group["lr"] = lr


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_lora(peft_model, output_dir: str, step: int | None = None):
    suffix = f"_step{step}" if step is not None else ""
    save_dir = os.path.join(output_dir, f"lora{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)
    print(f"Saved LoRA adapters to: {save_dir}")


class LocalModelShard(nn.Module):
    """Local shard: Embedding + Layers 0-15"""

    def __init__(self, full_model, split_layer: int = 16):
        super().__init__()
        self.split_layer = split_layer

        if hasattr(full_model, "base_model"):
            base_model = full_model.base_model.model
        else:
            base_model = full_model

        self.embed_tokens = base_model.model.embed_tokens
        self.layers = nn.ModuleList(base_model.model.layers[:split_layer])
        self.config = base_model.config
        # Rotary embedding module required by decoder layers in HF >= 4.57
        self.rotary_emb = base_model.model.rotary_emb

    def forward(self, input_ids, attention_mask=None):
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Build position ids and cache position
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        device = hidden_states.device
        cache_position = torch.arange(0, seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Build causal mask compatible with HF attention
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        # Precompute rotary position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through local decoder layers 0..split-1
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states


class RemoteModelShard(nn.Module):
    """Remote shard: Layers 16-31 + LM Head + Loss"""

    def __init__(self, full_model, split_layer: int = 16):
        super().__init__()
        self.split_layer = split_layer

        if hasattr(full_model, "base_model"):
            base_model = full_model.base_model.model
        else:
            base_model = full_model

        self.layers = nn.ModuleList(base_model.model.layers[split_layer:])
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.config = base_model.config
        # Rotary embedding module required by decoder layers in HF >= 4.57
        self.rotary_emb = base_model.model.rotary_emb

    def forward(self, hidden_states, attention_mask=None, labels=None):
        # Build position ids and cache position based on incoming hidden states
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        device = hidden_states.device
        cache_position = torch.arange(0, seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Build causal mask compatible with HF attention
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        # Precompute rotary position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through remote decoder layers split..end
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        if labels is None:
            return hidden_states
        logits = self.lm_head(hidden_states)
        # Compute CE loss in full precision for stability
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", enabled=False):
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


class IntermediateRemoteShard(nn.Module):
    """Intermediate remote shard: returns hidden states for next stage.

    Slices decoder layers [start:end] and runs them with proper attention mask
    and rotary embeddings, returning the transformed hidden states.
    """

    def __init__(self, full_model, start: int, end: int):
        super().__init__()
        if hasattr(full_model, "base_model"):
            base_model = full_model.base_model.model
        else:
            base_model = full_model
        self.layers = nn.ModuleList(base_model.model.layers[start:end])
        self.config = base_model.config
        self.rotary_emb = base_model.model.rotary_emb

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        device = hidden_states.device
        cache_position = torch.arange(0, seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states


def load_config(config_path: str):
    """Load hetero_config.json"""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("   Run: python init_config.py")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)


def detect_role(hetero_config: dict):
    """Detect if this machine is coordinator or worker"""
    hostname = socket_module.gethostname()
    try:
        s = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "unknown"

    print(f"📍 This machine: {hostname} ({local_ip})")

    # Check if coordinator
    coord = hetero_config.get("coordinator", {})
    if coord.get("hostname") == hostname or coord.get("ip") == local_ip:
        print("✓ Role: COORDINATOR\n")
        return "coordinator", None

    # Check if worker
    for worker in hetero_config.get("workers", []):
        if worker["hostname"] == hostname or worker["ip"] == local_ip:
            print(f"✓ Role: WORKER {worker['id']}\n")
            return "worker", worker

    print("❌ This machine not found in config!")
    print("   Add this machine to hetero_config.json")
    sys.exit(1)


def run_worker(worker_config: dict):
    """Run as worker. Supports multi-stage relay if shard_plan is defined."""
    print("=" * 70)
    print("WORKER MODE - LLAMA 8B 4-BIT (LAYERS 16-31)")
    print("=" * 70)
    print()

    print("Loading model...")
    full_model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=torch.float16 if Config.use_fp16 else torch.float32,
        device_map="auto",
    )
    full_model = prepare_model_for_kbit_training(full_model)

    lora_cfg = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        target_modules=Config.target_modules,
        bias=Config.bias,
        task_type=Config.task_type,
    )
    full_model = get_peft_model(full_model, lora_cfg)

    # Try multi-stage shard_plan
    cfg_path = None
    if "--config" in sys.argv:
        try:
            cfg_path = sys.argv[sys.argv.index("--config") + 1]
        except Exception:
            cfg_path = None
    hetero_cfg = load_config(cfg_path or "hetero_config.json")
    shard_plan = hetero_cfg.get("shard_plan")
    if shard_plan:
        this_id = worker_config.get("id")
        stage = next((s for s in shard_plan if s.get("id") == this_id), None)
        if stage is None:
            print(f"❌ shard_plan has no entry for worker id {this_id}")
            sys.exit(1)
        start, end = int(stage["start"]), int(stage["end"])
        is_last = bool(stage.get("is_last", False))
        if is_last:
            remote_shard = RemoteModelShard(full_model, start)
        else:
            remote_shard = IntermediateRemoteShard(full_model, start, end)
        print(f"✓ Remote shard ready ({start}-{end}{' LAST' if is_last else ''})")
        print(f"Listening on 0.0.0.0:{worker_config['port']}...")
        run_relay_worker(
            remote_shard,
            listen_ip="0.0.0.0",
            listen_port=worker_config["port"],
            use_fp16=Config.use_fp16,
            lr=Config.lr,
            betas=Config.adam_betas,
            eps=Config.adam_eps,
            weight_decay=Config.weight_decay,
            clip_grad_norm=Config.clip_grad_norm,
            peft_model=full_model,
            adapter_save_dir=Config.output_dir,
            worker_id=worker_config.get("id"),
        )
        return

    # No shard_plan: instruct user to define explicit splits
    print("❌ shard_plan not found in config. Please define explicit layer splits for this worker.")
    sys.exit(1)


def run_coordinator(hetero_config: dict):
    """Run as coordinator. Uses multi-stage trainer if shard_plan defines >1 remote stages."""
    print("=" * 70)
    print("COORDINATOR MODE - LLAMA 8B 4-BIT TRAINING")
    print("=" * 70)
    print()

    workers = hetero_config["workers"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if len(workers) == 1:
        worker = workers[0]
        print(f"Worker: {worker['hostname']} ({worker['ip']}:{worker['port']})")
        print(
            f"Split: Layers 0-{Config.split_layer-1} (local) | {Config.split_layer}-31 (remote)\n"
        )
    else:
        print(f"Workers: {len(workers)} configured\n")

    # Tokenizer and full model (PEFT) for shard construction and sampling
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": Config.eos_token})

    print("Loading model...")
    full_model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=torch.float16 if Config.use_fp16 else torch.float32,
        device_map="auto",
    )
    full_model = prepare_model_for_kbit_training(full_model)
    full_model.resize_token_embeddings(len(tokenizer))
    lora_cfg = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        target_modules=Config.target_modules,
        bias=Config.bias,
        task_type=Config.task_type,
    )
    full_model = get_peft_model(full_model, lora_cfg)
    full_model.print_trainable_parameters()

    # Local shard and dataloader
    print("\nCreating local shard...")
    shard_plan = hetero_config.get("shard_plan")
    if shard_plan:
        local_stage = next((s for s in shard_plan if s.get("id") == "local"), None)
        if local_stage is None:
            raise RuntimeError("shard_plan missing local stage")
        local_start, local_end = int(local_stage["start"]), int(local_stage["end"])

        class _LocalShard(nn.Module):
            def __init__(self, full_model, start: int, end: int):
                super().__init__()
                if hasattr(full_model, "base_model"):
                    base_model = full_model.base_model.model
                else:
                    base_model = full_model
                self.embed_tokens = base_model.model.embed_tokens
                self.layers = nn.ModuleList(base_model.model.layers[start:end])
                self.config = base_model.config
                self.rotary_emb = base_model.model.rotary_emb

            def forward(self, input_ids, attention_mask=None):
                hidden_states = self.embed_tokens(input_ids)
                batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
                device = hidden_states.device
                cache_position = torch.arange(0, seq_len, device=device)
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
                causal_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=None,
                    position_ids=position_ids,
                )
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                for layer in self.layers:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                return hidden_states

        local_shard = _LocalShard(full_model, local_start, local_end).to(device)
        print(f"✓ Local shard ready (Embedding + Layers {local_start}-{local_end-1})\n")
    else:
        local_shard = LocalModelShard(full_model, Config.split_layer).to(device)
        print(f"✓ Local shard ready (Embedding + Layers 0-{Config.split_layer-1})\n")
    local_shard.train()

    print("Loading dataset...")
    dataset = AlpacaSFTDataset(tokenizer=tokenizer, max_length=Config.max_length)
    collator = DataCollator(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collator, drop_last=False
    )
    print(f"✓ Dataset: {len(dataset)} examples\n")

    # Build and run multi-stage trainer using explicit config values
    # Always use multi-stage trainer when shard_plan is provided; otherwise error.
    if not shard_plan:
        print("❌ shard_plan not found in config. Please define explicit layer splits.")
        return

    print("Connecting to workers...")
    ms_trainer = MultiStagePipelineTrainer(
        local_shard=local_shard,
        workers=workers,
        shard_plan=shard_plan,
        dataloader=loader,
        batch_size=Config.batch_size,
        grad_accum_steps=Config.grad_accum_steps,
        lr=Config.lr,
        weight_decay=Config.weight_decay,
        adam_eps=Config.adam_eps,
        adam_betas=Config.adam_betas,
        num_epochs=Config.num_epochs,
        warmup_steps=Config.warmup_steps,
        clip_grad_norm=Config.clip_grad_norm,
        use_fp16=Config.use_fp16,
        device=device,
        diagram_basename="architecture",
        save_every_n_steps=Config.save_every_n_steps,
        checkpoint_dir=hetero_config.get("checkpoint_dir", Config.output_dir),
        resume_step=hetero_config.get("resume_step"),
    )
    ms_trainer.train()

    # Save adapters and sample
    try:
        ensure_output_dir(Config.output_dir)
        save_lora(full_model, Config.output_dir)

        full_model.eval()
        test_prompt = (
            f"{Config.bos_token}{Config.system_prompt}\n"
            + Config.instruction_template.format(
                instruction="Write a short haiku about distributed training."
            )
            + "<|assistant|>\n"
        )
        inputs = tokenizer(test_prompt, return_tensors="pt")
        model_device = next(full_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = full_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        print("\nSample generation:\n", tokenizer.decode(gen[0], skip_special_tokens=True))
    except Exception as e:
        print(f"Warning: sampler generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Distributed Llama 8B training")
    parser.add_argument(
        "--config", type=str, default="hetero_config.json", help="Path to hetero config file"
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["coordinator", "worker", "auto"],
        default="auto",
        help="Force role (default: auto-detect)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory (defaults to output_dir)",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Resume from checkpoint at this global step",
    )
    parser.add_argument(
        "--resume_latest",
        action="store_true",
        help="Resume from the latest available checkpoint step in checkpoint_dir",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Override save cadence for coordinator checkpoints",
    )
    args = parser.parse_args()

    hetero_config = load_config(args.config)
    
    if args.checkpoint_dir is not None:
        hetero_config["checkpoint_dir"] = args.checkpoint_dir
    if args.resume_step is not None:
        hetero_config["resume_step"] = args.resume_step
    if args.save_every_n_steps is not None:
        Config.save_every_n_steps = args.save_every_n_steps

    # Resolve resume_latest if requested
    if args.resume_latest and hetero_config.get("checkpoint_dir"):
        ckpt_root = os.path.join(hetero_config["checkpoint_dir"], "checkpoints", "coordinator")
        try:
            steps = []
            if os.path.isdir(ckpt_root):
                for name in os.listdir(ckpt_root):
                    if name.startswith("step_") and name.endswith(".pt"):
                        try:
                            steps.append(int(name.replace("step_", "").replace(".pt", "")))
                        except Exception:
                            pass
            if steps:
                latest = max(steps)
                hetero_config["resume_step"] = latest
                print(f"Resuming from latest checkpoint step: {latest}")
            else:
                print("No coordinator checkpoints found; starting fresh.")
        except Exception as e:
            print(f"Could not scan checkpoints for resume_latest: {e}")

    # Detect role
    if args.role == "auto":
        role, worker_config = detect_role(hetero_config)
    else:
        role = args.role
        worker_config = hetero_config["workers"][0] if role == "worker" else None

    # Run
    if role == "worker":
        run_worker(worker_config)
    else:
        run_coordinator(hetero_config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
