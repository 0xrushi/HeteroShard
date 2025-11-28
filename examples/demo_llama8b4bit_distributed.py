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

import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import argparse
import json
import socket as socket_module

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hetero_framework.core.transport import send_tensor, recv_tensor
from transformers.models.llama.modeling_llama import create_causal_mask


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
    use_fp16 = True

    system_prompt = "You are a helpful assistant."
    instruction_template = "<|user|>\n{instruction}\n"
    response_template = "<|assistant|>\n{response}\n"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    save_every_n_steps = 100

    # Distributed
    split_layer = 16


class AlpacaSFTDataset(Dataset):
    """Stanford Alpaca dataset for SFT."""
    
    def __init__(self, tokenizer, max_length: int, split: str = "train[:100]",
                 add_bos_eos: bool = True, mask_user_tokens: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos_eos = add_bos_eos
        self.mask_user_tokens = mask_user_tokens
        self.dataset = load_dataset("tatsu-lab/alpaca", split=split)
        
        self.formatted_texts: List[str] = []
        for row in self.dataset:
            instruction = (row.get("instruction") or "").strip()
            input_text = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()

            if input_text:
                user_block = f"{instruction}\n\nInput:\n{input_text}"
            else:
                user_block = instruction

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.formatted_texts[idx]
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        labels = input_ids.clone()

        if self.mask_user_tokens:
            decoded = self.tokenizer.decode(input_ids)
            assistant_tag = "<|assistant|>"
            pos = decoded.find(assistant_tag)
            if pos != -1:
                prefix_text = decoded[:pos + len(assistant_tag)]
                prefix_ids = self.tokenizer(prefix_text, truncation=True, 
                                           max_length=self.max_length, return_tensors="pt")["input_ids"][0]
                cutoff = len(prefix_ids)
                labels[:cutoff] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
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
                progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            group["lr"] = lr


class LocalModelShard(nn.Module):
    """Local shard: Embedding + Layers 0-15"""
    def __init__(self, full_model, split_layer: int = 16):
        super().__init__()
        self.split_layer = split_layer
        
        if hasattr(full_model, 'base_model'):
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
        
        if hasattr(full_model, 'base_model'):
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
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss if loss is not None else torch.tensor(0.0, device=hidden_states.device)


def load_config(config_path: str):
    """Load hetero_config.json"""
    try:
        with open(config_path, 'r') as f:
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
        print(f"✓ Role: COORDINATOR\n")
        return "coordinator", None
    
    # Check if worker
    for worker in hetero_config.get("workers", []):
        if worker["hostname"] == hostname or worker["ip"] == local_ip:
            print(f"✓ Role: WORKER {worker['id']}\n")
            return "worker", worker
    
    print(f"❌ This machine not found in config!")
    print(f"   Add this machine to hetero_config.json")
    sys.exit(1)


def run_worker(worker_config: dict):
    """Run as worker"""
    print("=" * 70)
    print("WORKER MODE - LLAMA 8B 4-BIT (LAYERS 16-31)")
    print("=" * 70)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Port: {worker_config['port']}\n")
    
    # Load model
    print("Loading model...")
    full_model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=torch.float16 if Config.use_fp16 else torch.float32,
        device_map="auto",
    )
    full_model = prepare_model_for_kbit_training(full_model)
    
    lora_cfg = LoraConfig(
        r=Config.lora_r, lora_alpha=Config.lora_alpha, lora_dropout=Config.lora_dropout,
        target_modules=Config.target_modules, bias=Config.bias, task_type=Config.task_type,
    )
    full_model = get_peft_model(full_model, lora_cfg)
    
    # Create shard
    print("Creating remote shard...")
    remote_shard = RemoteModelShard(full_model, Config.split_layer).to(device)
    remote_shard.train()
    print(f"✓ Remote shard ready (Layers {Config.split_layer}-31)\n")
    
    # Start server
    print(f"Listening on 0.0.0.0:{worker_config['port']}...")
    server = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    server.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", worker_config['port']))
    server.listen(1)
    
    conn, addr = server.accept()
    print(f"✓ Connected to coordinator at {addr}\n")
    
    step = 0
    try:
        while True:
            print(f"[Step {step}] Waiting for data...")
            
            hidden_states = recv_tensor(conn)
            if hidden_states is None:
                print("Connection closed.")
                break
            
            attention_mask = recv_tensor(conn)
            labels = recv_tensor(conn)
            
            hidden_states = hidden_states.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            labels = labels.to(device) if labels is not None else None
            
            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                loss = remote_shard(hidden_states, attention_mask, labels)
            
            print(f"[Step {step}] Loss: {loss.item():.4f}")
            send_tensor(conn, loss.cpu())
            print(f"[Step {step}] ✓ Complete\n")
            step += 1
            
    except KeyboardInterrupt:
        print("\nWorker stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        server.close()


def run_coordinator(hetero_config: dict):
    """Run as coordinator"""
    print("=" * 70)
    print("COORDINATOR MODE - LLAMA 8B 4-BIT TRAINING")
    print("=" * 70)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker = hetero_config["workers"][0]
    
    print(f"Device: {device}")
    print(f"Worker: {worker['hostname']} ({worker['ip']}:{worker['port']})")
    print(f"Split: Layers 0-{Config.split_layer-1} (local) | {Config.split_layer}-31 (remote)\n")
    
    # Connect to worker
    print(f"Connecting to worker...")
    sock = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    try:
        sock.connect((worker['ip'], worker['port']))
        print("✓ Connected\n")
    except ConnectionRefusedError:
        print(f"❌ Connection failed!")
        print(f"   Start worker first: python {sys.argv[0]} --config {sys.argv[2]}")
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": Config.eos_token})
    
    # Load model
    print("Loading model...")
    full_model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=torch.float16 if Config.use_fp16 else torch.float32,
        device_map="auto",
    )
    full_model = prepare_model_for_kbit_training(full_model)
    full_model.resize_token_embeddings(len(tokenizer))
    
    lora_cfg = LoraConfig(
        r=Config.lora_r, lora_alpha=Config.lora_alpha, lora_dropout=Config.lora_dropout,
        target_modules=Config.target_modules, bias=Config.bias, task_type=Config.task_type,
    )
    full_model = get_peft_model(full_model, lora_cfg)
    full_model.print_trainable_parameters()
    
    # Create local shard
    print("\nCreating local shard...")
    local_shard = LocalModelShard(full_model, Config.split_layer).to(device)
    local_shard.train()
    print(f"✓ Local shard ready (Embedding + Layers 0-{Config.split_layer-1})\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = AlpacaSFTDataset(tokenizer=tokenizer, max_length=Config.max_length)
    collator = DataCollator(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, 
                       collate_fn=collator, drop_last=False)
    print(f"✓ Dataset: {len(dataset)} examples\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(local_shard.parameters(), lr=Config.lr, 
                                  betas=Config.adam_betas, eps=Config.adam_eps, 
                                  weight_decay=Config.weight_decay)
    
    total_steps = Config.num_epochs * math.ceil(len(loader) / Config.grad_accum_steps)
    scheduler = CosineScheduler(optimizer, warmup_steps=Config.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)
    
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Steps: {total_steps} | Batch: {Config.batch_size} | Accum: {Config.grad_accum_steps}\n")
    
    # Training loop
    global_step = 0
    accum_steps = 0
    running_loss = 0.0
    
    for epoch in range(Config.num_epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                hidden_states = local_shard(input_ids, attention_mask)
            
            # Send to worker
            send_tensor(sock, hidden_states.cpu())
            send_tensor(sock, attention_mask.cpu())
            send_tensor(sock, labels)
            
            # Receive loss
            loss = recv_tensor(sock).to(device)
            loss = loss / Config.grad_accum_steps
            
            scaler.scale(loss).backward()
            running_loss += loss.item()
            accum_steps += 1
            
            if accum_steps % Config.grad_accum_steps == 0:
                if Config.clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(local_shard.parameters(), Config.clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                global_step += 1
                avg_loss = running_loss
                running_loss = 0.0
                accum_steps = 0
                
                lr = scheduler.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} | Step {global_step}/{total_steps} | Loss {avg_loss:.4f} | LR {lr:.6f}")
    
    print("\n✓ Training complete!")
    sock.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed Llama 8B training")
    parser.add_argument("--config", type=str, default="hetero_config.json",
                       help="Path to hetero config file")
    parser.add_argument("--role", type=str, choices=["coordinator", "worker", "auto"], 
                       default="auto", help="Force role (default: auto-detect)")
    args = parser.parse_args()
    
    # Load config
    hetero_config = load_config(args.config)
    
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
