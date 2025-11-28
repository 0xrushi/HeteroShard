# filename: finetune_unsloth_no_trainer.py
import os
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class Config:
    base_model = "unsloth/llama-3.1-8b-bnb-4bit"
    output_dir = "./lora_unsloth_sft"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # typical for LLaMA-like
    bias = "none"
    task_type = "CAUSAL_LM"

    # Tokenization/training params
    max_length = 2048
    batch_size = 1            # increase with more VRAM
    grad_accum_steps = 16     # effective batch size = batch_size * grad_accum_steps
    lr = 2e-4
    weight_decay = 0.0
    adam_eps = 1e-8
    adam_betas = (0.9, 0.95)
    num_epochs = 1
    warmup_steps = 100
    clip_grad_norm = 1.0

    # Mixed precision
    use_fp16 = True # disable for strix halo

    system_prompt = "You are a helpful assistant."
    instruction_template = "<|user|>\n{instruction}\n"
    response_template = "<|assistant|>\n{response}\n"

    # Special tokens for chat-y formatting
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"

    # Save frequency
    save_every_n_steps = 1000


class AlpacaSFTDataset(Dataset):
    """
    Loads a small slice of the Stanford Alpaca dataset and formats each row into
    a single prompt+response text suitable for causal LM SFT.
    - Hugging Face dataset: "tatsu-lab/stanford_alpaca"
    - Fields: instruction, input (optional), output
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        split: str = "train[:1000]",
        add_bos_eos: bool = True,
        mask_user_tokens: bool = False,
    ):
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

            # Build a user instruction string: include input when present
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

        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]

        labels = input_ids.clone()

        if self.mask_user_tokens:
            # mask loss on user/prompt tokens, train only on assistant response.
            decoded = self.tokenizer.decode(input_ids)
            # We expect something like: ... <|assistant|>\n RESPONSE ... </s>
            assistant_tag = "<|assistant|>"
            pos = decoded.find(assistant_tag)
            if pos != -1:
                # Re-tokenize segments to find index boundary robustly
                # Split into prefix (user/system) and rest (assistant)
                prefix_text = decoded[:pos + len(assistant_tag)]
                prefix_ids = self.tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )["input_ids"][0]
                cutoff = len(prefix_ids)
                labels[:cutoff] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Pad to max length in batch
        input_ids = [b["input_ids"] for b in batch]
        attention_masks = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 ignores loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

# Optimizer & scheduler
class CosineScheduler:
    """
    Simple cosine decay with linear warmup.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
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
                # cosine from 1.0 down to min_lr_ratio
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            group["lr"] = lr

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_lora(peft_model, output_dir: str, step: Optional[int] = None):
    suffix = f"_step{step}" if step is not None else ""
    save_dir = os.path.join(output_dir, f"lora{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)
    print(f"Saved LoRA adapters to: {save_dir}")

def main():
    start_time = time.perf_counter()
    ensure_output_dir(Config.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(Config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": Config.eos_token})
    pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        Config.base_model,
        torch_dtype=torch.float16 if Config.use_fp16 else torch.float32,
        device_map="auto",
    )

    # If the base is a k-bit quantized model, prep it for PEFT training
    model = prepare_model_for_kbit_training(model)

    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        target_modules=Config.target_modules,
        bias=Config.bias,
        task_type=Config.task_type,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = AlpacaSFTDataset(tokenizer=tokenizer, max_length=Config.max_length)
    collator = DataCollator(tokenizer=tokenizer, pad_token_id=pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        betas=Config.adam_betas,
        eps=Config.adam_eps,
        weight_decay=Config.weight_decay,
    )

    total_steps = Config.num_epochs * math.ceil(len(loader) / Config.grad_accum_steps)
    scheduler = CosineScheduler(optimizer, warmup_steps=Config.warmup_steps, total_steps=total_steps)

    # Mixed precision autocast
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_fp16)

    model.train()

    global_step = 0
    accum_steps = 0
    running_loss = 0.0

    for epoch in range(Config.num_epochs):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.use_fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss = loss / Config.grad_accum_steps

            scaler.scale(loss).backward()

            running_loss += loss.item()
            accum_steps += 1

            if accum_steps % Config.grad_accum_steps == 0:
                # Gradient clipping
                if Config.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                avg_loss = running_loss
                running_loss = 0.0
                accum_steps = 0

                if global_step % 10 == 0:
                    current_lr = scheduler.optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch} | Step {global_step} | Loss {avg_loss:.4f} | LR {current_lr:.6f}")

                if global_step % Config.save_every_n_steps == 0:
                    save_lora(model, Config.output_dir, step=global_step)

    save_lora(model, Config.output_dir)
    print("Training complete.")

    model.eval()
    test_prompt = f"{Config.bos_token}{Config.system_prompt}\n" + Config.instruction_template.format(
        instruction="Write a haiku about GPUs."
    ) + "<|assistant|>\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    print("\nSample generation:\n", tokenizer.decode(gen[0], skip_special_tokens=True))
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
