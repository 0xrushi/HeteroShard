"""
Relay Gradient Worker (multi-stage)

One process per GPU. Handles three commands:
- CMD_FWD:    forward only, returns activations
- CMD_FWD_LOSS: forward + CE loss, returns (loss, dL/dx)
- CMD_BWD:    recompute forward, backprop with upstream grad, returns dL/dx
"""

from __future__ import annotations

import contextlib
import socket
import os
import torch
import torch.nn as nn

from ..core.commands import (
    CMD_BWD,
    CMD_FWD,
    CMD_FWD_LOSS,
    CMD_OPT_STEP,
    CMD_SAVE_CKPT,
    CMD_LOAD_CKPT,
    CMD_REPORT_STATE,
    CMD_HAS_CKPT,
)
from ..core.transport import recv_tensor, send_tensor


class RelayGradientWorker:
    def __init__(
        self,
        model_shard: nn.Module,
        listen_ip: str = "0.0.0.0",
        listen_port: int = 9999,
        use_fp16: bool = True,
        lr: float = 2e-4,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = 1.0,
        peft_model: object | None = None,
        adapter_save_dir: str | None = None,
        worker_id: int | None = None,
    ):
        self.model_shard = model_shard
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        self.server_sock: socket.socket | None = None
        self.client_sock: socket.socket | None = None

        self.model_shard = self.model_shard.to(self.device).train()
        self.optimizer = torch.optim.AdamW(
            self.model_shard.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.clip_grad_norm = clip_grad_norm
        self._opt_steps = 0
        self._peft_model = peft_model
        self._adapter_save_dir = adapter_save_dir
        self._worker_id = worker_id
        # Optional checkpoint directory (defaults to adapter_save_dir if provided)
        self._ckpt_dir = adapter_save_dir

    def start(self) -> None:
        print("=" * 60)
        print("RELAY GRADIENT WORKER")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Listening on {self.listen_ip}:{self.listen_port}")

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.listen_ip, self.listen_port))
        self.server_sock.listen(1)

        self.client_sock, addr = self.server_sock.accept()
        print(f"Connected to coordinator at {addr}\n")
        self._loop()

    def _loop(self) -> None:
        step = 0
        while True:
            try:
                cmd_t = recv_tensor(self.client_sock)
                if cmd_t is None:
                    print("Connection closed by coordinator.")
                    break
                cmd = int(cmd_t.item())

                if cmd == CMD_FWD:
                    x = recv_tensor(self.client_sock)
                    attn = recv_tensor(self.client_sock)
                    x = x.to(self.device)
                    attn = attn.to(self.device) if attn is not None else None
                    with torch.amp.autocast("cuda", enabled=self.use_fp16):
                        y = self.model_shard(x, attn, labels=None)
                    send_tensor(self.client_sock, y.detach().cpu())
                    step += 1
                    continue

                if cmd == CMD_FWD_LOSS:
                    x = recv_tensor(self.client_sock)
                    attn = recv_tensor(self.client_sock)
                    labels = recv_tensor(self.client_sock)
                    x = x.to(self.device).detach()
                    x.requires_grad_(True)
                    x.retain_grad()
                    attn = attn.to(self.device) if attn is not None else None
                    labels = labels.to(self.device) if labels is not None else None
                    with torch.amp.autocast("cuda", enabled=self.use_fp16):
                        loss = self.model_shard(x, attn, labels)
                    if not torch.isfinite(loss):
                        grad_x = torch.zeros_like(x)
                    else:
                        # Accumulate gradients across micro-batches
                        loss.backward()
                        grad_x = x.grad.detach()
                    grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=1e4, neginf=-1e4)
                    send_tensor(self.client_sock, loss.detach().cpu())
                    send_tensor(self.client_sock, grad_x.cpu())
                    print(f"[Step {step}] Loss: {loss.item():.4f}")
                    step += 1
                    continue

                if cmd == CMD_BWD:
                    x = recv_tensor(self.client_sock)
                    attn = recv_tensor(self.client_sock)
                    upstream = recv_tensor(self.client_sock)
                    x = x.to(self.device).detach()
                    x.requires_grad_(True)
                    x.retain_grad()
                    attn = attn.to(self.device) if attn is not None else None
                    upstream = upstream.to(self.device)
                    with torch.amp.autocast("cuda", enabled=self.use_fp16):
                        y = self.model_shard(x, attn, labels=None)
                    # Accumulate gradients across micro-batches
                    y.backward(upstream)
                    grad_x = x.grad.detach()
                    grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=1e4, neginf=-1e4)
                    send_tensor(self.client_sock, grad_x.cpu())
                    continue

                if cmd == CMD_OPT_STEP:
                    # Gradient sync boundary: step and zero grads
                    if self.clip_grad_norm is not None:
                        with contextlib.suppress(Exception):
                            torch.nn.utils.clip_grad_norm_(
                                self.model_shard.parameters(), self.clip_grad_norm
                            )
                    try:
                        self.optimizer.step()
                    finally:
                        self.optimizer.zero_grad(set_to_none=True)
                    self._opt_steps += 1
                    # Save LoRA adapters by default if peft_model provided
                    if self._peft_model and self._adapter_save_dir:
                        subdir = self._adapter_save_dir
                        if self._worker_id is not None:
                            subdir = f"{subdir}/worker{self._worker_id}"
                        with contextlib.suppress(Exception):
                            self._peft_model.save_pretrained(subdir)
                    send_tensor(self.client_sock, torch.tensor(self._opt_steps, dtype=torch.int32))
                    continue

                if cmd == CMD_SAVE_CKPT:
                    # Expect: step (int tensor)
                    step_t = recv_tensor(self.client_sock)
                    if step_t is None:
                        print("Missing step for CMD_SAVE_CKPT")
                        break
                    target_step = int(step_t.item())
                    # Build path
                    if self._ckpt_dir is None:
                        # If no directory set, acknowledge but skip saving to avoid crash
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                        continue
                    # Structure: {ckpt_dir}/checkpoints/worker{ID}/step_{N}.pt
                    subdir = os.path.join(self._ckpt_dir, "checkpoints")
                    if self._worker_id is not None:
                        subdir = os.path.join(subdir, f"worker{self._worker_id}")
                    os.makedirs(subdir, exist_ok=True)
                    ckpt_path = os.path.join(subdir, f"step_{target_step:06d}.pt")
                    state = {
                        "model": self.model_shard.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "opt_steps": self._opt_steps,
                        "worker_id": self._worker_id,
                    }
                    try:
                        torch.save(state, ckpt_path)
                        send_tensor(self.client_sock, torch.tensor(1, dtype=torch.int32))
                    except Exception as e:
                        print(f"Failed to save worker ckpt: {e}")
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                    continue

                if cmd == CMD_LOAD_CKPT:
                    # Expect: step (int tensor)
                    step_t = recv_tensor(self.client_sock)
                    if step_t is None:
                        print("Missing step for CMD_LOAD_CKPT")
                        break
                    target_step = int(step_t.item())
                    if self._ckpt_dir is None:
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                        continue

                    subdir = os.path.join(self._ckpt_dir, "checkpoints")
                    if self._worker_id is not None:
                        subdir = os.path.join(subdir, f"worker{self._worker_id}")
                    ckpt_path = os.path.join(subdir, f"step_{target_step:06d}.pt")
                    try:
                        state = torch.load(ckpt_path, map_location=self.device)
                        saved = state.get("model", {})
                        if isinstance(saved, dict):
                            current = self.model_shard.state_dict()
                            filtered = {
                                k: v
                                for k, v in saved.items()
                                if k in current
                                and getattr(current[k], "shape", None) == getattr(v, "shape", None)
                            }
                            missing = [k for k in current.keys() if k not in filtered]
                            unexpected = [k for k in saved.keys() if k not in current]
                            if unexpected:
                                print(
                                    f"Note: ignoring {len(unexpected)} unexpected keys when loading worker shard"
                                )
                            if missing:
                                print(
                                    f"Note: {len(missing)} keys missing in checkpoint for worker shard; loading partial state"
                                )
                            self.model_shard.load_state_dict(filtered, strict=False)
                        else:
                            self.model_shard.load_state_dict(saved, strict=False)
                        opt_state = state.get("optimizer", None)
                        if opt_state:
                            try:
                                self.optimizer.load_state_dict(opt_state)
                            except Exception as e:
                                print(f"Note: skipping optimizer state load: {e}")
                        self._opt_steps = int(state.get("opt_steps", 0))
                        print(f"Loaded checkpoint step {target_step} (opt_steps={self._opt_steps})")
                        send_tensor(self.client_sock, torch.tensor(1, dtype=torch.int32))
                    except Exception as e:
                        print(f"Failed to load worker ckpt: {e}")
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                    continue

                if cmd == CMD_REPORT_STATE:
                    # Respond with current optimizer step as int
                    send_tensor(self.client_sock, torch.tensor(self._opt_steps, dtype=torch.int32))
                    continue

                if cmd == CMD_HAS_CKPT:
                    step_t = recv_tensor(self.client_sock)
                    if step_t is None:
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                        continue
                    target_step = int(step_t.item())
                    if self._ckpt_dir is None:
                        send_tensor(self.client_sock, torch.tensor(0, dtype=torch.int32))
                        continue
                    subdir = os.path.join(self._ckpt_dir, "checkpoints")
                    if self._worker_id is not None:
                        subdir = os.path.join(subdir, f"worker{self._worker_id}")
                    ckpt_path = os.path.join(subdir, f"step_{target_step:06d}.pt")
                    exists = 1 if os.path.exists(ckpt_path) else 0
                    send_tensor(self.client_sock, torch.tensor(exists, dtype=torch.int32))
                    continue

                print(f"Unknown command: {cmd}")
                break

            except Exception as e:
                print(f"Worker error: {e}")
                break

    def stop(self) -> None:
        if self.client_sock:
            with contextlib.suppress(Exception):
                self.client_sock.close()
        if self.server_sock:
            with contextlib.suppress(Exception):
                self.server_sock.close()


def run_worker(
    model_shard: nn.Module,
    listen_ip: str = "0.0.0.0",
    listen_port: int = 9999,
    use_fp16: bool = True,
    lr: float = 2e-4,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    clip_grad_norm: float | None = 1.0,
    peft_model: object | None = None,
    adapter_save_dir: str | None = None,
    worker_id: int | None = None,
) -> None:
    w = RelayGradientWorker(
        model_shard,
        listen_ip,
        listen_port,
        use_fp16,
        lr,
        betas,
        eps,
        weight_decay,
        clip_grad_norm,
        peft_model,
        adapter_save_dir,
        worker_id,
    )
    try:
        w.start()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        w.stop()
