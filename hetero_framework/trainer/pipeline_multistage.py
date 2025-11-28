"""
Multi-Stage Pipeline Trainer (Coordinator)

Implements a star-orchestrated multi-hop pipeline across N workers with an
explicit shard plan. The coordinator relays activations and gradients between
stages so workers do not need to communicate with each other.

Protocol commands are raw int tensors defined in hetero_framework.core.commands.
"""

from __future__ import annotations

import contextlib
import os
import math
import socket
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core.commands import (
    CMD_BWD,
    CMD_FWD,
    CMD_FWD_LOSS,
    CMD_HAS_CKPT,
    CMD_LOAD_CKPT,
    CMD_OPT_STEP,
    CMD_SAVE_CKPT,
)
from ..core.transport import recv_tensor, send_tensor
from ..visualizer.graph_gen import GraphGenerator
from ..core.dal import DeviceAbstractionLayer


class MultiStagePipelineTrainer:
    def __init__(
        self,
        local_shard: nn.Module,
        workers: list[dict[str, Any]],
        shard_plan: list[dict[str, Any]],
        dataloader: DataLoader,
        batch_size: int,
        grad_accum_steps: int,
        lr: float,
        weight_decay: float,
        adam_eps: float,
        adam_betas: tuple[float, float],
        num_epochs: int,
        warmup_steps: int,
        clip_grad_norm: float | None,
        use_fp16: bool,
        device: torch.device | None = None,
        diagram_basename: str = "architecture",
        save_every_n_steps: int | None = None,
        checkpoint_dir: str | None = None,
        resume_step: int | None = None,
    ):
        self.local_shard = local_shard
        self.workers_cfg = workers
        self.shard_plan = shard_plan
        self.loader = dataloader
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.clip_grad_norm = clip_grad_norm
        dev_str = None
        if device is None:
            dev_str = "auto"
        elif isinstance(device, torch.device):
            dev_str = device.type
        elif isinstance(device, str):
            dev_str = device
        else:
            dev_str = "auto"
        self.dal = DeviceAbstractionLayer(dev_str)
        self.device = self.dal.device
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")

        self.local_shard = self.local_shard.to(self.device)

        # Build ordered remote stages from shard_plan (exclude local)
        self.remote_stage_ids: list[Any] = [s["id"] for s in self.shard_plan if s["id"] != "local"]
        # Map id -> worker cfg
        id_to_worker = {w["id"]: w for w in self.workers_cfg}
        self.remote_workers: list[dict[str, Any]] = [
            id_to_worker[sid] for sid in self.remote_stage_ids
        ]

        self.sockets: list[socket.socket] = []
        self.diagram_basename = diagram_basename
        self.save_every_n_steps = save_every_n_steps or 0
        self.checkpoint_dir = checkpoint_dir
        self.resume_step = resume_step

    def _connect_all(self) -> None:
        self.sockets = []
        for w in self.remote_workers:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((w["ip"], w.get("port", 9999)))
            self.sockets.append(s)

    def _close_all(self) -> None:
        for s in self.sockets:
            with contextlib.suppress(Exception):
                s.close()
        self.sockets = []

    def _generate_diagram(self) -> None:
        graph = GraphGenerator()

        local_stage = next((s for s in self.shard_plan if s.get("id") == "local"), None)
        remote_stages = [s for s in self.shard_plan if s.get("id") != "local"]

        if local_stage is not None:
            l_start, l_end = int(local_stage["start"]), int(local_stage["end"])
            local_layers = ["Embedding", f"Layers {l_start}-{l_end-1}"]
        else:
            local_layers = ["LocalShard"]

        graph.add_node(
            "local",
            "Localhost (Coordinator)",
            "local",
            local_layers,
            device_info=self.dal.device_name,
        )

        # Remote nodes: show explicit layer ranges; last stage mentions Norm+LM Head
        prev = "local"
        for idx, (w, stg) in enumerate(zip(self.remote_workers, remote_stages), start=1):
            nid = f"worker{idx}"
            r_start, r_end = int(stg["start"]), int(stg["end"])
            layers = [f"Layers {r_start}-{r_end-1}"]
            if stg.get("is_last", False):
                layers.append("Norm + LM Head")
            graph.add_node(
                nid,
                f"Worker {idx}",
                "remote",
                layers,
                device_info=f"{w['ip']}:{w.get('port',9999)}",
            )
            graph.add_edge(prev, nid, label="Ethernet (TCP)", edge_type="network")
            prev = nid
        try:
            graph.generate_graphviz(self.diagram_basename, format="png")
        except Exception as e:
            print(f"Warning: could not generate diagram: {e}")

    def train(self) -> None:
        self._connect_all()
        try:
            self._train_inner()
        finally:
            self._close_all()

    def _train_inner(self) -> None:
        optimizer = torch.optim.AdamW(
            self.local_shard.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        total_steps = self.num_epochs * math.ceil(len(self.loader) / self.grad_accum_steps)
        scheduler = _CosineScheduler(
            optimizer, warmup_steps=self.warmup_steps, total_steps=total_steps
        )

        # Optionally resume
        start_global_step = 0
        if self.checkpoint_dir:
            if self.resume_step is not None:
                try:
                    step = self._resolve_resume_step(self.resume_step)
                    if step is None:
                        print(
                            "No common checkpoint found across coordinator and workers; starting fresh."
                        )
                    else:
                        self._load_checkpoint(step, optimizer, scheduler)
                        print(f"Resumed from step {step}")
                        start_global_step = step
                except Exception as e:
                    print(f"Warning: resume failed ({e}); starting fresh.")

        print("=" * 70)
        print("TRAINING")
        print("=" * 70)
        print(f"Steps: {total_steps} | Batch: {self.batch_size} | Accum: {self.grad_accum_steps}\n")

        self._generate_diagram()
        t0 = time.perf_counter()

        global_step = start_global_step
        accum = 0
        running_loss = 0.0

        # Identify last stage socket index (last remote)
        last_sock_idx = len(self.sockets) - 1

        done = False
        for epoch in range(self.num_epochs):
            if done:
                break
            for batch in self.loader:
                input_ids = self.dal.to_device(batch["input_ids"])
                attention_mask = self.dal.to_device(batch["attention_mask"])
                labels = batch["labels"]

                # Local forward
                if self.use_fp16:
                    with torch.amp.autocast("cuda"):
                        x = self.local_shard(input_ids, attention_mask)
                else:
                    x = self.local_shard(input_ids, attention_mask)

                boundary_inputs: list[torch.Tensor] = []
                for i, sock in enumerate(self.sockets):
                    is_last = i == last_sock_idx
                    if not is_last:
                        # FWD
                        boundary_inputs.append(x.detach().cpu())
                        send_tensor(sock, torch.tensor(CMD_FWD, dtype=torch.int32))
                        send_tensor(sock, x.detach().cpu())
                        send_tensor(sock, attention_mask.cpu())
                        x = recv_tensor(sock)
                        if x is None:
                            raise RuntimeError("Connection lost during forward")
                        x = x.to(self.device)
                    else:
                        # LAST: FWD_LOSS
                        send_tensor(sock, torch.tensor(CMD_FWD_LOSS, dtype=torch.int32))
                        send_tensor(sock, x.detach().cpu())
                        send_tensor(sock, attention_mask.cpu())
                        send_tensor(sock, labels)
                        loss = recv_tensor(sock)
                        if loss is None:
                            raise RuntimeError("Connection lost receiving loss from last worker")
                        upstream = recv_tensor(sock)
                        if upstream is None:
                            raise RuntimeError(
                                "Connection lost receiving gradient from last worker"
                            )
                        loss = loss.to(self.device)
                        upstream = upstream.to(self.device)

                # Backward chain: from last-1 down to 0
                for i in range(last_sock_idx - 1, -1, -1):
                    sock = self.sockets[i]
                    xi = boundary_inputs[i]
                    send_tensor(sock, torch.tensor(CMD_BWD, dtype=torch.int32))
                    send_tensor(sock, xi)
                    send_tensor(sock, attention_mask.cpu())
                    send_tensor(sock, upstream.detach().cpu())
                    upstream = recv_tensor(sock)
                    if upstream is None:
                        raise RuntimeError("Connection lost receiving upstream gradient")
                    upstream = upstream.to(self.device)

                # Accumulate and step on coordinator local shard
                upstream = upstream / self.grad_accum_steps
                x.backward(upstream)
                running_loss += float(loss.item())
                accum += 1
                if accum % self.grad_accum_steps == 0:
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.local_shard.parameters(), self.clip_grad_norm
                        )
                    # Signal remote workers to optimizer.step() and zero_grad()
                    for sock in self.sockets:
                        send_tensor(sock, torch.tensor(CMD_OPT_STEP, dtype=torch.int32))

                    for sock in self.sockets:
                        _ = recv_tensor(sock)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    avg_loss = running_loss / self.grad_accum_steps
                    running_loss = 0.0
                    accum = 0
                    lr = scheduler.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch} | Step {global_step}/{total_steps} | Loss {avg_loss:.4f} | LR {lr:.6f}"
                    )

                    # Periodic checkpointing
                    if self.checkpoint_dir and self.save_every_n_steps:
                        if global_step % self.save_every_n_steps == 0:
                            with contextlib.suppress(Exception):
                                self._save_checkpoint(global_step, optimizer, scheduler)

                    if global_step >= total_steps:
                        done = True
                        break

        t1 = time.perf_counter()
        print(f"\nTotal training time: {t1 - t0:.2f} seconds")

    def _ckpt_paths(self, step: int) -> tuple[str, str]:
        base = (
            os.path.join(self.checkpoint_dir, "checkpoints")
            if self.checkpoint_dir
            else "./checkpoints"
        )
        coord_dir = os.path.join(base, "coordinator")
        os.makedirs(coord_dir, exist_ok=True)
        coord_path = os.path.join(coord_dir, f"step_{step:06d}.pt")
        return base, coord_path

    def _save_checkpoint(
        self, step: int, optimizer: torch.optim.Optimizer, scheduler: "_CosineScheduler"
    ) -> None:
        base, coord_path = self._ckpt_paths(step)
        state = {
            "model": self.local_shard.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "shard_plan": self.shard_plan,
        }
        torch.save(state, coord_path)

        for sock in self.sockets:
            send_tensor(sock, torch.tensor(CMD_SAVE_CKPT, dtype=torch.int32))
            send_tensor(sock, torch.tensor(step, dtype=torch.int32))
        acks = []
        for sock in self.sockets:
            ack = recv_tensor(sock)
            acks.append(int(ack.item()) if ack is not None else 0)
        ok = sum(acks) == len(self.sockets)
        print(f"Checkpoint step {step} saved | workers ok: {ok}")

    def _resolve_resume_step(self, target_step: int | None) -> int | None:
        """Find a common checkpoint step present on coordinator and all workers.

        If target_step is provided, try it first, else try latest available locally.
        Falls back to lower steps until a common step is found or None.
        """
        base, _ = self._ckpt_paths(0)
        coord_dir = os.path.join(base, "coordinator")
        local_steps: list[int] = []
        if os.path.isdir(coord_dir):
            for name in os.listdir(coord_dir):
                if name.startswith("step_") and name.endswith(".pt"):
                    try:
                        local_steps.append(int(name.replace("step_", "").replace(".pt", "")))
                    except Exception:
                        pass
        if not local_steps:
            return None
        local_steps = sorted(local_steps, reverse=True)
        candidates = [target_step] if target_step is not None else []
        for s in local_steps:
            if s not in candidates:
                candidates.append(s)

        for step in candidates:
            if step is None:
                continue
            # Check each worker has this step without mutating their state
            ok_all = True
            for sock in self.sockets:
                send_tensor(sock, torch.tensor(CMD_HAS_CKPT, dtype=torch.int32))
                send_tensor(sock, torch.tensor(step, dtype=torch.int32))
            for sock in self.sockets:
                ack = recv_tensor(sock)
                if ack is None or int(ack.item()) != 1:
                    ok_all = False
            if ok_all:
                return step
        return None

    def _load_checkpoint(
        self, step: int, optimizer: torch.optim.Optimizer, scheduler: "_CosineScheduler"
    ) -> None:
        # Load coordinator
        _, coord_path = self._ckpt_paths(step)
        if not os.path.exists(coord_path):
            raise FileNotFoundError(f"Coordinator checkpoint not found: {coord_path}")
        state = torch.load(coord_path, map_location=self.device)
        # Be tolerant of bitsandbytes/PEFT buffers that may drift between versions
        saved = state.get("model", {})
        if isinstance(saved, dict):
            current = self.local_shard.state_dict()
            filtered = {
                k: v
                for k, v in saved.items()
                if k in current and getattr(current[k], "shape", None) == getattr(v, "shape", None)
            }
            missing = [k for k in current.keys() if k not in filtered]
            unexpected = [k for k in saved.keys() if k not in current]
            if unexpected:
                print(
                    f"Note: ignoring {len(unexpected)} unexpected keys when loading coordinator shard"
                )
            if missing:
                print(
                    f"Note: {len(missing)} keys missing in checkpoint for coordinator shard; loading partial state"
                )
            self.local_shard.load_state_dict(filtered, strict=False)
        else:
            # Fallback
            self.local_shard.load_state_dict(saved, strict=False)
        optimizer.load_state_dict(state.get("optimizer", {}))
        sched_state = state.get("scheduler", None)
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

        for sock in self.sockets:
            send_tensor(sock, torch.tensor(CMD_LOAD_CKPT, dtype=torch.int32))
            send_tensor(sock, torch.tensor(step, dtype=torch.int32))
        for sock in self.sockets:
            ack = recv_tensor(sock)
            if ack is None or int(ack.item()) != 1:
                raise RuntimeError("Worker failed to load checkpoint")


class _CosineScheduler:
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

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "step_count": self.step_count,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: dict) -> None:
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.min_lr_ratio = float(state.get("min_lr_ratio", self.min_lr_ratio))
        self.step_count = int(state.get("step_count", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        # Re-apply current LR immediately based on restored step_count
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
