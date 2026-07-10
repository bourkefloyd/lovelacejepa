"""Offline JEPA training loop (MPS-safe).

MPS hygiene applied throughout (hard-won on this machine):
- one throwaway warmup forward before the loop (first-batch reductions can
  return garbage on a fresh MPS graph);
- all metric scalars captured as Python floats BEFORE ``backward()``;
- non-finite losses skip the step instead of poisoning the weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TransitionWindows
from .model import ModelConfig, WorldModel, jepa_loss, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 64
    lr: float = 3e-4
    pred_steps: int = 2
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    motion_weight: float = 1.0
    motion_margin: float = 0.25
    max_trajectories: int | None = None
    seed: int = 0
    log_every: int = 100


def train(
    data_path: Path,
    out_path: Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict:
    torch.manual_seed(train_cfg.seed)
    dataset = TransitionWindows(
        data_path,
        pred_steps=train_cfg.pred_steps,
        max_trajectories=train_cfg.max_trajectories,
    )
    loader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True
    )
    model = WorldModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # MPS warmup pass (throwaway).
    batch = next(iter(loader))
    with torch.no_grad():
        _ = model.encode(batch["obs"][:4].to(device))

    history: list[dict] = []
    step = 0
    t0 = time.time()
    for epoch in range(train_cfg.epochs):
        for batch in loader:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)  # (B, K, C, H, W)
            actions = batch["actions"].to(device)  # (B, K, 2)
            b, k = actions.shape[:2]

            z = model.encode(obs)
            with torch.no_grad():
                z_targets = model.encode(next_obs.reshape(b * k, *next_obs.shape[2:]))
                z_targets = z_targets.reshape(b, k, -1)

            z_preds = model.rollout(z, actions)  # (B, K, D)
            loss, metrics = jepa_loss(
                z_preds,
                z_targets,
                z,  # encoder branch, with grad: anti-collapse must reach it
                variance_weight=train_cfg.variance_weight,
                covariance_weight=train_cfg.covariance_weight,
                motion_weight=train_cfg.motion_weight,
                motion_margin=train_cfg.motion_margin,
            )
            loss_val = float(loss.detach().cpu())  # capture BEFORE backward
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % train_cfg.log_every == 0:
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_val, 5),
                    **{k_: round(v, 5) for k_, v in metrics.items()},
                    "elapsed_s": round(time.time() - t0, 1),
                }
                history.append(row)
                print(
                    f"[train] step={row['step']} epoch={epoch} "
                    f"loss={row['loss']:.4f} pred={row['pred_loss']:.4f} "
                    f"embed_std={row['embed_std']:.3f}",
                    flush=True,
                )
            step += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        out_path,
        extra={
            "train_config": asdict(train_cfg),
            "data_path": str(data_path),
            "history": history,
        },
    )
    log_path = out_path.with_suffix(".train.json")
    log_path.write_text(json.dumps({"history": history}, indent=2))
    return {
        "checkpoint": str(out_path),
        "steps": step,
        "final": history[-1] if history else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


__all__ = ["TrainConfig", "train"]
