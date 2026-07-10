"""MPC trajectory optimizers over the latent world model (paper eq. 3).

Both planners minimize sum_k alpha_k * ||z_hat_{t+k} - z_g||^2 over an action
sequence, matching the paper's two optimizer choices:

- ``cem_plan``: Cross-Entropy Method (sampling-based).
- ``gd_plan``: gradient descent through the latent rollout (Adam).

Receding horizon: the caller executes the first ``execute_actions`` actions
and replans; warm-starting shifts the previous solution forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import WorldModel


@dataclass
class PlannerConfig:
    kind: str = "cem"  # "cem" | "gd"
    horizon: int = 10
    execute_actions: int = 2
    final_weight: float = 3.0  # alpha_H (intermediate steps get alpha_k = 1)
    # CEM
    cem_samples: int = 96
    cem_iters: int = 5
    cem_elite: int = 12
    cem_init_std: float = 0.7
    # GD
    gd_steps: int = 40
    gd_lr: float = 0.2


def _step_weights(horizon: int, final_weight: float, device) -> torch.Tensor:
    w = torch.ones(horizon, device=device)
    w[-1] = final_weight
    return w


def plan_cost(
    model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor,
    final_weight: float,
) -> torch.Tensor:
    """Latent goal-reaching cost for a batch of action sequences (B, H, 2)."""
    b, h = actions.shape[:2]
    z_hat = model.rollout(z0.expand(b, -1), actions)  # (B, H, D)
    dists = ((z_hat - z_goal.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, H)
    return (dists * _step_weights(h, final_weight, actions.device)).sum(dim=1)


@torch.no_grad()
def cem_plan(
    model: WorldModel,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    cfg: PlannerConfig,
    init_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """CEM over action sequences; returns the best (H, 2) plan."""
    device = z0.device
    h = cfg.horizon
    mean = (
        init_mean.clone()
        if init_mean is not None
        else torch.zeros(h, 2, device=device)
    )
    std = torch.full((h, 2), cfg.cem_init_std, device=device)
    best_actions = mean.clone()
    best_cost = torch.tensor(float("inf"), device=device)
    for _ in range(cfg.cem_iters):
        samples = mean + std * torch.randn(cfg.cem_samples, h, 2, device=device)
        samples = samples.clamp(-1.0, 1.0)
        samples[0] = mean.clamp(-1.0, 1.0)  # keep the incumbent in the pool
        costs = plan_cost(model, z0, z_goal, samples, cfg.final_weight)
        elite_idx = costs.topk(cfg.cem_elite, largest=False).indices
        elite = samples[elite_idx]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False) + 1e-4
        if costs[elite_idx[0]] < best_cost:
            best_cost = costs[elite_idx[0]]
            best_actions = samples[elite_idx[0]].clone()
    return best_actions


def gd_plan(
    model: WorldModel,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    cfg: PlannerConfig,
    init_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient-descent planning through the latent rollout."""
    device = z0.device
    if init_mean is not None:
        # Warm start lives in action space; map into the tanh pre-image.
        init = torch.atanh(init_mean.clamp(-0.999, 0.999))
    else:
        init = torch.zeros(cfg.horizon, 2, device=device)
    actions = init.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([actions], lr=cfg.gd_lr)
    z0 = z0.detach()
    z_goal = z_goal.detach()
    for _ in range(cfg.gd_steps):
        cost = plan_cost(
            model, z0, z_goal, torch.tanh(actions).unsqueeze(0), cfg.final_weight
        ).squeeze(0)
        optimizer.zero_grad(set_to_none=True)
        cost.backward()
        optimizer.step()
    return torch.tanh(actions).detach()


def plan(
    model: WorldModel,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    cfg: PlannerConfig,
    init_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.kind == "cem":
        return cem_plan(model, z0, z_goal, cfg, init_mean)
    if cfg.kind == "gd":
        return gd_plan(model, z0, z_goal, cfg, init_mean)
    raise ValueError(f"unknown planner {cfg.kind!r}")


def shift_plan(prev: torch.Tensor, executed: int) -> torch.Tensor:
    """Warm start: drop executed actions, pad the tail with zeros."""
    pad = torch.zeros(executed, 2, device=prev.device)
    return torch.cat([prev[executed:], pad], dim=0)


__all__ = ["PlannerConfig", "cem_plan", "gd_plan", "plan", "plan_cost", "shift_plan"]
