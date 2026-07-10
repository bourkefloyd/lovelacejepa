"""Frozen probe heads: downstream consumers of the pretrained latent space.

The paper's TTA objective is computed *in the adapting encoder's own latent
space*. Any downstream consumer calibrated on the pretrained space - success
detectors, progress estimators, state readouts - is silently re-mapped when
the encoder adapts. These probes make that damage measurable on the public
benchmarks:

- ``state``: latent -> state readout (maze: agent pos; pushobj: block pose).
- ``progress``: (z_t, z_goal) -> negative task distance (regressor).
- ``success``: (z_t, z_goal) -> goal-reached logit (classifier).

Probes are trained ONCE on the frozen pretrained model's latents (offline
dataset), then never updated. At eval time each arm feeds the probes the
latents it actually planned with; if adaptation moved the encoder, the
probes' inputs are off-manifold and their AUC/error degrades.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import WorldModel


@dataclass
class ProbeConfig:
    env_kind: str = "pointmaze"  # "pointmaze" | "pushobj"
    hidden: int = 128
    pairs_per_traj: int = 10
    epochs: int = 6
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    # Success tolerances (match the envs' own task tolerances).
    maze_tol: float = 0.5
    push_pos_tol: float = 0.5
    push_ang_tol: float = 0.6


def state_targets(states: np.ndarray, env_kind: str) -> np.ndarray:
    """Regression targets for the state probe."""
    if env_kind == "pushobj":
        # Block pose as (x, y, sin, cos) - angle wraps, so regress the circle.
        x, y, ang = states[..., 2], states[..., 3], states[..., 4]
        return np.stack([x, y, np.sin(ang), np.cos(ang)], axis=-1)
    return states[..., :2]


def _push_ang_err(
    s_a: np.ndarray, s_b: np.ndarray, periods: np.ndarray | None
) -> np.ndarray:
    """Angle error, modulo each sample's shape-symmetry period when given."""
    d_ang = s_a[..., 4] - s_b[..., 4]
    if periods is None:
        return np.abs((d_ang + np.pi) % (2 * np.pi) - np.pi)
    d = np.mod(d_ang, periods)
    return np.minimum(d, periods - d)


def task_distance(
    s_a: np.ndarray, s_b: np.ndarray, env_kind: str, periods: np.ndarray | None = None
) -> np.ndarray:
    """Task distance between two state vectors (same metric as the envs)."""
    if env_kind == "pushobj":
        pos = np.linalg.norm(s_a[..., 2:4] - s_b[..., 2:4], axis=-1)
        return pos + _push_ang_err(s_a, s_b, periods)
    return np.linalg.norm(s_a[..., :2] - s_b[..., :2], axis=-1)


def success_label(
    s_a: np.ndarray, s_b: np.ndarray, cfg: ProbeConfig,
    periods: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.env_kind == "pushobj":
        pos = np.linalg.norm(s_a[..., 2:4] - s_b[..., 2:4], axis=-1)
        ang = _push_ang_err(s_a, s_b, periods)
        return ((pos < cfg.push_pos_tol) & (ang < cfg.push_ang_tol)).astype(np.float32)
    dist = np.linalg.norm(s_a[..., :2] - s_b[..., :2], axis=-1)
    return (dist < cfg.maze_tol).astype(np.float32)


class ProbeHeads(nn.Module):
    """Small frozen MLP heads on top of (pretrained) latents."""

    def __init__(self, latent_dim: int, state_dim: int, cfg: ProbeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        h = cfg.hidden
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, h), nn.GELU(), nn.Linear(h, state_dim)
        )
        self.pair_trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU()
        )
        self.progress_head = nn.Linear(h, 1)
        self.success_head = nn.Linear(h, 1)

    def forward(
        self, z: torch.Tensor, z_goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(state readout, progress = -distance estimate, success logit)."""
        state = self.state_head(z)
        h = self.pair_trunk(torch.cat([z, z_goal], dim=-1))
        return state, self.progress_head(h).squeeze(-1), self.success_head(h).squeeze(-1)


def _stacked(frames: np.ndarray, traj: int, t: int, stack: int) -> np.ndarray:
    t0 = max(t - stack + 1, 0)
    fs = list(frames[traj, t0 : t + 1])
    while len(fs) < stack:  # left-pad with the first frame
        fs.insert(0, fs[0])
    return np.concatenate(fs, axis=-1)


@torch.no_grad()
def _encode_batch(model: WorldModel, imgs: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    return model.encode(x)


def train_probes(
    model: WorldModel,
    npz_path: Path,
    cfg: ProbeConfig,
    device: torch.device,
) -> tuple[ProbeHeads, dict]:
    """Train probes on the FROZEN model's latents over the offline dataset."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = np.load(npz_path)
    frames, states = data["frames"], data["positions"]
    stack = int(data["frame_stack"])
    n, t_plus_1 = frames.shape[:2]

    # Sample (t, g) pairs per trajectory across the full gap range, so both
    # success-positive (small gap / block static) and far pairs appear.
    pairs: list[tuple[int, int, int]] = []
    for i in range(n):
        for _ in range(cfg.pairs_per_traj):
            t = int(rng.integers(0, t_plus_1))
            g = int(rng.integers(0, t_plus_1))
            pairs.append((i, t, g))
    idx = np.array(pairs)

    model.eval()
    z_t_list, z_g_list = [], []
    bs = 256
    for lo in range(0, len(idx), bs):
        chunk = idx[lo : lo + bs]
        obs_t = np.stack([_stacked(frames, i, t, stack) for i, t, _ in chunk])
        obs_g = np.stack([_stacked(frames, i, g, stack) for i, _, g in chunk])
        z_t_list.append(_encode_batch(model, obs_t, device))
        z_g_list.append(_encode_batch(model, obs_g, device))
    z_t = torch.cat(z_t_list)
    z_g = torch.cat(z_g_list)

    s_t = states[idx[:, 0], idx[:, 1]]
    s_g = states[idx[:, 0], idx[:, 2]]
    periods = None
    if cfg.env_kind == "pushobj" and "shapes" in data and "shape_ids" in data:
        from .pushobj import SHAPE_SYMMETRY

        shape_names = [str(s) for s in data["shapes"]]
        traj_periods = np.array(
            [SHAPE_SYMMETRY[shape_names[i]] for i in data["shape_ids"]]
        )
        periods = traj_periods[idx[:, 0]]
    y_state = torch.from_numpy(state_targets(s_t, cfg.env_kind)).float().to(device)
    y_prog = torch.from_numpy(
        -task_distance(s_t, s_g, cfg.env_kind, periods).astype(np.float32)
    ).to(device)
    y_succ = torch.from_numpy(success_label(s_t, s_g, cfg, periods)).to(device)

    heads = ProbeHeads(z_t.shape[-1], y_state.shape[-1], cfg).to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr=cfg.lr)
    n_pairs = len(idx)
    pos_frac = float(y_succ.mean().cpu())
    history = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_pairs, device=device)
        ep_loss = 0.0
        for lo in range(0, n_pairs, cfg.batch_size):
            sel = perm[lo : lo + cfg.batch_size]
            state, prog, succ = heads(z_t[sel], z_g[sel])
            loss = (
                F.mse_loss(state, y_state[sel])
                + F.mse_loss(prog, y_prog[sel])
                + F.binary_cross_entropy_with_logits(succ, y_succ[sel])
            )
            loss_val = float(loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ep_loss += loss_val
        history.append(round(ep_loss / max(1, n_pairs // cfg.batch_size), 5))

    # Final-fit diagnostics on the training pairs (probes are consumers, not
    # the object of study; fit quality just needs to be "good enough to break").
    with torch.no_grad():
        state, prog, succ = heads(z_t, z_g)
        state_mse = float(F.mse_loss(state, y_state).cpu())
        prog_corr = float(
            torch.corrcoef(torch.stack([prog, y_prog]))[0, 1].cpu()
        )
        succ_auc = auc_score(succ.cpu().numpy(), y_succ.cpu().numpy())
    info = {
        "pairs": n_pairs,
        "success_pos_frac": round(pos_frac, 4),
        "loss_history": history,
        "fit_state_mse": round(state_mse, 4),
        "fit_progress_corr": round(prog_corr, 4),
        "fit_success_auc": round(succ_auc, 4),
    }
    heads.eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return heads, info


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via rank statistic (no sklearn dependency)."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # Midranks for ties.
    allscores = np.concatenate([neg, pos])
    sorted_scores = allscores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[len(neg) :].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_probes(heads: ProbeHeads, path: Path, info: dict | None = None) -> None:
    torch.save(
        {
            "config": asdict(heads.cfg),
            "latent_dim": heads.state_head[0].in_features,
            "state_dim": heads.state_dim,
            "state_dict": heads.state_dict(),
            "info": info or {},
        },
        path,
    )


def load_probes(path: Path, device) -> ProbeHeads:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    heads = ProbeHeads(ckpt["latent_dim"], ckpt["state_dim"], ProbeConfig(**ckpt["config"]))
    heads.load_state_dict(ckpt["state_dict"])
    heads.to(device)
    heads.eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return heads


__all__ = [
    "ProbeConfig",
    "ProbeHeads",
    "auc_score",
    "load_probes",
    "save_probes",
    "state_targets",
    "success_label",
    "task_distance",
    "train_probes",
]
