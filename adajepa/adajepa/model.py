"""Miniature JEPA world model (encoder + action encoder + predictor).

Follows the paper's eq. (1): z_t = E_s(o_t), u_t = E_a(a_t),
z_{t+1} = f(z_t, u_t), trained with the latent prediction objective of eq. (2)
using a stop-gradient target plus VICReg-style variance/covariance insurance
(the paper notes either stabilizer is admissible).

The module names deliberately expose the paper's adaptation targets:

- ``encoder.stem`` / ``encoder.proj``  -> "encfirst" / "enclast"
- ``predictor.in_proj``+``blocks[0]``  -> "predfirst"
- ``predictor.blocks[-1]``+``norm``+``head`` -> "predlast"
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    img_size: int = 64
    frame_stack: int = 2
    latent_dim: int = 128
    action_dim: int = 2
    action_embed_dim: int = 64
    predictor_hidden: int = 256
    predictor_blocks: int = 3


class Encoder(nn.Module):
    """Small CNN over channel-stacked frames -> LayerNorm'd latent."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_ch = 3 * cfg.frame_stack
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.GELU(),  # 64 -> 32
        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GELU(),  # 32 -> 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GELU(),  # 16 -> 8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.GELU(),  # 8 -> 4
        )
        flat = 128 * (cfg.img_size // 16) ** 2
        # "enclast" adaptation target: the projection head.
        self.proj = nn.Sequential(
            nn.Linear(flat, 256),
            nn.GELU(),
            nn.Linear(256, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(x))
        return self.proj(h.flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class Predictor(nn.Module):
    """Residual-MLP latent dynamics: (z_t, u_t) -> z_{t+1} (predicts a delta)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(cfg.latent_dim + cfg.action_embed_dim, cfg.latent_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(cfg.latent_dim, cfg.predictor_hidden)
            for _ in range(cfg.predictor_blocks)
        )
        self.norm = nn.LayerNorm(cfg.latent_dim)
        self.head = nn.Linear(cfg.latent_dim, cfg.latent_dim)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([z, u], dim=-1))
        for block in self.blocks:
            h = block(h)
        return z + self.head(self.norm(h))


class WorldModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = Encoder(self.config)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.config.action_dim, self.config.action_embed_dim),
            nn.GELU(),
            nn.Linear(self.config.action_embed_dim, self.config.action_embed_dim),
        )
        self.predictor = Predictor(self.config)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3*frame_stack, H, W) float in [0, 1]."""
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One latent step. action: (B, 2) in [-1, 1]."""
        return self.predictor(z, self.action_encoder(action))

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll the predictor over an action sequence.

        z0: (B, D); actions: (B, H, 2). Returns (B, H, D) predicted latents.
        """
        z = z0
        outs = []
        for k in range(actions.shape[1]):
            z = self.predict(z, actions[:, k])
            outs.append(z)
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _var_cov(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = flat.shape[-1]
    std = flat.std(dim=0, unbiased=False)
    var_loss = F.relu(1.0 - std).mean()
    centered = flat - flat.mean(dim=0)
    n = max(flat.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag**2).sum() / d
    return var_loss, cov_loss, std


def jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_enc: torch.Tensor,
    *,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    motion_weight: float = 0.0,
    motion_margin: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Latent prediction loss on LayerNorm'd embeddings + VICReg insurance.

    ``z_pred``/``z_target``: (B, K, D) multi-step rollout and stop-grad targets
    (paper eq. 2/4); ``z_enc``: (B, D) encoder embeddings at time t, WITH grad.
    Smooth-L1 for prediction.

    Two shortcut guards learned the hard way:
    - The variance/covariance terms must include the encoder branch: applied
      to the predicted branch only, the encoder collapses to a constant while
      the predictor manufactures batch variance from actions alone.
    - ``motion_weight`` > 0 adds a temporal-contrast hinge pushing LN'd
      embeddings of frames K steps apart to differ by at least
      ``motion_margin`` (per-dim MSE). Needed for multi-layout training:
      batch variance is otherwise satisfied by *inter-layout* variance, so the
      encoder may encode the maze identity and drop the agent position -
      prediction loss goes to ~0 and planning becomes position-blind.
    """
    d = z_pred.shape[-1]
    zp = F.layer_norm(z_pred, (d,))
    zt = F.layer_norm(z_target, (d,))
    pred = F.smooth_l1_loss(zp, zt)

    zel = F.layer_norm(z_enc, (d,))
    enc_var, enc_cov, enc_std = _var_cov(zel.reshape(-1, d))
    pred_var, pred_cov, _ = _var_cov(zp.reshape(-1, d))
    var_loss = 0.5 * (enc_var + pred_var)
    cov_loss = 0.5 * (enc_cov + pred_cov)

    if motion_weight > 0.0:
        travel = ((zel - zt[:, -1]) ** 2).mean(dim=-1)  # per-dim MSE, t vs t+K
        motion = F.relu(motion_margin - travel).mean()
    else:
        motion = zp.new_zeros(())

    loss = (
        pred
        + variance_weight * var_loss
        + covariance_weight * cov_loss
        + motion_weight * motion
    )
    metrics = {
        "pred_loss": float(pred.detach().cpu()),
        "var_loss": float(var_loss.detach().cpu()),
        "cov_loss": float(cov_loss.detach().cpu()),
        "motion_loss": float(motion.detach().cpu()),
        "embed_std": float(enc_std.mean().detach().cpu()),  # encoder batch spread
    }
    return loss, metrics


def adaptation_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Test-time loss (paper eq. 4): prediction term only, stop-grad target.

    No variance/covariance terms: with tiny buffers batch statistics are
    meaningless, and the paper relies on stop-grad + restricted parameters.
    """
    d = z_pred.shape[-1]
    return F.smooth_l1_loss(F.layer_norm(z_pred, (d,)), F.layer_norm(z_target, (d,)))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(model: WorldModel, path, extra: dict | None = None) -> None:
    torch.save(
        {
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, device: str | torch.device = "cpu") -> WorldModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = [
    "Encoder",
    "ModelConfig",
    "Predictor",
    "WorldModel",
    "adaptation_loss",
    "jepa_loss",
    "load_checkpoint",
    "save_checkpoint",
]
