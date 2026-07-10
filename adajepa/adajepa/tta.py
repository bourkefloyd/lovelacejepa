"""AdaJEPA test-time adaptation (paper section 3.2, algorithm 1) + LACE.

After each executed action chunk, the observed transition becomes a
self-supervised target: one (or a few) gradient steps on

    L_ada(B) = mean over buffer of  l( f(E(o_i), E_a(a_i)), TARGET(o_{i+1}) )

updating only a small parameter subset. Weights are episode-local: build one
adapter per episode on a fresh model copy; nothing is written back to the
checkpoint.

``target_source`` selects TARGET (the LovelaceJEPA/LACE knob):

- ``student`` (AdaJEPA, the paper): ``sg(E_theta(o_{i+1}))`` - the target is
  produced by the very encoder being adapted, so adaptation may reduce the
  loss by relocating the latent space itself ("unlaced").
- ``frozen`` (LACE): ``E_frozen(o_{i+1})`` - a frozen copy of the pretrained
  encoder anchors adaptation to the pretrained manifold ("laced-frozen").
- ``ema`` (LACE): a slow exponential-moving-average copy of the adapting
  encoder; the anchor itself may drift slowly toward a persistent shift
  ("laced-ema", decay ``ema_decay``).

Paper defaults: 1 gradient step per replan at the training learning rate, a
recent-5 transition buffer, and the ``predlast+enclast`` target.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass

import torch
from torch import nn

from .model import WorldModel, adaptation_loss

TARGET_SOURCES = ("student", "frozen", "ema")

ADAPT_TARGETS = (
    "predlast+enclast",
    "predfirst+enclast",
    "predlast",
    "enclast",
    "all",
)


def select_adapt_params(
    model: WorldModel, target: str
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(predictor_params, encoder_params) a target updates (rest stay frozen).

    The two groups get separate learning rates: at test time there is no
    variance/covariance term protecting the encoder from collapse, so it must
    move much more slowly than the predictor (the paper trains and adapts with
    encoder lr 1e-5 vs predictor lr 5e-4 for the same reason).
    """
    if target not in ADAPT_TARGETS:
        raise ValueError(f"unknown adapt target {target!r}; choose from {ADAPT_TARGETS}")
    if target == "all":
        pred = list(model.predictor.parameters()) + list(model.action_encoder.parameters())
        return pred, list(model.encoder.parameters())
    pred_params: list[nn.Parameter] = []
    enc_params: list[nn.Parameter] = []
    if "predlast" in target:
        pred_params += list(model.predictor.blocks[-1].parameters())
        pred_params += list(model.predictor.norm.parameters())
        pred_params += list(model.predictor.head.parameters())
    if "predfirst" in target:
        pred_params += list(model.predictor.in_proj.parameters())
        pred_params += list(model.predictor.blocks[0].parameters())
    if "enclast" in target:
        enc_params += list(model.encoder.proj.parameters())
    return pred_params, enc_params


@dataclass
class AdaptConfig:
    lr: float = 3e-4  # predictor lr; default = the training lr (paper default)
    enc_lr: float = 1e-5  # encoder lr, much smaller (paper: 1e-5 vs 5e-4)
    steps: int = 1
    buffer_size: int = 5
    target: str = "predlast+enclast"
    # --- LACE (LovelaceJEPA) knobs -----------------------------------------
    target_source: str = "student"  # "student" (AdaJEPA) | "frozen" | "ema"
    ema_decay: float = 0.996  # only used when target_source == "ema"
    # Which encoder embeds the GOAL each replan: "model" re-encodes with the
    # adapting encoder (AdaJEPA behavior); "anchor" uses the anchor encoder,
    # keeping the planning cost pinned to the pretrained space (LACE default).
    goal_encoder: str = "model"


class TestTimeAdapter:
    """Plan-execute-adapt-replan companion: buffers transitions, adapts the model."""

    def __init__(self, model: WorldModel, config: AdaptConfig | None = None) -> None:
        self.model = model
        self.config = config or AdaptConfig()
        if self.config.buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if self.config.target_source not in TARGET_SOURCES:
            raise ValueError(
                f"unknown target_source {self.config.target_source!r}; "
                f"choose from {TARGET_SOURCES}"
            )
        if self.config.goal_encoder not in ("model", "anchor"):
            raise ValueError("goal_encoder must be 'model' or 'anchor'")
        if self.config.goal_encoder == "anchor" and self.config.target_source == "student":
            raise ValueError("goal_encoder='anchor' requires an anchored target_source")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        pred_params, enc_params = select_adapt_params(model, self.config.target)
        self._params = pred_params + enc_params
        for p in self._params:
            p.requires_grad_(True)
        groups = []
        if pred_params:
            groups.append({"params": pred_params, "lr": self.config.lr})
        if enc_params:
            groups.append({"params": enc_params, "lr": self.config.enc_lr})
        self.optimizer = torch.optim.Adam(groups)
        self._buffer: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(
            maxlen=self.config.buffer_size
        )
        self.updates = 0

        # LACE anchor: a frozen (or slow-EMA) copy of the pretrained encoder
        # supplies adaptation targets, pinning the student to the pretrained
        # latent manifold. Episode-local like everything else here.
        self.anchor_encoder: nn.Module | None = None
        if self.config.target_source in ("frozen", "ema"):
            self.anchor_encoder = copy.deepcopy(model.encoder)
            self.anchor_encoder.eval()
            for p in self.anchor_encoder.parameters():
                p.requires_grad_(False)

    @property
    def encodes_goal_with_anchor(self) -> bool:
        return self.anchor_encoder is not None and self.config.goal_encoder == "anchor"

    @torch.no_grad()
    def anchor_encode(self, obs: torch.Tensor) -> torch.Tensor:
        assert self.anchor_encoder is not None
        return self.anchor_encoder(obs)

    @torch.no_grad()
    def _ema_update(self) -> None:
        assert self.anchor_encoder is not None
        d = self.config.ema_decay
        for p_a, p_s in zip(
            self.anchor_encoder.parameters(), self.model.encoder.parameters()
        ):
            p_a.mul_(d).add_(p_s, alpha=1.0 - d)
        for b_a, b_s in zip(self.anchor_encoder.buffers(), self.model.encoder.buffers()):
            b_a.copy_(b_s)

    def _target(self, nxt: torch.Tensor) -> torch.Tensor:
        """Adaptation target latents for the buffered next observations."""
        with torch.no_grad():
            if self.anchor_encoder is not None:
                return self.anchor_encoder(nxt)
            return self.model.encode(nxt)  # sg(z_{i+1}) - the paper's target

    def observe(
        self, before: torch.Tensor, action: torch.Tensor, after: torch.Tensor
    ) -> float:
        """Buffer one transition and adapt; returns the pre-update loss.

        before/after: (1, C, H, W) float obs; action: (1, 2).
        """
        self._buffer.append((before.detach(), action.detach(), after.detach()))
        b = torch.cat([t[0] for t in self._buffer])
        a = torch.cat([t[1] for t in self._buffer])
        nxt = torch.cat([t[2] for t in self._buffer])

        first_loss = 0.0
        for step in range(max(1, self.config.steps)):
            z = self.model.encode(b)
            z_pred = self.model.predict(z, a)
            z_tgt = self._target(nxt)
            loss = adaptation_loss(z_pred, z_tgt)
            loss_val = float(loss.detach().cpu())  # capture BEFORE backward (MPS)
            if step == 0:
                first_loss = loss_val
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return first_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 1.0)
            self.optimizer.step()
            self.updates += 1
            if self.config.target_source == "ema":
                self._ema_update()
        return first_loss


def episode_model(pretrained: WorldModel) -> WorldModel:
    """A fresh episode-local copy of the pretrained model (never mutated back)."""
    model = copy.deepcopy(pretrained)
    model.eval()
    return model


@torch.no_grad()
def prediction_loss(
    model: WorldModel,
    before: torch.Tensor,
    action: torch.Tensor,
    after: torch.Tensor,
) -> float:
    """The adaptation objective, evaluated without updating (for diagnostics)."""
    z_pred = model.predict(model.encode(before), action)
    z_tgt = model.encode(after)
    return float(adaptation_loss(z_pred, z_tgt).cpu())


__all__ = [
    "ADAPT_TARGETS",
    "TARGET_SOURCES",
    "AdaptConfig",
    "TestTimeAdapter",
    "episode_model",
    "prediction_loss",
    "select_adapt_params",
]
