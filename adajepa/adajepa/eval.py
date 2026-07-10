"""Closed-loop MPC evaluation: frozen vs. AdaJEPA test-time adaptation.

``run_episode`` implements algorithm 1 of the paper for one goal-reaching
episode; ``run_suite`` sweeps shifts x arms x seeds and writes a JSON blob the
notebook consumes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .data import obs_to_tensor
from .env import EnvConfig, PointMazeEnv, make_shift_config
from .model import WorldModel
from .planner import PlannerConfig, plan, shift_plan
from .probes import ProbeHeads, auc_score, state_targets
from .pushobj import PushObjConfig, PushObjEnv, make_pushobj_shift_config
from .tta import AdaptConfig, TestTimeAdapter, episode_model, prediction_loss


def make_env(env_cfg) -> "PointMazeEnv | PushObjEnv":
    """Env factory dispatching on the config dataclass type."""
    if isinstance(env_cfg, PushObjConfig):
        return PushObjEnv(env_cfg)
    return PointMazeEnv(env_cfg)


def shift_config(env_cfg, shift: str):
    """Shift-config factory dispatching on the config dataclass type."""
    if isinstance(env_cfg, PushObjConfig):
        return make_pushobj_shift_config(env_cfg, shift)
    return make_shift_config(env_cfg, shift)


@dataclass
class EvalConfig:
    max_replans: int = 30
    success_radius: float = 0.5  # cell units
    min_goal_cells: int = 3
    max_goal_cells: int = 5
    episodes: int = 15
    seeds: tuple[int, ...] = (0, 1)


@dataclass
class EpisodeResult:
    success: bool
    steps_to_success: int | None  # replans consumed until success
    pred_losses: list[float]  # per-replan latent prediction loss (pre-update)
    goal_dist: list[float]  # per-replan Euclidean distance to goal
    positions: list[list[float]]
    goal: list[float]
    adapt_time_s: float
    plan_time_s: float
    # Frozen-probe readouts per replan (empty when no probes are supplied).
    # The probes see the latents the arm actually planned with, so encoder
    # drift shows up here as degraded AUC/correlation vs the labels.
    probe_succ_logit: list[float] = field(default_factory=list)
    probe_prog: list[float] = field(default_factory=list)
    probe_state_err: list[float] = field(default_factory=list)
    label_dist: list[float] = field(default_factory=list)
    label_succ: list[int] = field(default_factory=list)
    # E4 diagnostic: L2 drift of the goal latent from its episode-initial
    # (= pretrained) value. The goal OBSERVATION never changes, so any drift
    # is pure encoder relocation - 0 for frozen and anchored-goal arms.
    goal_drift: list[float] = field(default_factory=list)


def run_episode(
    pretrained: WorldModel,
    env,
    start: np.ndarray,
    goal: np.ndarray,
    planner_cfg: PlannerConfig,
    eval_cfg: EvalConfig,
    adapt_cfg: AdaptConfig | None,
    device: torch.device,
    probes: ProbeHeads | None = None,
) -> EpisodeResult:
    """One plan-execute-(adapt)-replan episode. ``adapt_cfg=None`` => frozen.

    ``env`` follows the generic task protocol (``reset_to`` / ``state`` /
    ``goal_observation`` / ``goal_distance`` / ``is_success``), satisfied by
    both :class:`PointMazeEnv` and :class:`PushObjEnv`. ``probes`` (optional)
    are FROZEN heads scored on the latents this arm plans with.
    """
    model = episode_model(pretrained).to(device)
    adapter = TestTimeAdapter(model, adapt_cfg) if adapt_cfg is not None else None

    obs = env.reset_to(start)
    goal_obs_t = obs_to_tensor(env.goal_observation(goal), device)

    pred_losses: list[float] = []
    goal_dists: list[float] = []
    positions: list[list[float]] = [list(map(float, env.state()))]
    success = False
    steps_to_success: int | None = None
    prev_plan: torch.Tensor | None = None
    adapt_time = 0.0
    plan_time = 0.0

    probe_succ_logit: list[float] = []
    probe_prog: list[float] = []
    probe_state_err: list[float] = []
    label_dist: list[float] = []
    label_succ: list[int] = []
    goal_drift: list[float] = []
    z_goal_init: torch.Tensor | None = None

    for replan in range(eval_cfg.max_replans):
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            z0 = model.encode(obs_t)
            # Re-encode the goal each replan: when the encoder itself adapts,
            # the goal latent must move with it (otherwise the planning cost
            # compares latents from two different encoders). The LACE arms
            # can override this via adapt_cfg.goal_encoder="anchor" (the goal
            # is then encoded by the adapter's anchor encoder instead).
            if adapter is not None and adapter.encodes_goal_with_anchor:
                z_goal = adapter.anchor_encode(goal_obs_t)
            else:
                z_goal = model.encode(goal_obs_t)
        if z_goal_init is None:
            z_goal_init = z_goal.detach().clone()
        goal_drift.append(float(torch.linalg.norm(z_goal - z_goal_init).cpu()))

        if probes is not None:
            with torch.no_grad():
                st_pred, prog, succ = probes(z0, z_goal)
            st = env.state()
            tgt = state_targets(np.asarray(st)[None], probes.cfg.env_kind)[0]
            probe_state_err.append(
                float(np.linalg.norm(st_pred.squeeze(0).cpu().numpy() - tgt))
            )
            probe_succ_logit.append(float(succ.squeeze(0).cpu()))
            probe_prog.append(float(prog.squeeze(0).cpu()))
            label_dist.append(env.goal_distance(goal))
            label_succ.append(int(env.is_success(goal, eval_cfg.success_radius)))

        t0 = time.time()
        actions = plan(model, z0, z_goal, planner_cfg, init_mean=prev_plan)
        plan_time += time.time() - t0
        prev_plan = shift_plan(actions, planner_cfg.execute_actions)

        # Execute the first action chunk, adapting on each observed transition.
        for k in range(planner_cfg.execute_actions):
            a_np = actions[k].detach().cpu().numpy()
            before_t = obs_to_tensor(obs, device)
            obs, info = env.step(a_np)
            positions.append(list(map(float, info.pos)))
            after_t = obs_to_tensor(obs, device)
            a_t = actions[k].detach().unsqueeze(0)
            if k == 0:
                # Diagnostic: one-step latent prediction error before any update
                # this replan (the quantity the paper plots per replanning step).
                pred_losses.append(prediction_loss(model, before_t, a_t, after_t))
            if adapter is not None:
                t0 = time.time()
                adapter.observe(before_t, a_t, after_t)
                adapt_time += time.time() - t0
            if env.is_success(goal, eval_cfg.success_radius):
                success = True
                break

        goal_dists.append(env.goal_distance(goal))
        if success:
            steps_to_success = replan + 1
            break

    if probes is not None:
        # Final probe row at the episode's terminal state. Without it the
        # success head never sees a positive label (successful episodes break
        # out before the next replan's probe row), making AUC undefined.
        obs_t = obs_to_tensor(obs, device)
        with torch.no_grad():
            z0 = model.encode(obs_t)
            if adapter is not None and adapter.encodes_goal_with_anchor:
                z_goal = adapter.anchor_encode(goal_obs_t)
            else:
                z_goal = model.encode(goal_obs_t)
            st_pred, prog, succ = probes(z0, z_goal)
        st = env.state()
        tgt = state_targets(np.asarray(st)[None], probes.cfg.env_kind)[0]
        probe_state_err.append(
            float(np.linalg.norm(st_pred.squeeze(0).cpu().numpy() - tgt))
        )
        probe_succ_logit.append(float(succ.squeeze(0).cpu()))
        probe_prog.append(float(prog.squeeze(0).cpu()))
        label_dist.append(env.goal_distance(goal))
        label_succ.append(int(env.is_success(goal, eval_cfg.success_radius)))

    return EpisodeResult(
        success=success,
        steps_to_success=steps_to_success,
        pred_losses=pred_losses,
        goal_dist=goal_dists,
        positions=positions,
        goal=list(map(float, goal)),
        adapt_time_s=round(adapt_time, 3),
        plan_time_s=round(plan_time, 3),
        probe_succ_logit=probe_succ_logit,
        probe_prog=probe_prog,
        probe_state_err=probe_state_err,
        label_dist=label_dist,
        label_succ=label_succ,
        goal_drift=goal_drift,
    )


def run_setting(
    pretrained: WorldModel,
    env_cfg: EnvConfig,
    planner_cfg: PlannerConfig,
    eval_cfg: EvalConfig,
    adapt_cfg: AdaptConfig | None,
    device: torch.device,
    probes: ProbeHeads | None = None,
) -> dict:
    """All episodes x seeds for one (shift, arm) cell; returns aggregates."""
    per_seed: list[float] = []
    episodes: list[dict] = []
    for seed in eval_cfg.seeds:
        rng = np.random.default_rng(10_000 + seed)
        torch.manual_seed(20_000 + seed)  # CEM sampling noise, for reproducibility
        env = make_env(env_cfg)
        wins = 0
        for _ in range(eval_cfg.episodes):
            start, goal = env.sample_task(
                rng, min_cells=eval_cfg.min_goal_cells, max_cells=eval_cfg.max_goal_cells
            )
            result = run_episode(
                pretrained, env, start, goal, planner_cfg, eval_cfg, adapt_cfg,
                device, probes=probes,
            )
            wins += int(result.success)
            episodes.append(asdict(result))
        per_seed.append(100.0 * wins / eval_cfg.episodes)
    cell = {
        "success_rate": round(float(np.mean(per_seed)), 2),
        "success_std": round(float(np.std(per_seed)), 2),
        "per_seed": per_seed,
        "episodes": episodes,
    }
    final_drifts = [ep["goal_drift"][-1] for ep in episodes if ep["goal_drift"]]
    if final_drifts:
        cell["goal_drift_final_mean"] = round(float(np.mean(final_drifts)), 4)
    if probes is not None:
        cell["probes"] = probe_metrics(episodes)
    return cell


def probe_metrics(episodes: list[dict]) -> dict:
    """Pooled frozen-probe metrics for one (shift, arm) cell.

    - ``success_auc``: does the frozen success head still rank goal-reached
      states above others, given this arm's latents?
    - ``progress_corr``: Pearson correlation of the frozen progress head with
      the true negative task distance.
    - ``state_err``: mean readout error of the frozen state head.
    """
    logits = np.array([v for ep in episodes for v in ep["probe_succ_logit"]])
    labels = np.array([v for ep in episodes for v in ep["label_succ"]])
    prog = np.array([v for ep in episodes for v in ep["probe_prog"]])
    dist = np.array([v for ep in episodes for v in ep["label_dist"]])
    err = np.array([v for ep in episodes for v in ep["probe_state_err"]])
    out: dict = {"n_steps": int(len(logits))}
    if len(logits):
        out["success_auc"] = round(auc_score(logits, labels), 4)
        if np.std(prog) > 1e-9 and np.std(dist) > 1e-9:
            out["progress_corr"] = round(float(np.corrcoef(prog, -dist)[0, 1]), 4)
        out["state_err"] = round(float(err.mean()), 4)
    return out


def run_suite(
    pretrained: WorldModel,
    base_env_cfg: EnvConfig,
    shifts: list[str],
    planner_cfg: PlannerConfig,
    eval_cfg: EvalConfig,
    adapt_cfg: AdaptConfig,
    out_path: Path,
    device: torch.device,
    probes: ProbeHeads | None = None,
    arms: list[tuple[str, AdaptConfig | None]] | None = None,
) -> dict:
    """Arms x shifts; writes JSON incrementally to out_path.

    Default arms are (frozen, adapt); pass ``arms`` explicitly for multi-arm
    suites (e.g. frozen / unlaced / laced-frozen / laced-ema).
    """
    if arms is None:
        arms = [("frozen", None), ("adapt", adapt_cfg)]
    results: dict = {
        "planner": asdict(planner_cfg),
        "eval": asdict(eval_cfg),
        "adapt": asdict(adapt_cfg),
        "arms": {name: (asdict(cfg) if cfg else None) for name, cfg in arms},
        "env": asdict(base_env_cfg),
        "env_kind": type(base_env_cfg).__name__,
        "shifts": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for shift in shifts:
        env_cfg = shift_config(base_env_cfg, shift)
        results["shifts"][shift] = {}
        for arm, cfg in arms:
            t0 = time.time()
            cell = run_setting(
                pretrained, env_cfg, planner_cfg, eval_cfg, cfg, device, probes=probes
            )
            cell["elapsed_s"] = round(time.time() - t0, 1)
            results["shifts"][shift][arm] = cell
            probe_str = ""
            if probes is not None and cell.get("probes", {}).get("success_auc") is not None:
                probe_str = f" succAUC={cell['probes']['success_auc']:.3f}"
            print(
                f"[eval] {shift:>14s} {arm:>14s}: "
                f"{cell['success_rate']:5.1f}% +/- {cell['success_std']:.1f} "
                f"({cell['elapsed_s']}s){probe_str}",
                flush=True,
            )
            out_path.write_text(json.dumps(results, indent=2))
    return results


def success_by_step(cell: dict, max_replans: int) -> np.ndarray:
    """Cumulative success rate (%) as a function of MPC replanning step."""
    curve = np.zeros(max_replans)
    n = len(cell["episodes"])
    for ep in cell["episodes"]:
        if ep["success"] and ep["steps_to_success"] is not None:
            curve[ep["steps_to_success"] - 1 :] += 1
    return 100.0 * curve / max(n, 1)


__all__ = [
    "EpisodeResult",
    "EvalConfig",
    "make_env",
    "probe_metrics",
    "run_episode",
    "run_setting",
    "run_suite",
    "shift_config",
    "success_by_step",
]
