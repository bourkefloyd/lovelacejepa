"""Offline trajectory generation and dataset windows for JEPA training.

Reward-free trajectories are collected with an Ornstein-Uhlenbeck exploration
policy (smooth random forces give good maze coverage). Stored as compressed
npz shards of uint8 frames + float actions/positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .env import EnvConfig, PointMazeEnv
from .pushobj import PushObjConfig, PushObjEnv


@dataclass
class DataConfig:
    n_trajectories: int = 1500
    traj_len: int = 32  # actions per trajectory (frames = traj_len + 1)
    layout_seeds: tuple[int, ...] = (0,)  # maze only
    shapes: tuple[str, ...] = ("T",)  # pushobj only
    seed: int = 0
    ou_theta: float = 0.3
    ou_sigma: float = 0.7
    min_block_motion: float = 0.25  # pushobj contact bias: reject static rolls
    max_retries: int = 8


def _alloc_trajectory_arrays(
    n: int, t: int, img_size: int, state_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared array layout for every generator: frames/actions/positions."""
    frames = np.zeros((n, t + 1, img_size, img_size, 3), dtype=np.uint8)
    actions = np.zeros((n, t, 2), dtype=np.float32)
    positions = np.zeros((n, t + 1, state_dim), dtype=np.float32)
    return frames, actions, positions


def _roll_trajectory(
    env, t: int, rng: np.random.Generator, action_fn, init_state
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll one trajectory on an already-reset env.

    ``action_fn(a)`` maps the previous action to the next one (the exploration
    policy). Returns ``(frames (t+1, ...), actions (t, 2), states (t+1, d))``.
    """
    frames = [env.render_frame()]
    states = [np.asarray(init_state, dtype=np.float32)]
    actions = np.zeros((t, 2), dtype=np.float32)
    a = rng.uniform(-1, 1, size=2)
    for k in range(t):
        a = action_fn(a)
        _, info = env.step(a)
        actions[k] = a
        frames.append(info.frame)
        states.append(np.asarray(info.pos, dtype=np.float32))
    return np.stack(frames), actions, np.stack(states)


def _save_dataset(
    out_path: Path,
    *,
    frames: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray,
    layouts: np.ndarray,
    grid,
    frame_stack: int,
    **extra,
) -> None:
    """Common npz schema consumed by :class:`TransitionWindows` and the probes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=frames,
        actions=actions,
        positions=positions,
        layouts=layouts,
        grid=grid,
        frame_stack=frame_stack,
        **extra,
    )


def generate_dataset(env_cfg: EnvConfig, data_cfg: DataConfig, out_path: Path) -> dict:
    """Roll exploration trajectories and save them to ``out_path`` (.npz)."""
    rng = np.random.default_rng(data_cfg.seed)
    n, t = data_cfg.n_trajectories, data_cfg.traj_len
    frames, actions, positions = _alloc_trajectory_arrays(
        n, t, env_cfg.img_size, state_dim=2
    )
    layouts = np.zeros(n, dtype=np.int64)

    def ou_action(a: np.ndarray) -> np.ndarray:
        a = a + data_cfg.ou_theta * (-a) + data_cfg.ou_sigma * rng.normal(size=2)
        return np.clip(a, -1, 1)

    envs = {
        s: PointMazeEnv(env_cfg.shifted(layout_seed=s)) for s in data_cfg.layout_seeds
    }
    for i in range(n):
        layout = data_cfg.layout_seeds[i % len(data_cfg.layout_seeds)]
        env = envs[layout]
        env.reset(seed=int(rng.integers(1 << 31)))
        layouts[i] = layout
        frames[i], actions[i], positions[i] = _roll_trajectory(
            env, t, rng, ou_action, env.pos
        )

    _save_dataset(
        out_path,
        frames=frames,
        actions=actions,
        positions=positions,
        layouts=layouts,
        grid=env_cfg.grid,
        frame_stack=env_cfg.frame_stack,
    )
    return {
        "path": str(out_path),
        "trajectories": n,
        "traj_len": t,
        "layouts": list(data_cfg.layout_seeds),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
    }


def generate_pushobj_dataset(
    env_cfg: PushObjConfig, data_cfg: DataConfig, out_path: Path
) -> dict:
    """Contact-biased exploration trajectories on PushObj-mini (.npz).

    Mirrors the paper's appendix A.2: trajectories are re-rolled (up to
    ``max_retries``) until the block actually moves, so the dataset is biased
    toward contact-rich interaction instead of free-space pusher motion.

    Same npz schema as the maze so :class:`TransitionWindows` is unchanged;
    ``positions`` holds the 5-vector state (pusher xy + block pose) per frame
    for the probe heads, and ``shapes``/``shape_ids`` record the block type.
    """
    rng = np.random.default_rng(data_cfg.seed)
    n, t = data_cfg.n_trajectories, data_cfg.traj_len
    frames, actions, positions = _alloc_trajectory_arrays(
        n, t, env_cfg.img_size, state_dim=5
    )
    shape_ids = np.zeros(n, dtype=np.int64)
    shape_names = list(data_cfg.shapes)

    envs = {s: PushObjEnv(env_cfg.shifted(shape=s)) for s in shape_names}
    contact_rich = 0
    for i in range(n):
        shape = shape_names[i % len(shape_names)]
        env = envs[shape]
        shape_ids[i] = shape_names.index(shape)
        # Mix exploration styles: mostly block-seeking (contact-rich pushes at
        # varying aggressiveness), but keep a pure-OU slice so the model also
        # sees "pusher moves, block does NOT" - without it, CEM candidates
        # that never touch the block get hallucinated block motion.
        pure_ou = rng.random() < 0.3
        toward = 0.0 if pure_ou else float(rng.uniform(0.35, 0.7))

        def seek_action(a: np.ndarray) -> np.ndarray:
            return env.explore_action(a, rng, toward=toward)

        for attempt in range(data_cfg.max_retries):
            env.reset(seed=int(rng.integers(1 << 31)))
            start = env.state()
            traj_frames, traj_actions, traj_states = _roll_trajectory(
                env, t, rng, seek_action, start
            )
            moved = float(np.linalg.norm(env.state()[2:4] - start[2:4]))
            if pure_ou or moved >= data_cfg.min_block_motion \
                    or attempt == data_cfg.max_retries - 1:
                if moved >= data_cfg.min_block_motion:
                    contact_rich += 1
                break
        frames[i] = traj_frames
        actions[i] = traj_actions
        positions[i] = traj_states

    _save_dataset(
        out_path,
        frames=frames,
        actions=actions,
        positions=positions,
        layouts=shape_ids,  # schema parity with the maze npz
        grid=env_cfg.world,
        frame_stack=env_cfg.frame_stack,
        shape_ids=shape_ids,
        shapes=np.array(shape_names),
        env_kind="pushobj",
    )
    return {
        "path": str(out_path),
        "trajectories": n,
        "traj_len": t,
        "shapes": shape_names,
        "contact_rich_frac": round(contact_rich / n, 3),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
    }


class TransitionWindows(torch.utils.data.Dataset):
    """(stacked obs_t, actions t..t+K-1, stacked obs at t+1..t+K) windows.

    Frame stacking is assembled on the fly from raw frames; ``pred_steps`` is
    the multi-step horizon K of the training objective (paper eq. 2).
    """

    def __init__(
        self,
        npz_path: Path,
        *,
        pred_steps: int = 2,
        max_trajectories: int | None = None,
    ) -> None:
        data = np.load(npz_path)
        self.frames = data["frames"]
        self.actions = data["actions"]
        self.frame_stack = int(data["frame_stack"])
        self.pred_steps = pred_steps
        if max_trajectories is not None:
            self.frames = self.frames[:max_trajectories]
            self.actions = self.actions[:max_trajectories]
        n, t_plus_1 = self.frames.shape[:2]
        t = t_plus_1 - 1
        # A window at (i, s) uses frames s-stack+1..s+K -> valid s range below.
        self.starts_per_traj = t - self.frame_stack + 2 - pred_steps
        if self.starts_per_traj <= 0:
            raise ValueError("trajectories too short for this frame_stack/pred_steps")
        self.n_traj = n

    def __len__(self) -> int:
        return self.n_traj * self.starts_per_traj

    def _stack(self, traj: int, t: int) -> np.ndarray:
        """Stacked obs at time t: frames [t-stack+1 .. t], channel-concat."""
        fs = self.frames[traj, t - self.frame_stack + 1 : t + 1]
        return np.concatenate(list(fs), axis=-1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, s = divmod(idx, self.starts_per_traj)
        t = s + self.frame_stack - 1  # first fully-stacked timestep
        obs = self._stack(traj, t)
        next_obs = np.stack(
            [self._stack(traj, t + k + 1) for k in range(self.pred_steps)]
        )
        acts = self.actions[traj, t : t + self.pred_steps]
        return {
            "obs": _to_chw(obs),
            "next_obs": torch.stack([_to_chw(o) for o in next_obs]),
            "actions": torch.from_numpy(acts.copy()),
        }


def _to_chw(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0


def obs_to_tensor(obs: np.ndarray, device) -> torch.Tensor:
    """(H, W, C) uint8 -> (1, C, H, W) float batch on device."""
    return _to_chw(obs).unsqueeze(0).to(device)


__all__ = [
    "DataConfig",
    "TransitionWindows",
    "generate_dataset",
    "generate_pushobj_dataset",
    "obs_to_tensor",
]
