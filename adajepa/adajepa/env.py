"""PointMaze-style environment for the AdaJEPA reproduction.

A point mass navigates a randomly generated grid maze under 2D force actions,
rendered to small RGB frames. Pure numpy - no MuJoCo/gym - so the whole
reproduction stays dependency-light and deterministic.

The environment exposes the paper's three shift axes:

- **Dynamics shifts**: ``mass`` (low mass -> faster under same force) and
  ``damping`` (high damping -> velocity decays faster), matching the
  PointMaze-Medium shifts (x0.2 mass, x20 damping).
- **Visual shifts**: per-frame corruptions (``blur``, ``snp``, ``dark``) and a
  color change (``red_agent``), matching the PushT visual-shift suite.
- **Layout shifts**: ``layout_seed`` selects a randomly generated connected
  maze; train on some seeds, evaluate on held-out ones.

Observations are channel-stacked frame histories (``frame_stack`` frames) so a
single latent can capture velocity; the paper uses a 3-frame history window
for the same reason.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

VISUAL_SHIFTS = ("none", "blur", "snp", "dark", "red_agent")


# ---------------------------------------------------------------------------
# Maze layout
# ---------------------------------------------------------------------------


class Maze:
    """A ``grid x grid`` maze generated with a randomized DFS (always connected).

    Walls are stored on cell edges: ``v_walls[r, c]`` blocks movement between
    ``(r, c-1)`` and ``(r, c)``; ``h_walls[r, c]`` blocks movement between
    ``(r-1, c)`` and ``(r, c)``. Outer boundary is always walled.
    """

    def __init__(self, grid: int, layout_seed: int, extra_open: float = 0.15) -> None:
        self.grid = grid
        self.layout_seed = layout_seed
        rng = np.random.default_rng(layout_seed)
        g = grid
        self.v_walls = np.ones((g, g + 1), dtype=bool)
        self.h_walls = np.ones((g + 1, g), dtype=bool)

        # Randomized DFS over cells carves a spanning tree of open edges.
        visited = np.zeros((g, g), dtype=bool)
        stack = [(int(rng.integers(g)), int(rng.integers(g)))]
        visited[stack[0]] = True
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < g and 0 <= nc < g and not visited[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                stack.pop()
                continue
            nr, nc = neighbors[int(rng.integers(len(neighbors)))]
            if nr == r:
                self.v_walls[r, max(c, nc)] = False
            else:
                self.h_walls[max(r, nr), c] = False
            visited[nr, nc] = True
            stack.append((nr, nc))

        # Knock out a few extra interior walls so mazes have loops (less
        # corridor-like, closer to PointMaze-Medium connectivity).
        for r in range(g):
            for c in range(1, g):
                if self.v_walls[r, c] and rng.random() < extra_open:
                    self.v_walls[r, c] = False
        for r in range(1, g):
            for c in range(g):
                if self.h_walls[r, c] and rng.random() < extra_open:
                    self.h_walls[r, c] = False

    def blocked(self, r: int, c: int, dr: int, dc: int) -> bool:
        """Whether moving from cell (r, c) by (dr, dc) crosses a wall."""
        if dr == 0 and dc == 1:
            return bool(self.v_walls[r, c + 1])
        if dr == 0 and dc == -1:
            return bool(self.v_walls[r, c])
        if dr == 1 and dc == 0:
            return bool(self.h_walls[r + 1, c])
        if dr == -1 and dc == 0:
            return bool(self.h_walls[r, c])
        raise ValueError(f"bad step ({dr}, {dc})")

    def bfs_distances(self, start_cell: tuple[int, int]) -> np.ndarray:
        """Shortest-path distance (in cells) from ``start_cell`` to every cell."""
        g = self.grid
        dist = np.full((g, g), -1, dtype=int)
        dist[start_cell] = 0
        queue: deque[tuple[int, int]] = deque([start_cell])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < g and 0 <= nc < g and dist[nr, nc] < 0:
                    if not self.blocked(r, c, dr, dc):
                        dist[nr, nc] = dist[r, c] + 1
                        queue.append((nr, nc))
        return dist


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    grid: int = 5
    img_size: int = 64
    frame_stack: int = 2
    # dt is deliberately large (the paper's frameskip-5 analogue): big per-step
    # displacement makes wall contacts frequent, which forces *position* into
    # the latent - one-step velocity prediction alone can't explain contacts.
    dt: float = 0.25
    substeps: int = 8
    mass: float = 1.0
    damping: float = 1.0
    force_scale: float = 4.0
    max_speed: float = 5.0
    layout_seed: int = 0
    visual: str = "none"
    agent_color: tuple[int, int, int] = (60, 90, 235)
    wall_px: int = 2
    agent_radius_px: float = 3.5

    def shifted(self, **kwargs) -> "EnvConfig":
        return replace(self, **kwargs)


@dataclass
class StepInfo:
    pos: np.ndarray
    vel: np.ndarray
    frame: np.ndarray  # (H, W, 3) uint8, corrupted view


class PointMazeEnv:
    """Point-mass maze navigation with image observations."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        if self.config.visual not in VISUAL_SHIFTS:
            raise ValueError(f"unknown visual shift {self.config.visual!r}")
        self.maze = Maze(self.config.grid, self.config.layout_seed)
        self._wall_mask = self._build_wall_mask()
        self._frames: deque[np.ndarray] = deque(maxlen=self.config.frame_stack)
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self._rng = np.random.default_rng(0)

    # -- geometry / rendering ------------------------------------------------

    def _build_wall_mask(self) -> np.ndarray:
        cfg = self.config
        size, g = cfg.img_size, cfg.grid
        cell = size / g
        mask = np.zeros((size, size), dtype=bool)
        half = cfg.wall_px

        def px(x: float) -> int:
            return int(round(x * cell))

        for r in range(g + 1):
            for c in range(g):
                if self.maze.h_walls[r, c]:
                    y = min(max(px(r), half), size - half)
                    mask[y - half : y + half, px(c) : px(c + 1)] = True
        for r in range(g):
            for c in range(g + 1):
                if self.maze.v_walls[r, c]:
                    x = min(max(px(c), half), size - half)
                    mask[px(r) : px(r + 1), x - half : x + half] = True
        return mask

    def render_frame(self, pos: np.ndarray | None = None, *, corrupt: bool = True) -> np.ndarray:
        """Render one (H, W, 3) uint8 frame, optionally with the visual shift."""
        cfg = self.config
        size = cfg.img_size
        frame = np.full((size, size, 3), 255, dtype=np.uint8)
        frame[self._wall_mask] = (40, 40, 40)

        p = self.pos if pos is None else pos
        cell = size / cfg.grid
        cy, cx = p[0] * cell, p[1] * cell
        yy, xx = np.mgrid[0:size, 0:size]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= cfg.agent_radius_px**2
        color = (220, 40, 40) if cfg.visual == "red_agent" else cfg.agent_color
        frame[disc] = color

        if corrupt and cfg.visual not in ("none", "red_agent"):
            frame = self._apply_corruption(frame)
        return frame

    def _apply_corruption(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.visual == "dark":
            return (frame.astype(np.float32) * 0.4).astype(np.uint8)
        if cfg.visual == "blur":
            return _box_blur(frame, k=5)
        if cfg.visual == "snp":
            out = frame.copy()
            noise = self._rng.random(frame.shape[:2])
            out[noise < 0.06] = 0
            out[noise > 0.94] = 255
            return out
        return frame

    # -- dynamics -------------------------------------------------------------

    def _move(self, delta: np.ndarray) -> None:
        """Move the point by ``delta`` (cell units), stopping at walls."""
        g = self.config.grid
        eps = 1e-3
        for axis in (0, 1):
            if delta[axis] == 0.0:
                continue
            new = self.pos[axis] + delta[axis]
            new = float(np.clip(new, eps, g - eps))
            old_cell = int(self.pos[axis])
            new_cell = int(new)
            if new_cell != old_cell:
                step = 1 if new_cell > old_cell else -1
                r, c = int(self.pos[0]), int(self.pos[1])
                dr, dc = (step, 0) if axis == 0 else (0, step)
                if self.maze.blocked(r, c, dr, dc):
                    boundary = old_cell + (1 if step > 0 else 0)
                    new = boundary - eps * step
                    self.vel[axis] = 0.0
            self.pos[axis] = new
        # Clamped at outer edges too: zero velocity components that hit them.
        for axis in (0, 1):
            if self.pos[axis] <= eps or self.pos[axis] >= g - eps:
                self.vel[axis] = 0.0

    def step_physics(self, action: np.ndarray) -> None:
        cfg = self.config
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        dt_sub = cfg.dt / cfg.substeps
        for _ in range(cfg.substeps):
            acc = cfg.force_scale * a / cfg.mass
            self.vel = self.vel + acc * dt_sub
            self.vel = self.vel * max(0.0, 1.0 - cfg.damping * dt_sub)
            speed = float(np.linalg.norm(self.vel))
            if speed > cfg.max_speed:
                self.vel = self.vel * (cfg.max_speed / speed)
            self._move(self.vel * dt_sub)

    # -- episode API ----------------------------------------------------------

    def reset_to(self, start: np.ndarray) -> np.ndarray:
        """Generic task protocol: reset to a start state (here, a position)."""
        return self.reset(pos=start)

    def state(self) -> np.ndarray:
        """Generic task protocol: the environment's state vector (here, pos)."""
        return self.pos.copy()

    def goal_distance(self, goal: np.ndarray) -> float:
        """Generic task protocol: distance to goal in state space (cells)."""
        return float(np.linalg.norm(self.pos - np.asarray(goal, dtype=np.float64)))

    def is_success(self, goal: np.ndarray, radius: float = 0.5) -> bool:
        """Generic task protocol: within ``radius`` cells of the goal."""
        return self.goal_distance(goal) < radius

    def reset(
        self,
        *,
        pos: np.ndarray | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if pos is None:
            g = self.config.grid
            cell = self._rng.integers(0, g, size=2)
            pos = cell + self._rng.uniform(0.3, 0.7, size=2)
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.zeros(2)
        frame = self.render_frame()
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, StepInfo]:
        self.step_physics(action)
        frame = self.render_frame()
        self._frames.append(frame)
        return self.observation(), StepInfo(self.pos.copy(), self.vel.copy(), frame)

    def observation(self) -> np.ndarray:
        """Channel-stacked frame history, (H, W, 3 * frame_stack) uint8."""
        return np.concatenate(list(self._frames), axis=-1)

    def goal_observation(self, goal_pos: np.ndarray) -> np.ndarray:
        """Goal obs: agent at rest at the goal (same frame repeated)."""
        frame = self.render_frame(np.asarray(goal_pos, dtype=np.float64))
        return np.concatenate([frame] * self.config.frame_stack, axis=-1)

    # -- task sampling ---------------------------------------------------------

    def sample_task(
        self, rng: np.random.Generator, *, min_cells: int = 3, max_cells: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Start/goal with BFS shortest-path distance in [min_cells, max_cells]."""
        g = self.config.grid
        for _ in range(500):
            start_cell = (int(rng.integers(g)), int(rng.integers(g)))
            dist = self.maze.bfs_distances(start_cell)
            candidates = np.argwhere((dist >= min_cells) & (dist <= max_cells))
            if len(candidates) == 0:
                continue
            goal_cell = candidates[int(rng.integers(len(candidates)))]
            start = np.array(start_cell) + rng.uniform(0.35, 0.65, size=2)
            goal = goal_cell + rng.uniform(0.35, 0.65, size=2)
            return start, goal
        raise RuntimeError("could not sample a task; maze too small?")


def _box_blur(frame: np.ndarray, k: int = 5) -> np.ndarray:
    """Separable box blur without scipy (edge-padded)."""
    x = frame.astype(np.float32)
    pad = k // 2
    for axis in (0, 1):
        xp = np.concatenate(
            [np.repeat(x.take([0], axis=axis), pad, axis=axis), x,
             np.repeat(x.take([-1], axis=axis), pad, axis=axis)],
            axis=axis,
        )
        csum = np.cumsum(xp, axis=axis)
        zero = np.zeros_like(csum.take([0], axis=axis))
        csum = np.concatenate([zero, csum], axis=axis)
        hi = csum.take(range(k, k + x.shape[axis]), axis=axis)
        lo = csum.take(range(0, x.shape[axis]), axis=axis)
        x = (hi - lo) / k
    return np.clip(x, 0, 255).astype(np.uint8)


# Named evaluation settings (the paper's shift suite, adapted to this env).
def make_shift_config(base: EnvConfig, shift: str) -> EnvConfig:
    """An EnvConfig for a named train->test shift on top of ``base``."""
    if shift == "default":
        return base
    if shift == "low_mass":
        return base.shifted(mass=base.mass * 0.2)
    if shift == "high_damping":
        # The paper uses x20 in MuJoCo; with this integrator's semantics x20
        # makes 3-5-cell goals physically unreachable within the step budget
        # (terminal speed ~ 1/damping). x8 degrades without crippling.
        return base.shifted(damping=base.damping * 8.0)
    if shift in ("blur", "snp", "dark", "red_agent"):
        return base.shifted(visual=shift)
    if shift.startswith("layout:"):
        return base.shifted(layout_seed=int(shift.split(":", 1)[1]))
    raise ValueError(f"unknown shift {shift!r}")


SHIFT_NAMES = ("default", "low_mass", "high_damping", "blur", "snp", "dark", "red_agent")

__all__ = [
    "EnvConfig",
    "Maze",
    "PointMazeEnv",
    "SHIFT_NAMES",
    "StepInfo",
    "VISUAL_SHIFTS",
    "make_shift_config",
]
