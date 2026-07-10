"""PushObj-mini: a miniature PushT/PushObj environment for the reproduction.

A circular pusher (velocity-controlled) interacts with a rigid polyomino
block on a bounded 2D table (pymunk physics, top-down, quasi-static damping).
Rendered to the same small RGB frames the maze env produces, so the world
model code is unchanged - that is the point of the benchmark.

Shape suite mirrors the paper's PushObj (appendix A.2): train on
``{T, L, Z, +}``, hold out ``{I, smallT, cube}`` (their "square"; we adopt
``cube`` naming). Every shape is a union of unit cells (a polyomino), which
makes both collision (one pymunk box per cell) and rendering (fill rotated
squares) exact and simple.

Shifts exposed (same ``shift`` string convention as the maze):

- **Shape shifts**: ``shape:<name>`` swaps the block geometry; contact
  dynamics are identical, only geometry is unseen.
- **Visual shifts**: ``blur`` / ``snp`` / ``dark`` / ``red_agent`` reuse the
  maze env's corruptions.

Goals follow the paper: a goal is a *reachable* future state - sampled by
rolling the environment forward ``goal_steps`` steps with a contact-biased
exploration policy and taking the resulting state (their "goals are sampled
25 steps away").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import pymunk

from .env import VISUAL_SHIFTS, _box_blur

# Polyomino cell offsets (x, y) in cell units, centered on the centroid below.
SHAPES: dict[str, tuple[tuple[float, float], ...]] = {
    "T": ((-1, 1), (0, 1), (1, 1), (0, 0), (0, -1)),
    "L": ((0, 1), (0, 0), (0, -1), (1, -1)),
    "Z": ((-1, 1), (0, 1), (0, 0), (1, 0)),
    "plus": ((0, 1), (-1, 0), (0, 0), (1, 0), (0, -1)),
    "I": ((0, 2), (0, 1), (0, 0), (0, -1)),
    "smallT": ((-1, 0), (0, 0), (1, 0), (0, -1)),
    "cube": ((0, 0), (1, 0), (0, 1), (1, 1)),
}
TRAIN_SHAPES = ("T", "L", "Z", "plus")
TEST_SHAPES = ("I", "smallT", "cube")

# Rotational symmetry period per shape: angle errors are scored modulo this
# (a cube rotated 90 degrees is visually identical; penalizing it would make
# success unmeasurable from images for symmetric shapes).
SHAPE_SYMMETRY: dict[str, float] = {
    "T": 2 * np.pi,
    "L": 2 * np.pi,
    "Z": np.pi,
    "plus": np.pi / 2,
    "I": np.pi,
    "smallT": 2 * np.pi,
    "cube": np.pi / 2,
}


def _centered_cells(name: str) -> np.ndarray:
    cells = np.asarray(SHAPES[name], dtype=np.float64)
    return cells - cells.mean(axis=0)


@dataclass
class PushObjConfig:
    shape: str = "T"
    world: float = 5.0  # square table side, world units
    img_size: int = 64
    frame_stack: int = 2
    dt: float = 0.25
    substeps: int = 8
    block_cell: float = 0.55  # polyomino cell side, world units
    block_mass: float = 0.8
    pusher_radius: float = 0.28
    pusher_mass: float = 3.0
    pusher_speed: float = 2.0  # action [-1,1] -> velocity units/s
    damping: float = 0.1  # pymunk space damping (quasi-static block)
    friction: float = 0.5
    visual: str = "none"
    agent_color: tuple[int, int, int] = (60, 90, 235)
    block_color: tuple[int, int, int] = (120, 120, 120)
    # Task tolerances (success = both satisfied). Calibrated against a
    # ground-truth oracle CEM: goal_steps<=15 with these tolerances is 8/8
    # oracle-solvable within the replan budget (goal_steps=25 drops to 5/8),
    # so learned-model failures are attributable to the model, not the task.
    # goal_steps=8 is the paper protocol: the frozen model lands mid-range
    # on seen shapes (~30-50%), leaving headroom in both directions.
    success_pos_tol: float = 0.5
    success_ang_tol: float = 0.6  # radians
    goal_steps: int = 8
    min_goal_motion: float = 0.3  # block must move this far for a valid task

    def shifted(self, **kwargs) -> "PushObjConfig":
        return replace(self, **kwargs)


@dataclass
class PushStepInfo:
    pos: np.ndarray  # 5-vector state: pusher (x, y), block (x, y, angle)
    frame: np.ndarray  # (H, W, 3) uint8, corrupted view


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class PushObjEnv:
    """Pusher + polyomino block manipulation with image observations.

    State vector convention (used by ``state()`` / ``reset_to`` / goals):
    ``[pusher_x, pusher_y, block_x, block_y, block_angle]``.
    """

    def __init__(self, config: PushObjConfig | None = None) -> None:
        self.config = config or PushObjConfig()
        if self.config.visual not in VISUAL_SHIFTS:
            raise ValueError(f"unknown visual shift {self.config.visual!r}")
        if self.config.shape not in SHAPES:
            raise ValueError(
                f"unknown shape {self.config.shape!r}; choose from {sorted(SHAPES)}"
            )
        self._cells = _centered_cells(self.config.shape)
        self._frames: deque[np.ndarray] = deque(maxlen=self.config.frame_stack)
        self._rng = np.random.default_rng(0)
        self._build_space()

    # -- physics ---------------------------------------------------------------

    def _build_space(self) -> None:
        cfg = self.config
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.damping = cfg.damping

        # Table walls.
        w = cfg.world
        static = space.static_body
        for a, b in (((0, 0), (w, 0)), ((w, 0), (w, w)), ((w, w), (0, w)), ((0, w), (0, 0))):
            seg = pymunk.Segment(static, a, b, 0.02)
            seg.friction = cfg.friction
            space.add(seg)

        # Block: one box shape per polyomino cell.
        s = cfg.block_cell
        n_cells = len(self._cells)
        cell_mass = cfg.block_mass / n_cells
        moment = 0.0
        for cx, cy in self._cells:
            verts = [
                (cx * s - s / 2, cy * s - s / 2),
                (cx * s + s / 2, cy * s - s / 2),
                (cx * s + s / 2, cy * s + s / 2),
                (cx * s - s / 2, cy * s + s / 2),
            ]
            moment += pymunk.moment_for_poly(cell_mass, verts)
        block = pymunk.Body(cfg.block_mass, moment)
        block.position = (cfg.world / 2, cfg.world / 2)
        space.add(block)
        for cx, cy in self._cells:
            box = pymunk.Poly(
                block,
                [
                    (cx * s - s / 2, cy * s - s / 2),
                    (cx * s + s / 2, cy * s - s / 2),
                    (cx * s + s / 2, cy * s + s / 2),
                    (cx * s - s / 2, cy * s + s / 2),
                ],
            )
            box.friction = cfg.friction
            space.add(box)

        # Pusher: dynamic circle, velocity commanded each substep.
        pusher = pymunk.Body(
            cfg.pusher_mass, pymunk.moment_for_circle(cfg.pusher_mass, 0, cfg.pusher_radius)
        )
        pusher.position = (cfg.world / 4, cfg.world / 4)
        circle = pymunk.Circle(pusher, cfg.pusher_radius)
        circle.friction = cfg.friction
        space.add(pusher, circle)

        self.space = space
        self.block = block
        self.pusher = pusher

    def _set_state(self, state: np.ndarray) -> None:
        px, py, bx, by, ang = (float(v) for v in state)
        self.pusher.position = (px, py)
        self.pusher.velocity = (0.0, 0.0)
        self.pusher.angular_velocity = 0.0
        self.block.position = (bx, by)
        self.block.angle = ang
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        self.space.reindex_shapes_for_body(self.block)
        self.space.reindex_shapes_for_body(self.pusher)

    def state(self) -> np.ndarray:
        return np.array(
            [
                self.pusher.position.x,
                self.pusher.position.y,
                self.block.position.x,
                self.block.position.y,
                _wrap_angle(self.block.angle),
            ],
            dtype=np.float64,
        )

    def step_physics(self, action: np.ndarray) -> None:
        cfg = self.config
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v = a * cfg.pusher_speed
        dt_sub = cfg.dt / cfg.substeps
        for _ in range(cfg.substeps):
            self.pusher.velocity = (v[0], v[1])
            self.space.step(dt_sub)
        # Keep the pusher on the table (walls stop it, but clamp for safety).
        px = float(np.clip(self.pusher.position.x, 0.05, cfg.world - 0.05))
        py = float(np.clip(self.pusher.position.y, 0.05, cfg.world - 0.05))
        self.pusher.position = (px, py)

    # -- rendering --------------------------------------------------------------

    def _fill_quad(self, frame: np.ndarray, corners: np.ndarray, color) -> None:
        """Fill a convex quad given (4, 2) world coords (half-plane test)."""
        size = self.config.img_size
        scale = size / self.config.world
        pts = corners * scale  # world -> pixel (x, y); y is the row axis
        yy, xx = np.mgrid[0:size, 0:size]
        # Points are in CCW or CW order; use signed areas with consistent sign.
        inside = np.ones((size, size), dtype=bool)
        sign = 0.0
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            cross = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)
            if sign == 0.0:
                # Determine orientation from the polygon's own area.
                area = 0.0
                for j in range(4):
                    xa, ya = pts[j]
                    xb, yb = pts[(j + 1) % 4]
                    area += xa * yb - xb * ya
                sign = 1.0 if area > 0 else -1.0
            inside &= (sign * cross) >= 0
        frame[inside] = color

    def render_frame(
        self, state: np.ndarray | None = None, *, corrupt: bool = True
    ) -> np.ndarray:
        cfg = self.config
        size = cfg.img_size
        frame = np.full((size, size, 3), 255, dtype=np.uint8)

        st = self.state() if state is None else np.asarray(state, dtype=np.float64)
        px, py, bx, by, ang = st
        s = cfg.block_cell
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        for cx, cy in self._cells:
            local = np.array(
                [
                    [cx * s - s / 2, cy * s - s / 2],
                    [cx * s + s / 2, cy * s - s / 2],
                    [cx * s + s / 2, cy * s + s / 2],
                    [cx * s - s / 2, cy * s + s / 2],
                ]
            )
            world = local @ rot.T + np.array([bx, by])
            self._fill_quad(frame, world, cfg.block_color)

        scale = size / cfg.world
        cy_px, cx_px = py * scale, px * scale
        yy, xx = np.mgrid[0:size, 0:size]
        disc = (yy - cy_px) ** 2 + (xx - cx_px) ** 2 <= (cfg.pusher_radius * scale) ** 2
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

    # -- episode API -------------------------------------------------------------

    def reset(
        self, *, state: np.ndarray | None = None, seed: int | None = None
    ) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if state is None:
            state = self._sample_state(self._rng)
        self._set_state(np.asarray(state, dtype=np.float64))
        frame = self.render_frame()
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)
        return self.observation()

    def reset_to(self, start: np.ndarray) -> np.ndarray:
        return self.reset(state=start)

    def _sample_state(self, rng: np.random.Generator) -> np.ndarray:
        """Random block pose away from walls; pusher near the block (contact bias)."""
        cfg = self.config
        margin = 1.2
        bx, by = rng.uniform(margin, cfg.world - margin, size=2)
        ang = rng.uniform(-np.pi, np.pi)
        # Pusher at a ring around the block: close enough that exploration
        # makes contact, far enough not to start overlapping.
        r = rng.uniform(0.7, 1.3)
        theta = rng.uniform(0, 2 * np.pi)
        px = float(np.clip(bx + r * np.cos(theta), 0.3, cfg.world - 0.3))
        py = float(np.clip(by + r * np.sin(theta), 0.3, cfg.world - 0.3))
        return np.array([px, py, bx, by, ang])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, PushStepInfo]:
        self.step_physics(action)
        frame = self.render_frame()
        self._frames.append(frame)
        return self.observation(), PushStepInfo(self.state(), frame)

    def observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)

    def goal_observation(self, goal: np.ndarray) -> np.ndarray:
        frame = self.render_frame(np.asarray(goal, dtype=np.float64))
        return np.concatenate([frame] * self.config.frame_stack, axis=-1)

    # -- task API ------------------------------------------------------------------

    def _ang_err(self, a: float, b: float) -> float:
        """Angle error modulo the shape's rotational symmetry period."""
        period = SHAPE_SYMMETRY[self.config.shape]
        d = (a - b) % period
        return float(min(d, period - d))

    def goal_distance(self, goal: np.ndarray) -> float:
        """Block pose error: position + angular error (0.5 rad ~ one pos unit)."""
        st = self.state()
        pos_err = float(np.linalg.norm(st[2:4] - goal[2:4]))
        return pos_err + self._ang_err(float(st[4]), float(goal[4]))

    def is_success(self, goal: np.ndarray, radius: float | None = None) -> bool:
        st = self.state()
        pos_err = float(np.linalg.norm(st[2:4] - goal[2:4]))
        ang_err = self._ang_err(float(st[4]), float(goal[4]))
        return pos_err < self.config.success_pos_tol and ang_err < self.config.success_ang_tol

    def explore_action(
        self, a_prev: np.ndarray, rng: np.random.Generator, *, toward: float = 0.55
    ) -> np.ndarray:
        """Contact-biased OU exploration: OU noise blended with block direction."""
        st = self.state()
        to_block = st[2:4] - st[0:2]
        norm = float(np.linalg.norm(to_block))
        direction = to_block / norm if norm > 1e-6 else np.zeros(2)
        a = a_prev + 0.3 * (-a_prev) + 0.6 * rng.normal(size=2)
        a = (1 - toward) * a + toward * direction
        return np.clip(a, -1, 1)

    def sample_task(
        self, rng: np.random.Generator, **_: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """Start state + reachable goal (roll ``goal_steps`` of biased exploration).

        Matches the paper: goals are states ``goal_steps`` away under the data
        policy, rejected until the block actually moved (their contact filter).
        """
        cfg = self.config
        for _ in range(50):
            self.reset(seed=int(rng.integers(1 << 31)))
            start = self.state()
            a = rng.uniform(-1, 1, size=2)
            for _ in range(cfg.goal_steps):
                a = self.explore_action(a, rng)
                self.step(a)
            goal = self.state()
            block_moved = float(np.linalg.norm(goal[2:4] - start[2:4]))
            ang_moved = abs(_wrap_angle(float(goal[4] - start[4])))
            if block_moved >= cfg.min_goal_motion or ang_moved >= 0.4:
                self.reset(state=start)
                return start, goal
        raise RuntimeError("could not sample a contact-rich task")


# Named evaluation settings (shape + visual shifts).
def make_pushobj_shift_config(base: PushObjConfig, shift: str) -> PushObjConfig:
    if shift == "default":
        return base
    if shift.startswith("shape:"):
        return base.shifted(shape=shift.split(":", 1)[1])
    if shift in ("blur", "snp", "dark", "red_agent"):
        return base.shifted(visual=shift)
    raise ValueError(f"unknown pushobj shift {shift!r}")


PUSHOBJ_SHIFT_NAMES = tuple(
    ["default"]
    + [f"shape:{s}" for s in TRAIN_SHAPES + TEST_SHAPES]
    + ["blur", "snp", "dark", "red_agent"]
)

__all__ = [
    "PUSHOBJ_SHIFT_NAMES",
    "PushObjConfig",
    "PushObjEnv",
    "PushStepInfo",
    "SHAPES",
    "TEST_SHAPES",
    "TRAIN_SHAPES",
    "make_pushobj_shift_config",
]
