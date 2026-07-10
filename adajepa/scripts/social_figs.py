"""Render shareable summary figures from the eval JSONs into figures/.

Run from adajepa/:  ../.venv/bin/python scripts/social_figs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adajepa.device import torch_device
from adajepa.env import EnvConfig, PointMazeEnv, make_shift_config
from adajepa.eval import EvalConfig, run_episode, success_by_step
from adajepa.model import load_checkpoint
from adajepa.planner import PlannerConfig
from adajepa.tta import AdaptConfig

RUNS = ROOT / "runs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

FROZEN_C = "#7f9fc4"
ADAPT_C = "#f28522"
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

suite = json.loads((RUNS / "suite_cem.json").read_text())
layout = json.loads((RUNS / "suite_layout.json").read_text())
lowdata = json.loads((RUNS / "suite_lowdata.json").read_text())

# ---------------------------------------------------------------- fig 1: bars
labels = {
    "default": "in-distribution",
    "low_mass": "low mass ×0.2",
    "high_damping": "high damping ×8",
    "blur": "blur",
    "snp": "s&p noise",
    "dark": "dark",
    "red_agent": "agent recolor",
    "layout:100": "unseen maze A",
    "layout:101": "unseen maze B",
    "layout:102": "unseen maze C",
}
rows = []
for k, lbl in labels.items():
    src = suite if k in suite["shifts"] else layout
    cell = src["shifts"][k]
    rows.append((lbl, cell["frozen"], cell["adapt"]))

fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(rows))
ax.bar(x - 0.2, [r[1]["success_rate"] for r in rows], 0.4,
       yerr=[r[1]["success_std"] for r in rows], capsize=3,
       label="frozen world model", color=FROZEN_C)
ax.bar(x + 0.2, [r[2]["success_rate"] for r in rows], 0.4,
       yerr=[r[2]["success_std"] for r in rows], capsize=3,
       label="AdaJEPA (1 grad step / replan)", color=ADAPT_C)
for xi, r in zip(x, rows):
    d = r[2]["success_rate"] - r[1]["success_rate"]
    if abs(d) >= 5:
        ax.annotate(f"{d:+.0f}", (xi + 0.2, r[2]["success_rate"] + r[2]["success_std"] + 2),
                    ha="center", fontsize=10, fontweight="bold",
                    color="#1a7d36" if d > 0 else "#b03030")
ax.set_xticks(x, [r[0] for r in rows], rotation=20, ha="right")
ax.set_ylabel("planning success (%)")
ax.set_ylim(0, 112)
ax.set_title("AdaJEPA reproduction: test-time adaptation vs frozen world model\n"
             "(PointMaze goal-reaching, CEM MPC, 15 episodes × 2 seeds per cell)")
ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "1_headline_bars.png", bbox_inches="tight")

# ------------------------------------------------- fig 2: success-vs-replans
max_replans = suite["eval"]["max_replans"]
show = ["default", "high_damping", "dark", "red_agent"]
titles = ["in-distribution", "dynamics shift (damping ×8)",
          "visual shift (dark)", "visual shift (agent recolor)"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, shift, t in zip(axes, show, titles):
    cell = suite["shifts"][shift]
    steps = range(1, max_replans + 1)
    ax.plot(steps, success_by_step(cell["frozen"], max_replans), color=FROZEN_C,
            lw=2.5, label="frozen")
    ax.plot(steps, success_by_step(cell["adapt"], max_replans), color=ADAPT_C,
            lw=2.5, label="AdaJEPA")
    ax.set_title(t, fontsize=11)
    ax.set_xlabel("MPC replanning step")
axes[0].set_ylabel("cumulative success (%)")
axes[0].legend(loc="upper left")
fig.suptitle("Frozen models saturate early — the adapted model keeps improving as it acts",
             y=1.06, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "2_success_by_step.png", bbox_inches="tight")

# ------------------------------------------ fig 3: one episode, frozen vs ada
device = torch_device()
model = load_checkpoint(RUNS / "base_l0.pt", device)
env_cfg = make_shift_config(EnvConfig(), "high_damping")
env = PointMazeEnv(env_cfg)
start, goal = env.sample_task(np.random.default_rng(5))
torch.manual_seed(0)
ep_frozen = run_episode(model, env, start, goal, PlannerConfig(kind="cem"),
                        EvalConfig(), None, device)
torch.manual_seed(0)
ep_adapt = run_episode(model, env, start, goal, PlannerConfig(kind="cem"),
                       EvalConfig(), AdaptConfig(), device)

def draw(ax, ep, title, color):
    bg = PointMazeEnv(env_cfg)
    bg.reset(pos=start)
    ax.imshow(bg.render_frame(pos=np.array([-10, -10]), corrupt=False),
              extent=[0, 5, 5, 0])
    xy = np.array(ep.positions)
    ax.plot(xy[:, 1], xy[:, 0], "-", color=color, lw=2)
    ax.plot(xy[0, 1], xy[0, 0], "o", color="black", ms=6)
    ax.plot(xy[-1, 1], xy[-1, 0], "s", color=color, ms=8)
    ax.plot(ep.goal[1], ep.goal[0], "*", color="goldenrod", ms=18, mec="k", mew=0.5)
    ax.set_title(title, fontsize=12)
    ax.grid(False)
    ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
draw(axes[0], ep_frozen, "frozen → " + ("reaches goal" if ep_frozen.success else "fails"),
     FROZEN_C)
draw(axes[1], ep_adapt, "AdaJEPA → " + ("reaches goal" if ep_adapt.success else "fails"),
     ADAPT_C)
axes[2].plot(ep_frozen.pred_losses, marker="o", ms=4, color=FROZEN_C, label="frozen")
axes[2].plot(ep_adapt.pred_losses, marker="o", ms=4, color=ADAPT_C, label="AdaJEPA")
axes[2].set_xlabel("MPC replanning step")
axes[2].set_ylabel("latent prediction loss")
axes[2].set_title("the mechanism: each executed action\nis a free training example", fontsize=11)
axes[2].legend()
fig.suptitle("Same start, same goal, unseen dynamics (damping ×8): plan–execute–adapt–replan",
             y=1.03, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "3_episode_mechanism.png", bbox_inches="tight")

# ------------------------------------------------------- fig 4: low-data bars
fig, ax = plt.subplots(figsize=(7.5, 4.4))
shifts4 = ["default", "high_damping", "red_agent"]
names4 = ["in-distribution", "damping ×8", "agent recolor"]
x = np.arange(len(shifts4))
full_frozen = [suite["shifts"][s]["frozen"]["success_rate"] for s in shifts4]
low_frozen = [lowdata["shifts"][s]["frozen"]["success_rate"] for s in shifts4]
low_adapt = [lowdata["shifts"][s]["adapt"]["success_rate"] for s in shifts4]
ax.bar(x - 0.28, low_frozen, 0.28, label="10% data, frozen", color=FROZEN_C)
ax.bar(x, low_adapt, 0.28, label="10% data + AdaJEPA", color=ADAPT_C)
ax.bar(x + 0.28, full_frozen, 0.28, label="100% data, frozen", color="#9e9e9e")
ax.set_xticks(x, names4)
ax.set_ylabel("planning success (%)")
ax.set_title("Test-time adaptation partially buys back training data:\n"
             "on the hardest shift, 10%-data + TTA beats the full-data frozen model")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "4_lowdata.png", bbox_inches="tight")

print("wrote:", *[str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.png"))], sep="\n  ")
