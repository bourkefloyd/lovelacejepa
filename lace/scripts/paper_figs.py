"""Generate every LACE paper figure from the run JSONs.

Usage (from lace/):

    python scripts/paper_figs.py            # all figures found
    python scripts/paper_figs.py --only e1  # one family

Reads lace/runs/*.json (written by the adajepa CLI + the deployment replay harness)
and writes lace/paper/figures/*.pdf (+ .png for quick viewing).
Every figure in the paper is produced here - no hand-made figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT.parent / "adajepa"))  # for adajepa.env (numpy-only)

ARM_COLORS = {
    "frozen": "#8a8a8a",
    "unlaced": "#d1495b",
    "adapt": "#d1495b",
    "laced-frozen": "#1f7a8c",
    "laced-ema": "#7fb069",
}
ARM_LABELS = {
    "frozen": "Frozen",
    "unlaced": "Unlaced (AdaJEPA)",
    "adapt": "Unlaced (AdaJEPA)",
    "laced-frozen": "LACE (frozen anchor)",
    "laced-ema": "LACE (EMA anchor)",
}

SEEN_SHAPES = ("shape:T", "shape:L", "shape:Z", "shape:plus")
UNSEEN_SHAPES = ("shape:I", "shape:smallT", "shape:cube")


def _load(name: str) -> dict | None:
    path = RUNS / name
    if not path.exists():
        print(f"  [skip] {name} not found")
        return None
    return json.loads(path.read_text())


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {(OUT / name).relative_to(ROOT)}.pdf")


def _style(ax, title: str = "") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    if title:
        ax.set_title(title, fontsize=10)


def _fig_legend(fig, handles, labels, *, ncol: int, y: float = 0.92) -> None:
    """One shared legend centered above the panels (readable at print size)."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=9, frameon=False, columnspacing=1.4,
               handlelength=1.6)


# ---------------------------------------------------------------------------
# Fig (hook): one dark-shift episode, all three arms on the identical task.
# ---------------------------------------------------------------------------


def fig_hook(shift: str = "dark", ep_idx: int = 6) -> None:
    """Walkthrough: frozen fails, unlaced drifts its goal and fails, LACE
    succeeds -- same maze, same start, same goal observation."""
    data = _load("e2_maze_cem.json")
    if data is None:
        return
    from adajepa.env import EnvConfig, PointMazeEnv, make_shift_config

    env_cfg = make_shift_config(EnvConfig(**data["env"]), shift)
    env = PointMazeEnv(env_cfg)
    cell_px = env_cfg.img_size / env_cfg.grid

    arms = ["frozen", "unlaced", "laced-frozen"]
    titles = {
        "frozen": "frozen: no adaptation",
        "unlaced": "unlaced (AdaJEPA)",
        "laced-frozen": "LACE (frozen anchor)",
    }
    eps = {arm: data["shifts"][shift][arm]["episodes"][ep_idx] for arm in arms}
    goal = np.asarray(eps["frozen"]["goal"])

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.55]})
    for ax, arm in zip(axes[:3], arms):
        ep = eps[arm]
        pos = np.asarray(ep["positions"])
        env.pos = pos[-1].astype(float)
        frame = env.render_frame()
        ax.imshow(frame)
        ax.plot(pos[:, 1] * cell_px, pos[:, 0] * cell_px,
                color=ARM_COLORS[arm], lw=1.8, alpha=0.95)
        ax.plot(pos[0, 1] * cell_px, pos[0, 0] * cell_px, "o", ms=6,
                mfc="white", mec="black", mew=1.0)
        ax.plot(goal[1] * cell_px, goal[0] * cell_px, "*", ms=13,
                mfc="gold", mec="black", mew=0.8)
        outcome = "success" if ep["success"] else "failure"
        ax.set_title(f"{titles[arm]}\n{outcome}", fontsize=9.5,
                     color=ARM_COLORS[arm])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax = axes[3]
    for arm in arms:
        gd = eps[arm]["goal_drift"]
        ls = "--" if arm == "frozen" else "-"
        ax.plot(np.arange(1, len(gd) + 1), gd, ls, color=ARM_COLORS[arm],
                lw=2.0, label=ARM_LABELS[arm])
    ax.set_xlabel("MPC replanning step", fontsize=9)
    ax.set_ylabel(r"goal-latent drift $\|z_g^{(t)}-z_g^{(0)}\|$", fontsize=9)
    _style(ax, "same goal image, re-encoded each step")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "hook_episode")


# ---------------------------------------------------------------------------
# Fig 1 (E1 hook): the trade-off and its dissolution.
# ---------------------------------------------------------------------------


def fig_e1() -> None:
    """Divergence gain vs frozen-head damage, unlaced vs laced (maze sweep)."""
    data = _load("e1_maze_high_damping.json")
    sym = _load("e1_maze_high_damping_symlr.json")
    if data is None:
        return
    frozen_auc = data["arms"]["frozen"]["probes"]["success_auc"]

    def cells(d, source):
        out = {}
        for name, cell in d["arms"].items():
            if name.startswith(source + ":"):
                out[name.split(":", 1)[1]] = cell
        return out

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=False)
    recipes = ["predlast@x0.2", "predlast@x1", "predlast+enclast@x0.2",
               "predlast+enclast@x1"]
    labels = ["pred tail\n0.2x LR", "pred tail\n1x LR", "pred+enc tail\n0.2x LR",
              "pred+enc tail\n1x LR (paper)"]
    x = np.arange(len(recipes))
    w = 0.38
    for i, (source, color, lab) in enumerate((
        ("student", ARM_COLORS["unlaced"], "Unlaced (AdaJEPA)"),
        ("frozen", ARM_COLORS["laced-frozen"], "LACE (frozen anchor)"),
    )):
        cc = cells(data, source)
        succ = [cc[r]["success_rate"] for r in recipes]
        auc = [cc[r]["probes"]["success_auc"] for r in recipes]
        axes[0].bar(x + (i - 0.5) * w, succ, w, color=color, label=lab)
        axes[1].bar(x + (i - 0.5) * w, auc, w, color=color, label=lab)
    axes[0].axhline(data["arms"]["frozen"]["success_rate"], color="k", ls="--",
                    lw=1, label="Frozen model")
    axes[1].axhline(frozen_auc, color="k", ls="--", lw=1, label="Frozen model")
    axes[0].set_ylabel("Planning success (%)")
    axes[1].set_ylabel("Frozen success-head AUC")
    axes[1].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7.5)
        _style(ax)
    handles, hlabels = axes[0].get_legend_handles_labels()
    fig.suptitle("PointMaze high-damping shift: adaptation recipes x target source",
                 fontsize=11, y=1.14)
    _fig_legend(fig, handles, hlabels, ncol=3, y=1.06)
    _save(fig, "e1_maze_grid")

    if sym is not None:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        names, aucs, colors = [], [], []
        names.append("frozen"); aucs.append(sym["arms"]["frozen"]["probes"]["success_auc"])
        colors.append(ARM_COLORS["frozen"])
        for name, cell in sym["arms"].items():
            if name == "frozen":
                continue
            src = name.split(":")[0]
            names.append(name.replace("predlast+enclast", "p+e"))
            aucs.append(cell["probes"]["success_auc"])
            colors.append(ARM_COLORS["unlaced" if src == "student" else "laced-frozen"])
        ax.bar(range(len(names)), aucs, color=colors)
        ax.set_xticks(range(len(names)), names, rotation=30, fontsize=7, ha="right")
        ax.set_ylabel("Frozen success-head AUC")
        ax.set_ylim(0.5, 1.0)
        _style(ax, "Symmetric encoder LR (3e-4)")
        _save(fig, "e1_maze_symlr")


# ---------------------------------------------------------------------------
# Fig 2 (G0/E2): PushObj shape suite bars + success-by-step curves.
# ---------------------------------------------------------------------------


def _suite_bars(data: dict, name: str, shifts: list[str], title: str,
                shift_labels: list[str] | None = None) -> None:
    arms = [a for a in next(iter(data["shifts"].values())).keys()]
    x = np.arange(len(shifts))
    w = 0.8 / len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
    for i, arm in enumerate(arms):
        succ = [data["shifts"][s][arm]["success_rate"] for s in shifts]
        errs = [data["shifts"][s][arm]["success_std"] for s in shifts]
        aucs = [
            (data["shifts"][s][arm].get("probes") or {}).get("success_auc", np.nan)
            for s in shifts
        ]
        color = ARM_COLORS.get(arm, None)
        axes[0].bar(x + (i - len(arms) / 2 + 0.5) * w, succ, w, yerr=errs,
                    color=color, label=ARM_LABELS.get(arm, arm), capsize=2,
                    error_kw={"lw": 0.8})
        axes[1].bar(x + (i - len(arms) / 2 + 0.5) * w, aucs, w, color=color,
                    label=ARM_LABELS.get(arm, arm))
    labels = shift_labels or [s.replace("shape:", "") for s in shifts]
    for ax, ylab in ((axes[0], "Planning success (%)"),
                     (axes[1], "Frozen success-head AUC")):
        ax.set_xticks(x, labels, fontsize=8, rotation=18, ha="right")
        ax.set_ylabel(ylab)
        _style(ax)
    axes[1].set_ylim(0.4, 1.02)
    handles, hlabels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=11, y=1.16)
    _fig_legend(fig, handles, hlabels, ncol=len(arms), y=1.07)
    _save(fig, name)


def fig_e2() -> None:
    push = _load("e2_pushobj_cem.json")
    if push is not None:
        shifts = [s for s in list(SEEN_SHAPES) + list(UNSEEN_SHAPES)
                  if s in push["shifts"]]
        _suite_bars(push, "e2_pushobj_shapes", shifts,
                    "PushObj-mini: seen {T, L, Z, +} vs unseen {I, smallT, cube}")
        _success_by_step(push, "e2_pushobj_curves",
                         [("seen", [s for s in SEEN_SHAPES if s in push["shifts"]]),
                          ("unseen", [s for s in UNSEEN_SHAPES if s in push["shifts"]])])
    maze = _load("e2_maze_cem.json")
    if maze is not None:
        shifts = [s for s in maze["shifts"]]
        _suite_bars(maze, "e2_maze_shifts", shifts,
                    "PointMaze: dynamics / visual shifts",
                    shift_labels=[s.replace("layout:", "layout ") for s in shifts])
        _success_by_step(maze, "e2_maze_curves",
                         [("dynamics", [s for s in ("low_mass", "high_damping")
                                        if s in maze["shifts"]]),
                          ("visual", [s for s in ("blur", "snp", "dark", "red_agent")
                                      if s in maze["shifts"]])])
    layout = _load("e2_maze_layout_cem.json")
    if layout is not None:
        shifts = [s for s in layout["shifts"]]
        _suite_bars(layout, "e2_maze_layouts", shifts,
                    "PointMaze: held-out maze layouts (diverse model)",
                    shift_labels=[s.replace("layout:", "layout ") for s in shifts])


def _success_by_step(data: dict, name: str, groups: list[tuple[str, list[str]]]) -> None:
    max_replans = data["eval"]["max_replans"]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.0),
                             sharey=True, squeeze=False)
    for ax, (gname, shifts) in zip(axes[0], groups):
        if not shifts:
            continue
        arms = list(data["shifts"][shifts[0]].keys())
        for arm in arms:
            curves = []
            for s in shifts:
                cell = data["shifts"][s][arm]
                curve = np.zeros(max_replans)
                n = len(cell["episodes"])
                for ep in cell["episodes"]:
                    if ep["success"] and ep["steps_to_success"] is not None:
                        curve[ep["steps_to_success"] - 1:] += 1
                curves.append(100.0 * curve / max(n, 1))
            mean = np.mean(curves, axis=0)
            ax.plot(np.arange(1, max_replans + 1), mean,
                    color=ARM_COLORS.get(arm), label=ARM_LABELS.get(arm, arm), lw=1.8)
        ax.set_xlabel("MPC replanning step")
        _style(ax, gname)
    axes[0][0].set_ylabel("Cumulative success (%)")
    axes[0][0].legend(fontsize=7.5, frameon=False)
    _save(fig, name)


# ---------------------------------------------------------------------------
# Fig (E3): encoder-LR sweep - collapse safety.
# ---------------------------------------------------------------------------


def fig_e3() -> None:
    data = _load("e3_lr_sweep.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    lrs = data["enc_lrs"]
    # Frozen baseline for the same shift/checkpoint (E3 sweeps adaptation
    # arms only, so the no-TTA reference comes from the E1 grid).
    ref = _load("e1_maze_high_damping.json")
    if ref is not None:
        frozen = ref["arms"]["frozen"]
        axes[0].axhline(frozen["success_rate"], color="k", ls="--", lw=1.2,
                        label="Frozen model (no TTA)")
        axes[1].axhline(frozen["probes"]["success_auc"], color="k", ls="--",
                        lw=1.2)
    for source, color in (("student", ARM_COLORS["unlaced"]),
                          ("frozen", ARM_COLORS["laced-frozen"])):
        rows = data["cells"][source]
        axes[0].plot(lrs, [r["success_rate"] for r in rows], "o-", color=color,
                     label=ARM_LABELS["unlaced" if source == "student" else "laced-frozen"])
        axes[1].plot(lrs, [r["probes"]["success_auc"] for r in rows], "o-", color=color)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Encoder learning rate")
        _style(ax)
    axes[0].set_ylabel("Planning success (%)")
    axes[1].set_ylabel("Frozen success-head AUC")
    handles, hlabels = axes[0].get_legend_handles_labels()
    fig.suptitle("E3: symmetric-LR safety (high_damping shift)", fontsize=11,
                 y=1.14)
    _fig_legend(fig, handles, hlabels, ncol=3, y=1.06)
    _save(fig, "e3_lr_sweep")


# ---------------------------------------------------------------------------
# Fig (E4): goal-latent drift traces.
# ---------------------------------------------------------------------------


def fig_e4() -> None:
    data = _load("e2_pushobj_cem.json") or _load("e2_maze_cem.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    shifts = list(data["shifts"].keys())
    arms = list(data["shifts"][shifts[0]].keys())
    for arm in arms:
        traces = []
        for s in shifts:
            for ep in data["shifts"][s][arm]["episodes"]:
                gd = ep.get("goal_drift", [])
                if gd:
                    traces.append(gd)
        if not traces:
            continue
        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, : len(t)] = t
        mean = np.nanmean(padded, axis=0)
        ax.plot(np.arange(1, max_len + 1), mean, color=ARM_COLORS.get(arm),
                label=ARM_LABELS.get(arm, arm), lw=1.8)
    ax.set_xlabel("MPC replanning step")
    ax.set_ylabel(r"$\|z_{goal}(t) - z_{goal}(0)\|_2$")
    _style(ax, "Goal-latent drift (identical goal observation)")
    ax.legend(fontsize=7.5, frameon=False)
    _save(fig, "e4_goal_drift")


# ---------------------------------------------------------------------------
# Fig (E5): low-data regime.
# ---------------------------------------------------------------------------


def fig_e5() -> None:
    data = _load("e5_lowdata.json")
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    budgets = data["budgets"]
    for arm in data["cells"]:
        rows = data["cells"][arm]
        ax.plot(budgets, [r["success_rate"] for r in rows], "o-",
                color=ARM_COLORS.get(arm), label=ARM_LABELS.get(arm, arm), lw=1.8)
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("Pretraining shape diversity $K$")
    ax.set_ylabel("Planning success (%)")
    _style(ax, "E5: low-data pretraining (unseen shapes)")
    ax.legend(fontsize=7.5, frameon=False)
    _save(fig, "e5_lowdata")


# ---------------------------------------------------------------------------
# Fig (E6): EMA-decay ablation.
# ---------------------------------------------------------------------------


def fig_e6() -> None:
    data = _load("e6_ema_decay.json")
    if data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    decays = [str(d) for d in data["decays"]]
    rows = data["cells"]
    x = np.arange(len(decays))
    axes[0].bar(x, [r["success_rate"] for r in rows], 0.6, color=ARM_COLORS["laced-ema"])
    axes[1].bar(x, [r["probes"]["success_auc"] for r in rows], 0.6,
                color=ARM_COLORS["laced-ema"])
    for ax, ylab in ((axes[0], "Planning success (%)"),
                     (axes[1], "Frozen success-head AUC")):
        ax.set_xticks(x, decays)
        ax.set_xlabel(r"EMA decay $\tau$ (1.0 = frozen anchor)")
        ax.set_ylabel(ylab)
        _style(ax)
    fig.suptitle("E6: how slow must the anchor be?", fontsize=11)
    _save(fig, "e6_ema_decay")


# ---------------------------------------------------------------------------
# Fig (E7): SWM deployment replay 2x2.
# ---------------------------------------------------------------------------


def fig_e7() -> None:
    files = {
        ("unlaced", "full"): "e7_swm_unlaced_full.json",
        ("unlaced", "low"): "e7_swm_unlaced_low.json",
        ("laced", "full"): "e7_swm_laced_full.json",
        ("laced", "low"): "e7_swm_laced_low.json",
    }
    cells = {}
    for key, fname in files.items():
        d = _load(fname)
        if d is not None:
            cells[key] = d
    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    order = [("unlaced", "low"), ("unlaced", "full"), ("laced", "low"), ("laced", "full")]
    labels = ["Unlaced\nlow", "Unlaced\nfull", "LACE\nlow", "LACE\nfull"]
    present = [(k, l) for k, l in zip(order, labels) if k in cells]
    x = np.arange(len(present))
    div = [cells[k]["second_half"]["div_reduction_pct"] for k, _ in present]
    auc = [cells[k].get("adapted_success_auc", np.nan) for k, _ in present]
    frozen_auc = next(iter(cells.values())).get("frozen_success_auc", np.nan)
    colors = [ARM_COLORS["unlaced"] if k[0] == "unlaced" else ARM_COLORS["laced-frozen"]
              for k, _ in present]
    axes[0].bar(x, div, 0.6, color=colors)
    axes[0].set_ylabel("2nd-half divergence reduction (%)")
    axes[1].bar(x, auc, 0.6, color=colors)
    axes[1].axhline(frozen_auc, color="k", ls="--", lw=1, label="Frozen heads")
    axes[1].set_ylabel("Deployed success-head AUC")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend(fontsize=7.5, frameon=False)
    for ax in axes:
        ax.set_xticks(x, [l for _, l in present], fontsize=8)
        _style(ax)
    fig.suptitle("E7: deployed screen-agent replay (106+ runs, 5k+ steps)", fontsize=11)
    _save(fig, "e7_swm_replay")


# ---------------------------------------------------------------------------
# Fig (E7b): mechanism decomposition - encoder-only vs predictor-only arms.
# ---------------------------------------------------------------------------


def fig_e7b() -> None:
    files = {
        ("pred-only", "student"): "e7b_swm_predonly_student.json",
        ("pred-only", "frozen"): "e7b_swm_predonly_laced.json",
        ("enc-only", "student"): "e7b_swm_enconly_student.json",
        ("enc-only", "frozen"): "e7b_swm_enconly_laced.json",
    }
    cells = {k: d for k, f in files.items() if (d := _load(f)) is not None}
    if not cells:
        return
    order = [("pred-only", "student"), ("pred-only", "frozen"),
             ("enc-only", "student"), ("enc-only", "frozen")]
    labels = ["Pred-only\nunlaced", "Pred-only\nLACE",
              "Enc-only\nunlaced", "Enc-only\nLACE"]
    present = [(k, l) for k, l in zip(order, labels) if k in cells]
    x = np.arange(len(present))
    colors = [ARM_COLORS["unlaced" if k[1] == "student" else "laced-frozen"]
              for k, _ in present]
    frozen_auc = next(iter(cells.values()))["frozen_success_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    axes[0].bar(x, [cells[k]["adapted_success_auc"] for k, _ in present],
                0.6, color=colors)
    axes[0].set_ylabel("Success-head AUC (predicted latent)")
    axes[1].bar(x, [cells[k]["adapted_success_auc_enc"] for k, _ in present],
                0.6, color=colors)
    axes[1].set_ylabel("Success-head AUC (encoded next frame)")
    for ax in axes:
        ax.axhline(frozen_auc, color="k", ls="--", lw=1)
        ax.set_xticks(x, [l for _, l in present], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        _style(ax)
    axes[0].text(0.02, frozen_auc + 0.01, "frozen model", fontsize=7,
                 transform=axes[0].get_yaxis_transform())
    fig.suptitle("E7b: which parameters cause the damage?", fontsize=11)
    _save(fig, "e7b_mechanism")


FAMILIES = {
    "hook": fig_hook,
    "e1": fig_e1,
    "e2": fig_e2,
    "e3": fig_e3,
    "e4": fig_e4,
    "e5": fig_e5,
    "e6": fig_e6,
    "e7": fig_e7,
    "e7b": fig_e7b,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated families (e1..e7)")
    args = ap.parse_args()
    wanted = args.only.split(",") if args.only else list(FAMILIES)
    for fam in wanted:
        print(f"[{fam}]")
        FAMILIES[fam]()


if __name__ == "__main__":
    main()
