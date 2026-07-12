"""E3 / E5 / E6 ablation runners for the LACE paper.

Run from ``adajepa/`` (imports the adajepa package):

    python ../lace/scripts/run_ablations.py e3   # encoder-LR / collapse sweep
    python ../lace/scripts/run_ablations.py e5   # shape-diversity low-data grid
    python ../lace/scripts/run_ablations.py e6   # EMA-decay ablation

Each writes one JSON into lace/runs/ consumed by paper_figs.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adajepa"))

from adajepa.device import torch_device  # noqa: E402
from adajepa.env import EnvConfig, make_shift_config  # noqa: E402
from adajepa.eval import EvalConfig, run_setting, shift_config  # noqa: E402
from adajepa.model import load_checkpoint  # noqa: E402
from adajepa.planner import PlannerConfig  # noqa: E402
from adajepa.probes import load_probes  # noqa: E402
from adajepa.pushobj import PushObjConfig  # noqa: E402
from adajepa.tta import AdaptConfig  # noqa: E402

ADAJEPA = Path(__file__).resolve().parents[2] / "adajepa"
RUNS = Path(__file__).resolve().parents[1] / "runs"

# Ablation scale: half the E2 budget (ablations need trends, not tight CIs).
MAZE_PLANNER = PlannerConfig(kind="cem", horizon=10, execute_actions=2)
MAZE_EVAL = EvalConfig(max_replans=30, episodes=15, seeds=(0,))
PUSH_PLANNER = PlannerConfig(
    kind="cem", horizon=5, execute_actions=1, cem_samples=160, cem_iters=6
)
PUSH_EVAL = EvalConfig(max_replans=40, episodes=10, seeds=(0,))


def _strip(cell: dict) -> dict:
    cell.pop("episodes", None)
    return cell


def e3() -> None:
    """Encoder-LR sweep at SYMMETRIC LRs (G3): student vs frozen source."""
    device = torch_device("auto")
    model = load_checkpoint(ADAJEPA / "runs" / "base_l0.pt", device)
    probes = load_probes(ADAJEPA / "runs" / "maze_probes.pt", device)
    env_cfg = make_shift_config(EnvConfig(), "high_damping")
    enc_lrs = [1e-5, 1e-4, 3e-4, 1e-3]
    out: dict = {"enc_lrs": enc_lrs, "shift": "high_damping", "cells": {}}
    for source in ("student", "frozen"):
        rows = []
        for lr in enc_lrs:
            cfg = AdaptConfig(
                lr=lr, enc_lr=lr, target="predlast+enclast",
                target_source=source,
                goal_encoder="anchor" if source == "frozen" else "model",
            )
            t0 = time.time()
            cell = _strip(run_setting(
                model, env_cfg, MAZE_PLANNER, MAZE_EVAL, cfg, device, probes=probes
            ))
            rows.append(cell)
            print(f"[e3] {source} enc_lr={lr:g}: {cell['success_rate']:.1f}% "
                  f"succAUC={cell['probes'].get('success_auc')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out["cells"][source] = rows
        (RUNS / "e3_lr_sweep.json").write_text(json.dumps(out, indent=2))
    print("wrote e3_lr_sweep.json")


def e5() -> None:
    """Shape-diversity low-data grid: K in {1, 2, 4} train shapes, unseen evals."""
    device = torch_device("auto")
    models = {
        1: ("pushobj_k1.pt", "pushobj_k1_probes.pt"),
        2: ("pushobj_k2.pt", "pushobj_k2_probes.pt"),
        4: ("pushobj_base.pt", "pushobj_probes.pt"),
    }
    unseen = ["shape:I", "shape:cube"]
    arms = {
        "frozen": None,
        "unlaced": AdaptConfig(target_source="student"),
        "laced-frozen": AdaptConfig(target_source="frozen", goal_encoder="anchor"),
    }
    out: dict = {"budgets": list(models), "unseen": unseen, "cells": {}}
    for arm_name, cfg in arms.items():
        rows = []
        for k, (ckpt, probe_ckpt) in models.items():
            model = load_checkpoint(ADAJEPA / "runs" / ckpt, device)
            probes = load_probes(ADAJEPA / "runs" / probe_ckpt, device)
            succ, aucs = [], []
            for shift in unseen:
                env_cfg = shift_config(PushObjConfig(), shift)
                cell = run_setting(
                    model, env_cfg, PUSH_PLANNER, PUSH_EVAL, cfg, device, probes=probes
                )
                succ.append(cell["success_rate"])
                auc = (cell.get("probes") or {}).get("success_auc")
                if auc is not None and auc == auc:
                    aucs.append(auc)
            row = {
                "success_rate": round(sum(succ) / len(succ), 2),
                "per_shift": dict(zip(unseen, succ)),
                "probes": {
                    "success_auc": round(sum(aucs) / len(aucs), 4) if aucs else None
                },
            }
            rows.append(row)
            print(f"[e5] {arm_name} K={k}: {row['success_rate']:.1f}% "
                  f"succAUC={row['probes']['success_auc']}", flush=True)
        out["cells"][arm_name] = rows
        (RUNS / "e5_lowdata.json").write_text(json.dumps(out, indent=2))
    print("wrote e5_lowdata.json")


def e6() -> None:
    """EMA-decay ablation for the laced-ema anchor on unseen shapes."""
    device = torch_device("auto")
    model = load_checkpoint(ADAJEPA / "runs" / "pushobj_base.pt", device)
    probes = load_probes(ADAJEPA / "runs" / "pushobj_probes.pt", device)
    decays = [0.9, 0.996, 1.0]
    unseen = ["shape:I", "shape:cube"]
    out: dict = {"decays": decays, "unseen": unseen, "cells": []}
    for tau in decays:
        cfg = AdaptConfig(
            target_source="frozen" if tau >= 1.0 else "ema",
            ema_decay=tau, goal_encoder="anchor",
        )
        succ, aucs = [], []
        for shift in unseen:
            env_cfg = shift_config(PushObjConfig(), shift)
            cell = run_setting(
                model, env_cfg, PUSH_PLANNER, PUSH_EVAL, cfg, device, probes=probes
            )
            succ.append(cell["success_rate"])
            auc = (cell.get("probes") or {}).get("success_auc")
            if auc is not None and auc == auc:
                aucs.append(auc)
        row = {
            "success_rate": round(sum(succ) / len(succ), 2),
            "probes": {"success_auc": round(sum(aucs) / len(aucs), 4) if aucs else None},
        }
        out["cells"].append(row)
        print(f"[e6] tau={tau}: {row['success_rate']:.1f}% "
              f"succAUC={row['probes']['success_auc']}", flush=True)
        (RUNS / "e6_ema_decay.json").write_text(json.dumps(out, indent=2))
    print("wrote e6_ema_decay.json")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e3"
    {"e3": e3, "e5": e5, "e6": e6}[which]()
