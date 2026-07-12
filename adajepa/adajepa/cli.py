"""CLI for the AdaJEPA reproduction: gen-data / train / eval / sweep.

Run from ``adajepa/``:

    python -m adajepa.cli gen-data --out data/pointmaze.npz
    python -m adajepa.cli train --data data/pointmaze.npz --out runs/base.pt
    python -m adajepa.cli eval --ckpt runs/base.pt --out runs/suite_cem.json
    python -m adajepa.cli sweep --ckpt runs/base.pt --shift high_damping \
        --out runs/ablation_targets.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DataConfig, generate_dataset, generate_pushobj_dataset
from .device import torch_device
from .env import SHIFT_NAMES, EnvConfig
from .eval import EvalConfig, run_setting, run_suite
from .model import ModelConfig, load_checkpoint
from .planner import PlannerConfig
from .pushobj import PUSHOBJ_SHIFT_NAMES, PushObjConfig
from .tta import ADAPT_TARGETS, AdaptConfig


def _add_env_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--env", choices=("pointmaze", "pushobj"), default="pointmaze")
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--frame-stack", type=int, default=2)
    p.add_argument("--layout-seed", type=int, default=0)
    p.add_argument("--shape", default="T", help="pushobj block shape")
    p.add_argument("--goal-steps", type=int, default=8, help="pushobj task length")


def _env_cfg(args) -> EnvConfig | PushObjConfig:
    if args.env == "pushobj":
        return PushObjConfig(
            shape=args.shape,
            img_size=args.img_size,
            frame_stack=args.frame_stack,
            goal_steps=args.goal_steps,
        )
    return EnvConfig(
        grid=args.grid,
        img_size=args.img_size,
        frame_stack=args.frame_stack,
        layout_seed=args.layout_seed,
    )


def cmd_gen_data(args) -> None:
    if args.env == "pushobj":
        data_cfg = DataConfig(
            n_trajectories=args.trajectories,
            traj_len=args.traj_len,
            shapes=tuple(args.shapes.split(",")),
            seed=args.seed,
        )
        info = generate_pushobj_dataset(_env_cfg(args), data_cfg, Path(args.out))
    else:
        layout_seeds = tuple(int(s) for s in args.layouts.split(","))
        data_cfg = DataConfig(
            n_trajectories=args.trajectories,
            traj_len=args.traj_len,
            layout_seeds=layout_seeds,
            seed=args.seed,
        )
        info = generate_dataset(_env_cfg(args), data_cfg, Path(args.out))
    print(json.dumps(info, indent=2))


def cmd_train(args) -> None:
    from .train import TrainConfig, train  # torch-heavy import kept lazy

    device = torch_device(args.device)
    model_cfg = ModelConfig(
        img_size=args.img_size,
        frame_stack=args.frame_stack,
        latent_dim=args.latent_dim,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pred_steps=args.pred_steps,
        motion_weight=args.motion_weight,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
    )
    info = train(Path(args.data), Path(args.out), model_cfg, train_cfg, device)
    print(json.dumps(info, indent=2))


def _planner_cfg(args) -> PlannerConfig:
    return PlannerConfig(
        kind=args.planner,
        horizon=args.horizon,
        execute_actions=args.execute_actions,
        cem_samples=args.cem_samples,
        cem_iters=args.cem_iters,
    )


def _eval_cfg(args) -> EvalConfig:
    return EvalConfig(
        max_replans=args.max_replans,
        episodes=args.episodes,
        seeds=tuple(int(s) for s in args.eval_seeds.split(",")),
    )


def _base_adapt_cfg(args) -> AdaptConfig:
    return AdaptConfig(
        lr=args.adapt_lr,
        enc_lr=args.adapt_enc_lr,
        steps=args.adapt_steps,
        buffer_size=args.buffer_size,
        target=args.adapt_target,
        target_source=args.target_source,
        ema_decay=args.ema_decay,
        goal_encoder=args.goal_encoder,
    )


def make_arm(name: str, base: AdaptConfig) -> AdaptConfig | None:
    """Named arms for the LACE suites (unlaced = AdaJEPA baseline)."""
    from dataclasses import replace

    if name == "frozen":
        return None
    if name in ("adapt", "unlaced"):
        return replace(base, target_source="student", goal_encoder="model")
    if name == "laced-frozen":
        return replace(base, target_source="frozen", goal_encoder="anchor")
    if name == "laced-ema":
        return replace(base, target_source="ema", goal_encoder="anchor")
    raise ValueError(f"unknown arm {name!r}")


def cmd_eval(args) -> None:
    device = torch_device(args.device)
    model = load_checkpoint(args.ckpt, device)
    default_shifts = PUSHOBJ_SHIFT_NAMES if args.env == "pushobj" else SHIFT_NAMES
    shifts = args.shifts.split(",") if args.shifts else list(default_shifts)
    adapt_cfg = _base_adapt_cfg(args)
    probes = None
    if args.probes:
        from .probes import load_probes

        probes = load_probes(Path(args.probes), device)
    arms = None
    if args.arms:
        arms = [(name, make_arm(name, adapt_cfg)) for name in args.arms.split(",")]
    run_suite(
        model,
        _env_cfg(args),
        shifts,
        _planner_cfg(args),
        _eval_cfg(args),
        adapt_cfg,
        Path(args.out),
        device,
        probes=probes,
        arms=arms,
    )
    print(f"wrote {args.out}")


def cmd_train_probes(args) -> None:
    from .probes import ProbeConfig, save_probes, train_probes

    device = torch_device(args.device)
    model = load_checkpoint(args.ckpt, device)
    cfg = ProbeConfig(
        env_kind=args.env,
        pairs_per_traj=args.pairs_per_traj,
        epochs=args.epochs,
        seed=args.seed,
    )
    heads, info = train_probes(model, Path(args.data), cfg, device)
    save_probes(heads, Path(args.out), info)
    print(json.dumps(info, indent=2))


def cmd_sweep(args) -> None:
    """Ablation sweep on one shift: (targets x lr multipliers) x target sources."""
    from .eval import shift_config

    device = torch_device(args.device)
    model = load_checkpoint(args.ckpt, device)
    env_cfg = shift_config(_env_cfg(args), args.shift)
    planner_cfg = _planner_cfg(args)
    eval_cfg = _eval_cfg(args)
    probes = None
    if args.probes:
        from .probes import load_probes

        probes = load_probes(Path(args.probes), device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {"shift": args.shift, "arms": {}}

    arms: list[tuple[str, AdaptConfig | None]] = [("frozen", None)]
    for source in args.target_sources.split(","):
        for target in args.targets.split(","):
            for mult in (float(m) for m in args.lr_mults.split(",")):
                name = f"{source}:{target}@x{mult:g}"
                arms.append(
                    (
                        name,
                        AdaptConfig(
                            lr=args.adapt_lr * mult,
                            enc_lr=args.adapt_enc_lr * mult,
                            target=target,
                            target_source=source,
                            ema_decay=args.ema_decay,
                            goal_encoder="anchor" if source != "student" else "model",
                        ),
                    )
                )
    for name, cfg in arms:
        cell = run_setting(
            model, env_cfg, planner_cfg, eval_cfg, cfg, device, probes=probes
        )
        cell.pop("episodes")  # keep the sweep file small
        results["arms"][name] = cell
        probe_str = ""
        if probes is not None and cell.get("probes", {}).get("success_auc") is not None:
            probe_str = f" succAUC={cell['probes']['success_auc']:.3f}"
        print(
            f"[sweep] {name:>36s}: {cell['success_rate']:5.1f}% "
            f"+/- {cell['success_std']:.1f}{probe_str}",
            flush=True,
        )
        out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="adajepa", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-data", help="generate offline exploration trajectories")
    _add_env_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--trajectories", type=int, default=1500)
    p.add_argument("--traj-len", type=int, default=32)
    p.add_argument("--layouts", default="0", help="comma-separated layout seeds (maze)")
    p.add_argument("--shapes", default="T,L,Z,plus", help="comma-separated shapes (pushobj)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_gen_data)

    p = sub.add_parser("train", help="train the JEPA world model offline")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--frame-stack", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pred-steps", type=int, default=2)
    p.add_argument("--motion-weight", type=float, default=1.0)
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("train-probes", help="train frozen probe heads on a checkpoint")
    p.add_argument("--env", choices=("pointmaze", "pushobj"), default="pointmaze")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="offline npz the probes train on")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs-per-traj", type=int, default=10)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_train_probes)

    for name, fn in (("eval", cmd_eval), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        _add_env_args(p)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--planner", choices=("cem", "gd"), default="cem")
        p.add_argument("--horizon", type=int, default=10)
        p.add_argument("--execute-actions", type=int, default=2)
        p.add_argument("--cem-samples", type=int, default=96)
        p.add_argument("--cem-iters", type=int, default=5)
        p.add_argument("--max-replans", type=int, default=30)
        p.add_argument("--episodes", type=int, default=15)
        p.add_argument("--eval-seeds", default="0,1")
        p.add_argument("--adapt-lr", type=float, default=3e-4)
        p.add_argument("--adapt-enc-lr", type=float, default=1e-5)
        p.add_argument("--adapt-steps", type=int, default=1)
        p.add_argument("--buffer-size", type=int, default=5)
        p.add_argument("--adapt-target", default="predlast+enclast")
        p.add_argument(
            "--target-source",
            choices=("student", "frozen", "ema"),
            default="student",
            help="adaptation target: student = AdaJEPA, frozen/ema = LACE anchor",
        )
        p.add_argument("--ema-decay", type=float, default=0.996)
        p.add_argument(
            "--goal-encoder",
            choices=("model", "anchor"),
            default="model",
            help="which encoder embeds the goal each replan (anchor = LACE)",
        )
        p.add_argument("--probes", default="", help="frozen probe-head checkpoint")
        p.add_argument("--device", default="auto")
        if name == "eval":
            p.add_argument(
                "--shifts",
                default="",
                help=f"comma-separated; default all of {SHIFT_NAMES} "
                "(layout:<seed> and shape:<name> also accepted)",
            )
            p.add_argument(
                "--arms",
                default="",
                help="comma-separated arm names: frozen,unlaced,laced-frozen,"
                "laced-ema (default: frozen,adapt with the flags above)",
            )
        else:
            p.add_argument("--shift", required=True)
            p.add_argument("--targets", default=",".join(ADAPT_TARGETS[:4]))
            p.add_argument("--lr-mults", default="1")
            p.add_argument("--target-sources", default="student")
        p.set_defaults(fn=fn)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
