# AdaJEPA reproduction (arXiv 2606.32026)

An isolated, self-contained reproduction of **"AdaJEPA: An Adaptive Latent
World Model"** (Wang, Bounou, LeCun, Ren - NYU, 2026): a JEPA world model that
is *adapted at test time* inside the closed loop of MPC. After each executed
action chunk, the observed transition `(o_t, a_t, o_{t+1})` provides a
self-supervised latent prediction target; one gradient step on a small
parameter subset recalibrates the model before the next replan.

**Isolation**: this directory imports nothing from `lab/` or `backend/` - it
is a standalone research area intended to back its own paper. The environments
are a from-scratch numpy PointMaze (no MuJoCo/gym) and a miniature pymunk
PushObj (pusher + polyomino block), and the world model is a miniature JEPA
(1.2M params) that trains in minutes on Apple MPS.

This reproduction also hosts the **LovelaceJEPA / LACE** extensions
(`../lace/`): frozen probe heads (`probes.py`), the anchored adaptation
target (`tta.py: target_source=student|frozen|ema`), and the named arms
`frozen / unlaced / laced-frozen / laced-ema` in `eval --arms`.

## Claims under test (mapped from the paper)

| # | Paper claim | Where tested here |
|---|---|---|
| 1 | One TTA gradient step per replan improves planning success under **dynamics shifts** (low mass x0.2, high damping x20) | `eval` on `low_mass` / `high_damping` |
| 2 | Same for **visual shifts** (blur, salt-and-pepper, dark, agent recolor) | `eval` on `blur` / `snp` / `dark` / `red_agent` |
| 3 | Same for **layout shifts** (held-out mazes) | `eval` with `layout:<seed>` on the diverse-maze checkpoint |
| 4 | TTA is **safe in-distribution** (no harm when the frozen model is good) | `eval` on `default` |
| 5 | Adapted success keeps **rising with replanning steps** while frozen saturates | `success_by_step` curves in the notebook |
| 6 | The gains are **not sensitive to the adaptation target**; last-layer updates suffice | `sweep --targets ...` |
| 7 | TTA **compensates for scarce training data** | train with `--max-trajectories`, compare |
| 8 | Adaptation reduces the per-replan **latent prediction loss** | per-episode `pred_losses` traces |

## Layout

```
adajepa/            the library
  env.py            numpy PointMaze: dynamics / visual / layout shifts
  pushobj.py        pymunk PushObj-mini: shape shifts (train T/L/Z/+, test I/smallT/cube)
  data.py           OU-exploration trajectory generation + training windows
  model.py          JEPA world model (CNN encoder, residual-MLP predictor)
  train.py          offline training loop (MPS-safe)
  planner.py        CEM + GD trajectory optimizers (MPC, eq. 3)
  tta.py            TestTimeAdapter (algorithm 1; eq. 4-5) + LACE anchor knob
  probes.py         frozen probe heads (success / progress / state readout)
  eval.py           plan-execute-adapt-replan episodes + suites
  cli.py            gen-data / train / train-probes / eval / sweep
notebooks/
  adajepa_paper_walkthrough.ipynb   the paper-assumption walkthrough
data/               generated datasets (npz, gitignored)
runs/               checkpoints + result JSONs
```

## Quickstart

All commands from `adajepa/`, using a venv with the dependencies in
`requirements.txt` installed:

```bash
PY=python  # or path/to/venv/bin/python

# 1. Offline exploration data (single fixed maze; ~20 s)
$PY -m adajepa.cli gen-data --out data/pointmaze_l0.npz \
    --trajectories 1500 --layouts 0

# 2. Train the JEPA world model (~1 min on MPS)
$PY -m adajepa.cli train --data data/pointmaze_l0.npz --out runs/base_l0.pt

# 3. Frozen vs adapt across the shift suite (writes JSON incrementally)
$PY -m adajepa.cli eval --ckpt runs/base_l0.pt --out runs/suite_cem.json

# 4. Ablation: adaptation targets x learning rates on one shift
$PY -m adajepa.cli sweep --ckpt runs/base_l0.pt --shift high_damping \
    --targets predlast+enclast,predlast,enclast,predfirst+enclast \
    --lr-mults 0.2,1,5 --out runs/ablation_high_damping.json

# Layout shifts: train on 8 mazes, evaluate held-out mazes
$PY -m adajepa.cli gen-data --out data/pointmaze_diverse.npz \
    --trajectories 2400 --layouts 0,1,2,3,4,5,6,7
$PY -m adajepa.cli train --data data/pointmaze_diverse.npz --out runs/base_diverse.pt
$PY -m adajepa.cli eval --ckpt runs/base_diverse.pt \
    --shifts layout:100,layout:101,layout:102 --out runs/suite_layout.json

# PushObj-mini: shape shifts (train shapes T,L,Z,+; test I, smallT, cube)
$PY -m adajepa.cli gen-data --env pushobj --out data/pushobj_tlzp.npz \
    --trajectories 8000 --shapes T,L,Z,plus
$PY -m adajepa.cli train --data data/pushobj_tlzp.npz --out runs/pushobj_base.pt \
    --epochs 10 --pred-steps 4
$PY -m adajepa.cli train-probes --env pushobj --ckpt runs/pushobj_base.pt \
    --data data/pushobj_tlzp.npz --out runs/pushobj_probes.pt
$PY -m adajepa.cli eval --env pushobj --ckpt runs/pushobj_base.pt \
    --probes runs/pushobj_probes.pt \
    --shifts shape:T,shape:L,shape:Z,shape:plus,shape:I,shape:smallT,shape:cube \
    --arms frozen,unlaced,laced-frozen,laced-ema \
    --horizon 5 --execute-actions 1 --max-replans 40 --cem-samples 160 \
    --out runs/pushobj_suite.json
```

## Deliberate miniaturizations vs the paper

- Environment: numpy point-maze instead of MuJoCo PointMaze / pymunk PushT
  (same dynamics knobs: force -> mass -> damping -> wall collisions).
- Encoder: 4-conv CNN with a global latent (theirs: ResNet global features);
  predictor: residual MLP (theirs: transformer). Module names still expose the
  paper's adaptation targets (predfirst/predlast/enclast).
- History: 2 stacked frames (theirs: 3), frameskip 1 (theirs: 5).
- Anti-collapse: stop-grad + VICReg variance/covariance insurance on
  LayerNorm'd embeddings (the paper permits either stabilizer), plus a
  temporal-contrast "motion" hinge. Both additions guard against shortcut
  solutions we actually hit at this scale (constant encoder; layout-identity
  encoding on multi-maze data) - see the notebook's findings section.
- Scale: 15-25 episodes x 2 seeds per cell (theirs: 50 x 3), 30 max replans.

The shape-shift axis (PushObj) is covered by `pushobj.py`: a pymunk pusher +
polyomino block env with the paper's train/test shape split, contact-biased
data generation, and symmetry-aware success scoring (a cube rotated 90 deg is
identical; angle errors are scored modulo each shape's symmetry period).
PushObj tasks are goal states 8 steps ahead under the data policy, verified
oracle-solvable (8/8 by ground-truth-state CEM) so learned-model failures are
attributable to the model, not the task.

## Dependencies

Only `torch`, `numpy`, `pymunk` (PushObj), `matplotlib` (notebook), `jupyter`.
See `requirements.txt`.
