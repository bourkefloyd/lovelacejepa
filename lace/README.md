# LACE paper materials

Everything paper-specific for **"LACE: Online Encoder Adaptation by
Learning Anchored Consistent Embeddings"** - see the
[root README](../README.md) for the method overview and results.

The method implementation lives in the sibling testbed
([../adajepa](../adajepa)): `tta.py` (`target_source` knob + anchored goal
encoding), `probes.py` (frozen probe heads), `pushobj.py` (PushObj-mini
benchmark). This directory holds the pre-registered gates, result JSONs,
figure/ablation scripts, and the LaTeX source.

## Layout

```
GATES.md            pre-registered pass/fail gates (written before the grid ran)
runs/               every experiment result JSON (single source of numbers)
scripts/
  paper_figs.py     regenerates every paper figure from runs/*.json
  run_ablations.py  E3 (LR sweep), E5 (low-data), E6 (EMA decay) runners
notebooks/
  lace_paper.ipynb  tables + gate verdicts derived from runs/*.json
paper/
  main.tex, refs.bib, figures/
LICENSE             MIT
```

## Arms naming

| arm | target source | goal encoder | meaning |
|---|---|---|---|
| `frozen` | - | pretrained | no TTA (baseline) |
| `unlaced` | `student` | adapting model | AdaJEPA (paper eq. 4) |
| `laced-frozen` | `frozen` | anchor | LACE, frozen anchor |
| `laced-ema` | `ema` | anchor | LACE, slow-EMA anchor |

## Reproducing the experiment grid

All commands from `adajepa/`; each writes JSON into `lace/runs/`. Pretraining artifacts (`pushobj_base.pt`,
`base_l0.pt`, `base_diverse.pt`, probes) come from the README quickstart in
[../adajepa](../adajepa).

```bash
PY=python  # or path/to/venv/bin/python

# E1 - characterization grids (maze, recipes x target source)
$PY -m adajepa.cli sweep --ckpt runs/base_l0.pt --probes runs/maze_probes.pt \
    --shift high_damping --targets predlast+enclast,predlast --lr-mults 1,0.2 \
    --target-sources student,frozen --out ../lace/runs/e1_maze_high_damping.json
$PY -m adajepa.cli sweep --ckpt runs/base_l0.pt --probes runs/maze_probes.pt \
    --shift high_damping --targets predlast+enclast --lr-mults 1,0.2 \
    --target-sources student,frozen --adapt-enc-lr 3e-4 \
    --out ../lace/runs/e1_maze_high_damping_symlr.json

# E2 - full shift suites (4 arms; CEM + GD)
$PY -m adajepa.cli eval --env pushobj --ckpt runs/pushobj_base.pt \
    --probes runs/pushobj_probes.pt --arms frozen,unlaced,laced-frozen,laced-ema \
    --shifts shape:T,shape:L,shape:Z,shape:plus,shape:I,shape:smallT,shape:cube \
    --horizon 5 --execute-actions 1 --max-replans 40 --cem-samples 160 --cem-iters 6 \
    --out ../lace/runs/e2_pushobj_cem.json
$PY -m adajepa.cli eval --ckpt runs/base_l0.pt --probes runs/maze_probes.pt \
    --arms frozen,unlaced,laced-frozen,laced-ema \
    --shifts default,low_mass,high_damping,blur,snp,dark,red_agent \
    --out ../lace/runs/e2_maze_cem.json
$PY -m adajepa.cli eval --ckpt runs/base_diverse.pt --probes runs/maze_diverse_probes.pt \
    --arms frozen,unlaced,laced-frozen,laced-ema \
    --shifts layout:0,layout:100,layout:101,layout:102 \
    --out ../lace/runs/e2_maze_layout_cem.json

# E3 / E5 / E6 - ablations
$PY ../lace/scripts/run_ablations.py e3
$PY ../lace/scripts/run_ablations.py e5   # needs pushobj_k1/k2 checkpoints
$PY ../lace/scripts/run_ablations.py e6

# Figures
cd ../lace && python scripts/paper_figs.py
```

E7 (deployed screen-agent replay) uses proprietary deployment telemetry and
is not reproducible from this release; `runs/e7_*.json` contain the aggregate
summaries only (no screenshots, product names, or run identifiers), which is
what the paper reports.

## Honest-reporting policy

Every gate's table ships in the paper (or appendix) whether it passes or
fails, with the pre-registered criterion stated (see `GATES.md`). Notably,
the E7 gate FAILED as pre-registered: anchoring fixes the encoder-relocation
mechanism (zero goal drift, encoder-space head AUC intact) but full-recipe
predictor-side damage persists - the paper reports the two-mechanism
decomposition rather than claiming a universal fix.
