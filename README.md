# LovelaceJEPA / LACE

Code, experiments, and paper for **"LovelaceJEPA: Online Encoder Adaptation by
Learning Anchored Consistent Embeddings"** (Bourke Floyd IV, 2026).

Test-time adaptation (TTA) lets a latent world model track distribution shift
inside the closed loop of MPC, but the standard objective is self-referential:
the target `sg(E_student(o_t+1))` is produced by the very encoder being
adapted, so adaptation can reduce prediction error by *relocating the latent
space* rather than modeling the new dynamics - silently damaging every frozen
consumer (goal embeddings, success/progress heads, the planner's cost) while
the adaptation loss looks excellent. **LACE** (Learning Anchored Consistent
Embeddings) replaces the target with `E_frozen(o_t+1)` from a frozen or
slow-EMA copy of the pretrained encoder, anchoring adaptation to the
pretrained manifold. The change is one symbol in the objective and adds no
planning-time cost.

On 110 replayed deployment runs (5,434 steps), anchoring is necessary but not
sufficient: prequential replay decomposes the head damage into an
encoder-relocation component that LACE removes and a predictor-drift
component that persists (isolated via parameter-subset ablations). Two
pre-registered gates failed and the paper says so explicitly - see
`lace/GATES.md` and the honest-reporting policy in `lace/README.md`.

## Layout

```
adajepa/   isolated AdaJEPA reproduction: numpy PointMaze + pymunk PushObj-mini
           envs, miniature JEPA world model (1.2M params), CEM/GD planners,
           TestTimeAdapter with the LACE anchor knob, frozen probe heads.
           See adajepa/README.md for the quickstart.
lace/      everything paper-specific: pre-registered gates (GATES.md), result
           JSONs (runs/), figure + ablation scripts, the paper source
           (paper/main.tex), and the gate-verdict notebook.
           See lace/README.md for the experiment grid.
```

## Reproducing

1. Install dependencies: `pip install -r adajepa/requirements.txt`
   (torch, numpy, pymunk, matplotlib, jupyter).
2. Follow the quickstart in [adajepa/README.md](adajepa/README.md) to generate
   data and pretrain the world models (minutes on Apple MPS or CPU).
3. Follow [lace/README.md](lace/README.md) to run the E1-E6 experiment grid;
   each command writes a JSON into `lace/runs/`.
4. `cd lace && python scripts/paper_figs.py` regenerates every paper figure
   from `lace/runs/*.json`.

All committed result JSONs in `lace/runs/` and checkpoints in `adajepa/runs/`
are the exact artifacts behind the paper's numbers, with seeds recorded in
each file. E7 (deployed screen-agent replay) uses proprietary deployment
telemetry and is not reproducible from this release; `lace/runs/e7_*.json`
contain aggregate summaries only, which is what the paper reports.

## Citation

```bibtex
@misc{floyd2026lovelacejepa,
  title  = {LovelaceJEPA: Online Encoder Adaptation by Learning Anchored
            Consistent Embeddings},
  author = {Floyd IV, Bourke},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```

## License

MIT - see [LICENSE](LICENSE).
