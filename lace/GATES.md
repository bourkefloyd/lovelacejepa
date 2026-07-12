# LACE: pre-registered gates

Written BEFORE the experiment grid runs (playbook discipline: "write the gate
first"). Each gate has a pass/fail criterion fixed in advance so results are
tuning signals or verdicts, never judgment calls after the fact.

Date registered: 2026-07-07.

## Terms

- **unlaced** = AdaJEPA baseline: adaptation target `sg(E_student(o_t+1))`
  from the adapting encoder itself (`target_source=student`).
- **laced-frozen** = LACE: target `E_frozen(o_t+1)` from a frozen copy of the
  pretrained encoder (`target_source=frozen`, `goal_encoder=anchor`).
- **laced-ema** = LACE with a slow-EMA anchor (`target_source=ema`).
- **full recipe** = the paper's default: `predlast+enclast`, predictor LR
  3e-4 (training LR), encoder LR 1e-5, buffer 5, 1 step per replan.
- **probes** = frozen MLP heads (success classifier, progress regressor,
  state readout) trained once on the pretrained latents, never updated.
- **divergence** = latent prediction MSE on upcoming transitions, scored
  before adapting on them (prequential).

## G0 - benchmark validity (PushObj-mini earns the right to carry claims)

PASS requires all of:

1. Frozen model shows a clear seen-vs-unseen success gap: mean success on
   seen shapes {T, L, Z, plus} exceeds mean success on unseen shapes
   {I, smallT, cube} by >= 10 pp.
2. `unlaced` TTA improves mean success on unseen shapes vs frozen (the
   paper's headline result reproduces qualitatively).
3. Tasks are oracle-solvable: ground-truth-state CEM >= 80% success within
   the replan budget (verified 2026-07-07: 8/8 at goal_steps=15,
   tol=(0.5, 0.6); recorded in `adajepa/pushobj.py` docstring).

If G0 fails: fix env/task scale (descope smallT, adjust tolerances) before
running anything else; PushObj results cannot enter the paper until G0 passes.

## G1 - the headline (phenomenon transfers + LACE dissolves it)

On PushObj-mini (seen + unseen shapes) and PointMaze (dynamics + layout
shifts), with probes attached:

1. **Phenomenon**: `unlaced` at full recipe must measurably damage frozen
   consumers: probe success-AUC or progress-corr drop >= 0.03 vs the frozen
   arm (pooled over shifts), or goal-latent drift visibly corrupts planning
   cost (E4). This shows the SWM finding is not SWM-specific.
2. **Fix**: `laced-frozen` at the SAME full recipe must keep >= 80% of
   `unlaced`'s divergence reduction (per-replan pred-loss drop vs frozen)
   AND keep probe success-AUC within 0.02 of the frozen arm.

If G1.1 fails on public benchmarks (heads only break on SWM): reframe the
paper around the SWM evidence plus boundary conditions - weaker but honest.
If G1.2 fails: LACE does not dissolve the trade-off; the paper becomes a
characterization/negative-result paper (the 2x2 table stands alone).

## G2 - planning success (LACE must not cost the original gains)

`laced-frozen` planning success >= `unlaced` - 5 pp on every shift family
(seen shapes, unseen shapes, dynamics, layout), and >= frozen - 3 pp
in-distribution (no in-distribution harm). Comparisons at matched episode
counts and seeds; "within seed noise" = one pooled std.

## G3 - collapse safety (anchoring removes the asymmetric-LR hack)

With SYMMETRIC learning rates (encoder LR = predictor LR = 3e-4, target
`predlast+enclast` or `all`), over the E3 sweep:

1. `unlaced` must show latent degradation: `embed_std` decay > 20% from the
   pretrained value, or probe success-AUC drop >= 0.05.
2. `laced-frozen` under the same symmetric LRs must avoid both.

If G3.1 fails (student target is stable even at symmetric LR at this scale),
report it: the collapse-risk argument is then theoretical at miniature scale
and the paper leans on G1/G2 only.

## E7 (SWM replay) - supporting evidence gate

Same gate as the original Phase-1 replay eval: adapted second-half
divergence reduction > 5% AND cf_top1 regression <= 0.02, now required to
hold for `laced-frozen` at FULL recipe (where `unlaced` at full recipe
failed with success-AUC 0.947 -> 0.846).

## Honest-reporting rule

Every gate's table ships in the paper (or its appendix) whether it passes or
fails, with the pre-registered criterion stated.
