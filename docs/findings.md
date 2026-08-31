# Findings

Working notes for the paper's results section. Numbers come from
`results/summary.csv` (59 runs: 5 streams x 5 strategies x seeds).

**The MLflow store holds only the most recent run.** A branch checkout restored
an older tracked `mlflow.db` over the populated one, discarding the per-step
`windowed_accuracy` curves. Every final metric survives in
`results/raw_runs.csv`, which is what these notes are based on. Re-run
`src/experiments.py` if the over-time curves are needed for figures.

## The headline result is not "ARF wins"

ARF wins on three of five streams, but the interesting finding is **where
adaptation stops helping and starts hurting**.

| Stream | Drift | Best strategy | Baseline | Adaptation helps? |
|---|---|---|---|---|
| `sea_abrupt` | Abrupt, injected | ARF (0.976, sep.) | 0.9366 | Yes — clearly |
| `sea_gradual` | Gradual, injected | ARF | 0.9348 | Yes — clearly |
| `hyperplane` | Continuous rotation | detector_full_retrain — **not separable** (margin 0.0006 < std 0.0077) | 0.8839 | No |
| `elec2` | Recurring, real | ARF | 0.7983 | Yes — ARF by ~7.6 pts |
| `airlines` | Implicit, real | ARF 0.6541 +/- 0.0012 (+0.005 for 46x runtime) | 0.6488 | **No — retraining hurts** |

## Three claims the data supports

**1. Drift type determines the winner; no strategy dominates.**
ARF takes `sea_abrupt` by ~2.5 points but comes **last** on `hyperplane`
(0.832 vs. the baseline's 0.884). Continuous rotation gives ARF's internal
detectors no sharp signal, so it discards trees that were still useful. A
benchmark reporting only abrupt synthetic drift would have concluded ARF is
universally best. It is not.

**2. On weak-drift real data, adaptation machinery costs accuracy.**
Airlines inverts the synthetic ordering completely:

    baseline 0.6488 > blind_periodic 0.6464 > detector_full_retrain 0.6415

Every adaptation strategy underperforms doing nothing. Resetting on a fixed
schedule, or on a detector's say-so, discards learned structure that the stream
never invalidated. This is the most practically useful result here: it tells a
practitioner when *not* to deploy adaptation, which the literature's focus on
synthetic abrupt drift tends to obscure.

**3. Accuracy gains carry an order-of-magnitude compute cost.**
ARF's runtime against the cheapest strategy on the same stream:

| Stream | ARF | Cheapest | Ratio |
|---|---|---|---|
| `sea_abrupt` | ~22 s | ~1.4 s | ~15x |
| `elec2` | ~127 s | ~6 s | ~21x |
| `airlines` | ~702 s | ~14 s | ~46x |

On Airlines that buys +0.004 accuracy. Whether that trade is worth it is a
budget question, and the answer is plainly no for most deployments.

## What the design deliberately controls for

**`detector_incremental` is a control, not a strategy.** ADWIN fires but nothing
is reset. It matches `baseline` exactly on all five streams — expected, since a
Hoeffding Tree already adapts incrementally. It isolates how much adaptation
comes from the detector versus the learner's own updating: on these streams, all
of it comes from the learner.

**Seeding differs by stream type.** Synthetic streams regenerate from the seed,
so 3 seeds is a real sample. Elec2 and Airlines are fixed rows in fixed order;
only ARF is stochastic there, so deterministic strategies run once rather than
producing three identical rows reported as "mean ± 0.000".

**`separable` distinguishes three states**, not two: `True` (margin exceeds the
winner's spread), `False` (top two indistinguishable — as on `hyperplane`, where
a 0.0006 margin sits inside a 0.0077 std), and `None` (unreplicated; the
question cannot be answered).

The test is deliberately one-sided: it compares the margin against the
*winner's* spread only. On the real-world streams the runner-up is usually a
deterministic strategy that ran once and therefore has no spread at all, so a
two-sided test is not available. This is a screening heuristic for "is this
gap larger than run-to-run noise", not a significance test — say so rather
than letting a reviewer discover it. With 3 seeds, a proper test would have
little power anyway.

## Recovery time is too noisy to rank strategies at n=3

The accuracy comparisons hold up: standard deviations of ~0.002-0.004 against
margins of ~0.025 on `sea_abrupt`. **Recovery time does not.** Per-seed values on
`sea_abrupt`:

| Strategy | Mean | Std | Range |
|---|---|---|---|
| `blind_periodic` | 533 | 115 | 400-600 |
| `detector_full_retrain` | 533 | 231 | 400-800 |
| ARF | 600 | 200 | 400-800 |
| `baseline` | 1000 | 529 | 400-1400 |
| `detector_incremental` | 1000 | 529 | 400-1400 |

The means suggest retraining halves recovery time. The distributions say
otherwise: baseline's fastest run (400) ties the fastest run of every other
strategy, and its std is over half its mean. **The distributions overlap
heavily and n=3 cannot separate them.**

Two causes, both fixable:

1. **Resolution.** `eval_every=200` quantises recovery to multiples of 200, so
   the metric can only take a handful of values. Every measured recovery is
   400, 600, 800, 1200, or 1400 — the granularity is a large fraction of the
   signal.
2. **Sample size.** A metric with std ~50% of its mean needs far more than 3
   seeds. 10-20 would be a reasonable target, and they are cheap for every
   strategy except ARF.

**Do not report a recovery-time ranking from the current runs.** Either report
it with the spreads shown and state explicitly that the strategies are not
separable, or re-run with a tighter `eval_every` and more seeds. The accuracy
results stand on their own and do not depend on this.

## Known limitations — state these in the paper

1. **`recovery_mean` is invalid on `sea_gradual`.** It reads 100 for all five
   strategies including the baseline. That is the metric bottoming out at one
   `eval_every=200` window, not a real tie. Only `sea_abrupt` recovery numbers
   (533 retrain vs. 1000 baseline) are trustworthy. Fix by tightening
   `eval_every` or widening the transition, then re-run.
2. **Recovery time does not exist for real-world streams.** No ground-truth
   drift index, so `elec2` and `airlines` are compared on accuracy, macro-F1,
   compute, and detector firing counts only. The benchmark has a two-tier
   results story and should say so.
3. **Airlines uses the first 100k of 539k rows**, in temporal order. The cap is
   a compute decision; ordering is preserved because the drift is temporal.
4. **One base learner.** Everything uses a Hoeffding Tree. Conclusions may not
   transfer to Naive Bayes or logistic regression.
5. **One detector.** ADWIN only, at default `delta=0.002`. DDM and a `delta`
   sensitivity analysis remain unrun.
