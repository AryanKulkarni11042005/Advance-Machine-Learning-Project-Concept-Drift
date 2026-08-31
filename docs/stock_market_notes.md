# S&P 500 stream — review notes

Review of `notebooks/stock_market.ipynb` before wiring it into the benchmark.

## What was already right

- **No lookahead in features.** Every indicator (`rolling`, `diff`, `pct_change`,
  `ewm`) is backward-looking. No global normalisation, no full-sample statistics.
  This is the error that silently inflates financial results, and it was avoided.
- **Target correctly formed.** `Close.shift(-1) > Close` is genuine next-day
  direction, and `dropna` removes the final row where no label exists.
- **Class balance is realistic.** 54.4% up / 45.6% down reflects real market
  drift, not a labelling bug.

## Three defects found

**1. `volume_change` contained `inf`** (row 4581): a zero-volume prior day made
`pct_change` divide by zero. A Hoeffding Tree computing split statistics over an
infinite value corrupts that node.

**2. Price-level features are non-stationary.** `ma_50` averages 1187 early and
5660 late — a 4.8x drift. A tree learns `ma_50 < 1200`, a rule that is true in
2005 and never fires after ~2013. This manufactures apparent concept drift that
is really just price inflation, contaminating the exact quantity the benchmark
measures. `stock_sp500(scale_free=True)` replaces levels with ratios.

**3. Labels were learned before they existed.** The target for row t is only
known after t+1 closes, but standard prequential calls `learn_one(x_t, y_t)` at
t. `run_strategy(..., delay=1)` holds each sample until its label would actually
be available.

## The finding that matters: there is no signal

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| Raw price features, delay=0 (original) | 0.5417 | 0.3682 |
| Scale-free, delay=0 | 0.5423 | 0.3636 |
| Scale-free, delay=1 (correct) | 0.5452 | 0.3680 |
| **Always predict "up"** | **0.5445** | — |

**The model does not beat the majority class.** It predicts "up" on 4906 of 4980
samples (98.5%); precision on "up" is 0.5455, identical to the base rate. The
0.37 macro-F1 against 0.54 accuracy is the signature of a classifier collapsed
onto one class.

Note the leak was worth almost nothing (0.5417 vs 0.5452) — because there was no
signal to leak. Fixing it changed the number by less than half a point.

## What this means for the project

**Do not report strategy comparisons on this stream as though differences were
meaningful.** Every strategy will land within noise of 0.5445, and ranking them
would be ranking noise. Daily index direction from technical indicators is close
to unpredictable; this is the expected result, not a bug in the pipeline.

Three honest options:

1. **Report it as a negative result.** This is defensible and arguably the most
   valuable use: it extends the Airlines finding (adaptation machinery does not
   help when there is no exploitable drift) to a domain where practitioners
   routinely assume otherwise. State the majority-class baseline alongside every
   number so the absence of signal is visible.
2. **Change the target to something with more signal** — multi-day direction,
   volatility regime, or large-move detection. Volatility is far more predictable
   than direction and has genuine regime shifts, which would suit a drift
   benchmark much better.
3. **Drop the stream** and keep the five that produced usable results.

Option 2 is the most promising if the goal is a working drift benchmark;
option 1 if the goal is an honest paper with what already exists.

## Usage

```python
import streams, pipeline
samples, _ = streams.stock_sp500()          # scale-free features
pipeline.run_strategy(samples, "baseline", "stock_sp500", delay=1)
```

Always pass `delay=1` on this stream. Always report the 0.5445 majority baseline.
