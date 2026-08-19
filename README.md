[README.md](https://github.com/user-attachments/files/31220828/README.md)
# The Alpha Factory — Candidate Submission

EGARCH-conditioned volatility breakout (single-asset) + cross-asset
relative-value reversion (pairs), pushed through one pre-registered 4-layer
gate. **72 candidates in, 0 survivors out** — see `report.pdf` for the full
writeup. Short version: an XAUUSD breakout family looks good (full-sample
Sharpe up to 0.75, OoS Sharpe up to 1.4) right up until a Deflated Sharpe
Ratio correction for the 72 trials we ran shows that's close to what the
*best of pure noise* would produce at that trial count. A relative-value
instance (SPXUSD/USDJPY) also survives three layers despite our own
cointegration diagnostics showing no structural basis for the pair — and
is caught decisively by the same Layer 4.

## Quick start

The raw CSVs aren't included in this repo (provided confidentially for the assignment). Point the pipeline at
wherever you have them:

```bash
pip install -r requirements.txt
export ALPHA_RAW_DIR=/path/to/raw/csvs    # defaults to ./data if not set
PYTHONPATH=. python3 run_funnel.py         # reproduces the full funnel end to end (~16 min)
PYTHONPATH=. python3 extra_diagnostics.py  # cost sensitivity, baselines, vol-regime (~1 min)
PYTHONPATH=. python3 make_figures.py       # regenerates the figures used in report.pdf
```

First run builds an H1-resampled pickle cache in `data_cache/` (or wherever `ALPHA_CACHE_DIR` points, if set);
reused on subsequent runs. Runtime is dominated by EGARCH(1,1) MLE re-estimation (custom-implemented; `arch` is
not available in this environment) — cached aggressively per (symbol, date-range) so every (L,Q) breakout instance
on the same symbol and slice reuses one fit.

## Architecture

```
factory/
  root_ideas.py     -- breakout_egarch + relval signal generators, hypotheses
  egarch.py         -- EGARCH(1,1) MLE fit + causal rolling-refit filter
  expansion.py      -- grids -> 72-instance Candidate population
backtest/
  data.py           -- raw M1 CSV -> cleaned, resampled H1 bars (cached)
  engine.py         -- single-asset no-lookahead backtest + cost model + stats
                        (cost_multiplier param for the cost-sensitivity diagnostic)
  pairs_engine.py   -- two-asset (relval) backtest + cost model
diagnostics/
  cointegration.py  -- correlation + custom Engle-Granger/ADF pre-screen
  rolling_correlation.py   -- 30/90-day rolling correlation surface per pair
  rolling_cointegration.py -- rolling Engle-Granger (90-day window, 30-day
                               step): is any pair cointegrated in subperiods
                               even if not over the full sample? (no)
  pca.py            -- PCA (eigendecomposition) on standardized H1 returns
                        across all 4 symbols: explained variance + loadings
  regimes.py        -- ATR-vol level, efficiency-ratio trend/chop, 200-day
                        MA bull/bear, and spread/liquidity regime labels
                        (descriptive only)
gate/
  pre_registration.yaml   -- SINGLE SOURCE OF TRUTH: every grid, cost-model
                              parameter, and layer threshold; criteria.py
                              loads this file directly, never hardcodes
  criteria.py             -- thin loader around pre_registration.yaml
  taxonomy.py             -- systematic failure taxonomy (failed_layer,
                              primary/secondary reason, worst_metric)
  instance_runner.py      -- one interface (.bt/.bt_slice/.bt_stress) for
                              single- and multi-asset instances
  layer1_is_oos.py        -- IS/OoS + parameter-neighbor robustness
  layer2_walkforward.py   -- hybrid: group re-optimization + own-param OOW
  layer3_stress.py        -- COVID replay + post-warmup real stress window
                              + synthetic vol-scaled shock
  layer4_montecarlo.py    -- permutation test, trade bootstrap, Deflated Sharpe Ratio
run_funnel.py       -- orchestrates the pre-registered gate, writes results/*.csv
extra_diagnostics.py -- SUPPLEMENTARY, not part of the gate: cost sensitivity
                         (0.5x/1x/2x), 3-way vol-filter ablation (EGARCH vs.
                         realized-vol vs. unconditioned breakout vs.
                         buy-and-hold), volatility-regime breakdown, and the
                         full ATR/efficiency-ratio/MA regime tables
make_figures.py     -- builds results/figures/*.png from the run's output
report.tex / .pdf   -- the (max 6-page) research report
```

Adding a third root idea needs: one new signal function, one new
`InstanceRunner` subclass if it's multi-asset, one new grid entry in
`factory/expansion.py` AND `gate/pre_registration.yaml`. Layers 1-4 never
change.

## Outputs

- `results/diagnostics/cointegration_tests.csv`, `pca_explained_variance.csv`, `pca_loadings.csv`,
  `rolling_correlation.csv`, `rolling_cointegration.csv` — pre-screen + common-factor + stability checks for Family B
- `results/funnel_summary.csv` / `results/instance_level.csv` — funnel +
  per-instance verdicts, including the failure-taxonomy columns
- `results/walkforward_groups.csv` — Layer 2 group-level re-optimization detail
- `results/cost_sensitivity.csv` / `results/baselines.csv` / `results/vol_regime.csv` /
  `results/regime_performance.csv` — supplementary diagnostics (`extra_diagnostics.py`), not gate layers
- `results/run_meta.json` — cross-sectional Sharpe std, best Sharpe, calm/stress windows
- `results/figures/*.png` — figures used in `report.pdf`, including
  parameter heatmaps, the walk-forward-instability scatter, and the
  efficiency-ratio regime contrast

## Honesty / pre-registration

Every numeric threshold in the gate is defined exactly once, in
`gate/pre_registration.yaml` — `gate/criteria.py` loads it rather than
re-declaring values, so what's documented is what ran. Its `change_log` has
exactly one entry (a disclosed scope addition: a second, real post-warmup
stress window for Layer 3, added before re-running, not a loosened
threshold) and is otherwise empty. Two further judgment calls (the DSR pass
threshold, Layer 2's re-optimization/per-instance mapping) are documented at
the top of that file and in `report.pdf` §4.

Known limitations, matching `report.pdf` §8: the custom Engle-Granger
implementation (no network access to install `statsmodels`), the COVID-window
data-start-date artifact for `breakout_egarch`, and Granger causality
testing, the one item on the original diagnostic list that didn't make it
in. Rolling correlation, rolling cointegration stability, PCA, and all four
regime breakdowns (volatility, trend/chop, bull/bear, spread/liquidity) are
all implemented — see Outputs below. On the DSR trial count: it uses the
raw population size ($N=72$), not an estimated effective-independent-trials
count. The Layer 1-3 survivors are correlated with each other (mostly
through shared XAUUSD trend exposure), so exact calibration is uncertain;
DSR is treated as an approximate guardrail rather than a precise number, and
that uncertainty is never used to argue a candidate should be rescued.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, `pandas`, `numpy`, `scipy`, `matplotlib`, `pyyaml`. No ML libraries (out of scope). No
`statsmodels`/`arch` dependency — both EGARCH and Engle-Granger are implemented from scratch in this repo
(see `report.pdf` §8 for why, and `compare_statsmodels.py` if you want to check the custom Engle-Granger/ADF
implementation against `statsmodels`' canonical one yourself, wherever you have network access).
