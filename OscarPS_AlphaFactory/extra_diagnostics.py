"""
extra_diagnostics.py — supplementary analyses, NOT part of the pre-
registered gate (disclosed as such throughout). Run after run_funnel.py:

    PYTHONPATH=. python3 extra_diagnostics.py

Produces:
  results/cost_sensitivity.csv   -- 0.5x/1x/2x cost, applied to Layer 1-3 survivors
  results/baselines.csv          -- buy-and-hold + un-conditioned (Q=0) breakout
                                     ablation, per symbol -- "did EGARCH help?"
  results/vol_regime.csv         -- breakout_egarch Layer1-3 survivors' mean
                                     bar return by EGARCH vol-percentile quartile
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from backtest.data import load_all
from backtest.engine import perf_stats, trade_returns, run_backtest
from factory.expansion import expand_population, GRIDS
from factory.root_ideas import breakout_egarch_signal, _get_vol_pct, session_reversion_signal
from gate.instance_runner import make_runner
from gate.criteria import COST_SENSITIVITY
from diagnostics.pca import run_pca
from diagnostics.regimes import regime_performance_table
from diagnostics.rolling_correlation import rolling_correlation_summary
from diagnostics.rolling_cointegration import rolling_cointegration

RESULTS_DIR = "results"


def session_direction_check(data: dict, lookback_days: int = 40, z_entry: float = 1.5) -> pd.DataFrame:
    """For each symbol: does fading the Asian session (as specified) beat
    following it (momentum)? Run both directions at one representative
    parameter setting, per symbol, to see whether the reversion sign even
    has the right theoretical direction before the gate gets involved."""
    rows = []
    for sym, df in data.items():
        sig = session_reversion_signal(df, lookback_days=lookback_days, z_entry=z_entry)
        bt_rev = run_backtest(df, sig)
        bt_mom = run_backtest(df, -sig)
        s_rev, s_mom = perf_stats(bt_rev["strat_ret"]), perf_stats(bt_mom["strat_ret"])
        rows.append(dict(symbol=sym, sharpe_reversion=s_rev["sharpe"], sharpe_momentum=s_mom["sharpe"],
                          reversion_total_return=s_rev["total_return"], momentum_total_return=s_mom["total_return"],
                          theory_matches_data=bool(s_rev["sharpe"] > s_mom["sharpe"])))
    return pd.DataFrame(rows)


def cost_sensitivity(df_inst: pd.DataFrame, data: dict) -> pd.DataFrame:
    survivors = df_inst[df_inst.l1_pass & df_inst.l2_pass & df_inst.l3_pass]
    pop = {c.id: c for c in expand_population()}
    rows = []
    for _, row in survivors.iterrows():
        c = pop[row["instance_id"]]
        runner = make_runner(c.family, c.assets, c.params)
        results = {}
        for mult in COST_SENSITIVITY["multipliers"]:
            bt = runner.bt(data, cost_multiplier=mult)
            s = perf_stats(bt["strat_ret"])
            n_trades = len(trade_returns(bt))
            turnover = float(bt["trade"].sum())
            results[mult] = (s["sharpe"], s["total_return"], n_trades, turnover)
        sharpe_1x = results[1.0][0]
        sharpe_2x = results[2.0][0]
        drag = results[1.0][1] - results[0.5][1]
        rows.append(dict(
            instance_id=c.id, family=c.family,
            sharpe_0p5x=results[0.5][0], sharpe_1x=sharpe_1x, sharpe_2x=sharpe_2x,
            return_1x=results[1.0][1], turnover_1x=results[1.0][3], n_trades=results[1.0][2],
            cost_drag_0p5x_to_1x=drag,
            dies_at_2x_cost=bool(sharpe_2x <= 0 and sharpe_1x > 0),
        ))
    return pd.DataFrame(rows)


def _realized_vol_pct(df: pd.DataFrame, lookback_bars: int = 2160) -> pd.Series:
    """Same percentile-rank construction as the EGARCH filter (_get_vol_pct),
    but using a naive rolling-std realized volatility instead of an EGARCH
    conditional-vol forecast -- isolates whether EGARCH's sophistication
    (vs. a simple realized-vol filter) is what's adding value, per the
    suggestions doc's 3-way ablation (EGARCH-conditioned vs. realized-vol-
    conditioned vs. unfiltered breakout)."""
    r = df["close"].pct_change()
    rv = r.rolling(24 * 5).std()  # ~5-day realized vol, smoothed, before percentile-ranking
    return rv.rolling(lookback_bars).rank(pct=True).fillna(0.0)


def baselines(data: dict) -> pd.DataFrame:
    """Three-way ablation per the suggestions doc: EGARCH-conditioned (the
    actual Family A) vs. realized-vol-conditioned vs. unconditioned (Q=0)
    breakout, plus buy-and-hold, per symbol, across the breakout L grid.
    NOT new candidate families -- reference points only."""
    rows = []
    for sym, df in data.items():
        bh_ret = df["close"].pct_change().fillna(0.0)
        bh_stats = perf_stats(bh_ret)
        rows.append(dict(symbol=sym, strategy="buy_and_hold", L=None, Q=None,
                          sharpe=bh_stats["sharpe"], total_return=bh_stats["total_return"],
                          max_dd=bh_stats["max_dd"]))

        rv_pct = _realized_vol_pct(df)
        for L in GRIDS["breakout_egarch"]["L"]:
            sig_unfiltered = breakout_egarch_signal(df, L=L, Q=0.0)
            bt_u = run_backtest(df, sig_unfiltered)
            s_u = perf_stats(bt_u["strat_ret"])
            rows.append(dict(symbol=sym, strategy="breakout_unconditioned", L=L, Q=0.0,
                              sharpe=s_u["sharpe"], total_return=s_u["total_return"], max_dd=s_u["max_dd"]))

            for Q in GRIDS["breakout_egarch"]["Q"]:
                sig_rv = breakout_egarch_signal(df, L=L, Q=Q, vol_pct_override=rv_pct)
                bt_rv = run_backtest(df, sig_rv)
                s_rv = perf_stats(bt_rv["strat_ret"])
                rows.append(dict(symbol=sym, strategy="breakout_realizedvol_filtered", L=L, Q=Q,
                                  sharpe=s_rv["sharpe"], total_return=s_rv["total_return"], max_dd=s_rv["max_dd"]))
    return pd.DataFrame(rows)


def vol_regime_breakdown(df_inst: pd.DataFrame, data: dict) -> pd.DataFrame:
    """For breakout_egarch Layer1-3 survivors: mean per-bar strategy return
    by EGARCH vol-percentile quartile (reuses the already-cached vol_pct
    series -- near-zero marginal compute)."""
    survivors = df_inst[(df_inst.family == "breakout_egarch") & df_inst.l1_pass & df_inst.l2_pass & df_inst.l3_pass]
    pop = {c.id: c for c in expand_population()}
    rows = []
    for _, row in survivors.iterrows():
        c = pop[row["instance_id"]]
        runner = make_runner(c.family, c.assets, c.params)
        bt = runner.bt(data)
        df = data[c.assets[0]]
        vol_pct = _get_vol_pct(df).reindex(bt.index)
        quartile = pd.qcut(vol_pct.rank(method="first"), 4, labels=["Q1 (calm)", "Q2", "Q3", "Q4 (volatile)"])
        g = bt["strat_ret"].groupby(quartile).agg(["mean", "count"])
        for q, vals in g.iterrows():
            rows.append(dict(instance_id=c.id, symbol=c.assets[0], vol_quartile=str(q),
                              mean_bar_return=vals["mean"], n_bars=int(vals["count"])))
    return pd.DataFrame(rows)


def main():
    print("Loading data and instance_level.csv...")
    data = load_all()
    df_inst = pd.read_csv(f"{RESULTS_DIR}/instance_level.csv")

    print("PCA on H1 returns across the 4 symbols...")
    df_var, df_load = run_pca(data, list(data.keys()))
    df_var.to_csv(f"{RESULTS_DIR}/diagnostics/pca_explained_variance.csv", index=False)
    df_load.to_csv(f"{RESULTS_DIR}/diagnostics/pca_loadings.csv")
    print(df_var.to_string(index=False))
    print(df_load.to_string())

    print("\nRolling 30/90-day correlation surface...")
    df_rollcorr = rolling_correlation_summary(data, list(data.keys()))
    df_rollcorr.to_csv(f"{RESULTS_DIR}/diagnostics/rolling_correlation.csv", index=False)
    print(df_rollcorr.to_string(index=False))

    print("\nRolling cointegration stability (90-day windows, 30-day step)...")
    df_rollcoint = rolling_cointegration(data, list(data.keys()))
    df_rollcoint.to_csv(f"{RESULTS_DIR}/diagnostics/rolling_cointegration.csv", index=False)
    print(df_rollcoint.to_string(index=False))

    print("\nCost sensitivity (Layer 1-3 survivors, 0.5x/1x/2x cost)...")
    cs = cost_sensitivity(df_inst, data)
    cs.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    print(cs.to_string(index=False))

    print("\nBaselines: buy-and-hold + EGARCH vs. realized-vol vs. unconditioned breakout...")
    bl = baselines(data)
    bl.to_csv(f"{RESULTS_DIR}/baselines.csv", index=False)
    print(bl.to_string(index=False))

    print("\nVolatility-regime breakdown (breakout_egarch survivors, EGARCH vol_pct quartiles)...")
    vr = vol_regime_breakdown(df_inst, data)
    vr.to_csv(f"{RESULTS_DIR}/vol_regime.csv", index=False)
    print(vr.to_string(index=False))

    print("\nSession-reversion direction check: does fading the Asian session beat following it?")
    dc = session_direction_check(data)
    dc.to_csv(f"{RESULTS_DIR}/session_direction_check.csv", index=False)
    print(dc.to_string(index=False))

    print("\nFull regime tables (ATR vol level, efficiency-ratio trend/chop, MA bull/bear)...")
    survivors = df_inst[df_inst.l1_pass & df_inst.l2_pass & df_inst.l3_pass]
    pop = {c.id: c for c in expand_population()}
    regime_rows = []
    for _, row in survivors.iterrows():
        c = pop[row["instance_id"]]
        runner = make_runner(c.family, c.assets, c.params)
        bt = runner.bt(data)
        df = data[c.assets[0]] if c.family == "breakout_egarch" else data[c.assets[-1]]
        regime_rows.append(regime_performance_table(bt, df, c.id))
    df_regime = pd.concat(regime_rows, ignore_index=True)
    df_regime.to_csv(f"{RESULTS_DIR}/regime_performance.csv", index=False)
    print(df_regime.to_string(index=False))


if __name__ == "__main__":
    main()
