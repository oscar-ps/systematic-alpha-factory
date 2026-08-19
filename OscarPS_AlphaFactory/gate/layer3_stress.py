"""
Layer 3 -- stress replay.

Implemented: COVID crash replay + a synthetic volatility-scaled shock on a
real calm window (per-symbol for single-asset instances; applied to BOTH
legs over the same window for relval instances).

Deferred (disclosed, not faked): "worst 30-day realized-vol window per
asset", "worst 30-day return window per asset", and "worst spread/liquidity
window per asset" from the original plan draft. Given the timebox we kept
the two stress replays the assignment itself explicitly requires (COVID +
one synthetic shock) and did not implement the three additional per-asset
worst-window replays; see report limitations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtest.engine import perf_stats
from gate.criteria import L3


def window_total_return(bt: pd.DataFrame, start, end) -> float:
    win = bt.loc[start:end]
    if len(win) < 5:
        return 0.0
    return float((1 + win["strat_ret"]).prod() - 1)


def window_drawdown(bt: pd.DataFrame, start, end) -> float:
    win = bt.loc[start:end]
    if len(win) < 5:
        return 0.0
    eq = (1 + win["strat_ret"]).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def window_n_active(bt: pd.DataFrame, start, end) -> int:
    win = bt.loc[start:end]
    if len(win) == 0:
        return 0
    return int((win["position"] != 0).sum())


def make_synthetic_shock(df: pd.DataFrame, calm_start, calm_end, scale: float):
    """Scale close-to-close returns and intrabar O/H/L deviations (relative
    to close) by a fixed factor, applied to a real calm window. Preceding
    bars (for indicator warm-up) are left untouched."""
    warmup = df.loc[:calm_start].iloc[:-1]
    seg = df.loc[calm_start:calm_end].copy()

    close_orig = seg["close"].to_numpy()
    open_orig = seg["open"].to_numpy()
    high_orig = seg["high"].to_numpy()
    low_orig = seg["low"].to_numpy()

    ret = np.diff(close_orig) / close_orig[:-1]
    ret = np.insert(ret, 0, 0.0)
    ret_scaled = ret * scale

    anchor = close_orig[0]
    close_synth = anchor * np.cumprod(1 + ret_scaled)
    close_synth[0] = anchor

    prev_close_synth = np.roll(close_synth, 1)
    prev_close_synth[0] = anchor
    prev_close_orig = np.roll(close_orig, 1)
    prev_close_orig[0] = close_orig[0]

    ratio_o = open_orig / prev_close_orig
    open_synth = prev_close_synth * (1 + (ratio_o - 1) * scale)
    ratio_h = high_orig / close_orig
    high_synth = close_synth * (1 + (ratio_h - 1) * scale)
    ratio_l = low_orig / close_orig
    low_synth = close_synth * (1 + (ratio_l - 1) * scale)

    high_synth = np.maximum.reduce([high_synth, open_synth, close_synth])
    low_synth = np.minimum.reduce([low_synth, open_synth, close_synth])

    seg["open"], seg["high"], seg["low"], seg["close"] = open_synth, high_synth, low_synth, close_synth
    out = pd.concat([warmup, seg])
    return out, len(warmup)


def find_worst_vol_window(df: pd.DataFrame, window_days: int, after: str = "2020-05-15",
                           min_warmup_bars: int = 3000) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Worst (highest realized-vol) real window of length window_days,
    restricted to dates after `after` so breakout_egarch's EGARCH warm-up
    has definitely completed -- gives that family a real (non-synthetic,
    non-warmup-confounded) historical stress test, since COVID itself is
    confounded by warm-up (see report.pdf S6.3)."""
    ret = df["close"].pct_change()
    win_bars = window_days * 24
    vol = ret.rolling(win_bars).std().dropna()
    after_ts = pd.Timestamp(after)
    mask = []
    for ts in vol.index:
        w_start = ts - pd.Timedelta(days=window_days)
        enough_warmup = df.index.get_indexer([w_start], method="nearest")[0] > min_warmup_bars
        mask.append((w_start >= after_ts) and enough_warmup)
    vol = vol[mask]
    best_end = vol.idxmax()
    return best_end - pd.Timedelta(days=window_days), best_end


def find_calm_window(df: pd.DataFrame, window_days: int, exclude_start="2020-01-15", exclude_end="2020-06-01",
                      min_warmup_bars: int = 3000) -> tuple[pd.Timestamp, pd.Timestamp]:
    ret = df["close"].pct_change()
    win_bars = window_days * 24
    vol = ret.rolling(win_bars).std().dropna()
    excl_s, excl_e = pd.Timestamp(exclude_start), pd.Timestamp(exclude_end)
    mask = []
    for ts in vol.index:
        w_start = ts - pd.Timedelta(days=window_days)
        overlaps_covid = not (w_start > excl_e or ts < excl_s)
        enough_warmup = df.index.get_indexer([w_start], method="nearest")[0] > min_warmup_bars
        mask.append((not overlaps_covid) and enough_warmup)
    vol = vol[mask]
    best_end = vol.idxmin()
    return best_end - pd.Timedelta(days=window_days), best_end


def evaluate_instance_layer3(runner, data: dict, calm_window: tuple, postwarmup_window: tuple | None = None) -> dict:
    full_bt = runner.bt(data)
    full_stats = perf_stats(full_bt["strat_ret"])

    covid_ret = window_total_return(full_bt, L3["covid_start"], L3["covid_end"])
    covid_dd = window_drawdown(full_bt, L3["covid_start"], L3["covid_end"])
    covid_active = window_n_active(full_bt, L3["covid_start"], L3["covid_end"])
    covid_untested = covid_active == 0
    covid_pass = True if covid_untested else (covid_ret > L3["min_covid_return"])

    postwarmup_pass = True
    postwarmup_ret = None
    if postwarmup_window is not None:
        postwarmup_ret = window_total_return(full_bt, postwarmup_window[0], postwarmup_window[1])
        postwarmup_pass = postwarmup_ret > L3["min_postwarmup_return"]

    stress_bt, cw = runner.bt_stress(data, calm_window, L3["synthetic_vol_scale"])
    synth_ret = window_total_return(stress_bt, cw[0], cw[1])
    synth_dd = window_drawdown(stress_bt, cw[0], cw[1])
    synth_pass = synth_ret > L3["min_synthetic_return"]

    dd_cap = L3["max_dd_multiple_of_full_sample"] * abs(full_stats["max_dd"]) if full_stats["max_dd"] != 0 else abs(L3["min_covid_return"])
    dd_pass = (covid_dd >= -dd_cap) and (synth_dd >= -dd_cap)

    beta_pass = True
    if hasattr(runner, "pair"):
        pre = full_bt["beta"].loc[:calm_window[0]].tail(500).abs().median()
        during = stress_bt["beta"].loc[cw[0]:cw[1]].abs().median()
        if pre and pre > 1e-8:
            beta_pass = (during / pre) <= L3["max_beta_blowup_mult"]

    passed = covid_pass and synth_pass and dd_pass and beta_pass and postwarmup_pass
    fail_reason = None
    if not passed:
        fail_reason = ("covid_loss_exceeded" if not covid_pass else
                        "postwarmup_stress_loss_exceeded" if not postwarmup_pass else
                        "synthetic_loss_exceeded" if not synth_pass else
                        "stress_dd_exceeded_full_sample_multiple" if not dd_pass else
                        "pair_relationship_breakdown")

    return dict(
        passed=passed, covid_return=covid_ret, covid_dd=covid_dd, covid_untested=covid_untested,
        postwarmup_return=postwarmup_ret, postwarmup_pass=postwarmup_pass,
        synthetic_return=synth_ret, synthetic_dd=synth_dd, full_sample_dd=full_stats["max_dd"],
        beta_pass=beta_pass, fail_reason=fail_reason,
    )
