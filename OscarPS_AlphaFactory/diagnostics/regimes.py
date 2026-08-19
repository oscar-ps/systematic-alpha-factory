"""
Regime labels: ATR-based volatility level, efficiency-ratio trend/chop, and
moving-average bull/bear. These are descriptive labels for EXPLAINING where
a candidate's performance comes from (Section 6 of the suggestions doc) --
they are never fed back into any signal or used to gate any instance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from factory.root_ideas import atr


def atr_regime(df: pd.DataFrame, atr_period: int = 14, lookback_bars: int = 2160) -> pd.Series:
    """Low <30th pct, Normal 30-70th, High 70-90th, Extreme >90th -- ATR
    percentile-ranked against its own trailing history (lookback_bars,
    ~90 days), independent of the EGARCH vol filter used by Family A."""
    a = atr(df, atr_period)
    pct = a.rolling(lookback_bars).rank(pct=True)
    bins = pd.cut(pct, bins=[-0.01, 0.30, 0.70, 0.90, 1.01],
                   labels=["Low", "Normal", "High", "Extreme"])
    return bins


def efficiency_ratio_regime(df: pd.DataFrame, L: int = 48) -> pd.Series:
    """efficiency_ratio = |close[t]-close[t-L]| / sum(|close[i]-close[i-1]|)
    over the trailing L bars. Choppy <0.25, Neutral 0.25-0.50, Trending >0.50."""
    close = df["close"]
    net_change = (close - close.shift(L)).abs()
    path_length = close.diff().abs().rolling(L).sum()
    er = (net_change / path_length.replace(0, np.nan)).clip(0, 1)
    bins = pd.cut(er, bins=[-0.01, 0.25, 0.50, 1.01], labels=["Choppy", "Neutral", "Trending"])
    return bins


def ma_regime(df: pd.DataFrame, ma_window: int = 24 * 200) -> pd.Series:
    """Bull = price above its trailing ma_window (~200-day) moving average,
    Bear = below."""
    close = df["close"]
    ma = close.rolling(ma_window).mean()
    return pd.Series(np.where(close > ma, "Bull", "Bear"), index=df.index)


def spread_regime(df: pd.DataFrame, lookback_bars: int = 2160) -> pd.Series:
    """Normal <70th pct, Wide 70-90th, Extreme >90th -- realized broker
    spread (spread_frac) percentile-ranked against its own trailing
    history. A strategy that only works when spreads are favorable is not
    execution-robust, regardless of what the gate says about its returns."""
    pct = df["spread_frac"].rolling(lookback_bars).rank(pct=True)
    bins = pd.cut(pct, bins=[-0.01, 0.70, 0.90, 1.01], labels=["Normal", "Wide", "Extreme"])
    return bins


def regime_performance_table(bt: pd.DataFrame, df: pd.DataFrame, instance_id: str) -> pd.DataFrame:
    """Mean per-bar strategy return broken down by each of the 4 regime
    dimensions, for one instance's full-history backtest."""
    rows = []
    for name, regime in [("atr_vol", atr_regime(df)), ("efficiency_ratio", efficiency_ratio_regime(df)),
                          ("moving_average", ma_regime(df)), ("spread_liquidity", spread_regime(df))]:
        regime = regime.reindex(bt.index)
        g = bt["strat_ret"].groupby(regime, observed=True).agg(["mean", "count"])
        for label, vals in g.iterrows():
            rows.append(dict(instance_id=instance_id, regime_type=name, regime_label=str(label),
                              mean_bar_return=vals["mean"], n_bars=int(vals["count"])))
    return pd.DataFrame(rows)
