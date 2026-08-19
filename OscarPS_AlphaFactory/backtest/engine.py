"""
Backtest engine.

No-lookahead contract: signal generators in factory/root_ideas.py compute a
"raw position" series using only information available up to and including
bar t's close. We then SHIFT that series by one bar before applying it to
returns, so the position used for bar t's return was decided strictly using
information known at bar t-1's close ("signals on closed bars only").

Cost model (Section 3.2 default): realized spread from <SPREAD> at the fill
bar, half the spread charged on each side of a round trip, plus 0.5 bps
commission per side. A trade occurs whenever target position changes;
the cost is proportional to the size of that change (a flip from -1 to +1
pays the full round-trip cost twice, i.e. close + open).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COMMISSION_PER_SIDE = 0.00005  # 0.5 bps


def run_backtest(df: pd.DataFrame, raw_position: pd.Series, cost_multiplier: float = 1.0) -> pd.DataFrame:
    """
    df: OHLC(+spread_frac) bars.
    raw_position: position decided using info up to bar t (NOT yet shifted).
    cost_multiplier: scales the cost model (1.0 = the pre-registered default
      cost; used by the cost-sensitivity diagnostic in
      diagnostics/cost_sensitivity.py, NOT part of the gate itself).
    Returns a DataFrame with columns: ret, position, trade, cost, strat_ret, equity.
    """
    pos = raw_position.shift(1).fillna(0.0)  # apply at t using info known at t-1
    bar_ret = df["close"].pct_change().fillna(0.0)

    trade_size = pos.diff().abs().fillna(pos.abs())  # |Δposition|, first bar = |pos|
    spread_frac = df["spread_frac"].fillna(df["spread_frac"].median())
    cost_per_unit = cost_multiplier * (spread_frac / 2.0 + COMMISSION_PER_SIDE)
    cost = trade_size * cost_per_unit

    strat_ret = pos * bar_ret - cost
    equity = (1.0 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            "ret": bar_ret,
            "position": pos,
            "trade": trade_size,
            "cost": cost,
            "strat_ret": strat_ret,
            "equity": equity,
        },
        index=df.index,
    )
    return out


def perf_stats(strat_ret: pd.Series, bars_per_year: float = 24 * 365.25) -> dict:
    r = strat_ret.dropna()
    if len(r) == 0 or r.std(ddof=0) == 0:
        return dict(n=len(r), ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_dd=0.0, total_return=0.0)
    mean = r.mean()
    std = r.std(ddof=0)
    sharpe = (mean / std) * np.sqrt(bars_per_year) if std > 0 else 0.0
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    dd = (equity / running_max - 1.0)
    max_dd = dd.min()
    total_return = equity.iloc[-1] - 1.0
    n_bars = len(r)
    ann_return = (1 + total_return) ** (bars_per_year / n_bars) - 1 if n_bars > 0 else 0.0
    return dict(
        n=n_bars,
        ann_return=ann_return,
        ann_vol=std * np.sqrt(bars_per_year),
        sharpe=sharpe,
        max_dd=max_dd,
        total_return=total_return,
    )


def trade_returns(bt: pd.DataFrame) -> pd.Series:
    """Collapse the bar-level strat_ret series into one return per discrete
    trade (round trip), used by Layer 4's bootstrap & permutation tests."""
    pos = bt["position"]
    # group consecutive bars with the same nonzero-or-zero position state into "trades"
    # a new trade starts whenever position changes
    change = pos.diff().fillna(pos.iloc[0] if len(pos) else 0) != 0
    grp = change.cumsum()
    df = bt.copy()
    df["grp"] = grp
    # only count groups where the held position (not the bar that opens it) is nonzero
    trade_rets = []
    for _, g in df.groupby("grp"):
        if (g["position"].iloc[0] == 0):
            continue
        gross = (1 + g["strat_ret"]).prod() - 1
        trade_rets.append(gross)
    return pd.Series(trade_rets)
