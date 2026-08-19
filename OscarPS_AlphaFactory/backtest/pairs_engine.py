"""
Backtest engine for two-asset relative-value instances.

Same no-lookahead contract as backtest/engine.py (signals shift by 1 bar
before being applied), but P&L and costs are computed across BOTH legs:
  - position = +1 (long spread)  -> long 1 unit of asset B, short beta units of asset A
  - position = -1 (short spread) -> short 1 unit of asset B, long beta units of asset A
P&L per unit position ~= r_B - beta * r_A (first-order log-return approximation
of the change in the spread y - alpha - beta*x). Costs are charged on BOTH
legs whenever the spread position changes, weighted by each leg's notional
(1 for B, |beta| for A).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COMMISSION_PER_SIDE = 0.00005


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df_a.index.intersection(df_b.index)
    return df_a.loc[idx], df_b.loc[idx]


def run_pairs_backtest(df_a: pd.DataFrame, df_b: pd.DataFrame, raw_position: pd.Series, beta: pd.Series, cost_multiplier: float = 1.0) -> pd.DataFrame:
    pos = raw_position.shift(1).fillna(0.0)
    beta_lag = beta.shift(1).fillna(0.0)

    ret_a = df_a["close"].pct_change().fillna(0.0)
    ret_b = df_b["close"].pct_change().fillna(0.0)
    bar_ret = ret_b - beta_lag * ret_a

    trade_size = pos.diff().abs().fillna(pos.abs())
    spread_frac_a = df_a["spread_frac"].fillna(df_a["spread_frac"].median())
    spread_frac_b = df_b["spread_frac"].fillna(df_b["spread_frac"].median())
    cost_per_unit = cost_multiplier * ((spread_frac_b / 2.0 + COMMISSION_PER_SIDE) + beta_lag.abs() * (spread_frac_a / 2.0 + COMMISSION_PER_SIDE))
    cost = trade_size * cost_per_unit

    strat_ret = pos * bar_ret - cost
    equity = (1.0 + strat_ret).cumprod()

    return pd.DataFrame(
        {"ret": bar_ret, "position": pos, "beta": beta_lag, "trade": trade_size, "cost": cost,
         "strat_ret": strat_ret, "equity": equity},
        index=df_a.index,
    )
