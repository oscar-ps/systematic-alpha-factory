"""
InstanceRunner gives every gate layer ONE interface regardless of whether a
candidate is a single-asset instance (breakout_egarch) or a two-asset
instance (relval). Adding a third root idea -- single- or multi-asset --
means adding one new subclass here; layer1-4 never need to change.
"""
from __future__ import annotations
import pandas as pd
from backtest.engine import run_backtest
from backtest.pairs_engine import run_pairs_backtest, align_pair
from factory.root_ideas import breakout_egarch_signal, relval_signal, session_reversion_signal
from gate.layer3_stress import make_synthetic_shock


class BreakoutRunner:
    family = "breakout_egarch"

    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.assets = (symbol,)

    def bt(self, data: dict, params: dict | None = None, cost_multiplier: float = 1.0) -> pd.DataFrame:
        p = params or self.params
        df = data[self.symbol]
        sig = breakout_egarch_signal(df, **p)
        return run_backtest(df, sig, cost_multiplier=cost_multiplier)

    def bt_slice(self, data: dict, start, end, params: dict | None = None) -> pd.DataFrame:
        p = params or self.params
        df = data[self.symbol].loc[start:end]
        sig = breakout_egarch_signal(df, **p)
        return run_backtest(df, sig)

    def bt_stress(self, data: dict, calm_window, scale: float):
        df = data[self.symbol]
        synth_df, _ = make_synthetic_shock(df, calm_window[0], calm_window[1], scale)
        sig = breakout_egarch_signal(synth_df, **self.params)
        return run_backtest(synth_df, sig), calm_window


class SessionRunner:
    family = "session_reversion"

    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.params = params
        self.assets = (symbol,)

    def bt(self, data: dict, params: dict | None = None, cost_multiplier: float = 1.0) -> pd.DataFrame:
        p = params or self.params
        df = data[self.symbol]
        sig = session_reversion_signal(df, **p)
        return run_backtest(df, sig, cost_multiplier=cost_multiplier)

    def bt_slice(self, data: dict, start, end, params: dict | None = None) -> pd.DataFrame:
        p = params or self.params
        df = data[self.symbol].loc[start:end]
        sig = session_reversion_signal(df, **p)
        return run_backtest(df, sig)

    def bt_stress(self, data: dict, calm_window, scale: float):
        df = data[self.symbol]
        synth_df, _ = make_synthetic_shock(df, calm_window[0], calm_window[1], scale)
        sig = session_reversion_signal(synth_df, **self.params)
        return run_backtest(synth_df, sig), calm_window


class RelvalRunner:
    family = "relval"

    def __init__(self, pair: tuple, params: dict):
        self.pair = pair
        self.params = params
        self.assets = pair

    def bt(self, data: dict, params: dict | None = None, cost_multiplier: float = 1.0) -> pd.DataFrame:
        p = params or self.params
        a, b = self.pair
        df_a, df_b = align_pair(data[a], data[b])
        pos, beta = relval_signal(df_a["close"], df_b["close"], **p)
        return run_pairs_backtest(df_a, df_b, pos, beta, cost_multiplier=cost_multiplier)

    def bt_slice(self, data: dict, start, end, params: dict | None = None) -> pd.DataFrame:
        p = params or self.params
        a, b = self.pair
        df_a, df_b = align_pair(data[a].loc[start:end], data[b].loc[start:end])
        pos, beta = relval_signal(df_a["close"], df_b["close"], **p)
        return run_pairs_backtest(df_a, df_b, pos, beta)

    def bt_stress(self, data: dict, calm_window, scale: float):
        a, b = self.pair
        synth_a, _ = make_synthetic_shock(data[a], calm_window[0], calm_window[1], scale)
        synth_b, _ = make_synthetic_shock(data[b], calm_window[0], calm_window[1], scale)
        synth_a, synth_b = align_pair(synth_a, synth_b)
        pos, beta = relval_signal(synth_a["close"], synth_b["close"], **self.params)
        return run_pairs_backtest(synth_a, synth_b, pos, beta), calm_window


def make_runner(family: str, assets, params: dict):
    if family == "breakout_egarch":
        return BreakoutRunner(assets[0], params)
    elif family == "session_reversion":
        return SessionRunner(assets[0], params)
    elif family == "relval":
        return RelvalRunner(tuple(assets), params)
    raise ValueError(f"unknown family {family}")
