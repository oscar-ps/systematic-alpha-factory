"""
Variant expansion (Step 2), v3.

  breakout_egarch (single-asset):
    L in {24, 48, 96, 168}   H1 bars
    Q in {0.40, 0.60, 0.80}  EGARCH vol-percentile threshold
    -> 12 combos x 4 symbols = 48 instances

  session_reversion (single-asset, the redesigned Family B):
    The original Family B (cross-asset pairs reversion, relval) was dropped
    after Section 1's diagnostics showed no structural relationship between
    any two of the four assets (zero of 6 pairs cointegrated, PC1 explaining
    only 36% of variance). session_reversion needs no cross-asset
    relationship at all -- it only compares an asset's own Asian-session
    return against its own history. See factory/root_ideas.py for the full
    hypothesis. relval's code (backtest/pairs_engine.py,
    factory/root_ideas.relval_signal) is kept in the repo but is no longer
    part of the active population.
    lookback_days in {20, 60}     trading days
    z_entry in {1.0, 1.5, 2.0}
    -> 6 combos x 4 symbols = 24 instances

48 + 24 = 72 instances total.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field

SYMBOLS = ["SPXUSD", "USDJPY", "XAUUSD", "ETHUSD"]
PAIRS = list(itertools.combinations(SYMBOLS, 2))  # kept for relval, no longer used in expand_population

GRIDS = {
    "breakout_egarch": {
        "L": [24, 48, 96, 168],
        "Q": [0.40, 0.60, 0.80],
    },
    "session_reversion": {
        "lookback_days": [20, 60],
        "z_entry": [1.0, 1.5, 2.0],
    },
    "relval": {  # retained for reference / compare_statsmodels-style reruns; unused by expand_population
        "L": [336, 720],
        "Z": [1.5, 2.5],
    },
}


@dataclass
class Candidate:
    id: str
    family: str
    assets: tuple
    params: dict = field(default_factory=dict)

    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.family}|{'/'.join(self.assets)}|{p}"


def expand_population() -> list:
    pop = []
    grid = GRIDS["breakout_egarch"]
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        for sym in SYMBOLS:
            cid = f"breakout_egarch_{sym}_" + "_".join(f"{k}{v}" for k, v in params.items())
            pop.append(Candidate(id=cid, family="breakout_egarch", assets=(sym,), params=params))

    grid = GRIDS["session_reversion"]
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        for sym in SYMBOLS:
            cid = f"session_reversion_{sym}_" + "_".join(f"{k}{v}" for k, v in params.items())
            pop.append(Candidate(id=cid, family="session_reversion", assets=(sym,), params=params))
    return pop


def family_param_grid(family: str) -> list:
    grid = GRIDS[family]
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]


if __name__ == "__main__":
    pop = expand_population()
    print(f"Population size: {len(pop)}")
    from collections import Counter
    print(Counter(c.family for c in pop))
