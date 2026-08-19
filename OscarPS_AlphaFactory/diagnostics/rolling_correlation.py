"""
Rolling correlation surface. Static full-sample correlation (used in
cointegration.py) can hide a lot: two assets could spend stretches of time
strongly correlated and other stretches anti-correlated, averaging out to
something near zero. This checks whether that's happening, and whether any
pair has a correlation regime stable enough to be worth trading.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def rolling_correlation_summary(data: dict, symbols: list[str], windows_days=(30, 90)) -> pd.DataFrame:
    rets = {}
    common_idx = None
    for s in symbols:
        r = data[s]["close"].pct_change()
        rets[s] = r
        common_idx = r.index if common_idx is None else common_idx.intersection(r.index)
    R = pd.DataFrame({s: rets[s].reindex(common_idx) for s in symbols})

    rows = []
    for a, b in itertools.combinations(symbols, 2):
        for wd in windows_days:
            win_bars = wd * 24
            roll_corr = R[a].rolling(win_bars).corr(R[b]).dropna()
            rows.append(dict(
                pair=f"{a}/{b}", window_days=wd,
                mean_corr=float(roll_corr.mean()), std_corr=float(roll_corr.std()),
                min_corr=float(roll_corr.min()), max_corr=float(roll_corr.max()),
                frac_sign_flips=float(np.mean(np.diff(np.sign(roll_corr.values)) != 0)),
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from backtest.data import load_all
    data = load_all()
    df = rolling_correlation_summary(data, list(data.keys()))
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
