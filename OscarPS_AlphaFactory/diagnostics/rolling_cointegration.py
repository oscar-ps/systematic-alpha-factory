"""
Rolling cointegration stability. The full-sample Engle-Granger test
(diagnostics/cointegration.py) asks "is this pair cointegrated over the
whole 2020-2026 sample". It's possible a pair isn't cointegrated overall
but spends meaningful subperiods cointegrated, which a single full-sample
test would miss entirely and which would matter a lot for whether relval
has any real edge in some regimes. This checks that directly: rolling
Engle-Granger over 90-day windows, stepped by 30 days.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from diagnostics.cointegration import engle_granger, EG_CRIT


def rolling_cointegration(data: dict, symbols: list[str], window_days: int = 90, step_days: int = 30) -> pd.DataFrame:
    log_close = {s: np.log(data[s]["close"]) for s in symbols}
    common_idx = None
    for s in symbols:
        common_idx = log_close[s].index if common_idx is None else common_idx.intersection(log_close[s].index)
    win_bars, step_bars = window_days * 24, step_days * 24

    rows = []
    for a, b in itertools.combinations(symbols, 2):
        xa = log_close[a].reindex(common_idx).to_numpy()
        xb = log_close[b].reindex(common_idx).to_numpy()
        n = len(xa)
        adf_stats, betas = [], []
        t = win_bars
        while t < n:
            wa, wb = xa[t - win_bars:t], xb[t - win_bars:t]
            eg = engle_granger(wb, wa)  # b = alpha + beta*a
            if np.isfinite(eg["adf_stat"]):
                adf_stats.append(eg["adf_stat"])
                betas.append(eg["beta"])
            t += step_bars
        adf_stats = np.array(adf_stats)
        betas = np.array(betas)
        frac_sig = float(np.mean(adf_stats < EG_CRIT[0.10])) if len(adf_stats) else np.nan
        rows.append(dict(
            pair=f"{a}/{b}", n_windows=len(adf_stats),
            frac_windows_p10=frac_sig, median_adf_stat=float(np.median(adf_stats)) if len(adf_stats) else np.nan,
            beta_mean=float(np.mean(betas)) if len(betas) else np.nan,
            beta_std=float(np.std(betas)) if len(betas) else np.nan,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from backtest.data import load_all
    data = load_all()
    df = rolling_cointegration(data, list(data.keys()))
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
