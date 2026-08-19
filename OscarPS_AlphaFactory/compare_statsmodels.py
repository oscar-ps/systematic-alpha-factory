"""
compare_statsmodels.py

Checks the custom Engle-Granger / ADF implementation in
diagnostics/cointegration.py against statsmodels' canonical implementation,
on the exact same data. Needs internet access (to install statsmodels),
which this sandbox doesn't have -- run it wherever you do.

    pip install statsmodels
    PYTHONPATH=. python3 compare_statsmodels.py

Two comparisons are run:
  1. Full Engle-Granger pipeline (cointegrating OLS regression + ADF on the
     residuals), for every pair, both regression directions, same as
     diagnostics/cointegration.py does internally.
  2. The ADF step in isolation: take the exact residuals my OLS regression
     produces, and run both my adf_stat() and statsmodels' adfuller() on
     them with the same lag length and no constant, to separate "did my
     regression differ" from "did my ADF test statistic differ".
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from backtest.data import load_all
from diagnostics.cointegration import engle_granger, adf_stat as my_adf_stat, EG_CRIT

try:
    from statsmodels.tsa.stattools import coint, adfuller
except ImportError:
    raise SystemExit("statsmodels not installed -- run: pip install statsmodels")

LAGS = 5  # must match diagnostics/cointegration.py's default


def compare_full_pipeline(symbols: list[str], log_close: dict, common_idx) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(symbols, 2):
        xa = log_close[a].reindex(common_idx).to_numpy()
        xb = log_close[b].reindex(common_idx).to_numpy()

        my_ab = engle_granger(xb, xa)  # b = alpha + beta*a
        my_ba = engle_granger(xa, xb)
        my_best = my_ab if my_ab["adf_stat"] < my_ba["adf_stat"] else my_ba
        my_dir = "B_on_A" if my_best is my_ab else "A_on_B"

        sm_ab = coint(xb, xa, maxlag=LAGS, autolag=None)  # (tstat, pvalue, crit_values)
        sm_ba = coint(xa, xb, maxlag=LAGS, autolag=None)
        if sm_ab[0] < sm_ba[0]:
            sm_tstat, sm_pval, sm_crit, sm_dir = sm_ab[0], sm_ab[1], sm_ab[2], "B_on_A"
        else:
            sm_tstat, sm_pval, sm_crit, sm_dir = sm_ba[0], sm_ba[1], sm_ba[2], "A_on_B"

        rows.append(dict(
            pair=f"{a}/{b}",
            my_adf_stat=round(my_best["adf_stat"], 4), my_direction=my_dir,
            sm_adf_stat=round(sm_tstat, 4), sm_direction=sm_dir,
            stat_diff=round(my_best["adf_stat"] - sm_tstat, 4),
            sm_pvalue=round(sm_pval, 4), sm_crit_10pct=round(sm_crit[2], 4),
            my_cointegrated_10pct=my_best["adf_stat"] < EG_CRIT[0.10],
            sm_cointegrated_10pct=bool(sm_pval < 0.10),
        ))
    return pd.DataFrame(rows)


def compare_adf_step_only(symbols: list[str], log_close: dict, common_idx) -> pd.DataFrame:
    """Isolate the ADF test itself: same residuals, both implementations."""
    rows = []
    for a, b in itertools.combinations(symbols, 2):
        xa = log_close[a].reindex(common_idx).to_numpy()
        xb = log_close[b].reindex(common_idx).to_numpy()
        X = np.column_stack([np.ones(len(xa)), xa])
        beta, *_ = np.linalg.lstsq(X, xb, rcond=None)
        resid = xb - X @ beta

        mine = my_adf_stat(resid, lags=LAGS, include_const=False)
        sm_result = adfuller(resid, maxlag=LAGS, regression="n", autolag=None)
        sm_stat = sm_result[0]

        rows.append(dict(
            pair=f"{a}/{b} (resid of B on A)",
            my_adf_stat=round(mine, 4), sm_adf_stat=round(sm_stat, 4),
            diff=round(mine - sm_stat, 6),
        ))
    return pd.DataFrame(rows)


def main():
    data = load_all()
    symbols = list(data.keys())
    log_close = {s: np.log(data[s]["close"]) for s in symbols}
    common_idx = None
    for s in symbols:
        common_idx = log_close[s].index if common_idx is None else common_idx.intersection(log_close[s].index)

    pd.set_option("display.width", 160)

    print("=" * 70)
    print("1. Full Engle-Granger pipeline: mine vs. statsmodels.coint()")
    print("=" * 70)
    df1 = compare_full_pipeline(symbols, log_close, common_idx)
    print(df1.to_string(index=False))
    n_agree = (df1["my_cointegrated_10pct"] == df1["sm_cointegrated_10pct"]).sum()
    print(f"\nVerdict agrees with statsmodels on {n_agree} of {len(df1)} pairs at the 10% level.")
    print(f"Mean |difference| in ADF statistic: {df1['stat_diff'].abs().mean():.4f}")

    print("\n" + "=" * 70)
    print("2. ADF step only, same residuals, same lags, no constant")
    print("=" * 70)
    df2 = compare_adf_step_only(symbols, log_close, common_idx)
    print(df2.to_string(index=False))
    print(f"\nMean |difference|: {df2['diff'].abs().mean():.6f} (should be ~0 if the ADF math matches exactly)")


if __name__ == "__main__":
    main()
