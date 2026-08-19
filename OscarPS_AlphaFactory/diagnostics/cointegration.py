"""
Cross-asset diagnostic layer (Section 3 of the plan).

`statsmodels` is not available in this environment (no network access to
install it), so the Engle-Granger cointegration test below is a from-scratch
implementation: OLS for the cointegrating regression, then an Augmented
Dickey-Fuller test on the residuals. This is disclosed explicitly, the same
way the plan asks the Deflated Sharpe Ratio's approximations to be disclosed.

Approximations / simplifications, disclosed:
  - ADF critical values for the residual-based EG test use the commonly
    cited MacKinnon (2010) asymptotic approximations for a 2-variable,
    no-trend cointegrating regression (-3.90 / -3.34 / -3.04 for 1/5/10%).
  - Lag length for the ADF regression is fixed at 5 (no AIC/BIC search).
  - Granger lead-lag testing and rolling 30/90-day correlation surfaces are
    DEFERRED given the timebox -- noted explicitly in the report's
    limitations rather than faked.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import itertools

EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


def adf_stat(series: np.ndarray, lags: int = 5, include_const: bool = True) -> float:
    y = series
    dy = np.diff(y)
    n = len(dy) - lags
    if n < 30:
        return np.nan
    Y = dy[lags:]
    X_cols = [y[lags:-1]]
    for i in range(1, lags + 1):
        X_cols.append(dy[lags - i: len(dy) - i])
    X = np.column_stack(X_cols)
    if include_const:
        X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(Y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    rho_idx = 1 if include_const else 0
    se_rho = np.sqrt(cov[rho_idx, rho_idx])
    rho = beta[rho_idx]
    return float(rho / se_rho)


def engle_granger(y: np.ndarray, x: np.ndarray, lags: int = 5) -> dict:
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    stat = adf_stat(resid, lags=lags, include_const=False)
    r0, r1 = resid[:-1], resid[1:]
    phi = float(np.dot(r0, r1) / np.dot(r0, r0)) if np.dot(r0, r0) > 0 else np.nan
    half_life = -np.log(2) / np.log(phi) if (phi is not None and 0 < phi < 1) else np.nan
    return dict(adf_stat=stat, alpha=float(beta[0]), beta=float(beta[1]), phi=phi, half_life_bars=half_life)


def pair_diagnostics(data: dict, symbols: list[str]) -> pd.DataFrame:
    rows = []
    common_idx = None
    log_close, ret = {}, {}
    for s in symbols:
        c = np.log(data[s]["close"])
        log_close[s] = c
        ret[s] = data[s]["close"].pct_change()
        common_idx = c.index if common_idx is None else common_idx.intersection(c.index)

    for a, b in itertools.combinations(symbols, 2):
        ra, rb = ret[a].reindex(common_idx), ret[b].reindex(common_idx)
        corr = ra.corr(rb)
        xa, xb = log_close[a].reindex(common_idx).to_numpy(), log_close[b].reindex(common_idx).to_numpy()
        eg_ab = engle_granger(xb, xa)
        eg_ba = engle_granger(xa, xb)
        best = eg_ab if eg_ab["adf_stat"] < eg_ba["adf_stat"] else eg_ba
        direction = "B_on_A" if best is eg_ab else "A_on_B"
        rows.append(dict(
            pair=f"{a}/{b}", asset_a=a, asset_b=b, return_corr=corr,
            adf_stat=best["adf_stat"], cointegrated_1pct=best["adf_stat"] < EG_CRIT[0.01],
            cointegrated_5pct=best["adf_stat"] < EG_CRIT[0.05],
            cointegrated_10pct=best["adf_stat"] < EG_CRIT[0.10],
            direction=direction, hedge_ratio=best["beta"], half_life_bars=best["half_life_bars"],
        ))
    return pd.DataFrame(rows).sort_values("adf_stat")


if __name__ == "__main__":
    from backtest.data import load_all
    data = load_all()
    df = pair_diagnostics(data, list(data.keys()))
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
