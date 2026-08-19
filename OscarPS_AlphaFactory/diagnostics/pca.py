"""
PCA on H1 returns across the four symbols -- a direct, complementary check
on the cross-asset diagnostic story in cointegration.py. Cointegration asks
"do any two of these have a stable long-run relationship"; PCA asks "how
much of their *short-run co-movement* is explained by a small number of
common factors at all". Both inform whether Family B (relval) has a
believable common driver.

Implemented via eigendecomposition of the (standardized) return covariance
matrix -- classical multivariate statistics, not a fitted/learned model, so
it sits outside the assignment's "no ML signal models" restriction the same
way EGARCH and Engle-Granger do. No `sklearn` dependency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def run_pca(data: dict, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rets = {}
    common_idx = None
    for s in symbols:
        r = data[s]["close"].pct_change()
        rets[s] = r
        common_idx = r.index if common_idx is None else common_idx.intersection(r.index)

    R = pd.DataFrame({s: rets[s].reindex(common_idx) for s in symbols}).dropna()
    Z = (R - R.mean()) / R.std(ddof=0)  # standardize so no single asset's raw vol dominates

    cov = np.cov(Z.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    explained = eigvals / eigvals.sum()
    pc_names = [f"PC{i+1}" for i in range(len(symbols))]

    df_var = pd.DataFrame({
        "component": pc_names,
        "eigenvalue": eigvals,
        "explained_variance_ratio": explained,
        "cumulative_variance_ratio": np.cumsum(explained),
    })
    df_loadings = pd.DataFrame(eigvecs, index=symbols, columns=pc_names)
    return df_var, df_loadings


if __name__ == "__main__":
    from backtest.data import load_all
    data = load_all()
    syms = list(data.keys())
    df_var, df_load = run_pca(data, syms)
    print(df_var.to_string(index=False))
    print()
    print(df_load.to_string())
