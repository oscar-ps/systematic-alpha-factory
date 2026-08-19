from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as st
from backtest.engine import trade_returns
from gate.criteria import L4

EULER_MASCHERONI = 0.5772156649015329


def permutation_test(bt: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    """Null: no genuine timing skill -- shuffle the realized bar returns
    (breaking any temporal structure the strategy might be exploiting) while
    holding the strategy's position sequence (and hence its trading costs)
    fixed, and recompute performance."""
    pos = bt["position"].to_numpy()
    cost = bt["cost"].to_numpy()
    ret = bt["ret"].to_numpy()

    def sharpe_of(r):
        sr = pos * r - cost
        s = sr.std(ddof=0)
        return (sr.mean() / s) * np.sqrt(24 * 365.25) if s > 0 else 0.0

    observed = sharpe_of(ret)
    perm_sharpes = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(ret)
        perm_sharpes[i] = sharpe_of(shuffled)

    pvalue = float(np.mean(perm_sharpes >= observed))
    return dict(observed_sharpe=observed, pvalue=pvalue, perm_mean=float(perm_sharpes.mean()), perm_std=float(perm_sharpes.std()))


def bootstrap_drawdown(bt: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    """Resample discrete trades (with replacement) to build an empirical
    distribution of maximum drawdown, rather than trusting the single
    historical path."""
    trades = trade_returns(bt)
    if len(trades) < 5:
        return dict(p50_dd=0.0, p95_dd=0.0, n_trades=len(trades))
    trades_arr = trades.to_numpy()
    n = len(trades_arr)
    mdds = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(trades_arr, size=n, replace=True)
        eq = np.cumprod(1 + sample)
        running_max = np.maximum.accumulate(eq)
        dd = eq / running_max - 1.0
        mdds[i] = dd.min()
    return dict(p50_dd=float(np.percentile(mdds, 50)), p95_dd=float(np.percentile(mdds, 5)), n_trades=n)
    # note: "p95_dd" = the 5th percentile of the drawdown distribution, i.e. the
    # *worse* tail (drawdowns are negative numbers, so the 5th percentile of the
    # distribution is the 95th-percentile-worst outcome)


def expected_max_sharpe_under_null(sigma_sr: float, n_trials: int) -> float:
    if n_trials <= 1 or sigma_sr <= 0:
        return 0.0
    z1 = st.norm.ppf(1 - 1.0 / n_trials)
    z2 = st.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return sigma_sr * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(strat_ret: pd.Series, sigma_sr_cross_section: float, n_trials: int) -> dict:
    r = strat_ret.dropna().to_numpy()
    T = len(r)
    s = r.std(ddof=0)
    sr_hat = (r.mean() / s) if s > 0 else 0.0  # per-period (NOT annualized) Sharpe
    skew = float(st.skew(r)) if T > 2 else 0.0
    kurt = float(st.kurtosis(r, fisher=False)) if T > 3 else 3.0  # non-excess; normal=3

    sr0 = expected_max_sharpe_under_null(sigma_sr_cross_section, n_trials)
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat**2))
    z = (sr_hat - sr0) * np.sqrt(max(T - 1, 1)) / denom
    dsr = float(st.norm.cdf(z))
    return dict(dsr=dsr, sr_hat_per_period=float(sr_hat), sr0_benchmark=float(sr0), skew=skew, kurtosis=kurt, T=T)


def evaluate_instance_layer4(bt: pd.DataFrame, sigma_sr_cross_section: float, n_trials: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    perm = permutation_test(bt, L4["n_permutations"], rng)
    boot = bootstrap_drawdown(bt, L4["n_bootstrap"], rng)
    dsr_res = deflated_sharpe_ratio(bt["strat_ret"], sigma_sr_cross_section, n_trials)

    perm_pass = perm["pvalue"] < L4["perm_pvalue_max"]
    dsr_pass = dsr_res["dsr"] >= L4["dsr_min"]
    boot_pass = boot["p95_dd"] >= -L4["max_dd_bootstrap_p95"]

    passed = perm_pass and dsr_pass and boot_pass
    fail_reason = None
    if not passed:
        if not perm_pass:
            fail_reason = "permutation_test_failed"
        elif not dsr_pass:
            fail_reason = "deflated_sharpe_below_threshold"
        else:
            fail_reason = "bootstrap_drawdown_tail_too_severe"

    return dict(
        passed=passed,
        perm_pvalue=perm["pvalue"],
        perm_observed_sharpe=perm["observed_sharpe"],
        bootstrap_p95_dd=boot["p95_dd"],
        bootstrap_p50_dd=boot["p50_dd"],
        n_trades=boot["n_trades"],
        dsr=dsr_res["dsr"],
        sr_hat_per_period=dsr_res["sr_hat_per_period"],
        sr0_benchmark=dsr_res["sr0_benchmark"],
        fail_reason=fail_reason,
    )
