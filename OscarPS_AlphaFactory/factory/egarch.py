"""
EGARCH(1,1) -- Nelson (1991) exponential GARCH, fit by maximum likelihood.

Why EGARCH and not a plain rolling std: conditional volatility clusters and
reacts ASYMMETRICALLY to shocks (the "leverage effect" -- a negative return
raises future volatility more than a positive return of the same size).
A naive rolling-std Z-score implicitly assumes constant volatility within
the window and treats +1% and -1% moves identically; it understates how
extreme a move was during a volatility expansion (right when reversion is
most dangerous to trade) and overstates it in calm periods. EGARCH gives a
single-step-ahead conditional volatility forecast that adapts to this
asymmetry, which is the statistical engine behind root idea #2 below.

This is a classical parametric econometric model fit by numerical MLE
(5 free parameters, closed-form recursion, Gaussian innovations) -- not a
flexible/learned function approximator, so it sits outside the assignment's
"no ML signal models" restriction the same way an ARIMA or a GARCH(1,1)
would.

Model:
    r_t = mu + eps_t,  eps_t = sigma_t * z_t,  z_t ~ N(0,1) iid
    ln(sigma_t^2) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1} + beta*ln(sigma_{t-1}^2)
    E|z| = sqrt(2/pi) for standard normal innovations.

Causal usage (no lookahead): parameters are estimated ONLY on a trailing
window of returns ending strictly before the bar being scored, and re-
estimated periodically (every `refit_window` bars) on the most recent
`refit_window` returns. Between re-estimations the conditional variance
recursion is filtered forward bar-by-bar using only realized returns (the
recursion itself is causal: sigma_t depends only on r_{t-1}, sigma_{t-1}).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

E_ABS_Z = np.sqrt(2.0 / np.pi)


def _negloglik(theta, r):
    mu, omega, alpha, gamma, beta = theta
    T = len(r)
    ln_s2 = np.empty(T)
    z = np.empty(T)
    ln_s2[0] = np.log(np.var(r) + 1e-12)
    z[0] = (r[0] - mu) / np.sqrt(np.exp(ln_s2[0]))
    for t in range(1, T):
        ln_s2[t] = omega + alpha * (abs(z[t - 1]) - E_ABS_Z) + gamma * z[t - 1] + beta * ln_s2[t - 1]
        ln_s2[t] = np.clip(ln_s2[t], -30, 10)
        s2 = np.exp(ln_s2[t])
        z[t] = (r[t] - mu) / np.sqrt(s2)
    nll = 0.5 * np.sum(np.log(2 * np.pi) + ln_s2 + z**2)
    if not np.isfinite(nll):
        return 1e10
    return nll


def fit_egarch(r: np.ndarray, x0: tuple | None = None) -> tuple[float, float, float, float, float]:
    """MLE fit of EGARCH(1,1) on a trailing return window. Returns
    (mu, omega, alpha, gamma, beta). x0 lets the caller warm-start from the
    previous window's fit (faster, more stable across rolling refits)."""
    if x0 is None:
        x0 = (float(np.mean(r)), float(np.log(np.var(r) + 1e-12)) * 0.1, 0.1, -0.05, 0.85)
    bounds = [(-0.05, 0.05), (-3, 3), (0.001, 0.5), (-0.5, 0.5), (-0.995, 0.995)]
    res = minimize(_negloglik, x0=x0, args=(r,), method="L-BFGS-B", bounds=bounds,
                    options=dict(maxiter=200, ftol=1e-8))
    return tuple(res.x)


def rolling_egarch_sigma(returns: np.ndarray, refit_window: int) -> np.ndarray:
    """Like rolling_egarch_zscore, but returns the filtered conditional
    volatility (sigma_t, not standardized residual z_t) -- used as a vol-
    regime FILTER (breakout family) rather than as a standardized extremity
    score."""
    T = len(returns)
    sigma = np.zeros(T)
    if T <= refit_window + 2:
        return sigma

    cur_ln_s2 = np.log(np.var(returns[:refit_window]) + 1e-12)
    sigma[:refit_window] = np.sqrt(np.exp(cur_ln_s2))
    cur_z = 0.0
    x0 = None
    t = refit_window
    while t < T:
        train = returns[max(0, t - refit_window):t]
        mu, omega, alpha, gamma, beta = fit_egarch(train, x0=x0)
        x0 = (mu, omega, alpha, gamma, beta)
        end = min(t + refit_window, T)
        for tt in range(t, end):
            cur_ln_s2 = omega + alpha * (abs(cur_z) - E_ABS_Z) + gamma * cur_z + beta * cur_ln_s2
            cur_ln_s2 = float(np.clip(cur_ln_s2, -30, 10))
            s2 = np.exp(cur_ln_s2)
            cur_z = float(np.clip((returns[tt] - mu) / np.sqrt(s2), -8.0, 8.0))
            sigma[tt] = np.sqrt(s2)
        t = end
    return sigma


def rolling_egarch_zscore(returns: np.ndarray, refit_window: int) -> np.ndarray:
    """
    Causal, rolling-refit EGARCH conditional-volatility Z-score for an
    entire return series.

    For bars [0, refit_window): no model yet -> z = 0 (no trade).
    At each refit point k*refit_window (k=1,2,...): re-estimate EGARCH on the
    trailing `refit_window` returns ending at that point, then filter the
    conditional variance forward bar-by-bar (continuing the sigma recursion
    continuously across refit boundaries) using the newly estimated params,
    until the next refit point.
    """
    T = len(returns)
    z = np.zeros(T)
    ln_s2 = np.zeros(T)
    if T <= refit_window + 2:
        return z

    ln_s2[:refit_window] = np.log(np.var(returns[:refit_window]) + 1e-12)
    params = None
    next_refit = refit_window

    # warm-start the recursion's running state at the first refit point
    mu = float(np.mean(returns[:refit_window]))
    cur_ln_s2 = np.log(np.var(returns[:refit_window]) + 1e-12)
    cur_z = 0.0

    x0 = None
    t = refit_window
    while t < T:
        train = returns[max(0, t - refit_window):t]
        mu, omega, alpha, gamma, beta = fit_egarch(train, x0=x0)
        x0 = (mu, omega, alpha, gamma, beta)

        end = min(t + refit_window, T)
        for tt in range(t, end):
            cur_ln_s2 = omega + alpha * (abs(cur_z) - E_ABS_Z) + gamma * cur_z + beta * cur_ln_s2
            cur_ln_s2 = float(np.clip(cur_ln_s2, -30, 10))
            s2 = np.exp(cur_ln_s2)
            cur_z = (returns[tt] - mu) / np.sqrt(s2)
            cur_z = float(np.clip(cur_z, -8.0, 8.0))  # numerical safety net, not a strategy parameter
            ln_s2[tt] = cur_ln_s2
            z[tt] = cur_z
        t = end

    return z
