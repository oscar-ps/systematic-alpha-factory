"""
Root idea #1 — EGARCH-conditioned volatility breakout
-------------------------------------------------------
Hypothesis: a range breakout is more likely to be a genuine regime change
(macro repricing, institutional flow, forced deleveraging, reflexive
momentum) -- and therefore more likely to persist -- when it happens during
a period of elevated forecasted volatility, versus a breakout during a
quiet, low-vol regime which is more likely to be noise/a false break that
quickly mean-reverts. EGARCH is used ONLY to estimate conditional
volatility (a regime filter), never to predict direction.

EGARCH specification is FIXED (not optimized, to avoid overfitting the
vol model itself): EGARCH(1,1), Gaussian innovations, refit every 720 H1
bars (~1 month). The volatility-percentile lookback (used to rank today's
forecasted vol against its own recent history) is also fixed at 2160 H1
bars (~90 days). Both are computed ONCE per symbol and reused across every
(L, Q) instance on that symbol -- they do not vary with the free parameters
below, which keeps the EGARCH cost flat regardless of grid size.

Free parameters (2): breakout lookback in bars (L), EGARCH volatility-
percentile threshold (Q, in [0,1]).
Fixed by design, disclosed: the plan's literal rule ("otherwise: hold
previous position") has no exit mechanism at all, which we regard as an
unacceptable tail-risk omission -- we add a fixed ATR(14) catastrophic stop
(5x ATR, looser than a typical trend-following stop since this family is
deliberately patient) on top of the literal rule. This is a risk control,
not an alpha parameter, and is identical across all instances.

Root idea #2 — Rolling cross-asset relative-value reversion
-----------------------------------------------------------
Hypothesis: shared macro forces (risk sentiment, USD liquidity, safe-haven
demand, inflation expectations, deleveraging pressure) can temporarily
dislocate related assets, with a spread between them reverting once the
dislocation fades. This is NOT assumed to hold structurally -- the
diagnostic layer (diagnostics/cointegration.py) tests it empirically first.
As it happens, none of the 6 available pairs clears even a 10% cointegration
bar over the full sample (see results/diagnostics/cointegration_tests.csv),
which is itself an informative, honestly-reported finding: four assets
spanning equity, FX, gold and crypto have no good structural reason to
share a stable long-run relationship, and the data agrees. Per the
project's design, the family is run through the gate regardless (we do not
cherry-pick away from a clean negative result) -- we simply do not claim
the relationship is real going in.

Free parameters (2): rolling hedge-ratio/spread lookback in bars (L),
spread Z-score entry threshold (Z). Fixed by design: hedge ratio re-
estimated every bar via rolling OLS (no separate refit schedule); exit on
reversion through Z=0; fixed catastrophic stop at 1.5x Z (same convention
used throughout this project for mean-reversion-style exits).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from factory.egarch import rolling_egarch_sigma

_EGARCH_SIGMA_CACHE: dict[tuple, np.ndarray] = {}
EGARCH_REFIT_WINDOW = 720       # ~1 month, fixed, not a free parameter
VOL_PCT_LOOKBACK = 2160         # ~90 days, fixed, not a free parameter
ATR_STOP_MULT = 5.0             # fixed catastrophic-stop control, disclosed addition


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _df_fingerprint(df: pd.DataFrame) -> tuple:
    sym = df["symbol"].iloc[0] if "symbol" in df.columns else "?"
    c = df["close"]
    return (sym, len(df), str(df.index[0]), str(df.index[-1]), round(float(c.iloc[0]), 6), round(float(c.sum()), 4))


def _get_vol_pct(df: pd.DataFrame) -> pd.Series:
    """EGARCH conditional vol -> rolling percentile rank, fixed schedule,
    cached per (symbol, slice) so every (L, Q) instance reuses it for free."""
    key = _df_fingerprint(df)
    if key not in _EGARCH_SIGMA_CACHE:
        r = df["close"].pct_change().fillna(0.0).to_numpy()
        sigma = rolling_egarch_sigma(r, refit_window=EGARCH_REFIT_WINDOW)
        _EGARCH_SIGMA_CACHE[key] = sigma
    sigma = _EGARCH_SIGMA_CACHE[key]
    sigma_s = pd.Series(sigma, index=df.index)
    vol_pct = sigma_s.rolling(VOL_PCT_LOOKBACK).rank(pct=True)
    return vol_pct.fillna(0.0)


def breakout_egarch_signal(df: pd.DataFrame, L: int, Q: float, vol_pct_override: pd.Series | None = None) -> pd.Series:
    close = df["close"]
    upper = close.rolling(L).max().shift(1)
    lower = close.rolling(L).min().shift(1)
    vol_pct = vol_pct_override if vol_pct_override is not None else _get_vol_pct(df)
    a = atr(df, 14)

    close_v, upper_v, lower_v = close.to_numpy(), upper.to_numpy(), lower.to_numpy()
    vol_v, atr_v = vol_pct.to_numpy(), a.to_numpy()
    n = len(close_v)
    out = np.zeros(n)
    pos = 0.0
    entry_price = np.nan
    for i in range(n):
        if np.isnan(upper_v[i]) or np.isnan(lower_v[i]) or np.isnan(atr_v[i]):
            out[i] = pos
            continue
        breakout_long = close_v[i] > upper_v[i] and vol_v[i] >= Q
        breakout_short = close_v[i] < lower_v[i] and vol_v[i] >= Q

        if breakout_long:
            pos = 1.0
            entry_price = close_v[i]
        elif breakout_short:
            pos = -1.0
            entry_price = close_v[i]
        elif pos != 0.0 and not np.isnan(entry_price):
            stop_dist = ATR_STOP_MULT * atr_v[i]
            if pos > 0 and close_v[i] <= entry_price - stop_dist:
                pos = 0.0
            elif pos < 0 and close_v[i] >= entry_price + stop_dist:
                pos = 0.0
        # else: "otherwise, hold previous position" -- pos unchanged
        out[i] = pos
    return pd.Series(out, index=df.index, name="position_raw")


def relval_signal(x_close: pd.Series, y_close: pd.Series, L: int, Z: float, stop_mult: float = 1.5):
    """Rolling-hedge-ratio relative-value reversion between two assets.
    Returns (position, beta) -- position +1 = long spread = long y / short
    beta*x, -1 = short spread. beta is re-estimated every bar via rolling
    OLS (causal, trailing L bars up to and including the current bar; the
    backtest engine's single-bar shift handles no-lookahead before this is
    used to trade)."""
    x, y = np.log(x_close), np.log(y_close)
    cov_xy = x.rolling(L).cov(y)
    var_x = x.rolling(L).var()
    beta = (cov_xy / var_x).replace([np.inf, -np.inf], np.nan)
    alpha = y.rolling(L).mean() - beta * x.rolling(L).mean()
    spread = y - alpha - beta * x
    z = (spread - spread.rolling(L).mean()) / spread.rolling(L).std(ddof=0)
    z = z.fillna(0.0)

    z_v = z.to_numpy()
    n = len(z_v)
    out = np.zeros(n)
    cur = 0.0
    stop_z = stop_mult * Z
    for i in range(n):
        zi = z_v[i]
        if cur == 0.0:
            if zi >= Z:
                cur = -1.0
            elif zi <= -Z:
                cur = 1.0
        elif cur > 0:
            if zi >= 0 or zi <= -stop_z:
                cur = 0.0
        elif cur < 0:
            if zi <= 0 or zi >= stop_z:
                cur = 0.0
        out[i] = cur
    pos = pd.Series(out, index=x_close.index, name="position_raw")
    return pos, beta.fillna(0.0)


def session_reversion_signal(df: pd.DataFrame, lookback_days: int, z_entry: float, atr_stop_mult: float = 3.0) -> pd.Series:
    """
    Root idea #2 (v2) -- session-based liquidity reversion.

    Hypothesis: the Asian session (00:00-08:00, using the timestamps as given
    in the data -- see the disclosed assumption in the report about server
    timezone) is thin: fewer participants, wider effective spreads, more
    room for a single order or a stop run to push price further than
    fundamentals justify. When the London/NY session opens and real
    liquidity shows up, unusually large Asian-session moves tend to get
    partially corrected. This replaced the original Family B (cross-asset
    pairs reversion) after the diagnostics in Section 1 showed no structural
    relationship between any two of the four assets -- this version doesn't
    need one, since it only ever compares an asset's own Asian-session
    return against its own history.

    Mechanism: for each calendar day, compute the Asian-session return
    (close at 08:00 / close at 00:00 - 1), Z-score it against its own
    trailing history, and fade extreme days: short at 08:00 if the Asian
    session ran up too far, long if it ran down too far. Exit at the next
    day's 00:00 (a 16-hour hold, i.e. through the London/NY session).

    Free parameters (2): the Z-score lookback in trading days
    (lookback_days), and the entry threshold (z_entry).
    Fixed by design, disclosed: session boundaries (00:00/08:00) and the
    16-hour hold are part of the literal rule, not tuned. A fixed
    3x ATR(14) catastrophic stop is added on top, the same risk-control
    pattern used for breakout_egarch.
    """
    hours = df.index.hour
    close = df["close"]

    close0 = close[hours == 0]
    close0 = pd.Series(close0.to_numpy(), index=close0.index.normalize())
    close8 = close[hours == 8]
    close8 = pd.Series(close8.to_numpy(), index=close8.index.normalize())

    common_dates = close0.index.intersection(close8.index)
    asian_ret = (close8.reindex(common_dates) / close0.reindex(common_dates) - 1).sort_index()
    asian_ret = asian_ret[~asian_ret.index.duplicated(keep="first")]

    roll_mean = asian_ret.rolling(lookback_days).mean()
    roll_std = asian_ret.rolling(lookback_days).std(ddof=0)
    z = (asian_ret - roll_mean) / roll_std.replace(0, np.nan)
    z = z.fillna(0.0)

    direction = pd.Series(0.0, index=z.index)
    direction[z > z_entry] = -1.0   # Asian session ran up too far -> fade short
    direction[z < -z_entry] = 1.0   # ran down too far -> fade long
    direction_map = direction.to_dict()

    a = atr(df, 14)
    close_v, atr_v = close.to_numpy(), a.to_numpy()
    idx = df.index
    n = len(df)
    out = np.zeros(n)
    cur_dir = 0.0
    entry_price = np.nan

    for i in range(n):
        ts = idx[i]
        h = ts.hour
        if h == 0:
            cur_dir = 0.0
            entry_price = np.nan
        if h == 8:
            dir_today = direction_map.get(ts.normalize(), 0.0)
            if dir_today != 0.0:
                cur_dir = dir_today
                entry_price = close_v[i]
        if cur_dir != 0.0 and not np.isnan(entry_price) and not np.isnan(atr_v[i]):
            stop_dist = atr_stop_mult * atr_v[i]
            if cur_dir > 0 and close_v[i] <= entry_price - stop_dist:
                cur_dir = 0.0
            elif cur_dir < 0 and close_v[i] >= entry_price + stop_dist:
                cur_dir = 0.0
        out[i] = cur_dir

    return pd.Series(out, index=df.index, name="position_raw")


FAMILIES = {
    "breakout_egarch": {
        "func": breakout_egarch_signal,
        "param_names": ["L", "Q"],
    },
    "session_reversion": {
        "func": session_reversion_signal,
        "param_names": ["lookback_days", "z_entry"],
    },
    # relval (the original Family B, cross-asset pairs reversion) is kept in
    # the repo -- backtest/pairs_engine.py, factory root_ideas.relval_signal
    # -- but is no longer part of the active candidate population. The
    # diagnostics in Section 1 (zero of 6 pairs cointegrated, PC1 explaining
    # only 36% of variance, unstable rolling correlation) showed it had no
    # structural basis for this asset universe, so Family B was redesigned
    # as session_reversion above, which needs no cross-asset relationship
    # at all. It is intentionally NOT registered here for the same reason
    # it wasn't before: it needs the pairs-aware backtest path in
    # backtest/pairs_engine.py instead of the generic single-asset
    # run_backtest(); see gate/instance_runner.py for dispatch.
}

