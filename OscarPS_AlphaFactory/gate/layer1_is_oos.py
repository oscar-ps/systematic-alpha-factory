from __future__ import annotations
import numpy as np
from backtest.engine import perf_stats, trade_returns
from gate.criteria import L1


def _perturbed_param_sets(params: dict, pct: float, bounds: dict) -> list[dict]:
    out = []
    for k, v in params.items():
        for sign in (+1, -1):
            nv = v * (1 + sign * pct)
            nv = round(nv) if isinstance(v, int) else round(nv, 3)
            if isinstance(v, int):
                nv = max(bounds.get(k, (2, None))[0], int(nv))
            lo, hi = bounds.get(k, (None, None))
            if lo is not None and nv < lo:
                continue
            if hi is not None and nv > hi:
                continue
            p2 = dict(params)
            p2[k] = nv
            out.append(p2)
    return out


PARAM_BOUNDS = {
    "breakout_egarch": {"L": (10, None), "Q": (0.01, 0.99)},
    "session_reversion": {"lookback_days": (5, None), "z_entry": (0.2, None)},
    "relval": {"L": (20, None), "Z": (0.2, None)},
}


def evaluate_instance_layer1(runner, data: dict) -> dict:
    full_bt = runner.bt(data)
    full_index = full_bt.index
    split_idx = int(len(full_index) * L1["is_oos_split"])
    oos_start, oos_end = full_index[split_idx], full_index[-1]

    def stats_for(params):
        bt = runner.bt_slice(data, oos_start, oos_end, params=params)
        s = perf_stats(bt["strat_ret"])
        n_trades = len(trade_returns(bt))
        return s, n_trades

    base_stats, base_trades = stats_for(runner.params)

    bounds = PARAM_BOUNDS[runner.family]
    neighbors = _perturbed_param_sets(runner.params, L1["perturbation_pct"], bounds)
    neighbor_sharpes, neighbor_dds = [], []
    for p2 in neighbors:
        s, _ = stats_for(p2)
        neighbor_sharpes.append(s["sharpe"])
        neighbor_dds.append(s["max_dd"])

    median_neighbor_sharpe = float(np.median(neighbor_sharpes)) if neighbor_sharpes else np.nan
    worst_neighbor_dd = float(np.min(neighbor_dds)) if neighbor_dds else 0.0

    checks = dict(
        oos_sharpe_ok=base_stats["sharpe"] >= L1["min_oos_sharpe"],
        oos_return_ok=base_stats["total_return"] > L1["min_oos_total_return"],
        oos_dd_ok=base_stats["max_dd"] >= -L1["max_oos_drawdown"],
        oos_trades_ok=base_trades >= L1["min_oos_trades"],
        neighbor_sharpe_ok=median_neighbor_sharpe > L1["min_median_neighbor_sharpe"],
        neighbor_dd_ok=worst_neighbor_dd >= -L1["max_neighbor_drawdown"],
    )
    passed = all(checks.values())
    fail_reason = None
    if not passed:
        for name, ok in checks.items():
            if not ok:
                fail_reason = name
                break

    return dict(
        passed=passed, oos_sharpe=base_stats["sharpe"], oos_total_return=base_stats["total_return"],
        oos_max_dd=base_stats["max_dd"], oos_trades=base_trades,
        median_neighbor_sharpe=median_neighbor_sharpe, worst_neighbor_dd=worst_neighbor_dd,
        n_neighbors_tested=len(neighbors), fail_reason=fail_reason,
    )
