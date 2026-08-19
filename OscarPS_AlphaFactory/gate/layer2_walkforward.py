from __future__ import annotations
import numpy as np
import pandas as pd
from backtest.engine import perf_stats, trade_returns
from gate.criteria import L2


def rolling_windows(index: pd.DatetimeIndex, train_weeks: int, test_weeks: int):
    start, end = index.min(), index.max()
    train_td, test_td = pd.Timedelta(weeks=train_weeks), pd.Timedelta(weeks=test_weeks)
    windows = []
    cur = start
    while cur + train_td + test_td <= end:
        windows.append((cur, cur + train_td, cur + train_td, cur + train_td + test_td))
        cur = cur + test_td
    return windows


def compute_group_reoptimization(runners_by_paramkey: dict, data: dict) -> tuple[dict, list]:
    any_runner = next(iter(runners_by_paramkey.values()))
    full_idx = any_runner.bt(data).index
    windows = rolling_windows(full_idx, L2["train_weeks"], L2["test_weeks"])

    chosen_map, detail = {}, []
    for (tr_s, tr_e, te_s, te_e) in windows:
        best_params, best_sharpe = None, -np.inf
        for key, r in runners_by_paramkey.items():
            train_bt = r.bt_slice(data, tr_s, tr_e)
            if len(train_bt) < 24 * 7:
                continue
            s = perf_stats(train_bt["strat_ret"])
            n_trades = len(trade_returns(train_bt))
            if s["total_return"] > 0 and n_trades >= 5 and s["sharpe"] > best_sharpe:
                best_sharpe, best_params = s["sharpe"], dict(key)
        chosen_map[(te_s, te_e)] = best_params
        if best_params is not None:
            test_bt = runners_by_paramkey[tuple(sorted(best_params.items()))].bt_slice(data, te_s, te_e)
            ts = perf_stats(test_bt["strat_ret"])
            detail.append(dict(test_start=te_s, test_end=te_e, chosen_params=best_params,
                                train_sharpe=best_sharpe, test_sharpe=ts["sharpe"], test_return=ts["total_return"]))
    return chosen_map, detail


def evaluate_instance_layer2(runner, data: dict, group_chosen: dict | None = None) -> dict:
    full_bt = runner.bt(data)
    windows = rolling_windows(full_bt.index, L2["train_weeks"], L2["test_weeks"])

    own_results = []
    for (tr_s, tr_e, te_s, te_e) in windows:
        test_bt = runner.bt_slice(data, te_s, te_e)
        if len(test_bt) < 24 * 7:
            continue
        s = perf_stats(test_bt["strat_ret"])
        n_trades = len(trade_returns(test_bt))
        own_results.append(dict(test_start=te_s, test_end=te_e, sharpe=s["sharpe"],
                                 total_return=s["total_return"], max_dd=s["max_dd"], n_trades=n_trades))

    if len(own_results) < L2["min_windows"]:
        return dict(passed=False, n_windows=len(own_results), path=None,
                     fail_reason="insufficient_walk_forward_windows", own_results=own_results)

    mean_sharpe = float(np.mean([w["sharpe"] for w in own_results]))
    frac_pos = float(np.mean([w["total_return"] > 0 for w in own_results]))
    worst_dd = float(min(w["max_dd"] for w in own_results))
    agg_trades = int(sum(w["n_trades"] for w in own_results))

    own_pass = (
        mean_sharpe >= L2["min_wf_sharpe"]
        and frac_pos >= L2["min_frac_windows_positive"]
        and agg_trades >= L2["min_wf_trades"]
        and worst_dd >= -L2["max_wf_drawdown"]
    )

    selected_pass = False
    n_selected = 0
    if group_chosen is not None:
        own_key = tuple(sorted(runner.params.items()))
        sel_sharpes, sel_returns = [], []
        for (tr_s, tr_e, te_s, te_e) in windows:
            chosen = group_chosen.get((te_s, te_e))
            if chosen is not None and tuple(sorted(chosen.items())) == own_key:
                n_selected += 1
                match = next((w for w in own_results if w["test_start"] == te_s), None)
                if match:
                    sel_sharpes.append(match["sharpe"])
                    sel_returns.append(match["total_return"])
        if n_selected >= 1 and len(sel_sharpes) > 0:
            selected_pass = (np.mean(sel_sharpes) >= L2["min_wf_sharpe"]) and all(r > 0 for r in sel_returns)

    passed = own_pass or selected_pass
    path = "own_params" if own_pass else ("selected_winner" if selected_pass else None)
    fail_reason = None
    if not passed:
        fail_reason = "mean_wf_sharpe_below_min" if mean_sharpe < L2["min_wf_sharpe"] else (
            "too_few_wf_trades" if agg_trades < L2["min_wf_trades"] else (
                "wf_drawdown_exceeded" if worst_dd < -L2["max_wf_drawdown"] else "inconsistent_across_windows"))

    return dict(
        passed=passed, n_windows=len(own_results), mean_oow_sharpe=mean_sharpe, frac_positive=frac_pos,
        worst_oow_dd=worst_dd, agg_trades=agg_trades, n_times_selected=n_selected, path=path,
        own_results=own_results, fail_reason=fail_reason,
    )
