"""
run_funnel.py v2 — EGARCH-conditioned breakout (single-asset) + cross-asset
relative-value reversion (pairs), pushed through the same pre-registered
4-layer gate. One command:

    PYTHONPATH=. python3 run_funnel.py

Outputs -> results/funnel_summary.csv, results/instance_level.csv,
results/walkforward_groups.csv, results/diagnostics/cointegration_tests.csv,
results/run_meta.json, results/figures/*.png (via make_figures.py).
"""
from __future__ import annotations
import time
import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from backtest.data import load_all
from factory.expansion import expand_population, family_param_grid, SYMBOLS
from gate.instance_runner import make_runner
from gate.layer1_is_oos import evaluate_instance_layer1
from gate.layer2_walkforward import evaluate_instance_layer2, compute_group_reoptimization
from gate.layer3_stress import evaluate_instance_layer3, find_calm_window, find_worst_vol_window
from gate.layer4_montecarlo import evaluate_instance_layer4
from gate.criteria import L3
from gate.taxonomy import classify_failure
from diagnostics.cointegration import pair_diagnostics

RESULTS_DIR = "results"


def per_period_sharpe(strat_ret: pd.Series) -> float:
    r = strat_ret.dropna()
    s = r.std(ddof=0)
    return float(r.mean() / s) if s > 0 else 0.0


def main():
    t0 = time.time()
    print("=" * 70)
    print("ALPHA FACTORY v3 — EGARCH breakout + session-based liquidity reversion")
    print("=" * 70)

    print("\n[1/8] Loading & resampling market data (H1, cached)...")
    data = load_all()
    for sym, df in data.items():
        print(f"   {sym}: {len(df)} H1 bars, {df.index.min()} -> {df.index.max()}")

    print("\n[2/8] Cross-asset diagnostics (correlation + Engle-Granger cointegration)...")
    diag = pair_diagnostics(data, SYMBOLS)
    os.makedirs(f"{RESULTS_DIR}/diagnostics", exist_ok=True)
    diag.to_csv(f"{RESULTS_DIR}/diagnostics/cointegration_tests.csv", index=False)
    print(diag.to_string(index=False))
    print("   -> no pair clears even a 10% cointegration bar (see column cointegrated_10pct).")
    print("   -> this is why Family B was redesigned as session_reversion (single-asset, no")
    print("      cross-asset relationship needed) -- see gate/pre_registration.yaml change_log.")

    print("\n[3/8] Expanding variant population...")
    population = expand_population()
    print(f"   {len(population)} candidate instances ({Counter(c.family for c in population)})")
    runners = {c.id: make_runner(c.family, c.assets, c.params) for c in population}

    print("\n[4/8] Pre-computing calm + post-warmup stress windows per symbol...")
    calm_windows = {}
    postwarmup_windows = {}
    for sym, df in data.items():
        cs, ce = find_calm_window(df, L3["synthetic_window_days"])
        calm_windows[sym] = (cs, ce)
        pws, pwe = find_worst_vol_window(df, L3["postwarmup_window_days"])
        postwarmup_windows[sym] = (pws, pwe)
        print(f"   {sym}: calm window {cs.date()} -> {ce.date()}; "
              f"post-warmup worst-vol window {pws.date()} -> {pwe.date()}")

    def calm_for(c):
        return calm_windows[c.assets[-1]]

    def postwarmup_for(c):
        return postwarmup_windows[c.assets[-1]]

    print("\n[5/8] Full-history backtest for every instance (warms EGARCH cache; needed for Layer 3 & cross-sectional Sharpe)...")
    bt_full = {}
    full_sharpes = []
    for i, c in enumerate(population):
        bt = runners[c.id].bt(data)
        bt_full[c.id] = bt
        full_sharpes.append(per_period_sharpe(bt["strat_ret"]))
        if (i + 1) % 12 == 0:
            print(f"   ...{i+1}/{len(population)} full-history backtests done ({time.time()-t0:.0f}s elapsed)")
    sigma_sr_cross_section = float(np.std(full_sharpes, ddof=1))
    print(f"   cross-sectional std of per-period Sharpe across {len(population)} trials = {sigma_sr_cross_section:.5f}")

    print("\n[6/8] Layer 2 group-level re-optimization (one pass per symbol / pair group)...")
    group_chosen = {}
    wf_group_rows = []
    for sym in SYMBOLS:
        grid = family_param_grid("breakout_egarch")
        runners_by_key = {tuple(sorted(p.items())): make_runner("breakout_egarch", (sym,), p) for p in grid}
        chosen_map, detail = compute_group_reoptimization(runners_by_key, data)
        group_chosen[("breakout_egarch", (sym,))] = chosen_map
        for d in detail:
            wf_group_rows.append(dict(family="breakout_egarch", assets=sym, **d))
        print(f"   breakout_egarch/{sym}: {len(detail)} re-optimized windows ({time.time()-t0:.0f}s elapsed)")
    for sym in SYMBOLS:
        grid = family_param_grid("session_reversion")
        runners_by_key = {tuple(sorted(p.items())): make_runner("session_reversion", (sym,), p) for p in grid}
        chosen_map, detail = compute_group_reoptimization(runners_by_key, data)
        group_chosen[("session_reversion", (sym,))] = chosen_map
        for d in detail:
            wf_group_rows.append(dict(family="session_reversion", assets=sym, **d))
        print(f"   session_reversion/{sym}: {len(detail)} re-optimized windows ({time.time()-t0:.0f}s elapsed)")
    pd.DataFrame(wf_group_rows).to_csv(f"{RESULTS_DIR}/walkforward_groups.csv", index=False)

    print("\n[7/8] Running gate Layers 1-3 independently on the FULL population...")
    rows = []
    for i, c in enumerate(population):
        runner = runners[c.id]
        l1 = evaluate_instance_layer1(runner, data)
        l2 = evaluate_instance_layer2(runner, data, group_chosen=group_chosen.get((c.family, c.assets)))
        l3 = evaluate_instance_layer3(runner, data, calm_for(c), postwarmup_window=postwarmup_for(c))

        row = dict(
            instance_id=c.id, family=c.family, assets="/".join(c.assets), params=str(c.params),
            l1_pass=l1["passed"], l1_oos_sharpe=l1["oos_sharpe"], l1_oos_return=l1["oos_total_return"],
            l1_oos_dd=l1["oos_max_dd"], l1_oos_trades=l1["oos_trades"], l1_fail=l1["fail_reason"],
            l2_pass=l2["passed"], l2_n_windows=l2.get("n_windows"), l2_mean_oow_sharpe=l2.get("mean_oow_sharpe"),
            l2_path=l2.get("path"), l2_n_selected=l2.get("n_times_selected"), l2_fail=l2["fail_reason"],
            l3_pass=l3["passed"], l3_covid_return=l3["covid_return"], l3_covid_untested=l3["covid_untested"],
            l3_postwarmup_return=l3["postwarmup_return"], l3_postwarmup_pass=l3["postwarmup_pass"],
            l3_synth_return=l3["synthetic_return"], l3_fail=l3["fail_reason"],
        )
        rows.append(row)
        if (i + 1) % 12 == 0 or i == len(population) - 1:
            print(f"   ...{i+1}/{len(population)} instances through Layers 1-3 ({time.time()-t0:.0f}s elapsed)")

    df_inst = pd.DataFrame(rows)

    print("\n[8/8] Running Layer 4 (Monte Carlo) on the Layer1∩Layer2∩Layer3 survivors...")
    survivors_123 = df_inst[df_inst["l1_pass"] & df_inst["l2_pass"] & df_inst["l3_pass"]]["instance_id"].tolist()
    print(f"   {len(survivors_123)} instances entered Layer 4: {survivors_123}")

    l4_results = {}
    for cid in survivors_123:
        l4 = evaluate_instance_layer4(bt_full[cid], sigma_sr_cross_section, len(population))
        l4_results[cid] = l4

    df_inst["l4_pass"] = df_inst["instance_id"].map(lambda x: l4_results[x]["passed"] if x in l4_results else False)
    df_inst["l4_dsr"] = df_inst["instance_id"].map(lambda x: l4_results[x]["dsr"] if x in l4_results else np.nan)
    df_inst["l4_perm_pvalue"] = df_inst["instance_id"].map(lambda x: l4_results[x]["perm_pvalue"] if x in l4_results else np.nan)
    df_inst["l4_bootstrap_p95_dd"] = df_inst["instance_id"].map(lambda x: l4_results[x]["bootstrap_p95_dd"] if x in l4_results else np.nan)
    df_inst["l4_fail"] = df_inst["instance_id"].map(lambda x: l4_results[x]["fail_reason"] if x in l4_results else "did_not_reach_layer4")
    df_inst["final_survivor"] = df_inst["l1_pass"] & df_inst["l2_pass"] & df_inst["l3_pass"] & df_inst["l4_pass"]

    taxonomy_rows = df_inst.apply(lambda r: classify_failure(r.to_dict()), axis=1, result_type="expand")
    df_inst = pd.concat([df_inst, taxonomy_rows], axis=1)

    n0 = len(df_inst)
    n1 = int(df_inst["l1_pass"].sum())
    n2 = int((df_inst["l1_pass"] & df_inst["l2_pass"]).sum())
    n3 = int((df_inst["l1_pass"] & df_inst["l2_pass"] & df_inst["l3_pass"]).sum())
    n4 = int(df_inst["final_survivor"].sum())

    def dominant_fail(mask, col):
        sub = df_inst[mask][col].dropna()
        return Counter(sub).most_common(1)[0][0] if len(sub) else "n/a"

    funnel_rows = [
        dict(layer="0_population", candidates_in=n0, candidates_out=n0, dominant_fail_reason="n/a"),
        dict(layer="1_is_oos_sensitivity", candidates_in=n0, candidates_out=n1, dominant_fail_reason=dominant_fail(df_inst["instance_id"].notna(), "l1_fail")),
        dict(layer="2_walkforward", candidates_in=n1, candidates_out=n2, dominant_fail_reason=dominant_fail(df_inst["l1_pass"], "l2_fail")),
        dict(layer="3_stress_replay", candidates_in=n2, candidates_out=n3, dominant_fail_reason=dominant_fail(df_inst["l1_pass"] & df_inst["l2_pass"], "l3_fail")),
        dict(layer="4_montecarlo_dsr", candidates_in=n3, candidates_out=n4, dominant_fail_reason=dominant_fail(df_inst["l1_pass"] & df_inst["l2_pass"] & df_inst["l3_pass"], "l4_fail")),
    ]
    df_funnel = pd.DataFrame(funnel_rows)

    df_inst.to_csv(f"{RESULTS_DIR}/instance_level.csv", index=False)
    df_funnel.to_csv(f"{RESULTS_DIR}/funnel_summary.csv", index=False)

    with open(f"{RESULTS_DIR}/bt_full_survivors123.pkl", "wb") as f:
        pickle.dump({cid: bt_full[cid][["equity", "strat_ret"]] for cid in survivors_123}, f)

    meta = dict(
        sigma_sr_cross_section=sigma_sr_cross_section,
        n_population=len(population),
        best_full_sample_sharpe_ann=float(max(full_sharpes) * np.sqrt(24 * 365.25)),
        calm_windows={k: [str(v[0].date()), str(v[1].date())] for k, v in calm_windows.items()},
        n_breakout=int((df_inst.family == "breakout_egarch").sum()),
        n_session_reversion=int((df_inst.family == "session_reversion").sum()),
    )
    with open(f"{RESULTS_DIR}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("FUNNEL SUMMARY")
    print("=" * 70)
    print(df_funnel.to_string(index=False))
    print(f"\nFinal survivors ({n4}):")
    if n4 > 0:
        print(df_inst[df_inst["final_survivor"]][["instance_id", "l4_dsr", "l4_perm_pvalue"]].to_string(index=False))
    else:
        print("   none")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    return df_inst, df_funnel


if __name__ == "__main__":
    main()
