"""
Systematic failure taxonomy. Every candidate gets: failed_layer,
primary_failure_reason, secondary_failure_reason, worst_metric -- not just
a free-text reason string, so the funnel's "why" is queryable.
"""
from __future__ import annotations

L1_MAP = {
    "oos_sharpe_ok": "LOW_OOS_SHARPE",
    "oos_return_ok": "NEGATIVE_OOS_RETURN",
    "oos_dd_ok": "EXCESSIVE_OOS_DRAWDOWN",
    "oos_trades_ok": "TOO_FEW_TRADES",
    "neighbor_sharpe_ok": "PARAMETER_FRAGILITY",
    "neighbor_dd_ok": "PARAMETER_FRAGILITY",
}
L2_MAP = {
    "insufficient_walk_forward_windows": "TOO_FEW_TRADES",
    "mean_wf_sharpe_below_min": "WALK_FORWARD_INSTABILITY",
    "too_few_wf_trades": "TOO_FEW_TRADES",
    "wf_drawdown_exceeded": "WALK_FORWARD_INSTABILITY",
    "inconsistent_across_windows": "WALK_FORWARD_INSTABILITY",
}
L3_MAP = {
    "covid_loss_exceeded": "STRESS_DRAWDOWN",
    "postwarmup_stress_loss_exceeded": "STRESS_DRAWDOWN",
    "synthetic_loss_exceeded": "STRESS_DRAWDOWN",
    "stress_dd_exceeded_full_sample_multiple": "STRESS_DRAWDOWN",
    "pair_relationship_breakdown": "PAIR_RELATIONSHIP_BREAKDOWN",
}
L4_MAP = {
    "permutation_test_failed": "PERMUTATION_FAIL",
    "deflated_sharpe_below_threshold": "DSR_FAIL",
    "bootstrap_drawdown_tail_too_severe": "BOOTSTRAP_DRAWDOWN_FAIL",
}

WORST_METRIC_COL = {
    "LOW_OOS_SHARPE": "l1_oos_sharpe", "NEGATIVE_OOS_RETURN": "l1_oos_return",
    "EXCESSIVE_OOS_DRAWDOWN": "l1_oos_dd", "TOO_FEW_TRADES": "l1_oos_trades",
    "PARAMETER_FRAGILITY": "l1_oos_sharpe", "WALK_FORWARD_INSTABILITY": "l2_mean_oow_sharpe",
    "STRESS_DRAWDOWN": "l3_covid_return", "PAIR_RELATIONSHIP_BREAKDOWN": "l3_synth_return",
    "PERMUTATION_FAIL": "l4_perm_pvalue", "DSR_FAIL": "l4_dsr",
    "BOOTSTRAP_DRAWDOWN_FAIL": "l4_bootstrap_p95_dd",
}


def classify_failure(row: dict) -> dict:
    """row must have l1_pass..l4_pass, l*_fail strings, family, and the
    numeric columns referenced in WORST_METRIC_COL."""
    failed_layer, primary = None, None
    if not row.get("l1_pass", True):
        failed_layer, primary = 1, L1_MAP.get(row.get("l1_fail"), "LOW_OOS_SHARPE")
    elif not row.get("l2_pass", True):
        failed_layer, primary = 2, L2_MAP.get(row.get("l2_fail"), "WALK_FORWARD_INSTABILITY")
    elif not row.get("l3_pass", True):
        failed_layer, primary = 3, L3_MAP.get(row.get("l3_fail"), "STRESS_DRAWDOWN")
    elif not row.get("l4_pass", True):
        failed_layer, primary = 4, L4_MAP.get(row.get("l4_fail"), "DSR_FAIL")

    secondary = None
    if row.get("family") == "relval":
        secondary = "NO_COINTEGRATION_PRIOR"
    elif row.get("family") == "breakout_egarch" and row.get("l3_covid_untested"):
        secondary = "STRESS_UNTESTED" if secondary is None else secondary

    worst_metric = None
    if primary is not None:
        col = WORST_METRIC_COL.get(primary)
        worst_metric = row.get(col) if col else None

    return dict(failed_layer=failed_layer, primary_failure_reason=primary,
                secondary_failure_reason=secondary, worst_metric=worst_metric)
