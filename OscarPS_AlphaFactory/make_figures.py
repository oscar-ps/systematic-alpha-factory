"""Generate the figures used in report.tex. Run after run_funnel.py."""
import json
import pickle
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from gate.layer4_montecarlo import expected_max_sharpe_under_null

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
NAVY, BLUE, RED, GREY, GOLD, GREEN, PURPLE = "#1b2a4a", "#2f6fa3", "#b5443a", "#888888", "#c9982f", "#5a8f5a", "#9b59b6"

df_inst = pd.read_csv("results/instance_level.csv")
df_funnel = pd.read_csv("results/funnel_summary.csv")
with open("results/run_meta.json") as f:
    meta = json.load(f)

# ---------------------------------------------------------------- Figure 1: funnel
fig, ax = plt.subplots(figsize=(6.3, 2.6))
labels = ["Population", "L1: IS/OoS +\nsensitivity", "L2: Walk-\nforward", "L3: Stress\nreplay", "L4: Monte\nCarlo / DSR"]
vals = df_funnel["candidates_out"].tolist()
bars = ax.bar(labels, vals, color=[NAVY, BLUE, BLUE, BLUE, RED], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Candidates surviving")
ax.set_ylim(0, max(vals) * 1.12)
ax.set_title(f"Survival funnel — {meta['n_population']} candidates entered "
             f"({meta['n_breakout']} breakout_egarch + {meta['n_session_reversion']} session_reversion)",
             fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig("results/figures/funnel.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: DSR benchmark
sigma_sr = meta["sigma_sr_cross_section"]
n_trials_range = np.arange(1, 121)
sr0 = [expected_max_sharpe_under_null(sigma_sr, n) * np.sqrt(24 * 365.25) for n in n_trials_range]
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.plot(n_trials_range, sr0, color=RED, lw=2, label="Expected max Sharpe under null, $SR_0(N)$")
ax.axvline(meta["n_population"], color=GREY, ls="--", lw=1)
ax.text(meta["n_population"] + 1, max(sr0) * 0.12, f"N={meta['n_population']}\n(our population)", fontsize=8, color=GREY)
best = meta["best_full_sample_sharpe_ann"]
ax.axhline(best, color=NAVY, ls=":", lw=1.5, label=f"Best observed full-sample Sharpe ({best:.2f})")
ax.set_xlabel("Number of independent trials (N)")
ax.set_ylabel("Annualized Sharpe ratio")
ax.set_title(f"Why {meta['n_population']} trials deflates an apparently-good Sharpe", fontsize=10, loc="left")
ax.legend(fontsize=7.5, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("results/figures/dsr_benchmark.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: survivor equity curves
with open("results/bt_full_survivors123.pkl", "rb") as f:
    bts = pickle.load(f)
fig, ax = plt.subplots(figsize=(6.3, 3.1))
palette = [NAVY, BLUE, RED, GOLD, GREEN, PURPLE, "#555555", "#1b9e77", "#d95f02"]
for (cid, bt), col in zip(bts.items(), palette):
    label = (cid.replace("breakout_egarch_", "").replace("session_reversion_", "sr:")
              .replace("_L", " L=").replace("_Q", " Q=")
              .replace("_lookback_days", " lb=").replace("_z_entry", " z="))
    ax.plot(bt.index, bt["equity"], lw=1.2, label=label, color=col)
ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
ax.set_ylabel("Equity (start = 1.0)")
ax.set_title(f"Full-history equity — the {len(bts)} instances that survived Layers 1–3\n(all die in Layer 4)", fontsize=10, loc="left")
ax.legend(fontsize=6.0, ncol=2, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("results/figures/survivors123_equity.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: L1 pass rate by family/symbol
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.3))
fam = df_inst.groupby("family")["l1_pass"].mean() * 100
axes[0].bar(fam.index, fam.values, color=[BLUE, GOLD])
axes[0].set_ylabel("% passing Layer 1")
axes[0].set_title("By root idea", fontsize=8.5, loc="left")
axes[0].tick_params(axis="x", labelsize=7)

brk_single = df_inst[df_inst.family == "breakout_egarch"]
sym1 = brk_single.groupby("assets")["l1_pass"].mean() * 100
axes[1].bar(sym1.index, sym1.values, color=NAVY)
axes[1].set_title("breakout_egarch by symbol", fontsize=8.5, loc="left")
axes[1].tick_params(axis="x", labelsize=7, rotation=20)

sr_single = df_inst[df_inst.family == "session_reversion"]
sym2 = sr_single.groupby("assets")["l1_pass"].mean() * 100
axes[2].bar(sym2.index, sym2.values, color=RED)
axes[2].set_title("session_reversion by symbol", fontsize=8.5, loc="left")
axes[2].tick_params(axis="x", labelsize=7, rotation=20)
fig.tight_layout()
fig.savefig("results/figures/l1_breakdown.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 5: walk-forward parameter instability
wf = pd.read_csv("results/walkforward_groups.csv")
wf_xau = wf[(wf.family == "breakout_egarch") & (wf.assets == "XAUUSD")].copy()
wf_xau["test_start"] = pd.to_datetime(wf_xau["test_start"])
fig, ax = plt.subplots(figsize=(6.3, 2.2))
labels_seen = sorted(wf_xau["chosen_params"].unique())
y_map = {lab: i for i, lab in enumerate(labels_seen)}
ax.scatter(wf_xau["test_start"], wf_xau["chosen_params"].map(y_map), color=GOLD, s=60, zorder=3)
ax.plot(wf_xau["test_start"], wf_xau["chosen_params"].map(y_map), color=GOLD, lw=1, alpha=0.5, zorder=2)
ax.set_yticks(range(len(labels_seen)))
ax.set_yticklabels([l.replace("'", "").replace("{", "").replace("}", "") for l in labels_seen], fontsize=6.5)
ax.set_title("XAUUSD breakout_egarch: re-optimized winner, window by window", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig("results/figures/walkforward_instability.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure 6: parameter heatmaps (L1 OOS Sharpe)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
brk = df_inst[df_inst.family == "breakout_egarch"].copy()
brk["L"] = brk["params"].str.extract(r"'L': (\d+)").astype(int)
brk["Q"] = brk["params"].str.extract(r"'Q': ([\d.]+)").astype(float)
piv = brk[brk.assets == "XAUUSD"].groupby(["L", "Q"])["l1_oos_sharpe"].mean().unstack()
im0 = axes[0].imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1.5)
axes[0].set_xticks(range(len(piv.columns))); axes[0].set_xticklabels(piv.columns)
axes[0].set_yticks(range(len(piv.index))); axes[0].set_yticklabels(piv.index)
axes[0].set_xlabel("Q"); axes[0].set_ylabel("L")
axes[0].set_title("breakout_egarch / XAUUSD: OoS Sharpe", fontsize=8.5)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im0, ax=axes[0], shrink=0.8)

sr = df_inst[df_inst.family == "session_reversion"].copy()
sr["lookback_days"] = sr["params"].str.extract(r"'lookback_days': (\d+)").astype(int)
sr["z_entry"] = sr["params"].str.extract(r"'z_entry': ([\d.]+)").astype(float)
piv2 = sr[sr.assets == "SPXUSD"].groupby(["lookback_days", "z_entry"])["l1_oos_sharpe"].mean().unstack()
im1 = axes[1].imshow(piv2.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1.5)
axes[1].set_xticks(range(len(piv2.columns))); axes[1].set_xticklabels(piv2.columns)
axes[1].set_yticks(range(len(piv2.index))); axes[1].set_yticklabels(piv2.index)
axes[1].set_xlabel("z entry"); axes[1].set_ylabel("lookback days")
axes[1].set_title("session_reversion / SPXUSD: OoS Sharpe", fontsize=8.5)
for i in range(piv2.shape[0]):
    for j in range(piv2.shape[1]):
        v = piv2.values[i, j]
        if not np.isnan(v):
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im1, ax=axes[1], shrink=0.8)
fig.tight_layout()
fig.savefig("results/figures/param_heatmaps.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print("Figures 1-6 written to results/figures/")

# ---------------------------------------------------------------- Figure 7: regime contrast (efficiency ratio)
rg = pd.read_csv("results/regime_performance.csv")
er = rg[rg.regime_type == "efficiency_ratio"].copy()
er["mean_bps"] = er["mean_bar_return"] * 1e4
breakout_ids = [i for i in er.instance_id.unique() if i.startswith("breakout_egarch")]
session_ids = [i for i in er.instance_id.unique() if i.startswith("session_reversion")]
breakout_er = er[er.instance_id.isin(breakout_ids)].groupby("regime_label")["mean_bps"].mean()
session_er = er[er.instance_id.isin(session_ids)].groupby("regime_label")["mean_bps"].mean()
order = ["Choppy", "Neutral", "Trending"]
breakout_er = breakout_er.reindex(order)
session_er = session_er.reindex(order)

fig, ax = plt.subplots(figsize=(6.3, 2.3))
x = np.arange(len(order))
w = 0.35
ax.bar(x - w/2, breakout_er.values, width=w, label=f"breakout_egarch survivors (mean of {len(breakout_ids)})", color=GOLD)
ax.bar(x + w/2, session_er.values, width=w, label=f"session_reversion survivors (mean of {len(session_ids)})", color=BLUE)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("Mean bar return (bps)")
ax.set_title("Opposite mechanisms, opposite regimes: efficiency-ratio breakdown", fontsize=9.5, loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("results/figures/regime_contrast.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print("Figure 7 (regime contrast) written too.")
