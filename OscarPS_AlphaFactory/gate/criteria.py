"""
PRE-REGISTERED GATE CRITERIA (v2 -- breakout_egarch + relval design)
======================================================================
Loaded directly from gate/pre_registration.yaml, which is now the single
source of truth for every numeric threshold in the gate (candidate grids,
cost model, all 4 layers' pass criteria, the random seed, and the trial
count fed to the DSR). This module just exposes that file as importable
dicts so the rest of the codebase doesn't need to parse YAML itself.

Two deliberate deviations from the first draft of this project's plan,
decided BEFORE the gate was run (both are judgment calls made explicit
here, not post-hoc tuning):
  1. Layer 4's Deflated Sharpe Ratio pass criterion is set at >= 0.95 (95%
     confidence the deflation-adjusted Sharpe is still positive), not the
     literal ">0" in the original plan draft. DSR is a probability (a CDF
     value in (0,1)); ">0" is satisfied by nearly anything that isn't
     actively losing money and would make this layer almost vacuous.
  2. Layer 2 ("walk-forward re-optimization") creates a mapping problem
     (see pre_registration.yaml: layer2_walkforward.reoptimization_rule),
     resolved with a hybrid group-winner-or-own-params rule.

One disclosed post-hoc ADDITION (not a changed threshold -- see
pre_registration.yaml's change_log): a second, real (non-synthetic) Layer 3
stress window after the EGARCH warm-up period, since the COVID window is
confounded by breakout_egarch's warm-up requirement.
"""
from __future__ import annotations
import yaml
from pathlib import Path

_MANIFEST_PATH = Path(__file__).parent / "pre_registration.yaml"
with open(_MANIFEST_PATH) as f:
    MANIFEST = yaml.safe_load(f)

CHANGE_LOG = MANIFEST.get("change_log", [])
RANDOM_SEED = MANIFEST["random_seed"]
COST_MODEL = MANIFEST["cost_model"]
CANDIDATE_FAMILIES = MANIFEST["candidate_families"]

L1 = MANIFEST["layer1_is_oos"]
L2 = MANIFEST["layer2_walkforward"]
L3 = MANIFEST["layer3_stress"]
L4 = MANIFEST["layer4_montecarlo"]
COST_SENSITIVITY = MANIFEST["cost_sensitivity_diagnostic"]
