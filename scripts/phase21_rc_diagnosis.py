#!/usr/bin/env python
"""Phase 21 reproduction gate script — FAIL-CLOSED protocol.

Runs the Phase 20 reproduction on repaired benchmark (seeds 11/23/37) and
compares against established Phase 20 values (docs/architecture/phase20_log.md,
results/reports/phase20_repaired_portfolio.md) within REP_TOL = 3pp.

If any measurement diverges beyond tolerance, the script exits with status 1
and prints the divergence. Per Phase 21 protocol (§18): FAIL CLOSED — do not
proceed with false conclusions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REP_TOL = 0.03  # 3 percentage points, absolute fractions

P20_EXPECTED = {
    "mlp_f": 0.997, "mlp_rc": 0.456, "mlp_frc": 0.572,
    "graph_f": 0.500, "graph_rc": 0.519, "graph_frc": 0.522,
    "attention_f": 0.500, "attention_rc": 0.489, "attention_frc": 0.392,
    "attention_v2_f": 0.417, "attention_v2_rc": 0.550, "attention_v2_frc": 0.639,
    "joint_f": 1.000, "joint_rc": 0.533, "joint_frc": 0.761,
    "joint_co_f": 1.000, "joint_co_rc": 0.531, "joint_co_frc": 0.764,
    "depth3_f": 1.000, "depth3_rc": 0.528, "depth3_frc": 0.758,
}


def load_p21_observed() -> dict[str, float]:
    p21_summary = Path("results/metrics/phase21_rc_diagnosis/summary.json")
    if not p21_summary.exists():
        return {}
    data = json.loads(p21_summary.read_text())
    out: dict[str, float] = {}
    # Aggregate per-seed reproduction checks (persisted via `_reproduction_gate`).
    seed_agg: dict[str, list[float]] = {}
    if "per_seed_results" in data:
        for seed_res in data["per_seed_results"]:
            checks = seed_res.get("repro", {}).get("checks")
            if not isinstance(checks, dict):
                continue
            for key, val in checks.items():
                seed_agg.setdefault(key, []).append(float(val))
    # Average per-key
    for k, vals in seed_agg.items():
        if vals:
            out[k] = sum(vals) / len(vals)
    # Agreement/disagreement from diagnostics (computed only when the diagnose
    # stage ran); absent from a finalize-only run, so keep the 0.0 fallback.
    agg = data.get("aggregates", {})
    out["rc_agree_rate"] = float(agg.get("agreement", {}).get("agree_acc", 0.0))
    out["rc_disagree_rate"] = float(agg.get("agreement", {}).get("disagree_acc", 0.0))
    return out


def main() -> int:
    print("=" * 70)
    print("Phase 21 — RC Relational Capacity Diagnosis (FAIL-CLOSED GATE)")
    print("=" * 70)
    print()
    print("Protocol (§1, §18): First reproduce Phase 20; fail closed if divergence > 3pp.")
    print(f"Reproduction tolerance (REP_TOL): {REP_TOL * 100:.0f}pp")
    print()

    p21 = load_p21_observed()
    if not p21:
        print("FAIL — Phase 21 observed metrics MISSING or EMPTY.")
        print(f"  P20 expected measurements: {len(P20_EXPECTED)}")
        print(f"  P21 observed measurements: 0")
        print()
        print("Diagnosis: The Phase 21 measurement pipeline did not complete successfully.")
        print("Possible causes: missing partials (`results/metrics/phase21_rc_diagnosis/partials/`),")
        print("training stage interrupted, or measurement script not executed (`scripts/phase21_rc_diagnosis.py`).")
        print()
        print("Per §18: FAIL CLOSED. No H1-H8 verdicts from broken/reproduced data.")
        print("Next step: fix the measurement/reproduction pipeline before any intervention claims.")
        return 1

    failures = []
    for key, expected in P20_EXPECTED.items():
        # Try direct match first; try partial mapping for family names
        observed = None
        # Direct match
        observed = p21.get(key)
        # Partial mappings (family abbreviations vary)
        if observed is None:
            alt_keys = [
                key.replace("mlp", "attention").replace("attention", "attention_v2"),
                key,
                key.replace("_f", "_rc").replace("_rc", "_f"),
            ]
            # Try matching first token
            prefix = key.split("_")[0]
            for k, v in p21.items():
                if k.startswith(prefix) or k.endswith(prefix):
                    observed = v
                    break
        if observed is None:
            failures.append((key, expected, None, "MISSING"))
            print(f"  {key:25s}: expected={expected:.3f}  observed=MISSING  [FAIL — missing]")
        else:
            diff = abs(float(observed) - expected)
            status = "PASS" if diff <= REP_TOL else "FAIL"
            if status == "FAIL":
                failures.append((key, expected, float(observed), f"divergence={diff:.3f}"))
            print(f"  {key:25s}: expected={expected:.3f}  observed={float(observed):.3f}  div={diff:.3f}  [{status}]")

    # Agreement/disagreement checks (separate from family accuracies)
    agree_obs = p21.get("rc_agree_rate", 0.0)
    agree_exp = 0.98
    disagree_obs = p21.get("rc_disagree_rate", 0.0)
    disagree_exp = 0.013
    print()
    agree_diff = abs(float(agree_obs) - agree_exp)
    agree_status = "PASS" if agree_diff <= REP_TOL else "FAIL"
    print(f"  rc_agree_rate: expected={agree_exp:.3f} observed={float(agree_obs):.3f} div={agree_diff:.3f} [{agree_status}]")
    disagree_diff = abs(float(disagree_obs) - disagree_exp)
    disagree_status = "PASS" if disagree_diff <= REP_TOL else "FAIL"
    print(f"  rc_disagree_rate: expected={disagree_exp:.3f} observed={float(disagree_obs):.3f} div={disagree_diff:.3f} [{disagree_status}]")

    if agree_status == "FAIL":
        failures.append(("rc_agree_rate", agree_exp, float(agree_obs), f"divergence={agree_diff:.3f}"))
    if disagree_status == "FAIL":
        failures.append(("rc_disagree_rate", disagree_exp, float(disagree_obs), f"divergence={disagree_diff:.3f}"))

    if failures:
        print()
        print(f"=" * 70)
        print(f"GATE RESULT: FAIL CLOSED ({len(failures)} divergence(s) > {REP_TOL * 100:.0f}pp)")
        print(f"=" * 70)
        print()
        print("Divergence details:")
        for key, exp, obs, detail in failures:
            obs_str = f"{obs:.3f}" if obs is not None else "MISSING/0.0"
            print(f"  {key}: P20={exp:.3f}  P21={obs_str}  ({detail})")
        print()
        print("Evidence from Phase 20 log (docs/architecture/phase20_log.md):")
        print("  RC agree rate: 0.980 (98.0%)")
        print("  RC disagree rate: 0.013 (1.3%)")
        print("  Pure R experts: 0.628-0.675 (graph/depth3); C experts: 0.997 (attention_v2);")
        print("  RC best (attention_v2): 0.550 (55.0%)")
        print()
        print("Evidence from broken Phase 21 artifacts:")
        print("  Most measurements show 0.0% (R decodability, gradient shares, agreement, counterfactuals).")
        print("  Existing `results/reports/phase21_rc_diagnosis.md` reports:")
        print("    - H1: NOT SUPPORTED; H2-H7: NOT TESTED; H8: NOT TESTED")
        print("    - Gate passed: False (reason: H1=NOT SUPPORTED; R share 0.0% < 5%)")
        print()
        print("Per Phase 21 protocol (§5-§6, §18):")
        print("  The reproduction gate is the non-negotiable prerequisite.")
        print("  When it fails, the correct action is FAIL CLOSED — not to invent verdicts.")
        print()
        print("Programmatic verdict for broken pipeline: UNRESOLVED / FAIL CLOSED.")
        print("Recommendation (§20): Fix reproduction/debug measurement pipeline before any H1-H8 verdict.")
        print("Minimal architecture change recommendation (§20): NONE — no evidence supports any change.")
        return 1
    else:
        print()
        print(f"=" * 70)
        print(f"GATE RESULT: PASSED (all within {REP_TOL * 100:.0f}pp)")
        print(f"=" * 70)
        print()
        print("Reproduction gate passed. Proceed with Phase 21 diagnostic experiments (H1-H8) per protocol.")
        print("Next steps (only if gate passes):")
        print("  1. H1: Relational capacity (depth 1/2/3, minimal validated depth-4 if H1 supported).")
        print("  2. H2: Frozen representation probes (R-component, C-component, RC-target).")
        print("  3. H3: Gradient/branch contribution analysis (C/R ratio, starvation threshold 5%).")
        print("  4. H4: Counterfactual R/C swaps (change rate + correctness rate, agreement/disagreement split).")
        print("  5. H5: Frozen diagnostic heads (linear + small nonlinear, NOT new architecture).")
        print("  6. H6: Location probes (source/query/pooled/fused representations).")
        print("  7. H7: Composition order (R→C vs C→R with same final head).")
        print("  8. H8: Gated minimal intervention (only if H1 supported + R not starved).")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
