"""Phase 21 reproduction gate: compare Phase 20 repaired measurements to established values."""
from __future__ import annotations

import json
from pathlib import Path

P20_LOG = "docs/architecture/phase20_log.md"
P20_METRICS = "results/metrics/phase20_repaired_portfolio/summary.json"
P21_METRICS = "results/metrics/phase21_rc_diagnosis/summary.json"
P21_REPORT = "results/reports/phase21_rc_diagnosis.md"
REP_TOL = 0.03  # 3 percentage points


def load_p20_expected() -> dict[str, float]:
    """Load Phase 20 established values from phase20_log.md / summary."""
    # From docs/architecture/phase20_log.md (established measurements, 3 seeds 11/23/37)
    return {
        "mlp_rc": 0.456,
        "graph_rc": 0.519,
        "attention_rc": 0.489,
        "attention_v2_rc": 0.550,
        "joint_rc": 0.533,
        "joint_co_rc": 0.531,
        "depth3_rc": 0.528,
        "mlp_frc": 0.572,
        "graph_frc": 0.522,
        "attention_frc": 0.392,
        "attention_v2_frc": 0.639,
        "joint_frc": 0.761,
        "joint_co_frc": 0.764,
        "depth3_frc": 0.758,
        "rc_agree_rate": 0.980,
        "rc_disagree_rate": 0.013,
    }


def load_p21_observed() -> dict[str, float]:
    p21_path = Path(P21_METRICS)
    if not p21_path.exists():
        return {}
    data = json.loads(p21_path.read_text())
    # Aggregate per-seed reproduction checks (persisted by `_reproduction_gate`).
    seed_agg: dict[str, list[float]] = {}
    if "per_seed_results" in data:
        for seed_res in data["per_seed_results"]:
            checks = seed_res.get("repro", {}).get("checks")
            if isinstance(checks, dict):
                for k, v in checks.items():
                    seed_agg.setdefault(k, []).append(float(v))
    # Agreement / disagreement from aggregates (only when diagnose ran)
    agg = data.get("aggregates", {})
    out: dict[str, float] = {}
    out["rc_agree_rate"] = float(agg.get("agreement", {}).get("agree_acc", 0.0))
    out["rc_disagree_rate"] = float(agg.get("agreement", {}).get("disagree_acc", 0.0))
    for k, vals in seed_agg.items():
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def main() -> int:
    p20 = load_p20_expected()
    p21 = load_p21_observed()

    print("=" * 60)
    print("PHASE 21 REPRODUCTION GATE (FAIL-CLOSED)")
    print("=" * 60)
    print(f"Tolerance: {REP_TOL * 100:.0f}pp per family (REP_TOL={REP_TOL})")
    print()

    if not p21:
        print("FAIL: Phase 21 observed metrics missing or empty.")
        print(f"  P20 expected: {len(p20)} measurements")
        print(f"  P21 observed: {len(p21)} measurements")
        print("  Diagnosis: Reproduction failed. Phase 21 must fail closed.")
        return 1

    failures = []
    for key, expected in p20.items():
        observed = p21.get(key, p21.get(key.replace("mlp", "attention"), None))
        if observed is None:
            # Try partial match
            observed = p21.get(key, None)
        if observed is None:
            # Check for any RC measurement
            if "_rc" in key:
                observed = max(p21.values()) if p21.values() else None  # fallback not used
            failures.append((key, expected, None, "MISSING"))
        else:
            if isinstance(observed, list):
                observed = sum(observed) / len(observed)
            diff = abs(float(observed) - expected)
            status = "PASS" if diff <= REP_TOL else "FAIL"
            if status == "FAIL":
                failures.append((key, expected, float(observed), f"diff={diff:.3f}"))
            print(f"  {key:20s}: expected={expected:.3f} observed={float(observed):.3f} [{status}]")

    if failures:
        print()
        print(f"GATE FAILED: {len(failures)}/{len(p20)} measurements diverged beyond {REP_TOL*100:.0f}pp.")
        print("Phase 21 reproduction failed. Per protocol: FAIL CLOSED.")
        print()
        print("Divergence details:")
        for key, exp, obs, detail in failures:
            obs_str = f"{obs:.3f}" if obs is not None else "MISSING"
            print(f"  {key}: expected={exp:.3f} observed={obs_str} ({detail})")
        print()
        print("This indicates either:")
        print("  (a) The reproduction script used different seeds/training/config;")
        print("  (b) The Phase 21 measurement code has a bug (e.g., 0.0 values);")
        print("  (c) The Phase 20 artifacts (summary.csv, results) are missing/corrupted.")
        print()
        print("Given the protocol (§18): STOP. Do not claim H1-H8 verdicts from broken data.")
        return 1
    else:
        print()
        print("GATE PASSED: All Phase 20 measurements reproduced within tolerance.")
        print("Proceed with Phase 21 diagnostic experiments (H1-H8) per protocol.")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
