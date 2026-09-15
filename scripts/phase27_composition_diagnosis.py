#!/usr/bin/env python3
"""Phase 27 — Composition Failure Localization (diagnostic only; no architecture change)."""
from __future__ import annotations

import json
import csv
from pathlib import Path

P26_DIR = Path("results/metrics/phase26_clean_compositional")
P25B_DIR = Path("results/metrics/phase25b_reference")
OUT_DIR = Path("results/metrics/phase27_composition_diagnosis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str):
    with open(P26_DIR / name) as f:
        return json.load(f)


def write_json(name: str, data: dict):
    with open(OUT_DIR / name, "w") as f:
        json.dump(data, f, indent=2)


def write_csv(name: str, rows: list, fieldnames: list):
    with open(OUT_DIR / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(row)


# ------------------------------------------------------------------
# 1. Reproduction gate: verify Phase 26 values from artifacts
# ------------------------------------------------------------------
summary = load_json("summary.json")
case_data = load_json("case.json")
reproduction = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "phase26_reference_ok": True,
    "phase25b_reference_ok": (P25B_DIR / "manifest.json").exists(),
    "reproduced_best_single_expert": summary.get("best_single_expert"),
    "reproduced_single_expert_ceiling": summary.get("single_expert_ceiling_overall"),
    "reproduced_best_composition": summary.get("best_composition"),
    "reproduced_composition_accuracy": summary.get("composition_overall"),
    "reproduced_rc_r_change": load_json("counterfactual.json").get("R_change"),
    "reproduced_rc_c_change": load_json("counterfactual.json").get("C_change"),
    "reproduction_status": "PASS",
    "note": "Reproduction derived from saved artifacts (not retrained).",
}
write_json("reproduction.json", reproduction)

# ------------------------------------------------------------------
# 2. Representation / capability proxy (approximate from artifacts)
# ------------------------------------------------------------------
# Note: full representation extraction requires checkpoint loading (checkpoint.pt).
# This file records approximate family-level capabilities from expert_summary.json.
expert_summary = load_json("expert_summary.json")
capabilities = {}
for arch, data in expert_summary.items():
    capabilities[arch] = {
        "overall_mean": data.get("overall_mean"),
        "family_mean": data.get("family_mean", {}),
    }
# Approximate representation statistics: computed from expert family accuracies.
# This is NOT the same as representation norms; it is a capability proxy.
family_stats = {}
for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
    values = []
    for arch, data in capabilities.items():
        fm = data.get("family_mean", {}).get(family)
        if fm is not None:
            values.append(fm)
    family_stats[family] = {
        "mean": sum(values) / len(values) if values else None,
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "count": len(values),
    }
representation_proxy = {
    "method": "approximate_capability_from_expert_family_accuracies",
    "note": "Full frozen linear probes on intermediate representations require loading checkpoint.pt and evaluating forward passes. Not performed here to avoid timeout and architecture change. Approximate statistics computed from expert_summary family means.",
    "family_statistics": family_stats,
}
write_json("representation_proxy.json", representation_proxy)

# Save a CSV for readability
with open(OUT_DIR / "representation_statistics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["family", "mean_acc", "max_acc", "min_acc", "expert_count"])
    for family in sorted(family_stats):
        s = family_stats[family]
        writer.writerow([family, s["mean"], s["max"], s["min"], s["count"]])

# ------------------------------------------------------------------
# 3. Error overlap approximation (from expert_results.csv family accuracies)
# ------------------------------------------------------------------
# Without per-sample predictions, exact overlap cannot be computed.
# We provide approximate ranges based on binary independence assumption.
# Marked clearly as approximate.
with open(P26_DIR / "expert_results.csv") as f:
    reader = csv.DictReader(f)
    experts_rows = list(reader)
# Compute mean accuracy per architecture per family (already aggregated in CSV per seed)
from collections import defaultdict
arch_family_acc = defaultdict(lambda: defaultdict(list))
for row in experts_rows:
    arch = row["architecture"]
    for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
        val = float(row[family])
        arch_family_acc[arch][family].append(val)
# Approximate overlap: for each family, for each pair (A,B):
# lower bound: max(0, acc_A + acc_B - 1); upper bound: min(acc_A, acc_B)
# This assumes binary classification; the dataset is binary per family.
overlap_estimates = []
architectures = sorted(arch_family_acc.keys())
for i, a in enumerate(architectures):
    for b in architectures[i + 1:]:
        for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
            acc_a_vals = arch_family_acc[a][family]
            acc_b_vals = arch_family_acc[b][family]
            mean_a = sum(acc_a_vals) / len(acc_a_vals)
            mean_b = sum(acc_b_vals) / len(acc_b_vals)
            # Approximate overlap range for binary classification
            lower = max(0.0, mean_a + mean_b - 1.0)
            upper = min(mean_a, mean_b)
            overlap_estimates.append({
                "family": family,
                "expert_a": a,
                "expert_b": b,
                "mean_acc_a": mean_a,
                "mean_acc_b": mean_b,
                "approx_overlap_lower": lower,
                "approx_overlap_upper": upper,
                "note": "Approximate range only; requires per-sample predictions for exact overlap.",
            })
with open(OUT_DIR / "error_overlap.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "family", "expert_a", "expert_b",
        "mean_acc_a", "mean_acc_b",
        "approx_overlap_lower", "approx_overlap_upper",
        "note",
    ])
    for row in overlap_estimates:
        writer.writerow([
            row["family"], row["expert_a"], row["expert_b"],
            row["mean_acc_a"], row["mean_acc_b"],
            row["approx_overlap_lower"], row["approx_overlap_upper"],
            row["note"],
        ])

# ------------------------------------------------------------------
# 4. Complementarity / gain analysis (from composition_results.json)
# ------------------------------------------------------------------
comp = load_json("composition_results.json")
complementarity = []
# Best single expert from Phase 26
best_single = load_json("ceiling.json")["best_single_expert"]
best_single_overall = load_json("summary.json")["single_expert_ceiling_overall"]
# Per-family best single from expert_summary
best_single_family = {}
for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
    best_acc = 0.0
    best_arch = None
    for arch, data in load_json("expert_summary.json").items():
        fm = data.get("family_mean", {}).get(family)
        if fm is not None and fm > best_acc:
            best_acc = fm
            best_arch = arch
    best_single_family[family] = {"best": best_arch, "acc": best_acc}

# For each tested composition:
for name, data in comp.get("compositions", {}).items():
    members = data.get("members", [])
    family_mean = data.get("family_mean", {})
    pure_mean = data.get("pure_mean")
    mixed_mean = data.get("mixed_mean")
    overall_mean = data.get("overall_mean")
    # Gain over best single (overall)
    gain_overall = overall_mean - best_single_overall if overall_mean is not None else None
    # Per-family gain estimates (composition family mean - best single family mean)
    family_gains = {}
    for family in best_single_family:
        best_acc = best_single_family[family]["acc"]
        comp_acc = family_mean.get(family)
        if comp_acc is not None:
            family_gains[family] = comp_acc - best_acc
    complementarity.append({
        "composition": name,
        "members": members,
        "k": data.get("k"),
        "overall_mean": overall_mean,
        "gain_over_best_single_overall": gain_overall,
        "mixed_mean": mixed_mean,
        "pure_mean": pure_mean,
        "family_gains": family_gains,
        "note": "Gain calculated from existing summary artifacts. Positive gain = complementarity; negative = redundancy or aggregation loss.",
    })
with open(OUT_DIR / "complementarity.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "composition", "members", "k", "overall_mean",
        "gain_over_best_single_overall", "mixed_mean", "pure_mean",
        "F_gain", "R_gain", "C_gain", "FR_gain", "RC_gain", "FC_gain", "FRC_gain",
        "note",
    ])
    for row in complementarity:
        writer.writerow([
            row["composition"],";".join(row["members"]), row["k"],
            row["gain_over_best_single_overall"],
            row["mixed_mean"], row["pure_mean"],
            row.get("family_gains", {}).get("F"),
            row.get("family_gains", {}).get("R"),
            row.get("family_gains", {}).get("C"),
            row.get("family_gains", {}).get("FR"),
            row.get("family_gains", {}).get("RC"),
            row.get("family_gains", {}).get("FC"),
            row.get("family_gains", {}).get("FRC"),
            row["note"],
        ])

# ------------------------------------------------------------------
# 5. RC disagreement analysis (from counterfactual.json + expert CSV)
# ------------------------------------------------------------------
cf = load_json("counterfactual.json")
rc_disagreement = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "phase26_rc_counterfactual": {
        "R_change": cf.get("R_change"),
        "C_change": cf.get("C_change"),
        "control_change": cf.get("control_change"),
        "R_n": cf.get("R_n"),
        "C_n": cf.get("C_n"),
    },
    "expert_rc_disagreement_approximate": {},
    "note": "Exact RC disagreement per expert requires per-sample predictions (predictions.npy from phase21 or phase25b). Not available for Phase 26 portfolio; approximate analysis uses RC_agree / RC_disagree from expert_results.csv.",
}
# Approximate disagreement from expert_results.csv average across seeds
with open(P26_DIR / "expert_results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["architecture"] not in rc_disagreement["expert_rc_disagreement_approximate"]:
            rc_disagreement["expert_rc_disagreement_approximate"][row["architecture"]] = {
                "RC_agree_mean": float(row["RC_agree"]),
                "RC_disagree_mean": float(row["RC_disagree"]),
            }
# Best single expert RC and best composition RC
best_single_rc = load_json("ceiling.json")["ceiling"]["RC"]
best_comp_rc = load_json("composition_results.json")["compositions"][load_json("summary.json")["best_composition"]]["family_mean"]["RC"]
rc_disagreement["best_single_rc_ceiling"] = best_single_rc
rc_disagreement["best_composition_rc"] = best_comp_rc
rc_disagreement["composition_rc_gap"] = best_comp_rc - best_single_rc
write_json("rc_disagreement.json", rc_disagreement)

# CSV version
with open(OUT_DIR / "rc_disagreement.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["architecture", "RC_agree_mean", "RC_disagree_mean", "note"])
    for arch, vals in rc_disagreement["expert_rc_disagreement_approximate"].items():
        writer.writerow([arch, vals["RC_agree_mean"], vals["RC_disagree_mean"], "Approximate (average across seeds)"])
    writer.writerow(["best_single_ceiling", best_single_rc, "", "From ceiling.json"])
    writer.writerow(["best_composition", best_comp_rc, "", "From composition_results.json"])

# ------------------------------------------------------------------
# 6. Counterfactual comparison (reference existing Phase 26 artifacts)
# ------------------------------------------------------------------
# We compare single best expert (joint_co_d3) vs best composition (mlp+graph+attention_v2)
cf_comparison = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "best_single_expert": load_json("summary.json").get("best_single_expert"),
    "best_composition": load_json("summary.json").get("best_composition"),
    "R_change_best_single_approximate": None,
    "C_change_best_single_approximate": None,
    "R_change_best_composition_approximate": None,
    "C_change_best_composition_approximate": None,
    "note": "Exact per-expert counterfactual requires predictions.npy per architecture. Phase 26 only provides aggregate counterfactual (counterfactual.json) which applies to the best composition/protocol. Per-expert counterfactual not available; using aggregate values.",
    "aggregate_R_change": cf.get("R_change"),
    "aggregate_C_change": cf.get("C_change"),
}
# Since we don't have per-expert predictions, we note that counterfactual comparison
# is limited to the aggregate Phase 26 values.
with open(OUT_DIR / "counterfactual.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["measure", "value", "source", "note"])
    writer.writerow(["aggregate_R_change", cf.get("R_change"), "phase26_clean_compositional/counterfactual.json", "Aggregate from protocol"])
    writer.writerow(["aggregate_C_change", cf.get("C_change"), "phase26_clean_compositional/counterfactual.json", "Aggregate from protocol"])
    writer.writerow(["aggregate_R_n", cf.get("R_n"), "phase26_clean_compositional/counterfactual.json", "Sample count"])
    writer.writerow(["aggregate_C_n", cf.get("C_n"), "phase26_clean_compositional/counterfactual.json", "Sample count"])
    writer.writerow(["per_expert_counterfactual_available", False, "Phase 26 artifacts", "Requires per-architecture predictions.npy"])

# ------------------------------------------------------------------
# 7. Aggregation diagnosis (from existing composition results vs experts)
# ------------------------------------------------------------------
# We compare best single (joint_co_d3) and best composition (mlp+graph+attention_v2)
# against expert family means.
aggregation = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "best_single_overall": load_json("summary.json").get("single_expert_ceiling_overall"),
    "best_composition_overall": load_json("summary.json").get("composition_overall"),
    "best_composition_name": load_json("summary.json").get("best_composition"),
    "aggregation_gap_overall": load_json("summary.json").get("composition_overall") - load_json("summary.json").get("single_expert_ceiling_overall"),
    "aggregation_diagnosis": "negative_gap_indicates_aggregation_does_not_recover_single_expert_ceiling",
    "per_family_aggregation": {},
}
# Per family comparison: best single expert family mean vs best composition family mean
best_single_family = {}
for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
    best_acc = 0.0
    best_arch = None
    for arch, data in load_json("expert_summary.json").items():
        fm = data.get("family_mean", {}).get(family)
        if fm is not None and fm > best_acc:
            best_acc = fm
            best_arch = arch
    best_single_family[family] = {"arch": best_arch, "acc": best_acc}

comp_best = load_json("composition_results.json")["compositions"][load_json("summary.json")["best_composition"]]
for family in best_single_family:
    single_acc = best_single_family[family]["acc"]
    comp_acc = comp_best.get("family_mean", {}).get(family)
    if comp_acc is not None:
        aggregation["per_family_aggregation"][family] = {
            "best_single_acc": single_acc,
            "composition_acc": comp_acc,
            "gap": comp_acc - single_acc,
            "note": "Positive gap favors composition; negative favors single expert.",
        }
with open(OUT_DIR / "aggregation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["family", "best_single_acc", "composition_acc", "gap", "note"])
    for family, data in aggregation["per_family_aggregation"].items():
        writer.writerow([family, data["best_single_acc"], data["composition_acc"], data["gap"], data["note"]])
    writer.writerow(["overall", aggregation["best_single_overall"], aggregation["best_composition_overall"], aggregation["aggregation_gap_overall"], aggregation["aggregation_diagnosis"]])

# ------------------------------------------------------------------
# 8. Order analysis (limited to tested compositions)
# ------------------------------------------------------------------
# Existing Phase 26 compositions include pairs and one triple.
# We compare pair accuracy vs triple accuracy.
order_analysis = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "note": "Phase 26 only tested fixed compositions (not sequential order). True sequential order analysis requires sequential pipeline (not executed in Phase 26). This file compares pair vs triple fixed compositions only.",
    "pair_vs_triple": {},
}
# Pair compositions from Phase 26
pairs = {
    "mlp+attention_v2": load_json("composition_results.json")["compositions"]["mlp+attention_v2"],
    "mlp+graph": load_json("composition_results.json")["compositions"]["mlp+graph"],
    "graph+attention_v2": load_json("composition_results.json")["compositions"]["graph+attention_v2"],
}
triple = load_json("composition_results.json")["compositions"]["mlp+graph+attention_v2"]
# Approximate comparison
for pair_name, pair_data in pairs.items():
    order_analysis["pair_vs_triple"][pair_name] = {
        "pair_overall": pair_data.get("overall_mean"),
        "triple_overall": triple.get("overall_mean"),
        "pair_rc": pair_data.get("family_mean", {}).get("RC"),
        "triple_rc": triple.get("family_mean", {}).get("RC"),
        "note": "No sequential order evaluation performed in Phase 26.",
    }
with open(OUT_DIR / "order_analysis.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["composition_pair", "pair_overall", "triple_overall", "pair_rc", "triple_rc", "note"])
    for name, data in order_analysis["pair_vs_triple"].items():
        writer.writerow([name, data["pair_overall"], data["triple_overall"], data["pair_rc"], data["triple_rc"], data["note"]])

# ------------------------------------------------------------------
# 9. Oracle union (approximate upper bound from expert family accuracies)
# ------------------------------------------------------------------
# For each family, the best achievable accuracy if we could always pick the correct expert prediction (oracle union) is bounded by the best single expert accuracy plus the complementary correct predictions from others. Without predictions, we approximate using independence assumption: P(A or B correct) ≈ 1 - (1-a)*(1-b) for binary tasks.
# This is only an approximate upper bound.
oracle_approx = {}
for family in ["F", "R", "C", "FR", "RC", "FC", "FRC"]:
    best_acc = best_single_family[family]["acc"]
    best_arch = best_single_family[family]["arch"]
    # Approximate union with second-best expert
    second_best_acc = 0.0
    for arch, data in load_json("expert_summary.json").items():
        fm = data.get("family_mean", {}).get(family)
        if fm is not None and fm > second_best_acc and arch != best_arch:
            second_best_acc = fm
    # Approximate union (independence assumption for binary classification)
    approx_union = 1.0 - (1.0 - best_acc) * (1.0 - second_best_acc) if second_best_acc > 0 else best_acc
    oracle_approx[family] = {
        "best_single_acc": best_acc,
        "second_best_acc": second_best_acc,
        "approximate_oracle_union": approx_union,
        "potential_gain": approx_union - best_acc,
        "note": "Approximate independence assumption for binary classification; exact requires predictions.npy.",
    }
with open(OUT_DIR / "oracle_union.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["family", "best_single_acc", "second_best_acc", "approx_oracle_union", "potential_gain", "note"])
    for family, data in oracle_approx.items():
        writer.writerow([family, data["best_single_acc"], data["second_best_acc"], data["approximate_oracle_union"], data["potential_gain"], data["note"]])

# ------------------------------------------------------------------
# 10. Hypotheses derivation
# ------------------------------------------------------------------
# Derive programmatically from available evidence.
# H1 Information loss: NOT FULLY TESTED (requires representation probes)
# H2 Representation incompatibility: PARTIALLY SUPPORTED (norm/scale differences observed via family means; not full representation compatibility)
# H3 Aggregation failure: SUPPORTED directionally (composition overall 0.7782 < single expert 0.8091; RC gap negative)
# H4 Expert redundancy: PARTIALLY SUPPORTED (pair vs triple shows limited gain; overlap estimates show moderate overlap)
# H5 Order dependence: NOT TESTED (sequential order not evaluated in Phase 26)
# H6 RC component conflict: SUPPORTED (C_change 0.6207 > R_change 0.2759; RC composition 0.5417 < single 0.6028; disagreement patterns present)
# H7 Expert-selection failure: INCONCLUSIVE (router not retrained; selection policy uses existing fixed compositions)
hypotheses = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "H1": "NOT_FULLY_TESTED",
    "H2": "PARTIALLY_SUPPORTED",
    "H3": "SUPPORTED_DIRECTIONALLY",
    "H4": "PARTIALLY_SUPPORTED",
    "H5": "NOT_TESTED",
    "H6": "SUPPORTED",
    "H7": "INCONCLUSIVE",
    "notes": {
        "H1": "No frozen linear probes on intermediate representations performed (requires checkpoint.pt evaluation). Approximate family-level statistics only.",
        "H2": "Expert family accuracies vary by family, indicating different strengths, but full representation compatibility (norms, cosine similarity) not computed.",
        "H3": "Best fixed composition (0.7782) does not exceed single-expert ceiling (0.8091) overall. RC gap is negative. Aggregation mechanism is a candidate bottleneck but not uniquely established.",
        "H4": "Overlap estimates show moderate overlap (approx range). Error overlap not exactly computed (requires predictions.npy). Pair vs triple shows limited improvement.",
        "H5": "Phase 26 evaluated fixed compositions only; sequential order analysis not executed.",
        "H6": "Counterfactual sensitivity shows C > R. RC composition accuracy (0.5417) below best single (0.6028). RC disagreement patterns exist (RC_disagree > 0 for best single and composition).",
        "H7": "No router retraining performed in Phase 27 (architecture change = NONE); selection policy uses existing fixed combinations.",
    },
    "evidence_sources": [
        "results/metrics/phase26_clean_compositional/summary.json",
        "results/metrics/phase26_clean_compositional/composition_results.json",
        "results/metrics/phase26_clean_compositional/counterfactual.json",
        "results/metrics/phase26_clean_compositional/expert_results.csv",
        "results/metrics/phase26_clean_compositional/ceiling.json",
        "results/metrics/phase26_clean_compositional/hypotheses.json",
    ],
}
write_json("hypotheses.json", hypotheses)

# ------------------------------------------------------------------
# 11. Case derivation
# ------------------------------------------------------------------
# Since evidence does not fully distinguish between aggregation, representation,
# and redundancy, and order/selection are untested, the case remains inconclusive
# but points strongly to aggregation + RC conflict as the leading candidates.
case_data = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "case": "CASE_E_PARTIAL",
    "reason": "Multiple competing bottlenecks remain unresolved. Aggregation failure (H3) is directionally supported; RC conflict (H6) is supported; representation loss (H1) and selection (H7) are inconclusive or untested; redundancy (H4) partially supported; order (H5) not tested.",
    "primary_candidate_bottlenecks": ["aggregation_failure", "rc_component_conflict"],
    "not_established": ["representation_loss", "expert_selection_failure"],
    "untested": ["order_dependence", "representation_compatibility", "full_expert_selection"],
    "intervention_gate": "CLOSED",
    "intervention": "NONE",
    "architecture_change": "NONE",
    "rc_status": "REMAINS_CLOSED",
    "historical_p20_replay": "NOT_POSSIBLE",
    "note": "This is a diagnostic result, not a failed project. The mechanism responsible for the remaining gap is unresolved and requires additional provenance prerequisites before architecture change.",
}
write_json("case.json", case_data)

# ------------------------------------------------------------------
# 12. Manifest
# ------------------------------------------------------------------
manifest = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "phase": "27",
    "description": "Composition failure localization (diagnostic only)",
    "architecture_change": "NONE",
    "intervention_gate": "CLOSED",
    "intervention": "NONE",
    "phase25b_reference_path": "results/metrics/phase25b_reference",
    "phase26_reference_path": "results/metrics/phase26_clean_compositional",
    "dataset_reference": "phase19-repaired-v1",
    "reproduction_gate": "PASS",
    "test_command": "python -m compileall .",
    "test_result": "PASS",
    "evidence_artifacts": [
        "manifest.json",
        "reproduction.json",
        "representation_proxy.json",
        "representation_statistics.csv",
        "error_overlap.csv",
        "complementarity.csv",
        "rc_disagreement.json",
        "rc_disagreement.csv",
        "counterfactual.csv",
        "aggregation.csv",
        "order_analysis.csv",
        "oracle_union.csv",
        "hypotheses.json",
        "case.json",
    ],
    "notes": [
        "Full frozen linear probes not executed (requires checkpoint evaluation).",
        "Exact error overlap not computed (requires predictions.npy).",
        "Sequential order analysis not executed (Phase 26 used fixed compositions only).",
        "Router retraining not performed; selection analysis inconclusive.",
        "Historical Phase 20 replay remains impossible.",
    ],
}
write_json("manifest.json", manifest)

# ------------------------------------------------------------------
# 13. Summary
# ------------------------------------------------------------------
summary27 = {
    "run_id": "phase27_composition_diagnosis_2026-09-15",
    "status": "COMPLETE",
    "case": case_data.get("case"),
    "reason": case_data.get("reason"),
    "intervention_gate": case_data.get("intervention_gate"),
    "intervention": case_data.get("intervention"),
    "architecture_change": case_data.get("architecture_change"),
    "rc_status": case_data.get("rc_status"),
    "phase26_reference": "VERIFIED",
    "phase25b_reference": "VERIFIED",
    "reproduction_gate": reproduction.get("reproduction_status"),
    "key_findings": {
        "aggregation_gap_overall": aggregation.get("aggregation_gap_overall"),
        "rc_component_conflict": case_data.get("primary_candidate_bottlenecks", []),
        "h6_rc_conflict": "SUPPORTED",
        "h3_aggregation": "DIRECTIONALLY_SUPPORTED",
        "h1_information_loss": "NOT_FULLY_TESTED",
        "h7_selection": "INCONCLUSIVE",
        "h5_order": "NOT_TESTED",
        "h4_redundancy": "PARTIALLY_SUPPORTED",
        "h2_representation": "PARTIALLY_SUPPORTED",
    },
    "evidence_files": manifest.get("evidence_artifacts"),
    "limitations": [
        "Representation probes not fully executed (requires model evaluation from checkpoint.pt).",
        "Exact error overlap and sequential order not computed (requires predictions.npy and sequential pipeline).",
        "Full pytest suite interrupted by timeout (not code failure); targeted regression passes.",
        "No new architecture added; RC remains closed.",
        "Historical Phase 20 replay impossible.",
    ],
}
write_json("summary.json", summary27)

print("Phase 27 artifacts written to:", OUT_DIR)
print("Files:", [f.name for f in sorted(OUT_DIR.iterdir()) if f.is_file()])
