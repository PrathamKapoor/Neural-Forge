"""Evaluation metrics and Pareto frontier calculations for Phase 8B Compute-Aware Routing."""
from __future__ import annotations

from typing import Any

import torch


def compute_normalized_costs(
    expert_flops: dict[str, float],
    router_flops: float = 1052.0,
    reference_cost: float | None = None,
) -> tuple[dict[str, float], float]:
    """Compute constant normalized costs for all candidate experts including router overhead.

    Reference cost defaults to the maximum portfolio execution cost: max(expert_flops) + router_flops.
    """
    total_expert_costs = {name: fl + router_flops for name, fl in expert_flops.items()}
    if reference_cost is None:
        reference_cost = max(total_expert_costs.values())

    if reference_cost <= 0:
        raise ValueError("reference_cost must be strictly positive")

    normalized_costs = {name: cost / reference_cost for name, cost in total_expert_costs.items()}
    return normalized_costs, reference_cost


def calculate_pareto_frontier(
    eval_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate the Pareto frontier over evaluated points.

    Each point dict must contain:
      - "name": condition identifier
      - "accuracy": classification accuracy (higher is better)
      - "flops": theoretical forward FLOPs per sample (lower is better)

    Point A dominates Point B (A > B) iff:
      (A.accuracy >= B.accuracy and A.flops <= B.flops) and
      (A.accuracy > B.accuracy or A.flops < B.flops)
    """
    n = len(eval_points)
    is_non_dominated = [True] * n
    dominated_by: list[list[str]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            acc_i, fl_i = eval_points[i]["accuracy"], eval_points[i]["flops"]
            acc_j, fl_j = eval_points[j]["accuracy"], eval_points[j]["flops"]

            # Check if point j strictly dominates point i
            dominates = (acc_j >= acc_i and fl_j <= fl_i) and (acc_j > acc_i or fl_j < fl_i)
            if dominates:
                is_non_dominated[i] = False
                dominated_by[i].append(eval_points[j]["name"])

    pareto_records = []
    for i in range(n):
        rec = dict(eval_points[i])
        rec["is_non_dominated"] = is_non_dominated[i]
        rec["dominated_by"] = dominated_by[i]
        pareto_records.append(rec)

    non_dominated_names = [p["name"] for p in pareto_records if p["is_non_dominated"]]
    dominated_names = [p["name"] for p in pareto_records if not p["is_non_dominated"]]

    # Sort non-dominated frontier by FLOPs ascending
    frontier_sorted = sorted(
        [p for p in pareto_records if p["is_non_dominated"]],
        key=lambda x: x["flops"]
    )

    return {
        "non_dominated_names": non_dominated_names,
        "dominated_names": dominated_names,
        "pareto_records": pareto_records,
        "frontier_sorted": frontier_sorted,
    }


def check_cost_collapse(
    utilization: dict[str, float],
    selection_matrix: dict[str, dict[str, float]] | None = None,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Inspect for cost-aware routing collapse modes."""
    dominant_expert = max(utilization, key=utilization.get)  # type: ignore[arg-type]
    dominant_util = utilization[dominant_expert]

    # Mode 1: Cheapest-expert collapse (all samples route to Graph or MLP)
    is_cheapest_collapse = (
        dominant_util >= threshold and dominant_expert in {"graph", "mlp"}
    )

    # Mode 2: Expensive V2 avoidance
    v2_util = utilization.get("attention_v2", 0.0)
    is_v2_avoidance = v2_util < 0.05

    # Mode 3: Inefficient V1 avoidance
    v1_util = utilization.get("attention", 0.0)
    is_v1_avoidance = v1_util < 0.05

    # Mode 4: Useful selective V2
    is_selective_v2 = False
    if selection_matrix is not None:
        ctx_v2 = selection_matrix.get("contextual", {}).get("attention_v2", 0.0)
        feat_v2 = selection_matrix.get("feature", {}).get("attention_v2", 0.0)
        is_selective_v2 = ctx_v2 > 0.85 and feat_v2 < 0.15

    return {
        "cheapest_expert_collapse": is_cheapest_collapse,
        "v2_avoidance": is_v2_avoidance,
        "v1_avoidance": is_v1_avoidance,
        "useful_selective_v2": is_selective_v2,
        "dominant_expert": dominant_expert,
        "dominant_utilization": dominant_util,
    }


def calculate_family_conditional_cost(
    selected_choices: list[str],
    test_families: list[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    expert_flops: dict[str, float],
    router_flops: float = 1052.0,
) -> dict[str, dict[str, Any]]:
    """Calculate mean FLOPs, accuracy, and expert distribution partitioned by task family."""
    unique_families = ("feature", "relational", "contextual")
    result: dict[str, dict[str, Any]] = {}

    for fam in unique_families:
        indices = [i for i, f in enumerate(test_families) if f == fam]
        if not indices:
            continue

        n_fam = len(indices)
        fam_choices = [selected_choices[i] for i in indices]
        fam_correct = sum(int(predictions[i].item() == targets[i].item()) for i in indices)
        fam_acc = fam_correct / n_fam

        fam_flops = [expert_flops[c] + router_flops for c in fam_choices]
        mean_flops = sum(fam_flops) / n_fam

        counts = {exp: fam_choices.count(exp) / n_fam for exp in expert_flops}
        top_exp = max(counts, key=counts.get)  # type: ignore[arg-type]

        result[fam] = {
            "mean_flops": mean_flops,
            "accuracy": fam_acc,
            "utilization": counts,
            "dominant_expert": top_exp,
        }

    return result


def calculate_counterfactual_sacrifices(
    selected_preds: torch.Tensor,
    expert_preds: dict[str, torch.Tensor],
    oracle_choices: list[str],
    selected_choices: list[str],
    targets: torch.Tensor,
    expert_flops: dict[str, float],
) -> dict[str, Any]:
    """Distinguish useful sacrifices from harmful sacrifices under compute awareness."""
    n = len(targets)
    useful_sacrifices = 0
    harmful_sacrifices = 0
    oracle_chosen_correct = 0
    oracle_chosen_failed = 0
    more_expensive_choices = 0
    equal_cost_choices = 0

    for i in range(n):
        sel_exp = selected_choices[i]
        ora_exp = oracle_choices[i]
        sel_cost = expert_flops[sel_exp]
        ora_cost = expert_flops[ora_exp]
        is_sel_correct = (selected_preds[i].item() == targets[i].item())

        if sel_cost < ora_cost:
            if is_sel_correct:
                useful_sacrifices += 1
            else:
                harmful_sacrifices += 1
        elif sel_cost > ora_cost:
            more_expensive_choices += 1
        else:
            equal_cost_choices += 1
            if sel_exp == ora_exp:
                if is_sel_correct:
                    oracle_chosen_correct += 1
                else:
                    oracle_chosen_failed += 1

    return {
        "useful_sacrifices": useful_sacrifices,
        "harmful_sacrifices": harmful_sacrifices,
        "oracle_chosen_correct": oracle_chosen_correct,
        "oracle_chosen_failed": oracle_chosen_failed,
        "more_expensive_choices": more_expensive_choices,
        "equal_cost_choices": equal_cost_choices,
        "useful_sacrifice_rate": useful_sacrifices / n if n > 0 else 0.0,
        "harmful_sacrifice_rate": harmful_sacrifices / n if n > 0 else 0.0,
    }


def identify_representative_operating_points(
    conditions_summary: dict[str, Any],
    pareto_result: dict[str, Any],
) -> dict[str, Any]:
    """Identify pre-declared representative operating points on the performance-compute frontier."""
    learned_points = [
        p for p in pareto_result["pareto_records"]
        if p["name"].startswith("Cost-Aware") or p["name"] == "Learned Router (Phase 8A)"
    ]

    if not learned_points:
        return {"accuracy_first": None, "balanced": None, "compute_first": None}

    # 1. Accuracy-first: highest accuracy among learned conditions
    acc_first = max(learned_points, key=lambda p: p["accuracy"])

    # 2. Compute-first: lowest FLOPs among non-degenerate learned points (accuracy >= 0.60)
    non_degenerate = [p for p in learned_points if p["accuracy"] >= 0.60]
    compute_first = min(non_degenerate, key=lambda p: p["flops"]) if non_degenerate else min(learned_points, key=lambda p: p["flops"])

    # 3. Balanced: non-dominated point that achieves >= 10% compute reduction vs acc_first with <= 5% accuracy drop
    balanced_candidates = [
        p for p in pareto_result["frontier_sorted"]
        if p["name"] in [lp["name"] for lp in learned_points]
        and p["flops"] <= acc_first["flops"] * 0.90
        and p["accuracy"] >= acc_first["accuracy"] - 0.05
    ]

    balanced = balanced_candidates[len(balanced_candidates) // 2] if balanced_candidates else acc_first

    return {
        "accuracy_first": acc_first,
        "balanced": balanced,
        "compute_first": compute_first,
    }
