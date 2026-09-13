"""Phase 8A metrics: route agreement, oracle recovery, and counterfactual analysis."""
from __future__ import annotations

from typing import Any

import torch


def calculate_route_accuracy(
    chosen_experts: list[str],
    oracle_experts: list[str],
    families: list[str],
    family_names: tuple[str, ...] = ("feature", "relational", "contextual"),
) -> dict[str, float]:
    """Calculate overall and per-family route agreement with the generator oracle."""
    total = len(chosen_experts)
    if total == 0:
        raise ValueError("chosen_experts cannot be empty")

    matches = [int(c == o) for c, o in zip(chosen_experts, oracle_experts)]
    overall = sum(matches) / total

    per_family: dict[str, float] = {}
    for fam in family_names:
        fam_matches = [m for m, f in zip(matches, families) if f == fam]
        per_family[fam] = sum(fam_matches) / len(fam_matches) if fam_matches else 0.0

    return {
        "overall_route_agreement": overall,
        "feature_route_agreement": per_family.get("feature", 0.0),
        "relational_route_agreement": per_family.get("relational", 0.0),
        "contextual_route_agreement": per_family.get("contextual", 0.0),
    }


def calculate_oracle_recovery(
    learned_acc: float,
    random_acc: float,
    oracle_acc: float,
) -> dict[str, float]:
    """Calculate the fraction of the oracle routing advantage recovered by the learned router."""
    denom = oracle_acc - random_acc
    if abs(denom) < 1e-7:
        recovery_fraction = 0.0
    else:
        recovery_fraction = (learned_acc - random_acc) / denom

    return {
        "oracle_recovery_fraction": recovery_fraction,
        "oracle_gap": oracle_acc - learned_acc,
        "random_baseline_accuracy": random_acc,
        "learned_router_accuracy": learned_acc,
        "oracle_accuracy": oracle_acc,
    }


def calculate_counterfactual_analysis(
    selected_preds: torch.Tensor,
    all_expert_preds: dict[str, torch.Tensor],
    oracle_choices: list[str],
    chosen_choices: list[str],
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Analyze whether prediction errors stem from routing error or underlying expert failure.

    Categorizes each sample:
      - oracle_expert_and_correct: Router selected oracle expert, and prediction was correct.
      - non_oracle_but_correct: Router selected different expert, but prediction was still correct.
      - router_selection_error: Oracle expert was correct, but router chose a failing expert.
      - intrinsic_expert_failure: Router chose the oracle expert, but that expert was wrong.
      - mutual_failure: Neither the oracle expert nor the chosen expert was correct.
    """
    total = len(targets)
    oracle_preds = torch.tensor(
        [all_expert_preds[oracle_choices[i]][i].item() for i in range(total)],
        dtype=torch.long,
    )

    selected_correct = (selected_preds == targets)
    oracle_correct = (oracle_preds == targets)
    chose_oracle = torch.tensor([chosen_choices[i] == oracle_choices[i] for i in range(total)], dtype=torch.bool)

    c1 = (chose_oracle & selected_correct).sum().item()
    c2 = ((~chose_oracle) & selected_correct).sum().item()
    c3 = (oracle_correct & (~selected_correct)).sum().item()
    c4 = (chose_oracle & (~selected_correct)).sum().item()
    c5 = ((~chose_oracle) & (~selected_correct) & (~oracle_correct)).sum().item()

    return {
        "total_samples": total,
        "oracle_expert_and_correct": c1,
        "non_oracle_but_correct": c2,
        "router_selection_error": c3,
        "intrinsic_expert_failure": c4,
        "mutual_failure": c5,
        "selection_error_rate": c3 / total if total > 0 else 0.0,
        "intrinsic_failure_rate": c4 / total if total > 0 else 0.0,
    }


def check_routing_collapse(
    utilization: dict[str, float],
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Flag optimization pathologies such as single-module collapse."""
    max_expert = max(utilization, key=utilization.get)
    max_val = utilization[max_expert]
    collapsed = max_val >= threshold
    active_count = sum(1 for v in utilization.values() if v >= 0.05)

    return {
        "collapsed": collapsed,
        "dominant_expert": max_expert,
        "dominant_utilization": max_val,
        "active_experts_count": active_count,
        "is_uniform_like": all(abs(v - 0.25) < 0.06 for v in utilization.values()),
    }
