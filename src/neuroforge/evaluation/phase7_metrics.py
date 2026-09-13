"""Routing statistics, regret analysis, and Pareto evaluations for Phase 7."""
from __future__ import annotations

import math
from typing import Any

import torch


def calculate_routing_statistics(
    chosen_experts: list[str],
    families: list[str],
    expert_names: tuple[str, ...] = ("mlp", "graph", "attention", "attention_v2"),
    family_names: tuple[str, ...] = ("feature", "relational", "contextual"),
) -> dict[str, Any]:
    """Calculate module utilization, selection matrix, and routing entropy."""
    total = len(chosen_experts)
    if total == 0:
        raise ValueError("chosen_experts cannot be empty")

    utilization = {name: 0.0 for name in expert_names}
    for exp in chosen_experts:
        if exp in utilization:
            utilization[exp] += 1.0
    for name in expert_names:
        utilization[name] /= total

    # Selection matrix [len(family_names), len(expert_names)]
    matrix = {fam: {exp: 0 for exp in expert_names} for fam in family_names}
    family_counts = {fam: 0 for fam in family_names}
    for fam, exp in zip(families, chosen_experts):
        matrix[fam][exp] += 1
        family_counts[fam] += 1

    matrix_pct = {
        fam: {
            exp: (matrix[fam][exp] / family_counts[fam]) if family_counts[fam] > 0 else 0.0
            for exp in expert_names
        }
        for fam in family_names
    }

    # Routing entropy: - sum p * log2(p)
    entropy = 0.0
    for p in utilization.values():
        if p > 0.0:
            entropy -= p * math.log2(p)

    active_modules = sum(1 for p in utilization.values() if p > 0.0)

    return {
        "utilization": utilization,
        "selection_matrix_counts": matrix,
        "selection_matrix_proportions": matrix_pct,
        "routing_entropy_bits": entropy,
        "active_modules_count": active_modules,
        "average_active_modules_per_sample": 1.0,
    }


def calculate_oracle_regret(
    oracle_preds: torch.Tensor,
    best_fixed_preds: torch.Tensor,
    targets: torch.Tensor,
    families: list[str],
    family_names: tuple[str, ...] = ("feature", "relational", "contextual"),
) -> dict[str, Any]:
    """Quantify sample-level agreement, advantage, and regret between Oracle and Best Fixed."""
    total = len(targets)
    disagreements = int((oracle_preds != best_fixed_preds).sum().item())
    oracle_corr = (oracle_preds == targets)
    fixed_corr = (best_fixed_preds == targets)

    oracle_acc = float(oracle_corr.float().mean().item())
    best_fixed_acc = float(fixed_corr.float().mean().item())

    # Oracle wins (oracle correct, fixed wrong) vs Fixed wins (fixed correct, oracle wrong)
    oracle_wins = int((oracle_corr & (~fixed_corr)).sum().item())
    fixed_wins = int((fixed_corr & (~oracle_corr)).sum().item())

    family_breakdown = {}
    for fam in family_names:
        mask = torch.tensor([f == fam for f in families], dtype=torch.bool)
        fam_total = int(mask.sum().item())
        fam_oracle = float(oracle_corr[mask].float().mean().item()) if fam_total > 0 else 0.0
        fam_fixed = float(fixed_corr[mask].float().mean().item()) if fam_total > 0 else 0.0
        family_breakdown[fam] = {
            "samples": fam_total,
            "oracle_accuracy": fam_oracle,
            "best_fixed_accuracy": fam_fixed,
            "accuracy_delta": fam_oracle - fam_fixed,
        }

    return {
        "disagreement_count": disagreements,
        "disagreement_rate": disagreements / total if total > 0 else 0.0,
        "oracle_accuracy": oracle_acc,
        "best_fixed_accuracy": best_fixed_acc,
        "net_accuracy_advantage": oracle_acc - best_fixed_acc,
        "oracle_win_count": oracle_wins,
        "fixed_win_count": fixed_wins,
        "family_breakdown": family_breakdown,
    }
