"""Evaluation metrics, generalization gaps, and failure diagnostics for Phase 9."""
from __future__ import annotations

from typing import Any
import torch


def calculate_generalization_gap(
    baseline_acc: float,
    eval_acc: float,
    baseline_flops: float,
    eval_flops: float,
) -> dict[str, float]:
    """Calculate accuracy and compute generalization gaps relative to in-distribution baseline."""
    acc_gap = baseline_acc - eval_acc
    rel_acc_drop = (baseline_acc - eval_acc) / baseline_acc if baseline_acc > 0 else 0.0
    flops_shift = eval_flops - baseline_flops
    rel_flops_shift = (eval_flops - baseline_flops) / baseline_flops if baseline_flops > 0 else 0.0

    return {
        "accuracy_gap": acc_gap,
        "relative_accuracy_drop": rel_acc_drop,
        "flops_shift": flops_shift,
        "relative_flops_shift": rel_flops_shift,
    }


def calculate_decision_preservation(
    clean_choices: list[str],
    transformed_choices: list[str],
) -> float:
    """Calculate the proportion of samples whose selected expert is preserved under transformation."""
    if not clean_choices or not transformed_choices:
        return 0.0
    n = min(len(clean_choices), len(transformed_choices))
    preserved = sum(1 for i in range(n) if clean_choices[i] == transformed_choices[i])
    return preserved / n


def calculate_single_expert_ceiling(
    expert_accuracies: dict[str, float],
) -> dict[str, Any]:
    """Calculate the best single-expert accuracy for a given task family."""
    if not expert_accuracies:
        return {"best_expert": None, "ceiling_accuracy": 0.0}
    best_expert = max(expert_accuracies, key=lambda k: expert_accuracies[k])
    return {
        "best_expert": best_expert,
        "ceiling_accuracy": expert_accuracies[best_expert],
    }


def diagnose_mixed_structure_failure(
    router_acc: float,
    ceiling_acc: float,
    expert_accuracies: dict[str, float],
    desired_threshold: float = 0.85,
) -> dict[str, Any]:
    """Diagnose and categorize performance outcomes on mixed-structure tasks.

    Failure Modes (Section 22):
      - Failure A: Expert incapability (no expert can solve it, ceiling < 0.60)
      - Failure B: Router selection failure (an expert achieves >= desired_threshold, router picks wrong one)
      - Failure C: Representation failure (router cannot distinguish mixture, selects at chance)
      - Failure D: Single-expert limitation (ceiling is partial, task genuinely requires composition)
      - Success: Router meets or exceeds desired threshold
    """
    if router_acc >= desired_threshold and ceiling_acc >= desired_threshold:
        mode = "Success — Robust Generalization"
        desc = f"Router achieves {router_acc*100:.1f}% >= {desired_threshold*100:.1f}% threshold."
    elif ceiling_acc >= desired_threshold and router_acc < ceiling_acc - 0.08:
        mode = "Failure B — Router Selection Failure"
        best_exp = max(expert_accuracies, key=lambda k: expert_accuracies[k])
        desc = (
            f"Expert '{best_exp}' achieves {ceiling_acc*100:.1f}% >= {desired_threshold*100:.1f}%, "
            f"but router achieved only {router_acc*100:.1f}% (gap: {(ceiling_acc - router_acc)*100:.1f}%)."
        )
    elif ceiling_acc < 0.60:
        mode = "Failure A — Expert Incapability"
        desc = f"No existing expert can solve the task (best single expert: {ceiling_acc*100:.1f}% < 60%)."
    elif ceiling_acc < desired_threshold:
        mode = "Failure D — Single-Expert Compositional Limitation"
        desc = (
            f"Best single-expert ceiling is bounded at {ceiling_acc*100:.1f}% < {desired_threshold*100:.1f}%. "
            f"Task genuinely requires multi-expert composition."
        )
    else:
        mode = "Failure C — Representation Limitation"
        desc = f"Router representation cannot resolve mixed requirements (router acc: {router_acc*100:.1f}%)."

    return {
        "failure_mode": mode,
        "description": desc,
        "ceiling_accuracy": ceiling_acc,
        "router_accuracy": router_acc,
        "router_to_ceiling_gap": ceiling_acc - router_acc,
    }


def calculate_mixed_counterfactual_analysis(
    selected_choices: list[str],
    selected_preds: torch.Tensor,
    expert_preds: dict[str, torch.Tensor],
    targets: torch.Tensor,
    expert_flops: dict[str, float],
    router_flops: float = 1052.0,
) -> dict[str, Any]:
    """Perform offline counterfactual analysis on mixed-structure samples."""
    n = len(targets)
    if n == 0:
        return {}

    expert_accs = {
        name: float((expert_preds[name] == targets).float().mean().item())
        for name in expert_preds
    }
    ceiling_info = calculate_single_expert_ceiling(expert_accs)
    best_expert = ceiling_info["best_expert"]
    ceiling_acc = ceiling_info["ceiling_accuracy"]
    router_acc = float((selected_preds == targets).float().mean().item())

    # Count how often router matched the best performing expert for each sample
    sample_best_matches = 0
    for i in range(n):
        best_for_sample = max(expert_preds.keys(), key=lambda e: int(expert_preds[e][i].item() == targets[i].item()))
        if selected_choices[i] == best_for_sample or (selected_preds[i].item() == targets[i].item()):
            sample_best_matches += 1

    selected_flops = [expert_flops[c] + router_flops for c in selected_choices]
    mean_selected_flops = sum(selected_flops) / n

    return {
        "router_accuracy": router_acc,
        "ceiling_accuracy": ceiling_acc,
        "best_single_expert": best_expert,
        "router_to_ceiling_gap": ceiling_acc - router_acc,
        "expert_accuracies": expert_accs,
        "mean_selected_flops": mean_selected_flops,
        "sample_effective_match_rate": sample_best_matches / n,
    }
