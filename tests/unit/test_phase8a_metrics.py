"""Unit tests for Phase 8A metrics and diagnostics."""
from __future__ import annotations

import pytest
import torch

from neuroforge.evaluation.phase8a_metrics import (
    calculate_counterfactual_analysis,
    calculate_oracle_recovery,
    calculate_route_accuracy,
    check_routing_collapse,
)


def test_calculate_route_accuracy():
    chosen = ["mlp", "graph", "attention_v2", "mlp"]
    oracle = ["mlp", "mlp", "attention_v2", "mlp"]
    families = ["feature", "relational", "contextual", "feature"]

    res = calculate_route_accuracy(chosen, oracle, families)
    assert res["overall_route_agreement"] == 0.75
    assert res["feature_route_agreement"] == 1.0
    assert res["relational_route_agreement"] == 0.0
    assert res["contextual_route_agreement"] == 1.0


def test_calculate_oracle_recovery():
    # Standard recovery
    res = calculate_oracle_recovery(learned_acc=0.85, random_acc=0.60, oracle_acc=0.90)
    # (0.85 - 0.60) / (0.90 - 0.60) = 0.25 / 0.30 = 0.8333...
    assert pytest.approx(res["oracle_recovery_fraction"], rel=1e-3) == 0.8333
    assert pytest.approx(res["oracle_gap"], rel=1e-3) == 0.05

    # Zero denominator safety
    res_zero = calculate_oracle_recovery(learned_acc=0.70, random_acc=0.70, oracle_acc=0.70)
    assert res_zero["oracle_recovery_fraction"] == 0.0


def test_calculate_counterfactual_analysis():
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    all_expert_preds = {
        "mlp": torch.tensor([1, 1, 0, 0], dtype=torch.long),
        "graph": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "attention_v2": torch.tensor([1, 0, 1, 0], dtype=torch.long),
    }
    oracle_choices = ["mlp", "graph", "attention_v2", "mlp"]
    chosen_choices = ["mlp", "graph", "mlp", "mlp"]
    selected_preds = torch.tensor([1, 0, 0, 0], dtype=torch.long)

    cf = calculate_counterfactual_analysis(
        selected_preds, all_expert_preds, oracle_choices, chosen_choices, targets
    )
    assert cf["total_samples"] == 4
    total_categorized = (
        cf["oracle_expert_and_correct"]
        + cf["non_oracle_but_correct"]
        + cf["router_selection_error"]
        + cf["intrinsic_expert_failure"]
        + cf["mutual_failure"]
    )
    assert total_categorized == 4


def test_check_routing_collapse():
    # Non-collapsed
    balanced_util = {"mlp": 0.35, "graph": 0.30, "attention": 0.0, "attention_v2": 0.35}
    res = check_routing_collapse(balanced_util, threshold=0.85)
    assert not res["collapsed"]
    assert res["active_experts_count"] == 3

    # Collapsed to MLP
    collapsed_util = {"mlp": 0.92, "graph": 0.03, "attention": 0.01, "attention_v2": 0.04}
    res_col = check_routing_collapse(collapsed_util, threshold=0.85)
    assert res_col["collapsed"]
    assert res_col["dominant_expert"] == "mlp"
    assert res_col["dominant_utilization"] == 0.92
