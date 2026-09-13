"""Unit tests for Phase 8B metrics: Pareto frontier, normalized costs, and sacrifice attribution."""
from __future__ import annotations

import pytest
import torch

from neuroforge.evaluation.phase8b_metrics import (
    calculate_counterfactual_sacrifices,
    calculate_family_conditional_cost,
    calculate_pareto_frontier,
    check_cost_collapse,
    compute_normalized_costs,
    identify_representative_operating_points,
)


def test_compute_normalized_costs():
    expert_flops = {
        "mlp": 8928.0,
        "graph": 8112.0,
        "attention": 39168.0,
        "attention_v2": 46296.0,
    }
    router_flops = 1052.0
    norm_costs, ref_cost = compute_normalized_costs(expert_flops, router_flops)

    # Reference cost should be max total cost = 46296 + 1052 = 47348
    assert ref_cost == 47348.0
    assert norm_costs["attention_v2"] == 1.0
    assert norm_costs["graph"] == (8112.0 + 1052.0) / 47348.0
    assert norm_costs["mlp"] == (8928.0 + 1052.0) / 47348.0
    assert norm_costs["attention"] == (39168.0 + 1052.0) / 47348.0


def test_calculate_pareto_frontier():
    points = [
        {"name": "Point A (Optimal Low-Compute)", "accuracy": 0.85, "flops": 10000.0},
        {"name": "Point B (Dominated by A)", "accuracy": 0.80, "flops": 12000.0},
        {"name": "Point C (Optimal High-Acc)", "accuracy": 0.92, "flops": 25000.0},
        {"name": "Point D (Dominated by C)", "accuracy": 0.90, "flops": 30000.0},
    ]

    res = calculate_pareto_frontier(points)
    assert set(res["non_dominated_names"]) == {"Point A (Optimal Low-Compute)", "Point C (Optimal High-Acc)"}
    assert set(res["dominated_names"]) == {"Point B (Dominated by A)", "Point D (Dominated by C)"}

    # Frontier sorted ascending by flops
    frontier = res["frontier_sorted"]
    assert len(frontier) == 2
    assert frontier[0]["name"] == "Point A (Optimal Low-Compute)"
    assert frontier[1]["name"] == "Point C (Optimal High-Acc)"


def test_check_cost_collapse():
    # Normal balanced routing
    util_healthy = {"mlp": 0.30, "graph": 0.35, "attention": 0.05, "attention_v2": 0.30}
    matrix_healthy = {
        "feature": {"mlp": 0.80, "graph": 0.10, "attention": 0.0, "attention_v2": 0.10},
        "relational": {"mlp": 0.05, "graph": 0.95, "attention": 0.0, "attention_v2": 0.0},
        "contextual": {"mlp": 0.0, "graph": 0.0, "attention": 0.0, "attention_v2": 1.0},
    }
    healthy_diag = check_cost_collapse(util_healthy, matrix_healthy)
    assert not healthy_diag["cheapest_expert_collapse"]
    assert not healthy_diag["v2_avoidance"]
    assert not healthy_diag["v1_avoidance"]
    assert healthy_diag["useful_selective_v2"]

    # Collapse to cheapest expert (Graph)
    util_collapsed = {"mlp": 0.05, "graph": 0.90, "attention": 0.02, "attention_v2": 0.03}
    col_diag = check_cost_collapse(util_collapsed)
    assert col_diag["cheapest_expert_collapse"]
    assert col_diag["v2_avoidance"]


def test_calculate_family_conditional_cost():
    targets = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.long)
    preds = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.long)  # last contextual is wrong
    families = ["feature", "feature", "relational", "relational", "contextual", "contextual"]
    choices = ["mlp", "mlp", "graph", "graph", "attention_v2", "attention_v2"]
    flops = {"mlp": 8000.0, "graph": 7000.0, "attention": 30000.0, "attention_v2": 40000.0}

    res = calculate_family_conditional_cost(choices, families, targets, preds, flops, router_flops=1000.0)
    assert res["feature"]["accuracy"] == 1.0
    assert res["feature"]["mean_flops"] == 9000.0
    assert res["feature"]["dominant_expert"] == "mlp"

    assert res["relational"]["accuracy"] == 1.0
    assert res["relational"]["mean_flops"] == 8000.0
    assert res["relational"]["dominant_expert"] == "graph"

    assert res["contextual"]["accuracy"] == 0.5
    assert res["contextual"]["mean_flops"] == 41000.0
    assert res["contextual"]["dominant_expert"] == "attention_v2"


def test_calculate_counterfactual_sacrifices():
    targets = torch.tensor([1, 1, 0], dtype=torch.long)
    oracle_choices = ["attention_v2", "attention_v2", "graph"]
    # Sample 0: picked MLP (cheaper than V2: 8928 < 46296), correct -> useful sacrifice
    # Sample 1: picked MLP (cheaper than V2: 8928 < 46296), wrong -> harmful sacrifice
    # Sample 2: picked Graph (same as oracle: 8112 == 8112), correct -> oracle chosen correct
    selected_choices = ["mlp", "mlp", "graph"]
    selected_preds = torch.tensor([1, 0, 0], dtype=torch.long)
    dummy_expert_preds = {
        "mlp": torch.tensor([1, 0, 0], dtype=torch.long),
        "graph": torch.tensor([0, 0, 0], dtype=torch.long),
        "attention_v2": torch.tensor([1, 1, 0], dtype=torch.long),
        "attention": torch.tensor([0, 0, 0], dtype=torch.long),
    }
    flops = {"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0}

    cf = calculate_counterfactual_sacrifices(
        selected_preds, dummy_expert_preds, oracle_choices, selected_choices, targets, flops
    )
    assert cf["useful_sacrifices"] == 1
    assert cf["harmful_sacrifices"] == 1
    assert cf["oracle_chosen_correct"] == 1
    assert cf["oracle_chosen_failed"] == 0
    assert pytest.approx(cf["useful_sacrifice_rate"], rel=1e-3) == 0.3333
    assert pytest.approx(cf["harmful_sacrifice_rate"], rel=1e-3) == 0.3333


def test_identify_representative_operating_points():
    eval_records = [
        {"name": "Cost-Aware (λ=0.000)", "accuracy": 0.905, "flops": 27000.0},
        {"name": "Cost-Aware (λ=0.050)", "accuracy": 0.890, "flops": 22000.0},
        {"name": "Cost-Aware (λ=0.500)", "accuracy": 0.720, "flops": 12000.0},
        {"name": "Cost-Aware (λ=2.000)", "accuracy": 0.520, "flops": 9500.0},  # degenerate (<0.60)
    ]
    pareto_res = calculate_pareto_frontier(eval_records)
    op = identify_representative_operating_points({}, pareto_res)

    assert op["accuracy_first"]["name"] == "Cost-Aware (λ=0.000)"
    assert op["balanced"]["name"] == "Cost-Aware (λ=0.050)"
    assert op["compute_first"]["name"] == "Cost-Aware (λ=0.500)"
