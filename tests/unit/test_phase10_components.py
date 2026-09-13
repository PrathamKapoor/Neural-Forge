"""Unit tests for Phase 10 Multi-Expert Routing, Composition, and Evaluation Metrics."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from neuroforge.evaluation.composition_metrics import (
    calculate_accuracy_constrained_compute,
    calculate_active_expert_statistics,
    calculate_composition_entropy,
    calculate_counterfactual_composition_analysis,
    calculate_pair_utilization,
    calculate_triple_utilization,
    check_composition_collapses,
    diagnose_phase10_failure,
    generate_family_composition_matrix,
)
from neuroforge.models.composition import (
    ParallelMultiExpert,
    SequentialSpecialistComposition,
)
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.multi_expert import (
    AdaptiveKRouter,
    TopKRouter,
    aggregate_parallel_normalized,
    aggregate_parallel_uniform,
    aggregate_parallel_weighted,
)


def _create_mock_specialists() -> dict[str, StandaloneSpecialist]:
    """Create lightweight mock specialists for testing."""
    specialists = {
        "mlp": StandaloneSpecialist(architecture="mlp", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "graph": StandaloneSpecialist(architecture="graph", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "attention": StandaloneSpecialist(architecture="attention", input_dim=8, hidden_dim=24, depth=1, num_classes=2),
        "attention_v2": StandaloneSpecialist(architecture="attention_v2", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
    }
    for m in specialists.values():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return specialists


def test_topk_router_shapes_and_k_selection():
    """Verify TopKRouter outputs correct dimensions, distinct indices, and weights."""
    router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(5, 12, 8)

    # Test k=1
    dec1 = router(x, k=1, mode="hard")
    assert dec1.logits.shape == (5, 4)
    assert dec1.weights.shape == (5, 4)
    assert len(dec1.selected_indices) == 5
    assert all(len(indices) == 1 for indices in dec1.selected_indices)
    assert dec1.k_values.tolist() == [1, 1, 1, 1, 1]

    # Test k=2
    dec2 = router(x, k=2, mode="hard")
    assert len(dec2.selected_indices) == 5
    for indices in dec2.selected_indices:
        assert len(indices) == 2
        assert indices[0] != indices[1], "Selected indices for a sample must be distinct"
        assert all(0 <= idx < 4 for idx in indices)
    assert dec2.k_values.tolist() == [2, 2, 2, 2, 2]

    # Test k=3
    dec3 = router(x, k=3, mode="hard")
    assert len(dec3.selected_indices) == 5
    for indices in dec3.selected_indices:
        assert len(indices) == 3
        assert len(set(indices)) == 3

    # Test invalid k
    with pytest.raises(ValueError, match="k must be between 1 and 4"):
        router(x, k=0)
    with pytest.raises(ValueError, match="k must be between 1 and 4"):
        router(x, k=5)


def test_topk_router_straight_through_gradients():
    """Verify straight-through mode produces hard selections while passing soft gradients."""
    router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(4, 12, 8)

    decision = router(x, k=2, mode="straight_through")
    loss = decision.weights.sum()
    loss.backward()

    # Router parameters must have received gradients
    has_grads = any(p.grad is not None and torch.any(p.grad != 0) for p in router.parameters())
    assert has_grads, "Gradients must flow to router parameters under straight_through mode"


def test_adaptive_k_router_strategies():
    """Verify AdaptiveKRouter under learned, entropy, and margin strategies."""
    router = AdaptiveKRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(6, 12, 8)

    # 1. Learned strategy
    dec_learned = router(x, mode="hard", strategy="learned")
    assert len(dec_learned.selected_indices) == 6
    assert dec_learned.k_values.shape == (6,)
    for i in range(6):
        ki = dec_learned.k_values[i].item()
        assert ki in {1, 2, 3}
        assert len(dec_learned.selected_indices[i]) == ki

    # 2. Entropy strategy
    dec_entropy = router(x, mode="hard", strategy="entropy", entropy_thresholds=(0.5, 1.0))
    for ki in dec_entropy.k_values.tolist():
        assert ki in {1, 2, 3}

    # 3. Margin strategy
    dec_margin = router(x, mode="hard", strategy="margin", margin_thresholds=(0.3, 0.1))
    for ki in dec_margin.k_values.tolist():
        assert ki in {1, 2, 3}

    # 4. Straight-through learned gradient check
    router.zero_grad()
    dec_st = router(x, mode="straight_through", strategy="learned")
    loss = dec_st.weights.sum() + dec_st.k_logits.sum()
    loss.backward()
    has_khead_grads = any(p.grad is not None for p in router.k_head.parameters())
    assert has_khead_grads, "Gradients must flow to k_head parameters"


def test_parallel_aggregations():
    """Verify C1 uniform, C2 weighted, and C3 normalized aggregations."""
    z1 = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    z2 = torch.tensor([[0.0, 4.0], [3.0, 1.0]])
    logits_list = [z1, z2]

    # C1 Uniform: mean = ([2+0, 0+4]/2, [1+3, 3+1]/2) = ([1.0, 2.0], [2.0, 2.0])
    c1 = aggregate_parallel_uniform(logits_list)
    expected_c1 = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
    assert torch.allclose(c1, expected_c1)

    # C2 Weighted: weights [0.75, 0.25]
    w = torch.tensor([[0.75, 0.25], [0.5, 0.5]])
    c2 = aggregate_parallel_weighted(logits_list, w)
    # For sample 0: 0.75 * [2, 0] + 0.25 * [0, 4] = [1.5, 1.0]
    # For sample 1: 0.5 * [1, 3] + 0.5 * [3, 1] = [2.0, 2.0]
    expected_c2 = torch.tensor([[1.5, 1.0], [2.0, 2.0]])
    assert torch.allclose(c2, expected_c2)

    # C3 Normalized: Softmax then weighted sum in probability space
    c3 = aggregate_parallel_normalized(logits_list, w)
    assert c3.shape == (2, 2)
    # Probabilities must be log-probabilities (negative or zero)
    assert torch.all(c3 <= 0.0)

    # Empty raises ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        aggregate_parallel_uniform([])


def test_parallel_multi_expert_execution():
    """Verify end-to-end execution of ParallelMultiExpert wrapper."""
    specialists = _create_mock_specialists()
    router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)
    model = ParallelMultiExpert(
        experts=specialists,
        router=router,
        aggregation="uniform",
    )

    x = torch.randn(4, 12, 8)
    logits, decision, flops = model(x, k=2, mode="hard")

    assert logits.shape == (4, 2)
    assert len(decision.selected_indices) == 4
    assert len(flops) == 4
    assert all(f > 10000.0 for f in flops), "FLOPs must include router and expert forwards"


def test_sequential_specialist_composition():
    """Verify SequentialSpecialistComposition forward in block_chain and cascade modes."""
    specialists = _create_mock_specialists()
    seq_model = SequentialSpecialistComposition(
        experts=specialists,
        sequence=("graph", "mlp"),
        mode="block_chain",
    )

    x = torch.randn(4, 12, 8)
    logits_bc, flops_bc = seq_model(x)
    assert logits_bc.shape == (4, 2)
    assert flops_bc == 8112.0 + 8928.0

    cascade_model = SequentialSpecialistComposition(
        experts=specialists,
        sequence=("graph", "mlp"),
        mode="cascade",
    )
    logits_cas, flops_cas = cascade_model(x)
    assert logits_cas.shape == (4, 2)
    assert flops_cas == 8112.0 + 8928.0


def test_composition_metrics_active_stats():
    """Verify calculate_active_expert_statistics."""
    k_vals = [1, 1, 2, 2, 2, 3]
    stats = calculate_active_expert_statistics(k_vals)
    assert pytest.approx(stats["mean_k"], rel=1e-3) == 11.0 / 6.0
    assert pytest.approx(stats["p_k1"], rel=1e-3) == 2.0 / 6.0
    assert pytest.approx(stats["p_k2"], rel=1e-3) == 3.0 / 6.0
    assert pytest.approx(stats["p_k3"], rel=1e-3) == 1.0 / 6.0
    assert stats["k_entropy"] > 0.0

    empty_stats = calculate_active_expert_statistics([])
    assert empty_stats["mean_k"] == 0.0
    assert empty_stats["k_entropy"] == 0.0


def test_pair_and_triple_utilization():
    """Verify pair and triple canonical frequencies."""
    tuples = [
        ("mlp", "graph"),
        ("graph", "mlp"),
        ("attention_v2", "graph"),
    ]
    pair_stats = calculate_pair_utilization(tuples)
    assert "Graph + MLP" in pair_stats
    assert pytest.approx(pair_stats["Graph + MLP"], rel=1e-3) == 2.0 / 3.0
    assert pytest.approx(pair_stats["Attn V2 + Graph"], rel=1e-3) == 1.0 / 3.0

    triples = [
        ("mlp", "graph", "attention_v2"),
        ("attention_v2", "graph", "mlp"),
    ]
    triple_stats = calculate_triple_utilization(triples)
    assert "Attn V2 + Graph + MLP" in triple_stats
    assert pytest.approx(triple_stats["Attn V2 + Graph + MLP"]) == 1.0


def test_composition_entropy_and_matrix():
    """Verify composition entropy and family matrix calculation."""
    tuples = [("mlp", "graph"), ("mlp", "graph"), ("attention_v2", "mlp")]
    ent = calculate_composition_entropy(tuples)
    assert ent > 0.0

    families = ["feature_relational", "feature_relational", "feature_contextual"]
    mat = generate_family_composition_matrix(families, tuples)
    assert "feature_relational" in mat
    assert "feature_contextual" in mat
    assert "Graph+MLP" in mat["feature_relational"]
    assert pytest.approx(mat["feature_relational"]["Graph+MLP"]) == 1.0


def test_composition_collapses_detection():
    """Verify detection of the 4 pre-declared collapse conditions."""
    # 1. Clean run (no collapses)
    clean = check_composition_collapses(
        mean_k_pure=1.2,
        p_k1_mixed=0.20,
        expert_frequencies={"mlp": 0.4, "graph": 0.3, "attention_v2": 0.3},
        mean_flops=25000.0,
        all_expert_flops=60000.0,
        delta_acc=0.08,
    )
    assert not clean["any_collapse"]

    # 2. All-expert collapse
    all_col = check_composition_collapses(
        mean_k_pure=2.90,
        p_k1_mixed=0.20,
        expert_frequencies={"mlp": 0.4, "graph": 0.3, "attention_v2": 0.3},
        mean_flops=55000.0,
        all_expert_flops=60000.0,
        delta_acc=0.08,
    )
    assert all_col["all_expert_collapse"]
    assert all_col["any_collapse"]

    # 3. Single-expert collapse on mixed
    single_col = check_composition_collapses(
        mean_k_pure=1.1,
        p_k1_mixed=0.98,
        expert_frequencies={"mlp": 0.4, "graph": 0.3, "attention_v2": 0.3},
        mean_flops=12000.0,
        all_expert_flops=60000.0,
        delta_acc=0.01,
    )
    assert single_col["single_expert_collapse"]

    # 4. Expert dominance collapse
    dom_col = check_composition_collapses(
        mean_k_pure=1.5,
        p_k1_mixed=0.3,
        expert_frequencies={"mlp": 0.95, "graph": 0.05},
        mean_flops=20000.0,
        all_expert_flops=60000.0,
        delta_acc=0.04,
    )
    assert dom_col["expert_collapse"]

    # 5. Compute collapse
    comp_col = check_composition_collapses(
        mean_k_pure=1.5,
        p_k1_mixed=0.3,
        expert_frequencies={"mlp": 0.5, "graph": 0.5},
        mean_flops=58000.0,
        all_expert_flops=60000.0,
        delta_acc=0.01,
    )
    assert comp_col["compute_collapse"]


def test_accuracy_constrained_compute():
    """Verify identification of min compute for accuracy milestones."""
    pareto_pts = [
        {"name": "k1", "accuracy": 0.74, "flops": 12000.0},
        {"name": "k2", "accuracy": 0.88, "flops": 24000.0},
        {"name": "k3", "accuracy": 0.92, "flops": 50000.0},
    ]
    res = calculate_accuracy_constrained_compute(pareto_pts, thresholds=[0.70, 0.80, 0.90, 0.95])
    assert res["70%"] == 12000.0
    assert res["80%"] == 24000.0
    assert res["90%"] == 50000.0
    assert res["95%"] is None


def test_counterfactual_composition_analysis():
    """Verify leave-one-expert-out ablation on multi-expert predictions."""
    targets = torch.tensor([1, 0, 1])
    selected_tuples = [("mlp", "graph"), ("mlp", "graph"), ("graph",)]
    expert_logits = {
        "mlp": torch.tensor([[2.0, -2.0], [1.0, 3.0], [0.0, 2.0]]),
        "graph": torch.tensor([[-1.0, 3.0], [4.0, -1.0], [0.0, 2.0]]),
    }

    res = calculate_counterfactual_composition_analysis(
        selected_tuples=selected_tuples,
        expert_logits_dict=expert_logits,
        targets=targets,
    )
    assert res["multi_sample_count"] == 2
    assert "mean_leave_one_out_drop_impact" in res
    assert "mlp" in res["mean_leave_one_out_drop_impact"]
    assert "graph" in res["mean_leave_one_out_drop_impact"]


def test_diagnose_phase10_failure():
    """Verify diagnostic classification across success and failure modes."""
    # 1. Success
    succ = diagnose_phase10_failure(
        k1_acc=0.74,
        multi_acc=0.91,
        ceiling_acc=0.75,
        best_single_exp="attention_v2",
        multi_exp_combinations={"mlp+graph": 0.92},
        desired_threshold=0.85,
    )
    assert "Success" in succ["failure_mode"]

    # 2. Failure A: Incapability
    fail_a = diagnose_phase10_failure(
        k1_acc=0.52,
        multi_acc=0.55,
        ceiling_acc=0.53,
        best_single_exp="mlp",
        multi_exp_combinations={"mlp+graph": 0.56},
        desired_threshold=0.85,
    )
    assert "Failure A" in fail_a["failure_mode"]

    # 3. Failure B: Selection failure
    fail_b = diagnose_phase10_failure(
        k1_acc=0.68,
        multi_acc=0.72,
        ceiling_acc=0.70,
        best_single_exp="mlp",
        multi_exp_combinations={"mlp+graph": 0.90},
        desired_threshold=0.85,
    )
    assert "Failure B" in fail_b["failure_mode"]

    # 4. Failure C: Aggregation failure
    fail_c = diagnose_phase10_failure(
        k1_acc=0.74,
        multi_acc=0.74,
        ceiling_acc=0.75,
        best_single_exp="mlp",
        multi_exp_combinations={"mlp+graph": 0.76},
        desired_threshold=0.85,
    )
    assert "Failure C" in fail_c["failure_mode"]
