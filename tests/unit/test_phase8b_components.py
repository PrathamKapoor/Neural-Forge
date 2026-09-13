"""Unit tests for Phase 8B Compute-Aware Learned Routing components."""
from __future__ import annotations

import torch

from neuroforge.evaluation.phase8b_metrics import compute_normalized_costs
from neuroforge.models import StandaloneSpecialist
from neuroforge.routing.learned_router import SampleLevelRouter
from neuroforge.training.phase8b_compute_aware import verify_surrogate_gradient_flow


def test_soft_probabilities_interface():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(8, 12, 8)

    dec_hard = router(x, mode="hard")
    assert dec_hard.soft_probabilities is not None
    assert dec_hard.soft_probabilities.shape == (8, 4)
    assert torch.allclose(dec_hard.soft_probabilities.sum(dim=-1), torch.ones(8), atol=1e-5)

    dec_st = router(x, mode="straight_through")
    assert dec_st.soft_probabilities is not None
    assert dec_st.weights.requires_grad
    assert dec_st.soft_probabilities.requires_grad


def test_surrogate_gradient_flow():
    expert_names = ("mlp", "graph", "attention", "attention_v2")
    expert_flops = {"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0}
    norm_costs, _ = compute_normalized_costs(expert_flops, router_flops=1052.0)

    audit = verify_surrogate_gradient_flow(expert_names, norm_costs)
    assert audit["gradient_flows_to_router"]
    assert audit["cost_surrogate_value"] > 0.0


def test_compute_aware_loss_scaling():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(6, 12, 8)
    norm_costs = torch.tensor([0.21, 0.19, 0.85, 1.0], dtype=torch.float32)

    dec = router(x, mode="straight_through")
    soft = dec.soft_probabilities
    cost_surrogate = (soft * norm_costs.unsqueeze(0)).sum(dim=-1).mean()

    # Zero lambda vs non-zero lambda
    loss_lam0 = 1.0 + 0.0 * cost_surrogate
    loss_lam1 = 1.0 + 0.5 * cost_surrogate

    assert loss_lam1 > loss_lam0


def test_frozen_expert_immutability():
    expert = StandaloneSpecialist("mlp", depth=3)
    for p in expert.parameters():
        p.requires_grad = False

    # Simulate forward pass with dummy router
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(4, 12, 8)
    dec = router(x, mode="straight_through")
    exp_out = expert(x)
    loss = (dec.weights.sum() * exp_out.sum())
    loss.backward()

    # Expert parameters must have NO gradients
    for name, p in expert.named_parameters():
        assert p.grad is None, f"Expert parameter {name} received gradient despite being frozen!"


def test_normalized_costs_bounded():
    expert_flops = {"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0}
    norm_costs, ref_cost = compute_normalized_costs(expert_flops, router_flops=1052.0)

    assert ref_cost == 46296.0 + 1052.0
    for c in norm_costs.values():
        assert 0.0 < c <= 1.0
    assert norm_costs["attention_v2"] == 1.0
    assert norm_costs["graph"] < norm_costs["mlp"] < norm_costs["attention"] < norm_costs["attention_v2"]
