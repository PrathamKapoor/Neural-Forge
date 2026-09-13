import torch


def test_router_probabilities_sum_to_one_and_compute_is_weighted():
    """Break caught: unnormalized routes make expected-compute penalties meaningless."""
    from neuroforge.routing import SoftRouter

    router = SoftRouter(hidden_dim=8, module_names=("mlp", "graph", "attention"))
    decision = router(torch.randn(5, 6, 8))

    assert torch.allclose(decision.weights.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert decision.expected_module_cost(torch.tensor([1.0, 2.0, 3.0])).shape == (5,)


def test_diagnostics_flags_single_module_collapse():
    """Break caught: collapsed router reports itself as meaningfully adaptive."""
    from neuroforge.routing import routing_diagnostics

    weights = torch.tensor([[0.995, 0.003, 0.002]]).repeat(10, 1)
    report = routing_diagnostics(weights)

    assert report.single_module_collapse
    assert report.utilization["mlp"] > 0.99
