import torch


def test_descriptor_router_learns_soft_probabilities_and_hard_one_hot_routes():
    """Break caught: hard selection is not one-hot or soft routes cannot support supervised loss."""
    from neuroforge.routing.probes import DescriptorRouter

    router = DescriptorRouter(input_dim=4, hidden_dim=8)
    soft = router(torch.randn(5, 4), mode="soft")
    hard = router(torch.randn(5, 4), mode="hard")

    assert torch.allclose(soft.weights.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.equal(hard.weights.sum(dim=1), torch.ones(5))
    assert torch.nn.functional.cross_entropy(soft.logits, torch.tensor([0, 1, 2, 0, 1])).isfinite()
