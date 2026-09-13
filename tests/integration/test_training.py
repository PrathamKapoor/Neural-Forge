import torch


def test_adaptive_model_trains_one_finite_step_and_exposes_routes():
    """Break caught: block/router integration either loses gradients or produces non-finite objectives."""
    from neuroforge.models import AdaptiveHybridClassifier

    model = AdaptiveHybridClassifier(input_dim=8, hidden_dim=12, num_classes=2)
    features = torch.randn(10, 6, 8)
    targets = torch.randint(0, 2, (10,))
    output = model(features, routing_mode="soft")
    loss = torch.nn.functional.cross_entropy(output.logits, targets) + 0.01 * output.expected_compute.mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert output.routing_weights.shape == (10, 3)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_fixed_baseline_exposes_tensor_costs_for_compute_comparison():
    """Break caught: baseline compute accounting cannot multiply route weights by a Python list."""
    from neuroforge.models import FixedHybridClassifier

    output = FixedHybridClassifier(input_dim=8, hidden_dim=12, num_classes=2, active_modules=("mlp",))(torch.randn(4, 6, 8))

    assert output.block_costs.shape == (3,)
    assert torch.isfinite(output.expected_compute).all()
