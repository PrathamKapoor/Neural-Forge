import torch


def test_standalone_specialists_reuse_primitives_and_share_logits_shape():
    from neuroforge.blocks import AttentionBlock, GraphBlock, MLPBlock
    from neuroforge.models import StandaloneSpecialist

    for architecture, block in (("mlp", MLPBlock), ("graph", GraphBlock), ("attention", AttentionBlock)):
        model = StandaloneSpecialist(architecture, input_dim=8, hidden_dim=24, depth=1)
        assert isinstance(model.blocks[0], block)
        assert model(torch.randn(5, 12, 8)).shape == (5, 2)


def test_capacity_selection_and_specialization_metrics_are_deterministic():
    from neuroforge.evaluation.specialization import select_capacity, specialization_metrics

    selected = select_capacity(input_dim=8, hidden_dim=24, max_depth=4)
    assert selected.max_relative_gap <= 0.20
    assert select_capacity(input_dim=8, hidden_dim=24, max_depth=4) == selected
    result = specialization_metrics({"mlp": [0.8, 0.7, 0.9], "graph": [0.6, 0.85, 0.8], "attention": [0.7, 0.75, 0.95]})
    assert result["winners"] == ["mlp", "graph", "attention"]
    assert result["margins"]["contextual"] == 0.05
