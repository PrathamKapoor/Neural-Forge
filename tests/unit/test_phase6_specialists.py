import torch
from neuroforge.blocks import AttentionBlock, AttentionBlockV2, GraphBlock, MLPBlock
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models import StandaloneSpecialist


def test_standalone_specialists_include_attention_v2():
    for arch, block_cls in (
        ("mlp", MLPBlock),
        ("graph", GraphBlock),
        ("attention", AttentionBlock),
        ("attention_v2", AttentionBlockV2),
    ):
        model = StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=1)
        assert isinstance(model.blocks[0], block_cls)
        # Check forward pass preserves [B, 2]
        sample = torch.randn(4, 12, 8)
        sample[:, 0, 4] = 1.0  # marked query
        logits = model(sample)
        assert logits.shape == (4, 2)


def test_capacity_selection_phase6():
    selected = select_capacity_phase6(input_dim=8, hidden_dim=24, max_depth=4)
    assert selected.depths == {"mlp": 3, "graph": 2, "attention": 1, "attention_v2": 3}
    assert selected.parameter_counts == {"mlp": 3914, "graph": 3866, "attention": 3314, "attention_v2": 3914}
    assert selected.max_relative_gap <= 0.16
