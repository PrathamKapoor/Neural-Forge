import torch


def test_mixed_dataset_is_seed_deterministic_and_hides_oracle_from_features():
    """Break caught: a seed change controls only randomness, not the task contract."""
    from neuroforge.datasets import MixedStructureDataset

    first = MixedStructureDataset(samples=12, sequence_length=6, input_dim=8, seed=7)
    second = MixedStructureDataset(samples=12, sequence_length=6, input_dim=8, seed=7)

    assert torch.equal(first.features, second.features)
    assert torch.equal(first.targets, second.targets)
    assert first.features.shape == (12, 6, 8)
    assert set(first.oracle_routes.tolist()) == {0, 1, 2}


def test_adapters_preserve_batch_and_advertise_nonzero_cost():
    """Break caught: silently dropping a sample or omitting adapter cost corrupts efficiency metrics."""
    from neuroforge.core.representations import RingGraphAdapter, SequenceToVectorAdapter

    state = torch.randn(4, 6, 8)
    vector = SequenceToVectorAdapter()(state)
    graph = RingGraphAdapter()(state)

    assert vector.values.shape == (4, 8)
    assert graph.node_features.shape == (4, 6, 8)
    assert vector.estimated_cost > 0 and graph.estimated_cost > 0
