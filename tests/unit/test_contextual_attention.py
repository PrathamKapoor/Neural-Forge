import torch


def test_contextual_association_is_deterministic_balanced_permutation_invariant_and_pool_leak_free():
    from neuroforge.datasets.contextual_attention import ContextualAssociationDataset, association_oracle
    data = ContextualAssociationDataset("train", 80, 17, sequence_length=12)
    clone = ContextualAssociationDataset("train", 80, 17, sequence_length=12)
    assert torch.equal(data.features, clone.features)
    assert data.targets.bincount().tolist() == [40, 40]
    assert association_oracle(data.features).equal(data.targets)
    assert association_oracle(data.features[:, torch.randperm(data.features.shape[1]), :]).equal(data.targets)
    assert abs(float(data.features[:, :, 3].mean(1)[data.targets == 1].mean() - data.features[:, :, 3].mean(1)[data.targets == 0].mean())) < .02
    assert ContextualAssociationDataset("test", 20, 17, sequence_length=16).features.shape[1] == 16


def test_contextual_controls_break_association_or_value_but_preserve_shape():
    from neuroforge.datasets.contextual_attention import ContextualAssociationDataset, break_association, shuffle_values
    data = ContextualAssociationDataset("test", 40, 3)
    assert break_association(data.features, 9).shape == data.features.shape
    assert shuffle_values(data.features, 9).shape == data.features.shape
