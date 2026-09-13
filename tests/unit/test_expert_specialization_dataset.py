import torch


def test_phase3_generators_are_deterministic_balanced_and_split_separated():
    from neuroforge.datasets import ExpertSpecializationDataset

    first = ExpertSpecializationDataset("feature", "train", 120, 7)
    second = ExpertSpecializationDataset("feature", "train", 120, 7)
    validation = ExpertSpecializationDataset("feature", "validation", 120, 7)
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.targets, second.targets)
    assert first.features.shape == (120, 12, 8)
    assert first.targets.bincount().tolist() == [60, 60]
    assert not torch.equal(first.features, validation.features)


def test_phase3_controls_preserve_shape_but_change_relevant_correspondence():
    from neuroforge.datasets import ExpertSpecializationDataset, apply_structural_control

    relational = ExpertSpecializationDataset("relational", "test", 60, 9)
    permuted = apply_structural_control(relational.features, "relational", seed=19)
    assert permuted.shape == relational.features.shape
    assert torch.equal(permuted.sort(dim=1).values, relational.features.sort(dim=1).values)
    assert not torch.equal(permuted, relational.features)
    contextual = ExpertSpecializationDataset("contextual", "test", 60, 9)
    broken = apply_structural_control(contextual.features, "contextual", seed=19)
    assert broken.shape == contextual.features.shape
    assert not torch.equal(broken, contextual.features)


def test_contextual_value_channel_does_not_leak_label_through_pooling():
    from neuroforge.datasets import ExpertSpecializationDataset
    data = ExpertSpecializationDataset("contextual", "train", 240, 13)
    pooled = data.features[:, :, 4].mean(1)
    assert abs(float(pooled[data.targets == 1].mean() - pooled[data.targets == 0].mean())) < 0.02
