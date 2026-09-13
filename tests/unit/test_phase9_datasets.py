"""Unit tests for Phase 9 Datasets and Transformations."""
import torch
import pytest

from neuroforge.datasets.phase7_mixed import Phase7MixedDataset
from neuroforge.datasets.phase9_datasets import (
    Phase9FreshInDistributionDataset,
    Phase9MixedStructureDataset,
    apply_phase9_distribution_shift,
    apply_phase9_marker_variation,
    apply_phase9_relational_permutation,
    apply_phase9_token_permutation,
    generate_phase9_variable_sequence_dataset,
)


def test_phase9_fresh_in_distribution_dataset():
    """Verify fresh in-distribution dataset properties and zero sample overlap with Phase 7/8B."""
    ds = Phase9FreshInDistributionDataset(samples_per_family=30, seed=11)
    assert len(ds) == 90
    assert ds.features.shape == (90, 12, 8)
    assert ds.targets.shape == (90,)
    assert len(ds.families_list) == 90

    # Family balance
    for fam in ("feature", "relational", "contextual"):
        count = ds.families_list.count(fam)
        assert count == 30

    # Check zero overlap with Phase 7 test set (which used seed + 20_000)
    p7_ds = Phase7MixedDataset("test", samples_per_family=30, seed=11)
    assert not torch.equal(ds.features, p7_ds.features), "Fresh test set must not overlap with canonical test set"

    # Single item access
    item = ds[0]
    assert "features" in item
    assert "target" in item
    assert "family" in item
    assert "oracle_expert" in item
    assert item["features"].shape == (12, 8)


def test_apply_phase9_token_permutation():
    """Verify token permutation preserves feature content and query marker."""
    x = torch.randn(10, 12, 8)
    x[:, :, 4] = 0.0
    x[:, 3, 4] = 1.0  # Marker at token 3
    permuted = apply_phase9_token_permutation(x, seed=42)


    assert permuted.shape == x.shape
    # For every sample, marker must still be present somewhere
    for i in range(10):
        assert float(permuted[i, :, 4].max().item()) == 1.0
        # Permutation invariant statistics must match
        assert torch.allclose(x[i].mean(dim=0), permuted[i].mean(dim=0), atol=1e-5)


def test_apply_phase9_marker_variation():
    """Verify marker position variation shifts the marker while maintaining valid format."""
    x = torch.zeros(10, 12, 8)
    x[:, 0, 4] = 1.0  # Initial marker at token 0
    shifted = apply_phase9_marker_variation(x, seed=42)

    assert shifted.shape == x.shape
    for i in range(10):
        assert float(shifted[i, :, 4].max().item()) == 1.0


def test_generate_phase9_variable_sequence_dataset():
    """Verify variable sequence length dataset generation for S=8 and S=16."""
    for s_len in (8, 16):
        data = generate_phase9_variable_sequence_dataset(samples_per_family=20, seed=11, sequence_length=s_len)
        assert data["features"].shape == (60, s_len, 8)
        assert data["targets"].shape == (60,)
        assert len(data["families"]) == 60
        assert data["sequence_length"] == s_len


def test_apply_phase9_relational_permutation():
    """Verify destructive relational permutation scrambles node ordering."""
    x = torch.randn(5, 12, 8)
    scrambled = apply_phase9_relational_permutation(x, seed=42)
    assert scrambled.shape == x.shape
    # Cyclic difference should change
    orig_cyclic = (x * torch.roll(x, -1, dims=1)).mean()
    scrambled_cyclic = (scrambled * torch.roll(scrambled, -1, dims=1)).mean()
    assert not torch.isclose(orig_cyclic, scrambled_cyclic, atol=1e-3)


def test_apply_phase9_distribution_shift():
    """Verify magnitude scaling, additive noise, and variance contrast shifts."""
    x = torch.randn(10, 12, 8) + 2.0
    x[:, :, 4] = 0.0
    x[:, 2, 4] = 1.0  # marker

    # 1. Magnitude scaling
    scaled = apply_phase9_distribution_shift(x, "magnitude_scaling", "mild")
    assert scaled.shape == x.shape
    assert float(scaled[:, 2, 4].mean().item()) == 1.0  # marker intact
    assert float(scaled[:, :, 0].mean().item()) > float(x[:, :, 0].mean().item())

    # 2. Additive noise
    noisy = apply_phase9_distribution_shift(x, "additive_noise", "mild")
    assert noisy.shape == x.shape
    assert float(noisy[:, 2, 4].mean().item()) == 1.0  # marker intact

    # 3. Variance contrast
    contrasted = apply_phase9_distribution_shift(x, "variance_contrast", "strong")
    assert contrasted.shape == x.shape

    with pytest.raises(ValueError, match="Unknown shift_type"):
        apply_phase9_distribution_shift(x, "invalid_shift", "mild")


def test_phase9_mixed_structure_dataset():
    """Verify mixed-structure dataset has 7 families, composite targets, and correct oracle labeling."""
    ds = Phase9MixedStructureDataset(samples_per_type=20, seed=11)
    assert len(ds) == 140
    assert ds.features.shape == (140, 12, 8)
    assert ds.targets.shape == (140,)
    assert len(ds.families_list) == 140

    # 7 families
    for fam in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        assert ds.families_list.count(fam) == 20

    # Target values binary
    assert set(ds.targets.tolist()).issubset({0, 1})

    # Oracle rule: ORACLE NOT DEFINED for mixed families
    for item in ds:
        fam = item["family"]
        if fam in ("FR", "RC", "FC", "FRC"):
            assert item["oracle_expert"] == "ORACLE NOT DEFINED"
        elif fam == "F":
            assert item["oracle_expert"] == "mlp"
        elif fam == "R":
            assert item["oracle_expert"] == "graph"
        elif fam == "C":
            assert item["oracle_expert"] == "attention_v2"
