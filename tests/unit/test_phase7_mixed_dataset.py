import torch
from neuroforge.datasets import Phase7MixedDataset


def test_phase7_mixed_dataset_balance_and_shapes():
    ds = Phase7MixedDataset(split="train", samples_per_family=30, seed=42)
    assert len(ds) == 90
    assert ds.features.shape == (90, 12, 8)
    assert ds.targets.shape == (90,)

    # Exact family balance
    fam_counts = {"feature": 0, "relational": 0, "contextual": 0}
    for item in ds:
        fam_counts[item["family"]] += 1
    assert fam_counts["feature"] == 30
    assert fam_counts["relational"] == 30
    assert fam_counts["contextual"] == 30


def test_phase7_oracle_mapping_and_no_label_dependence():
    ds = Phase7MixedDataset(split="test", samples_per_family=40, seed=123)

    for item in ds:
        fam = item["family"]
        oracle = item["oracle_expert"]
        if fam == "feature":
            assert oracle == "mlp"
        elif fam == "relational":
            assert oracle == "graph"
        elif fam == "contextual":
            assert oracle == "attention_v2"
        else:
            raise AssertionError(f"Unknown family: {fam}")

    # Verify oracle mapping does NOT depend on target label
    targets_for_mlp = [item["target"].item() for item in ds if item["oracle_expert"] == "mlp"]
    assert 0 in targets_for_mlp and 1 in targets_for_mlp


def test_phase7_marker_preservation_and_interleaving():
    ds = Phase7MixedDataset(split="test", samples_per_family=20, seed=999)

    # Marker preservation: exactly one position in channel 4 is >= 0.9
    for i in range(len(ds)):
        ch4 = ds.features[i, :, 4]
        assert (ch4 >= 0.9).sum().item() == 1

    # Interleaving verification: consecutive items should not all be the same family
    families = ds.families_list
    same_consecutive = sum(1 for i in range(len(families) - 1) if families[i] == families[i + 1])
    # In a properly shuffled list of 60 items, consecutive matches should be small (approx 1/3)
    assert same_consecutive < len(families) * 0.7
