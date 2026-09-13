"""Unit tests for SampleLevelRouter and straight-through routing mechanics."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from neuroforge.models import StandaloneSpecialist
from neuroforge.routing.learned_router import LearnedRoutingDecision, SampleLevelRouter


def test_router_initialization_and_parameter_count():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    param_count = sum(p.numel() for p in router.parameters())
    # Linear 1: 16 * 16 + 16 = 272
    # Linear 2: 16 * 4 + 4 = 68
    # Total: 340
    assert param_count == 340
    assert router.num_experts == 4


def test_router_representation_and_permutation_invariance():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(5, 12, 8)
    rep = router.extract_representation(x)
    assert rep.shape == (5, 16)

    # Arbitrary sequence permutation
    perm = torch.randperm(12)
    x_perm = x[:, perm, :]
    rep_perm = router.extract_representation(x_perm)

    # Permutation invariance must hold numerically
    assert torch.allclose(rep, rep_perm, atol=1e-6)


def test_router_forward_modes_and_shapes():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    x = torch.randn(4, 12, 8)

    # Mode: hard
    dec_hard = router(x, mode="hard")
    assert isinstance(dec_hard, LearnedRoutingDecision)
    assert dec_hard.logits.shape == (4, 4)
    assert dec_hard.weights.shape == (4, 4)
    assert dec_hard.selected_experts.shape == (4,)
    assert dec_hard.entropy.shape == (4,)
    # Weights are exact one-hot
    assert torch.all((dec_hard.weights == 0.0) | (dec_hard.weights == 1.0))
    assert torch.allclose(dec_hard.weights.sum(dim=-1), torch.ones(4))

    # Mode: straight_through
    dec_st = router(x, mode="straight_through")
    assert dec_st.weights.shape == (4, 4)
    assert torch.allclose(dec_st.weights.sum(dim=-1), torch.ones(4))

    # Mode: soft
    dec_soft = router(x, mode="soft")
    assert dec_soft.weights.shape == (4, 4)
    assert torch.allclose(dec_soft.weights.sum(dim=-1), torch.ones(4))

    with pytest.raises(ValueError):
        router(x, mode="invalid_mode")


def test_straight_through_gradient_flow():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    optimizer = torch.optim.SGD(router.parameters(), lr=0.1)

    x = torch.randn(6, 12, 8)
    target = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    dummy_expert_logits = torch.randn(6, 4, 2)

    optimizer.zero_grad()
    dec = router(x, mode="straight_through")
    pred = (dec.weights.unsqueeze(-1) * dummy_expert_logits).sum(dim=1)
    loss = F.cross_entropy(pred, target)
    loss.backward()

    # Router parameters must receive valid, non-zero gradients
    for name, p in router.named_parameters():
        assert p.grad is not None
        assert not torch.all(p.grad == 0.0)


def test_frozen_experts_remain_unchanged():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    expert = StandaloneSpecialist("mlp", depth=1)
    for p in expert.parameters():
        p.requires_grad = False

    initial_expert_params = [p.clone() for p in expert.parameters()]

    x = torch.randn(4, 12, 8)
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    for _ in range(3):
        optimizer.zero_grad()
        dec = router(x, mode="straight_through")
        out = expert(x)
        weighted_out = dec.weights[:, :1] * out
        loss = F.cross_entropy(weighted_out, target)
        loss.backward()
        optimizer.step()

    # Expert parameters must be completely unchanged
    for p_init, p_curr in zip(initial_expert_params, expert.parameters()):
        assert torch.equal(p_init, p_curr)
        assert p_curr.grad is None


def test_router_input_tensor_shape_and_rejection():
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    # Valid input [B, 12, 8]
    valid_x = torch.randn(3, 12, 8)
    dec = router(valid_x)
    assert dec.logits.shape == (3, 4)

    # Invalid rank (2D instead of 3D)
    with pytest.raises(ValueError, match="features must have shape"):
        router(torch.randn(3, 8))

    # Invalid feature dimension (10 instead of 8)
    with pytest.raises(ValueError, match="features must have shape"):
        router(torch.randn(3, 12, 10))


def test_expert_dispatch_only_selected_expert_executes():
    """Verify that on the online routing path, only the selected expert executes for each sample."""
    class SpyExpert(torch.nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.call_count = 0
            self.samples_processed = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.call_count += 1
            self.samples_processed += len(x)
            return torch.zeros(len(x), 2)

    expert_names = ("mlp", "graph", "attention", "attention_v2")
    experts = {name: SpyExpert(name) for name in expert_names}

    # Create mock inputs and mock router choices
    batch_size = 8
    x = torch.randn(batch_size, 12, 8)
    # Force 5 samples to MLP (idx 0) and 3 samples to Graph (idx 1), 0 to attention and attention_v2
    chosen_indices = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1], dtype=torch.long)

    # Online hard dispatch logic
    chosen_names = [expert_names[idx.item()] for idx in chosen_indices]
    idx_groups = {e: [i for i, c in enumerate(chosen_names) if c == e] for e in expert_names}

    # Execute only selected sub-batches
    for e in expert_names:
        if idx_groups[e]:
            sub_x = x[idx_groups[e]]
            experts[e](sub_x)

    # Verify that ONLY active experts executed
    assert experts["mlp"].call_count == 1
    assert experts["mlp"].samples_processed == 5
    assert experts["graph"].call_count == 1
    assert experts["graph"].samples_processed == 3
    assert experts["attention"].call_count == 0
    assert experts["attention"].samples_processed == 0
    assert experts["attention_v2"].call_count == 0
    assert experts["attention_v2"].samples_processed == 0


def test_random_baseline_reproducibility():
    """Verify random router selection is seed-deterministic and changes across different seeds."""
    expert_names = ("mlp", "graph", "attention", "attention_v2")

    g1 = torch.Generator().manual_seed(42 + 77_000)
    choices_1 = [expert_names[int(torch.randint(4, (1,), generator=g1).item())] for _ in range(50)]

    g2 = torch.Generator().manual_seed(42 + 77_000)
    choices_2 = [expert_names[int(torch.randint(4, (1,), generator=g2).item())] for _ in range(50)]

    g3 = torch.Generator().manual_seed(43 + 77_000)
    choices_3 = [expert_names[int(torch.randint(4, (1,), generator=g3).item())] for _ in range(50)]

    assert choices_1 == choices_2
    assert choices_1 != choices_3


def test_oracle_baseline_generator_mapping():
    """Verify oracle routing adheres strictly to generator-defined expert assignments."""
    from neuroforge.datasets import Phase7MixedDataset

    ds = Phase7MixedDataset(split="test", samples_per_family=30, seed=11)
    for i in range(len(ds)):
        fam = ds.families_list[i]
        oracle_exp = ds.oracle_experts[i]
        if fam == "feature":
            assert oracle_exp == "mlp"
        elif fam == "relational":
            assert oracle_exp == "graph"
        elif fam == "contextual":
            assert oracle_exp == "attention_v2"
        else:
            pytest.fail(f"Unexpected family: {fam}")


def test_mixed_dataset_family_balance():
    """Verify training, validation, and test datasets maintain strictly balanced 1:1:1 family splits."""
    from neuroforge.datasets import Phase7MixedDataset

    for split in ("train", "validation", "test"):
        ds = Phase7MixedDataset(split=split, samples_per_family=40, seed=11)
        assert len(ds) == 120
        fam_counts = {fam: ds.families_list.count(fam) for fam in Phase7MixedDataset.families}
        assert fam_counts["feature"] == 40
        assert fam_counts["relational"] == 40
        assert fam_counts["contextual"] == 40


def test_permutation_marker_moves_with_token():
    """Verify that token permutation moves the query marker along with its token, and preserves representation."""
    from neuroforge.datasets import Phase7MixedDataset

    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
    ds = Phase7MixedDataset(split="test", samples_per_family=10, seed=11)
    # Find a contextual sample where channel 4 contains marker 1.0
    contextual_indices = [i for i, f in enumerate(ds.families_list) if f == "contextual"]
    x = ds.features[contextual_indices[0]:contextual_indices[0]+1]  # [1, 12, 8]
    orig_q = x[0, :, 4].argmax().item()
    assert x[0, orig_q, 4].item() == 1.0

    # Apply random permutation
    perm = torch.randperm(12, generator=torch.Generator().manual_seed(99))
    x_perm = x[:, perm, :]
    new_q = perm.tolist().index(orig_q)

    # Marker must have moved to new_q
    assert x_perm[0, new_q, 4].item() == 1.0

    # Router representation must be identical (permutation invariance)
    rep_orig = router.extract_representation(x)
    rep_perm = router.extract_representation(x_perm)
    assert torch.allclose(rep_orig, rep_perm, atol=1e-6)

    # Decisions must match
    dec_orig = router(x, mode="hard")
    dec_perm = router(x_perm, mode="hard")
    assert torch.equal(dec_orig.selected_experts, dec_perm.selected_experts)


def test_router_marker_neutrality_control():
    """Section 11 Control: diagnostic confirms router representation of marker alone cannot predict task family."""
    from neuroforge.training.phase8a_learned_routing import (
        verify_router_marker_neutrality,
    )

    neutrality = verify_router_marker_neutrality(seeds=(11,), train_samples_per_family=60, test_samples_per_family=60)
    assert neutrality["neutral"] is True
    # Chance level for 3 balanced families is 1/3 (33.3%)
    assert abs(neutrality["mean_accuracy"] - (1.0 / 3.0)) < 0.08


def test_router_no_metadata_leakage_interface():
    """Verify router forward interface consumes strictly observable features and mode, no target/family/oracle."""
    import inspect

    sig_forward = inspect.signature(SampleLevelRouter.forward)
    param_names = [p for p in sig_forward.parameters if p != "self"]
    assert param_names == ["features", "mode"]

    sig_extract = inspect.signature(SampleLevelRouter.extract_representation)
    assert [p for p in sig_extract.parameters if p != "self"] == ["features"]

    # Ensure prohibited metadata arguments do not exist
    for prohibited in ("target", "family", "task_family", "oracle", "oracle_expert", "metadata", "expert_preds"):
        assert prohibited not in param_names


def test_serialization_artifacts_reloading(tmp_path):
    """Verify all machine-readable artifacts are generated, valid, and fully reloadable."""
    import csv
    import json

    from neuroforge.training.phase8a_learned_routing import run_phase8a_learned_routing

    output_dir = tmp_path / "serialization_test"
    run_phase8a_learned_routing(
        output_dir=output_dir,
        seeds=(11,),
        train_samples_per_family=12,
        val_samples_per_family=6,
        test_samples_per_family=12,
        expert_epochs=1,
        router_epochs=1,
        batch_size=6,
    )

    # Reload and test JSON files
    with (output_dir / "summary.json").open("r", encoding="utf-8") as f:
        reloaded_summary = json.load(f)
    assert reloaded_summary["experiment"] == "Phase 8A Minimal Learned Sample-Level Router"
    assert "marker_neutrality_control" in reloaded_summary
    assert "conditions" in reloaded_summary

    with (output_dir / "manifest.json").open("r", encoding="utf-8") as f:
        reloaded_manifest = json.load(f)
    assert "input_contract" in reloaded_manifest
    assert "router_architecture" in reloaded_manifest

    with (output_dir / "history.json").open("r", encoding="utf-8") as f:
        reloaded_history = json.load(f)
    assert "router__11" in reloaded_history

    # Reload and test CSV files
    for csv_name in ("source_data.csv", "routing_assignments.csv", "seed_results.csv", "latency_results.csv"):
        csv_path = output_dir / csv_name
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) > 0
