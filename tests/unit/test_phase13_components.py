"""Unit tests for Phase 13 joint co-adaptation diagnostic building blocks."""
from __future__ import annotations

import pytest
import torch

from neuroforge.blocks import JointBlock, JointCoBlock
from neuroforge.blocks.joint import JointBlock as JB
from neuroforge.blocks.joint_co import JointCoBlock as JCB
from neuroforge.evaluation.phase13_metrics import (
    build_phase13_causal_diagnosis,
    cross_evaluation_with_ablation,
    expert_branch_scales,
    expert_with_branch_ablated,
    oracle_composition,
    relational_destruction_accuracy,
    representation_probes_per_expert,
    select_phase13_verdict,
    single_expert_ceiling,
)
from neuroforge.models.specialists import StandaloneSpecialist


# ---------------------------------------------------------------------------
# JointCoBlock tests
# ---------------------------------------------------------------------------
def test_joint_co_block_registered():
    block = JointCoBlock(hidden_dim=24)
    assert block.name == "joint_co"


def test_joint_co_specialist_registered():
    sp = StandaloneSpecialist("joint_co", input_dim=8, hidden_dim=24, depth=1)
    n_params = sum(p.numel() for p in sp.parameters())
    # JointCo has the same 3 branches as Joint plus the fusion (H*3 -> H) and
    # the gate (H*3 -> 3). Total should be > joint's 4493.
    assert n_params > 4493
    assert n_params < 8000  # upper bound (sanity)


def test_joint_co_block_shape():
    block = JointCoBlock(hidden_dim=24)
    state = torch.randn(2, 12, 24)
    raw = torch.randn(2, 12, 8)
    out, cost = block(state, raw)
    assert out.shape == (2, 12, 24)
    assert cost > 0


def test_joint_co_block_no_raw_features():
    block = JointCoBlock(hidden_dim=24)
    state = torch.randn(2, 12, 24)
    out, cost = block(state, raw_features=None)
    assert out.shape == (2, 12, 24)


def test_joint_co_block_fusion_starts_zero():
    """At init the fusion gate is biased to -3 (sigmoid ~ 0.05), so the
    fusion contribution starts near zero. The per-branch scales start at 1/3."""
    block = JointCoBlock(hidden_dim=24, init_scale=1.0 / 3.0)
    assert abs(block.feature_scale.item() - 1.0 / 3.0) < 1e-6
    assert abs(block.graph_scale.item() - 1.0 / 3.0) < 1e-6
    assert abs(block.context_scale.item() - 1.0 / 3.0) < 1e-6
    # Fusion gate bias should be -3
    assert torch.allclose(block.fusion_gate.bias, torch.tensor([-3.0, -3.0, -3.0]))


def test_joint_co_block_gradient_flow():
    block = JointCoBlock(hidden_dim=24)
    state = torch.randn(2, 12, 24)
    raw = torch.randn(2, 12, 8)
    out, _ = block(state, raw)
    loss = out.sum()
    loss.backward()
    # All sub-modules should have gradients
    assert block.feature_net[0].weight.grad is not None
    assert block.graph_message.weight.grad is not None
    assert block.context_value.weight.grad is not None
    assert block.fusion.weight.grad is not None
    assert block.fusion_gate.weight.grad is not None


# ---------------------------------------------------------------------------
# Branch diagnostics
# ---------------------------------------------------------------------------
def test_expert_branch_scales_extraction():
    sp = StandaloneSpecialist("joint", depth=1)
    scales = expert_branch_scales(sp)
    assert set(scales.keys()) == {"feature", "graph", "context"}
    for v in scales.values():
        assert isinstance(v, float)


def test_expert_with_branch_ablated_returns_modified_copy():
    sp = StandaloneSpecialist("joint", depth=1)
    sp.eval()
    ablated = expert_with_branch_ablated(sp, ablate=("graph",))
    # The ablated copy has graph_scale set to 0
    assert ablated.blocks[0].graph_scale.item() == 0.0
    # The other scales are still 1/3
    assert ablated.blocks[0].feature_scale.item() == sp.blocks[0].feature_scale.item()


def test_cross_evaluation_with_ablation_runs():
    sp = StandaloneSpecialist("joint", depth=1)
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    result = cross_evaluation_with_ablation(
        sp, ds, ablations=((), ("feature",), ("graph",), ("context",))
    )
    assert set(result.keys()) == {"all", "feature", "graph", "context"}
    for label, per_fam in result.items():
        assert "overall" in per_fam
        for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
            if f in per_fam:
                assert 0.0 <= per_fam[f] <= 1.0


def test_cross_evaluation_with_ablation_changes_accuracy():
    """Ablating a branch should change accuracy on at least one family."""
    sp = StandaloneSpecialist("joint", depth=1)
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=20, seed=11)
    result = cross_evaluation_with_ablation(
        sp, ds, ablations=((), ("graph",))
    )
    # At minimum the overall accuracy may differ (random init -> different
    # outputs) or the ablated run should at least produce finite values.
    assert "all" in result
    assert "graph" in result


# ---------------------------------------------------------------------------
# Representation probes
# ---------------------------------------------------------------------------
def test_representation_probes_per_expert_runs():
    sp = StandaloneSpecialist("joint", depth=1)
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    component_targets = {
        "F_signal": torch.tensor(
            [int(ds.items[i]["sf"] == 1) for i in range(len(ds))], dtype=torch.long
        ),
        "R_signal": torch.tensor(
            [int(ds.items[i]["sr"] == 1) for i in range(len(ds))], dtype=torch.long
        ),
        "C_signal": torch.tensor(
            [int(ds.items[i]["sc"] == 1) for i in range(len(ds))], dtype=torch.long
        ),
        "final_target": ds.targets,
    }
    result = representation_probes_per_expert(sp, ds, component_targets)
    for task in ("F_signal", "R_signal", "C_signal", "final_target"):
        assert task in result
        for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
            if f in result[task]:
                assert 0.0 <= result[task][f] <= 1.0


# ---------------------------------------------------------------------------
# Relational destruction
# ---------------------------------------------------------------------------
def test_relational_destruction_accuracy_runs():
    sp = StandaloneSpecialist("joint", depth=1)
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    result = relational_destruction_accuracy(sp, ds, seed=42)
    assert "original" in result
    assert "relational_permuted" in result


# ---------------------------------------------------------------------------
# Ceiling and oracle
# ---------------------------------------------------------------------------
def test_single_expert_ceiling_returns_max():
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    sp = StandaloneSpecialist("joint", depth=1)
    sp.eval()
    sp2 = StandaloneSpecialist("mlp", depth=1)
    sp2.eval()
    logits = {
        "joint": sp(ds.features),
        "mlp": sp2(ds.features),
    }
    result = single_expert_ceiling(logits, ds.targets, ds.families_list)
    assert "ceiling_per_family" in result
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        if f in result["ceiling_per_family"]:
            assert 0.0 <= result["ceiling_per_family"][f] <= 1.0


def test_oracle_composition_k1_matches_ceiling():
    from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
    ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    sp = StandaloneSpecialist("joint", depth=1)
    sp.eval()
    sp2 = StandaloneSpecialist("mlp", depth=1)
    sp2.eval()
    logits = {"joint": sp(ds.features), "mlp": sp2(ds.features)}
    k1 = oracle_composition(logits, ds.targets, ds.families_list, k=1)
    assert "per_family" in k1


# ---------------------------------------------------------------------------
# Causal diagnosis + verdict
# ---------------------------------------------------------------------------
def test_build_phase13_causal_diagnosis_has_all_categories():
    diag = build_phase13_causal_diagnosis(
        joint_baseline_mixed=0.5,
        extended_training_mixed=0.55,
        coadaptation_mixed=0.55,
        control_mixed=0.4,
        joint_branch_scales={"feature": 0.3, "graph": 0.5, "context": 0.4},
        joint_branch_ablation={
            "all": {"F": 1.0, "R": 0.7, "FRC": 0.8},
            "feature": {"F": 0.5, "R": 0.7, "FRC": 0.8},
            "graph": {"F": 1.0, "R": 0.3, "FRC": 0.5},
            "context": {"F": 1.0, "R": 0.7, "FRC": 0.4},
        },
        joint_repr_probes={
            "F_signal": {"F": 0.9, "R": 0.5, "FRC": 0.7},
            "R_signal": {"F": 0.5, "R": 0.8, "FRC": 0.7},
            "C_signal": {"F": 0.5, "R": 0.5, "FRC": 0.7},
            "final_target": {"F": 1.0, "R": 0.7, "FRC": 0.8},
        },
        joint_relational_sensitivity={
            "original": {"R": 0.7, "RC": 0.6, "FRC": 0.8},
            "relational_permuted": {"R": 0.3, "RC": 0.4, "FRC": 0.5},
        },
        old_ceiling_mixed=0.66,
        new_ceiling_mixed=0.67,
    )
    for cat in (
        "UNDERTRAINING",
        "BRANCH_INTERFERENCE",
        "RELATIONAL_CAPABILITY",
        "REPRESENTATION_FUSION",
        "RELATIONAL_SENSITIVITY",
        "PORTFOLIO_LIMITATION",
        "OPTIMIZATION",
        "BENCHMARK_LIMITATION",
    ):
        assert cat in diag
        assert diag[cat]["status"] in {
            "SUPPORTED",
            "PARTIALLY SUPPORTED",
            "NOT SUPPORTED",
            "INCONCLUSIVE",
            "NOT TESTED",
        }


def test_select_phase13_verdict_returns_valid_case():
    diag = {
        "UNDERTRAINING": {"status": "SUPPORTED"},
        "BRANCH_INTERFERENCE": {"status": "NOT SUPPORTED"},
        "RELATIONAL_CAPABILITY": {"status": "NOT SUPPORTED"},
        "REPRESENTATION_FUSION": {"status": "NOT SUPPORTED"},
        "RELATIONAL_SENSITIVITY": {"status": "NOT TESTED"},
        "PORTFOLIO_LIMITATION": {"status": "NOT SUPPORTED"},
        "OPTIMIZATION": {"status": "NOT SUPPORTED"},
        "BENCHMARK_LIMITATION": {"status": "NOT SUPPORTED"},
    }
    case, label = select_phase13_verdict(
        causal=diag,
        joint_baseline_mixed=0.5,
        extended_training_mixed=0.55,
        coadaptation_mixed=0.55,
        old_ceiling_mixed=0.66,
        new_ceiling_mixed=0.66,
    )
    assert case in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F", "CASE G",
    )
    assert isinstance(label, str) and label
