"""Unit tests for Phase 14 readout/head bottleneck diagnostic building blocks."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase14_metrics import (
    LinearHead,
    SmallNonlinearHead,
    branch_combination_evaluate,
    build_phase14_causal_diagnosis,
    encode_state,
    encode_state_grad,
    evaluate_readout,
    head_flops,
    head_parameters,
    pool_max,
    pool_mean,
    pool_mean_plus_query,
    pool_query,
    select_phase14_verdict,
    train_readout_head,
)
from neuroforge.models.phase11_diagnostics import extract_state_after_blocks
from neuroforge.models.specialists import StandaloneSpecialist


# ---------------------------------------------------------------------------
# Pooling tests
# ---------------------------------------------------------------------------
def test_pool_mean_shape():
    state = torch.randn(3, 12, 24)
    out = pool_mean(state)
    assert out.shape == (3, 24)


def test_pool_max_shape():
    state = torch.randn(3, 12, 24)
    out = pool_max(state)
    assert out.shape == (3, 24)


def test_pool_query_shape():
    state = torch.randn(3, 12, 24)
    feats = torch.randn(3, 12, 8)
    # set channel 4 marker for at least one token
    feats[:, 0, 4] = 1.0
    out = pool_query(state, feats)
    assert out.shape == (3, 24)


def test_pool_query_raises_without_raw_features():
    state = torch.randn(3, 12, 24)
    with pytest.raises(ValueError):
        pool_query(state, None)


def test_pool_mean_plus_query_shape():
    state = torch.randn(3, 12, 24)
    feats = torch.randn(3, 12, 8)
    feats[:, 0, 4] = 1.0
    out = pool_mean_plus_query(state, feats)
    assert out.shape == (3, 24)


# ---------------------------------------------------------------------------
# Head tests
# ---------------------------------------------------------------------------
def test_linear_head_shape_and_params():
    h = LinearHead(hidden_dim=24, num_classes=2)
    x = torch.randn(3, 24)
    y = h(x)
    assert y.shape == (3, 2)
    assert head_parameters(h) == 24 * 2 + 2  # 50


def test_small_nonlinear_head_shape_and_params():
    h = SmallNonlinearHead(hidden_dim=24, num_classes=2)
    x = torch.randn(3, 24)
    y = h(x)
    assert y.shape == (3, 2)
    n = head_parameters(h)
    # H*H + H + H*2 + 2 = 576 + 24 + 48 + 2 = 650
    assert n == 24 * 24 + 24 + 24 * 2 + 2


def test_head_flops_scales_with_batch():
    h = LinearHead(24, 2)
    f1 = head_flops(h, batch_size=1)
    f10 = head_flops(h, batch_size=10)
    assert f10 > f1


# ---------------------------------------------------------------------------
# Encode-state tests
# ---------------------------------------------------------------------------
def test_encode_state_returns_block_output():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    x = torch.randn(3, 12, 8)
    with torch.no_grad():
        s1 = encode_state(sp, x)
        s2 = extract_state_after_blocks(sp, x)
    assert torch.allclose(s1, s2, atol=1e-5)


def test_encode_state_grad_preserves_grad():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    x = torch.randn(3, 12, 8)
    s = encode_state_grad(sp, x)
    assert s.requires_grad


# ---------------------------------------------------------------------------
# Train + evaluate tests
# ---------------------------------------------------------------------------
def test_train_readout_head_returns_curve():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    h = LinearHead(24, 2)
    x = torch.randn(20, 12, 8)
    y = torch.tensor([0, 1] * 10, dtype=torch.long)
    curve = train_readout_head(h, sp, x, y, pool_mean, epochs=3, lr=1e-2, batch_size=10)
    assert len(curve) == 3


def test_train_readout_head_with_branch_pool():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    h = LinearHead(24, 2)
    x = torch.randn(20, 12, 8)
    y = torch.tensor([0, 1] * 10, dtype=torch.long)
    curve = train_readout_head(
        h, sp, x, y, None, epochs=2, lr=1e-2, batch_size=10, use_branch_pool="rel_delta"
    )
    assert len(curve) == 2


def test_evaluate_readout_runs():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    h = LinearHead(24, 2)
    # First train the head so it produces meaningful outputs
    x = torch.randn(20, 12, 8)
    y = torch.tensor([0, 1] * 10, dtype=torch.long)
    train_readout_head(h, sp, x, y, pool_mean, epochs=2, lr=1e-2, batch_size=10)
    # Evaluate on a synthetic "dataset" wrapper
    class _DS:
        features = x
        targets = y
        families_list = ["F", "R", "C", "FR", "RC", "FC", "FRC"] * (20 // 7 + 1)
        families_list = families_list[:20]
        items = [{"family": f, "sf": 1, "sr": 1, "sc": 1, "target": t, "features": xx} for f, t, xx in zip(families_list, y, x)]
        def __len__(self): return 20
        def __getitem__(self, i): return self.items[i]
    ds = _DS()
    out = evaluate_readout(sp, h, ds, pool_mean)
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        if f in out:
            assert 0.0 <= out[f] <= 1.0


# ---------------------------------------------------------------------------
# Branch combination
# ---------------------------------------------------------------------------
def test_branch_combination_evaluate_runs():
    sp = StandaloneSpecialist("joint_co", depth=1)
    sp.eval()
    ds = Phase9MixedStructureDataset(samples_per_type=5, seed=11)
    train_x = torch.randn(20, 12, 8)
    train_y = torch.tensor([0, 1] * 10, dtype=torch.long)
    res = branch_combination_evaluate(
        sp, ds, train_x, train_y, head_epochs=2, families=("F", "R", "C", "FR", "RC", "FC", "FRC")
    )
    for combo in ("feat", "rel", "ctx", "feat+rel", "feat+ctx", "rel+ctx", "feat+rel+ctx"):
        assert combo in res
        for fam in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
            if fam in res[combo]:
                assert 0.0 <= res[combo][fam] <= 1.0


# ---------------------------------------------------------------------------
# Causal diagnosis + verdict
# ---------------------------------------------------------------------------
def test_build_phase14_causal_diagnosis_has_all_categories():
    diag = build_phase14_causal_diagnosis(
        baseline_r=0.5, baseline_rc=0.5, baseline_frc=0.8,
        pool_results={
            "P1_query": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "P2_mean": {"R": 0.55, "RC": 0.55, "FRC": 0.8, "overall": 0.65},
        },
        head_results={
            "linear": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "small_nonlinear": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "linear_params": 50,
            "small_nonlinear_params": 650,
        },
        branch_readout_results={
            "rel_only": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "feat_only": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "ctx_only": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
        },
        branch_combination_results={
            "feat": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "rel": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "feat+rel+ctx": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
        },
        fusion_results={
            "fused_delta": {"R": 0.5, "RC": 0.5, "FRC": 0.8, "overall": 0.6},
            "branches_concat": {"R": 0.55, "RC": 0.55, "FRC": 0.8, "overall": 0.65},
        },
        relational_sensitivity={
            "original": {"R": 0.6, "RC": 0.5, "FRC": 0.8},
            "relational_permuted": {"R": 0.5, "RC": 0.5, "FRC": 0.8},
            "R_probe": 0.7,
        },
    )
    for cat in ("POOLING", "CLASSIFIER_HEAD", "FUSION", "RELATIONAL_BRANCH",
                "REPRESENTATION_TRANSFER", "BENCHMARK"):
        assert cat in diag
        assert diag[cat]["status"] in {
            "SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED"
        }


def test_select_phase14_verdict_returns_valid_case():
    diag = {
        "POOLING": {"status": "NOT SUPPORTED"},
        "CLASSIFIER_HEAD": {"status": "NOT SUPPORTED"},
        "FUSION": {"status": "SUPPORTED"},
        "RELATIONAL_BRANCH": {"status": "NOT SUPPORTED"},
        "REPRESENTATION_TRANSFER": {"status": "SUPPORTED"},
        "BENCHMARK": {"status": "NOT SUPPORTED"},
    }
    case, label = select_phase14_verdict(
        causal=diag,
        baseline_r=0.5,
        branch_readout_best_r=0.5,
        best_pool_r=0.5,
        best_head_r=0.5,
    )
    assert case in ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F", "CASE G")
    assert isinstance(label, str) and label


def test_select_phase14_verdict_case_a_pooling_supported():
    diag = {
        "POOLING": {"status": "SUPPORTED"},
        "CLASSIFIER_HEAD": {"status": "NOT SUPPORTED"},
        "FUSION": {"status": "NOT SUPPORTED"},
        "RELATIONAL_BRANCH": {"status": "NOT SUPPORTED"},
        "REPRESENTATION_TRANSFER": {"status": "SUPPORTED"},
        "BENCHMARK": {"status": "NOT SUPPORTED"},
    }
    case, _ = select_phase14_verdict(
        causal=diag, baseline_r=0.5, branch_readout_best_r=0.5,
        best_pool_r=0.6, best_head_r=0.5,
    )
    assert case == "CASE A"


def test_select_phase14_verdict_case_d_relational_branch_bottleneck():
    diag = {
        "POOLING": {"status": "NOT SUPPORTED"},
        "CLASSIFIER_HEAD": {"status": "NOT SUPPORTED"},
        "FUSION": {"status": "NOT SUPPORTED"},
        "RELATIONAL_BRANCH": {"status": "NOT SUPPORTED"},
        "REPRESENTATION_TRANSFER": {"status": "NOT SUPPORTED"},
        "BENCHMARK": {"status": "NOT SUPPORTED"},
    }
    case, _ = select_phase14_verdict(
        causal=diag, baseline_r=0.5, branch_readout_best_r=0.5,
        best_pool_r=0.5, best_head_r=0.5,
    )
    assert case == "CASE D"
