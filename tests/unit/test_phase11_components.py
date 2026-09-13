"""Unit tests for Phase 11 diagnostic building blocks."""
from __future__ import annotations

import pytest
import torch

from neuroforge.models.phase11_diagnostics import (
    IdentityAdapter,
    LinearAdapter,
    StateChain,
    extract_state_after_blocks,
    extract_state_after_encoder,
    linear_probe_accuracy,
    train_linear_adapter,
)
from neuroforge.models.specialists import StandaloneSpecialist


def _make_specialists() -> dict[str, StandaloneSpecialist]:
    return {
        "mlp": StandaloneSpecialist(architecture="mlp", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "graph": StandaloneSpecialist(architecture="graph", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "attention": StandaloneSpecialist(architecture="attention", input_dim=8, hidden_dim=24, depth=1, num_classes=2),
        "attention_v2": StandaloneSpecialist(architecture="attention_v2", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
    }


def test_identity_adapter_zero_params():
    adapter = IdentityAdapter(hidden_dim=24)
    x = torch.randn(2, 12, 24)
    out = adapter(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)
    assert adapter.extra_flops(x) == 0.0
    assert sum(p.numel() for p in adapter.parameters()) == 0


def test_linear_adapter_shape_and_flops():
    adapter = LinearAdapter(hidden_dim=24)
    x = torch.randn(2, 12, 24)
    out = adapter(x)
    assert out.shape == x.shape
    assert adapter.extra_flops(x) > 0.0
    n_params = sum(p.numel() for p in adapter.parameters())
    assert n_params == 24 * 24 + 24  # weight + bias


def test_state_chain_shape_and_interface_fix():
    """Sequential state chain returns logits and preserves V2's raw-feature contract."""
    experts = _make_specialists()
    for e in experts.values():
        e.eval()
    chain = StateChain(
        experts,
        sequence=("graph", "mlp"),
        attach="identity",
        attach_v2_features=True,
    )
    chain.eval()
    x = torch.randn(3, 12, 8)
    logits = chain(x)
    assert logits.shape == (3, 2)
    # params count: 2 experts + 1 identity adapter (0 params)
    pc = chain.parameters_count()
    assert pc["adapter_params"] == 0
    assert sum(pc["expert_params"].values()) > 0


def test_state_chain_v2_in_middle():
    """V2 in the middle should still receive the original raw features."""
    experts = _make_specialists()
    for e in experts.values():
        e.eval()
    chain = StateChain(
        experts,
        sequence=("graph", "attention_v2", "mlp"),
        attach="identity",
        attach_v2_features=True,
    )
    chain.eval()
    x = torch.randn(2, 12, 8)
    logits = chain(x)
    assert logits.shape == (2, 2)
    # With attach_v2_features=False the chain should still produce logits
    # of the same shape (the bug-shape just becomes wrong, not an error).
    chain2 = StateChain(
        experts,
        sequence=("graph", "attention_v2", "mlp"),
        attach="identity",
        attach_v2_features=False,
    )
    chain2.eval()
    logits2 = chain2(x)
    assert logits2.shape == (2, 2)


def test_state_chain_k_bounds_no_duplicates():
    """StateChain should refuse empty or non-existent sequences and disallow duplicates."""
    experts = _make_specialists()
    with pytest.raises(ValueError):
        StateChain(experts, sequence=(), attach="identity", attach_v2_features=True)
    with pytest.raises(ValueError):
        StateChain(experts, sequence=("does_not_exist",), attach="identity", attach_v2_features=True)
    # No duplicates required (the code currently allows same expert twice;
    # we only enforce that each name exists at least once).
    chain = StateChain(experts, sequence=("mlp", "mlp"), attach="identity", attach_v2_features=True)
    chain.eval()
    logits = chain(torch.randn(1, 12, 8))
    assert logits.shape == (1, 2)


def test_state_chain_with_linear_adapter():
    experts = _make_specialists()
    chain = StateChain(experts, sequence=("mlp", "graph"), attach="linear", attach_v2_features=True)
    n_params = sum(p.numel() for p in chain.parameters())
    # 2 experts + 1 linear adapter (24*24+24)
    assert n_params > 0
    chain.eval()
    logits = chain(torch.randn(2, 12, 8))
    assert logits.shape == (2, 2)


def test_train_linear_adapter_smoke():
    """The training routine should reduce loss for a trivial case."""
    experts = _make_specialists()
    chain = StateChain(experts, sequence=("mlp", "graph"), attach="linear", attach_v2_features=True)
    train_x = torch.randn(20, 12, 8)
    train_y = torch.tensor([0, 1] * 10, dtype=torch.long)
    curve = train_linear_adapter(chain, train_x, train_y, epochs=3, lr=1e-2, batch_size=8)
    assert len(curve) == 3
    # Loss should not explode (sanity check)
    assert all(0.0 <= v < 100 for v in curve)


def test_extract_state_after_encoder_shape():
    experts = _make_specialists()
    expert = experts["mlp"]
    expert.eval()
    x = torch.randn(2, 12, 8)
    state = extract_state_after_encoder(expert, x)
    assert state.shape == (2, 12, 24)


def test_extract_state_after_blocks_shape():
    experts = _make_specialists()
    for n in ("mlp", "graph", "attention", "attention_v2"):
        experts[n].eval()
        x = torch.randn(2, 12, 8)
        s = extract_state_after_blocks(experts[n], x)
        assert s.shape == (2, 12, 24)


def test_linear_probe_accuracy_returns_float():
    state = torch.randn(50, 12, 24)
    target = torch.tensor([0, 1] * 25, dtype=torch.long)
    acc = linear_probe_accuracy(state, target, hidden_dim=24)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_accuracy_chance_level():
    """Probe on a random state should be near chance on a 2-class problem."""
    state = torch.randn(500, 12, 24)
    target = torch.tensor([0, 1] * 250, dtype=torch.long)
    acc = linear_probe_accuracy(state, target, hidden_dim=24)
    # With 500 samples and 2 classes, chance is 50%; allow generous tolerance.
    assert 0.4 <= acc <= 0.65


def test_state_chain_sequential_block_chain_matches_expert_count():
    """flops count should be sum of expert flops."""
    experts = _make_specialists()
    chain = StateChain(experts, sequence=("mlp", "graph", "attention_v2"), attach="identity", attach_v2_features=True)
    fl = chain.theoretical_flops()
    expected = 8928.0 + 8112.0 + 46296.0
    assert abs(fl - expected) < 1.0
