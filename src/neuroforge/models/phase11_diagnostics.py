"""Phase 11 Diagnostic Building Blocks.

This module provides the minimal diagnostic interface required to localize the
Phase 10 composition bottleneck WITHOUT modifying any Phase 1-10 module.

Three diagnostic primitives are provided:

  1. `EncoderOnlyProbe` — extracts intermediate `[B, S, H]` state from any
     `StandaloneSpecialist` after the encoder and (optionally) after N blocks,
     without modifying the expert. This is the state that any future sequential
     composition mechanism would have to consume.

  2. `StateChain` — explicitly composes two or more experts by passing
     intermediate `[B, S, H]` state through a frozen "downstream" expert. To
     make this a controlled diagnostic, the chain supports:
        * `attach=v2` to force passing the original raw features to V2
          (preserves V2's query/key semantics) — this is the **interface fix**
          that Phase 10's `SequentialSpecialistComposition` does NOT do.
        * `attach=identity` to add no adapter.
        * `attach=linear` to add a single linear `H -> H` adapter between
          experts (with parameter count reported for accountability).

  3. `train_linear_adapter` — minimal helper to train a `H -> H` linear adapter
     on the mixed benchmark (used only when evidence from 11B/11C indicates
     interface incompatibility). Kept deliberately small.

No new expert architecture is introduced. No additional neural primitives.
No replacement of Phase 10 modules.

Diagnostic outputs are deliberately interpretable: every method returns
*states* (tensors) and *scalar metrics*; no learned black boxes.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from neuroforge.models.specialists import StandaloneSpecialist


@torch.no_grad()
def extract_state_after_encoder(
    expert: StandaloneSpecialist,
    features: torch.Tensor,
) -> torch.Tensor:
    """Return the state `[B, S, H]` after the shared `encoder` linear layer.

    This is the natural first-stage representation that any future
    composition mechanism would consume.
    """
    return expert.encoder(features)


@torch.no_grad()
def extract_state_after_blocks(
    expert: StandaloneSpecialist,
    features: torch.Tensor,
    n_blocks: int | None = None,
) -> torch.Tensor:
    """Return the state `[B, S, H]` after `n_blocks` blocks of the expert.

    For `attention_v2`, blocks require the raw features (it reads the
    channel-4 marker and channels 0:3 keys), so this function passes the
    raw features alongside the state — preserving V2's interface contract.
    """
    state = expert.encoder(features)
    n = len(expert.blocks) if n_blocks is None else min(n_blocks, len(expert.blocks))
    for i in range(n):
        block = expert.blocks[i]
        if expert.architecture == "attention_v2":
            state, _ = block(state, features)
        else:
            state, _ = block(state)
    return state


class IdentityAdapter(nn.Module):
    """Identity H -> H adapter. Zero parameters, zero cost."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def extra_flops(self, state: torch.Tensor) -> float:
        return 0.0


class LinearAdapter(nn.Module):
    """Single linear H -> H adapter. Reports its parameter count + cost."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.lin(state)

    def extra_flops(self, state: torch.Tensor) -> float:
        # 2 * H * H per token (matmul) plus same per row of the bmm.
        return float(2 * state.shape[-1] * state.shape[-1] * state.shape[1])


class StateChain(nn.Module):
    """Diagnostic sequential composition: pass state from expert A into expert B.

    Args:
        experts: dict of frozen StandaloneSpecialists.
        sequence: ordered expert names, e.g. `("graph", "mlp")` or `("mlp", "v2")`.
        attach: adapter class for the state transition between experts:
            "identity" (default) or "linear" (single linear H -> H, must be
            trained via `train_linear_adapter`).
        attach_v2_features: when True (default), V2 always receives the original
            raw features (preserves its query/key interface). This is the
            **controlled interface fix** that Phase 10's chain omits.

    Notes:
        * All expert parameters remain frozen. Only the optional linear adapter
          (if `attach="linear"`) is trainable.
        * This module does NOT replace `SequentialSpecialistComposition`. It
          exists solely to enable controlled interface experiments (11B-11E).
    """

    EXPERT_FLOPS = {
        "mlp": 8928.0,
        "graph": 8112.0,
        "attention": 39168.0,
        "attention_v2": 46296.0,
    }

    def __init__(
        self,
        experts: dict[str, StandaloneSpecialist],
        sequence: tuple[str, ...],
        attach: str = "identity",
        attach_v2_features: bool = True,
        hidden_dim: int = 24,
    ) -> None:
        super().__init__()
        if not sequence:
            raise ValueError("sequence must be non-empty")
        for name in sequence:
            if name not in experts:
                raise ValueError(f"expert {name!r} not in experts dict")
        self.sequence = tuple(sequence)
        self.attach_v2_features = attach_v2_features
        self.experts = nn.ModuleDict({name: experts[name] for name in sequence})
        # Freeze the chain's experts; we never train them.
        for p in self.experts.parameters():
            p.requires_grad = False

        # One adapter per inter-expert transition (length = len(sequence) - 1).
        if attach == "identity":
            adapters: list[nn.Module] = [
                IdentityAdapter(hidden_dim) for _ in range(len(sequence) - 1)
            ]
        elif attach == "linear":
            adapters = [LinearAdapter(hidden_dim) for _ in range(len(sequence) - 1)]
        else:
            raise ValueError(f"unknown attach: {attach!r}")
        self.adapters = nn.ModuleList(adapters)
        self._attach_name = attach
        self.total_adapter_params = sum(
            sum(p.numel() for p in a.parameters()) for a in adapters
        )

    def parameters_count(self) -> dict[str, int]:
        """Return per-expert and adapter parameter counts (adapters may be 0)."""
        expert_params = {
            name: sum(p.numel() for p in self.experts[name].parameters())
            for name in self.sequence
        }
        adapter_params = self.total_adapter_params
        return {
            "expert_params": expert_params,
            "adapter_params": adapter_params,
            "total_params": sum(expert_params.values()) + adapter_params,
        }

    def theoretical_flops(self) -> float:
        total = sum(self.EXPERT_FLOPS.get(n, 10000.0) for n in self.sequence)
        # adapter cost is negligible (at most 2*H*H*S per transition) — use
        # analytic constant of 2 * 24 * 24 * 12 = 13824 for one linear adapter
        # and 0 for identity.
        for a in self.adapters:
            total += a.extra_flops(torch.zeros(1, 12, 24))
        return float(total)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward through the ordered chain; returns logits `[B, 2]`."""
        b_size = len(features)
        first = self.experts[self.sequence[0]]
        state = first.encoder(features)
        # Run blocks of the first expert as well — to be consistent with how
        # a full standalone specialist consumes input.
        for block in first.blocks:
            if first.architecture == "attention_v2":
                state, _ = block(state, features)
            else:
                state, _ = block(state)

        # Transition through remaining experts.
        for i in range(1, len(self.sequence)):
            name = self.sequence[i]
            exp = self.experts[name]
            # Adapter
            state = self.adapters[i - 1](state)
            # Blocks: V2 needs raw features to locate query / compute keys.
            for block in exp.blocks:
                if exp.architecture == "attention_v2":
                    raw = features if self.attach_v2_features else state
                    state, _ = block(state, raw)
                else:
                    state, _ = block(state)

        # Final head: V2 pools at the query, others mean-pool.
        last = self.experts[self.sequence[-1]]
        if last.architecture == "attention_v2":
            q = features[:, :, 4].argmax(1)
            logits = last.head(state[torch.arange(b_size), q])
        else:
            logits = last.head(state.mean(1))
        return logits


def train_linear_adapter(
    chain: StateChain,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> list[float]:
    """Train only the linear adapter(s) in a `StateChain` (chain experts stay frozen).

    Returns the training-loss curve. Used only when `attach="linear"`.
    """
    # The chain is assumed to have exactly one or more LinearAdapters.
    adapter_params = [p for a in chain.adapters for p in a.parameters() if p.requires_grad]
    if not adapter_params:
        return []
    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=1e-4)
    n = len(train_features)
    curve: list[float] = []
    for _ in range(epochs):
        perm = torch.randperm(n)
        running_loss = 0.0
        n_batches = 0
        chain.train()
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x = train_features[idx]
            y = train_targets[idx]
            optimizer.zero_grad()
            logits = chain(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        chain.eval()
        curve.append(running_loss / max(1, n_batches))
    return curve


@torch.no_grad()
def linear_probe_accuracy(
    state: torch.Tensor,
    target_task: torch.Tensor,
    hidden_dim: int,
    seed: int = 0,
) -> float:
    """Train a small linear probe on `state` to predict `target_task` and return accuracy.

    A simple analytic solution: ridge regression of one-hot target onto
    pooled `state` (no sklearn dependency). This is a DIAGNOSTIC, not a model.
    """
    if state.ndim != 3:
        raise ValueError("state must be [B, S, H]")
    pooled = state.mean(dim=1)  # [B, H]
    b, h = pooled.shape
    if b < 2:
        return 0.0
    # one-hot
    y = target_task.long()
    n_classes = int(y.max().item()) + 1
    Y = torch.zeros(b, n_classes)
    Y[torch.arange(b), y] = 1.0
    # Ridge closed-form: (X^T X + lambda I) W = X^T Y
    lam = 1e-2
    X = pooled
    XtX = X.t() @ X + lam * torch.eye(h)
    XtY = X.t() @ Y
    try:
        W = torch.linalg.solve(XtX, XtY)
    except Exception:
        W = torch.linalg.lstsq(X, Y).solution
    preds = (pooled @ W).argmax(dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def linear_probe_accuracy_per_class(
    state: torch.Tensor,
    target_task: torch.Tensor,
    hidden_dim: int,
) -> dict[str, float]:
    """Variant of `linear_probe_accuracy` that also returns per-class accuracy.

    For binary tasks returns the accuracy on class 0 and class 1 separately
    (because imbalanced classes can mask decodability).
    """
    overall = linear_probe_accuracy(state, state.new_zeros(state.shape[0]), hidden_dim)  # placeholder
    pooled = state.mean(dim=1)
    b, h = pooled.shape
    y = target_task.long()
    n_classes = int(max(int(y.max().item()) + 1, 2))
    Y = torch.zeros(b, n_classes)
    Y[torch.arange(b), y] = 1.0
    lam = 1e-2
    XtX = pooled.t() @ pooled + lam * torch.eye(h)
    XtY = pooled.t() @ Y
    try:
        W = torch.linalg.solve(XtX, XtY)
    except Exception:
        W = torch.linalg.lstsq(pooled, Y).solution
    preds = (pooled @ W).argmax(dim=1)
    overall = float((preds == y).float().mean().item())
    per_class: dict[str, float] = {}
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            per_class[f"class_{c}"] = float((preds[mask] == y[mask]).float().mean().item())
        else:
            per_class[f"class_{c}"] = 0.0
    return {"overall": overall, **per_class}
