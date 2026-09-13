"""Phase 14 Readout/Head Diagnostic Metrics.

Additive evaluators. Everything here operates on a frozen JointCo specialist
whose encoder and blocks are not modified. We only vary:

  - the pooling strategy (mean / max / query / mean+query);
  - the head (linear H -> 2 or small nonlinear H -> H -> 2);
  - which internal representation the head reads from
    (scaled_sum, fused_delta, block_output, feat_delta, rel_delta, ctx_delta).

The encoder is FROZEN in all primary diagnostics (Section 19 of the spec).

This module is diagnostic only. The Phase 14 runner uses these functions and
records per-family accuracy, parameter counts, FLOPs, and latency.
"""
from __future__ import annotations

import math
import statistics
from collections import OrderedDict
from itertools import combinations
from typing import Any

import torch
from torch import nn

from neuroforge.datasets.phase9_datasets import (
    Phase9MixedStructureDataset,
    apply_phase9_relational_permutation,
)
from neuroforge.models.phase11_diagnostics import (
    extract_state_after_blocks,
    linear_probe_accuracy_per_class,
)
from neuroforge.models.specialists import StandaloneSpecialist


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

def pool_mean(state: torch.Tensor, raw_features: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool across the sequence dimension. Returns [B, H]."""
    return state.mean(dim=1)



def pool_max(state: torch.Tensor, raw_features: torch.Tensor | None = None) -> torch.Tensor:
    """Element-wise max across the sequence dimension. Returns [B, H]."""
    return state.max(dim=1).values



def pool_query(state: torch.Tensor, raw_features: torch.Tensor | None) -> torch.Tensor:
    """Pool at the query-token position (channel-4 marker). Returns [B, H]."""
    if raw_features is None:
        raise ValueError("pool_query requires raw_features")
    q = raw_features[:, :, 4].argmax(dim=1)
    return state[torch.arange(state.shape[0]), q]



def pool_mean_plus_query(
    state: torch.Tensor, raw_features: torch.Tensor | None
) -> torch.Tensor:
    """Concatenate mean-pool and query-token, then halve to keep dim H.

    To keep the head size comparable, we use (mean + query) / 2 (so output is
    still [B, H]). This is a reduced-dim compromise; we report it explicitly.
    """
    if raw_features is None:
        raise ValueError("pool_mean_plus_query requires raw_features")
    mean = state.mean(dim=1)
    q = raw_features[:, :, 4].argmax(dim=1)
    query = state[torch.arange(state.shape[0]), q]
    return (mean + query) / 2.0


def pool_branch(
    state: torch.Tensor,
    raw_features: torch.Tensor | None,
    branch_name: str,
    expert: StandaloneSpecialist,
) -> torch.Tensor:
    """Mean-pool a SPECIFIC branch's output (only available with `forward_with_intermediates`).

    Returns [B, H]. This function does NOT detach gradients so the head can
    backpropagate through the (frozen-but-still-attached) encoder to the
    branch output. The encoder parameters are not updated (requires_grad=False
    is set on them by the Phase 6 training protocol), but the gradient can
    still flow through the computation graph.
    """
    if not hasattr(expert.blocks[0], "forward_with_intermediates"):
        raise ValueError(
            f"expert block does not expose forward_with_intermediates: {type(expert.blocks[0])}"
        )
    enc = expert.encoder(raw_features)
    inter = expert.blocks[0].forward_with_intermediates(enc, raw_features)
    branch = inter.get(branch_name)
    if branch is None:
        raise ValueError(f"unknown branch {branch_name}")
    return branch.mean(dim=1)


POOLING_REGISTRY = OrderedDict([
    ("P1_query", "query"),
    ("P2_mean", "mean"),
    ("P3_max", "max"),
    ("P4_mean_plus_query", "mean_plus_query"),
])


# ---------------------------------------------------------------------------
# Head variants
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    """Plain Linear(H, 2) head with no nonlinearity."""

    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SmallNonlinearHead(nn.Module):
    """Small nonlinear head: H -> H -> 2 with GELU."""

    def __init__(self, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def head_parameters(head: nn.Module) -> int:
    return sum(p.numel() for p in head.parameters())


def head_flops(head: nn.Module, batch_size: int = 1, seq_len: int = 1) -> float:
    """Rough FLOPs for a 2-layer head on a per-token input."""
    n = 0
    for module in head.modules():
        if isinstance(module, nn.Linear):
            n += 2 * module.in_features * module.out_features * batch_size
    return float(n)


# ---------------------------------------------------------------------------
# Full forward: frozen encoder + (pool, head)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_state(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    """Return the per-position state [B, S, H] after the block, no pooling.

    Uses the existing @torch.no_grad-decorated `extract_state_after_blocks`; safe
    for evaluation but no gradients flow.
    """
    return extract_state_after_blocks(expert, features)


def encode_state_grad(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    """Same as `encode_state` but allows gradients to flow through the encoder
    to the head. Used during head training; the encoder is still FROZEN
    (parameters are not updated) but the head can backpropagate through it.
    """
    expert.eval()  # ensure no dropout etc.
    state = expert.encoder(features)
    if expert.architecture in ("attention_v2", "joint", "joint_co"):
        for block in expert.blocks:
            state, _ = block(state, features)
    else:
        for block in expert.blocks:
            state, _ = block(state)
    return state


def readout_forward(
    expert: StandaloneSpecialist,
    features: torch.Tensor,
    pooling_fn,
    head: nn.Module,
    use_branch_pool: str | None = None,
) -> torch.Tensor:
    """Frozen-encoder + pooling + head forward.

    Args:
        expert: a frozen (eval mode) StandaloneSpecialist.
        features: raw input features [B, S, D].
        pooling_fn: callable(state, raw_features) -> [B, H].
        head: a small nn.Module (LinearHead or SmallNonlinearHead).
        use_branch_pool: if set, pool that branch's output (requires
            forward_with_intermediates); otherwise pool the post-block state.
    """
    expert.eval()
    # IMPORTANT: use encode_state_grad (gradient-preserving) so the head can
    # backpropagate through the frozen encoder. The encoder parameters are
    # still requires_grad=False (frozen) so they are not updated.
    enc = encode_state_grad(expert, features)
    if use_branch_pool is not None:
        pooled = pool_branch(enc, features, use_branch_pool, expert)
    else:
        # For V2/joint/joint_co, the existing head pools at the query position.
        # We allow any pooling function for the diagnostic path.
        pooled = pooling_fn(enc, features)
    return head(pooled)

def train_readout_head(
    head: nn.Module,
    expert: StandaloneSpecialist,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    pooling_fn,
    epochs: int = 10,
    lr: float = 1e-2,
    batch_size: int = 60,
    use_branch_pool: str | None = None,
) -> list[float]:
    """Train a small head on a frozen JointCo encoder. Returns the loss curve.

    The head is trained; the expert is frozen. The spec requires this protocol
    (Section 19): "frozen JointCo encoder + trainable small diagnostic head".
    """
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    expert.eval()
    n = len(train_features)
    curve: list[float] = []
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        running = 0.0
        nbatches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = train_features[idx]
            y = train_targets[idx]
            optimizer.zero_grad()
            logits = readout_forward(expert, x, pooling_fn, head, use_branch_pool=use_branch_pool)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            nbatches += 1
        head.eval()
        curve.append(running / max(1, nbatches))
    return curve


@torch.no_grad()
def evaluate_readout(
    expert: StandaloneSpecialist,
    head: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    pooling_fn,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    use_branch_pool: str | None = None,
) -> dict[str, float]:
    """Evaluate a frozen-encoder + head on the mixed dataset."""
    expert.eval()
    head.eval()
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    logits = readout_forward(expert, feats, pooling_fn, head, use_branch_pool=use_branch_pool)
    preds = logits.argmax(dim=-1)
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["overall"] = float((preds == targets).float().mean().item())
    return out


# ---------------------------------------------------------------------------
# Branch-combination ablations (Section 15, 16)
# ---------------------------------------------------------------------------
def branch_combination_evaluate(
    expert: StandaloneSpecialist,
    eval_ds: Phase9MixedStructureDataset,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    combinations: list[tuple[str, ...]] | None = None,
    head_epochs: int = 10,
    pooling: str = "mean",
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, dict[str, float]]:
    """Train a small head on each branch combination and report per-family accuracy.

    Each combination is a subset of ("feat", "rel", "ctx") indicating which
    branch outputs are concatenated and mean-pooled before the head.
    """
    if combinations is None:
        combinations = [
            ("feat",), ("rel",), ("ctx",),
            ("feat", "rel"), ("feat", "ctx"), ("rel", "ctx"),
            ("feat", "rel", "ctx"),
        ]
    # Pre-extract per-branch outputs and pool them once (frozen).
    expert.eval()
    enc = expert.encoder(eval_ds.features)
    inter = expert.blocks[0].forward_with_intermediates(enc, eval_ds.features)
    branch_pooled = {b: inter[b].mean(dim=1) for b in ("feat_delta", "rel_delta", "ctx_delta")}
    # Re-extract training branch features PER BATCH so the autograd graph is
    # not freed between combos. The encoder is frozen (requires_grad=False)
    # but the head backpropagates through the pooled branches.
    name_map = {"feat": "feat_delta", "rel": "rel_delta", "ctx": "ctx_delta"}
    results: dict[str, dict[str, float]] = {}
    for combo in combinations:
        label = "+".join(combo)
        # Build the head input
        h = len(combo) * 24
        head = LinearHead(h, num_classes=2)
        # Train on the concatenated branch features
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = len(train_features)
        for _ in range(head_epochs):
            head.train()
            perm = torch.randperm(n)
            for i in range(0, n, 60):
                idx = perm[i:i + 60]
                xb = train_features[idx]
                enc_b = expert.encoder(xb)
                inter_b = expert.blocks[0].forward_with_intermediates(enc_b, xb)
                xs = torch.cat([inter_b[name_map[b]].mean(dim=1) for b in combo], dim=-1)
                ys = train_targets[idx]
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(xs), ys)
                loss.backward()
                optimizer.step()
        head.eval()
        # Evaluate
        with torch.no_grad():
            xs_eval = torch.cat([branch_pooled[name_map[b]] for b in combo], dim=-1)
            preds = head(xs_eval).argmax(dim=-1)
        per_fam: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == eval_ds.targets[idx]).float().mean().item())
        per_fam["overall"] = float((preds == eval_ds.targets).float().mean().item())
        results[label] = per_fam
    return results


# ---------------------------------------------------------------------------
# Relational destruction control (Section 21)
# ---------------------------------------------------------------------------
@torch.no_grad()
def relational_destruction_readout(
    expert: StandaloneSpecialist,
    head: nn.Module,
    pooling_fn,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("R", "RC", "FRC"),
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate the (frozen-encoder, trained-head) on original vs relationally-permuted features."""
    out: dict[str, dict[str, float]] = {}
    for label, f_to_eval in (
        ("original", eval_ds.features),
        ("relational_permuted", apply_phase9_relational_permutation(eval_ds.features, seed=seed)),
    ):
        logits = readout_forward(expert, f_to_eval, pooling_fn, head)
        preds = logits.argmax(dim=-1)
        per_fam: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == eval_ds.targets[idx]).float().mean().item())
        out[label] = per_fam
    return out


# ---------------------------------------------------------------------------
# Ceiling and oracle helpers (reused)
# ---------------------------------------------------------------------------
@torch.no_grad()
def single_expert_ceiling(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    ceilings: dict[str, float] = {}
    best_per: dict[str, str] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        best_acc = -1.0
        best_name = ""
        for name, logits in expert_logits_per_name.items():
            preds = logits.argmax(-1)
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_name = name
        ceilings[f] = best_acc
        best_per[f] = best_name
    return {"ceiling_per_family": ceilings, "best_per_family": best_per}


@torch.no_grad()
def oracle_composition(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    k: int,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    names = list(expert_logits_per_name.keys())
    if k > len(names):
        return {"per_family": {}}
    per_fam: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        if k == 1:
            best = -1.0
            for name, logits in expert_logits_per_name.items():
                preds = logits.argmax(-1)
                acc = float((preds[idx] == targets[idx]).float().mean().item())
                if acc > best:
                    best = acc
            per_fam[f] = best
            continue
        best_acc = -1.0
        for combo in combinations(names, k):
            stacked = torch.stack([expert_logits_per_name[n] for n in combo], dim=0)
            combined = stacked.mean(dim=0)
            preds = combined.argmax(dim=-1)
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc = acc
        per_fam[f] = best_acc
    return {"per_family": per_fam}


# ---------------------------------------------------------------------------
# Causal diagnosis and verdict selection
# ---------------------------------------------------------------------------
def build_phase14_causal_diagnosis(
    baseline_r: float,
    baseline_rc: float,
    baseline_frc: float,
    pool_results: dict[str, dict[str, float]],
    head_results: dict[str, dict[str, float]],
    branch_readout_results: dict[str, dict[str, float]],
    branch_combination_results: dict[str, dict[str, float]],
    fusion_results: dict[str, dict[str, float]],
    relational_sensitivity: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """Build the causal-diagnosis table for Phase 14.

    Each entry is ``{status, evidence}`` where status is one of:
    SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED, INCONCLUSIVE, NOT TESTED.
    """
    out: dict[str, dict[str, Any]] = {}

    # 1. POOLING: if changing pooling alone materially improves R/RC/FRC.
    if pool_results:
        # Compare the existing pooling (P1_query) vs the best of P2-P5
        existing_r = pool_results.get("P1_query", {}).get("R", 0.0)
        existing_rc = pool_results.get("P1_query", {}).get("RC", 0.0)
        best_r = max(
            (pool_results.get(k, {}).get("R", 0.0) for k in ("P2_mean", "P3_max", "P4_mean_plus_query")),
            default=existing_r,
        )
        best_rc = max(
            (pool_results.get(k, {}).get("RC", 0.0) for k in ("P2_mean", "P3_max", "P4_mean_plus_query")),
            default=existing_rc,
        )
        r_gain = best_r - existing_r
        rc_gain = best_rc - existing_rc
        if r_gain > 0.05 or rc_gain > 0.05:
            out["POOLING"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Changing only pooling (P1_query -> best of P2-P4) improves R by "
                    f"{r_gain*100:+.1f}pp and RC by {rc_gain*100:+.1f}pp."
                ),
            }
        elif r_gain > 0.01 or rc_gain > 0.01:
            out["POOLING"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Pooling change improves R by {r_gain*100:+.1f}pp and RC by "
                    f"{rc_gain*100:+.1f}pp (marginal)."
                ),
            }
        else:
            out["POOLING"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Changing pooling alone does not improve R ({r_gain*100:+.1f}pp) "
                    f"or RC ({rc_gain*100:+.1f}pp)."
                ),
            }
    else:
        out["POOLING"] = {"status": "NOT TESTED", "evidence": "Pooling experiment not run."}

    # 2. CLASSIFIER_HEAD: if a small nonlinear head improves over linear.
    if head_results:
        lin_r = head_results.get("linear", {}).get("R", 0.0)
        nl_r = head_results.get("small_nonlinear", {}).get("R", 0.0)
        nl_rc = head_results.get("small_nonlinear", {}).get("RC", 0.0)
        lin_rc = head_results.get("linear", {}).get("RC", 0.0)
        nl_n_params = head_results.get("small_nonlinear_params", 0)
        lin_n_params = head_results.get("linear_params", 0)
        gain_r = nl_r - lin_r
        gain_rc = nl_rc - lin_rc
        if gain_r > 0.05 or gain_rc > 0.05:
            out["CLASSIFIER_HEAD"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Small nonlinear head improves R by {gain_r*100:+.1f}pp and RC by "
                    f"{gain_rc*100:+.1f}pp over linear head ({nl_n_params} vs {lin_n_params} params)."
                ),
            }
        elif gain_r > 0.01 or gain_rc > 0.01:
            out["CLASSIFIER_HEAD"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Nonlinear head improves R by {gain_r*100:+.1f}pp and RC by "
                    f"{gain_rc*100:+.1f}pp (modest)."
                ),
            }
        else:
            out["CLASSIFIER_HEAD"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Nonlinear head does not improve over linear "
                    f"(R {gain_r*100:+.1f}pp, RC {gain_rc*100:+.1f}pp)."
                ),
            }
    else:
        out["CLASSIFIER_HEAD"] = {"status": "NOT TESTED", "evidence": "Head experiment not run."}

    # 3. FUSION: if branch-level representations contain complementary
    #    information that the fused representation loses.
    if fusion_results:
        # Compare "fused_delta -> head" vs "(feat+rel+ctx) -> head"
        fused_r = fusion_results.get("fused_delta", {}).get("R", 0.0)
        branches_r = fusion_results.get("branches_concat", {}).get("R", 0.0)
        if branches_r > fused_r + 0.03:
            out["FUSION"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Branch-concat head reads more relational information "
                    f"({branches_r*100:.1f}% on R) than the fused-delta head "
                    f"({fused_r*100:.1f}% on R); the fusion pathway loses "
                    f"{(branches_r - fused_r)*100:+.1f}pp of R accuracy."
                ),
            }
        elif branches_r > fused_r:
            out["FUSION"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Branch-concat head (R={branches_r*100:.1f}%) modestly "
                    f"exceeds fused-delta head (R={fused_r*100:.1f}%)."
                ),
            }
        else:
            out["FUSION"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Fused-delta head (R={fused_r*100:.1f}%) is at least as good as "
                    f"branch-concat head (R={branches_r*100:.1f}%); no fusion loss."
                ),
            }
    else:
        out["FUSION"] = {"status": "NOT TESTED", "evidence": "Fusion experiment not run."}

    # 4. RELATIONAL_BRANCH: if a direct relational-branch readout helps.
    if branch_readout_results:
        rel_only_r = branch_readout_results.get("rel_only", {}).get("R", 0.0)
        rel_only_rc = branch_readout_results.get("rel_only", {}).get("RC", 0.0)
        rel_only_frc = branch_readout_results.get("rel_only", {}).get("FRC", 0.0)
        baseline_avg = (baseline_r + baseline_rc + baseline_frc) / 3
        rel_avg = (rel_only_r + rel_only_rc + rel_only_frc) / 3
        if rel_avg > baseline_avg + 0.05:
            out["RELATIONAL_BRANCH"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Direct relational-branch readout achieves R/RC/FRC mean "
                    f"{rel_avg*100:.1f}% vs baseline head mean {baseline_avg*100:.1f}%; "
                    f"the existing head is losing relational information."
                ),
            }
        elif rel_avg > baseline_avg:
            out["RELATIONAL_BRANCH"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Direct relational-branch readout (mean {rel_avg*100:.1f}%) "
                    f"slightly exceeds baseline head (mean {baseline_avg*100:.1f}%)."
                ),
            }
        else:
            out["RELATIONAL_BRANCH"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Direct relational-branch readout (mean {rel_avg*100:.1f}%) does not "
                    f"exceed baseline head (mean {baseline_avg*100:.1f}%); the relational "
                    f"branch alone is not task-sufficient."
                ),
            }
    else:
        out["RELATIONAL_BRANCH"] = {"status": "NOT TESTED", "evidence": "Not run."}

    # 5. REPRESENTATION_TRANSFER: compare probe accuracy vs final head accuracy.
    r_probe = relational_sensitivity.get("R_probe", 0.0)
    r_acc = baseline_r
    if r_probe > r_acc + 0.05:
        out["REPRESENTATION_TRANSFER"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"R-signal probe = {r_probe*100:.1f}% but baseline R final accuracy = "
                f"{r_acc*100:.1f}%; the existing head fails to use the information."
            ),
        }
    else:
        out["REPRESENTATION_TRANSFER"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"Probe and final R accuracy are close (probe {r_probe*100:.1f}%, "
                f"final {r_acc*100:.1f}%); representation transfer is not the bottleneck."
            ),
        }

    # 6. BENCHMARK: if no condition improves.
    best_combined_r = max(
        baseline_r,
        *[pool_results.get(k, {}).get("R", 0.0) for k in pool_results],
        *[head_results.get(k, {}).get("R", 0.0) for k in ("linear", "small_nonlinear") if isinstance(head_results.get(k, {}), dict)],
        *[branch_readout_results.get(k, {}).get("R", 0.0) for k in branch_readout_results if isinstance(branch_readout_results.get(k, {}), dict)],
    ) if (pool_results or head_results or branch_readout_results) else baseline_r
    if best_combined_r <= baseline_r + 0.02:
        out["BENCHMARK"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"No diagnostic intervention improves R by more than 2pp over "
                f"baseline (best R = {best_combined_r*100:.1f}%, baseline = {baseline_r*100:.1f}%); "
                f"the benchmark may not differentiate between readout strategies."
            ),
        }
    else:
        out["BENCHMARK"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"At least one diagnostic intervention improves R by > 2pp over "
                f"baseline (best R = {best_combined_r*100:.1f}% vs {baseline_r*100:.1f}%)."
            ),
        }

    return out


def select_phase14_verdict(
    causal: dict[str, dict[str, Any]],
    baseline_r: float,
    branch_readout_best_r: float,
    best_pool_r: float,
    best_head_r: float,
) -> tuple[str, str]:
    """Programmatic Phase 14 verdict selection.

    Returns (CASE label, verdict label).
    """
    def st(cat: str) -> str:
        return causal.get(cat, {}).get("status", "INCONCLUSIVE")

    # CASE A: pooling bottleneck.
    if st("POOLING") == "SUPPORTED":
        return ("CASE A", "Pooling is the primary readout bottleneck")
    # CASE B: head capacity bottleneck.
    if st("CLASSIFIER_HEAD") == "SUPPORTED":
        return ("CASE B", "Head capacity is the primary bottleneck")
    # CASE C: fusion bottleneck.
    if st("FUSION") == "SUPPORTED":
        return ("CASE C", "Fusion is the primary bottleneck")
    # CASE D: relational branch bottleneck (no readout can extract enough R).
    if (st("RELATIONAL_BRANCH") in ("NOT SUPPORTED", "PARTIALLY SUPPORTED")
        and branch_readout_best_r <= baseline_r + 0.02
        and best_pool_r <= baseline_r + 0.02
        and best_head_r <= baseline_r + 0.02):
        return ("CASE D", "Relational branch itself requires improvement")
    # CASE E: decodable but not task-sufficient.
    if st("REPRESENTATION_TRANSFER") == "SUPPORTED" and st("POOLING") != "SUPPORTED" and st("CLASSIFIER_HEAD") != "SUPPORTED" and st("FUSION") != "SUPPORTED":
        return ("CASE E", "Decodable but not task-sufficient (information exists but is not usable for the target)")
    # CASE F: multiple interacting bottlenecks.
    n_supported = sum(1 for k, v in causal.items() if v.get("status") == "SUPPORTED")
    if n_supported >= 2:
        return ("CASE F", "Multiple interacting bottlenecks")
    return ("CASE G", "Unresolved or unclassified")
