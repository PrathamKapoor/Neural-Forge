"""Phase 16 embedded-vs-dedicated relational diagnosis metrics.

Pure evaluators; no Phase 1-15 module is modified. All thresholds are
pre-registered below; H1-H8 and the final CASE are programmatic functions of
executed-experiment data.
"""
from __future__ import annotations

import statistics
from typing import Any

import torch
from torch import nn

from neuroforge.blocks.primitives import GraphBlock
from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.models.phase11_diagnostics import linear_probe_accuracy_per_class
from neuroforge.models.specialists import StandaloneSpecialist


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")
REL_FAMS = ("R", "RC", "FRC")

GAIN_SUPPORTED = 0.03
GAIN_PARTIAL = 0.01
CEILING_SUPPORTED = 0.01
SEED_STD_UNSTABLE = 0.05

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")


# ---------------------------------------------------------------------------
# 16B — relational input equivalence (statistics, no inference)
# ---------------------------------------------------------------------------
@torch.no_grad()
def input_equivalence_stats(
    tensor: torch.Tensor,
    raw_features: torch.Tensor,
    label: str,
) -> dict[str, Any]:
    """Record shape/scale/marker facts about a relational input tensor."""
    flat = tensor.detach().float()
    per_channel_mean = flat.mean(dim=(0, 1))
    per_channel_std = flat.std(dim=(0, 1))
    marker = raw_features[:, :, 4]
    return {
        "label": label,
        "shape": list(tensor.shape),
        "global_mean": float(flat.mean().item()),
        "global_std": float(flat.std().item()),
        "mean_abs": float(flat.abs().mean().item()),
        "per_channel_mean": [float(v) for v in per_channel_mean],
        "per_channel_std": [float(v) for v in per_channel_std],
        "marker_max": float(marker.max().item()),
        "marker_mean": float(marker.mean().item()),
        "marker_positions_match": True,  # same raw batch feeds both paths
    }


def compare_input_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Summarise the material differences between two input representations."""
    std_ratio = (b["global_std"] / max(1e-9, a["global_std"]))
    mean_shift = abs(b["global_mean"] - a["global_mean"])
    chan_mean_l2 = float(
        torch.tensor(a["per_channel_mean"]) .sub(torch.tensor(b["per_channel_mean"])).pow(2).sum().sqrt().item()
    )
    return {
        "shape_equal": a["shape"] == b["shape"],
        "global_std_ratio_b_over_a": std_ratio,
        "global_mean_shift_abs": mean_shift,
        "per_channel_mean_l2": chan_mean_l2,
        "marker_magnitude_equal": abs(a["marker_max"] - b["marker_max"]) < 1e-9,
    }


# ---------------------------------------------------------------------------
# Representation extraction helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def joint_pre_relational(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    """Tensor entering the JointCo relational branch (shared encoder output)."""
    expert.eval()
    return expert.encoder(features)


@torch.no_grad()
def graph_native_input(graph_expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    """Tensor entering the standalone Graph computation (its encoder output)."""
    graph_expert.eval()
    return graph_expert.encoder(features)


@torch.no_grad()
def apply_graph_blocks(
    graph_expert: StandaloneSpecialist, state: torch.Tensor
) -> torch.Tensor:
    """Run the trained standalone Graph blocks on an arbitrary [B,S,H] state."""
    graph_expert.eval()
    h = state
    for block in graph_expert.blocks:
        h, _ = block(h)
    return h


@torch.no_grad()
def embedded_relational_output(
    joint_expert: StandaloneSpecialist, features: torch.Tensor
) -> torch.Tensor:
    """Output of the JointCo relational path (h_K) on the encoder state."""
    joint_expert.eval()
    enc = joint_expert.encoder(features)
    block = joint_expert.blocks[0]
    if hasattr(block, "_relational_delta"):  # JointCoRelationalBlock
        rel, _ = block._relational_delta(enc)
        return rel
    inter = block.forward_with_intermediates(enc, features)  # JointCoBlock
    return inter["rel_delta"]


# ---------------------------------------------------------------------------
# Controlled head protocol (identical for every 16C/16D/16E/16I comparison)
# ---------------------------------------------------------------------------
HEAD_LR = 1e-2
HEAD_WD = 1e-4
HEAD_EPOCHS = 10
HEAD_BATCH = 60


def train_linear_head_on_states(
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    hidden_dim: int,
    epochs: int = HEAD_EPOCHS,
    lr: float = HEAD_LR,
    seed: int = 0,
) -> nn.Linear:
    """Train Linear(H→2) on frozen mean-pooled states.

    Fully deterministic for a given seed and non-invasive: the global RNG
    state is forked and restored, so callers' randomness is unaffected.
    """
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        head = nn.Linear(hidden_dim, 2)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=HEAD_WD)
        n = len(train_states)
        for _ in range(epochs):
            perm = torch.randperm(n)
            head.train()
            for i in range(0, n, HEAD_BATCH):
                idx = perm[i : i + HEAD_BATCH]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(train_states[idx]), train_targets[idx])
                loss.backward()
                opt.step()
        head.eval()
        return head
    finally:
        torch.set_rng_state(rng_state)


@torch.no_grad()
def eval_states_head(
    head: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    families_list: list[str],
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, float]:
    head.eval()
    preds = head(states).argmax(-1)
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(families_list) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["overall"] = float((preds == targets).float().mean().item())
    return out


def train_graph_diagnostic_head(
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    hidden_dim: int = 24,
    epochs: int = HEAD_EPOCHS,
    lr: float = HEAD_LR,
    seed: int = 0,
) -> tuple[GraphBlock, nn.Linear]:
    """Train a fresh GraphBlock + linear head on frozen input states (16D).

    Fully deterministic for a given seed and non-invasive (RNG fork/restore).
    """
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        gblock = GraphBlock(hidden_dim)
        head = nn.Linear(hidden_dim, 2)
        opt = torch.optim.AdamW(
            list(gblock.parameters()) + list(head.parameters()), lr=lr, weight_decay=HEAD_WD
        )
        n = len(train_states)
        for _ in range(epochs):
            perm = torch.randperm(n)
            gblock.train()
            head.train()
            for i in range(0, n, HEAD_BATCH):
                idx = perm[i : i + HEAD_BATCH]
                opt.zero_grad()
                out, _ = gblock(train_states[idx])
                loss = torch.nn.functional.cross_entropy(head(out.mean(dim=1)), train_targets[idx])
                loss.backward()
                opt.step()
        gblock.eval()
        head.eval()
        return gblock, head
    finally:
        torch.set_rng_state(rng_state)


@torch.no_grad()
def eval_graph_diagnostic(
    gblock: GraphBlock,
    head: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    families_list: list[str],
) -> dict[str, float]:
    gblock.eval()
    head.eval()
    out, _ = gblock(states)
    return eval_states_head(head, out.mean(dim=1), targets, families_list)


# ---------------------------------------------------------------------------
# R-signal probe on arbitrary states
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_r_signal(
    state: torch.Tensor,
    eval_ds: Phase9MixedStructureDataset,
    hidden_dim: int = 24,
) -> float:
    """Linear-probe decodability of the R component on R-family samples."""
    idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "R"]
    if not idx:
        return 0.0
    sub = linear_probe_accuracy_per_class(
        state[idx],
        torch.tensor([int(eval_ds.items[i]["sr"] == 1) for i in idx], dtype=torch.long),
        hidden_dim=hidden_dim,
    )
    return sub.get("overall", 0.0)


# ---------------------------------------------------------------------------
# 16G — gradient instrumentation (diagnostic only, never alters optimisation)
# ---------------------------------------------------------------------------
GRAD_GROUPS = ("relational", "feature", "contextual", "fusion", "encoder", "head")


def grad_group_of(param_name: str) -> str:
    n = param_name.lower()
    if "graph_message" in n or "graph_update" in n or ("blocks.0.message" in n) or ("blocks.0.update" in n):
        return "relational"
    if "feature" in n:
        return "feature"
    if "context" in n:
        return "contextual"
    if "fusion" in n:
        return "fusion"
    if "encoder" in n:
        return "encoder"
    return "head"


def record_grad_norms(
    model: nn.Module, probe_x: torch.Tensor, probe_y: torch.Tensor
) -> dict[str, float]:
    """Forward+backward on a fixed probe batch; record per-group grad norms.

    Gradients are discarded afterwards (caller must zero_grad); no optimizer
    step is taken, so the training trajectory is unaffected.
    """
    model.train()
    model.zero_grad()
    logits = model(probe_x)
    loss = torch.nn.functional.cross_entropy(logits, probe_y)
    loss.backward()
    norms: dict[str, float] = {g: 0.0 for g in GRAD_GROUPS}
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        norms[grad_group_of(name)] += float(p.grad.norm().item()) ** 2
    model.zero_grad()
    return {g: float(v**0.5) for g, v in norms.items()}


# ---------------------------------------------------------------------------
# H1-H8 (programmatic)
# ---------------------------------------------------------------------------
def _status_gain(mean_gain: float, n_pos: int, n: int) -> str:
    need = max(2, n - 1)
    if mean_gain >= GAIN_SUPPORTED and n_pos >= need:
        return "SUPPORTED"
    if mean_gain >= GAIN_PARTIAL and n_pos >= need:
        return "PARTIALLY SUPPORTED"
    if mean_gain >= GAIN_PARTIAL:
        return "INCONCLUSIVE"
    return "NOT SUPPORTED"


def build_phase16_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = int(agg.get("n_seeds", 3))

    c = agg.get("computation", {})  # 16C
    if c.get("tested"):
        d = c["graph_on_joint_minus_embedded_R"]
        if d >= GAIN_SUPPORTED and c.get("n_pos", 0) >= max(2, n - 1):
            out["H2"] = {"status": "SUPPORTED",
                         "evidence": f"Graph-on-joint-input exceeds embedded branch by {d * 100:+.1f}pp on R (same input)."}
        elif d >= GAIN_PARTIAL:
            out["H2"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Graph-on-joint-input exceeds embedded branch by {d * 100:+.1f}pp (modest)."}
        else:
            out["H2"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Same-input computation gap {d * 100:+.1f}pp on R; computation is not the differentiator."}
        gap = c.get("native_minus_graph_on_joint_R", 0.0)
        cons = c.get("graph_on_joint_minus_embedded_R", 0.0)
        diag = agg.get("graph_diagnostic", {})
        diag_gap = diag.get("native_minus_diagnostic_R", 1.0) if diag.get("tested") else 1.0
        if abs(cons) < GAIN_PARTIAL and gap > 0.10 and diag_gap > 0.10:
            out["H1"] = {"status": "SUPPORTED",
                         "evidence": f"Both computations weak on joint input (gap {cons * 100:+.1f}pp) while native Graph leads by {gap * 100:+.1f}pp: input suspect."}
        elif gap <= 0.10 or diag_gap <= 0.05:
            out["H1"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Joint input supports Graph computation within {min(gap, diag_gap) * 100:+.1f}pp of native; input is suitable."}
        else:
            out["H1"] = {"status": "INCONCLUSIVE",
                         "evidence": f"Same-input gap {cons * 100:+.1f}pp, native lead {gap * 100:+.1f}pp: ambiguous."}
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Graph-on-joint-input not executed."}
        out["H2"] = {"status": "NOT TESTED", "evidence": "Graph-on-joint-input not executed."}

    co = agg.get("coadaptation", {})  # 16F
    if co.get("tested"):
        g = co.get("relfocused_minus_joint_R", 0.0)
        out["H3"] = {"status": _status_gain(g, co.get("n_pos", 0), n),
                     "evidence": f"Relational-focused training vs joint training on R: {g * 100:+.1f}pp. Rel-first: {co.get('relfirst_minus_joint_R', 0.0) * 100:+.1f}pp."}
    else:
        out["H3"] = {"status": "NOT TESTED", "evidence": "Branch-freeze conditions not executed."}

    st = agg.get("stages", {})  # 16E
    if st.get("tested"):
        pre, post, postf = st["pre_probe_R"], st["post_probe_R"], st["post_fusion_probe_R"]
        nat = st.get("native_input_probe_R", 1.0)
        if pre < 0.65 and (nat - pre) >= 0.10:
            out["H4"] = {"status": "SUPPORTED",
                         "evidence": f"Pre-relational probe {pre * 100:.1f}% vs native-input probe {nat * 100:.1f}%: information lost before."}
        elif pre >= 0.65:
            out["H4"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Pre-relational probe {pre * 100:.1f}%: representation already informative."}
        else:
            out["H4"] = {"status": "INCONCLUSIVE",
                         "evidence": f"Pre-relational probe {pre * 100:.1f}%, native {nat * 100:.1f}%."}
        if postf <= post - 0.05:
            out["H5"] = {"status": "SUPPORTED",
                         "evidence": f"Fusion-stage probe {postf * 100:.1f}% below post-relational {post * 100:.1f}%: loss after computation."}
        elif post >= 0.70 and st.get("final_R", 0.0) <= post - 0.10:
            out["H5"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Post-relational probe {post * 100:.1f}% but final R {st.get('final_R', 0.0) * 100:.1f}%: conversion loss downstream."}
        else:
            out["H5"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Post-relational {post * 100:.1f}%, post-fusion {postf * 100:.1f}%: no post-computation loss."}
    else:
        out["H4"] = {"status": "NOT TESTED", "evidence": "Stage probes not executed."}
        out["H5"] = {"status": "NOT TESTED", "evidence": "Stage probes not executed."}

    comp = agg.get("compositional", {})  # 16K
    if comp.get("tested"):
        rc, frc = comp["RC_gain"], comp["FRC_gain"]
        if rc >= 0.02 and frc >= 0.02:
            out["H6"] = {"status": "SUPPORTED", "evidence": f"RC {rc * 100:+.1f}pp, FRC {frc * 100:+.1f}pp."}
        elif rc >= 0.01 or frc >= 0.01:
            out["H6"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"RC {rc * 100:+.1f}pp, FRC {frc * 100:+.1f}pp (modest)."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED",
                         "evidence": f"RC {rc * 100:+.1f}pp, FRC {frc * 100:+.1f}pp: isolated capability did not transfer."}
    else:
        out["H6"] = {"status": "NOT TESTED", "evidence": "No strongest condition to assess."}

    sens = agg.get("sensitivity", {})  # 16J
    if sens.get("tested"):
        d = sens["candidate_drop_minus_baseline_drop_R"]
        if d >= 0.03:
            out["H7"] = {"status": "SUPPORTED", "evidence": f"Causal sensitivity increases by {d * 100:+.1f}pp on R."}
        elif d > 0:
            out["H7"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Sensitivity increases by {d * 100:+.1f}pp (modest)."}
        else:
            out["H7"] = {"status": "NOT SUPPORTED", "evidence": f"Sensitivity delta {d * 100:+.1f}pp on R."}
    else:
        out["H7"] = {"status": "NOT TESTED", "evidence": "Destruction comparison not executed."}

    ceil = agg.get("ceiling", {})  # 16L/H8
    if ceil.get("tested"):
        d = ceil["new_minus_old_mixed"]
        out["H8"] = {"status": "SUPPORTED" if d > CEILING_SUPPORTED else "NOT SUPPORTED",
                     "evidence": f"Ceiling delta {d * 100:+.1f}pp mixed mean."}
    else:
        out["H8"] = {"status": "NOT TESTED", "evidence": "Portfolio re-evaluation not executed."}
    return out


def select_phase16_case(hy: dict[str, dict[str, Any]], stages: dict[str, Any]) -> tuple[str, str]:
    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    if st("H3") == "SUPPORTED":
        return ("CASE B", "Joint optimization/co-adaptation suppresses relational capability")
    if st("H2") == "SUPPORTED":
        return ("CASE C", "Embedded relational computation is intrinsically weaker than the dedicated Graph computation")
    if st("H1") == "SUPPORTED":
        return ("CASE A", "Input/interface incompatibility is the primary bottleneck")
    if st("H4") == "SUPPORTED":
        return ("CASE D", "Shared representation is insufficient before relational computation")
    pre = stages.get("pre_probe_R", 0.0)
    post = stages.get("post_probe_R", 0.0)
    if max(pre, post) >= 0.70 and st("H6") != "SUPPORTED":
        return ("CASE E", "Relational information exists but remains difficult to convert into task-usable mixed reasoning")
    return ("CASE F", "No sufficiently localized bottleneck; further architecture is not yet justified")


def select_minimal_intervention(
    case: str, hy: dict[str, dict[str, Any]], agg: dict[str, Any]
) -> dict[str, str]:
    if case == "CASE B" and agg.get("coadaptation", {}).get("relfirst_minus_joint_R", 0.0) > 0:
        return {"intervention": "rel_first_warmup",
                "outcome": "Procedural co-adaptation warmup (no architecture change)",
                "detail": "Relational-first then full training improves over joint training; adopt as training protocol."}
    if case == "CASE A":
        return {"intervention": "document_adapter",
                "outcome": "Adapter design documented, no code added",
                "detail": "Input/interface implicated; smallest adapter deferred to a future phase with its own evidence gate."}
    return {"intervention": "none",
            "outcome": "No architectural intervention (localization only)",
            "detail": "Phase 16 is diagnostic; no subsystem earned a structural change."}


def recommendation_for_case(case: str) -> str:
    return {
        "CASE A": "Design the smallest input adapter for the relational branch and gate it on equivalent-input parity.",
        "CASE B": "Adopt relational-first warmup as the JointCo training protocol; test whether RC/FRC follow.",
        "CASE C": "Align the embedded computation with the standalone Graph configuration before adding anything new.",
        "CASE D": "Investigate the shared encoder representation; the relational branch cannot recover what is not there.",
        "CASE E": "Treat mixed composition (RC/FRC) as the open problem; isolated R gains do not transfer.",
        "CASE F": "STOP building architecture; the bottleneck is not localized. Report the boundary.",
    }[case]


def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({"category": category, "failed": bool(failed),
                     "criterion": criterion, "observed": observed})

    c = agg.get("computation", {})
    add("input_mismatch", bool(c.get("tested")) and not c.get("shapes_equal", True),
        "joint vs native input shapes differ", f"shapes_equal={c.get('shapes_equal')}")
    add("computation_gap", bool(c.get("tested")) and c.get("graph_on_joint_minus_embedded_R", 0.0) <= 0.0,
        "graph-on-joint R gain <= 0pp over embedded", f"{c.get('graph_on_joint_minus_embedded_R', 0.0) * 100:+.1f}pp")
    co = agg.get("coadaptation", {})
    add("coadaptation_suppression", bool(co.get("tested")) and co.get("relfocused_minus_joint_R", 0.0) <= -0.02,
        "rel-focused training >2pp worse than joint (suppression inverted)",
        f"{co.get('relfocused_minus_joint_R', 0.0) * 100:+.1f}pp")
    st = agg.get("stages", {})
    add("pre_computation_loss", bool(st.get("tested")) and st.get("pre_probe_R", 1.0) < 0.65,
        "pre-relational probe < 65%", f"{st.get('pre_probe_R', 0.0) * 100:.1f}%")
    add("post_computation_loss", bool(st.get("tested")) and st.get("post_fusion_probe_R", 1.0) <= st.get("post_probe_R", 1.0) - 0.05,
        "post-fusion probe >=5pp below post-relational", f"post {st.get('post_probe_R', 0.0) * 100:.1f}% vs fusion {st.get('post_fusion_probe_R', 0.0) * 100:.1f}%")
    comp = agg.get("compositional", {})
    add("compositional_transfer_failure",
        bool(comp.get("tested")) and comp.get("R_gain", 0.0) >= 0.02
        and comp.get("RC_gain", 0.0) < 0.01 and comp.get("FRC_gain", 0.0) < 0.01,
        "R gain >= 2pp but RC and FRC gains < 1pp",
        f"R {comp.get('R_gain', 0.0) * 100:+.1f}pp, RC {comp.get('RC_gain', 0.0) * 100:+.1f}pp, FRC {comp.get('FRC_gain', 0.0) * 100:+.1f}pp")
    unstable = [f"{s}/{l}" for s, dd in (agg.get("seed_stability") or {}).items() for l, sd in dd.items() if sd >= SEED_STD_UNSTABLE]
    add("seed_instability", len(unstable) > 0, "any R seed-SD >= 5pp", "; ".join(unstable) if unstable else "all stable")
    gr = agg.get("gradients", {})
    if gr.get("status") == "VERIFIED":
        rel_share = gr.get("rel_share_mean", 1.0)
        add("gradient_starvation", rel_share < 0.05,
            "relational grad share < 5% of total", f"{rel_share * 100:.1f}%")
    else:
        add("gradient_starvation", False, "instrumentation " + gr.get("status", "NOT VERIFIED"),
            gr.get("observed", "not verified"))
    lat = agg.get("latency", {})
    lr = lat.get("candidate_total_us", 0.0) / max(1e-9, lat.get("baseline_total_us", 1.0)) if lat.get("tested") else 1.0
    add("latency_regression", bool(lat.get("tested")) and lr > 1.5, "candidate latency > 1.5x baseline", f"{lr:.2f}x")
    return rows
