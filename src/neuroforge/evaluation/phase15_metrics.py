"""Phase 15 relational-substep diagnostic metrics.

Pure evaluators; do not mutate trained experts and do not modify any Phase
1-14 module. All scientific thresholds are pre-registered as constants below
so the H1-H8 verdicts and the final CASE classification are programmatic
functions of executed-experiment data, never of manual judgement.

Conventions (same as Phases 13/14):
  - accuracies are fractions in [0, 1];
  - gains/drops are absolute percentage-point deltas (fractions);
  - seed aggregation is mean +/- SD (never best-seed selection).
"""
from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import nn

from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock
from neuroforge.datasets.phase9_datasets import (
    Phase9MixedStructureDataset,
    apply_phase9_relational_permutation,
    apply_phase9_token_permutation,
)
from neuroforge.evaluation.phase13_metrics import (
    expert_with_branch_ablated,
    representation_probes_per_expert,
)
from neuroforge.models.specialists import StandaloneSpecialist


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
PURE_FAMS = ("F", "R", "C")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")
REL_FAMS = ("R", "RC", "FRC")

# ---------------------------------------------------------------------------
# Pre-registered decision thresholds (absolute fractions, i.e. 0.03 = 3pp)
# ---------------------------------------------------------------------------
GAIN_SUPPORTED = 0.03
GAIN_PARTIAL = 0.01
GAP_CLOSE_SUPPORTED = 0.03
CEILING_SUPPORTED = 0.01
TOPOLOGY_DROP_SUPPORTED = 0.03
SEED_STD_UNSTABLE = 0.05
LATENCY_REGRESSION_FACTOR = 1.5
PARAM_GROWTH_FLAG = 0.50
BRANCH_INTERFERENCE_DROP = 0.03

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")


# ---------------------------------------------------------------------------
# Basic evaluators
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_family_accuracy(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    features: torch.Tensor | None = None,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, float]:
    """Per-family accuracy of an expert on the mixed benchmark."""
    expert.eval()
    feats = eval_ds.features if features is None else features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    preds = expert(feats).argmax(-1)
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["pure_mean"] = statistics.mean([out.get(f, 0.0) for f in PURE_FAMS])
    out["mixed_mean"] = statistics.mean([out.get(f, 0.0) for f in MIXED_FAMS])
    out["overall"] = float((preds == targets).float().mean().item())
    return out


@torch.no_grad()
def relational_probe_triplet(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    hidden_dim: int = 24,
) -> dict[str, float]:
    """Return probe_R, final_R and the probe-final gap for one expert.

    probe_R: linear-probe decodability of the R component signal on R samples.
    final_R: the expert's own R accuracy.
    gap: probe_R - final_R (positive means decodable-but-unused signal).
    """
    component_targets = {
        "R_signal": torch.tensor(
            [int(eval_ds.items[i]["sr"] == 1) for i in range(len(eval_ds))],
            dtype=torch.long,
        ),
    }
    probes = representation_probes_per_expert(expert, eval_ds, component_targets, hidden_dim=hidden_dim)
    probe_r = probes.get("R_signal", {}).get("R", 0.0)
    final_r = per_family_accuracy(expert, eval_ds).get("R", 0.0)
    return {"probe_R": probe_r, "final_R": final_r, "gap": probe_r - final_r}


@torch.no_grad()
def marker_ablation_accuracy(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("R", "RC", "FRC"),
) -> dict[str, dict[str, float]]:
    """Neutralise the contextual query marker (channel 4 -> 0) and re-evaluate.

    A candidate that relies on marker leakage rather than relational structure
    will collapse under this control.
    """
    neutral = eval_ds.features.clone()
    neutral[:, :, 4] = 0.0
    out: dict[str, dict[str, float]] = {}
    for label, feats in (("original", eval_ds.features), ("marker_neutralised", neutral)):
        out[label] = {
            f: v for f, v in per_family_accuracy(expert, eval_ds, features=feats).items() if f in families
        }
    return out


@torch.no_grad()
def token_permutation_accuracy(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("R", "RC", "FRC"),
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Permute token order (marker moves with its token) and re-evaluate.

    Semantics-preserving permutation: a position-shortcut model degrades,
    a genuinely relational model is comparatively robust.
    """
    permuted = apply_phase9_token_permutation(eval_ds.features, seed=seed)
    out: dict[str, dict[str, float]] = {}
    for label, feats in (("original", eval_ds.features), ("token_permuted", permuted)):
        out[label] = {
            f: v for f, v in per_family_accuracy(expert, eval_ds, features=feats).items() if f in families
        }
    return out


def branch_7way_ablation(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, dict[str, float]]:
    """Evaluate all 7 non-empty branch combinations via scale zeroing.

    Labels use kept branches: "F", "R", "C", "F+R", "R+C", "F+C", "F+R+C".
    "F+R+C" (no ablation) equals the unmodified expert.
    """
    combos: dict[str, tuple[str, ...]] = {
        "F": ("graph", "context"),
        "R": ("feature", "context"),
        "C": ("feature", "graph"),
        "F+R": ("context",),
        "R+C": ("feature",),
        "F+C": ("graph",),
        "F+R+C": (),
    }
    out: dict[str, dict[str, float]] = {}
    for label, ablate in combos.items():
        mod = expert_with_branch_ablated(expert, ablate)
        out[label] = per_family_accuracy(mod, eval_ds, families=families)
    return out


# ---------------------------------------------------------------------------
# Compute / latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def relational_parameters_of(expert: StandaloneSpecialist) -> int:
    """Parameters of the relational substep (message + update linears)."""
    block = expert.blocks[0]
    if isinstance(block, JointCoRelationalBlock):
        return block.relational_param_count()
    params = 0
    for name in ("graph_message", "graph_update", "message", "update"):
        mod = getattr(block, name, None)
        if mod is not None:
            params += sum(p.numel() for p in mod.parameters())
    return params


@torch.no_grad()
def latency_microseconds(model: nn.Module, sample: torch.Tensor, n_runs: int = 20) -> float:
    """Mean CPU microseconds per sample (same methodology as Phases 13/14)."""
    model.eval()
    b_size = len(sample)
    times: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            times.append((t1 - t0) / b_size * 1e6)
    return statistics.mean(times)


def activation_bytes(batch: int, seq_len: int, hidden_dim: int, rel_depth: int) -> int:
    """FP32 bytes for retained relational intermediates (depth-K chain)."""
    return batch * seq_len * hidden_dim * 4 * max(1, rel_depth)


# ---------------------------------------------------------------------------
# Seed aggregation helpers
# ---------------------------------------------------------------------------
def mean_sd(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "sd": 0.0, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _gain_status(
    mean_gain: float,
    n_positive_seeds: int,
    n_seeds: int,
    tested: bool = True,
    not_tested_reason: str = "",
) -> tuple[str, str]:
    """Map (mean gain, seed consistency) to a verdict status + evidence fragment."""
    if not tested:
        return ("NOT TESTED", not_tested_reason or "Experiment gated off by the sequential protocol.")
    need = max(2, n_seeds - 1)  # at least 2/3 seeds must agree in sign
    consistent = n_positive_seeds >= need
    if mean_gain >= GAIN_SUPPORTED and consistent:
        return ("SUPPORTED", f"mean gain {mean_gain * 100:+.1f}pp, positive on {n_positive_seeds}/{n_seeds} seeds.")
    if mean_gain >= GAIN_PARTIAL and consistent:
        return ("PARTIALLY SUPPORTED", f"mean gain {mean_gain * 100:+.1f}pp (modest), positive on {n_positive_seeds}/{n_seeds} seeds.")
    if mean_gain >= GAIN_PARTIAL and not consistent:
        return ("INCONCLUSIVE", f"mean gain {mean_gain * 100:+.1f}pp but seed-inconsistent ({n_positive_seeds}/{n_seeds} positive).")
    return ("NOT SUPPORTED", f"mean gain {mean_gain * 100:+.1f}pp over baseline.")


# ---------------------------------------------------------------------------
# H1-H8 hypothesis verdicts (all programmatic)
# ---------------------------------------------------------------------------
def build_phase15_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build H1-H8 verdicts from cross-seed aggregates.

    Expected keys in `agg`:
      baseline_R_mean, depth (dict label -> {R_mean, RC_mean, per-seed R gains...}),
      aggregation, capacity, destruction drops, probe gaps, ceiling means,
      tested flags. Missing gated-off sections yield NOT TESTED.
    """
    out: dict[str, dict[str, Any]] = {}
    n_seeds = int(agg.get("n_seeds", 3))

    # ---- H1: message-passing depth limitation ----
    depth = agg.get("depth") or {}
    if depth.get("tested"):
        best = depth.get("best_label", "")
        g = depth.get("best_R_gain_mean", 0.0)
        npos = depth.get("best_R_gain_n_positive", 0)
        status, frag = _gain_status(g, npos, n_seeds, True)
        out["H1"] = {
            "status": status,
            "evidence": f"Depth {best} vs depth-1 baseline on R: {frag}",
        }
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Depth ablation not executed."}

    # ---- H2: aggregation limitation ----
    aggr = agg.get("aggregation") or {}
    if aggr.get("tested"):
        best = aggr.get("best_label", "")
        g = aggr.get("best_R_gain_mean", 0.0)
        npos = aggr.get("best_R_gain_n_positive", 0)
        status, frag = _gain_status(g, npos, n_seeds, True)
        out["H2"] = {
            "status": status,
            "evidence": f"Aggregation {best} vs mean baseline on R: {frag}",
        }
    else:
        out["H2"] = {
            "status": "NOT TESTED",
            "evidence": aggr.get("gate_reason", "Aggregation gated off: depth results fully explained the bottleneck."),
        }

    # ---- H3: relational update-capacity limitation ----
    cap = agg.get("capacity") or {}
    if cap.get("tested"):
        g = cap.get("R_gain_mean", 0.0)
        npos = cap.get("R_gain_n_positive", 0)
        status, frag = _gain_status(g, npos, n_seeds, True)
        out["H3"] = {
            "status": status,
            "evidence": (
                f"MLP relational update vs linear update on R: {frag} "
                f"(+{cap.get('param_delta', 0)} params, +{cap.get('flop_delta', 0.0):.0f} FLOPs)."
            ),
        }
    else:
        out["H3"] = {
            "status": "NOT TESTED",
            "evidence": cap.get("gate_reason", "Capacity gated off: depth/aggregation results fully explained the bottleneck."),
        }

    # ---- H4: fixed-topology limitation ----
    topo = agg.get("topology") or {}
    if topo.get("tested"):
        r_gain = topo.get("candidate_R_gain_mean", 0.0)
        drop_delta = topo.get("candidate_drop_minus_baseline_drop_R", 0.0)
        if r_gain >= 0.02 and drop_delta >= TOPOLOGY_DROP_SUPPORTED:
            out["H4"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Candidate improves R by {r_gain * 100:+.1f}pp and shows "
                    f"{drop_delta * 100:+.1f}pp stronger relational-destruction dependence on R."
                ),
            }
        elif drop_delta >= TOPOLOGY_DROP_SUPPORTED:
            out["H4"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Candidate shows {drop_delta * 100:+.1f}pp stronger destruction dependence "
                    f"but R gain is only {r_gain * 100:+.1f}pp (dependence without useful performance)."
                ),
            }
        else:
            out["H4"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Candidate destruction dependence delta {drop_delta * 100:+.1f}pp on R; "
                    f"no evidence of topology-limited dependence."
                ),
            }
    else:
        out["H4"] = {"status": "NOT TESTED", "evidence": "Topology controls not executed."}

    # ---- H5: intervention improves RC/FRC ----
    comp = agg.get("compositional") or {}
    if comp.get("tested"):
        rc_g = comp.get("RC_gain_mean", 0.0)
        frc_g = comp.get("FRC_gain_mean", 0.0)
        if rc_g >= 0.02 and frc_g >= 0.02:
            out["H5"] = {"status": "SUPPORTED", "evidence": f"RC {rc_g * 100:+.1f}pp, FRC {frc_g * 100:+.1f}pp."}
        elif rc_g >= 0.01 or frc_g >= 0.01:
            out["H5"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"RC {rc_g * 100:+.1f}pp, FRC {frc_g * 100:+.1f}pp (modest)."}
        else:
            out["H5"] = {
                "status": "NOT SUPPORTED",
                "evidence": f"RC {rc_g * 100:+.1f}pp, FRC {frc_g * 100:+.1f}pp: isolated R gains did not transfer compositionally.",
            }
    else:
        out["H5"] = {"status": "NOT TESTED", "evidence": "No candidate to assess compositional transfer."}

    # ---- H6: improvement is specifically relational ----
    spec = agg.get("specificity") or {}
    if spec.get("tested"):
        rel = spec.get("rel_gain_mean", 0.0)
        non = spec.get("nonrel_gain_mean", 0.0)
        diff = rel - non
        if diff >= 0.02 and rel > 0:
            out["H6"] = {"status": "SUPPORTED", "evidence": f"Relational-family gain {rel * 100:+.1f}pp exceeds F/C gain {non * 100:+.1f}pp."}
        elif rel > non:
            out["H6"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Relational gain {rel * 100:+.1f}pp vs F/C gain {non * 100:+.1f}pp (weak separation)."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED", "evidence": f"Gain is generic capacity, not relational: rel {rel * 100:+.1f}pp vs F/C {non * 100:+.1f}pp."}
    else:
        out["H6"] = {"status": "NOT TESTED", "evidence": "No candidate to assess specificity."}

    # ---- H7: representation-to-prediction gap decreases ----
    gap = agg.get("gap") or {}
    if gap.get("tested"):
        close = gap.get("gap_close_mean", 0.0)
        if close >= GAP_CLOSE_SUPPORTED:
            out["H7"] = {"status": "SUPPORTED", "evidence": f"Probe-final gap closes by {close * 100:+.1f}pp."}
        elif close > 0:
            out["H7"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Gap closes by {close * 100:+.1f}pp (modest)."}
        else:
            out["H7"] = {"status": "NOT SUPPORTED", "evidence": f"Gap does not close ({close * 100:+.1f}pp)."}
    else:
        out["H7"] = {"status": "NOT TESTED", "evidence": "Probe reassessment not executed."}

    # ---- H8: improved JointCo raises the empirical single-expert ceiling ----
    ceil = agg.get("ceiling") or {}
    if ceil.get("tested"):
        delta = ceil.get("new_minus_old_mixed", 0.0)
        if delta > CEILING_SUPPORTED:
            out["H8"] = {"status": "SUPPORTED", "evidence": f"New ceiling exceeds old by {delta * 100:+.1f}pp mixed mean."}
        else:
            out["H8"] = {"status": "NOT SUPPORTED", "evidence": f"Ceiling delta {delta * 100:+.1f}pp mixed mean."}
    else:
        out["H8"] = {"status": "NOT TESTED", "evidence": "Portfolio re-evaluation not executed."}

    return out


def select_phase15_case(
    hypotheses: dict[str, dict[str, Any]],
    baseline_gap: float,
    candidate_r_gain: float,
) -> tuple[str, str]:
    """Programmatic final CASE classification for Phase 15."""
    def st(h: str) -> str:
        return hypotheses.get(h, {}).get("status", "INCONCLUSIVE")

    if baseline_gap < 0.05:
        return ("CASE A", "Existing relational computation is sufficient; weakness originated elsewhere")
    if st("H1") in ("SUPPORTED", "PARTIALLY SUPPORTED") or st("H2") in ("SUPPORTED", "PARTIALLY SUPPORTED") or st("H3") in ("SUPPORTED", "PARTIALLY SUPPORTED"):
        if st("H8") == "SUPPORTED" and candidate_r_gain > 0:
            return ("CASE F", "A minimal relational intervention produces a validated new expert capability")
    if st("H1") == "SUPPORTED":
        return ("CASE B", "Additional relational propagation is sufficient")
    if st("H2") == "SUPPORTED" or st("H3") == "SUPPORTED":
        return ("CASE C", "Relational aggregation/update capacity is limiting")
    if st("H4") == "SUPPORTED" and st("H1") not in ("SUPPORTED", "PARTIALLY SUPPORTED") and st("H2") not in ("SUPPORTED", "PARTIALLY SUPPORTED") and st("H3") not in ("SUPPORTED", "PARTIALLY SUPPORTED"):
        return ("CASE D", "Fixed topology is limiting")
    return ("CASE E", "Relational bottleneck remains unresolved")


def select_minimal_intervention(agg: dict[str, Any]) -> dict[str, str]:
    """Select the single minimal validated intervention (15J), programmatically.

    Preference order: smallest useful depth increase, then aggregation, then
    capacity. Topology yields a document-only outcome. Otherwise Outcome E.
    """
    depth = agg.get("depth") or {}
    if depth.get("tested"):
        for label in ("depth2", "depth3", "depth4"):
            d = depth.get("variants", {}).get(label)
            if d is not None and d.get("status") in ("SUPPORTED", "PARTIALLY SUPPORTED") and d.get("R_gain_mean", 0.0) > 0:
                return {"intervention": label, "outcome": "Outcome A (depth)", "detail": f"smallest useful depth increase: {label}"}
    aggr = agg.get("aggregation") or {}
    if aggr.get("tested"):
        b = aggr.get("best_label", "")
        if aggr.get("best_status") in ("SUPPORTED", "PARTIALLY SUPPORTED") and aggr.get("best_R_gain_mean", 0.0) > 0:
            return {"intervention": f"aggregation_{b}", "outcome": "Outcome B (aggregation)", "detail": f"smallest supported aggregation change: {b}"}
    cap = agg.get("capacity") or {}
    if cap.get("tested") and cap.get("status") in ("SUPPORTED", "PARTIALLY SUPPORTED") and cap.get("R_gain_mean", 0.0) > 0:
        return {"intervention": "rel_update_mlp", "outcome": "Outcome C (capacity)", "detail": "smallest supported capacity increase: mlp update"}
    topo = agg.get("topology") or {}
    if (topo.get("tested") and agg.get("hypotheses", {}).get("H4", {}).get("status") == "SUPPORTED"):
        return {"intervention": "document_topology", "outcome": "Outcome D (topology)", "detail": "topology implicated; learned adjacency deferred to a later phase"}
    return {"intervention": "none", "outcome": "Outcome E (no validated intervention)", "detail": "No relational intervention produced a reproducible improvement"}


# ---------------------------------------------------------------------------
# Failure diagnosis (explicit criteria, §24)
# ---------------------------------------------------------------------------
def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Machine-readable failure diagnosis with explicit pass/fail criteria."""
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({
            "category": category,
            "failed": bool(failed),
            "criterion": criterion,
            "observed": observed,
        })

    depth = agg.get("depth") or {}
    best_gain = depth.get("best_R_gain_mean", 0.0) if depth.get("tested") else 0.0
    add("depth_failure", bool(depth.get("tested")) and best_gain <= 0.0,
        "best depth R gain <= 0pp", f"{best_gain * 100:+.1f}pp")
    aggr = agg.get("aggregation") or {}
    ag = aggr.get("best_R_gain_mean", 0.0) if aggr.get("tested") else 0.0
    add("aggregation_failure", bool(aggr.get("tested")) and ag <= 0.0,
        "best aggregation R gain <= 0pp", f"{ag * 100:+.1f}pp")
    cap = agg.get("capacity") or {}
    cg = cap.get("R_gain_mean", 0.0) if cap.get("tested") else 0.0
    add("capacity_failure", bool(cap.get("tested")) and cg <= 0.0,
        "capacity R gain <= 0pp", f"{cg * 100:+.1f}pp")
    topo = agg.get("topology") or {}
    cdrop = topo.get("candidate_drop_R", 0.0) if topo.get("tested") else 0.0
    add("topology_failure", bool(topo.get("tested")) and cdrop <= 0.0,
        "candidate relational-destruction drop on R <= 0pp", f"{cdrop * 100:+.1f}pp")
    gap = agg.get("gap") or {}
    g = gap.get("candidate_gap_mean", gap.get("baseline_gap_mean", 0.0))
    add("representation_prediction_gap", g >= 0.05,
        "candidate probe-final gap >= 5pp", f"{g * 100:.1f}pp")
    comp = agg.get("compositional") or {}
    add("compositional_transfer_failure",
        bool(comp.get("tested")) and comp.get("R_gain_mean", 0.0) >= 0.02
        and comp.get("RC_gain_mean", 0.0) < 0.01 and comp.get("FRC_gain_mean", 0.0) < 0.01,
        "R gain >= 2pp but RC and FRC gains < 1pp",
        f"R {comp.get('R_gain_mean', 0.0) * 100:+.1f}pp, RC {comp.get('RC_gain_mean', 0.0) * 100:+.1f}pp, FRC {comp.get('FRC_gain_mean', 0.0) * 100:+.1f}pp")
    unstable = []
    for sec in ("depth", "aggregation", "capacity"):
        for lab, sd in (agg.get("seed_stability") or {}).get(sec, {}).items():
            if sd >= SEED_STD_UNSTABLE:
                unstable.append(f"{sec}/{lab}")
    add("seed_instability", len(unstable) > 0,
        f"any condition R seed-SD >= {SEED_STD_UNSTABLE * 100:.0f}pp", "; ".join(unstable) if unstable else "all stable")
    compute = agg.get("compute") or {}
    cand_params = compute.get("candidate_params", 0)
    base_params = compute.get("baseline_params", 1)
    growth = (cand_params - base_params) / max(1, base_params)
    add("compute_regression", growth > PARAM_GROWTH_FLAG,
        f"candidate params growth > {PARAM_GROWTH_FLAG * 100:.0f}%", f"{growth * 100:+.1f}%")
    lat = agg.get("latency") or {}
    lr = (lat.get("candidate_total_us", 0.0) / max(1e-9, lat.get("baseline_total_us", 1.0)))
    add("latency_regression", bool(lat.get("tested")) and lr > LATENCY_REGRESSION_FACTOR,
        f"candidate latency > {LATENCY_REGRESSION_FACTOR}x baseline", f"{lr:.2f}x")
    bi = agg.get("branch_interference") or {}
    add("branch_interference", bool(bi.get("present")),
        f"candidate drops F or C by >= {BRANCH_INTERFERENCE_DROP * 100:.0f}pp vs baseline",
        bi.get("observed", "not observed"))
    return rows
