"""Phase 20 clean portfolio re-evaluation metrics (repaired benchmark only).

Pure evaluators; no Phase 1-19 module is modified. Thresholds pre-registered
below; H1-H8 and CASE A-F are programmatic. Historical (pre-repair) numbers
are never inputs to any verdict here.
"""
from __future__ import annotations

import hashlib
import statistics
from typing import Any

import torch
from torch import nn

from neuroforge.datasets.phase9_mixed_repaired import (
    CONSTRUCTION_VERSION,
    Phase9MixedRepairedDataset,
)
from neuroforge.models.specialists import StandaloneSpecialist


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
PURE_FAMS = ("F", "R", "C")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")

# Pre-registered thresholds (absolute fractions).
PURE_CAPABLE = 0.80
MIXED_CAPABLE_MEAN = 0.70
MIXED_CAPABLE_PARTIAL = 0.60
COMPOSITION_GAIN = 0.03
ROUTING_GAIN = 0.02
CEILING_GAIN = 0.02
COLLAPSE_GAP = 0.50
DISAGREE_LOW = 0.20
FAVOR_SYSTEMATIC = 0.90
SEED_STD_UNSTABLE = 0.05

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")

EXPERTS_20 = ("mlp", "graph", "attention", "attention_v2", "joint", "joint_co", "depth3")


def assert_repaired(ds: Any) -> None:
    """Guard against accidentally mixing the original generator (20A/§4)."""
    ver = getattr(ds, "construction_version", "phase9-mixed-original")
    if ver != CONSTRUCTION_VERSION:
        raise ValueError(f"Phase 20 requires {CONSTRUCTION_VERSION}, got {ver!r}")


def dataset_fingerprint(ds: Any) -> str:
    h = hashlib.sha256()
    h.update(ds.features.numpy().tobytes())
    h.update(ds.targets.numpy().tobytes())
    return h.hexdigest()[:16]


def check_zero_overlap(a: Any, b: Any) -> int:
    """Count exact sample overlaps between two splits (must be 0)."""
    seen = {t.numpy().tobytes() for t in a.features.detach().reshape(len(a), -1)}
    return sum(1 for t in b.features.detach().reshape(len(b), -1) if t.numpy().tobytes() in seen)


@torch.no_grad()
def per_family_accuracy(
    expert: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    families_list: list[str],
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, float]:
    expert.eval()
    preds = expert(features).argmax(-1)
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(families_list) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["mixed_mean"] = statistics.mean([out.get(f, 0.0) for f in MIXED_FAMS])
    out["overall"] = float((preds == targets).float().mean().item())
    return out


@torch.no_grad()
def analytical_flops(expert: StandaloneSpecialist, seq_len: int = 12) -> float:
    """Analytic forward FLOPs for one sample (documented convention).

    Encoder Linear(8→H) + sum of block-reported analytical costs + head
    (LayerNorm ≈ 5H + Linear(H→2) = 4H+2). Block costs come from the blocks'
    own forward accounting, so variants (fusion/depth) are counted consistently.
    """
    h = expert.encoder.out_features
    total = float(2 * 8 * h * seq_len)  # encoder matmuls over S tokens
    dummy_state = torch.zeros(1, seq_len, h)
    dummy_raw = torch.zeros(1, seq_len, 8)
    for block in expert.blocks:
        try:
            _, c = block(dummy_state, dummy_raw)
        except TypeError:
            _, c = block(dummy_state)
        total += float(c)
    total += float(5 * h + 4 * h + 2)  # LayerNorm + Linear(H→2)
    return total


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _cons(n_pos: int, n: int) -> bool:
    return n_pos >= max(2, n - 1)


def build_phase20_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = int(agg.get("n_seeds", 3))

    pure = agg.get("pure", {})
    if pure.get("tested"):
        worst = min(pure.get("best_F", 0.0), pure.get("best_R", 0.0), pure.get("best_C", 0.0))
        if worst >= PURE_CAPABLE:
            out["H1"] = {"status": "SUPPORTED", "evidence": f"Best pure acc F/R/C all ≥ {PURE_CAPABLE * 100:.0f}% (worst {worst * 100:.1f}%)."}
        elif worst >= PURE_CAPABLE - 0.10:
            out["H1"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Worst pure {worst * 100:.1f}%."}
        else:
            out["H1"] = {"status": "NOT SUPPORTED", "evidence": f"Worst pure {worst * 100:.1f}%."}
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Pure-expert baselines not executed."}

    mixed = agg.get("mixed", {})
    if mixed.get("tested"):
        mm = mixed.get("best_mixed_mean", 0.0)
        if mm >= MIXED_CAPABLE_MEAN:
            out["H2"] = {"status": "SUPPORTED", "evidence": f"Best-per-family mixed mean {mm * 100:.1f}%."}
        elif mm >= MIXED_CAPABLE_PARTIAL:
            out["H2"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Best-per-family mixed mean {mm * 100:.1f}% (modest)."}
        else:
            out["H2"] = {"status": "NOT SUPPORTED", "evidence": f"Best-per-family mixed mean {mm * 100:.1f}%."}
    else:
        out["H2"] = {"status": "NOT TESTED", "evidence": "Mixed evaluation not executed."}

    req = agg.get("requires_multi", {})
    if req.get("tested"):
        if req.get("rc_margin", 0.0) >= 0.20 and req.get("frc_margin", 0.0) >= 0.10:
            out["H3"] = {"status": "SUPPORTED", "evidence": f"RC rule-vs-single margin {req['rc_margin'] * 100:+.1f}pp."}
            out["H4"] = {"status": "SUPPORTED" if req.get("frc_margin", 0.0) >= 0.10 else "PARTIALLY SUPPORTED",
                         "evidence": f"FRC rule-vs-single margin {req['frc_margin'] * 100:+.1f}pp."}
        elif req.get("rc_margin", 0.0) >= 0.20:
            out["H3"] = {"status": "SUPPORTED", "evidence": f"RC margin {req['rc_margin'] * 100:+.1f}pp."}
            out["H4"] = {"status": "NOT SUPPORTED", "evidence": f"FRC margin {req.get('frc_margin', 0.0) * 100:+.1f}pp."}
        else:
            out["H3"] = {"status": "NOT SUPPORTED", "evidence": f"RC margin {req.get('rc_margin', 0.0) * 100:+.1f}pp."}
            out["H4"] = {"status": "NOT SUPPORTED", "evidence": f"FRC margin {req.get('frc_margin', 0.0) * 100:+.1f}pp."}
    else:
        out["H3"] = {"status": "NOT TESTED", "evidence": "Multi-component requirement not executed."}
        out["H4"] = {"status": "NOT TESTED", "evidence": "Multi-component requirement not executed."}

    comp = agg.get("composition", {})
    if comp.get("tested"):
        rcg, frcg = comp.get("RC_gain", 0.0), comp.get("FRC_gain", 0.0)
        if rcg >= COMPOSITION_GAIN and frcg >= COMPOSITION_GAIN and _cons(comp.get("n_pos", 0), n):
            out["H5"] = {"status": "SUPPORTED", "evidence": f"Fixed composition gains RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp."}
        elif rcg >= COMPOSITION_GAIN or frcg >= COMPOSITION_GAIN:
            out["H5"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp (partial)."}
        else:
            out["H5"] = {"status": "NOT SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp."}
    else:
        out["H5"] = {"status": "NOT TESTED", "evidence": "Fixed composition not executed."}

    rout = agg.get("routing", {})
    if rout.get("tested"):
        g = rout.get("learned_minus_best_single_mixed", 0.0)
        if g >= ROUTING_GAIN and _cons(rout.get("n_pos", 0), n):
            out["H6"] = {"status": "SUPPORTED", "evidence": f"Learned routing beats best fixed expert by {g * 100:+.1f}pp mixed."}
        elif g > 0:
            out["H6"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Routing gain {g * 100:+.1f}pp (weak/inconsistent)."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED", "evidence": f"Routing gain {g * 100:+.1f}pp."}
    else:
        out["H6"] = {"status": "NOT TESTED", "evidence": "Learned routing not executed."}

    beh = agg.get("behavior", {})
    if beh.get("tested"):
        gap = beh.get("agree_acc", 0.0) - beh.get("disagree_acc", 0.0)
        if gap >= COLLAPSE_GAP and beh.get("disagree_acc", 1.0) <= DISAGREE_LOW:
            out["H7"] = {"status": "SUPPORTED", "evidence": f"Single-component collapse persists (agree {beh['agree_acc'] * 100:.1f}% vs disagree {beh['disagree_acc'] * 100:.1f}%)."}
        elif beh.get("disagree_acc", 0.0) >= 0.30:
            out["H7"] = {"status": "NOT SUPPORTED", "evidence": f"Disagree accuracy recovered to {beh['disagree_acc'] * 100:.1f}%: collapse gone."}
        else:
            out["H7"] = {"status": "PARTIALLY SUPPORTED", "evidence": f"Gap {gap * 100:.1f}pp, disagree {beh.get('disagree_acc', 0.0) * 100:.1f}%."}
    else:
        out["H7"] = {"status": "NOT TESTED", "evidence": "Composition semantics not executed."}

    ceil = agg.get("ceiling", {})
    if ceil.get("tested"):
        d = ceil.get("portfolio_minus_single_mixed", 0.0)
        rc = ceil.get("portfolio_RC", 0.0) - ceil.get("single_RC", 0.0)
        frc = ceil.get("portfolio_FRC", 0.0) - ceil.get("single_FRC", 0.0)
        if d >= CEILING_GAIN and rc > 0 and frc > 0:
            out["H8"] = {"status": "SUPPORTED", "evidence": f"Portfolio beats ceiling by {d * 100:+.1f}pp mixed (RC {rc * 100:+.1f}, FRC {frc * 100:+.1f})."}
        else:
            out["H8"] = {"status": "NOT SUPPORTED", "evidence": f"Portfolio vs ceiling {d * 100:+.1f}pp mixed (RC {rc * 100:+.1f}, FRC {frc * 100:+.1f})."}
    else:
        out["H8"] = {"status": "NOT TESTED", "evidence": "Ceiling comparison not executed."}
    return out


def select_phase20_case(hy: dict[str, dict[str, Any]], agg: dict[str, Any]) -> tuple[str, str]:
    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    if st("H8") == "SUPPORTED":
        return ("CASE E", "Existing heterogeneous composition demonstrates a genuine capability gain")
    best_rc = agg.get("mixed", {}).get("best_RC", 1.0)
    if st("H2") == "NOT SUPPORTED" or (st("H7") == "SUPPORTED" and best_rc < 0.60):
        return ("CASE B", "Composition failure persists despite valid observability")
    if st("H2") in ("SUPPORTED", "PARTIALLY SUPPORTED") and st("H6") == "NOT SUPPORTED":
        return ("CASE C", "Existing experts solve mixed tasks but routing fails to exploit them")
    if st("H5") in ("SUPPORTED", "PARTIALLY SUPPORTED"):
        return ("CASE D", "Existing composition helps but does not exceed the empirical single-expert ceiling")
    if st("H2") == "SUPPORTED" and st("H7") == "NOT SUPPORTED" and st("H5") != "SUPPORTED":
        return ("CASE A", "Repair resolves the apparent composition failure")
    if st("H2") == "NOT SUPPORTED":
        return ("CASE B", "Composition failure persists despite valid observability")
    return ("CASE F", "Results are inconclusive and another diagnostic phase is required")


def recommendation_for_case(case: str) -> str:
    return {
        "CASE A": "Composition works on valid inputs: consolidate the repaired baseline; no new architecture.",
        "CASE B": "Treat post-repair composition failure as a legitimate research question for Phase 21.",
        "CASE C": "Routing is the next bottleneck: diagnose route quality before touching experts.",
        "CASE D": "Adaptive-composition claim remains unvalidated: find what beats the ceiling or report it.",
        "CASE E": "Characterize the mechanism behind the reproducible portfolio advantage.",
        "CASE F": "Run another diagnostic phase; do not add architecture on inconclusive evidence.",
    }[case]


def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({"category": category, "failed": bool(failed),
                     "criterion": criterion, "observed": observed})

    pure = agg.get("pure", {})
    add("pure_capability_failure", bool(pure.get("tested"))
        and min(pure.get("best_F", 1.0), pure.get("best_R", 1.0), pure.get("best_C", 1.0)) < PURE_CAPABLE,
        "a pure family best < 80%",
        f"F {pure.get('best_F', 0.0) * 100:.1f}%, R {pure.get('best_R', 0.0) * 100:.1f}%, C {pure.get('best_C', 0.0) * 100:.1f}%")
    mixed = agg.get("mixed", {})
    add("mixed_capability_failure", bool(mixed.get("tested")) and mixed.get("best_mixed_mean", 1.0) < MIXED_CAPABLE_PARTIAL,
        "best-per-family mixed mean < 60%", f"{mixed.get('best_mixed_mean', 0.0) * 100:.1f}%")
    comp = agg.get("composition", {})
    add("composition_no_gain", bool(comp.get("tested"))
        and comp.get("RC_gain", 1.0) < COMPOSITION_GAIN and comp.get("FRC_gain", 1.0) < COMPOSITION_GAIN,
        "fixed composition gains < 3pp on both RC and FRC",
        f"RC {comp.get('RC_gain', 0.0) * 100:+.1f}pp, FRC {comp.get('FRC_gain', 0.0) * 100:+.1f}pp")
    rout = agg.get("routing", {})
    add("routing_no_gain", bool(rout.get("tested")) and rout.get("learned_minus_best_single_mixed", 1.0) < ROUTING_GAIN,
        "learned routing gain < 2pp over best fixed expert",
        f"{rout.get('learned_minus_best_single_mixed', 0.0) * 100:+.1f}pp")
    beh = agg.get("behavior", {})
    gap = beh.get("agree_acc", 0.0) - beh.get("disagree_acc", 0.0)
    add("single_component_collapse", bool(beh.get("tested")) and gap >= COLLAPSE_GAP,
        "agree−disagree gap ≥ 50pp", f"{gap * 100:.1f}pp")
    ceil = agg.get("ceiling", {})
    add("ceiling_unmoved", bool(ceil.get("tested")) and ceil.get("portfolio_minus_single_mixed", 1.0) < CEILING_GAIN,
        "portfolio vs single-expert ceiling < 2pp", f"{ceil.get('portfolio_minus_single_mixed', 0.0) * 100:+.1f}pp")
    unstable = [f"{s}/{l}" for s, dd in (agg.get("seed_stability") or {}).items() for l, sd in dd.items() if sd >= SEED_STD_UNSTABLE]
    add("seed_instability", len(unstable) > 0, "any seed-SD ≥ 5pp", "; ".join(unstable) if unstable else "all stable")
    smoke = agg.get("smoke", {})
    add("smoke_failed", bool(smoke.get("tested")) and not smoke.get("passed", False),
        "repaired-validity smoke must pass", smoke.get("detail", ""))
    return rows
