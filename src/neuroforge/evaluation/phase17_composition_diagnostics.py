"""Phase 17 heterogeneous information composition diagnostics.

Pure evaluators; no Phase 1-16 module is modified. All thresholds are
pre-registered below; H1-H8 and the final CASE are programmatic functions of
executed-experiment data.

Distinguishes four claims (§23): information exists / is decodable /
is task-usable / is composable with another branch.
"""
from __future__ import annotations

import statistics
from typing import Any

import torch
from torch import nn

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.models.phase11_diagnostics import linear_probe_accuracy_per_class


FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")

# Pre-registered thresholds (absolute fractions; 0.05 = 5pp).
PROBE_DROP_SUPPORTED = 0.05
ORACLE_GAIN_SUPPORTED = 0.05
ORACLE_GAIN_PARTIAL = 0.02
INTERACTION_SUPPORTED = 0.03
ORDER_SPREAD_SUPPORTED = 0.05
ORDER_SPREAD_PARTIAL = 0.03
DOMINATION_SUPPORTED = 3.0
DOMINATION_PARTIAL = 2.0
CEILING_SUPPORTED = 0.01
SEED_STD_UNSTABLE = 0.05

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")

COMPONENT_TASKS = ("F_signal", "R_signal", "C_signal", "final_target")


def component_targets(eval_ds: Phase9MixedStructureDataset) -> dict[str, torch.Tensor]:
    """Component-level diagnostic targets (benchmark labels unchanged)."""
    n = len(eval_ds)
    return {
        "F_signal": torch.tensor([int(eval_ds.items[i]["sf"] == 1) for i in range(n)], dtype=torch.long),
        "R_signal": torch.tensor([int(eval_ds.items[i]["sr"] == 1) for i in range(n)], dtype=torch.long),
        "C_signal": torch.tensor([int(eval_ds.items[i]["sc"] == 1) for i in range(n)], dtype=torch.long),
        "final_target": eval_ds.targets,
    }


@torch.no_grad()
def probe_state_components(
    state: torch.Tensor,
    eval_ds: Phase9MixedStructureDataset,
    hidden_dim: int = 24,
) -> dict[str, float]:
    """Linear-probe decodability of each component label from a [B,S,H] state."""
    if state.ndim != 3:
        raise ValueError(f"probe_state_components expects [B, S, H], got {tuple(state.shape)}")
    out: dict[str, float] = {}
    for task, targets in component_targets(eval_ds).items():
        sub = linear_probe_accuracy_per_class(state, targets, hidden_dim=hidden_dim)
        out[task] = sub.get("overall", 0.0)
    return out


# ---------------------------------------------------------------------------
# Shared diagnostic-head protocol (frozen states, deterministic, non-invasive)
# ---------------------------------------------------------------------------
HEAD_LR = 1e-2
HEAD_WD = 1e-4
HEAD_EPOCHS = 10
HEAD_BATCH = 60


def train_diag_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    in_dim: int,
    seed: int = 0,
    epochs: int = HEAD_EPOCHS,
) -> nn.Linear:
    """Train Linear(in_dim→2) on frozen diagnostic features."""
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        head = nn.Linear(in_dim, 2)
        opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
        n = len(train_x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            head.train()
            for i in range(0, n, HEAD_BATCH):
                idx = perm[i : i + HEAD_BATCH]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(train_x[idx]), train_y[idx])
                loss.backward()
                opt.step()
        head.eval()
        return head
    finally:
        torch.set_rng_state(rng_state)


@torch.no_grad()
def eval_diag_head(
    head: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    families_list: list[str],
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, float]:
    head.eval()
    preds = head(x).argmax(-1)
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(families_list) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["overall"] = float((preds == targets).float().mean().item())
    out["mixed_mean"] = statistics.mean([out.get(f, 0.0) for f in MIXED_FAMS])
    return out


# ---------------------------------------------------------------------------
# 17F — scale / domination diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def branch_scale_stats(states: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """L2/RMS/mean-abs/variance norms for branch deltas and shared states."""
    out: dict[str, dict[str, float]] = {}
    for name, t in states.items():
        f = t.detach().float()
        out[name] = {
            "l2_mean": float(f.pow(2).sum(dim=(1, 2)).sqrt().mean().item()),
            "rms": float(f.pow(2).mean().sqrt().item()),
            "mean_abs": float(f.abs().mean().item()),
            "variance": float(f.var().item()),
        }
    return out


def domination_ratios(scale: dict[str, dict[str, float]]) -> dict[str, float]:
    """Relational/Feature and Relational/Contextual magnitude ratios (RMS)."""
    rel = max(1e-12, scale["rel_delta"]["rms"])
    return {
        "rel_over_feat": rel / max(1e-12, scale["feat_delta"]["rms"]),
        "rel_over_ctx": rel / max(1e-12, scale["ctx_delta"]["rms"]),
        "max_min_branch": max(v["rms"] for k, v in scale.items() if k in ("feat_delta", "rel_delta", "ctx_delta"))
        / max(1e-12, min(v["rms"] for k, v in scale.items() if k in ("feat_delta", "rel_delta", "ctx_delta"))),
    }


# ---------------------------------------------------------------------------
# 17G — interaction gains
# ---------------------------------------------------------------------------
def interaction_gain(
    pair_acc: float, single_a_acc: float, single_b_acc: float
) -> float:
    """accuracy(A+B) - max(accuracy(A), accuracy(B)). Negative = interference."""
    return pair_acc - max(single_a_acc, single_b_acc)


# ---------------------------------------------------------------------------
# H1-H8 (programmatic)
# ---------------------------------------------------------------------------
def _consistency(n_pos: int, n: int) -> bool:
    return n_pos >= max(2, n - 1)


def build_phase17_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = int(agg.get("n_seeds", 3))

    pres = agg.get("preservation", {})
    if pres.get("tested"):
        drops = {t: pres.get(f"{t}_drop_pre_to_fused", 0.0) for t in ("F_signal", "R_signal", "C_signal")}
        n_big = sum(1 for v in drops.values() if v >= PROBE_DROP_SUPPORTED)
        worst = min(drops.values())
        if n_big >= 2:
            out["H1"] = {"status": "SUPPORTED",
                         "evidence": f"Pre→fused probe drops ≥5pp on {n_big}/3 components (worst {worst * 100:+.1f}pp)."}
        elif n_big == 1:
            out["H1"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Pre→fused probe drop ≥5pp on 1/3 components (worst {worst * 100:+.1f}pp)."}
        else:
            out["H1"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Branch information remains decodable after fusion (worst drop {worst * 100:+.1f}pp)."}
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Preservation audit not executed."}

    dom = agg.get("domination", {})
    if dom.get("tested"):
        r = dom.get("max_min_branch_mean", 1.0)
        if r >= DOMINATION_SUPPORTED:
            out["H2"] = {"status": "SUPPORTED",
                         "evidence": f"Max/min branch RMS ratio {r:.2f}x ≥ 3x: severe scale imbalance."}
        elif r >= DOMINATION_PARTIAL:
            out["H2"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Max/min branch RMS ratio {r:.2f}x (moderate imbalance)."}
        else:
            out["H2"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Max/min branch RMS ratio {r:.2f}x: no domination."}
    else:
        out["H2"] = {"status": "NOT TESTED", "evidence": "Scale diagnostics not executed."}

    inter = agg.get("interaction", {})
    if inter.get("tested"):
        worst = inter.get("worst_pair_gain_mean", 0.0)
        npos = inter.get("n_neg_consistent", 0)
        if worst <= -INTERACTION_SUPPORTED and _consistency(npos, n):
            out["H3"] = {"status": "SUPPORTED",
                         "evidence": f"Worst pair interaction {worst * 100:+.1f}pp, negative on {npos}/{n} seeds."}
        elif worst <= -INTERACTION_SUPPORTED:
            out["H3"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Worst pair interaction {worst * 100:+.1f}pp but seed-inconsistent."}
        elif worst < 0:
            out["H3"] = {"status": "INCONCLUSIVE",
                         "evidence": f"Weak negative interaction {worst * 100:+.1f}pp."}
        else:
            out["H3"] = {"status": "NOT SUPPORTED",
                         "evidence": f"No negative interaction (worst {worst * 100:+.1f}pp)."}
    else:
        out["H3"] = {"status": "NOT TESTED", "evidence": "Branch interaction not executed."}

    order = agg.get("order", {})
    if order.get("tested"):
        # Raw spread is confounded by final-expert identity (the last expert's
        # head/pooling differs per order). The controlled test is same-final-
        # expert orientations; SUPPORTED additionally requires 3/3 unanimity.
        orient = order.get("orientations", {})
        spread = order.get("range_R_mean", 0.0)
        best = {"name": "", "diff": 0.0, "n_pos": 0, "n": int(agg.get("n_seeds", 3))}
        for name, o in orient.items():
            if abs(o.get("diff_mean", 0.0)) > abs(best["diff"]):
                best = {"name": name, "diff": o.get("diff_mean", 0.0),
                        "n_pos": o.get("n_pos", 0), "n": o.get("n_seeds", best["n"])}
        if abs(best["diff"]) >= ORDER_SPREAD_SUPPORTED and best["n_pos"] == best["n"]:
            out["H4"] = {"status": "SUPPORTED",
                         "evidence": f"Same-final-expert orientation {best['name']} {best['diff'] * 100:+.1f}pp on R, {best['n_pos']}/{best['n']} seeds (raw spread {spread * 100:.1f}pp)."}
        elif abs(best["diff"]) >= ORDER_SPREAD_PARTIAL and best["n_pos"] >= max(2, best["n"] - 1):
            out["H4"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Same-final-expert orientation {best['name']} {best['diff'] * 100:+.1f}pp on R but seed-inconsistent ({best['n_pos']}/{best['n']}); raw spread {spread * 100:.1f}pp is dominated by final-expert identity."}
        else:
            out["H4"] = {"status": "NOT SUPPORTED",
                         "evidence": f"No consistent same-final-expert order effect (best {best['name']} {best['diff'] * 100:+.1f}pp); raw spread {spread * 100:.1f}pp tracks the final expert, not order."}
    else:
        out["H4"] = {"status": "NOT TESTED", "evidence": "Composition order not executed."}

    suff = agg.get("sufficiency", {})
    if suff.get("tested"):
        rc, frc = suff.get("oracle_RC", 0.0), suff.get("oracle_FRC", 0.0)
        rcg, frcg = suff.get("oracle_RC_gain", 0.0), suff.get("oracle_FRC_gain", 0.0)
        if rcg >= ORACLE_GAIN_SUPPORTED and frcg >= ORACLE_GAIN_SUPPORTED and rc >= 0.60:
            out["H5"] = {"status": "SUPPORTED",
                         "evidence": f"Oracle RC {rc * 100:.1f}% ({rcg * 100:+.1f}pp), FRC {frc * 100:.1f}% ({frcg * 100:+.1f}pp)."}
        elif rcg >= ORACLE_GAIN_PARTIAL or frcg >= ORACLE_GAIN_PARTIAL:
            out["H5"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Oracle RC {rc * 100:.1f}% ({rcg * 100:+.1f}pp), FRC {frc * 100:.1f}% ({frcg * 100:+.1f}pp) (modest)."}
        else:
            out["H5"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Full branch information does not help RC/FRC (gains {rcg * 100:+.1f}/{frcg * 100:+.1f}pp)."}
        if rcg >= ORACLE_GAIN_SUPPORTED or frcg >= ORACLE_GAIN_SUPPORTED:
            out["H6"] = {"status": "SUPPORTED",
                         "evidence": f"Diagnostic readout exploits jointly available info (RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp) that fusion/readout does not."}
        elif rcg > 0 or frcg > 0:
            out["H6"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Weak diagnostic advantage (RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp)."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED",
                         "evidence": "Even the diagnostic readout cannot exploit branch information: limitation is deeper."}
    else:
        out["H5"] = {"status": "NOT TESTED", "evidence": "Oracle representation not executed."}
        out["H6"] = {"status": "NOT TESTED", "evidence": "Oracle representation not executed."}

    iv = agg.get("intervention", {})
    if iv.get("tested"):
        rcg, frcg = iv.get("RC_gain", 0.0), iv.get("FRC_gain", 0.0)
        if rcg >= 0.02 and frcg >= 0.02:
            out["H7"] = {"status": "SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp."}
        else:
            out["H7"] = {"status": "NOT SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp, FRC {frcg * 100:+.1f}pp."}
        ceil = agg.get("ceiling", {})
        if ceil.get("tested"):
            d = ceil["new_minus_old_mixed"]
            out["H8"] = {"status": "SUPPORTED" if d > CEILING_SUPPORTED else "NOT SUPPORTED",
                         "evidence": f"Ceiling delta {d * 100:+.1f}pp mixed mean."}
        else:
            out["H8"] = {"status": "NOT TESTED", "evidence": "Ceiling not re-evaluated."}
    else:
        out["H7"] = {"status": "NOT TESTED",
                     "evidence": iv.get("reason", "No composition mechanism earned an intervention.")}
        out["H8"] = {"status": "NOT TESTED", "evidence": "No validated candidate."}
    return out


def select_phase17_case(hy: dict[str, dict[str, Any]], agg: dict[str, Any]) -> tuple[str, str]:
    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    if st("H1") == "SUPPORTED":
        return ("CASE A", "Fusion destroys branch-specific information")
    if st("H2") == "SUPPORTED":
        return ("CASE B", "One branch dominates / scale mismatch limits composition")
    if st("H3") == "SUPPORTED":
        return ("CASE C", "Branches exhibit destructive interaction")
    if st("H4") == "SUPPORTED":
        return ("CASE D", "Composition order is the primary limitation")
    pres = agg.get("preservation", {})
    branch_probes = [pres.get(f"branch_{t}", 0.0) for t in ("F_signal", "R_signal", "C_signal")]
    if (max(branch_probes + [0.0]) >= 0.70
            and st("H5") != "SUPPORTED" and st("H1") != "SUPPORTED"):
        return ("CASE E", "Branch information exists but is not jointly task-usable")
    return ("CASE F", "No sufficiently localized composition bottleneck")


def select_minimal_intervention(
    case: str, hy: dict[str, dict[str, Any]], agg: dict[str, Any]
) -> dict[str, str]:
    iv = agg.get("intervention", {})
    if iv.get("tested") and hy.get("H7", {}).get("status") == "SUPPORTED":
        return {"intervention": iv.get("name", "candidate"),
                "outcome": "Single minimal composition intervention validated",
                "detail": iv.get("detail", "")}
    return {"intervention": "none",
            "outcome": "NO INTERVENTION (diagnosis only)",
            "detail": f"{case}: no single composition mechanism was sufficiently localized."}


def recommendation_for_case(case: str) -> str:
    return {
        "CASE A": "Redesign fusion to preserve branch-specific information; gate on pre/post probe parity.",
        "CASE B": "Test minimal branch normalization; gate on scale parity without R regression.",
        "CASE C": "Investigate pairwise interference mechanisms; gate on non-negative interaction gains.",
        "CASE D": "Adopt the reproducibly superior composition order; verify RC/FRC follow.",
        "CASE E": "Treat joint usability as the open problem; isolated gains do not compose.",
        "CASE F": "STOP adding architecture; report the composition boundary.",
    }[case]


def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({"category": category, "failed": bool(failed),
                     "criterion": criterion, "observed": observed})

    pres = agg.get("preservation", {})
    worst_drop = min([pres.get(f"{t}_drop_pre_to_fused", 0.0) for t in ("F_signal", "R_signal", "C_signal")] or [0.0])
    add("fusion_information_loss", bool(pres.get("tested")) and worst_drop >= PROBE_DROP_SUPPORTED,
        "pre→fused probe drop ≥5pp on a component", f"worst {worst_drop * 100:+.1f}pp")
    dom = agg.get("domination", {})
    add("scale_domination", bool(dom.get("tested")) and dom.get("max_min_branch_mean", 1.0) >= DOMINATION_SUPPORTED,
        "max/min branch RMS ≥ 3x", f"{dom.get('max_min_branch_mean', 0.0):.2f}x")
    inter = agg.get("interaction", {})
    add("destructive_interaction", bool(inter.get("tested")) and inter.get("worst_pair_gain_mean", 0.0) <= -INTERACTION_SUPPORTED,
        "worst pair interaction ≤ −3pp", f"{inter.get('worst_pair_gain_mean', 0.0) * 100:+.1f}pp")
    suff = agg.get("sufficiency", {})
    add("insufficient_joint_information",
        bool(suff.get("tested")) and suff.get("oracle_RC_gain", 0.0) < ORACLE_GAIN_PARTIAL and suff.get("oracle_FRC_gain", 0.0) < ORACLE_GAIN_PARTIAL,
        "oracle RC/FRC gains both < 2pp", f"RC {suff.get('oracle_RC_gain', 0.0) * 100:+.1f}pp, FRC {suff.get('oracle_FRC_gain', 0.0) * 100:+.1f}pp")
    comp = agg.get("compositional", {})
    add("compositional_transfer_failure",
        bool(comp.get("tested")) and comp.get("R_gain", 0.0) >= 0.02
        and comp.get("RC_gain", 0.0) < 0.01 and comp.get("FRC_gain", 0.0) < 0.01,
        "R gain ≥ 2pp but RC and FRC gains < 1pp",
        f"R {comp.get('R_gain', 0.0) * 100:+.1f}pp, RC {comp.get('RC_gain', 0.0) * 100:+.1f}pp, FRC {comp.get('FRC_gain', 0.0) * 100:+.1f}pp")
    unstable = [f"{s}/{l}" for s, dd in (agg.get("seed_stability") or {}).items() for l, sd in dd.items() if sd >= SEED_STD_UNSTABLE]
    add("seed_instability", len(unstable) > 0, "any seed-SD ≥ 5pp", "; ".join(unstable) if unstable else "all stable")
    order = agg.get("order", {})
    orient = order.get("orientations", {})
    _best_orient = max([abs(o.get("diff_mean", 0.0)) for o in orient.values()] + [0.0])
    add("order_sensitivity", bool(order.get("tested")) and _best_orient >= ORDER_SPREAD_SUPPORTED,
        "same-final-expert orientation effect ≥ 5pp on R", f"{_best_orient * 100:.1f}pp (seed-consistency in H4 evidence)")
    return rows
