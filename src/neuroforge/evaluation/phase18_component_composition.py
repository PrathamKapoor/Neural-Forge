"""Phase 18 component-composition and decision-semantics diagnostics.

Pure evaluators; no Phase 1-17 module is modified. Thresholds pre-registered
below; H1-H8, CASE A-F and the 18L intervention gate are programmatic.

Benchmark-semantics note (§18J): the Phase 9 construction defines RC from the
R and C signals (target = majority of sr+sc, ties broken by sample parity)
and FRC from F+R+C (majority of three, never tied). Where the Phase 18 text
frames RC as F+R, this module follows the construction (R+C for RC) and
documents the deviation; the F component on RC is retained as a negative
control (no F signal is injected, so F probes/heads must read ~chance).
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

# RC is constructed from R+C; FRC from F+R+C (verified against construction).
RC_COMPONENTS = ("R", "C")
FRC_COMPONENTS = ("F", "R", "C")

# Pre-registered thresholds (absolute fractions).
COMP_AVAILABLE = 0.70
JOINT_AVAILABLE = 0.60
DISAGREE_COLLAPSE_GAP = 0.50
DISAGREE_LOW = 0.20
FAVOR_SYSTEMATIC = 0.90
NONLINEAR_SUCCESS = 0.60
SHORTCUT_TOL = 0.03
CEILING_SUPPORTED = 0.01
SEED_STD_UNSTABLE = 0.05

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")


def comp_key(letter: str) -> str:
    return {"F": "sf", "R": "sr", "C": "sc"}[letter]


def component_labels(ds: Phase9MixedStructureDataset, letter: str) -> torch.Tensor:
    """Binary diagnostic labels for one component (benchmark labels unchanged)."""
    key = comp_key(letter)
    return torch.tensor([int(ds.items[i][key] == 1) for i in range(len(ds))], dtype=torch.long)


def joint_rc_labels(ds: Phase9MixedStructureDataset) -> torch.Tensor:
    """4-class joint (sr, sc) diagnostic target for RC analysis."""
    out = []
    for i in range(len(ds)):
        sr = int(ds.items[i]["sr"] == 1)
        sc = int(ds.items[i]["sc"] == 1)
        out.append(sr * 2 + sc)
    return torch.tensor(out, dtype=torch.long)


def joint_frc_labels(ds: Phase9MixedStructureDataset) -> torch.Tensor:
    """8-class joint (sf, sr, sc) diagnostic target for FRC analysis."""
    out = []
    for i in range(len(ds)):
        sf = int(ds.items[i]["sf"] == 1)
        sr = int(ds.items[i]["sr"] == 1)
        sc = int(ds.items[i]["sc"] == 1)
        out.append(sf * 4 + sr * 2 + sc)
    return torch.tensor(out, dtype=torch.long)


def rc_agreement_mask(ds: Phase9MixedStructureDataset) -> tuple[list[int], list[int]]:
    """Partition RC indices into component-agreement vs disagreement."""
    agree, disagree = [], []
    for i, fm in enumerate(ds.families_list):
        if fm != "RC":
            continue
        (agree if ds.items[i]["sr"] == ds.items[i]["sc"] else disagree).append(i)
    return agree, disagree


def verify_rc_parity_semantics(ds: Phase9MixedStructureDataset) -> dict[str, Any]:
    """Verify the 18J semantic fact: on RC-disagree, y == R-component always.

    Algebraic (not empirical): sc is set deterministically from the sample index
    (sc=+1 iff i even) and disagree labels are the index-parity tiebreak, so on
    disagree y == (sr>0) == NOT(sc>0) holds identically for every seed.
    """
    agree, disagree = rc_agreement_mask(ds)
    y_is_r = sum(1 for i in disagree if int(ds.targets[i]) == (1 if ds.items[i]["sr"] > 0 else 0))
    y_is_c = sum(1 for i in disagree if int(ds.targets[i]) == (1 if ds.items[i]["sc"] > 0 else 0))
    return {"n_agree": len(agree), "n_disagree": len(disagree),
            "disagree_y_equals_R": y_is_r, "disagree_y_equals_C": y_is_c,
            "parity_fact": y_is_r == len(disagree) and len(disagree) > 0}


@torch.no_grad()
def autocorr_alignment(
    ds: Phase9MixedStructureDataset, families: tuple[str, ...]
) -> float:
    """Agreement between the channel-0 autocorrelation sign and sr (18J).

    The relational signal, when present, lives in channel 0 as `cand`; its
    lag-1 autocorrelation sign IS sr by construction. High agreement on R
    validates the statistic; chance-level on RC/FRC proves input-level erasure
    (the contextual overwrite at construction replaces channels 0:3).
    """
    idx = [i for i, fm in enumerate(ds.families_list) if fm in families]
    if not idx:
        return 0.0
    x0 = ds.features[idx][:, :, 0]
    stat = (x0 * torch.roll(x0, shifts=-1, dims=1)).mean(dim=1).sign()
    sr = torch.tensor([ds.items[i]["sr"] for i in idx])
    return float((stat == sr).float().mean().item())


# ---------------------------------------------------------------------------
# Shared diagnostic-head protocol (frozen features, deterministic)
# ---------------------------------------------------------------------------
HEAD_LR = 1e-2
HEAD_WD = 1e-4
HEAD_EPOCHS = 10
HEAD_BATCH = 60


def train_diag_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    in_dim: int,
    num_classes: int = 2,
    seed: int = 0,
    epochs: int = HEAD_EPOCHS,
) -> nn.Linear:
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        head = nn.Linear(in_dim, num_classes)
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


def train_nonlinear_diag_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    in_dim: int,
    hidden: int | None = None,
    num_classes: int = 2,
    seed: int = 0,
    epochs: int = HEAD_EPOCHS,
) -> nn.Sequential:
    """Small nonlinear joint head: in_dim → hidden → num_classes (18H)."""
    hidden = hidden if hidden is not None else max(8, in_dim // 2)
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
        opt = torch.optim.AdamW(net.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
        n = len(train_x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            net.train()
            for i in range(0, n, HEAD_BATCH):
                idx = perm[i : i + HEAD_BATCH]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(net(train_x[idx]), train_y[idx])
                loss.backward()
                opt.step()
        net.eval()
        return net
    finally:
        torch.set_rng_state(rng_state)


def head_param_count(head: nn.Module) -> int:
    return sum(p.numel() for p in head.parameters())


@torch.no_grad()
def eval_head_on(
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
    return out


# ---------------------------------------------------------------------------
# 18F — logit / margin metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def logit_margin_stats(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Top-1 margin, confidence (max softmax) and entropy per sample."""
    probs = logits.softmax(dim=-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return {"margin": margin, "confidence": top2[:, 0], "entropy": entropy, "preds": logits.argmax(-1)}


# ---------------------------------------------------------------------------
# H1-H8 (programmatic)
# ---------------------------------------------------------------------------
def _cons(n_pos: int, n: int) -> bool:
    return n_pos >= max(2, n - 1)


def build_phase18_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = int(agg.get("n_seeds", 3))

    comp = agg.get("components", {})
    if comp.get("tested"):
        r_c = comp.get("best_R_comp_acc", 0.0)
        c_c = comp.get("best_C_comp_acc", 0.0)
        if min(r_c, c_c) >= COMP_AVAILABLE:
            out["H1"] = {"status": "SUPPORTED",
                         "evidence": f"R-component {r_c * 100:.1f}%, C-component {c_c * 100:.1f}% (best reps)."}
        elif max(r_c, c_c) >= COMP_AVAILABLE:
            out["H1"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Only one component available (R {r_c * 100:.1f}%, C {c_c * 100:.1f}%)."}
        else:
            out["H1"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Components unavailable (R {r_c * 100:.1f}%, C {c_c * 100:.1f}%)."}
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Component matrix not executed."}

    joint = agg.get("joint", {})
    if joint.get("tested"):
        jv = joint.get("joint_RC_acc", 0.0)
        if jv >= JOINT_AVAILABLE and _cons(joint.get("n_pos", 0), n):
            out["H2"] = {"status": "SUPPORTED",
                         "evidence": f"Joint (R&C) accuracy {jv * 100:.1f}% on RC."}
        elif jv >= JOINT_AVAILABLE:
            out["H2"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Joint accuracy {jv * 100:.1f}% but seed-inconsistent."}
        else:
            out["H2"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Joint (R&C) accuracy only {jv * 100:.1f}% on RC."}
    else:
        out["H2"] = {"status": "NOT TESTED", "evidence": "Joint decodability not executed."}

    agr = agg.get("agreement", {})
    if agr.get("tested"):
        gap = agr.get("agree_acc", 0.0) - agr.get("disagree_acc", 0.0)
        if gap >= DISAGREE_COLLAPSE_GAP and agr.get("disagree_acc", 1.0) <= DISAGREE_LOW:
            out["H3"] = {"status": "SUPPORTED",
                         "evidence": f"RC agree {agr['agree_acc'] * 100:.1f}% vs disagree {agr['disagree_acc'] * 100:.1f}% (gap {gap * 100:.1f}pp)."}
        elif gap >= DISAGREE_COLLAPSE_GAP:
            out["H3"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Large agree/disagree gap {gap * 100:.1f}pp but disagree not collapsed."}
        else:
            out["H3"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Agree/disagree gap only {gap * 100:.1f}pp."}
    else:
        out["H3"] = {"status": "NOT TESTED", "evidence": "Agreement audit not executed."}

    fav = agg.get("favor", {})
    if fav.get("tested"):
        frac_c = fav.get("disagree_pred_matches_C", 0.0)
        frac_r = fav.get("disagree_pred_matches_R", 0.0)
        if frac_c >= FAVOR_SYSTEMATIC:
            out["H4"] = {"status": "SUPPORTED",
                         "evidence": f"On RC-disagree, predictions match C {frac_c * 100:.1f}% vs R {frac_r * 100:.1f}%: systematic C-favoring."}
        elif frac_r >= FAVOR_SYSTEMATIC:
            out["H4"] = {"status": "SUPPORTED",
                         "evidence": f"On RC-disagree, predictions match R {frac_r * 100:.1f}% vs C {frac_c * 100:.1f}%: systematic R-favoring."}
        else:
            out["H4"] = {"status": "NOT SUPPORTED",
                         "evidence": f"No systematic favoring (C {frac_c * 100:.1f}%, R {frac_r * 100:.1f}%)."}
    else:
        out["H4"] = {"status": "NOT TESTED", "evidence": "Decision-behavior audit not executed."}

    hd = agg.get("heads", {})
    if hd.get("tested"):
        lin, nl = hd.get("linear_disagree_RC", 0.0), hd.get("nonlinear_disagree_RC", 0.0)
        if lin >= NONLINEAR_SUCCESS:
            out["H5"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Linear head already solves RC-disagree ({lin * 100:.1f}%): production state/head is the limitation, not expressivity."}
        elif nl >= NONLINEAR_SUCCESS and _cons(hd.get("n_pos", 0), n):
            out["H5"] = {"status": "SUPPORTED",
                         "evidence": f"Linear fails RC-disagree ({lin * 100:.1f}%) but nonlinear succeeds ({nl * 100:.1f}%): combination is nonlinear."}
        elif nl >= NONLINEAR_SUCCESS:
            out["H5"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Nonlinear helps RC-disagree ({nl * 100:.1f}%) but inconsistently."}
        else:
            out["H5"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Both fail RC-disagree (linear {lin * 100:.1f}%, nonlinear {nl * 100:.1f}%): deeper than expressivity."}
    else:
        out["H5"] = {"status": "NOT TESTED", "evidence": "Diagnostic heads not executed."}

    obj = agg.get("objective", {})
    if obj.get("tested"):
        # Feasible rules only: the R-rule (y=sr) is a label-oracle on RC because
        # sr is erased from RC inputs (see erasure check); the C-rule is feasible.
        gap = abs(obj.get("model_train_RC", 0.0) - obj.get("rule_C_train_RC", 0.0))
        beh = obj.get("model_matches_C_rule", 0.0)
        if gap <= SHORTCUT_TOL and beh >= FAVOR_SYSTEMATIC:
            out["H6"] = {"status": "SUPPORTED",
                         "evidence": f"Model train RC {obj['model_train_RC'] * 100:.1f}% ≈ feasible C-rule {obj['rule_C_train_RC'] * 100:.1f}%, behavior matches C-rule {beh * 100:.1f}% (R-rule is label-oracle/infeasible)."}
        elif gap <= SHORTCUT_TOL:
            out["H6"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Train accuracies match feasible C-rule (gap {gap * 100:.1f}pp) but behavior match only {beh * 100:.1f}%."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Model train RC differs from the feasible C-rule by {gap * 100:.1f}pp."}
    else:
        out["H6"] = {"status": "NOT TESTED", "evidence": "Objective diagnosis not executed."}

    sem = agg.get("semantics", {})
    if sem.get("tested"):
        erasure = bool(sem.get("erasure_demonstrated", False))
        if sem.get("parity_fact", False) and erasure:
            out["H7"] = {"status": "SUPPORTED",
                         "evidence": "Input-level erasure demonstrated (autocorr-stat recovers sr "
                                     f"{sem.get('align_R', 0.0) * 100:.1f}% on R vs {sem.get('align_RC', 0.0) * 100:.1f}% on RC; "
                                     "construction overwrites channels 0:3 after adding cand). RC-disagree labels equal R "
                                     "identically yet R is unobservable: composition untestable as constructed."}
        elif sem.get("parity_fact", False):
            out["H7"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": "Parity construction verified but erasure not demonstrated; "
                                     f"production↔C-rule match {sem.get('production_matches_C_rule', 0.0) * 100:.1f}%."}
        else:
            out["H7"] = {"status": "NOT SUPPORTED", "evidence": "Parity construction not verified."}
    else:
        out["H7"] = {"status": "NOT TESTED", "evidence": "Semantic contrast not executed."}

    iv = agg.get("intervention", {})
    if iv.get("tested"):
        rcg = iv.get("RC_gain", 0.0)
        if rcg >= 0.02 and _cons(iv.get("n_pos", 0), n):
            out["H8"] = {"status": "SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp reproducibly."}
        else:
            out["H8"] = {"status": "NOT SUPPORTED", "evidence": f"RC {rcg * 100:+.1f}pp."}
    else:
        out["H8"] = {"status": "NOT TESTED",
                     "evidence": iv.get("reason", "Intervention gate (18L) not satisfied.")}
    return out


def select_phase18_case(hy: dict[str, dict[str, Any]], agg: dict[str, Any]) -> tuple[str, str]:
    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    if st("H7") == "SUPPORTED" and agg.get("semantics", {}).get("erasure_demonstrated", False):
        return ("CASE D", "RC/FRC behavior is primarily explained by task semantics (input-level erasure)")
    if st("H2") == "NOT SUPPORTED" and st("H1") != "SUPPORTED":
        return ("CASE A", "Joint component information is unavailable")
    if st("H6") == "SUPPORTED":
        return ("CASE C", "A single-component shortcut is incentivized by the objective")
    if st("H2") == "SUPPORTED" and st("H4") == "SUPPORTED":
        return ("CASE B", "Joint information exists but the final decision mechanism cannot combine it")
    if st("H1") == "SUPPORTED" and st("H2") != "SUPPORTED":
        return ("CASE E", "Multiple component signals exist but their joint usability remains unresolved")
    if st("H7") == "SUPPORTED":
        return ("CASE D", "RC/FRC behavior is primarily explained by task semantics")
    return ("CASE F", "No sufficiently localized bottleneck justifies intervention")


def evaluate_intervention_gate(agg: dict[str, Any]) -> dict[str, Any]:
    """18L gate: TWO independent observations converging on one mechanism."""
    hy = agg.get("hypotheses", {})
    n = int(agg.get("n_seeds", 3))

    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    # Mechanism 1: decision-head bottleneck (H2 + H3-behavior + H5-linear-fails-nonlinear-succeeds).
    hd = agg.get("heads", {})
    m1 = (st("H2") == "SUPPORTED"
          and st("H3") == "SUPPORTED"
          and st("H5") == "SUPPORTED")
    # Mechanism 2: objective shortcut (H1 + H6 + train-behavior).
    m2 = (st("H1") == "SUPPORTED" and st("H6") == "SUPPORTED")
    # Mechanism 3: representation composition (individual decodable + joint lost + preserved recovers).
    pres = agg.get("preservation17", {})
    m3 = (st("H1") == "SUPPORTED" and st("H2") != "SUPPORTED"
          and pres.get("oracle_recovers", False))

    if m1:
        return {"passed": True, "mechanism": "decision_head",
                "candidate": "minimal_joint_decision_head",
                "evidence": ["H2 joint info available", "H3 disagree collapse",
                             "H5 linear fails / nonlinear succeeds"]}
    if m2:
        return {"passed": True, "mechanism": "objective_shortcut",
                "candidate": "objective_modification",
                "evidence": ["H1 components available", "H6 shortcut demonstrated"]}
    if m3:
        return {"passed": True, "mechanism": "representation_composition",
                "candidate": "minimal_composition_adapter",
                "evidence": ["H1 available", "H2 unavailable", "preserved recovers"]}
    return {"passed": False, "mechanism": "none", "candidate": "none",
            "evidence": ["18L requires two converging observations; not satisfied"],
            "detail": f"H2={st('H2')}, H3={st('H3')}, H5={st('H5')}, H6={st('H6')}"}


def select_minimal_intervention(gate: dict[str, Any]) -> dict[str, str]:
    if gate.get("passed"):
        return {"intervention": gate["candidate"],
                "outcome": f"Single minimal intervention ({gate['mechanism']})",
                "detail": "; ".join(gate.get("evidence", []))}
    return {"intervention": "none",
            "outcome": "NO INTERVENTION (default)",
            "detail": "; ".join(gate.get("evidence", ["gate not satisfied"]))}


def recommendation_for_case(case: str) -> str:
    return {
        "CASE A": "Recover joint component information before any decision work; probe which representation stage loses it.",
        "CASE B": "Fix the decision function (minimal joint head) and gate on RC-disagree accuracy.",
        "CASE C": "Remove the single-component shortcut incentive; gate on disagree accuracy, not aggregate RC.",
        "CASE D": "Accept the semantic boundary; do not relitigate parity-tie labels with architecture.",
        "CASE E": "Treat joint usability as the open problem; isolated component gains do not compose.",
        "CASE F": "STOP adding architecture; report the composition boundary.",
    }[case]


def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({"category": category, "failed": bool(failed),
                     "criterion": criterion, "observed": observed})

    comp = agg.get("components", {})
    add("component_unavailable", bool(comp.get("tested")) and min(comp.get("best_R_comp_acc", 1.0), comp.get("best_C_comp_acc", 1.0)) < COMP_AVAILABLE,
        "best R- or C-component acc < 70%",
        f"R {comp.get('best_R_comp_acc', 0.0) * 100:.1f}%, C {comp.get('best_C_comp_acc', 0.0) * 100:.1f}%")
    joint = agg.get("joint", {})
    add("joint_unavailable", bool(joint.get("tested")) and joint.get("joint_RC_acc", 1.0) < JOINT_AVAILABLE,
        "joint (R&C) acc < 60% on RC", f"{joint.get('joint_RC_acc', 0.0) * 100:.1f}%")
    agr = agg.get("agreement", {})
    gap = agr.get("agree_acc", 0.0) - agr.get("disagree_acc", 0.0)
    add("disagree_collapse", bool(agr.get("tested")) and gap >= DISAGREE_COLLAPSE_GAP,
        "agree−disagree gap ≥ 50pp", f"{gap * 100:.1f}pp")
    fav = agg.get("favor", {})
    add("systematic_favoring", bool(fav.get("tested")) and max(fav.get("disagree_pred_matches_C", 0.0), fav.get("disagree_pred_matches_R", 0.0)) >= FAVOR_SYSTEMATIC,
        "disagree preds match one component ≥ 90%",
        f"C {fav.get('disagree_pred_matches_C', 0.0) * 100:.1f}%, R {fav.get('disagree_pred_matches_R', 0.0) * 100:.1f}%")
    hd = agg.get("heads", {})
    add("head_inexpressive", bool(hd.get("tested")) and hd.get("linear_disagree_RC", 1.0) < NONLINEAR_SUCCESS and hd.get("nonlinear_disagree_RC", 1.0) < NONLINEAR_SUCCESS,
        "linear AND nonlinear fail RC-disagree (<60%)",
        f"lin {hd.get('linear_disagree_RC', 0.0) * 100:.1f}%, nl {hd.get('nonlinear_disagree_RC', 0.0) * 100:.1f}%")
    obj = agg.get("objective", {})
    add("shortcut_incentive", bool(obj.get("tested")) and abs(obj.get("model_train_RC", 0.0) - obj.get("rule_C_train_RC", 1.0)) <= SHORTCUT_TOL,
        "model train RC within 3pp of the feasible C-rule (R-rule is label-oracle)",
        f"model {obj.get('model_train_RC', 0.0) * 100:.1f}% vs C-rule {obj.get('rule_C_train_RC', 0.0) * 100:.1f}%")
    sem = agg.get("semantics", {})
    add("semantic_asymmetry", bool(sem.get("tested")) and sem.get("parity_fact", False),
        "RC-disagree labels equal R by construction (verify, not fail)",
        f"parity_fact={sem.get('parity_fact', False)}, erasure={sem.get('erasure_demonstrated', False)}, "
        f"align R/RC/FRC={[round(sem.get(k, 0.0) * 100, 1) for k in ('align_R', 'align_RC', 'align_FRC')]}")
    cf = agg.get("counterfactual", {})
    add("counterfactual_unresponsive", bool(cf.get("tested")) and cf.get("R_swap_response", 1.0) < 0.20 and cf.get("C_swap_response", 0.0) >= 0.20,
        "R-swap changes prediction < 20% while C-swap changes ≥ 20% (ignores R)",
        f"R-swap Δ {cf.get('R_swap_response', 0.0) * 100:.1f}%, C-swap Δ {cf.get('C_swap_response', 0.0) * 100:.1f}%")
    unstable = [f"{s}/{l}" for s, dd in (agg.get("seed_stability") or {}).items() for l, sd in dd.items() if sd >= SEED_STD_UNSTABLE]
    add("seed_instability", len(unstable) > 0, "any seed-SD ≥ 5pp", "; ".join(unstable) if unstable else "all stable")
    return rows
