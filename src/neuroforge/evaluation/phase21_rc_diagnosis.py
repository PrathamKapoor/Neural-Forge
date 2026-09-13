"""Phase 21 RC relational-capacity and decision-conversion diagnosis metrics.

Pure evaluators; no Phase 1-20 module is modified. All thresholds below are
pre-registered; H1-H8 verdicts, the H8 gate, and CASE A-F are programmatic
functions of executed-experiment data.

North-star chain under test (first broken link = bottleneck):
RC valid -> R observable -> R represented -> R reaches fusion ->
R+C jointly encode RC -> RC rep reaches head -> head converts correctly ->
decision changes correctly under R counterfactuals.
"""
from __future__ import annotations

import statistics
from typing import Any

import torch
from torch import nn

FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
PURE_FAMS = ("F", "R", "C")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")

# ---------------------------------------------------------------------------
# Pre-registered thresholds (absolute fractions; pp = percentage points)
# ---------------------------------------------------------------------------
REP_TOL = 0.03            # reproduction gate tolerance per family
R_DEC_HIGH = 0.80         # R-component "highly decodable"
RC_HEAD_HIGH = 0.80       # RC-target "high" from a frozen head
RC_HEAD_LOW = 0.60        # RC-target "low" from a frozen head
JOINT_HIGH = 0.60         # joint (sr,sc) probe "high"
AGREE_HIGH = 0.90         # RC-agree "high"
DIS_LOW = 0.20            # RC-disagree "collapsed"
STARVE = 0.05             # reused Phase 16 gradient-starvation threshold
DOM_RATIO = 3.0           # C/R grad-share ratio flag (reported, not a verdict alone)
HEAD_BEATS_PROD = 0.10    # frozen-head margin over production for CASE C
H1_GAIN = 0.02            # RC improvement counting as capacity signal
H1_GAIN_PARTIAL = 0.01
H8_GAIN = 0.02            # minimal RC improvement for the intervention
H8_REGRESS = 0.02         # tolerated FRC/mixed regression
SEED_CONSISTENT = 3       # 3/3 directional consistency required

VALID_STATUSES = ("SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTED")
VALID_CASES = ("CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F")


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def implied_label(sr: int, sc: int) -> int:
    """Official RC label algebra: agreement -> shared side, else R side.

    Disagree-equals-R holds identically by the index-parity tiebreak
    (Phase 18, algebraic not empirical).
    """
    if sr == sc:
        return 1 if sr > 0 else 0
    return 1 if sr > 0 else 0


@torch.no_grad()
def split_metrics(
    preds: torch.Tensor,
    eval_ds: Any,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, float]:
    """Per-family accuracy plus RC agree/disagree split accuracy."""
    targets = eval_ds.targets
    fams = eval_ds.families_list
    out: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    rc = [i for i, fm in enumerate(fams) if fm == "RC"]
    agree = [i for i in rc if eval_ds.items[i]["sr"] == eval_ds.items[i]["sc"]]
    disagree = [i for i in rc if eval_ds.items[i]["sr"] != eval_ds.items[i]["sc"]]
    out["RC_agree"] = float((preds[agree] == targets[agree]).float().mean().item()) if agree else 0.0
    out["RC_disagree"] = float((preds[disagree] == targets[disagree]).float().mean().item()) if disagree else 0.0
    out["n_RC_agree"] = len(agree)
    out["n_RC_disagree"] = len(disagree)
    return out


@torch.no_grad()
def stat_sr_repaired(features: torch.Tensor, channel: int = 5) -> torch.Tensor:
    """Deterministic R-component statistic on repaired inputs (ch5 carrier)."""
    x = features[:, :, channel].float()
    return ((x * torch.roll(x, shifts=-1, dims=1)).mean(dim=1) > 0).long()


@torch.no_grad()
def analytical_rc_baselines(eval_ds: Any) -> dict[str, dict[str, float]]:
    """Non-neural diagnostic baselines on repaired RC (train-free rules).

    R-only: predict the sr side (from the ch5 statistic). C-only: predict the
    sc side (from ch3 votes is unavailable without a probe, so C-only uses the
    probed-sc fitted below is NOT done here; instead majority-side constant is
    reported by the runner). Oracle combination: majority + parity algebra.
    Here: exact rule accuracies from ground-truth components (privileged,
    clearly separated from trainable models).
    """
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    agree = [i for i in rc if eval_ds.items[i]["sr"] == eval_ds.items[i]["sc"]]
    disagree = [i for i in rc if eval_ds.items[i]["sr"] != eval_ds.items[i]["sc"]]
    targets = eval_ds.targets
    out: dict[str, dict[str, float]] = {}
    # R-only rule: sr side everywhere.
    r_pred = torch.tensor([1 if eval_ds.items[i]["sr"] > 0 else 0 for i in rc])
    # C-only rule: sc side everywhere.
    c_pred = torch.tensor([1 if eval_ds.items[i]["sc"] > 0 else 0 for i in rc])
    # Oracle combination: exact label algebra.
    o_pred = torch.tensor([implied_label(eval_ds.items[i]["sr"], eval_ds.items[i]["sc"]) for i in rc])
    yt = targets[rc]
    for name, p in (("R_only", r_pred), ("C_only", c_pred), ("oracle_combo", o_pred)):
        acc = float((p == yt).float().mean().item()) if rc else 0.0
        ag = [i for i, j in enumerate(rc) if j in agree]
        dg = [i for i, j in enumerate(rc) if j in disagree]
        out[name] = {
            "RC": acc,
            "RC_agree": float((p[ag] == yt[ag]).float().mean().item()) if ag else 0.0,
            "RC_disagree": float((p[dg] == yt[dg]).float().mean().item()) if dg else 0.0,
        }
    return out


def counterfactual_swap_rates(
    expert: nn.Module,
    eval_ds: Any,
    seed: int = 0,
) -> dict[str, float]:
    """Input-level R/C swaps on repaired RC with change + correctness rates.

    R-swap: exchange ch5 with an opposite-sr donor matched on sc (C held fixed).
    C-swap: exchange ch0:4 with an opposite-sc donor matched on sr (R held fixed).
    Control: same-(sr,sc) donor swap (expect ~no change).
    Rates: change (pred flips), correct-change (flips toward new implied target),
    incorrect-change (flips away). Uses the production head (no new parameters).
    """
    from collections import defaultdict

    expert.eval()
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    disagree = [i for i in rc if eval_ds.items[i]["sr"] != eval_ds.items[i]["sc"]]
    pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in rc:
        pools[(eval_ds.items[i]["sr"], eval_ds.items[i]["sc"])].append(i)
    g = torch.Generator().manual_seed(seed + 2121)
    out: dict[str, float] = {}
    with torch.no_grad():
        base = expert(eval_ds.features).argmax(-1)

        def run(which: str, opposite: bool) -> dict[str, float]:
            ch = cng = tot = 0
            for i in disagree:
                sr, sc = eval_ds.items[i]["sr"], eval_ds.items[i]["sc"]
                if which == "R":
                    key = (-sr, sc) if opposite else (sr, sc)
                else:
                    key = (sr, -sc) if opposite else (sr, sc)
                donors = [j for j in pools.get(key, []) if j != i]
                if not donors:
                    continue
                j = donors[torch.randint(len(donors), (1,), generator=g).item()]
                mod = eval_ds.features[i].clone()
                if which == "R":
                    mod[:, 5] = eval_ds.features[j][:, 5]
                    nsr, nsc = eval_ds.items[j]["sr"], sc
                else:
                    mod[:, 0:5] = eval_ds.features[j][:, 0:5]
                    nsr, nsc = sr, eval_ds.items[j]["sc"]
                new = int(expert(mod.unsqueeze(0)).argmax(-1).item())
                old = int(base[i].item())
                if new != old:
                    ch += 1
                    if new == implied_label(nsr, nsc):
                        cng += 1
                tot += 1
            return {
                "change": ch / max(1, tot),
                "correct_change": cng / max(1, tot),
                "incorrect_change": (ch - cng) / max(1, tot),
                "n": tot,
            }

        for tag, which, opp in (("R", "R", True), ("C", "C", True), ("control", "R", False)):
            s = run(which, opp)
            out[f"{tag}_change"] = s["change"]
            out[f"{tag}_correct_change"] = s["correct_change"]
            out[f"{tag}_incorrect_change"] = s["incorrect_change"]
            out[f"{tag}_n"] = s["n"]
    return out


def grad_norms_unfrozen_clone(
    expert: nn.Module, probe_x: torch.Tensor, probe_y: torch.Tensor
) -> dict[str, float]:
    """Per-group grad norms on a deepcopy with grads temporarily enabled.

    The original (frozen) expert is never modified: no optimizer step is taken
    and the clone is discarded. Reuses Phase 16 group mapping.
    """
    import copy

    from neuroforge.evaluation.phase16_metrics import GRAD_GROUPS, grad_group_of

    clone = copy.deepcopy(expert)
    clone.train()
    for p in clone.parameters():
        p.requires_grad_(True)
    clone.zero_grad()
    logits = clone(probe_x)
    loss = torch.nn.functional.cross_entropy(logits, probe_y)
    loss.backward()
    norms: dict[str, float] = {g: 0.0 for g in GRAD_GROUPS}
    for name, p in clone.named_parameters():
        if p.grad is None:
            continue
        norms[grad_group_of(name)] += float(p.grad.norm().item()) ** 2
    return {g: float(v**0.5) for g, v in norms.items()}


@torch.no_grad()
def branch_rms(expert: nn.Module, features: torch.Tensor) -> dict[str, float]:
    """RMS activation of the relational vs contextual branch deltas."""
    expert.eval()
    enc = expert.encoder(features)
    block = expert.blocks[0]
    if hasattr(block, "forward_with_intermediates"):
        inter = block.forward_with_intermediates(enc, features)
        out: dict[str, float] = {}
        for k in ("rel_delta", "ctx_delta", "feat_delta", "fused_delta", "block_output"):
            if k in inter and isinstance(inter[k], torch.Tensor):
                out[k] = float(inter[k].float().pow(2).mean().sqrt().item())
        return out
    return {}


# ---------------------------------------------------------------------------
# Hypothesis builders (all programmatic, pre-registered thresholds)
# ---------------------------------------------------------------------------
def _consistency(vals: list[float], thresh: float) -> int:
    return sum(1 for v in vals if v >= thresh)


def build_phase21_hypotheses(agg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n = int(agg.get("n_seeds", 3))

    h1 = agg.get("capacity", {})
    if h1.get("tested"):
        gains = h1.get("RC_gains", [])
        mg = mean(gains)
        npos = sum(1 for g in gains if g > 0)
        r_kept = h1.get("R_kept", True)
        if mg >= H1_GAIN and npos >= SEED_CONSISTENT and r_kept:
            out["H1"] = {"status": "SUPPORTED",
                         "evidence": f"RC improves with relational depth by {mg * 100:+.1f}pp mean, {npos}/{n} seeds positive, R preserved."}
        elif mg >= H1_GAIN_PARTIAL and npos >= max(2, n - 1):
            out["H1"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"RC improves modestly with depth ({mg * 100:+.1f}pp, {npos}/{n} seeds)."}
        else:
            out["H1"] = {"status": "NOT SUPPORTED",
                         "evidence": f"No specific RC improvement with depth ({mg * 100:+.1f}pp, {npos}/{n} seeds positive)."}
    else:
        out["H1"] = {"status": "NOT TESTED", "evidence": "Capacity variants not executed."}

    h2 = agg.get("alignment", {})
    if h2.get("tested"):
        r_dec = h2.get("R_dec", 0.0)
        rc_dec = h2.get("RC_dec", 0.0)
        if r_dec >= R_DEC_HIGH and rc_dec < RC_HEAD_LOW:
            out["H2"] = {"status": "SUPPORTED",
                         "evidence": f"R decodable ({r_dec * 100:.1f}%) but RC-target decodability low ({rc_dec * 100:.1f}%): decision-conversion signature."}
        elif r_dec >= R_DEC_HIGH:
            out["H2"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"R decodable ({r_dec * 100:.1f}%), RC-target {rc_dec * 100:.1f}%."}
        else:
            out["H2"] = {"status": "NOT SUPPORTED",
                         "evidence": f"R itself poorly decodable ({r_dec * 100:.1f}%): information missing, not conversion."}
    else:
        out["H2"] = {"status": "NOT TESTED", "evidence": "Stage probes not executed."}

    h3 = agg.get("dominance", {})
    if h3.get("tested"):
        ratio = h3.get("C_over_R_grad", 0.0)
        r_share = h3.get("R_share", 1.0)
        if ratio >= DOM_RATIO and r_share < 0.10:
            out["H3"] = {"status": "SUPPORTED",
                         "evidence": f"C/R grad-norm ratio {ratio:.1f}x with R share {r_share * 100:.1f}%: dominance + weak relational signal."}
        elif ratio >= 2.0 or r_share < STARVE:
            out["H3"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"C/R ratio {ratio:.1f}x, R share {r_share * 100:.1f}% (continuous; weak dominance signal)."}
        else:
            out["H3"] = {"status": "NOT SUPPORTED",
                         "evidence": f"C/R ratio {ratio:.1f}x, R share {r_share * 100:.1f}%: no dominance."}
    else:
        out["H3"] = {"status": "NOT TESTED", "evidence": "Gradient instrumentation not executed."}

    h4 = agg.get("counterfactual", {})
    if h4.get("tested"):
        rcc = h4.get("R_correct_change", 0.0)
        if rcc >= 0.60 and h4.get("R_change", 0.0) > h4.get("control_change", 1.0):
            out["H4"] = {"status": "SUPPORTED",
                         "evidence": f"R changes convert correctly {rcc * 100:.1f}% of the time: R is used in the RC decision."}
        elif h4.get("R_change", 0.0) <= max(h4.get("control_change", 0.0), 0.10):
            out["H4"] = {"status": "NOT SUPPORTED",
                         "evidence": f"R counterfactuals rarely change the decision ({h4.get('R_change', 0.0) * 100:.1f}% vs control {h4.get('control_change', 0.0) * 100:.1f}%): conversion/use failure."}
        else:
            out["H4"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"R changes decision {h4.get('R_change', 0.0) * 100:.1f}% but correctly only {rcc * 100:.1f}%."}
    else:
        out["H4"] = {"status": "NOT TESTED", "evidence": "Counterfactuals not executed."}

    h5 = agg.get("heads", {})
    if h5.get("tested"):
        lin, nonlin, comp = h5.get("linear_RC", 0.0), h5.get("nonlinear_RC", 0.0), h5.get("component_RC", 0.0)
        best = max(lin, nonlin, comp)
        if best >= RC_HEAD_HIGH:
            out["H5"] = {"status": "SUPPORTED",
                         "evidence": f"Frozen rep + tiny head reaches RC {best * 100:.1f}% (lin {lin * 100:.1f} / nl {nonlin * 100:.1f} / comp {comp * 100:.1f}): encoder holds joint info."}
        elif best >= RC_HEAD_LOW:
            out["H5"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Best frozen head RC {best * 100:.1f}% (below {RC_HEAD_HIGH * 100:.0f}% bar)."}
        else:
            out["H5"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Frozen heads fail RC (best {best * 100:.1f}%): problem precedes the head."}
    else:
        out["H5"] = {"status": "NOT TESTED", "evidence": "Diagnostic heads not executed."}

    h6 = agg.get("locations", {})
    if h6.get("tested"):
        rows = h6.get("rows", {})
        rel_r = rows.get("relational_output", {}).get("R_dec", 0.0)
        rel_rc = rows.get("relational_output", {}).get("RC_dec", 0.0)
        fin_rc = rows.get("final_representation", {}).get("RC_dec", 0.0)
        if rel_r >= R_DEC_HIGH and rel_rc >= JOINT_HIGH:
            out["H6"] = {"status": "SUPPORTED",
                         "evidence": f"Relational output carries R ({rel_r * 100:.1f}%) and RC ({rel_rc * 100:.1f}%) locally."}
        elif rel_r >= R_DEC_HIGH:
            out["H6"] = {"status": "PARTIALLY SUPPORTED",
                         "evidence": f"Relational output carries R ({rel_r * 100:.1f}%) but RC only {rel_rc * 100:.1f}% (final {fin_rc * 100:.1f}%)."}
        else:
            out["H6"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Relational output lacks R ({rel_r * 100:.1f}%)."}
    else:
        out["H6"] = {"status": "NOT TESTED", "evidence": "Location probes not executed."}

    h7 = agg.get("order", {})
    if h7.get("tested"):
        spread = h7.get("spread_RC", 0.0)
        if h7.get("separable", False) and spread >= H1_GAIN:
            out["H7"] = {"status": "SUPPORTED",
                         "evidence": f"Order changes RC usability by {spread * 100:+.1f}pp with controlled readout."}
        elif not h7.get("separable", True):
            out["H7"] = {"status": "INCONCLUSIVE",
                         "evidence": "Order effects inseparable from expert identity/optimization noise."}
        else:
            out["H7"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Controlled order spread only {spread * 100:+.1f}pp on RC."}
    else:
        out["H7"] = {"status": "NOT TESTED", "evidence": "Order experiment not executed."}

    h8 = agg.get("intervention", {})
    if h8.get("tested"):
        g = h8.get("RC_gain", 0.0)
        reg = h8.get("max_regression", 1.0)
        if g >= H8_GAIN and h8.get("n_pos", 0) >= SEED_CONSISTENT and reg <= H8_REGRESS:
            out["H8"] = {"status": "SUPPORTED",
                         "evidence": f"Candidate improves RC by {g * 100:+.1f}pp ({h8.get('n_pos', 0)}/{n} seeds), max regression {reg * 100:.1f}pp."}
        else:
            out["H8"] = {"status": "NOT SUPPORTED",
                         "evidence": f"Candidate RC gain {g * 100:+.1f}pp, max regression {reg * 100:.1f}pp."}
    else:
        out["H8"] = {"status": "NOT TESTED", "evidence": "H8 gate did not pass; candidate not trained."}
    return out


def h8_gate(agg: dict[str, Any]) -> dict[str, Any]:
    """H8 intervention gate: H1 must implicate capacity AND R must not be starved.

    Gate passes iff H1 == SUPPORTED (monotonic, consistent RC gains with depth,
    R preserved) and the relational grad share >= STARVE (capacity is used, so
    more of it can help). Otherwise the candidate is not trained.
    """
    hy = agg.get("hypotheses", {})
    dom = agg.get("dominance", {})
    h1_ok = hy.get("H1", {}).get("status") == "SUPPORTED"
    used = dom.get("R_share", 0.0) >= STARVE
    passed = bool(h1_ok and used)
    return {
        "passed": passed,
        "reason": (
            "H1 implicates capacity and relational signal is used (share "
            f"{dom.get('R_share', 0.0) * 100:.1f}% >= {STARVE * 100:.0f}%)."
            if passed else
            f"H1={hy.get('H1', {}).get('status')} (need SUPPORTED); "
            f"R share {dom.get('R_share', 0.0) * 100:.1f}% (need >= {STARVE * 100:.0f}%)."
        ),
    }


def select_phase21_case(hy: dict[str, dict[str, Any]], agg: dict[str, Any]) -> tuple[str, str]:
    def st(h: str) -> str:
        return hy.get(h, {}).get("status", "INCONCLUSIVE")

    if st("H8") == "SUPPORTED":
        return ("CASE D", "Relational capacity bottleneck supported")
    comp_rc = agg.get("heads", {}).get("component_RC", 0.0)
    if st("H5") == "SUPPORTED" and comp_rc >= RC_HEAD_HIGH and comp_rc >= agg.get("production_RC", 0.0) + HEAD_BEATS_PROD:
        return ("CASE C", "Prediction interface bottleneck")
    agr = agg.get("agreement", {})
    if st("H2") == "SUPPORTED" and st("H5") != "SUPPORTED":
        return ("CASE B", "Joint decision/composition bottleneck")
    if agr.get("agree_acc", 0.0) >= AGREE_HIGH and agr.get("disagree_acc", 1.0) <= DIS_LOW:
        return ("CASE E", "Decision-rule / component-conflict bottleneck")
    loc = agg.get("locations", {}).get("rows", {})
    post = loc.get("relational_output", {}).get("RC_dec", 1.0)
    if st("H2") != "SUPPORTED" and post < JOINT_HIGH:
        return ("CASE A", "Representation formation bottleneck")
    return ("CASE F", "UNRESOLVED")


def recommendation_for_case(case: str) -> str:
    return {
        "CASE A": "Strengthen representation formation before any decision work.",
        "CASE B": "Target the minimal R+C combination mechanism, not relational depth.",
        "CASE C": "Modify only the minimal readout/head; keep encoder and depth fixed.",
        "CASE D": "Adopt the smallest successful capacity increase with full cost accounting.",
        "CASE E": "Make R/C-disagreement resolution the primary Phase 22 target.",
        "CASE F": "Run another targeted diagnosis; do not add architecture.",
    }[case]


def build_failure_diagnosis(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(category: str, failed: bool, criterion: str, observed: str) -> None:
        rows.append({"category": category, "failed": bool(failed),
                     "criterion": criterion, "observed": observed})

    rep = agg.get("reproduction", {})
    add("reproduction_breach", bool(rep.get("tested")) and not rep.get("passed", False),
        f"Phase 20 values within {REP_TOL * 100:.0f}pp", rep.get("detail", ""))
    cap = agg.get("capacity", {})
    add("capacity_no_signal", bool(cap.get("tested")) and mean(cap.get("RC_gains", [0.0])) < H1_GAIN_PARTIAL,
        "mean depth RC gain < 1pp", f"{mean(cap.get('RC_gains', [0.0])) * 100:+.1f}pp")
    al = agg.get("alignment", {})
    add("conversion_gap", bool(al.get("tested")) and al.get("R_dec", 0.0) >= R_DEC_HIGH
        and al.get("RC_dec", 0.0) < RC_HEAD_LOW,
        "R decodable but RC-target head weak", f"R {al.get('R_dec', 0.0) * 100:.1f}% vs RC {al.get('RC_dec', 0.0) * 100:.1f}%")
    dom = agg.get("dominance", {})
    add("gradient_starvation", bool(dom.get("tested")) and dom.get("R_share", 1.0) < STARVE,
        "relational grad share < 5% (reused Phase 16 bar)", f"{dom.get('R_share', 0.0) * 100:.1f}%")
    cf = agg.get("counterfactual", {})
    add("r_change_unresponsive", bool(cf.get("tested")) and cf.get("R_change", 1.0) < 0.20,
        "R-swap change rate < 20%", f"{cf.get('R_change', 0.0) * 100:.1f}%")
    hd = agg.get("heads", {})
    add("head_inexpressive", bool(hd.get("tested"))
        and max(hd.get("linear_RC", 0.0), hd.get("nonlinear_RC", 0.0), hd.get("component_RC", 0.0)) < RC_HEAD_LOW,
        "all frozen heads RC < 60%", f"best {max(hd.get('linear_RC', 0.0), hd.get('nonlinear_RC', 0.0), hd.get('component_RC', 0.0)) * 100:.1f}%")
    agr = agg.get("agreement", {})
    gap = agr.get("agree_acc", 0.0) - agr.get("disagree_acc", 0.0)
    add("disagree_collapse", bool(agr.get("tested")) and gap >= 0.50,
        "agree−disagree gap ≥ 50pp", f"{gap * 100:.1f}pp")
    return rows
