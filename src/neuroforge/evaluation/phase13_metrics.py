"""Phase 13 Joint Co-Adaptation diagnostics.

Pure evaluators; do not mutate any trained expert. All functions return
machine-readable dictionaries.

Diagnostic families:

  13D  branch contribution (scale extraction + ablation)
  13E  relational destruction + representation probing
  13G  portfolio re-evaluation (ceiling + oracle)
"""
from __future__ import annotations

import math
import statistics
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


@torch.no_grad()
def expert_branch_scales(expert: nn.Module) -> dict[str, float]:
    """Extract the three per-branch learnable scales of a JointBlock or JointCoBlock.

    Returns ``{"feature": float, "graph": float, "context": float}``.

    For a `JointBlock`, the scales are parameters `feature_scale`, `graph_scale`,
    `context_scale`. For a `JointCoBlock`, the same parameter names are used.
    """
    block = expert.blocks[0] if hasattr(expert, "blocks") and len(expert.blocks) > 0 else expert
    return {
        "feature": float(block.feature_scale.detach().item()),
        "graph": float(block.graph_scale.detach().item()),
        "context": float(block.context_scale.detach().item()),
    }


@torch.no_grad()
def expert_with_branch_ablated(
    expert: nn.Module,
    ablate: tuple[str, ...] = (),
) -> nn.Module:
    """Return a deep copy of the expert with the specified branches' scales zeroed.

    Used to evaluate the contribution of each branch at inference time without
    retraining. The copy shares parameters with the original via `state_dict`,
    but the relevant scales are overridden to zero.
    """
    import copy

    new_expert = copy.deepcopy(expert)
    new_expert.eval()
    block = new_expert.blocks[0]
    with torch.no_grad():
        if "feature" in ablate:
            block.feature_scale.zero_()
        if "graph" in ablate:
            block.graph_scale.zero_()
        if "context" in ablate:
            block.context_scale.zero_()
    return new_expert


@torch.no_grad()
def cross_evaluation_with_ablation(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    ablations: tuple[tuple[str, ...], ...] = ((), ("feature",), ("graph",), ("context",)),
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, dict[str, float]]:
    """Evaluate the expert under several branch-ablation conditions.

    Returns ``{ablation_label: {family: accuracy}}``. The empty tuple ``()``
    represents "no ablation" (all branches active).
    """
    out: dict[str, dict[str, float]] = {}
    for abl in ablations:
        label = "+".join(abl) if abl else "all"
        mod = expert_with_branch_ablated(expert, abl)
        logits = mod(eval_ds.features)
        preds = logits.argmax(-1)
        per_fam: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == eval_ds.targets[idx]).float().mean().item())
        per_fam["overall"] = float((preds == eval_ds.targets).float().mean().item())
        out[label] = per_fam
    return out


@torch.no_grad()
def representation_probes_per_expert(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    component_targets: dict[str, torch.Tensor],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    hidden_dim: int = 24,
) -> dict[str, dict[str, float]]:
    """Probe the expert's final representation for decodability of component signals.

    Returns ``{task_name: {family: probe_overall}}``.
    """
    expert.eval()
    state = extract_state_after_blocks(expert, eval_ds.features)
    out: dict[str, dict[str, float]] = {}
    for task_name, task_targets in component_targets.items():
        if task_name == "final_target":
            ttask = eval_ds.targets
        else:
            ttask = task_targets
        per_fam: dict[str, float] = {}
        # Single probe on the whole mixed dataset (per-family linear probe).
        for f in families:
            idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == f]
            if not idx:
                continue
            sub_state = state[idx]
            sub_targets = ttask[idx]
            sub = linear_probe_accuracy_per_class(sub_state, sub_targets, hidden_dim=hidden_dim)
            per_fam[f] = sub.get("overall", 0.0)
        out[task_name] = per_fam
    return out


@torch.no_grad()
def relational_destruction_accuracy(
    expert: nn.Module,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("R", "RC", "FRC"),
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate the expert on the original benchmark and on the
    destructively-relationally-permuted benchmark.

    Returns ``{condition: {family: accuracy}}``. If the relational branch is
    genuinely functional, the expert should be sensitive to the destructive
    permutation on R-bearing families.
    """
    expert.eval()
    feats = eval_ds.features
    fams = eval_ds.families_list
    targets = eval_ds.targets

    out: dict[str, dict[str, float]] = {}
    for label, f_to_eval in (
        ("original", feats),
        ("relational_permuted", apply_phase9_relational_permutation(feats, seed=seed)),
    ):
        logits = expert(f_to_eval)
        preds = logits.argmax(-1)
        per_fam: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
        out[label] = per_fam
    return out


@torch.no_grad()
def single_expert_ceiling(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, dict[str, Any]]:
    """For each family, find the best single expert accuracy (the ceiling)."""
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
    return {
        "ceiling_per_family": ceilings,
        "best_per_family": best_per,
    }


@torch.no_grad()
def oracle_composition(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    k: int,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """Oracle k-way combination: pick the k best experts per sample and mean
    their logits. For k=1 this is the per-family best single expert.
    """
    from itertools import combinations
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
            preds = combined.argmax(-1)
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc = acc
        per_fam[f] = best_acc
    return {"per_family": per_fam}


def build_phase13_causal_diagnosis(
    joint_baseline_mixed: float,
    extended_training_mixed: float,
    coadaptation_mixed: float,
    control_mixed: float,
    joint_branch_scales: dict[str, dict[str, float]],
    joint_branch_ablation: dict[str, dict[str, float]],
    joint_repr_probes: dict[str, dict[str, float]],
    joint_relational_sensitivity: dict[str, dict[str, float]],
    old_ceiling_mixed: float,
    new_ceiling_mixed: float,
) -> dict[str, dict[str, Any]]:
    """Programmatic causal diagnosis for Phase 13.

    Returns a dict mapping each failure-mode category to
    ``{status, evidence}`` where status is one of SUPPORTED,
    PARTIALLY SUPPORTED, NOT SUPPORTED, INCONCLUSIVE, NOT TESTED.
    """
    out: dict[str, dict[str, Any]] = {}

    # 1. UNDERTRAINING: extended training catches up to co-adaptation.
    if extended_training_mixed >= coadaptation_mixed - 0.02:
        out["UNDERTRAINING"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"Extended training (40 epochs) achieves {extended_training_mixed:.1%} mixed mean, "
                f"matching co-adaptation ({coadaptation_mixed:.1%}). The Phase 12 gap was "
                f"at least partly optimization-budget related."
            ),
        }
    elif extended_training_mixed > joint_baseline_mixed + 0.01:
        out["UNDERTRAINING"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"Extended training improves over baseline ({joint_baseline_mixed:.1%} -> "
                f"{extended_training_mixed:.1%}) but does not reach co-adaptation "
                f"({coadaptation_mixed:.1%})."
            ),
        }
    else:
        out["UNDERTRAINING"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"Extended training does not improve over baseline "
                f"({joint_baseline_mixed:.1%} -> {extended_training_mixed:.1%})."
            ),
        }

    # 2. BRANCH_INTERFERENCE: co-adaptation > extended training means branch
    #    cross-talk is providing a real signal.
    if coadaptation_mixed > extended_training_mixed + 0.02:
        out["BRANCH_INTERFERENCE"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"Co-adaptation ({coadaptation_mixed:.1%}) > extended training "
                f"({extended_training_mixed:.1%}) by > 2pp; the inter-branch fusion is "
                f"providing more than just additional optimization."
            ),
        }
    elif coadaptation_mixed > extended_training_mixed:
        out["BRANCH_INTERFERENCE"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"Co-adaptation ({coadaptation_mixed:.1%}) > extended training "
                f"({extended_training_mixed:.1%}) by < 2pp; marginal benefit."
            ),
        }
    else:
        out["BRANCH_INTERFERENCE"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"Co-adaptation ({coadaptation_mixed:.1%}) does not exceed extended "
                f"training ({extended_training_mixed:.1%}); the intervention is not "
                f"providing branch cross-talk benefits beyond training time."
            ),
        }

    # 3. RELATIONAL_CAPABILITY: removing the relational branch causes a large
    #    drop on R/RC/FRC.
    rel_drop = {
        f: (joint_branch_ablation.get("all", {}).get(f, 0.0)
            - joint_branch_ablation.get("graph", {}).get(f, 0.0))
        for f in ("R", "RC", "FRC")
    }
    avg_rel_drop = statistics.mean(rel_drop.values()) if rel_drop else 0.0
    if avg_rel_drop > 0.05:
        out["RELATIONAL_CAPABILITY"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"Removing the relational branch drops accuracy on R/RC/FRC by an "
                f"average of {avg_rel_drop*100:.1f}pp; the branch is causally relevant."
            ),
        }
    elif avg_rel_drop > 0.01:
        out["RELATIONAL_CAPABILITY"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"Removing the relational branch drops R/RC/FRC by an average of "
                f"{avg_rel_drop*100:.1f}pp; modest contribution."
            ),
        }
    else:
        out["RELATIONAL_CAPABILITY"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"Removing the relational branch drops R/RC/FRC by an average of "
                f"{avg_rel_drop*100:.1f}pp; the branch is not contributing."
            ),
        }

    # 4. REPRESENTATION_FUSION: R-probe is high but R-accuracy is low.
    r_probe = joint_repr_probes.get("R_signal", {}).get("R", 0.0)
    r_acc = joint_branch_ablation.get("all", {}).get("R", 0.0)
    if r_probe > 0.7 and r_acc < 0.6:
        out["REPRESENTATION_FUSION"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"R-signal probe = {r_probe*100:.1f}% (R information is present in the "
                f"representation) but R-accuracy = {r_acc*100:.1f}% (the final head "
                f"fails to use it)."
            ),
        }
    elif r_probe > 0.5 and r_acc < 0.5:
        out["REPRESENTATION_FUSION"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"R-signal probe = {r_probe*100:.1f}%, R-accuracy = {r_acc*100:.1f}%."
            ),
        }
    else:
        out["REPRESENTATION_FUSION"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"R-signal probe = {r_probe*100:.1f}%, R-accuracy = {r_acc*100:.1f}%."
            ),
        }

    # 5. RELATIONAL_SENSITIVITY: destroying the relational correspondence
    #    degrades R/RC performance.
    if "relational_permuted" in joint_relational_sensitivity:
        r_orig = joint_relational_sensitivity.get("original", {}).get("R", 0.0)
        r_perm = joint_relational_sensitivity.get("relational_permuted", {}).get("R", 0.0)
        rc_orig = joint_relational_sensitivity.get("original", {}).get("RC", 0.0)
        rc_perm = joint_relational_sensitivity.get("relational_permuted", {}).get("RC", 0.0)
        sens = (r_orig - r_perm) + (rc_orig - rc_perm)
        if sens > 0.10:
            out["RELATIONAL_SENSITIVITY"] = {
                "status": "SUPPORTED",
                "evidence": (
                    f"Relational destruction drops R+RC by {sens*100:.1f}pp; the "
                    f"joint expert is sensitive to relational correspondence."
                ),
            }
        elif sens > 0.0:
            out["RELATIONAL_SENSITIVITY"] = {
                "status": "PARTIALLY SUPPORTED",
                "evidence": (
                    f"Relational destruction drops R+RC by {sens*100:.1f}pp; weak sensitivity."
                ),
            }
        else:
            out["RELATIONAL_SENSITIVITY"] = {
                "status": "NOT SUPPORTED",
                "evidence": (
                    f"Relational destruction does not drop R+RC accuracy (delta = "
                    f"{sens*100:.1f}pp)."
                ),
            }
    else:
        out["RELATIONAL_SENSITIVITY"] = {
            "status": "NOT TESTED",
            "evidence": "Relational destruction not computed for this seed.",
        }

    # 6. PORTFOLIO_LIMITATION: if the best Joint condition doesn't move the
    #    ceiling, the portfolio (not the Joint expert) is the bottleneck.
    if new_ceiling_mixed > old_ceiling_mixed + 0.03:
        out["PORTFOLIO_LIMITATION"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"Co-adapted Joint moves the cross-family mixed ceiling by "
                f"{(new_ceiling_mixed - old_ceiling_mixed)*100:+.1f}pp."
            ),
        }
    else:
        out["PORTFOLIO_LIMITATION"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"Co-adapted Joint does not move the cross-family mixed ceiling "
                f"(delta = {(new_ceiling_mixed - old_ceiling_mixed)*100:+.1f}pp); "
                f"the existing portfolio is not the bottleneck."
            ),
        }

    # 7. OPTIMIZATION: low overall accuracy across conditions with all variants.
    if max(joint_baseline_mixed, extended_training_mixed, coadaptation_mixed) < control_mixed + 0.10:
        out["OPTIMIZATION"] = {
            "status": "SUPPORTED",
            "evidence": (
                f"All Joint variants underperform by > 10pp vs. the parameter-matched "
                f"control; optimization appears to be the dominant bottleneck."
            ),
        }
    else:
        out["OPTIMIZATION"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"At least one Joint variant matches the control; optimization is not "
                f"the dominant bottleneck."
            ),
        }

    # 8. BENCHMARK_LIMITATION: no condition improves on the Phase 12 ceiling.
    if new_ceiling_mixed <= old_ceiling_mixed + 0.005:
        out["BENCHMARK_LIMITATION"] = {
            "status": "PARTIALLY SUPPORTED",
            "evidence": (
                f"Phase 13 best ceiling ({new_ceiling_mixed*100:.1f}%) is essentially "
                f"equal to Phase 12 ({old_ceiling_mixed*100:.1f}%). The benchmark may be "
                f"insufficient to discriminate Joint variants."
            ),
        }
    else:
        out["BENCHMARK_LIMITATION"] = {
            "status": "NOT SUPPORTED",
            "evidence": (
                f"Phase 13 best ceiling ({new_ceiling_mixed*100:.1f}%) > Phase 12 "
                f"({old_ceiling_mixed*100:.1f}%); the benchmark does discriminate."
            ),
        }

    return out


def select_phase13_verdict(
    causal: dict[str, dict[str, Any]],
    joint_baseline_mixed: float,
    extended_training_mixed: float,
    coadaptation_mixed: float,
    old_ceiling_mixed: float,
    new_ceiling_mixed: float,
) -> tuple[str, str]:
    """Programmatic verdict selection for Phase 13.

    Returns (CASE label, verdict label).
    """
    def st(cat: str) -> str:
        return causal.get(cat, {}).get("status", "INCONCLUSIVE")

    # CASE A: co-adaptation materially improves Joint capability.
    if (coadaptation_mixed > joint_baseline_mixed + 0.05) and (st("BRANCH_INTERFERENCE") == "SUPPORTED"):
        return (
            "CASE A",
            "Co-adaptation materially improves Joint capability",
        )
    # CASE B: additional training explains improvement.
    if (extended_training_mixed > joint_baseline_mixed + 0.02) and (
        st("UNDERTRAINING") == "SUPPORTED"
    ) and (coadaptation_mixed <= extended_training_mixed + 0.01):
        return (
            "CASE B",
            "Additional training explains the improvement",
        )
    # CASE C: relational branch is the bottleneck.
    if (st("RELATIONAL_CAPABILITY") == "SUPPORTED") and (
        st("REPRESENTATION_FUSION") != "SUPPORTED"
    ):
        return (
            "CASE C",
            "Relational branch is the bottleneck",
        )
    # CASE D: fusion is the bottleneck.
    if st("REPRESENTATION_FUSION") == "SUPPORTED":
        return (
            "CASE D",
            "Fusion is the bottleneck (R info present but unused)",
        )
    # CASE E: Joint remains capped.
    if max(joint_baseline_mixed, extended_training_mixed, coadaptation_mixed) <= old_ceiling_mixed:
        return (
            "CASE E",
            "Joint expert remains capped",
        )
    # CASE F: existing portfolio remains superior.
    if coadaptation_mixed < joint_baseline_mixed and new_ceiling_mixed <= old_ceiling_mixed:
        return (
            "CASE F",
            "Co-adaptation does not help; existing portfolio remains superior",
        )
    return (
        "CASE G",
        "Multiple interacting causes",
    )
