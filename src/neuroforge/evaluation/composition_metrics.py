"""Phase 10 Composition Metrics, Failure Diagnostics, and Counterfactual Analysis."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch


def calculate_active_expert_statistics(k_selections: list[int]) -> dict[str, float]:
    """Calculate average active experts, distribution P(k=1,2,3), and k-entropy."""
    if not k_selections:
        return {"mean_k": 0.0, "p_k1": 0.0, "p_k2": 0.0, "p_k3": 0.0, "k_entropy": 0.0}

    n = len(k_selections)
    mean_k = sum(k_selections) / n
    p1 = sum(1 for k in k_selections if k == 1) / n
    p2 = sum(1 for k in k_selections if k == 2) / n
    p3 = sum(1 for k in k_selections if k >= 3) / n

    # Shannon entropy over k in bits
    k_entropy = 0.0
    for p in (p1, p2, p3):
        if p > 1e-8:
            k_entropy -= p * math.log2(p)

    return {
        "mean_k": mean_k,
        "p_k1": p1,
        "p_k2": p2,
        "p_k3": p3,
        "k_entropy": k_entropy,
    }


def _canonical_pair_key(pair: tuple[str, str]) -> str:
    display_names = {
        "mlp": "MLP",
        "graph": "Graph",
        "attention": "Attn V1",
        "attention_v2": "Attn V2",
    }
    sorted_pair = sorted([display_names.get(p, p) for p in pair])
    return f"{sorted_pair[0]} + {sorted_pair[1]}"


def _canonical_triple_key(triple: tuple[str, str, str]) -> str:
    display_names = {
        "mlp": "MLP",
        "graph": "Graph",
        "attention": "Attn V1",
        "attention_v2": "Attn V2",
    }
    sorted_triple = sorted([display_names.get(t, t) for t in triple])
    return f"{sorted_triple[0]} + {sorted_triple[1]} + {sorted_triple[2]}"


def calculate_pair_utilization(selected_tuples: list[tuple[str, ...]]) -> dict[str, float]:
    """Calculate frequencies of all canonical expert pairs."""
    pair_counts: Counter[str] = Counter()
    total_pairs = 0

    all_canonical_pairs = [
        "Graph + MLP",
        "Attn V1 + MLP",
        "Attn V2 + MLP",
        "Attn V1 + Graph",
        "Attn V2 + Graph",
        "Attn V1 + Attn V2",
    ]
    for cp in all_canonical_pairs:
        pair_counts[cp] = 0

    for tup in selected_tuples:
        if len(tup) == 2:
            key = _canonical_pair_key((tup[0], tup[1]))
            pair_counts[key] += 1
            total_pairs += 1
        elif len(tup) >= 3:
            # If 3 experts are selected, count constituent pairs
            for i in range(len(tup)):
                for j in range(i + 1, len(tup)):
                    key = _canonical_pair_key((tup[i], tup[j]))
                    pair_counts[key] += 1
                    total_pairs += 1

    if total_pairs == 0:
        return {cp: 0.0 for cp in all_canonical_pairs}
    return {cp: pair_counts[cp] / total_pairs for cp in all_canonical_pairs}


def calculate_triple_utilization(selected_tuples: list[tuple[str, ...]]) -> dict[str, float]:
    """Calculate frequencies of all canonical expert triples."""
    all_canonical_triples = [
        "Attn V1 + Graph + MLP",
        "Attn V2 + Graph + MLP",
        "Attn V1 + Attn V2 + MLP",
        "Attn V1 + Attn V2 + Graph",
    ]
    triple_counts: Counter[str] = Counter({ct: 0 for ct in all_canonical_triples})
    total_triples = 0

    for tup in selected_tuples:
        if len(tup) == 3:
            key = _canonical_triple_key((tup[0], tup[1], tup[2]))
            triple_counts[key] += 1
            total_triples += 1

    if total_triples == 0:
        return {ct: 0.0 for ct in all_canonical_triples}
    return {ct: triple_counts[ct] / total_triples for ct in all_canonical_triples}


def calculate_composition_entropy(selected_tuples: list[tuple[str, ...]]) -> float:
    """Calculate Shannon entropy over the full distribution of expert combinations."""
    if not selected_tuples:
        return 0.0

    counts = Counter(tuple(sorted(t)) for t in selected_tuples)
    n = len(selected_tuples)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 1e-8:
            entropy -= p * math.log2(p)
    return entropy


def generate_family_composition_matrix(
    families: list[str],
    selected_tuples: list[tuple[str, ...]],
) -> dict[str, dict[str, float]]:
    """Build task family -> composition frequency matrix."""
    family_types = sorted(list(set(families)))
    matrix: dict[str, dict[str, float]] = {}

    display_names = {
        "mlp": "MLP",
        "graph": "Graph",
        "attention": "V1",
        "attention_v2": "V2",
    }

    for fam in family_types:
        fam_tuples = [selected_tuples[i] for i, f in enumerate(families) if f == fam]
        n_fam = len(fam_tuples)
        if n_fam == 0:
            continue

        comp_strings = []
        for tup in fam_tuples:
            sorted_tup = sorted([display_names.get(x, x) for x in tup])
            comp_strings.append("+".join(sorted_tup))

        counts = Counter(comp_strings)
        matrix[fam] = {comp: counts[comp] / n_fam for comp in sorted(counts)}

    return matrix


def check_composition_collapses(
    mean_k_pure: float,
    p_k1_mixed: float,
    expert_frequencies: dict[str, float],
    mean_flops: float,
    all_expert_flops: float,
    delta_acc: float,
) -> dict[str, bool]:
    """Check pre-declared composition and compute collapse conditions (Section 23).
    
    1. all_expert_collapse: mean k on pure tasks >= 2.85.
    2. single_expert_collapse: P(k=1) on mixed tasks >= 0.95.
    3. expert_collapse: one expert present in > 90% of combinations.
    4. compute_collapse: compute >= 90% of all-expert compute without >= 3% accuracy gain.
    """
    all_exp_col = bool(mean_k_pure >= 2.85)
    single_exp_col = bool(p_k1_mixed >= 0.95)
    expert_col = any(freq > 0.90 for freq in expert_frequencies.values())
    compute_col = bool((mean_flops >= 0.90 * all_expert_flops) and (delta_acc < 0.03))

    return {
        "all_expert_collapse": all_exp_col,
        "single_expert_collapse": single_exp_col,
        "expert_collapse": expert_col,
        "compute_collapse": compute_col,
        "any_collapse": all_exp_col or single_exp_col or expert_col or compute_col,
    }


def calculate_accuracy_constrained_compute(
    pareto_points: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> dict[str, float | None]:
    """Identify the minimum average theoretical FLOPs required to achieve target accuracies."""
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

    res: dict[str, float | None] = {}
    for th in thresholds:
        qualifying = [pt["flops"] for pt in pareto_points if pt["accuracy"] >= th]
        key = f"{int(th * 100)}%"
        res[key] = min(qualifying) if qualifying else None
    return res


def calculate_counterfactual_composition_analysis(
    selected_tuples: list[tuple[str, ...]],
    expert_logits_dict: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Perform leave-one-expert-out ablation on multi-expert predictions.
    
    Determines whether each selected expert contributes non-redundantly to accuracy.
    """
    n = len(targets)
    if n == 0:
        return {}

    multi_indices = [i for i, tup in enumerate(selected_tuples) if len(tup) > 1]
    if not multi_indices:
        return {"multi_sample_count": 0, "leave_one_out_impact": {}}

    # Full multi-expert accuracy on these multi samples
    corr_full = 0
    drop_impacts: dict[str, list[float]] = {}

    for i in multi_indices:
        tup = selected_tuples[i]
        t = targets[i].item()
        sub_logits = [expert_logits_dict[e][i:i + 1] for e in tup]
        full_pred = torch.stack(sub_logits, dim=0).mean(dim=0).argmax(dim=-1).item()
        if full_pred == t:
            corr_full += 1

        # Leave-one-out for each expert in tup
        for e_drop in tup:
            remaining = [expert_logits_dict[e][i:i + 1] for e in tup if e != e_drop]
            if remaining:
                rem_pred = torch.stack(remaining, dim=0).mean(dim=0).argmax(dim=-1).item()
                acc_change = float(full_pred == t) - float(rem_pred == t)
            else:
                acc_change = 0.0
            if e_drop not in drop_impacts:
                drop_impacts[e_drop] = []
            drop_impacts[e_drop].append(acc_change)

    mean_drop_impacts = {
        e: sum(vals) / len(vals) for e, vals in drop_impacts.items()
    }

    return {
        "multi_sample_count": len(multi_indices),
        "full_multi_accuracy": corr_full / len(multi_indices),
        "mean_leave_one_out_drop_impact": mean_drop_impacts,
    }


def diagnose_phase10_failure(
    k1_acc: float,
    multi_acc: float,
    ceiling_acc: float,
    best_single_exp: str,
    multi_exp_combinations: dict[str, float],
    desired_threshold: float = 0.85,
) -> dict[str, Any]:
    """Diagnose and classify outcomes for Phase 10 multi-expert composition (Section 15).
    
    Failure Modes:
      - Success — Multi-Expert Composition: multi_acc >= desired_threshold and multi_acc > ceiling_acc + 0.05
      - Failure A — Expert Incapability: max combination accuracy < 0.60
      - Failure B — Router Selection Failure: an expert combination achieves >= desired_threshold, but router fails to select it
      - Failure C — Aggregation Failure: correct experts selected, but combining them fails to beat ceiling
      - Failure D — Composition-Order Failure: sequential order prevents useful feature computation
      - Failure E — Compute-Economic Failure: accuracy gain is minimal while compute doubles
      - Failure F — Calibration Failure: logit scale mismatch degrades combination
      - UNRESOLVED: ambiguous evidence
    """
    best_combo_acc = max(multi_exp_combinations.values()) if multi_exp_combinations else ceiling_acc

    if multi_acc >= desired_threshold and multi_acc > ceiling_acc:
        mode = "Success — Multi-Expert Composition"
        desc = f"Multi-expert policy achieves {multi_acc*100:.1f}% >= {desired_threshold*100:.1f}%, breaking ceiling ({ceiling_acc*100:.1f}%)."
    elif best_combo_acc < 0.60:
        mode = "Failure A — Expert Incapability"
        desc = f"No combination of experts achieves >60% (best combo: {best_combo_acc*100:.1f}%)."
    elif best_combo_acc >= desired_threshold and multi_acc < best_combo_acc - 0.10:
        mode = "Failure B — Router Selection Failure"
        desc = f"Optimal combination achieves {best_combo_acc*100:.1f}%, but router achieved {multi_acc*100:.1f}%."
    elif multi_acc <= ceiling_acc and best_combo_acc <= ceiling_acc + 0.03:
        mode = "Failure C — Aggregation Failure"
        desc = f"Combining multiple experts yields {multi_acc*100:.1f}%, failing to meaningfully exceed single-expert ceiling ({ceiling_acc*100:.1f}%)."
    else:
        mode = "Failure E — Compute-Economic Failure"
        desc = f"Multi-expert accuracy ({multi_acc*100:.1f}%) provides marginal gain over single expert ({k1_acc*100:.1f}%) despite multiplied compute."

    return {
        "failure_mode": mode,
        "description": desc,
        "k1_accuracy": k1_acc,
        "multi_accuracy": multi_acc,
        "ceiling_accuracy": ceiling_acc,
        "best_combination_accuracy": best_combo_acc,
    }
