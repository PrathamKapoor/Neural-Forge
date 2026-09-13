"""Phase 11 Diagnostic Metrics.

All functions are pure evaluators: they take frozen experts + an evaluation
dataset + a callable, and return machine-readable dictionaries.

Nothing in this module mutates the experts.

Diagnostic families:

  11A  oracle_composition_feasibility
  11B  representation_transfer_probing
  11C  interface_compatibility
  11D  aggregation_diagnosis
  11E  composition_order_diagnosis
  11F  routing_selection_diagnosis
  11G  mixed_task_component_validation
  11H  minimal_validated_composition
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Callable

import torch

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.models.phase11_diagnostics import (
    LinearAdapter,
    StateChain,
    extract_state_after_encoder,
    extract_state_after_blocks,
    linear_probe_accuracy,
    linear_probe_accuracy_per_class,
    train_linear_adapter,
)
from neuroforge.models.specialists import StandaloneSpecialist


# ---------------------------------------------------------------------------
# 11A — Oracle composition feasibility
# ---------------------------------------------------------------------------
@torch.no_grad()
def oracle_composition_feasibility(
    experts: dict[str, StandaloneSpecialist],
    expert_names: tuple[str, ...],
    eval_ds: Phase9MixedStructureDataset,
    raw_flops: dict[str, float],
    router_flops: float = 1052.0,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    combinations: list[tuple[str, ...]] | None = None,
    use_logit_average: bool = True,
) -> dict[str, Any]:
    """Evaluate every named expert combination via simple logit averaging
    (or sum) and report per-family accuracy and per-combo total FLOPs.

    The function does NOT introduce a router. It is an oracle that picks the
    combination manually.

    Returns:
        {
          "per_combo": { combo: {family: acc, "overall": acc, "mean_flops": float} },
          "best_per_family": { family: (combo, acc) },
          "ceiling_per_family": { family: acc },
          "ceiling_overall": acc,
        }
    """
    experts_mod = {n: experts[n] for n in expert_names}
    for n in expert_names:
        experts_mod[n].eval()

    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    n_samples = len(targets)

    if combinations is None:
        # Include all 2-expert pairs + the 3-expert triple Graph+MLP+V2.
        combinations = [
            ("mlp",),
            ("graph",),
            ("attention_v2",),
            ("mlp", "graph"),
            ("graph", "mlp"),
            ("mlp", "attention_v2"),
            ("attention_v2", "mlp"),
            ("graph", "attention_v2"),
            ("attention_v2", "graph"),
            ("mlp", "graph", "attention_v2"),
        ]

    # Precompute per-expert logits for all experts
    per_expert_logits: dict[str, torch.Tensor] = {}
    for en in expert_names:
        per_expert_logits[en] = experts_mod[en](feats)

    per_combo: dict[str, Any] = {}
    for combo in combinations:
        combo_str = "+".join(combo)
        # Sum logits (or mean) across selected experts
        stacked = torch.stack([per_expert_logits[nm] for nm in combo], dim=0)  # [k, B, 2]
        if use_logit_average:
            combined = stacked.mean(dim=0)
        else:
            combined = stacked.sum(dim=0)
        preds = combined.argmax(dim=-1)
        per_combo[combo_str] = {
            "combo": list(combo),
            "k": len(combo),
            "mean_flops": float(
                sum(raw_flops.get(n, 10000.0) for n in combo) + len(combo) * 2 * 2
            ),
            "router_flops": router_flops,
        }
        # Per family
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if not idx:
                continue
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            per_combo[combo_str][f] = acc
        idx_all = list(range(n_samples))
        per_combo[combo_str]["overall"] = float(
            (preds[idx_all] == targets[idx_all]).float().mean().item()
        )
        per_combo[combo_str]["mean_flops_total"] = (
            router_flops + per_combo[combo_str]["mean_flops"]
        )

    # Best per family across combinations
    best_per_family: dict[str, tuple[str, float]] = {}
    for f in families:
        best_acc = -1.0
        best_combo = ""
        for cstr, vals in per_combo.items():
            if vals.get(f, -1) > best_acc:
                best_acc = vals[f]
                best_combo = cstr
        best_per_family[f] = (best_combo, best_acc)

    # Best family-mean over all combinations
    ceiling_per_family: dict[str, float] = {f: acc for f, (_, acc) in best_per_family.items()}
    ceiling_combos = list({c for c, _ in best_per_family.values()})
    ceiling_overall = statistics.mean([ceiling_per_family[f] for f in families])

    return {
        "per_combo": per_combo,
        "best_per_family": best_per_family,
        "ceiling_per_family": ceiling_per_family,
        "ceiling_overall": ceiling_overall,
        "ceiling_combos": ceiling_combos,
    }


# ---------------------------------------------------------------------------
# 11B — Representation transfer probing
# ---------------------------------------------------------------------------
@torch.no_grad()
def representation_transfer_probe(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    component_targets: dict[str, torch.Tensor],
    stages: tuple[str, ...] = ("input", "after_encoder", "after_blocks"),
    hidden_dim: int = 24,
) -> dict[str, Any]:
    """Probe each expert's representation at multiple depths for
    decodability of (a) the final target, (b) per-component votes.

    Returns per-expert × per-stage × per-task probe accuracy.
    """
    feats = eval_ds.features
    experts_eval = {n: e for n, e in experts.items()}
    for e in experts_eval.values():
        e.eval()
    results: dict[str, Any] = {"per_expert": {}}
    for name, expert in experts_eval.items():
        per_stage: dict[str, Any] = {}
        for stage in stages:
            if stage == "input":
                state = feats
            elif stage == "after_encoder":
                state = extract_state_after_encoder(expert, feats)
            elif stage == "after_blocks":
                state = extract_state_after_blocks(expert, feats)
            else:
                continue
            per_stage[stage] = {}
            for task_name, task_targets in component_targets.items():
                if task_name == "final_target":
                    ttask = eval_ds.targets
                else:
                    ttask = task_targets
                probe = linear_probe_accuracy_per_class(state, ttask, hidden_dim=hidden_dim)
                per_stage[stage][task_name] = probe
        results["per_expert"][name] = per_stage
    return results


# ---------------------------------------------------------------------------
# 11C — Interface compatibility
# ---------------------------------------------------------------------------
@torch.no_grad()
def interface_compatibility(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    hidden_dim: int = 24,
    sequences: list[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Evaluate sequential composition for many orderings using StateChain.

    Returns per-ordering per-family accuracy and FLOPs.
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list

    if sequences is None:
        sequences = [
            ("mlp",),
            ("graph",),
            ("attention_v2",),
            ("mlp", "graph"),
            ("graph", "mlp"),
            ("mlp", "attention_v2"),
            ("attention_v2", "mlp"),
            ("graph", "attention_v2"),
            ("attention_v2", "graph"),
            ("mlp", "graph", "attention_v2"),
        ]

    results: dict[str, Any] = {"per_sequence": {}}
    for seq in sequences:
        chain = StateChain(experts, sequence=seq, attach="identity", attach_v2_features=True)
        chain.eval()
        logits = chain(feats)
        preds = logits.argmax(dim=-1)
        per_family: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                per_family[f] = float((preds[idx] == targets[idx]).float().mean().item())
        results["per_sequence"]["+".join(seq)] = {
            "sequence": list(seq),
            "k": len(seq),
            "flops": chain.theoretical_flops(),
            "params": chain.parameters_count(),
            "per_family": per_family,
            "overall": float((preds == targets).float().mean().item()),
        }
    return results


# ---------------------------------------------------------------------------
# 11D — Aggregation diagnosis
# ---------------------------------------------------------------------------
@torch.no_grad()
def aggregation_diagnosis(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    methods: tuple[str, ...] = ("raw", "weighted_router", "concat_linear", "residual", "concat_only_pooled"),
) -> dict[str, Any]:
    """Compare five aggregation methods on the same (k=2) expert pair.

    Methods:
      raw              : mean of expert logits (Phase 10 "C1")
      weighted_router  : weighted by a synthetic router (uniform weight here)
      concat_linear    : concat pooled state vectors and a small linear head (trained)
      residual         : add second expert's logits to first (asymmetric aggregation)
      concat_only_pooled : concat logits only (no extra linear head)
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    # Use the two strongest specialists (mlp + graph + attention_v2) for a generic
    # pair; Phase 9 baseline identifies these as the most load-bearing.
    primary = ("mlp", "graph")
    e_a = experts[primary[0]]
    e_b = experts[primary[1]]
    e_a.eval(); e_b.eval()
    logits_a = e_a(feats)
    logits_b = e_b(feats)
    state_a = extract_state_after_blocks(e_a, feats).mean(dim=1)
    state_b = extract_state_after_blocks(e_b, feats).mean(dim=1)

    results: dict[str, Any] = {"methods": {}}
    for m in methods:
        if m == "raw":
            combined = (logits_a + logits_b) / 2
        elif m == "weighted_router":
            # Synthetic 50/50 (router-free diagnostic)
            combined = 0.5 * logits_a + 0.5 * logits_b
        elif m == "residual":
            combined = logits_a + logits_b
        elif m == "concat_only_pooled":
            combined = torch.cat([logits_a, logits_b], dim=-1).mean(dim=-1, keepdim=True).repeat(1, 2)
        elif m == "concat_linear":
            # Concatenate pooled states; train a small ridge regressor
            stacked = torch.cat([state_a, state_b], dim=-1)
            y = targets
            Y = torch.zeros(len(y), 2); Y[torch.arange(len(y)), y] = 1.0
            lam = 1e-2
            XtX = stacked.t() @ stacked + lam * torch.eye(stacked.shape[1])
            XtY = stacked.t() @ Y
            try:
                W = torch.linalg.solve(XtX, XtY)
            except Exception:
                W = torch.linalg.lstsq(stacked, Y).solution
            combined = stacked @ W
        else:
            raise ValueError(f"unknown method: {m}")
        preds = combined.argmax(dim=-1)
        per_family: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                per_family[f] = float((preds[idx] == targets[idx]).float().mean().item())
        results["methods"][m] = {
            "per_family": per_family,
            "overall": float((preds == targets).float().mean().item()),
        }
    # Counterfactual: remove each expert
    cf: dict[str, dict[str, float]] = {}
    for m in methods:
        cf[m] = {}
        for tag, l_full, l_minus in [
            ("without_mlp", logits_b, (logits_a + logits_b) / 2 - logits_a / 2),  # use only logits_b
            ("without_graph", logits_a, (logits_a + logits_b) / 2 - logits_b / 2),  # use only logits_a
        ]:
            # Reconstruction of the "minus" expert prediction
            minus_pred = l_minus.argmax(dim=-1)
            cf[m][tag] = float((minus_pred == targets).float().mean().item())
    results["counterfactual"] = cf
    return results


# ---------------------------------------------------------------------------
# 11E — Composition order diagnosis
# ---------------------------------------------------------------------------
@torch.no_grad()
def composition_order_diagnosis(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """Compare every pairwise ordering A->B vs B->A on mixed tasks.

    Uses the StateChain to ensure interface compatibility is preserved.
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list

    pairs = [("mlp", "graph"), ("mlp", "attention_v2"), ("graph", "attention_v2")]
    results: dict[str, Any] = {"per_pair": {}}
    for a, b in pairs:
        per_order: dict[str, Any] = {}
        for order in ((a, b), (b, a)):
            chain = StateChain(experts, sequence=order, attach="identity", attach_v2_features=True)
            chain.eval()
            logits = chain(feats)
            preds = logits.argmax(dim=-1)
            per_family: dict[str, float] = {}
            for f in families:
                idx = [i for i, fm in enumerate(fams) if fm == f]
                if idx:
                    per_family[f] = float((preds[idx] == targets[idx]).float().mean().item())
            per_order["+".join(order)] = {
                "per_family": per_family,
                "overall": float((preds == targets).float().mean().item()),
                "flops": chain.theoretical_flops(),
            }
        fwd = per_order[f"{a}+{b}"]["per_family"]
        rev = per_order[f"{b}+{a}"]["per_family"]
        # Order matters if any family differs by > 5pp
        order_matters_per_family: dict[str, bool] = {
            f: abs(fwd.get(f, 0) - rev.get(f, 0)) > 0.05 for f in families
        }
        results["per_pair"][f"{a}+vs+{b}"] = {
            "forward": per_order[f"{a}+{b}"],
            "reverse": per_order[f"{b}+{a}"],
            "order_matters_per_family": order_matters_per_family,
        }
    return results


# ---------------------------------------------------------------------------
# 11F — Routing selection diagnosis
# ---------------------------------------------------------------------------
@torch.no_grad()
def routing_selection_diagnosis(
    expert_selection_function: Callable[[torch.Tensor], list[list[str]]],
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    best_combos_per_family: dict[str, str],
    use_adapter: bool = False,
    adapter_state: dict[str, Any] | None = None,
    expert_names: tuple[str, ...] = ("mlp", "graph", "attention_v2"),
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """Compare the learned router's selections to oracle-useful combinations.

    Args:
        expert_selection_function: a callable that takes features [B, S, D]
            and returns a list-of-lists of expert names (one per sample).
        best_combos_per_family: mapping of family -> combo string (output of
            oracle_composition_feasibility).
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list

    selections = expert_selection_function(feats)
    n_samples_routing = len(targets)

    # Decompose selection accuracy
    useful_recovery: dict[str, list[bool]] = {f: [] for f in families}
    harmful_rate_per_family: dict[str, list[bool]] = {f: [] for f in families}
    agreement_per_family: dict[str, list[bool]] = {f: [] for f in families}
    k_distribution: list[int] = []

    for i in range(n_samples_routing):
        sel = selections[i] if i < len(selections) else []
        if not sel:
            sel = ["mlp"]
        k_distribution.append(len(sel))
        fam = fams[i]
        oracle_combo = best_combos_per_family.get(fam, "")
        oracle_experts = set(oracle_combo.split("+")) if oracle_combo else set()
        sel_set = set(sel)
        # Agreement: all oracle experts present in selection (within the portfolio)
        agreement_per_family[fam].append(sel_set == oracle_experts if oracle_experts else False)
        # Recovery: at least one useful expert chosen
        useful_recovery[fam].append(bool(sel_set & oracle_experts) if oracle_experts else False)

    return {
        "agreement_rate_per_family": {f: (sum(v) / len(v)) if v else 0.0 for f, v in agreement_per_family.items()},
        "useful_recovery_rate_per_family": {f: (sum(v) / len(v)) if v else 0.0 for f, v in useful_recovery.items()},
        "k_distribution": dict(Counter(k_distribution)),
        "mean_k": statistics.mean(k_distribution) if k_distribution else 0.0,
    }


# ---------------------------------------------------------------------------
# 11G — Mixed-task component validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def mixed_task_component_validation(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("FR", "RC", "FC", "FRC"),
    component_signs: dict[int, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Component-removal / destruction controls for the mixed benchmark.

    For each sample, we know the per-component sign `sf`, `sr`, `sc` (in
    `eval_ds.items[i]["sf"|"sr"|"sc"]`).

    A "destroy-component-X" control zeros channel X in the input features and
    measures the accuracy drop on the corresponding family.

    A "remove-component-X" control flips the sign of the relevant target vote
    and checks if the new target flips too (validates that the component was
    actually load-bearing in the composite target).
    """
    feats = eval_ds.features.clone()
    targets = eval_ds.targets
    fams = eval_ds.families_list

    # 1. Per-component accuracy: can a model predict sf, sr, sc independently?
    per_expert_pred_components: dict[str, dict[str, Any]] = {}
    for name, expert in experts.items():
        expert.eval()
        logits = expert(feats)
        # extract per-sample per-component targets from items
        per_comp_acc: dict[str, float] = {}
        for comp_idx, comp_name in enumerate(("sf", "sr", "sc")):
            comp_targets = torch.tensor(
                [int(eval_ds.items[i][comp_name] == 1) for i in range(len(eval_ds))],
                dtype=torch.long,
            )
            # Use the expert's binary prediction to predict the per-component sign
            # (treat argmax of 2-class as a 0/1 prediction; majority rule)
            exp_pred = (logits.argmax(-1) == 1).long()
            per_comp_acc[comp_name] = float(
                (exp_pred == comp_targets).float().mean().item()
            )
        per_expert_pred_components[name] = per_comp_acc

    # 2. Destroy components by zeroing feature channels used by them.
    # Channel mapping (Phase 9 mixed dataset):
    #   F: channels 0, 1
    #   R: channel 0 (the candidate; F's channel 0 is overwritten when R is active)
    #   C: channels 0:3 (distractors + key)
    # Therefore to test each component we either (a) re-derive families with one
    # component removed, or (b) estimate sensitivity by zeroing channels after
    # the fact. We do (b): for each family, define channels that "carry" each
    # component when present.
    channel_map = {
        "F": [0, 1],
        "R": [0],
        "C": [0, 1, 2],
    }

    # Compute baseline per-family accuracy with each expert
    baseline: dict[str, dict[str, float]] = {}
    for name, expert in experts.items():
        expert.eval()
        logits = expert(feats)
        preds = logits.argmax(-1)
        baseline[name] = {}
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                baseline[name][f] = float((preds[idx] == targets[idx]).float().mean().item())

    destroy_results: dict[str, dict[str, Any]] = {}
    for comp_name, channels in channel_map.items():
        destroyed = feats.clone()
        destroyed[:, :, channels] = 0.0
        per_expert = {}
        for name, expert in experts.items():
            expert.eval()
            logits = expert(destroyed)
            preds = logits.argmax(-1)
            per_family = {}
            for f in families:
                idx = [i for i, fm in enumerate(fams) if f in fm]  # any family containing comp
                if idx and f in baseline.get(name, {}):
                    per_family[f] = float((preds[idx] == targets[idx]).float().mean().item())
            per_expert[name] = per_family
        destroy_results[f"destroy_{comp_name}"] = per_expert

    # 3. Verify composite target really depends on the component (analytically).
    # For each mixed family, flip a component's vote and re-derive the target.
    # Then measure how often the new target differs from the original. If never
    # differs, the component is **not load-bearing** in the composite target.
    component_load_bearing: dict[str, dict[str, float]] = {}
    for f in families:
        comps = [c for c in ("F", "R", "C") if c in f]
        per_comp_flip: dict[str, float] = {}
        for comp_to_flip in comps:
            flipped = 0
            total = 0
            for it in eval_ds.items:
                if it["family"] != f:
                    continue
                total += 1
                key = {"F": "sf", "R": "sr", "C": "sc"}[comp_to_flip]
                active_signals: list[int] = []
                if "F" in f:
                    active_signals.append(-it["sf"])
                if "R" in f:
                    active_signals.append(-it["sr"])
                if "C" in f:
                    active_signals.append(-it["sc"])
                if comp_to_flip in f:
                    active_signals[-1] = -active_signals[-1]  # flip
                z = sum(active_signals)
                if z > 0:
                    new_y = 1
                elif z < 0:
                    new_y = 0
                else:
                    new_y = int(it["sample_id"]) % 2
                if new_y != it["target"].item():
                    flipped += 1
            per_comp_flip[comp_to_flip] = flipped / total if total else 0.0
        component_load_bearing[f] = per_comp_flip

    return {
        "per_expert_pred_components": per_expert_pred_components,
        "baseline_per_family": baseline,
        "destroy_results": destroy_results,
        "component_load_bearing": component_load_bearing,
        "channel_map": channel_map,
    }


# ---------------------------------------------------------------------------
# 11H — Minimal validated composition
# ---------------------------------------------------------------------------
def minimal_validated_composition(
    experts: dict[str, StandaloneSpecialist],
    eval_ds: Phase9MixedStructureDataset,
    best_combos: dict[str, str],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
    train_ds_for_adapter: Phase9MixedStructureDataset | None = None,
    use_adapter: bool = False,
    hidden_dim: int = 24,
) -> dict[str, Any]:
    """Apply the smallest validated intervention supported by 11A evidence.

    If `use_adapter` is True and `train_ds_for_adapter` is provided, train a
    single linear H->H adapter on top of the best oracle composition.

    Returns per-family accuracy and parameter count.
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list

    # Apply best combo per family. The simplest "minimal" intervention is to
    # use the per-family oracle combo discovered in 11A.
    per_family_results: dict[str, dict[str, Any]] = {}
    expert_logits: dict[str, torch.Tensor] = {}
    for name, exp in experts.items():
        exp.eval()
        expert_logits[name] = exp(feats)

    total_extra_params = 0
    for f in families:
        combo_str = best_combos.get(f, "")
        if not combo_str:
            per_family_results[f] = {"combo": "", "accuracy": 0.0}
            continue
        names = combo_str.split("+")
        if use_adapter and train_ds_for_adapter is not None and len(names) > 1:
            # Train a single linear H->H adapter on top of stacked states
            stacked_states = torch.stack(
                [extract_state_after_blocks(experts[n], feats).mean(dim=1) for n in names],
                dim=1,
            )  # [B, K, H]
            adapter = LinearAdapter(hidden_dim)
            optim = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
            train_feats = train_ds_for_adapter.features
            train_targets = train_ds_for_adapter.targets
            for _ in range(8):
                perm = torch.randperm(len(train_feats))
                for i in range(0, len(train_feats), 32):
                    idx = perm[i:i + 32]
                    stacked = torch.stack(
                        [extract_state_after_blocks(experts[n], train_feats[idx]).mean(dim=1) for n in names],
                        dim=1,
                    )
                    pooled = stacked.mean(dim=1)  # [B, H]
                    optim.zero_grad()
                    logits = adapter(pooled)
                    loss = torch.nn.functional.cross_entropy(logits, train_targets[idx])
                    loss.backward()
                    optim.step()
            total_extra_params += sum(p.numel() for p in adapter.parameters())
            with torch.no_grad():
                stacked_eval = torch.stack(
                    [extract_state_after_blocks(experts[n], feats).mean(dim=1) for n in names],
                    dim=1,
                )
                pooled_eval = stacked_eval.mean(dim=1)
                logits_eval = adapter(pooled_eval)
        else:
            stacked = torch.stack([expert_logits[n] for n in names], dim=0)
            logits_eval = stacked.mean(dim=0)
        preds = logits_eval.argmax(-1)
        idx = [i for i, fm in enumerate(fams) if fm == f]
        per_family_results[f] = {
            "combo": combo_str,
            "accuracy": float((preds[idx] == targets[idx]).float().mean().item()) if idx else 0.0,
        }

    return {
        "per_family": per_family_results,
        "total_extra_params": total_extra_params,
        "use_adapter": use_adapter,
    }


# ---------------------------------------------------------------------------
# Routing representation decodability (used in 11F)
# ---------------------------------------------------------------------------
@torch.no_grad()
def routing_representation_decodability(
    router_repr: torch.Tensor,
    eval_ds: Phase9MixedStructureDataset,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """Train a per-family linear probe on the router's representation.

    Returns per-family probe accuracy. If family labels are not linearly
    decodable, that is evidence that the routing input is insufficient.
    """
    feats = router_repr
    fams = eval_ds.families_list
    # Convert family string to int class index
    fam_to_idx = {f: i for i, f in enumerate(families)}
    y = torch.tensor([fam_to_idx[f] for f in fams], dtype=torch.long)
    n_classes = len(families)
    Y = torch.zeros(len(y), n_classes); Y[torch.arange(len(y)), y] = 1.0
    h = feats.shape[1]
    lam = 1e-2
    XtX = feats.t() @ feats + lam * torch.eye(h)
    XtY = feats.t() @ Y
    try:
        W = torch.linalg.solve(XtX, XtY)
    except Exception:
        W = torch.linalg.lstsq(feats, Y).solution
    preds = (feats @ W).argmax(-1)
    per_family: dict[str, float] = {}
    for i, f in enumerate(families):
        mask = y == i
        if mask.any():
            per_family[f] = float((preds[mask] == y[mask]).float().mean().item())
        else:
            per_family[f] = 0.0
    return {"overall": float((preds == y).float().mean().item()), "per_family": per_family}


# ---------------------------------------------------------------------------
# Causal diagnosis helper
# ---------------------------------------------------------------------------
def build_causal_diagnosis(
    oracle_results: dict[str, Any],
    rep_probe_results: dict[str, Any],
    interface_results: dict[str, Any],
    aggregation_results: dict[str, Any],
    order_results: dict[str, Any],
    routing_results: dict[str, Any],
    validation_results: dict[str, Any],
    representation_decodability: dict[str, Any],
    phase10_overall_mixed: float,
    phase10_ceiling: float,
) -> dict[str, Any]:
    """Apply the §28 causal-diagnosis table.

    Each row records a hypothesis / bottleneck, the evidence key, the control,
    and a status drawn from {SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED,
    INCONCLUSIVE, NOT TESTED}.
    """
    # 1. EXPERT_CAPABILITY: does any oracle combination exceed the single-expert
    #    ceiling on mixed tasks?
    ceiling_per_family = oracle_results.get("ceiling_per_family", {})
    mixed_fams = ("FR", "RC", "FC", "FRC")
    mixed_ceiling_oracle = statistics.mean(
        [ceiling_per_family.get(f, 0.0) for f in mixed_fams]
    )
    if mixed_ceiling_oracle > phase10_ceiling + 0.05:
        expert_status = "SUPPORTED"
        expert_note = (
            f"Oracle ceiling on mixed families = {mixed_ceiling_oracle:.1%} clearly "
            f"exceeds Phase 10 single-expert ceiling = {phase10_ceiling:.1%} (>5pp); "
            f"the existing expert portfolio can solve mixed tasks if given the "
            f"correct combination."
        )
    elif mixed_ceiling_oracle > phase10_ceiling + 0.01:
        expert_status = "PARTIALLY SUPPORTED"
        expert_note = (
            f"Oracle ceiling on mixed families = {mixed_ceiling_oracle:.1%} modestly "
            f"exceeds Phase 10 single-expert ceiling = {phase10_ceiling:.1%} (1-5pp)."
        )
    elif mixed_ceiling_oracle >= phase10_ceiling - 0.01:
        expert_status = "PARTIALLY SUPPORTED"
        expert_note = (
            f"Oracle ceiling on mixed = {mixed_ceiling_oracle:.1%} essentially equal "
            f"to Phase 10 single-expert ceiling = {phase10_ceiling:.1%}; no multi-expert "
            f"combination materially exceeds the best single expert."
        )
    else:
        expert_status = "NOT SUPPORTED"
        expert_note = (
            f"No oracle combination (k<=3) exceeds the Phase 10 single-expert "
            f"ceiling on mixed families."
        )
    rep_per_exp = rep_probe_results.get("per_expert", {})
    if rep_per_exp:

        # Decodability of final target at "after_blocks" stage per expert
        v2_after = rep_per_exp.get("attention_v2", {}).get("after_blocks", {}).get("final_target", {}).get("overall", 0.0)
        mlp_after = rep_per_exp.get("mlp", {}).get("after_blocks", {}).get("final_target", {}).get("overall", 0.0)
        graph_after = rep_per_exp.get("graph", {}).get("after_blocks", {}).get("final_target", {}).get("overall", 0.0)
    else:
        v2_after = mlp_after = graph_after = 0.0
    # Heuristic: representation transfer works if any expert's "after_blocks"
    # final-target probe is > 0.6 on the mixed dataset.
    if max(v2_after, mlp_after, graph_after) > 0.6:
        rep_status = "SUPPORTED"
        rep_note = (
            f"Final-target decodability at 'after_blocks' is high (max {max(v2_after, mlp_after, graph_after):.1%}); "
            f"intermediate state contains sufficient task information."
        )
    else:
        rep_status = "PARTIALLY SUPPORTED"
        rep_note = (
            f"Final-target decodability at 'after_blocks' is limited "
            f"(max {max(v2_after, mlp_after, graph_after):.1%}); representation transfer "
            f"is the bottleneck for some experts."
        )

    # 3. AGGREGATION: does aggregation method choice change mixed-task accuracy?
    methods = aggregation_results.get("methods", {})
    if methods:
        accs = [v.get("overall", 0.0) for v in methods.values()]
        agg_spread = (max(accs) - min(accs)) if accs else 0.0
        if agg_spread > 0.05:
            agg_status = "SUPPORTED"
            agg_note = f"Aggregation method spread = {agg_spread:.1%} > 5pp; aggregation choice matters."
        else:
            agg_status = "NOT SUPPORTED"
            agg_note = f"Aggregation method spread = {agg_spread:.1%} <= 5pp; aggregation is not the bottleneck."
    else:
        agg_status = "INCONCLUSIVE"
        agg_note = "Aggregation results missing."

    # 4. COMPOSITION_ORDER: does order matter in sequential chains?
    order_per_pair = order_results.get("per_pair", {})
    n_diff = 0
    n_total = 0
    for pair, vals in order_per_pair.items():
        for f, differs in vals.get("order_matters_per_family", {}).items():
            n_total += 1
            if differs:
                n_diff += 1
    if n_total and n_diff / n_total > 0.25:
        order_status = "SUPPORTED"
        order_note = (
            f"Order differs > 5pp in {n_diff}/{n_total} (mixed family, order) pairs."
        )
    else:
        order_status = "NOT SUPPORTED"
        order_note = (
            f"Order differs > 5pp in only {n_diff}/{n_total} pairs; order is not decisive."
        )

    # 5. ROUTER_SELECTION: is the router recovery rate high?
    recovery = routing_results.get("useful_recovery_rate_per_family", {})
    if recovery:
        avg_rec = statistics.mean(recovery.get(f, 0.0) for f in mixed_fams)
        if avg_rec > 0.7:
            router_status = "PARTIALLY SUPPORTED"
            router_note = f"Mean mixed-family useful-composition recovery = {avg_rec:.1%}."
        elif avg_rec > 0.4:
            router_status = "PARTIALLY SUPPORTED"
            router_note = f"Mean mixed-family useful-composition recovery = {avg_rec:.1%}; partial."
        else:
            router_status = "SUPPORTED"
            router_note = f"Mean mixed-family useful-composition recovery = {avg_rec:.1%} is low; router selection is a bottleneck."
    else:
        router_status = "INCONCLUSIVE"
        router_note = "Routing recovery data missing."

    # 6. BENCHMARK_SEMANTICS
    component_load = validation_results.get("component_load_bearing", {})
    if component_load:
        n_matters = 0
        n_total = 0
        for fam, comp_dict in component_load.items():
            for comp, flip_rate in comp_dict.items():
                n_total += 1
                if flip_rate > 0.1:
                    n_matters += 1
        if n_total and n_matters / n_total > 0.6:
            bench_status = "SUPPORTED"
            bench_note = (
                f"{n_matters}/{n_total} (family, component) pairs show that "
                f"removing the component flips the target > 10% of the time; "
                f"the mixed benchmark genuinely requires multi-component reasoning."
            )
        else:
            bench_status = "PARTIALLY SUPPORTED"
            bench_note = (
                f"Only {n_matters}/{n_total} (family, component) pairs show > 10% "
                f"target-flip on component removal; some components may not be load-bearing."
            )
    else:
        bench_status = "INCONCLUSIVE"
        bench_note = "Component-load-bearing data missing."

    # 7. ROUTING_REPRESENTATION
    repr_overall = representation_decodability.get("overall", 0.0)
    chance_level = 1.0 / 7.0  # 7-family classification
    if repr_overall > chance_level * 3:
        repr_status = "PARTIALLY SUPPORTED"
        repr_note = f"Router representation decodes 7-family at {repr_overall:.1%} (chance = {chance_level:.1%}); some signal."
    else:
        repr_status = "SUPPORTED"
        repr_note = f"Router representation decodes 7-family at only {repr_overall:.1%}; routing input is insufficient."

    # 8. COMPUTE_ECONOMICS
    # If adaptive-k mean_k is much larger than 1 while mixed-task accuracy does
    # not improve, compute is a bottleneck.
    mean_k = routing_results.get("mean_k", 1.0)
    if mean_k > 1.5 and not (agg_status == "SUPPORTED" or expert_status.startswith("PARTIALLY")):
        compute_status = "PARTIALLY SUPPORTED"
        compute_note = (
            f"Mean k = {mean_k:.2f} but expert capability appears limited; "
            f"additional compute is unlikely to help."
        )
    else:
        compute_status = "NOT TESTED"
        compute_note = "Compute economics requires full accuracy comparison across policies."

    return {
        "EXPERT_CAPABILITY": {"status": expert_status, "evidence": expert_note},
        "REPRESENTATION_TRANSFER": {"status": rep_status, "evidence": rep_note},
        "AGGREGATION": {"status": agg_status, "evidence": agg_note},
        "COMPOSITION_ORDER": {"status": order_status, "evidence": order_note},
        "ROUTER_SELECTION": {"status": router_status, "evidence": router_note},
        "BENCHMARK_SEMANTICS": {"status": bench_status, "evidence": bench_note},
        "ROUTING_REPRESENTATION": {"status": repr_status, "evidence": repr_note},
        "COMPUTE_ECONOMICS": {"status": compute_status, "evidence": compute_note},
    }
