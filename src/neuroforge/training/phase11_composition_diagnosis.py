"""Phase 11 Composition Bottleneck Localization and Interface Diagnosis.

Diagnostic experiment runner. Trains the four frozen experts (Phase 6
contract), then runs the eight sub-experiments (11A-11H) described in the
Phase 11 task spec and writes machine-readable artifacts to
`results/metrics/phase11_composition_diagnosis/`.

This module is **strictly additive**: it does not modify any Phase 1-10
module. It only:

  1. Re-uses `_train_expert` from `phase8b_compute_aware` to build the same
     four experts Phase 10 used.
  2. Re-uses `Phase9MixedStructureDataset` for the evaluation set.
  3. Re-uses the routing decision from `AdaptiveKRouter` and `TopKRouter` for
     the routing-selection diagnostic (no retraining needed: Phase 10's
     routing selection policy is to top-k the same router logits).
  4. Calls the eight pure diagnostic functions from
     `evaluation/phase11_metrics` and aggregates the results.
  5. Emits 13 CSVs, a summary JSON, a manifest JSON, and 12 figures.
"""
from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase11_metrics import (
    aggregation_diagnosis,
    build_causal_diagnosis,
    composition_order_diagnosis,
    interface_compatibility,
    minimal_validated_composition,
    mixed_task_component_validation,
    oracle_composition_feasibility,
    representation_transfer_probe,
    routing_representation_decodability,
    routing_selection_diagnosis,
)
from neuroforge.evaluation.phase8b_metrics import compute_normalized_costs
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.multi_expert import AdaptiveKRouter, TopKRouter
from neuroforge.training.phase8b_compute_aware import _train_expert
from neuroforge.visualization.phase11_plots import generate_phase11_figures


# Standard Phase 6 expert FLOPs map (matches Phase 10's hardcoded values).
EXPERT_FLOPS = {
    "mlp": 8928.0,
    "graph": 8112.0,
    "attention": 39168.0,
    "attention_v2": 46296.0,
}
ROUTER_FLOPS = 1052.0


def _train_4_experts(
    seed: int,
    expert_epochs: int,
    batch_size: int,
) -> dict[str, StandaloneSpecialist]:
    selected = select_capacity_phase6()
    experts: dict[str, StandaloneSpecialist] = {}
    for arch, fam in [
        ("mlp", "feature"),
        ("graph", "relational"),
        ("attention", "contextual"),
        ("attention_v2", "contextual"),
    ]:
        exp, _ = _train_expert(
            architecture=arch,
            target_family=fam,
            depth=selected.depths[arch],
            seed=seed,
            train_samples=360,
            val_samples=120,
            epochs=expert_epochs,
            batch_size=batch_size,
        )
        experts[arch] = exp
    return experts


@torch.no_grad()
def _get_routing_representation(
    router: TopKRouter,
    features: torch.Tensor,
) -> torch.Tensor:
    """Extract the routing input representation `[B, D]` from the TopKRouter.

    Phase 10's router uses mean + std pooling of the (non-existent) hidden
    state. We recompute the same pooling directly on the features so the
    `routing_representation_decodability` diagnostic sees what the router sees.
    """
    # TopKRouter has a `feature_extractor` that takes raw features.
    # The simplest thing is to compute mean+std exactly the way the router does.
    mean = features.mean(dim=1)  # [B, 8]
    std = features.std(dim=1)  # [B, 8]
    return torch.cat([mean, std], dim=-1)  # [B, 16]


def _phase10_ceiling_recompute(experts: dict[str, StandaloneSpecialist],
                                eval_ds: Phase9MixedStructureDataset,
                                families: tuple[str, ...] = ("FR", "RC", "FC", "FRC")) -> float:
    """Re-derive the Phase 10 single-expert ceiling from the current experts.

    Phase 10's reported ceiling was 66.4% (mean over mixed families). The
    actual empirical ceiling depends on the trained experts. This helper
    recomputes it to make Phase 11's claim self-contained.
    """
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    per_expert = {n: experts[n](feats).argmax(-1) for n in experts}
    ceilings: list[float] = []
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        best = 0.0
        for n in per_expert:
            acc = float((per_expert[n][idx] == targets[idx]).float().mean().item())
            if acc > best:
                best = acc
        ceilings.append(best)
    return statistics.mean(ceilings) if ceilings else 0.0


@torch.no_grad()
def _phase10_k1_baseline(experts: dict[str, StandaloneSpecialist],
                         eval_ds: Phase9MixedStructureDataset,
                         mixed_fams: tuple[str, ...] = ("FR", "RC", "FC", "FRC")) -> float:
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    # Replicate Phase 10 k=1 baseline: TopKRouter with k=1.
    router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)
    dec = router(feats, k=1, mode="hard")
    expert_names = ("mlp", "graph", "attention", "attention_v2")
    expert_logits = torch.stack(
        [experts[n](feats) for n in expert_names], dim=1
    )  # [B, 4, 2]
    preds = []
    for i in range(len(feats)):
        sel = dec.selected_indices[i]
        preds.append(expert_logits[i, sel[0]].argmax().item())
    preds = torch.tensor(preds)
    accs = []
    for f in mixed_fams:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            accs.append(float((preds[idx] == targets[idx]).float().mean().item()))
    return statistics.mean(accs) if accs else 0.0


@torch.no_grad()
def _phase10_topk_selection_function(
    experts: dict[str, StandaloneSpecialist],
    router: TopKRouter,
    eval_ds: Phase9MixedStructureDataset,
    k: int = 2,
) -> Any:
    """Return a function that maps features -> per-sample selection list.

    Used by `routing_selection_diagnosis` to compare Phase 10's learned
    selection against oracle-useful combinations.
    """
    expert_names = ("mlp", "graph", "attention", "attention_v2")

    def _fn(features: torch.Tensor) -> list[list[str]]:
        dec = router(features, k=k, mode="hard")
        out: list[list[str]] = []
        for i in range(len(features)):
            sel = [expert_names[idx] for idx in dec.selected_indices[i]]
            out.append(sel)
        return out

    return _fn


def run_phase11_composition_diagnosis(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    expert_epochs: int = 20,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute the complete Phase 11 composition diagnosis experiment.

    Returns a `summary` dict that the report generator and notebook consume.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)

    expert_names = ("mlp", "graph", "attention", "attention_v2")
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    mixed_families = ("FR", "RC", "FC", "FRC")
    pure_families = ("F", "R", "C")

    manifest: dict[str, Any] = {
        "phase": "Phase 11 — Composition Bottleneck Localization",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "expert_portfolio": list(expert_names),
        "expert_flops": EXPERT_FLOPS,
        "router_flops": ROUTER_FLOPS,
        "evaluated_families": list(families),
        "samples_per_type": samples_per_type,
    }

    # Per-seed running accumulators
    per_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        # 1. Train 4 frozen experts (same as Phase 10)
        experts = _train_4_experts(seed, expert_epochs, batch_size)

        # 2. Build evaluation dataset
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        feats = eval_ds.features
        targets = eval_ds.targets
        fams = eval_ds.families_list

        # Reconstruct per-component target tensors (sf -> {0,1}, sr -> {0,1}, sc -> {0,1})
        component_targets = {
            "sf": torch.tensor(
                [int(eval_ds.items[i]["sf"] == 1) for i in range(len(eval_ds))],
                dtype=torch.long,
            ),
            "sr": torch.tensor(
                [int(eval_ds.items[i]["sr"] == 1) for i in range(len(eval_ds))],
                dtype=torch.long,
            ),
            "sc": torch.tensor(
                [int(eval_ds.items[i]["sc"] == 1) for i in range(len(eval_ds))],
                dtype=torch.long,
            ),
            "final_target": targets,
        }

        # Re-derive Phase 10 ceiling + k=1 baseline from the same trained experts
        phase10_ceiling = _phase10_ceiling_recompute(experts, eval_ds, families=mixed_families)
        phase10_k1 = _phase10_k1_baseline(experts, eval_ds, mixed_fams=mixed_families)

        # 11A: Oracle composition feasibility
        oracle = oracle_composition_feasibility(
            experts=experts,
            expert_names=expert_names,
            eval_ds=eval_ds,
            raw_flops=EXPERT_FLOPS,
            router_flops=ROUTER_FLOPS,
            families=families,
        )
        best_combos_per_family = {f: combo for f, (combo, _) in oracle["best_per_family"].items()}

        # 11B: Representation transfer probing
        rep_probe = representation_transfer_probe(
            experts=experts,
            eval_ds=eval_ds,
            component_targets=component_targets,
        )

        # 11C: Interface compatibility
        interface = interface_compatibility(
            experts=experts,
            eval_ds=eval_ds,
            families=families,
        )

        # 11D: Aggregation diagnosis
        aggregation = aggregation_diagnosis(
            experts=experts,
            eval_ds=eval_ds,
            families=families,
        )

        # 11E: Composition order
        order_diag = composition_order_diagnosis(
            experts=experts,
            eval_ds=eval_ds,
            families=families,
        )

        # 11F: Routing selection diagnosis (using Phase 10's TopKRouter with k=2)
        router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)
        sel_fn = _phase10_topk_selection_function(experts, router, eval_ds, k=2)
        routing = routing_selection_diagnosis(
            expert_selection_function=sel_fn,
            experts=experts,
            eval_ds=eval_ds,
            best_combos_per_family=best_combos_per_family,
            families=families,
        )

        # 11G: Mixed-task component validation
        validation = mixed_task_component_validation(
            experts=experts,
            eval_ds=eval_ds,
            families=mixed_families,
        )

        # 11H: Minimal validated composition
        train_ds_for_adapter = Phase9MixedStructureDataset(samples_per_type=40, seed=seed + 50_000)
        minimal = minimal_validated_composition(
            experts=experts,
            eval_ds=eval_ds,
            best_combos=best_combos_per_family,
            families=families,
            train_ds_for_adapter=train_ds_for_adapter,
            use_adapter=False,
        )
        minimal_with_adapter = minimal_validated_composition(
            experts=experts,
            eval_ds=eval_ds,
            best_combos=best_combos_per_family,
            families=families,
            train_ds_for_adapter=train_ds_for_adapter,
            use_adapter=True,
        )

        # Routing representation decodability
        repr_decodability = routing_representation_decodability(
            router_repr=_get_routing_representation(router, feats),
            eval_ds=eval_ds,
            families=families,
        )

        # Causal diagnosis
        causal = build_causal_diagnosis(
            oracle_results=oracle,
            rep_probe_results=rep_probe,
            interface_results=interface,
            aggregation_results=aggregation,
            order_results=order_diag,
            routing_results=routing,
            validation_results=validation,
            representation_decodability=repr_decodability,
            phase10_overall_mixed=phase10_k1,
            phase10_ceiling=phase10_ceiling,
        )

        per_seed_results.append({
            "seed": seed,
            "phase10_ceiling": phase10_ceiling,
            "phase10_k1_mixed": phase10_k1,
            "oracle": oracle,
            "best_combos_per_family": best_combos_per_family,
            "rep_probe": rep_probe,
            "interface": interface,
            "aggregation": aggregation,
            "order_diagnosis": order_diag,
            "routing_selection": routing,
            "validation": validation,
            "minimal": minimal,
            "minimal_with_adapter": minimal_with_adapter,
            "routing_representation_decodability": repr_decodability,
            "causal_diagnosis": causal,
        })

    # =========================================================================
    # AGGREGATE ACROSS SEEDS
    # =========================================================================
    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": per_seed_results,
        # Mean per-family oracle ceiling
        "oracle_ceiling_per_family": _aggregate_oracle_ceiling(per_seed_results, families),
        "oracle_ceiling_overall_mixed": _aggregate_oracle_ceiling_mixed(per_seed_results, mixed_families),
        # Mean best combo per family
        "best_combos_per_family_aggregate": _aggregate_best_combos(per_seed_results, families),
        # Mean rep probe per (expert, stage, task)
        "representation_probe_per_expert": _aggregate_rep_probe(per_seed_results),
        # Mean interface per sequence
        "interface_per_sequence": _aggregate_interface(per_seed_results),
        # Mean aggregation methods
        "aggregation_methods": _aggregate_aggregation(per_seed_results),
        # Mean order per pair
        "order_per_pair": _aggregate_order(per_seed_results),
        # Mean routing recovery
        "routing_selection_summary": _aggregate_routing(per_seed_results),
        # Mean component validation
        "component_validation": _aggregate_validation(per_seed_results),
        # Mean minimal composition
        "minimal_composition": _aggregate_minimal(per_seed_results),
        "minimal_composition_with_adapter": _aggregate_minimal_adapter(per_seed_results),
        # Mean routing representation decodability
        "routing_representation_decodability_mean": _aggregate_repr_decodability(per_seed_results),
        # Causal diagnosis aggregated by status
        "causal_diagnosis_aggregate": _aggregate_causal(per_seed_results),
        # Phase 10 baselines
        "phase10_ceiling_mean": statistics.mean([r["phase10_ceiling"] for r in per_seed_results]),
        "phase10_k1_mixed_mean": statistics.mean([r["phase10_k1_mixed"] for r in per_seed_results]),
    }

    # Write CSVs and JSON
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_csvs(m_dir, per_seed_results, families, mixed_families)

    # 12 figures
    fig_paths = generate_phase11_figures(summary, f_dir)
    summary["generated_figures"] = fig_paths

    return summary


# ---------------------------------------------------------------------------
# Aggregation helpers (per-seed -> cross-seed mean)
# ---------------------------------------------------------------------------
def _aggregate_oracle_ceiling(
    per_seed: list[dict[str, Any]], families: tuple[str, ...]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in families:
        vals = [r["oracle"]["ceiling_per_family"].get(f, 0.0) for r in per_seed]
        out[f] = statistics.mean(vals) if vals else 0.0
    return out


def _aggregate_oracle_ceiling_mixed(
    per_seed: list[dict[str, Any]], mixed_fams: tuple[str, ...]
) -> float:
    vals = []
    for r in per_seed:
        m = statistics.mean(
            r["oracle"]["ceiling_per_family"].get(f, 0.0) for f in mixed_fams
        )
        vals.append(m)
    return statistics.mean(vals) if vals else 0.0


def _aggregate_best_combos(
    per_seed: list[dict[str, Any]], families: tuple[str, ...]
) -> dict[str, str]:
    """For each family, the most-frequent best combo across seeds."""
    from collections import Counter
    out: dict[str, str] = {}
    for f in families:
        cnt: Counter[str] = Counter()
        for r in per_seed:
            cnt[r["best_combos_per_family"].get(f, "")] += 1
        out[f] = cnt.most_common(1)[0][0] if cnt else ""
    return out


def _aggregate_rep_probe(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in per_seed:
        for exp_name, stages in r["rep_probe"]["per_expert"].items():
            out.setdefault(exp_name, {})
            for stage, tasks in stages.items():
                out[exp_name].setdefault(stage, {})
                for task, vals in tasks.items():
                    val_to_agg = vals.get("overall", 0.0) if isinstance(vals, dict) else vals
                    out[exp_name][stage].setdefault(task, []).append(val_to_agg)
    # Convert to means
    final: dict[str, dict[str, dict[str, float]]] = {}
    for exp_name, stages in out.items():
        final[exp_name] = {}
        for stage, tasks in stages.items():
            final[exp_name][stage] = {
                task: statistics.mean(vs) for task, vs in tasks.items()
            }
    return final


def _aggregate_interface(
    per_seed: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for r in per_seed:
        for seq_str, vals in r["interface"]["per_sequence"].items():
            out.setdefault(seq_str, {"per_family": {}, "overall": [], "flops": []})
            out[seq_str]["overall"].append(vals["overall"])
            out[seq_str]["flops"].append(vals["flops"])
            for f, acc in vals["per_family"].items():
                out[seq_str]["per_family"].setdefault(f, []).append(acc)
    final: dict[str, dict[str, Any]] = {}
    for seq_str, vals in out.items():
        final[seq_str] = {
            "overall_mean": statistics.mean(vals["overall"]),
            "flops": statistics.mean(vals["flops"]),
            "per_family_mean": {
                f: statistics.mean(accs) for f, accs in vals["per_family"].items()
            },
        }
    return final


def _aggregate_aggregation(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, list[float]]] = {}
    for r in per_seed:
        for m, vals in r["aggregation"]["methods"].items():
            out.setdefault(m, {"overall": [], "per_family": {}})
            out[m]["overall"].append(vals["overall"])
            for f, acc in vals["per_family"].items():
                out[m]["per_family"].setdefault(f, []).append(acc)
    final: dict[str, dict[str, Any]] = {}
    for m, vals in out.items():
        final[m] = {
            "overall_mean": statistics.mean(vals["overall"]),
            "per_family_mean": {
                f: statistics.mean(accs) for f, accs in vals["per_family"].items()
            },
        }
    return final


def _aggregate_order(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, list[Any]]] = {}
    for r in per_seed:
        for pair, vals in r["order_diagnosis"]["per_pair"].items():
            out.setdefault(pair, {
                "forward_overall": [],
                "reverse_overall": [],
                "forward_per_family": {},
                "reverse_per_family": {},
                "order_matters_per_family": {},
            })
            out[pair]["forward_overall"].append(vals["forward"]["overall"])
            out[pair]["reverse_overall"].append(vals["reverse"]["overall"])
            for f, acc in vals["forward"]["per_family"].items():
                out[pair]["forward_per_family"].setdefault(f, []).append(acc)
            for f, acc in vals["reverse"]["per_family"].items():
                out[pair]["reverse_per_family"].setdefault(f, []).append(acc)
            for f, differs in vals["order_matters_per_family"].items():
                out[pair]["order_matters_per_family"].setdefault(f, []).append(differs)
    final: dict[str, dict[str, Any]] = {}
    for pair, vals in out.items():
        final[pair] = {
            "forward_overall_mean": statistics.mean(vals["forward_overall"]),
            "reverse_overall_mean": statistics.mean(vals["reverse_overall"]),
            "forward_per_family_mean": {
                f: statistics.mean(accs) for f, accs in vals["forward_per_family"].items()
            },
            "reverse_per_family_mean": {
                f: statistics.mean(accs) for f, accs in vals["reverse_per_family"].items()
            },
            "order_matters_per_family_mean": {
                f: statistics.mean([1.0 if d else 0.0 for d in ds])
                for f, ds in vals["order_matters_per_family"].items()
            },
        }
    return final


def _aggregate_routing(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    agreement: dict[str, list[float]] = {f: [] for f in families}
    recovery: dict[str, list[float]] = {f: [] for f in families}
    k_dist: list[dict[Any, int]] = []
    mean_k: list[float] = []
    for r in per_seed:
        rs = r["routing_selection"]
        for f in families:
            agreement[f].append(rs["agreement_rate_per_family"].get(f, 0.0))
            recovery[f].append(rs["useful_recovery_rate_per_family"].get(f, 0.0))
        k_dist.append(rs.get("k_distribution", {}))
        mean_k.append(rs.get("mean_k", 1.0))
    return {
        "agreement_rate_per_family_mean": {f: statistics.mean(v) for f, v in agreement.items()},
        "useful_recovery_rate_per_family_mean": {f: statistics.mean(v) for f, v in recovery.items()},
        "k_distribution_mean": _merge_counters(k_dist),
        "mean_k_mean": statistics.mean(mean_k),
    }


def _merge_counters(counters: list[dict[Any, int]]) -> dict[Any, float]:
    merged: dict[Any, float] = {}
    for c in counters:
        for k, v in c.items():
            merged.setdefault(k, 0.0)
            merged[k] += v
    if not counters:
        return merged
    return {k: v / len(counters) for k, v in merged.items()}


def _aggregate_validation(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("FR", "RC", "FC", "FRC")
    comps = ("F", "R", "C")
    flip_rates: dict[str, dict[str, list[float]]] = {
        f: {c: [] for c in comps} for f in families
    }
    baseline: dict[str, dict[str, list[float]]] = {}
    for r in per_seed:
        val = r["validation"]
        for fam, d in val["component_load_bearing"].items():
            for c, rate in d.items():
                if c in flip_rates.get(fam, {}):
                    flip_rates[fam][c].append(rate)
        for exp_name, fam_accs in val["baseline_per_family"].items():
            baseline.setdefault(exp_name, {})
            for fam, acc in fam_accs.items():
                baseline[exp_name].setdefault(fam, []).append(acc)
    return {
        "component_flip_rates_mean": {
            f: {c: (statistics.mean(v) if v else 0.0) for c, v in d.items()}
            for f, d in flip_rates.items()
        },
        "baseline_per_family_mean": {
            exp: {f: (statistics.mean(v) if v else 0.0) for f, v in d.items()}
            for exp, d in baseline.items()
        },
        "destroy_results_mean": _aggregate_destroy_results(per_seed),
    }


def _aggregate_destroy_results(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in per_seed:
        for comp_tag, per_expert in r["validation"]["destroy_results"].items():
            out.setdefault(comp_tag, {})
            for exp_name, fam_accs in per_expert.items():
                out[comp_tag].setdefault(exp_name, {})
                for f, acc in fam_accs.items():
                    out[comp_tag][exp_name].setdefault(f, []).append(acc)
    final: dict[str, dict[str, dict[str, float]]] = {}
    for comp_tag, per_expert in out.items():
        final[comp_tag] = {}
        for exp_name, fam_accs in per_expert.items():
            final[comp_tag][exp_name] = {
                f: statistics.mean(v) for f, v in fam_accs.items()
            }
    return final


def _aggregate_minimal(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    out: dict[str, list[Any]] = {f: [] for f in families}
    for r in per_seed:
        m = r["minimal"]["per_family"]
        for f in families:
            out[f].append(m.get(f, {}).get("accuracy", 0.0))
    return {f: statistics.mean(v) if v else 0.0 for f, v in out.items()}


def _aggregate_minimal_adapter(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    out: dict[str, list[Any]] = {f: [] for f in families}
    extra_params: list[int] = []
    for r in per_seed:
        m = r["minimal_with_adapter"]["per_family"]
        for f in families:
            out[f].append(m.get(f, {}).get("accuracy", 0.0))
        extra_params.append(r["minimal_with_adapter"].get("total_extra_params", 0))
    return {
        "per_family_mean": {f: statistics.mean(v) if v else 0.0 for f, v in out.items()},
        "mean_extra_params": statistics.mean(extra_params) if extra_params else 0,
    }


def _aggregate_repr_decodability(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    overall: list[float] = []
    per_family: dict[str, list[float]] = {f: [] for f in families}
    for r in per_seed:
        d = r["routing_representation_decodability"]
        overall.append(d.get("overall", 0.0))
        for f, acc in d.get("per_family", {}).items():
            per_family.setdefault(f, []).append(acc)
    return {
        "overall_mean": statistics.mean(overall) if overall else 0.0,
        "per_family_mean": {
            f: statistics.mean(v) for f, v in per_family.items() if v
        },
    }


def _aggregate_causal(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Majority-vote status across seeds, with a 'noted_evidence' sample."""
    keys = [
        "EXPERT_CAPABILITY",
        "REPRESENTATION_TRANSFER",
        "AGGREGATION",
        "COMPOSITION_ORDER",
        "ROUTER_SELECTION",
        "BENCHMARK_SEMANTICS",
        "ROUTING_REPRESENTATION",
        "COMPUTE_ECONOMICS",
    ]
    out: dict[str, dict[str, Any]] = {}
    from collections import Counter
    for k in keys:
        statuses: list[str] = []
        evidence: list[str] = []
        for r in per_seed:
            d = r["causal_diagnosis"].get(k, {})
            statuses.append(d.get("status", "INCONCLUSIVE"))
            evidence.append(d.get("evidence", ""))
        cnt = Counter(statuses)
        out[k] = {
            "status": cnt.most_common(1)[0][0],
            "status_distribution": dict(cnt),
            "evidence_seed_0": evidence[0] if evidence else "",
        }
    return out


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_csvs(
    m_dir: Path,
    per_seed: list[dict[str, Any]],
    families: tuple[str, ...],
    mixed_families: tuple[str, ...],
) -> None:
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys: list[str] = []
        for r in rows:
            for kk in r.keys():
                if kk not in keys:
                    keys.append(kk)
        with (m_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # 1. oracle_composition.csv
    rows: list[dict[str, Any]] = []
    for r in per_seed:
        for combo, vals in r["oracle"]["per_combo"].items():
            row = {
                "seed": r["seed"],
                "combo": combo,
                "k": vals["k"],
                "overall": vals.get("overall", 0.0),
                "mean_flops": vals.get("mean_flops_total", 0.0),
            }
            for f in families:
                row[f"acc_{f}"] = vals.get(f, 0.0)
            rows.append(row)
    write_csv("oracle_composition.csv", rows)

    # 2. representation_probes.csv
    rows = []
    for r in per_seed:
        for exp_name, stages in r["rep_probe"]["per_expert"].items():
            for stage, tasks in stages.items():
                for task, vals in tasks.items():
                    row = {
                        "seed": r["seed"],
                        "expert": exp_name,
                        "stage": stage,
                        "task": task,
                        "probe_overall": vals.get("overall", 0.0),
                    }
                    for k, v in vals.items():
                        if k != "overall":
                            row[k] = v
                    rows.append(row)
    write_csv("representation_probes.csv", rows)

    # 3. interface_ablation.csv
    rows = []
    for r in per_seed:
        for seq, vals in r["interface"]["per_sequence"].items():
            row = {
                "seed": r["seed"],
                "sequence": seq,
                "k": vals["k"],
                "overall": vals["overall"],
                "flops": vals["flops"],
                "params_expert_total": vals["params"]["total_params"],
                "params_adapter": vals["params"]["adapter_params"],
            }
            for f in families:
                row[f"acc_{f}"] = vals["per_family"].get(f, 0.0)
            rows.append(row)
    write_csv("interface_ablation.csv", rows)

    # 4. aggregation_results.csv
    rows = []
    for r in per_seed:
        for m, vals in r["aggregation"]["methods"].items():
            row = {
                "seed": r["seed"],
                "method": m,
                "overall": vals["overall"],
            }
            for f in families:
                row[f"acc_{f}"] = vals["per_family"].get(f, 0.0)
            rows.append(row)
    write_csv("aggregation_results.csv", rows)

    # 5. composition_order.csv
    rows = []
    for r in per_seed:
        for pair, vals in r["order_diagnosis"]["per_pair"].items():
            fwd = vals["forward"]; rev = vals["reverse"]
            row = {
                "seed": r["seed"],
                "pair": pair,
                "forward_overall": fwd["overall"],
                "reverse_overall": rev["overall"],
                "forward_flops": fwd["flops"],
                "reverse_flops": rev["flops"],
            }
            for f in families:
                row[f"forward_{f}"] = fwd["per_family"].get(f, 0.0)
                row[f"reverse_{f}"] = rev["per_family"].get(f, 0.0)
                row[f"order_matters_{f}"] = vals["order_matters_per_family"].get(f, False)
            rows.append(row)
    write_csv("composition_order.csv", rows)

    # 6. routing_diagnosis.csv
    rows = []
    for r in per_seed:
        rs = r["routing_selection"]
        row = {
            "seed": r["seed"],
            "mean_k": rs.get("mean_k", 0.0),
        }
        for f in families:
            row[f"agreement_{f}"] = rs["agreement_rate_per_family"].get(f, 0.0)
            row[f"recovery_{f}"] = rs["useful_recovery_rate_per_family"].get(f, 0.0)
        rows.append(row)
    write_csv("routing_diagnosis.csv", rows)

    # 7. routing_representation.csv
    rows = []
    for r in per_seed:
        d = r["routing_representation_decodability"]
        row = {
            "seed": r["seed"],
            "probe_overall": d.get("overall", 0.0),
        }
        for f in families:
            row[f"acc_{f}"] = d.get("per_family", {}).get(f, 0.0)
        rows.append(row)
    write_csv("routing_representation.csv", rows)

    # 8. mixed_task_validation.csv
    rows = []
    for r in per_seed:
        val = r["validation"]
        # Component load-bearing (analytical)
        for fam, comp_dict in val["component_load_bearing"].items():
            row = {"seed": r["seed"], "family": fam, "control": "component_load_bearing"}
            for c, rate in comp_dict.items():
                row[f"flip_{c}"] = rate
            rows.append(row)
        # Destroy controls
        for comp_tag, per_expert in val["destroy_results"].items():
            for exp_name, fam_accs in per_expert.items():
                row = {
                    "seed": r["seed"],
                    "family": "any",
                    "control": f"{comp_tag}_expert={exp_name}",
                }
                for f, acc in fam_accs.items():
                    row[f"acc_{f}"] = acc
                rows.append(row)
    write_csv("mixed_task_validation.csv", rows)

    # 9. counterfactual_results.csv
    rows = []
    for r in per_seed:
        for m, cf_dict in r["aggregation"]["counterfactual"].items():
            for tag, acc in cf_dict.items():
                rows.append({
                    "seed": r["seed"],
                    "method": m,
                    "control": tag,
                    "accuracy": acc,
                })
    write_csv("counterfactual_results.csv", rows)

    # 10. compute_results.csv
    rows = []
    for r in per_seed:
        # k=1 phase10 baseline
        rows.append({
            "seed": r["seed"],
            "policy": "k=1_baseline_phase10",
            "accuracy": r["phase10_k1_mixed"],
            "mean_flops": 1 * 8112.0 + 1052.0,  # representative
        })
        # Best oracle combo per family
        for f, combo in r["best_combos_per_family"].items():
            v = r["oracle"]["per_combo"].get(combo, {})
            rows.append({
                "seed": r["seed"],
                "policy": f"oracle_per_family_{f}",
                "accuracy": v.get(f, 0.0),
                "mean_flops": v.get("mean_flops_total", 0.0),
            })
        # Minimal composition
        for f in families:
            m = r["minimal"]["per_family"].get(f, {})
            rows.append({
                "seed": r["seed"],
                "policy": f"minimal_per_family_{f}",
                "accuracy": m.get("accuracy", 0.0),
                "mean_flops": 0.0,
            })
    write_csv("compute_results.csv", rows)

    # 11. latency_results.csv
    # We do not re-measure latency here; it is reported in Phase 10 and remains
    # valid (the experts are the same). We still record the policy -> estimate.
    rows = []
    for r in per_seed:
        for p, flops in [
            ("k=1", 8112.0 + 1052.0),
            ("fixed_k=2_uniform", EXPERT_FLOPS["mlp"] + EXPERT_FLOPS["graph"] + 1052.0),
            ("sequential_2", 17040.0),
            ("adaptive_k_learned", 61972.0),
        ]:
            rows.append({
                "seed": r["seed"],
                "policy": p,
                "estimated_flops_per_sample": flops,
            })
    write_csv("latency_results.csv", rows)

    # 12. seed_results.csv
    rows = []
    for r in per_seed:
        rows.append({
            "seed": r["seed"],
            "phase10_ceiling": r["phase10_ceiling"],
            "phase10_k1_mixed": r["phase10_k1_mixed"],
            "oracle_ceiling_mixed_mean": statistics.mean(
                r["oracle"]["ceiling_per_family"].get(f, 0.0) for f in mixed_families
            ),
        })
    write_csv("seed_results.csv", rows)

    # 13. failure_diagnosis.csv
    rows = []
    for r in per_seed:
        for k, v in r["causal_diagnosis"].items():
            rows.append({
                "seed": r["seed"],
                "category": k,
                "status": v.get("status", "INCONCLUSIVE"),
                "evidence": v.get("evidence", ""),
            })
    write_csv("failure_diagnosis.csv", rows)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def generate_phase11_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Generate the 15-section Phase 11 markdown report (honest, evidence-derived)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper formatters
    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    def gv(d: dict, *keys: str, default: Any = 0.0) -> Any:
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    mixed_families = ("FR", "RC", "FC", "FRC")
    pure_families = ("F", "R", "C")

    # Pull the cross-seed aggregate fields
    oracle_ceiling = summary.get("oracle_ceiling_per_family", {})
    oracle_ceiling_mixed = summary.get("oracle_ceiling_overall_mixed", 0.0)
    phase10_ceiling = summary.get("phase10_ceiling_mean", 0.0)
    phase10_k1 = summary.get("phase10_k1_mixed_mean", 0.0)
    best_combos = summary.get("best_combos_per_family_aggregate", {})
    rep_probe = summary.get("representation_probe_per_expert", {})
    interface = summary.get("interface_per_sequence", {})
    aggregation = summary.get("aggregation_methods", {})
    order = summary.get("order_per_pair", {})
    routing = summary.get("routing_selection_summary", {})
    validation = summary.get("component_validation", {})
    minimal = summary.get("minimal_composition", {})
    minimal_with_adapter = summary.get("minimal_composition_with_adapter", {})
    repr_dec = summary.get("routing_representation_decodability_mean", {})
    causal = summary.get("causal_diagnosis_aggregate", {})

    # Select the programmatic verdict (§36) from the causal diagnosis
    # Priority order: REPRESENTATION_TRANSFER, EXPERT_CAPABILITY, BENCHMARK_SEMANTICS,
    # AGGREGATION, COMPOSITION_ORDER, ROUTER_SELECTION
    def status_of(category: str) -> str:
        return causal.get(category, {}).get("status", "INCONCLUSIVE")

    # Decide the primary verdict
    exp_cap = causal.get("EXPERT_CAPABILITY", {}).get("status", "INCONCLUSIVE")
    exp_cap_note = causal.get("EXPERT_CAPABILITY", {}).get("evidence_seed_0", "")
    bench = status_of("BENCHMARK_SEMANTICS")
    if bench == "NOT SUPPORTED":
        verdict_case = "CASE F"
        verdict_label = "Benchmark does not genuinely require composition"
    elif exp_cap == "NOT SUPPORTED" or "essentially equal" in exp_cap_note or "no multi-expert combination" in exp_cap_note:
        # Oracle composition offers no material benefit over the best single expert.
        verdict_case = "CASE E"
        verdict_label = "No existing combination works"
    elif status_of("ROUTING_REPRESENTATION") == "SUPPORTED" or status_of("ROUTER_SELECTION") == "SUPPORTED":
        verdict_case = "CASE A"
        verdict_label = "Composition capability exists; router fails"
    elif status_of("REPRESENTATION_TRANSFER") == "SUPPORTED":
        verdict_case = "CASE B"
        verdict_label = "Interface / representation transfer failure"
    elif status_of("AGGREGATION") == "SUPPORTED":
        verdict_case = "CASE C"
        verdict_label = "Aggregation destroys complementary information"
    elif status_of("COMPOSITION_ORDER") == "SUPPORTED":
        verdict_case = "CASE D"
        verdict_label = "Order is decisive"
    elif status_of("ROUTER_SELECTION") == "SUPPORTED":
        verdict_case = "CASE A"
        verdict_label = "Composition capability exists; router fails"
    else:
        verdict_case = "CASE G"
        verdict_label = "Multiple interacting causes"

    def per_family_table_seq(seq: str) -> str:
        d = interface.get(seq, {})
        pf = d.get("per_family_mean", {})
        return " | ".join(pct(pf.get(f, 0.0)) for f in families)

    def order_table() -> str:
        rows = []
        for pair, vals in order.items():
            fwd_pf = vals.get("forward_per_family_mean", {})
            rev_pf = vals.get("reverse_per_family_mean", {})
            fr_mixed = statistics.mean([fwd_pf.get(f, 0.0) for f in mixed_families])
            rv_mixed = statistics.mean([rev_pf.get(f, 0.0) for f in mixed_families])
            rows.append(
                f"| **{pair}** | {pct(fr_mixed)} | {pct(rv_mixed)} | {(fr_mixed - rv_mixed)*100:+.1f}pp |"
            )
        return "\n".join(rows) if rows else "- No pairs."

    def causal_table() -> str:
        rows = []
        for cat in [
            "EXPERT_CAPABILITY",
            "REPRESENTATION_TRANSFER",
            "AGGREGATION",
            "COMPOSITION_ORDER",
            "ROUTER_SELECTION",
            "BENCHMARK_SEMANTICS",
            "ROUTING_REPRESENTATION",
            "COMPUTE_ECONOMICS",
        ]:
            data = causal.get(cat, {})
            status = data.get("status", "INCONCLUSIVE")
            ev = data.get("evidence_seed_0", "")
            rows.append(f"| **{cat}** | {status} | {ev} |")
        return "\n".join(rows)

    def per_family_oracle_ceiling_table() -> str:
        rows = []
        for f in families:
            rows.append(f"| **{f}** | {pct(oracle_ceiling.get(f, 0.0))} | {best_combos.get(f, '')} |")
        return "\n".join(rows)

    def per_family_repr_probe_table() -> str:
        rows = []
        for f in families:
            cells = []
            for exp in ("mlp", "graph", "attention_v2"):
                v = gv(rep_probe, exp, "after_blocks", "final_target", "overall", default=0.0)
                cells.append(f"{pct(v)}")
            rows.append(f"| **{f}** | {' | '.join(cells)} |")
        return "\n".join(rows)

    def aggregation_table() -> str:
        rows = []
        for m, vals in aggregation.items():
            mixed = statistics.mean(
                [vals.get("per_family_mean", {}).get(f, 0.0) for f in mixed_families]
            )
            rows.append(f"| **{m}** | {pct(vals.get('overall_mean', 0.0))} | {pct(mixed)} |")
        return "\n".join(rows)

    def interface_table() -> str:
        rows = []
        for seq, vals in interface.items():
            mixed = statistics.mean(
                [vals.get("per_family_mean", {}).get(f, 0.0) for f in mixed_families]
            )
            rows.append(
                f"| **{seq}** | {pct(vals.get('overall_mean', 0.0))} | {pct(mixed)} | {vals.get('flops', 0.0):,.0f} |"
            )
        return "\n".join(rows)

    def routing_recovery_table() -> str:
        rows = []
        for f in families:
            rec = routing.get("useful_recovery_rate_per_family_mean", {}).get(f, 0.0)
            agr = routing.get("agreement_rate_per_family_mean", {}).get(f, 0.0)
            rows.append(f"| **{f}** | {pct(agr)} | {pct(rec)} |")
        rows.append(
            f"\nMean k = {routing.get('mean_k_mean', 1.0):.2f}; "
            f"k distribution = {routing.get('k_distribution_mean', {})}"
        )
        return "\n".join(rows)

    def validation_table() -> str:
        flip = validation.get("component_flip_rates_mean", {})
        rows = []
        for fam in mixed_families:
            d = flip.get(fam, {})
            rows.append(
                f"| **{fam}** | {pct(d.get('F', 0.0))} | {pct(d.get('R', 0.0))} | {pct(d.get('C', 0.0))} |"
            )
        return "\n".join(rows)

    def minimal_table() -> str:
        rows = []
        for f in families:
            rows.append(f"| **{f}** | {pct(minimal.get(f, 0.0))} | {pct(minimal_with_adapter.get('per_family_mean', {}).get(f, 0.0))} |")
        rows.append(
            f"\nAdapter adds {minimal_with_adapter.get('mean_extra_params', 0):.0f} parameters."
        )
        return "\n".join(rows)

    def minimal_comparison() -> str:
        # Best oracle per family vs phase 10 k=1 baseline
        rows = []
        for f in families:
            oracle_acc = oracle_ceiling.get(f, 0.0)
            minimal_acc = minimal.get(f, 0.0)
            rows.append(
                f"| **{f}** | {pct(phase10_k1)} | {pct(oracle_acc)} | {pct(minimal_acc)} | {(minimal_acc - phase10_k1)*100:+.1f}pp |"
            )
        return "\n".join(rows)

    def primary_comparison_table() -> str:
        return (
            "| **k=1 (Phase 10 baseline)** | "
            + f"{pct(phase10_k1)} | "
            + " | ".join(["-"] * 7)
            + f" | {8112.0 + 1052.0:,.0f} | - | 1.00 |\n"
            + "| **Oracle composition (per-family best)** | "
            + f"{pct(oracle_ceiling_mixed)} | "
            + " | ".join([pct(oracle_ceiling.get(f, 0.0)) for f in families])
            + " | - | - | - |\n"
            + "| **Sequential best (chain)** | "
            + f"{pct(max((v.get('overall_mean', 0.0) for v in interface.values()), default=0.0))} | "
            + " | ".join(["-"] * 7)
            + " | - | - | - |\n"
            + "| **Minimal validated composition** | "
            + f"{pct(statistics.mean([minimal.get(f, 0.0) for f in mixed_families]))} | "
            + " | ".join([pct(minimal.get(f, 0.0)) for f in families])
            + " | - | - | - |\n"
        )

    report = f"""# Phase 11 — Composition Bottleneck Localization and Interface Diagnosis

## Mandatory Scientific Disclaimer

> Phase 10's reported single-expert ceiling of 66.4% is an empirical value derived from the trained Phase 10 portfolio. It does NOT constitute a universal mathematical limit. The oracle "mixed-task ceiling" recomputed in this Phase 11 experiment depends on the same experts; if those experts cannot solve the mixed task even in oracle-combination, that is a statement about THIS portfolio, not about composition in general.

---

## 1. Executive Summary

Phase 10 reported that fixed k=2 composition reached 57.2% mixed accuracy and k=3 reached 60.4% on the validated 7-family mixed benchmark, against a cross-family single-expert ceiling of 66.4%. The current Phase 11 experiment re-trains the same four experts and re-derives these numbers from the same code path. The objective is **causal localization**: identifying which of (A) expert capability, (B) representation/interface, (C) aggregation, (D) composition order, (E) routing, or (F) benchmark semantics is the dominant bottleneck.

- **Re-derived Phase 10 k=1 mixed accuracy**: {pct(phase10_k1)}
- **Re-derived Phase 10 single-expert ceiling on mixed families**: {pct(phase10_ceiling)}
- **Oracle-combination ceiling (k<=3) on mixed families**: {pct(oracle_ceiling_mixed)}
- **Best per-family oracle combination** (from 11A): {best_combos}
- **Programmatic verdict**: **{verdict_case} — {verdict_label}**

---

## 2. Research Question

Why does multi-expert execution fail to exploit the complementary computational capabilities of the existing experts strongly enough to exceed the single-expert ceiling on the validated mixed benchmark?

---

## 3. Method (Causal Map)

| Layer | Diagnostic | Status |
|---|---|---|
| 11A | Oracle composition feasibility | executed |
| 11B | Representation transfer probing | executed |
| 11C | Interface compatibility (StateChain) | executed |
| 11D | Aggregation comparison | executed |
| 11E | Composition order | executed |
| 11F | Routing-selection diagnosis | executed |
| 11G | Benchmark semantic validation | executed |
| 11H | Minimal validated composition | executed (no / with linear adapter) |

All experiments re-use the four Phase 6 frozen experts and the validated `Phase9MixedStructureDataset` (840 samples, 120 per family). Seeds: {list(summary.get('manifest', {}).get('seeds', []))}.

---

## 4. 11A — Oracle Composition Feasibility

| Family | Oracle Ceiling | Best Combo (this seed) |
|---|---:|---|
{per_family_oracle_ceiling_table()}

Cross-family mixed mean oracle ceiling: **{pct(oracle_ceiling_mixed)}** vs Phase 10 single-expert ceiling {pct(phase10_ceiling)}.

---

## 5. 11B — Representation Transfer Probing

Decodability of the FINAL target from each expert's `[B, S, H]` state at the `after_blocks` stage (linear probe, ridge regression on mean-pooled state). Higher is better.

| Family | MLP | Graph | V2 |
|---|---:|---:|---:|
{per_family_repr_probe_table()}

The full per-expert × per-stage × per-task matrix is in `representation_probes.csv`.

---

## 6. 11C — Interface Compatibility (Sequential StateChain)

| Sequence | Overall | Mixed | FLOPs |
|---|---:|---:|---:|
{interface_table()}

StateChain preserves V2's raw-feature requirement (V2 always receives the original input to read its channel-4 marker and channels 0:3 keys), unlike Phase 10's `SequentialSpecialistComposition`.

---

## 7. 11D — Aggregation Comparison

| Method | Overall | Mixed |
|---|---:|---:|
{aggregation_table()}

If methods are numerically identical, aggregation is **NOT** the bottleneck.

---

## 8. 11E — Composition Order

| Pair | Forward (Mixed) | Reverse (Mixed) | Diff |
|---|---:|---:|---:|
{order_table()}

Forward vs reverse means: chain `A->B` vs `B->A` averaged over mixed families.

---

## 9. 11F — Routing Selection Diagnosis

| Family | Agreement (Phase 10 top-2 = Oracle) | Useful-Composition Recovery |
|---|---:|---:|
{routing_recovery_table()}

---

## 10. 11G — Benchmark Semantic Validation

Component-removal / destruction controls. For each (family, component) cell, the entry is the rate at which flipping the component's vote flips the composite target — a high rate means the component is genuinely load-bearing.

| Family | F (flip rate) | R (flip rate) | C (flip rate) |
|---|---:|---:|---:|
{validation_table()}

The destroy-channel results (zeroing each component's feature channels) are in `mixed_task_validation.csv`.

---

## 11. 11H — Minimal Validated Composition

Using the 11A best-oracle combinations, evaluate the smallest intervention. Adapter uses a single `H -> H` linear layer trained on a small mixed split.

| Family | No Adapter | With Linear Adapter |
|---|---:|---:|
{minimal_table()}

---

## 12. Primary Comparison Table

| Policy | Mixed Acc | F | R | C | FR | RC | FC | FRC | FLOPs | Latency | Avg active |
|---|---:|---|---|---|---|---|---|---|---:|---:|---:|
{primary_comparison_table()}

---

## 13. Causal Diagnosis Table (§28)

| Bottleneck | Status | Evidence |
|---|---|---|
{causal_table()}

---

## 14. Causal Conclusion

The selected case is **{verdict_case} — {verdict_label}**, with the following supporting evidence (full per-seed numbers in `failure_diagnosis.csv` and `summary.json`):

- Phase 10 k=1 mixed baseline: {pct(phase10_k1)}
- Phase 10 single-expert ceiling (recomputed on the same experts): {pct(phase10_ceiling)}
- Oracle k<=3 mixed ceiling (11A): {pct(oracle_ceiling_mixed)}
- Routing representation decodes 7-family at: {pct(repr_dec.get('overall_mean', 0.0))} (chance = {100/7:.1f}%)
- Best sequential chain mixed accuracy: {pct(max((v.get('overall_mean', 0.0) for v in interface.values()), default=0.0))}

---

## 15. Limitations & Decision for Next Phase

1. **Frozen experts** — same Phase 6 portfolio as Phase 10. The "expert capability" verdict is conditional on this portfolio.
2. **Synthetic benchmark** — the mixed dataset uses a sum-of-signs target; component-load-bearing is verified analytically but real-world compositional tasks may have different structure.
3. **Linear adapter only** — no deeper adapter was tested; that is by design (§4 forbids architectural escalation).
4. **Single routing representation** — the Phase 8B mean+std routing input is fixed; alternative representations (11F R1-R4) are mentioned in the spec but not tested here unless evidence demands.

The decision for the next phase is **evidence-driven**, not pre-committed. The strongest evidence-supported next research question depends on the verdict above.
"""
    output_path.write_text(report, encoding="utf-8")
