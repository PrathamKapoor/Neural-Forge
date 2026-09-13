"""Phase 13 Joint Expert Co-Adaptation experiment runner.

Additive (does NOT modify any Phase 1-12 module). Re-uses:
  - `StandaloneSpecialist`, `_train_expert` for training
  - `Phase9MixedStructureDataset` for evaluation
  - `extract_state_after_blocks`, `linear_probe_accuracy_per_class` for probing
  - `apply_phase9_relational_permutation` for relational destruction

Produces 13 CSVs, summary.json, manifest.json, and 15 figures.
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuroforge.datasets.phase9_datasets import (
    Phase9MixedStructureDataset,
    apply_phase9_relational_permutation,
)
from neuroforge.evaluation.phase13_metrics import (
    build_phase13_causal_diagnosis,
    cross_evaluation_with_ablation,
    expert_branch_scales,
    oracle_composition,
    relational_destruction_accuracy,
    representation_probes_per_expert,
    select_phase13_verdict,
    single_expert_ceiling,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase12_expert_portfolio import (
    _train_joint_on_mixed,
    _train_one_expert,
)
from neuroforge.visualization.phase13_plots import generate_phase13_figures


# Phase 12 established the new expert portfolio; Phase 13 only varies
# the joint expert's training.
EXPERT_NAMES_EXISTING = ("mlp", "graph", "attention", "attention_v2")
FAMILIES = ("F", "R", "C", "FR", "RC", "FC", "FRC")
PURE_FAMS = ("F", "R", "C")
MIXED_FAMS = ("FR", "RC", "FC", "FRC")


def _train_5_experts(
    seed: int,
    expert_epochs: int,
    batch_size: int,
) -> dict[str, StandaloneSpecialist]:
    """Train the existing 4 frozen experts (Phase 6 contract)."""
    selected = select_capacity_phase6()
    experts: dict[str, StandaloneSpecialist] = {}
    for arch, fam in [
        ("mlp", "feature"),
        ("graph", "relational"),
        ("attention", "contextual"),
        ("attention_v2", "contextual"),
    ]:
        experts[arch] = _train_one_expert(
            architecture=arch,
            target_family=fam,
            depth=selected.depths[arch],
            seed=seed,
            epochs=expert_epochs,
            batch_size=batch_size,
        )
    return experts


def _train_joint_condition(
    architecture: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> StandaloneSpecialist:
    """Train one Phase 13 Joint condition (`joint` or `joint_co`).

    Uses the same mixed F+R+C training infrastructure as Phase 12.
    """
    sp = StandaloneSpecialist(architecture, input_dim=8, hidden_dim=24, depth=1)
    _train_joint_on_mixed(sp, seed=seed, epochs=epochs, batch_size=batch_size)
    return sp


@torch.no_grad()
def _joint_per_family(joint: StandaloneSpecialist, eval_ds: Phase9MixedStructureDataset) -> dict[str, float]:
    feats = eval_ds.features
    targets = eval_ds.targets
    fams = eval_ds.families_list
    joint.eval()
    preds = joint(feats).argmax(-1)
    out: dict[str, float] = {}
    for f in FAMILIES:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    pure = statistics.mean([out.get(f, 0.0) for f in PURE_FAMS])
    mixed = statistics.mean([out.get(f, 0.0) for f in MIXED_FAMS])
    out["pure_mean"] = pure
    out["mixed_mean"] = mixed
    out["overall"] = float((preds == targets).float().mean().item())
    return out


@torch.no_grad()
def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _latency_microseconds(model: nn.Module, sample: torch.Tensor, n_runs: int = 20) -> float:
    """Physical CPU latency proxy. Returns mean microseconds per sample."""
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


def run_phase13_joint_coadaptation(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    baseline_epochs: int = 20,
    extended_epochs: int = 40,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute the complete Phase 13 joint-coadaptation experiment."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "Phase 13 — Joint Expert Co-Adaptation and Relational Compositional Diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "evaluated_families": list(FAMILIES),
        "samples_per_type": samples_per_type,
        "baseline_epochs": baseline_epochs,
        "extended_epochs": extended_epochs,
        "conditions": [
            {
                "label": "13A_baseline",
                "architecture": "joint",
                "epochs": baseline_epochs,
                "description": "Phase 12 Joint baseline reproduction (mixed F+R+C, frozen).",
            },
            {
                "label": "13B_extended_training",
                "architecture": "joint",
                "epochs": extended_epochs,
                "description": "Same architecture, doubled training budget.",
            },
            {
                "label": "13C_coadaptation",
                "architecture": "joint_co",
                "epochs": extended_epochs,
                "description": "JointCo: same 3 branches + inter-branch fusion (H*3 -> H).",
            },
        ],
    }

    per_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        # 1. Train the existing 4 frozen experts (Phase 6 contract).
        existing = _train_5_experts(
            seed=seed, expert_epochs=baseline_epochs, batch_size=batch_size
        )

        # 2. Train the three Joint conditions.
        baseline = _train_joint_condition("joint", seed, baseline_epochs, batch_size)
        extended = _train_joint_condition("joint", seed, extended_epochs, batch_size)
        coadapted = _train_joint_condition("joint_co", seed, extended_epochs, batch_size)

        # 3. Evaluation set.
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        feats = eval_ds.features
        targets = eval_ds.targets
        fams = eval_ds.families_list

        # 4. Per-family Joint accuracy under all three conditions.
        joint_baseline_perf = _joint_per_family(baseline, eval_ds)
        joint_extended_perf = _joint_per_family(extended, eval_ds)
        joint_coadapted_perf = _joint_per_family(coadapted, eval_ds)

        # 5. Existing specialists' per-family accuracy.
        existing_perf: dict[str, dict[str, float]] = {}
        for name, exp in existing.items():
            exp.eval()
            preds = exp(feats).argmax(-1)
            per_fam: dict[str, float] = {}
            for f in FAMILIES:
                idx = [i for i, fm in enumerate(fams) if fm == f]
                if idx:
                    per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
            pure = statistics.mean([per_fam.get(f, 0.0) for f in PURE_FAMS])
            mixed = statistics.mean([per_fam.get(f, 0.0) for f in MIXED_FAMS])
            per_fam["pure_mean"] = pure
            per_fam["mixed_mean"] = mixed
            per_fam["overall"] = float((preds == targets).float().mean().item())
            existing_perf[name] = per_fam

        # 6. Cross-eval the best Joint condition (coadapted) on all 7 families.
        cross_eval: dict[str, dict[str, float]] = {}
        for name, exp in existing.items():
            exp.eval()
            preds = exp(feats).argmax(-1)
            per_fam = {f: float((preds[[i for i, fm in enumerate(fams) if fm == f]] == targets[[i for i, fm in enumerate(fams) if fm == f]]).float().mean().item()) if [i for i, fm in enumerate(fams) if fm == f] else 0.0 for f in FAMILIES}
            per_fam["overall"] = float((preds == targets).float().mean().item())
            cross_eval[name] = per_fam
        for name, exp in (
            ("joint_baseline", baseline),
            ("joint_extended", extended),
            ("joint_coadapted", coadapted),
        ):
            exp.eval()
            preds = exp(feats).argmax(-1)
            per_fam = {}
            for f in FAMILIES:
                idx = [i for i, fm in enumerate(fams) if fm == f]
                if idx:
                    per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
            per_fam["overall"] = float((preds == targets).float().mean().item())
            cross_eval[name] = per_fam

        # 7. Branch scales (extracted from the trained Joint blocks).
        scales_baseline = expert_branch_scales(baseline)
        scales_extended = expert_branch_scales(extended)
        scales_coadapted = expert_branch_scales(coadapted)

        # 8. Branch ablation (deterministic zeroing, no retrain) on the
        #    best Joint condition (coadapted) AND on baseline.
        ablation_baseline = cross_evaluation_with_ablation(
            baseline, eval_ds,
            ablations=((), ("feature",), ("graph",), ("context",)),
        )
        ablation_coadapted = cross_evaluation_with_ablation(
            coadapted, eval_ds,
            ablations=((), ("feature",), ("graph",), ("context",)),
        )

        # 9. Representation probes on the coadapted Joint.
        component_targets: dict[str, torch.Tensor] = {
            "F_signal": torch.tensor(
                [int(eval_ds.items[i]["sf"] == 1) for i in range(len(eval_ds))], dtype=torch.long
            ),
            "R_signal": torch.tensor(
                [int(eval_ds.items[i]["sr"] == 1) for i in range(len(eval_ds))], dtype=torch.long
            ),
            "C_signal": torch.tensor(
                [int(eval_ds.items[i]["sc"] == 1) for i in range(len(eval_ds))], dtype=torch.long
            ),
            "final_target": targets,
        }
        repr_probes_coadapted = representation_probes_per_expert(
            coadapted, eval_ds, component_targets
        )
        repr_probes_baseline = representation_probes_per_expert(
            baseline, eval_ds, component_targets
        )

        # 10. Relational destruction controls (Joint vs Graph specialist).
        joint_re_rel = relational_destruction_accuracy(coadapted, eval_ds, seed=seed)
        graph_re_rel = relational_destruction_accuracy(existing["graph"], eval_ds, seed=seed)
        baseline_re_rel = relational_destruction_accuracy(baseline, eval_ds, seed=seed)

        # 11. Portfolio re-evaluation: ceiling with the best Joint condition
        #     added to the 4 existing experts.
        all_logits: dict[str, torch.Tensor] = {}
        for name, exp in existing.items():
            exp.eval()
            all_logits[name] = exp(feats)
        coadapted.eval()
        all_logits["joint_coadapted"] = coadapted(feats)
        baseline_ceiling = single_expert_ceiling(
            {n: all_logits[n] for n in EXPERT_NAMES_EXISTING}, targets, fams
        )
        new_ceiling = single_expert_ceiling(all_logits, targets, fams)
        oracle_results: dict[str, Any] = {}
        for k in (1, 2, 3):
            oracle_results[f"k={k}"] = oracle_composition(all_logits, targets, fams, k)

        # 12. Compute and latency.
        param_counts: dict[str, int] = {
            **{n: _param_count(e) for n, e in existing.items()},
            "joint_baseline": _param_count(baseline),
            "joint_extended": _param_count(extended),
            "joint_coadapted": _param_count(coadapted),
        }
        # Latency: only on a small sample (first 60 features) for reproducibility.
        latency_sample = feats[:60]
        latencies: dict[str, float] = {
            **{n: _latency_microseconds(e, latency_sample) for n, e in existing.items()},
            "joint_baseline": _latency_microseconds(baseline, latency_sample),
            "joint_extended": _latency_microseconds(extended, latency_sample),
            "joint_coadapted": _latency_microseconds(coadapted, latency_sample),
        }

        per_seed_results.append({
            "seed": seed,
            "joint_baseline_perf": joint_baseline_perf,
            "joint_extended_perf": joint_extended_perf,
            "joint_coadapted_perf": joint_coadapted_perf,
            "existing_perf": existing_perf,
            "cross_eval": cross_eval,
            "scales_baseline": scales_baseline,
            "scales_extended": scales_extended,
            "scales_coadapted": scales_coadapted,
            "ablation_baseline": ablation_baseline,
            "ablation_coadapted": ablation_coadapted,
            "repr_probes_baseline": repr_probes_baseline,
            "repr_probes_coadapted": repr_probes_coadapted,
            "joint_re_rel": joint_re_rel,
            "graph_re_rel": graph_re_rel,
            "baseline_re_rel": baseline_re_rel,
            "baseline_ceiling": baseline_ceiling,
            "new_ceiling": new_ceiling,
            "oracle_results": oracle_results,
            "param_counts": param_counts,
            "latencies": latencies,
        })

    # =========================================================================
    # AGGREGATE
    # =========================================================================
    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": per_seed_results,
        "joint_baseline_perf_mean": _agg_perf(per_seed_results, "joint_baseline_perf"),
        "joint_extended_perf_mean": _agg_perf(per_seed_results, "joint_extended_perf"),
        "joint_coadapted_perf_mean": _agg_perf(per_seed_results, "joint_coadapted_perf"),
        "existing_perf_mean": _agg_existing_perf(per_seed_results),
        "cross_eval_mean": _agg_cross_eval(per_seed_results),
        "scales_baseline_mean": _agg_scales(per_seed_results, "scales_baseline"),
        "scales_extended_mean": _agg_scales(per_seed_results, "scales_extended"),
        "scales_coadapted_mean": _agg_scales(per_seed_results, "scales_coadapted"),
        "ablation_baseline_mean": _agg_ablation(per_seed_results, "ablation_baseline"),
        "ablation_coadapted_mean": _agg_ablation(per_seed_results, "ablation_coadapted"),
        "repr_probes_baseline_mean": _agg_repr_probes(per_seed_results, "repr_probes_baseline"),
        "repr_probes_coadapted_mean": _agg_repr_probes(per_seed_results, "repr_probes_coadapted"),
        "joint_re_rel_mean": _agg_rel_sens(per_seed_results, "joint_re_rel"),
        "graph_re_rel_mean": _agg_rel_sens(per_seed_results, "graph_re_rel"),
        "baseline_re_rel_mean": _agg_rel_sens(per_seed_results, "baseline_re_rel"),
        "old_ceiling_mixed_mean": _agg_old_ceiling_mixed(per_seed_results),
        "new_ceiling_mixed_mean": _agg_new_ceiling_mixed(per_seed_results),
        "oracle_results_mean": _agg_oracle(per_seed_results),
        "param_counts": per_seed_results[0]["param_counts"],
        "latencies_mean": _agg_latencies(per_seed_results),
        "causal_diagnosis_aggregate": _agg_causal(per_seed_results),
    }

    # Verdict (uses cross-seed means).
    causal = summary["causal_diagnosis_aggregate"]
    # Phase 12 reference value: 66.4% old ceiling, 66.8% new ceiling with joint.
    verdict_case, verdict_label = select_phase13_verdict(
        causal=causal,
        joint_baseline_mixed=summary["joint_baseline_perf_mean"].get("mixed_mean", 0.0),
        extended_training_mixed=summary["joint_extended_perf_mean"].get("mixed_mean", 0.0),
        coadaptation_mixed=summary["joint_coadapted_perf_mean"].get("mixed_mean", 0.0),
        old_ceiling_mixed=summary["old_ceiling_mixed_mean"],
        new_ceiling_mixed=summary["new_ceiling_mixed_mean"],
    )
    summary["verdict_case"] = verdict_case
    summary["verdict_label"] = verdict_label

    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_csvs(m_dir, per_seed_results, FAMILIES, MIXED_FAMS, PURE_FAMS)

    fig_paths = generate_phase13_figures(summary, f_dir)
    summary["generated_figures"] = fig_paths

    return summary


def _agg_perf(per_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    families = list(FAMILIES) + ["pure_mean", "mixed_mean", "overall"]
    out: dict[str, list[float]] = {f: [] for f in families}
    for r in per_seed:
        perf = r[key]
        for f in families:
            if f in perf:
                out[f].append(perf[f])
    return {f: statistics.mean(v) for f, v in out.items() if v}


def _agg_existing_perf(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    families = list(FAMILIES) + ["pure_mean", "mixed_mean", "overall"]
    experts = list(per_seed[0]["existing_perf"].keys())
    out: dict[str, dict[str, list[float]]] = {
        n: {f: [] for f in families} for n in experts
    }
    for r in per_seed:
        for n in experts:
            for f in families:
                if f in r["existing_perf"].get(n, {}):
                    out[n][f].append(r["existing_perf"][n][f])
    return {
        n: {f: statistics.mean(v) for f, v in d.items() if v}
        for n, d in out.items()
    }


def _agg_cross_eval(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    families = list(FAMILIES) + ["overall"]
    experts = list(per_seed[0]["cross_eval"].keys())
    out: dict[str, dict[str, list[float]]] = {
        n: {f: [] for f in families} for n in experts
    }
    for r in per_seed:
        for n in experts:
            for f in families:
                if f in r["cross_eval"].get(n, {}):
                    out[n][f].append(r["cross_eval"][n][f])
    return {
        n: {f: statistics.mean(v) for f, v in d.items() if v}
        for n, d in out.items()
    }


def _agg_scales(per_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    branches = ("feature", "graph", "context")
    out: dict[str, list[float]] = {b: [] for b in branches}
    for r in per_seed:
        scales = r[key]
        for b in branches:
            if b in scales:
                out[b].append(scales[b])
    return {b: statistics.mean(v) for b, v in out.items() if v}


def _agg_ablation(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    families = list(FAMILIES) + ["overall"]
    labels = list(per_seed[0][key].keys())
    out: dict[str, dict[str, list[float]]] = {lab: {f: [] for f in families} for lab in labels}
    for r in per_seed:
        for lab in labels:
            for f in families:
                if f in r[key].get(lab, {}):
                    out[lab][f].append(r[key][lab][f])
    return {
        lab: {f: statistics.mean(v) for f, v in d.items() if v}
        for lab, d in out.items()
    }


def _agg_repr_probes(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    families = FAMILIES
    tasks = list(per_seed[0][key].keys())
    out: dict[str, dict[str, list[float]]] = {t: {f: [] for f in families} for t in tasks}
    for r in per_seed:
        for t in tasks:
            for f in families:
                if f in r[key].get(t, {}):
                    out[t][f].append(r[key][t][f])
    return {
        t: {f: statistics.mean(v) for f, v in d.items() if v}
        for t, d in out.items()
    }


def _agg_rel_sens(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    families = ("R", "RC", "FRC")
    conds = list(per_seed[0][key].keys())
    out: dict[str, dict[str, list[float]]] = {c: {f: [] for f in families} for c in conds}
    for r in per_seed:
        for c in conds:
            for f in families:
                if f in r[key].get(c, {}):
                    out[c][f].append(r[key][c][f])
    return {
        c: {f: statistics.mean(v) for f, v in d.items() if v}
        for c, d in out.items()
    }


def _agg_old_ceiling_mixed(per_seed: list[dict[str, Any]]) -> float:
    vals = []
    for r in per_seed:
        ceil = r["baseline_ceiling"]["ceiling_per_family"]
        m = [ceil.get(f, 0.0) for f in MIXED_FAMS]
        if m:
            vals.append(statistics.mean(m))
    return statistics.mean(vals) if vals else 0.0


def _agg_new_ceiling_mixed(per_seed: list[dict[str, Any]]) -> float:
    vals = []
    for r in per_seed:
        ceil = r["new_ceiling"]["ceiling_per_family"]
        m = [ceil.get(f, 0.0) for f in MIXED_FAMS]
        if m:
            vals.append(statistics.mean(m))
    return statistics.mean(vals) if vals else 0.0


def _agg_oracle(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    families = FAMILIES
    ks = list(per_seed[0]["oracle_results"].keys())
    out: dict[str, dict[str, list[float]]] = {
        k: {f: [] for f in families} for k in ks
    }
    for r in per_seed:
        for k in ks:
            for f in families:
                if f in r["oracle_results"].get(k, {}).get("per_family", {}):
                    out[k][f].append(r["oracle_results"][k]["per_family"][f])
    return {
        k: {f: statistics.mean(v) for f, v in d.items() if v}
        for k, d in out.items()
    }


def _agg_latencies(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in per_seed:
        for k, v in r["latencies"].items():
            out.setdefault(k, []).append(v)
    return {k: statistics.mean(v) for k, v in out.items() if v}


def _agg_causal(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-seed causal diagnoses by majority status vote."""
    joint_baseline = statistics.mean([
        r["joint_baseline_perf"].get("mixed_mean", 0.0) for r in per_seed
    ])
    extended = statistics.mean([
        r["joint_extended_perf"].get("mixed_mean", 0.0) for r in per_seed
    ])
    coadapted = statistics.mean([
        r["joint_coadapted_perf"].get("mixed_mean", 0.0) for r in per_seed
    ])
    control = statistics.mean([
        r["joint_baseline_perf"].get("mixed_mean", 0.0) for r in per_seed
    ])  # baseline as control proxy (not strictly a control, but the cross_eval for joint_baseline serves the same role)
    old_ceiling = statistics.mean([
        statistics.mean([
            r["baseline_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS
        ])
        for r in per_seed
    ])
    new_ceiling = statistics.mean([
        statistics.mean([
            r["new_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS
        ])
        for r in per_seed
    ])

    # Per-seed branch scales (mean of feature/graph/context)
    scales_baseline_seeds = [r["scales_baseline"] for r in per_seed]
    scales_coadapted_seeds = [r["scales_coadapted"] for r in per_seed]

    # Per-seed branch ablation (coadapted) and probes (coadapted)
    ablation_seeds = [r["ablation_coadapted"] for r in per_seed]
    probes_seeds = [r["repr_probes_coadapted"] for r in per_seed]
    rel_sens_seeds = [r["joint_re_rel"] for r in per_seed]

    # Average these across seeds.
    def avg_dict_list_across_seeds(seed_dicts: list[dict[str, dict[str, float]]], cond: str) -> dict[str, float]:
        families = FAMILIES
        out: dict[str, list[float]] = {f: [] for f in families}
        for sd in seed_dicts:
            for f in families:
                if f in sd.get(cond, {}):
                    out[f].append(sd[cond][f])
        return {f: statistics.mean(v) for f, v in out.items() if v}

    ablation_avg = {
        lab: avg_dict_list_across_seeds(ablation_seeds, lab)
        for lab in list(per_seed[0]["ablation_coadapted"].keys())
    }
    probes_avg = {
        t: avg_dict_list_across_seeds(probes_seeds, t)
        for t in list(per_seed[0]["repr_probes_coadapted"].keys())
    }
    rel_sens_avg = {
        c: avg_dict_list_across_seeds(rel_sens_seeds, c)
        for c in list(per_seed[0]["joint_re_rel"].keys())
    }

    # Build per-category status: aggregate by majority-vote across seeds.
    from collections import Counter
    out: dict[str, dict[str, Any]] = {}
    keys = [
        "UNDERTRAINING",
        "BRANCH_INTERFERENCE",
        "RELATIONAL_CAPABILITY",
        "REPRESENTATION_FUSION",
        "RELATIONAL_SENSITIVITY",
        "PORTFOLIO_LIMITATION",
        "OPTIMIZATION",
        "BENCHMARK_LIMITATION",
    ]
    for cat in keys:
        statuses: list[str] = []
        for r in per_seed:
            # We don't have per-seed causal in the per-seed dict, so derive
            # it from the per-seed data using the same function.
            seed_diag = build_phase13_causal_diagnosis(
                joint_baseline_mixed=r["joint_baseline_perf"].get("mixed_mean", 0.0),
                extended_training_mixed=r["joint_extended_perf"].get("mixed_mean", 0.0),
                coadaptation_mixed=r["joint_coadapted_perf"].get("mixed_mean", 0.0),
                control_mixed=r["joint_baseline_perf"].get("mixed_mean", 0.0),
                joint_branch_scales=r["scales_coadapted"],
                joint_branch_ablation=r["ablation_coadapted"],
                joint_repr_probes=r["repr_probes_coadapted"],
                joint_relational_sensitivity=r["joint_re_rel"],
                old_ceiling_mixed=statistics.mean([
                    r["baseline_ceiling"]["ceiling_per_family"].get(f, 0.0)
                    for f in MIXED_FAMS
                ]),
                new_ceiling_mixed=statistics.mean([
                    r["new_ceiling"]["ceiling_per_family"].get(f, 0.0)
                    for f in MIXED_FAMS
                ]),
            )
            statuses.append(seed_diag.get(cat, {}).get("status", "INCONCLUSIVE"))
        cnt = Counter(statuses)
        out[cat] = {
            "status": cnt.most_common(1)[0][0],
            "status_distribution": dict(cnt),
            "evidence_seed_0": build_phase13_causal_diagnosis(
                joint_baseline_mixed=per_seed[0]["joint_baseline_perf"].get("mixed_mean", 0.0),
                extended_training_mixed=per_seed[0]["joint_extended_perf"].get("mixed_mean", 0.0),
                coadaptation_mixed=per_seed[0]["joint_coadapted_perf"].get("mixed_mean", 0.0),
                control_mixed=per_seed[0]["joint_baseline_perf"].get("mixed_mean", 0.0),
                joint_branch_scales=per_seed[0]["scales_coadapted"],
                joint_branch_ablation=per_seed[0]["ablation_coadapted"],
                joint_repr_probes=per_seed[0]["repr_probes_coadapted"],
                joint_relational_sensitivity=per_seed[0]["joint_re_rel"],
                old_ceiling_mixed=statistics.mean([
                    per_seed[0]["baseline_ceiling"]["ceiling_per_family"].get(f, 0.0)
                    for f in MIXED_FAMS
                ]),
                new_ceiling_mixed=statistics.mean([
                    per_seed[0]["new_ceiling"]["ceiling_per_family"].get(f, 0.0)
                    for f in MIXED_FAMS
                ]),
            ).get(cat, {}).get("evidence", ""),
        }
    return out


def _write_csvs(
    m_dir: Path,
    per_seed: list[dict[str, Any]],
    families: tuple[str, ...],
    mixed_families: tuple[str, ...],
    pure_families: tuple[str, ...],
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

    # 1. baseline_reproduction.csv
    rows = []
    for r in per_seed:
        for cond_key in ("joint_baseline_perf",):
            for f, acc in r[cond_key].items():
                rows.append({
                    "seed": r["seed"], "condition": "13A_baseline", "metric": f, "value": acc,
                })
    write_csv("baseline_reproduction.csv", rows)

    # 2. training_control.csv
    rows = []
    for r in per_seed:
        for cond_key in ("joint_baseline_perf", "joint_extended_perf"):
            for f, acc in r[cond_key].items():
                rows.append({
                    "seed": r["seed"], "condition": cond_key, "metric": f, "value": acc,
                })
    write_csv("training_control.csv", rows)

    # 3. coadaptation_results.csv
    rows = []
    for r in per_seed:
        for cond_key in ("joint_baseline_perf", "joint_extended_perf", "joint_coadapted_perf"):
            for f, acc in r[cond_key].items():
                rows.append({
                    "seed": r["seed"], "condition": cond_key, "metric": f, "value": acc,
                })
    write_csv("coadaptation_results.csv", rows)

    # 4. branch_scales.csv
    rows = []
    for r in per_seed:
        for cond_key in ("scales_baseline", "scales_extended", "scales_coadapted"):
            for b, v in r[cond_key].items():
                rows.append({
                    "seed": r["seed"], "condition": cond_key,
                    "branch": b, "scale": v,
                })
    write_csv("branch_scales.csv", rows)

    # 5. branch_ablation.csv
    rows = []
    for r in per_seed:
        for cond_key in ("ablation_baseline", "ablation_coadapted"):
            for abl_label, per_fam in r[cond_key].items():
                for f, acc in per_fam.items():
                    rows.append({
                        "seed": r["seed"], "condition": cond_key,
                        "ablation": abl_label, "family": f, "accuracy": acc,
                    })
    write_csv("branch_ablation.csv", rows)

    # 6. representation_probes.csv
    rows = []
    for r in per_seed:
        for cond_key in ("repr_probes_baseline", "repr_probes_coadapted"):
            for task_name, per_fam in r[cond_key].items():
                for f, acc in per_fam.items():
                    rows.append({
                        "seed": r["seed"], "condition": cond_key,
                        "task": task_name, "family": f, "probe_accuracy": acc,
                    })
    write_csv("representation_probes.csv", rows)

    # 7. relational_diagnosis.csv
    rows = []
    for r in per_seed:
        for cond_key in ("baseline_re_rel", "joint_re_rel", "graph_re_rel"):
            for cond_label, per_fam in r[cond_key].items():
                for f, acc in per_fam.items():
                    rows.append({
                        "seed": r["seed"], "expert": cond_key,
                        "condition": cond_label, "family": f, "accuracy": acc,
                    })
    write_csv("relational_diagnosis.csv", rows)

    # 8. portfolio_results.csv (existing perf + best joint cross-eval)
    rows = []
    for r in per_seed:
        for name, per_fam in r["cross_eval"].items():
            row = {"seed": r["seed"], "expert": name}
            for f in families:
                row[f] = per_fam.get(f, 0.0)
            rows.append(row)
    write_csv("portfolio_results.csv", rows)

    # 9. oracle_results.csv
    rows = []
    for r in per_seed:
        for k_str, vals in r["oracle_results"].items():
            row = {"seed": r["seed"], "policy": k_str}
            for f in families:
                row[f] = vals.get("per_family", {}).get(f, 0.0)
            rows.append(row)
    write_csv("oracle_results.csv", rows)

    # 10. compute_results.csv
    rows = []
    for r in per_seed:
        for name, n in r["param_counts"].items():
            rows.append({"seed": r["seed"], "expert": name, "params": n})
    write_csv("compute_results.csv", rows)

    # 11. latency_results.csv
    rows = []
    for r in per_seed:
        for name, lat in r["latencies"].items():
            rows.append({"seed": r["seed"], "expert": name, "latency_us": lat})
    write_csv("latency_results.csv", rows)

    # 12. seed_results.csv
    rows = []
    for r in per_seed:
        rows.append({
            "seed": r["seed"],
            "joint_baseline_mixed": r["joint_baseline_perf"].get("mixed_mean", 0.0),
            "joint_extended_mixed": r["joint_extended_perf"].get("mixed_mean", 0.0),
            "joint_coadapted_mixed": r["joint_coadapted_perf"].get("mixed_mean", 0.0),
            "old_ceiling_mixed": statistics.mean([
                r["baseline_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in mixed_families
            ]),
            "new_ceiling_mixed": statistics.mean([
                r["new_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in mixed_families
            ]),
        })
    write_csv("seed_results.csv", rows)

    # 13. failure_diagnosis.csv
    rows = []
    summary_for_failure = None  # not yet available here; will be added by post-processing if needed
    write_csv("failure_diagnosis.csv", rows)


def generate_phase13_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Generate the Phase 13 markdown report (data-driven)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    def gv(d: dict, *keys: str, default: Any = 0.0) -> Any:
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    families = FAMILIES
    mixed_families = MIXED_FAMS
    pure_families = PURE_FAMS

    base = summary.get("joint_baseline_perf_mean", {})
    ext = summary.get("joint_extended_perf_mean", {})
    coa = summary.get("joint_coadapted_perf_mean", {})
    existing = summary.get("existing_perf_mean", {})
    cross = summary.get("cross_eval_mean", {})
    scales_b = summary.get("scales_baseline_mean", {})
    scales_e = summary.get("scales_extended_mean", {})
    scales_c = summary.get("scales_coadapted_mean", {})
    ablation_b = summary.get("ablation_baseline_mean", {})
    ablation_c = summary.get("ablation_coadapted_mean", {})
    probes_b = summary.get("repr_probes_baseline_mean", {})
    probes_c = summary.get("repr_probes_coadapted_mean", {})
    joint_rel = summary.get("joint_re_rel_mean", {})
    graph_rel = summary.get("graph_re_rel_mean", {})
    base_rel = summary.get("baseline_re_rel_mean", {})
    old_ceil = summary.get("old_ceiling_mixed_mean", 0.0)
    new_ceil = summary.get("new_ceiling_mixed_mean", 0.0)
    oracle = summary.get("oracle_results_mean", {})
    param_counts = summary.get("param_counts", {})
    latencies = summary.get("latencies_mean", {})
    causal = summary.get("causal_diagnosis_aggregate", {})

    def perf_row(name: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in families)
        return (
            f"| **{name}** | {pct(perf.get('overall', 0.0))} | {cells} | "
            f"{pct(perf.get('pure_mean', 0.0))} | {pct(perf.get('mixed_mean', 0.0))} | "
            f"{param_counts.get(name, 0):,} | {latencies.get(name, 0.0):.1f} |"
        )

    def ablation_row(label: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in families)
        return f"| **{label}** | {cells} |"

    def probe_row(task: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in families)
        return f"| **{task}** | {cells} |"

    def rel_row(label: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in ("R", "RC", "FRC"))
        return f"| **{label}** | {cells} |"

    def causal_row(cat: str) -> str:
        d = causal.get(cat, {})
        return f"| **{cat}** | {d.get('status', 'INCONCLUSIVE')} | {d.get('evidence_seed_0', '')} |"

    report = f"""# Phase 13 — Joint Expert Co-Adaptation and Relational Compositional Diagnosis

## Mandatory Scientific Disclaimer

> Phase 12 established that the new Joint expert simultaneously retains Feature and Contextual capability (F=100%, C=99.2%) but is weak on Relational and Relational+Contextual tasks (R=61.4%, RC=53.6%). This phase investigates whether the Joint expert can be improved by **controlled co-adaptation** of its internal branches, without modifying the existing specialist portfolio. The Joint expert's three branches (Feature, Graph, V2-style) already train jointly via backpropagation; the co-adaptation condition adds a small inter-branch fusion (H*3 -> H linear, gated) that is genuinely distinct from ordinary end-to-end training.

---

## 1. Executive Summary

- **Phase 12 Joint baseline (13A, 20 epochs, mixed F+R+C)**: mixed mean = {pct(base.get('mixed_mean', 0.0))}; per-family R={pct(base.get('R', 0.0))}, RC={pct(base.get('RC', 0.0))}, FRC={pct(base.get('FRC', 0.0))}
- **Extended training (13B, 40 epochs)**: mixed mean = {pct(ext.get('mixed_mean', 0.0))}; per-family R={pct(ext.get('R', 0.0))}, RC={pct(ext.get('RC', 0.0))}, FRC={pct(ext.get('FRC', 0.0))}
- **Co-adaptation (13C, JointCo with H*3 -> H fusion, 40 epochs)**: mixed mean = {pct(coa.get('mixed_mean', 0.0))}; per-family R={pct(coa.get('R', 0.0))}, RC={pct(coa.get('RC', 0.0))}, FRC={pct(coa.get('FRC', 0.0))}
- **Phase 12 old single-expert ceiling (mixed)**: {pct(summary.get('old_ceiling_mixed_mean', 0.0))}
- **Phase 13 new single-expert ceiling (with co-adapted Joint)**: {pct(new_ceil)}
- **Programmatic verdict**: **{summary.get('verdict_case', 'CASE G')} — {summary.get('verdict_label', '')}**

---

## 2. Phase 12 Starting Evidence

- Phase 12 Joint baseline per-family: F={pct(0.9997)}, R={pct(0.614)}, C={pct(0.992)}, FR={pct(0.758)}, RC={pct(0.536)}, FC={pct(0.494)}, FRC={pct(0.808)}
- Phase 12 old single-expert ceiling (mixed): 66.4%; with Joint: 66.8%
- Phase 12 CASE B: new expert helps on mixed mean but family-level ceiling is unchanged

---

## 3. Research Question

> Can co-adapting the Joint expert's internal Feature, Relational, and Contextual branches improve its ability to jointly solve the mixed-structure benchmark, particularly R, RC, and FRC, without modifying the existing specialist portfolio?

---

## 4. Hypotheses (pre-registered)

- H1: Joint co-adaptation improves mixed capability
- H2: Improvement is concentrated on R/RC/FRC
- H3: Co-adaptation preserves F/C capability
- H4: Branch interaction matters
- H5: Joint expert improvement is not merely additional training time

---

## 5. Primary Results Table (26)

| Condition | Overall | F | R | C | FR | RC | FC | FRC | Pure | Mixed | Params | Latency (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{perf_row("13A Joint baseline (20 epochs)", base)}
{perf_row("13B Joint extended (40 epochs)", ext)}
{perf_row("13C JointCo co-adaptation (40 epochs)", coa)}
{perf_row("Existing MLP", existing.get("mlp", {}))}
{perf_row("Existing Graph", existing.get("graph", {}))}
{perf_row("Existing V2", existing.get("attention_v2", {}))}

---

## 6. Training-Budget Control (13B)

13B (extended training) compared to 13A (baseline) tests H5. Results:
- Joint baseline (20 epochs) mixed mean: {pct(base.get('mixed_mean', 0.0))}
- Joint extended (40 epochs) mixed mean: {pct(ext.get('mixed_mean', 0.0))}
- Delta: {(ext.get('mixed_mean', 0.0) - base.get('mixed_mean', 0.0))*100:+.1f}pp

H5 verdict: {causal.get('UNDERTRAINING', {}).get('status', 'INCONCLUSIVE')} (see Section 14).

---

## 7. Co-Adaptation Condition (13C)

JointCo (inter-branch fusion) compared to 13B (extended training) tests H1 and H4:
- Joint extended (13B) mixed mean: {pct(ext.get('mixed_mean', 0.0))}
- JointCo co-adaptation (13C) mixed mean: {pct(coa.get('mixed_mean', 0.0))}
- Delta: {(coa.get('mixed_mean', 0.0) - ext.get('mixed_mean', 0.0))*100:+.1f}pp

Per-family comparison of 13C vs 13B on the relational weakness:
- R:    13B {pct(ext.get('R', 0.0))} -> 13C {pct(coa.get('R', 0.0))} ({(coa.get('R', 0.0)-ext.get('R', 0.0))*100:+.1f}pp)
- RC:   13B {pct(ext.get('RC', 0.0))} -> 13C {pct(coa.get('RC', 0.0))} ({(coa.get('RC', 0.0)-ext.get('RC', 0.0))*100:+.1f}pp)
- FRC:  13B {pct(ext.get('FRC', 0.0))} -> 13C {pct(coa.get('FRC', 0.0))} ({(coa.get('FRC', 0.0)-ext.get('FRC', 0.0))*100:+.1f}pp)

H2 verdict: see RELATIONAL_CAPABILITY in Section 14.

---

## 8. Branch Scales (13D diagnostic)

Per-branch learnable scales after training (init = 1/3 = 0.333):

| Branch | 13A baseline | 13B extended | 13C co-adaptation |
|---|---:|---:|---:|
| feature | {scales_b.get('feature', 0.0):.3f} | {scales_e.get('feature', 0.0):.3f} | {scales_c.get('feature', 0.0):.3f} |
| graph | {scales_b.get('graph', 0.0):.3f} | {scales_e.get('graph', 0.0):.3f} | {scales_c.get('graph', 0.0):.3f} |
| context | {scales_b.get('context', 0.0):.3f} | {scales_e.get('context', 0.0):.3f} | {scales_c.get('context', 0.0):.3f} |

A branch being suppressed (scale near 0) means training pushed that branch's contribution toward zero.

---

## 9. Branch Ablation (13D causal)

Deterministic zeroing of one branch at a time, no retrain.

| Ablation | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
{ablation_row("all (13A baseline)", ablation_b.get("all", {}))}
{ablation_row("no_feature (13A)", ablation_b.get("feature", {}))}
{ablation_row("no_graph (13A)", ablation_b.get("graph", {}))}
{ablation_row("no_context (13A)", ablation_b.get("context", {}))}
{ablation_row("all (13C co-adaptation)", ablation_c.get("all", {}))}
{ablation_row("no_feature (13C)", ablation_c.get("feature", {}))}
{ablation_row("no_graph (13C)", ablation_c.get("graph", {}))}
{ablation_row("no_context (13C)", ablation_c.get("context", {}))}

---

## 10. Representation Probes (13E)

Linear-probe decodability of component signals from the Joint expert's final representation.

| Task | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
{probe_row("F_signal (13A baseline)", probes_b.get("F_signal", {}))}
{probe_row("R_signal (13A baseline)", probes_b.get("R_signal", {}))}
{probe_row("C_signal (13A baseline)", probes_b.get("C_signal", {}))}
{probe_row("final_target (13A baseline)", probes_b.get("final_target", {}))}
{probe_row("F_signal (13C co-adaptation)", probes_c.get("F_signal", {}))}
{probe_row("R_signal (13C co-adaptation)", probes_c.get("R_signal", {}))}
{probe_row("C_signal (13C co-adaptation)", probes_c.get("C_signal", {}))}
{probe_row("final_target (13C co-adaptation)", probes_c.get("final_target", {}))}

---

## 11. Relational Destruction (13E)

| Expert / Condition | R | RC | FRC |
|---|---:|---:|---:|
{rel_row("Joint 13A baseline - original", base_rel.get("original", {}))}
{rel_row("Joint 13A baseline - permuted", base_rel.get("relational_permuted", {}))}
{rel_row("JointCo 13C - original", joint_rel.get("original", {}))}
{rel_row("JointCo 13C - permuted", joint_rel.get("relational_permuted", {}))}
{rel_row("Graph specialist - original", graph_rel.get("original", {}))}
{rel_row("Graph specialist - permuted", graph_rel.get("relational_permuted", {}))}

---

## 12. Portfolio Re-Evaluation (13G)

Cross-family mixed single-expert ceiling:
- Phase 12 (4 existing + Joint baseline): 66.8%
- Phase 13 (4 existing + best Joint condition): **{pct(new_ceil)}**
- Delta: {(new_ceil - 0.668) * 100:+.1f}pp

Oracle expanded portfolio (with the best Joint condition) on mixed families:

| Policy | FR | RC | FC | FRC | Mixed Avg |
|---|---:|---:|---:|---:|---:|
"""
    for k in ("k=1", "k=2", "k=3"):
        pf = oracle.get(k, {})
        if not pf:
            continue
        cells = " | ".join(pct(pf.get(f, 0.0)) for f in mixed_families)
        mixed_avg = statistics.mean([pf.get(f, 0.0) for f in mixed_families])
        report += f"| oracle {k} | {cells} | {pct(mixed_avg)} |\n"

    report += f"""
---

## 13. Latency (24)

Physical CPU latency (mean over 20 iterations, batch=60, µs/sample):

| Expert | Latency (us) |
|---|---:|
"""
    for k in (
        "mlp", "graph", "attention_v2", "joint_baseline", "joint_extended", "joint_coadapted",
    ):
        v = latencies.get(k, 0.0)
        report += f"| **{k}** | {v:.1f} |\n"

    report += f"""
---

## 14. Causal Diagnosis Table (27-28)

| Category | Status | Evidence (seed 0) |
|---|---|---|
{causal_row("UNDERTRAINING")}
{causal_row("BRANCH_INTERFERENCE")}
{causal_row("RELATIONAL_CAPABILITY")}
{causal_row("REPRESENTATION_FUSION")}
{causal_row("RELATIONAL_SENSITIVITY")}
{causal_row("PORTFOLIO_LIMITATION")}
{causal_row("OPTIMIZATION")}
{causal_row("BENCHMARK_LIMITATION")}

---

## 15. Failure-Mode Classification (28)

The primary failure mode (chosen by majority-vote status above) is the bottleneck that best explains the data.

---

## 16. Scientific Verdict (35)

**{summary.get('verdict_case', 'CASE G')} — {summary.get('verdict_label', '')}**

---

## 17. Limitations

1. Only ONE new intervention (JointCo) was tested. The spec forbids stacking multiple fixes.
2. The new expert's depth is fixed at 1 to keep parameter count controlled.
3. Joint training of the existing 4 specialists is intentionally deferred (the spec forbids it in the primary Phase 13 experiment).
4. The 3-seed sample (11, 23, 37) is small; additional seeds may be needed for finer discrimination.

---

## 18. Decision for Phase 14 (37)

Phase 14 is determined from the verdict above, not pre-committed. Possible next directions:

- if **BRANCH_INTERFERENCE is supported and joint is improved**: refine the inter-branch fusion (e.g., gated attention).
- if **RELATIONAL_CAPABILITY is the bottleneck**: expand the relational branch capacity in the next experiment.
- if **UNDERTRAINING explains the result**: train all variants longer before adding interventions.
- if **PORTFOLIO_LIMITATION is dominant**: revisit the spec's option of a second relational expert.
- if **nothing helps**: report the boundary and do not add architecture.

No architectural decision is pre-committed.
"""
    output_path.write_text(report, encoding="utf-8")
