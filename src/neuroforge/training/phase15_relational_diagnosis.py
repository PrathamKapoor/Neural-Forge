"""Phase 15 relational-substep capability and message-passing diagnosis runner.

Additive: does NOT modify any Phase 1-14 module. Reuses:
  - `StandaloneSpecialist` + `_train_joint_on_mixed` (Phase 12) for training;
  - `_train_one_expert` (Phase 12) + `select_capacity_phase6` for reference experts;
  - Phase 13 evaluators (branch ablation, probes, destruction, ceiling);
  - Phase 9 causal controls (relational permutation, token permutation);
  - `JointCoRelationalBlock` (new, Phase 15) for depth/aggregation/capacity variants.

Experimental order (§5-§8):
  1. baseline (exact JointCo replica protocol) + depth ablation (1/2/3);
  2. sequential gate: aggregation runs ONLY if depth does not fully explain;
  3. sequential gate: capacity runs ONLY if depth/aggregation do not explain;
  4. topology + marker/position controls + branch interaction + probes + graph
     comparison + portfolio ceiling + compute/latency for the best candidate.

Per-seed/per-condition weights + evals are cached under
`<metrics_dir>/partials/` so interrupted runs resume without retraining.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock
from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase13_metrics import (
    expert_branch_scales,
    relational_destruction_accuracy,
    single_expert_ceiling,
)
from neuroforge.evaluation.phase15_metrics import (
    FAMILIES,
    MIXED_FAMS,
    activation_bytes,
    branch_7way_ablation,
    build_failure_diagnosis,
    build_phase15_hypotheses,
    count_parameters,
    latency_microseconds,
    marker_ablation_accuracy,
    per_family_accuracy,
    relational_parameters_of,
    relational_probe_triplet,
    select_minimal_intervention,
    select_phase15_case,
    token_permutation_accuracy,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase12_expert_portfolio import (
    _train_joint_on_mixed,
    _train_one_expert,
)

# ---------------------------------------------------------------------------
# Variant registry: cond label -> block kwargs (None = stock JointCo baseline)
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict[str, Any] | None] = {
    "baseline": None,  # exact Phase 14 JointCo
    "depth2": {"rel_depth": 2, "aggregation": "mean", "rel_update": "linear"},
    "depth3": {"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"},
    "agg_sum": {"rel_depth": 1, "aggregation": "sum", "rel_update": "linear"},
    "agg_max": {"rel_depth": 1, "aggregation": "max", "rel_update": "linear"},
    "cap_mlp": {"rel_depth": 1, "aggregation": "mean", "rel_update": "mlp"},
    "cap_control_featwide": {
        "rel_depth": 1,
        "aggregation": "mean",
        "rel_update": "linear",
        "feature_hidden": 36,
    },
}

DEPTH_CONDS = ("baseline", "depth2", "depth3")
AGG_CONDS = ("agg_sum", "agg_max")
CAP_CONDS = ("cap_mlp", "cap_control_featwide")


def build_expert_for_variant(variant_kwargs: dict[str, Any] | None) -> StandaloneSpecialist:
    """Construct a JointCo expert, swapping in the relational variant block."""
    expert = StandaloneSpecialist("joint_co", input_dim=8, hidden_dim=24, depth=1)
    if variant_kwargs is not None:
        expert.blocks[0] = JointCoRelationalBlock(hidden_dim=24, **variant_kwargs)  # type: ignore[arg-type]
    return expert


def _config_hash(cond: str, variant_kwargs: dict[str, Any] | None, seed: int, epochs: int, batch_size: int) -> str:
    raw = json.dumps({"cond": cond, "kw": variant_kwargs, "seed": seed, "epochs": epochs, "bs": batch_size}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def train_variant(
    cond: str,
    seed: int,
    epochs: int,
    batch_size: int,
    partials_dir: Path,
) -> StandaloneSpecialist:
    """Train one (condition, seed) model, resuming cached weights if present."""
    variant_kwargs = VARIANTS[cond]
    cfg = _config_hash(cond, variant_kwargs, seed, epochs, batch_size)
    pt_path = partials_dir / f"seed{seed}_{cond}.pt"
    meta_path = partials_dir / f"seed{seed}_{cond}.json"
    expert = build_expert_for_variant(variant_kwargs)
    if pt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("config_hash") == cfg:
                expert.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                expert.eval()
                return expert
        except Exception:
            pass  # fall through to retraining
    torch.manual_seed(seed)
    _train_joint_on_mixed(expert, seed=seed, epochs=epochs, batch_size=batch_size)
    expert.eval()
    partials_dir.mkdir(parents=True, exist_ok=True)
    torch.save(expert.state_dict(), pt_path)
    meta_path.write_text(json.dumps({"config_hash": cfg, "cond": cond, "seed": seed}), encoding="utf-8")
    return expert


def train_reference_expert(
    architecture: str,
    target_family: str,
    depth: int,
    seed: int,
    epochs: int,
    batch_size: int,
    partials_dir: Path,
) -> StandaloneSpecialist:
    """Train (or resume) a reference specialist (graph / mlp / attention / v2)."""
    tag = f"ref_{architecture}"
    raw = json.dumps({"arch": architecture, "fam": target_family, "depth": depth, "seed": seed, "epochs": epochs}, sort_keys=True)
    cfg = hashlib.sha256(raw.encode()).hexdigest()[:12]
    pt_path = partials_dir / f"seed{seed}_{tag}.pt"
    meta_path = partials_dir / f"seed{seed}_{tag}.json"
    if pt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("config_hash") == cfg:
                from neuroforge.models.specialists import StandaloneSpecialist as SS
                expert = SS(architecture, input_dim=8, hidden_dim=24, depth=depth)
                expert.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                expert.eval()
                return expert
        except Exception:
            pass
    torch.manual_seed(seed)
    expert = _train_one_expert(
        architecture=architecture, target_family=target_family, depth=depth,
        seed=seed, epochs=epochs, batch_size=batch_size,
    )
    expert.eval()
    partials_dir.mkdir(parents=True, exist_ok=True)
    torch.save(expert.state_dict(), pt_path)
    meta_path.write_text(json.dumps({"config_hash": cfg, "tag": tag, "seed": seed}), encoding="utf-8")
    return expert


def evaluate_condition(
    expert: StandaloneSpecialist,
    eval_ds: Phase9MixedStructureDataset,
    seed: int,
    do_branch7: bool = True,
) -> dict[str, Any]:
    """Full per-condition evaluation bundle (no retraining).

    `do_branch7=False` for single-branch reference specialists (e.g. the
    standalone Graph expert), where scale-zeroing branch ablation is not
    applicable; the bundle records an explicit not-applicable marker.
    """
    perf = per_family_accuracy(expert, eval_ds)
    probe = relational_probe_triplet(expert, eval_ds)
    destruction = relational_destruction_accuracy(expert, eval_ds, seed=seed)
    marker = marker_ablation_accuracy(expert, eval_ds)
    token = token_permutation_accuracy(expert, eval_ds, seed=seed)
    if do_branch7:
        branch7: dict[str, Any] = branch_7way_ablation(expert, eval_ds)
        scales: dict[str, Any] = expert_branch_scales(expert)
    else:
        branch7 = {"not_applicable": "single-branch specialist has no branch scales to ablate"}
        scales = {"not_applicable": "single-branch specialist"}
    params = count_parameters(expert)
    rel_params = relational_parameters_of(expert)
    lat = latency_microseconds(expert, eval_ds.features[:60])
    destruction_drop_r = destruction.get("original", {}).get("R", 0.0) - destruction.get("relational_permuted", {}).get("R", 0.0)
    return {
        "perf": perf,
        "probe": probe,
        "destruction": destruction,
        "destruction_drop_R": destruction_drop_r,
        "marker": marker,
        "token": token,
        "branch7": branch7,
        "scales": scales,
        "params": params,
        "rel_params": rel_params,
        "latency_us": lat,
        "throughput_per_s": 1e6 / max(lat, 1e-9),
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _agg_perf(per_seed: list[dict[str, Any]], cond: str) -> dict[str, Any]:
    keys = list(FAMILIES) + ["pure_mean", "mixed_mean", "overall"]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [r[cond]["perf"].get(k, 0.0) for r in per_seed if cond in r]
        if vals:
            out[k] = mean_sd(vals)
    return out


def _r_gains(per_seed: list[dict[str, Any]], cond: str, fam: str = "R") -> list[float]:
    return [
        r[cond]["perf"].get(fam, 0.0) - r["baseline"]["perf"].get(fam, 0.0)
        for r in per_seed if cond in r and "baseline" in r
    ]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_phase15_relational_diagnosis(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    jointco_epochs: int = 40,
    samples_per_type: int = 120,
    batch_size: int = 60,
    stages: tuple[str, ...] = ("baseline", "depth", "gated", "graph", "existing", "finalize"),
) -> dict[str, Any]:
    """Execute the Phase 15 relational-substep diagnosis (staged, resumable)."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)
    partials_dir = m_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "Phase 15 — Relational Substep Capability and Message-Passing Diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "evaluated_families": list(FAMILIES),
        "samples_per_type": samples_per_type,
        "jointco_epochs": jointco_epochs,
        "batch_size": batch_size,
        "stages": list(stages),
        "protocol": (
            "Same architecture/dataset/train-protocol/optimizer/budget as Phase 14 "
            "JointCo baseline (mixed F+R+C, AdamW lr=0.003, best-val checkpoint, "
            "frozen at eval). Only the relational substep varies (depth / "
            "aggregation / update capacity). Router frozen."
        ),
        "gate_thresholds": {
            "depth_fully_explains_R_gain": 0.05,
            "depth_fully_explains_gap_close": 0.03,
            "aggregation_adequate_R_gain": 0.05,
        },
    }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        seed_res: dict[str, Any] = {"seed": seed}

        def ensure(cond: str) -> StandaloneSpecialist:
            return train_variant(cond, seed, jointco_epochs, batch_size, partials_dir)

        if "baseline" in stages:
            seed_res["baseline_eval"] = evaluate_condition(ensure("baseline"), eval_ds, seed)
        if "depth" in stages:
            for cond in ("depth2", "depth3"):
                seed_res[f"{cond}_eval"] = evaluate_condition(ensure(cond), eval_ds, seed)
        per_seed.append(seed_res)

    # ---- Interim aggregate for sequential gating (baseline + depth) ----
    def interim_depth_stats() -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for cond in ("depth2", "depth3"):
            key = f"{cond}_eval"
            gains = [
                r[key]["perf"].get("R", 0.0) - r["baseline_eval"]["perf"].get("R", 0.0)
                for r in per_seed if key in r and "baseline_eval" in r
            ]
            gap_close = [
                r["baseline_eval"]["probe"].get("gap", 0.0) - r[key]["probe"].get("gap", 0.0)
                for r in per_seed if key in r and "baseline_eval" in r
            ]
            stats[cond] = {
                "R_gain_mean": statistics.mean(gains) if gains else 0.0,
                "R_gain_n_positive": sum(1 for g in gains if g > 0),
                "gap_close_mean": statistics.mean(gap_close) if gap_close else 0.0,
            }
        return stats

    gate_info: dict[str, Any] = {"aggregation_run": False, "capacity_run": False, "reasons": []}
    run_agg = False
    run_cap = False
    if "gated" in stages and "baseline" in stages and "depth" in stages:
        dstats = interim_depth_stats()
        best = max(dstats, key=lambda c: dstats[c]["R_gain_mean"])
        b = dstats[best]
        depth_fully_explains = (
            b["R_gain_mean"] >= 0.05 and b["gap_close_mean"] >= 0.03 and b["R_gain_n_positive"] >= 2
        )
        gate_info["depth_stats"] = dstats
        gate_info["depth_fully_explains"] = depth_fully_explains
        if depth_fully_explains:
            gate_info["reasons"].append(
                f"Depth {best} fully explains (R gain {b['R_gain_mean'] * 100:+.1f}pp, gap closes "
                f"{b['gap_close_mean'] * 100:+.1f}pp): aggregation/capacity gated off per §7/§8."
            )
        else:
            gate_info["reasons"].append(
                f"Depth does not fully explain (best {best}: R gain {b['R_gain_mean'] * 100:+.1f}pp, "
                f"gap closes {b['gap_close_mean'] * 100:+.1f}pp): running aggregation per §7."
            )
            run_agg = True
        gate_info["aggregation_run"] = run_agg

        if run_agg:
            for r in per_seed:
                eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
                for cond in AGG_CONDS:
                    r[f"{cond}_eval"] = evaluate_condition(
                        train_variant(cond, r["seed"], jointco_epochs, batch_size, partials_dir), eval_ds, r["seed"]
                    )
            agg_gains = {}
            for cond in AGG_CONDS:
                key = f"{cond}_eval"
                gains = [
                    rr[key]["perf"].get("R", 0.0) - rr["baseline_eval"]["perf"].get("R", 0.0)
                    for rr in per_seed if key in rr and "baseline_eval" in rr
                ]
                agg_gains[cond] = statistics.mean(gains) if gains else 0.0
            agg_best = max(agg_gains, key=lambda c: agg_gains[c])
            agg_adequate = agg_gains[agg_best] >= 0.05
            gate_info["aggregation_gains"] = agg_gains
            if agg_adequate:
                gate_info["reasons"].append(
                    f"Aggregation {agg_best} adequate (R gain {agg_gains[agg_best] * 100:+.1f}pp): capacity gated off per §8."
                )
            else:
                gate_info["reasons"].append(
                    f"Aggregation inadequate (best {agg_best}: {agg_gains[agg_best] * 100:+.1f}pp): running capacity per §8."
                )
                run_cap = True
        else:
            gate_info["reasons"].append("Capacity gated off: depth fully explained (§8).")
        gate_info["capacity_run"] = run_cap

        if run_cap:
            for r in per_seed:
                eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
                for cond in CAP_CONDS:
                    r[f"{cond}_eval"] = evaluate_condition(
                        train_variant(cond, r["seed"], jointco_epochs, batch_size, partials_dir), eval_ds, r["seed"]
                    )

    # ---- Graph reference (15I) ----
    if "graph" in stages:
        selected = select_capacity_phase6()
        for r in per_seed:
            eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
            g = train_reference_expert("graph", "relational", selected.depths["graph"], r["seed"], jointco_epochs, batch_size, partials_dir)
            r["graph_eval"] = evaluate_condition(g, eval_ds, r["seed"], do_branch7=False)

    # ---- Existing experts for portfolio ceiling (15K) ----
    if "existing" in stages:
        selected = select_capacity_phase6()
        for r in per_seed:
            eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
            logits: dict[str, torch.Tensor] = {}
            for arch, fam in [("mlp", "feature"), ("attention", "contextual"), ("attention_v2", "contextual")]:
                e = train_reference_expert(arch, fam, selected.depths[arch], r["seed"], jointco_epochs, batch_size, partials_dir)
                e.eval()
                with torch.no_grad():
                    logits[arch] = e(eval_ds.features)
            # graph logits from the graph-stage model (retrain cheap if absent)
            if "graph_eval" not in r:
                g = train_reference_expert("graph", "relational", selected.depths["graph"], r["seed"], jointco_epochs, batch_size, partials_dir)
                r["graph_eval"] = evaluate_condition(g, eval_ds, r["seed"], do_branch7=False)
            with torch.no_grad():
                g2 = train_reference_expert("graph", "relational", selected.depths["graph"], r["seed"], jointco_epochs, batch_size, partials_dir)
                g2.eval()
                logits["graph"] = g2(eval_ds.features)
            # candidate logits: best validated intervention (or best depth) vs baseline
            r["_ceiling_logits"] = {k: v.clone() for k, v in logits.items()}

    if "finalize" not in stages:
        return {"manifest": manifest, "per_seed_keys": [list(r.keys()) for r in per_seed], "gate_info": gate_info}

    # =====================================================================
    # FINALIZE: aggregates, hypotheses, case, CSVs, figures, report
    # =====================================================================
    from neuroforge.visualization.phase15_plots import generate_phase15_figures

    eval_key = lambda c: f"{c}_eval"  # noqa: E731
    tested_conds = [c for c in VARIANTS if any(eval_key(c) in r for r in per_seed)]

    # Candidate = minimal validated intervention inputs: best depth by mean R.
    depth_means = {}
    for cond in DEPTH_CONDS:
        if cond == "baseline":
            continue
        vals = [r[eval_key(cond)]["perf"].get("R", 0.0) for r in per_seed if eval_key(cond) in r]
        if vals:
            depth_means[cond] = statistics.mean(vals)
    base_r_mean = statistics.mean([r["baseline_eval"]["perf"].get("R", 0.0) for r in per_seed if "baseline_eval" in r])
    best_depth = max(depth_means, key=lambda c: depth_means[c]) if depth_means else "depth2"
    candidate_cond = best_depth  # provisional; refined by select_minimal_intervention below

    def cond_mean(cond: str, fam: str) -> float:
        vals = [r[eval_key(cond)]["perf"].get(fam, 0.0) for r in per_seed if eval_key(cond) in r]
        return statistics.mean(vals) if vals else 0.0

    # ---- depth aggregate ----
    depth_agg: dict[str, Any] = {"tested": any(eval_key(c) in r for r in per_seed for c in ("depth2", "depth3")), "variants": {}}
    for cond in ("depth2", "depth3"):
        gains = _r_gains_seed(per_seed, cond)
        if gains:
            npos = sum(1 for g in gains if g > 0)
            mg = statistics.mean(gains)
            status, _ = _status_for_gain(mg, npos, len(gains))
            depth_agg["variants"][cond] = {"R_gain_mean": mg, "R_gain_n_positive": npos, "status": status}
    if depth_agg["tested"]:
        bl = max(depth_agg["variants"], key=lambda c: depth_agg["variants"][c]["R_gain_mean"])
        depth_agg["best_label"] = bl
        depth_agg["best_R_gain_mean"] = depth_agg["variants"][bl]["R_gain_mean"]
        depth_agg["best_R_gain_n_positive"] = depth_agg["variants"][bl]["R_gain_n_positive"]

    # ---- aggregation aggregate ----
    agg_tested = any(eval_key(c) in r for r in per_seed for c in AGG_CONDS)
    aggregation_agg: dict[str, Any] = {"tested": agg_tested, "variants": {}}
    if agg_tested:
        for cond in AGG_CONDS:
            gains = _r_gains_seed(per_seed, cond)
            if gains:
                npos = sum(1 for g in gains if g > 0)
                mg = statistics.mean(gains)
                status, _ = _status_for_gain(mg, npos, len(gains))
                aggregation_agg["variants"][cond] = {"R_gain_mean": mg, "R_gain_n_positive": npos, "status": status}
        bl = max(aggregation_agg["variants"], key=lambda c: aggregation_agg["variants"][c]["R_gain_mean"])
        aggregation_agg["best_label"] = bl
        aggregation_agg["best_R_gain_mean"] = aggregation_agg["variants"][bl]["R_gain_mean"]
        aggregation_agg["best_R_gain_n_positive"] = aggregation_agg["variants"][bl]["R_gain_n_positive"]
        aggregation_agg["best_status"] = aggregation_agg["variants"][bl]["status"]
    else:
        aggregation_agg["gate_reason"] = "; ".join(gate_info.get("reasons", [])) or "Aggregation not executed."

    # ---- capacity aggregate ----
    cap_tested = any("cap_mlp_eval" in r for r in per_seed)
    capacity_agg: dict[str, Any] = {"tested": cap_tested}
    if cap_tested:
        gains = _r_gains_seed(per_seed, "cap_mlp")
        npos = sum(1 for g in gains if g > 0)
        mg = statistics.mean(gains) if gains else 0.0
        status, _ = _status_for_gain(mg, npos, len(gains))
        # param/flop delta of mlp update vs linear update (H=24): +600 params
        cap_params = statistics.mean([r["cap_mlp_eval"]["rel_params"] for r in per_seed if "cap_mlp_eval" in r])
        base_rel_params = statistics.mean([r["baseline_eval"]["rel_params"] for r in per_seed if "baseline_eval" in r])
        capacity_agg.update({
            "R_gain_mean": mg, "R_gain_n_positive": npos, "status": status,
            "param_delta": int(cap_params - base_rel_params), "flop_delta": float(24 * 24),
            "rel_params": cap_params,
        })
        # generic-capacity control comparison
        ctrl_gains = _r_gains_seed(per_seed, "cap_control_featwide")
        if ctrl_gains:
            capacity_agg["control_R_gain_mean"] = statistics.mean(ctrl_gains)
    else:
        capacity_agg["gate_reason"] = "; ".join(gate_info.get("reasons", [])) or "Capacity not executed."

    # ---- topology (15E): destruction on baseline vs candidate ----
    topology_agg: dict[str, Any] = {"tested": False}
    if "baseline_eval" in per_seed[0] and eval_key(candidate_cond) in per_seed[0]:
        base_drop = statistics.mean([r["baseline_eval"]["destruction_drop_R"] for r in per_seed if "baseline_eval" in r])
        cand_drop = statistics.mean([r[eval_key(candidate_cond)]["destruction_drop_R"] for r in per_seed if eval_key(candidate_cond) in r])
        cand_gain = statistics.mean([
            r[eval_key(candidate_cond)]["perf"].get("R", 0.0) - r["baseline_eval"]["perf"].get("R", 0.0)
            for r in per_seed if eval_key(candidate_cond) in r and "baseline_eval" in r
        ])
        topology_agg = {
            "tested": True,
            "baseline_drop_R": base_drop,
            "candidate_drop_R": cand_drop,
            "candidate_drop_minus_baseline_drop_R": cand_drop - base_drop,
            "candidate_R_gain_mean": cand_gain,
            "candidate": candidate_cond,
        }

    # ---- compositional (H5), specificity (H6), gap (H7) for candidate ----
    compositional_agg: dict[str, Any] = {"tested": False}
    specificity_agg: dict[str, Any] = {"tested": False}
    gap_agg: dict[str, Any] = {"tested": False}
    if eval_key(candidate_cond) in per_seed[0]:
        rc_g = statistics.mean([r[eval_key(candidate_cond)]["perf"].get("RC", 0.0) - r["baseline_eval"]["perf"].get("RC", 0.0) for r in per_seed])
        frc_g = statistics.mean([r[eval_key(candidate_cond)]["perf"].get("FRC", 0.0) - r["baseline_eval"]["perf"].get("FRC", 0.0) for r in per_seed])
        r_g = statistics.mean([r[eval_key(candidate_cond)]["perf"].get("R", 0.0) - r["baseline_eval"]["perf"].get("R", 0.0) for r in per_seed])
        compositional_agg = {"tested": True, "R_gain_mean": r_g, "RC_gain_mean": rc_g, "FRC_gain_mean": frc_g}
        rel_g = statistics.mean([r_g, rc_g, frc_g])
        non_g = statistics.mean([
            statistics.mean([r[eval_key(candidate_cond)]["perf"].get("F", 0.0) - r["baseline_eval"]["perf"].get("F", 0.0) for r in per_seed]),
            statistics.mean([r[eval_key(candidate_cond)]["perf"].get("C", 0.0) - r["baseline_eval"]["perf"].get("C", 0.0) for r in per_seed]),
        ])
        specificity_agg = {"tested": True, "rel_gain_mean": rel_g, "nonrel_gain_mean": non_g}
        base_gap = statistics.mean([r["baseline_eval"]["probe"].get("gap", 0.0) for r in per_seed])
        cand_gap = statistics.mean([r[eval_key(candidate_cond)]["probe"].get("gap", 0.0) for r in per_seed])
        gap_agg = {
            "tested": True,
            "baseline_gap_mean": base_gap, "candidate_gap_mean": cand_gap, "gap_close_mean": base_gap - cand_gap,
            "baseline_probe_R_mean": statistics.mean([r["baseline_eval"]["probe"].get("probe_R", 0.0) for r in per_seed]),
            "baseline_final_R_mean": statistics.mean([r["baseline_eval"]["probe"].get("final_R", 0.0) for r in per_seed]),
            "candidate_probe_R_mean": statistics.mean([r[eval_key(candidate_cond)]["probe"].get("probe_R", 0.0) for r in per_seed]),
            "candidate_final_R_mean": statistics.mean([r[eval_key(candidate_cond)]["probe"].get("final_R", 0.0) for r in per_seed]),
        }

    # ---- ceiling (15K): old (existing only) vs new (existing + candidate) ----
    ceiling_agg: dict[str, Any] = {"tested": False}
    if "_ceiling_logits" in per_seed[0]:
        new_mixed, old_mixed = [], []
        for r in per_seed:
            eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
            base_logits = r["_ceiling_logits"]
            old = single_expert_ceiling(base_logits, eval_ds.targets, eval_ds.families_list)
            old_mixed.append(statistics.mean([old["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
            # candidate logits: retrain-free forward of the candidate model
            cand_expert = train_variant(candidate_cond, r["seed"], jointco_epochs, batch_size, partials_dir)
            cand_expert.eval()
            with torch.no_grad():
                cand_logits = cand_expert(eval_ds.features)
            new = single_expert_ceiling({**base_logits, "jointco_candidate": cand_logits}, eval_ds.targets, eval_ds.families_list)
            new_mixed.append(statistics.mean([new["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
            r["ceiling_old_mixed"] = old_mixed[-1]
            r["ceiling_new_mixed"] = new_mixed[-1]
        ceiling_agg = {
            "tested": True,
            "old_ceiling_mixed_mean": statistics.mean(old_mixed),
            "new_ceiling_mixed_mean": statistics.mean(new_mixed),
            "new_minus_old_mixed": statistics.mean(new_mixed) - statistics.mean(old_mixed),
        }

    # ---- seed stability ----
    stability: dict[str, dict[str, float]] = {"baseline": {}, "depth": {}, "aggregation": {}, "capacity": {}}
    for sec, conds in (("baseline", ("baseline",)), ("depth", ("depth2", "depth3")), ("aggregation", AGG_CONDS), ("capacity", ("cap_mlp",))):
        for cond in conds:
            vals = [r[eval_key(cond)]["perf"].get("R", 0.0) for r in per_seed if eval_key(cond) in r]
            if len(vals) > 1:
                stability[sec][cond] = statistics.stdev(vals)

    # ---- branch interference for candidate ----
    bi: dict[str, Any] = {"present": False, "observed": "not observed"}
    if eval_key(candidate_cond) in per_seed[0]:
        f_drop = cond_mean("baseline", "F") - cond_mean(candidate_cond, "F")
        c_drop = cond_mean("baseline", "C") - cond_mean(candidate_cond, "C")
        if f_drop >= 0.03 or c_drop >= 0.03:
            bi = {"present": True, "observed": f"F drop {f_drop * 100:+.1f}pp, C drop {c_drop * 100:+.1f}pp"}

    # ---- compute ----
    cand_params_mean = statistics.mean([r[eval_key(candidate_cond)]["params"] for r in per_seed if eval_key(candidate_cond) in r]) if eval_key(candidate_cond) in per_seed[0] else 0
    base_params_mean = statistics.mean([r["baseline_eval"]["params"] for r in per_seed if "baseline_eval" in r])
    compute_agg = {"baseline_params": int(base_params_mean), "candidate_params": int(cand_params_mean), "candidate": candidate_cond}

    # ---- latency ----
    lat_base = statistics.mean([r["baseline_eval"]["latency_us"] for r in per_seed if "baseline_eval" in r])
    lat_cand = statistics.mean([r[eval_key(candidate_cond)]["latency_us"] for r in per_seed if eval_key(candidate_cond) in r]) if eval_key(candidate_cond) in per_seed[0] else 0.0
    latency_agg = {"tested": True, "baseline_total_us": lat_base, "candidate_total_us": lat_cand}

    agg_for_h: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "depth": depth_agg,
        "aggregation": aggregation_agg,
        "capacity": capacity_agg,
        "topology": topology_agg,
        "compositional": compositional_agg,
        "specificity": specificity_agg,
        "gap": gap_agg,
        "ceiling": ceiling_agg,
        "seed_stability": stability,
        "branch_interference": bi,
        "compute": compute_agg,
        "latency": latency_agg,
    }
    hypotheses = build_phase15_hypotheses(agg_for_h)
    agg_for_h["hypotheses"] = hypotheses
    intervention = select_minimal_intervention(agg_for_h)
    if intervention["intervention"] not in ("none", "document_topology"):
        candidate_cond = intervention["intervention"] if intervention["intervention"] in VARIANTS else candidate_cond
    cand_r_gain = float(compositional_agg.get("R_gain_mean", 0.0)) if compositional_agg.get("tested") else 0.0
    verdict_case, verdict_label = select_phase15_case(
        hypotheses, gap_agg.get("baseline_gap_mean", 1.0), cand_r_gain
    )
    failure_rows = build_failure_diagnosis(agg_for_h)

    # ---- baseline vs Phase 14 cross-check ----
    phase14_ref = _load_phase14_reference()
    baseline_mean_perf = {f: cond_mean("baseline", f) for f in list(FAMILIES) + ["pure_mean", "mixed_mean", "overall"]}
    baseline_check: dict[str, Any] = {"phase14_available": phase14_ref is not None, "deltas": {}, "reproduced": True}
    if phase14_ref is not None:
        for f in ("R", "RC", "FRC"):
            d = baseline_mean_perf.get(f, 0.0) - phase14_ref.get(f, 0.0)
            baseline_check["deltas"][f] = d
            if abs(d) > 0.05:
                baseline_check["reproduced"] = False

    summary: dict[str, Any] = {
        "manifest": manifest,
        "gate_info": gate_info,
        "per_seed_results": _serialisable_per_seed(per_seed),
        "baseline_perf_mean": baseline_mean_perf,
        "baseline_check_vs_phase14": baseline_check,
        "depth_results_mean": {c: {f: cond_mean(c, f) for f in list(FAMILIES) + ["overall"]} for c in DEPTH_CONDS if c in tested_conds or c == "baseline"},
        "aggregation_results_mean": {c: {f: cond_mean(c, f) for f in list(FAMILIES) + ["overall"]} for c in AGG_CONDS if c in tested_conds},
        "capacity_results_mean": {c: {f: cond_mean(c, f) for f in list(FAMILIES) + ["overall"]} for c in CAP_CONDS if c in tested_conds},
        "candidate_cond": candidate_cond,
        "hypotheses": hypotheses,
        "verdict_case": verdict_case,
        "verdict_label": verdict_label,
        "minimal_intervention": intervention,
        "failure_diagnosis": failure_rows,
        "aggregates": agg_for_h,
    }

    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_csvs(m_dir, per_seed, tested_conds, failure_rows)
    fig_paths = generate_phase15_figures(summary, f_dir)
    summary["generated_figures"] = fig_paths
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _r_gains_seed(per_seed: list[dict[str, Any]], cond: str) -> list[float]:
    key = f"{cond}_eval"
    return [
        r[key]["perf"].get("R", 0.0) - r["baseline_eval"]["perf"].get("R", 0.0)
        for r in per_seed if key in r and "baseline_eval" in r
    ]


def _status_for_gain(mean_gain: float, n_positive: int, n_seeds: int) -> tuple[str, str]:
    from neuroforge.evaluation.phase15_metrics import GAIN_PARTIAL, GAIN_SUPPORTED
    need = max(2, n_seeds - 1)
    if mean_gain >= GAIN_SUPPORTED and n_positive >= need:
        return ("SUPPORTED", "")
    if mean_gain >= GAIN_PARTIAL and n_positive >= need:
        return ("PARTIALLY SUPPORTED", "")
    if mean_gain >= GAIN_PARTIAL:
        return ("INCONCLUSIVE", "")
    return ("NOT SUPPORTED", "")


def _load_phase14_reference() -> dict[str, float] | None:
    try:
        p = Path("results/metrics/phase14_readout_diagnosis/summary.json")
        if not p.exists():
            return None
        s = json.loads(p.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in s.get("baseline_perf_mean", {}).items() if isinstance(v, (int, float))}
    except Exception:
        return None


def _serialisable_per_seed(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for k, v in r.items():
            if k.startswith("_") or k == "seed":
                continue
            row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV writers (only for executed experiments)
# ---------------------------------------------------------------------------
def _write_csvs(
    m_dir: Path,
    per_seed: list[dict[str, Any]],
    tested_conds: list[str],
    failure_rows: list[dict[str, Any]],
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

    fams = list(FAMILIES)

    rows = []
    for r in per_seed:
        if "baseline_eval" in r:
            for f in fams + ["pure_mean", "mixed_mean", "overall"]:
                rows.append({"seed": r["seed"], "metric": f, "value": r["baseline_eval"]["perf"].get(f, 0.0)})
    write_csv("baseline_reproduction.csv", rows)

    rows = []
    for r in per_seed:
        for cond in DEPTH_CONDS:
            key = f"{cond}_eval"
            if key in r:
                row = {"seed": r["seed"], "condition": cond}
                for f in fams:
                    row[f] = r[key]["perf"].get(f, 0.0)
                row["overall"] = r[key]["perf"].get("overall", 0.0)
                row["probe_R"] = r[key]["probe"].get("probe_R", 0.0)
                row["gap"] = r[key]["probe"].get("gap", 0.0)
                rows.append(row)
    write_csv("depth_ablation.csv", rows)

    if any(f"{c}_eval" in r for r in per_seed for c in AGG_CONDS):
        rows = []
        for r in per_seed:
            for cond in AGG_CONDS:
                key = f"{cond}_eval"
                if key in r:
                    row = {"seed": r["seed"], "condition": cond}
                    for f in fams:
                        row[f] = r[key]["perf"].get(f, 0.0)
                    row["overall"] = r[key]["perf"].get("overall", 0.0)
                    rows.append(row)
        write_csv("aggregation_ablation.csv", rows)

    if any("cap_mlp_eval" in r for r in per_seed):
        rows = []
        for r in per_seed:
            for cond in CAP_CONDS:
                key = f"{cond}_eval"
                if key in r:
                    row = {"seed": r["seed"], "condition": cond}
                    for f in fams:
                        row[f] = r[key]["perf"].get(f, 0.0)
                    row["overall"] = r[key]["perf"].get("overall", 0.0)
                    row["rel_params"] = r[key].get("rel_params", 0)
                    rows.append(row)
        write_csv("capacity_ablation.csv", rows)

    rows = []
    for r in per_seed:
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key not in r:
                continue
            for ctrl, fam_set in (
                ("destruction", ("R", "RC", "FRC")),
                ("marker", ("R", "RC", "FRC")),
                ("token", ("R", "RC", "FRC")),
            ):
                for sub in ("original", "relational_permuted", "marker_neutralised", "token_permuted"):
                    vals = r[key][ctrl].get(sub)
                    if vals is None:
                        continue
                    row = {"seed": r["seed"], "condition": cond, "control": ctrl, "variant": sub}
                    for f in fam_set:
                        row[f] = vals.get(f, 0.0)
                    rows.append(row)
    write_csv("causal_controls.csv", rows)

    rows = []
    for r in per_seed:
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key not in r:
                continue
            for combo, perf in r[key]["branch7"].items():
                row = {"seed": r["seed"], "condition": cond, "combination": combo}
                for f in fams:
                    row[f] = perf.get(f, 0.0)
                rows.append(row)
    write_csv("branch_ablation.csv", rows)

    rows = []
    for r in per_seed:
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key in r:
                rows.append({
                    "seed": r["seed"], "condition": cond,
                    "probe_R": r[key]["probe"].get("probe_R", 0.0),
                    "final_R": r[key]["probe"].get("final_R", 0.0),
                    "gap": r[key]["probe"].get("gap", 0.0),
                })
        if "graph_eval" in r:
            rows.append({
                "seed": r["seed"], "condition": "graph_standalone",
                "probe_R": r["graph_eval"]["probe"].get("probe_R", 0.0),
                "final_R": r["graph_eval"]["probe"].get("final_R", 0.0),
                "gap": r["graph_eval"]["probe"].get("gap", 0.0),
            })
    write_csv("representation_probe.csv", rows)

    if any("graph_eval" in r for r in per_seed):
        rows = []
        for r in per_seed:
            if "graph_eval" not in r:
                continue
            for cond_l, ev in (("graph_standalone", r["graph_eval"]), ("jointco_baseline", r.get("baseline_eval", {}))):
                if not ev:
                    continue
                row = {"seed": r["seed"], "condition": cond_l}
                for f in ("R", "RC", "FRC"):
                    row[f] = ev["perf"].get(f, 0.0)
                row["destruction_drop_R"] = ev.get("destruction_drop_R", 0.0)
                row["probe_R"] = ev["probe"].get("probe_R", 0.0)
                row["params"] = ev.get("params", 0)
                row["latency_us"] = ev.get("latency_us", 0.0)
                rows.append(row)
        write_csv("graph_comparison.csv", rows)

    rows = []
    for r in per_seed:
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key in r:
                rows.append({
                    "seed": r["seed"], "condition": cond,
                    "params": r[key].get("params", 0),
                    "rel_params": r[key].get("rel_params", 0),
                    "activation_bytes": activation_bytes(60, 12, 24, {"baseline": 1, "depth2": 2, "depth3": 3}.get(cond, 1)),
                })
    write_csv("compute.csv", rows)

    rows = []
    for r in per_seed:
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key in r:
                rows.append({
                    "seed": r["seed"], "condition": cond,
                    "latency_us": r[key].get("latency_us", 0.0),
                    "throughput_per_s": r[key].get("throughput_per_s", 0.0),
                })
    write_csv("latency.csv", rows)

    rows = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for cond in tested_conds:
            key = f"{cond}_eval"
            if key in r:
                row[f"{cond}_R"] = r[key]["perf"].get("R", 0.0)
                row[f"{cond}_RC"] = r[key]["perf"].get("RC", 0.0)
                row[f"{cond}_FRC"] = r[key]["perf"].get("FRC", 0.0)
        rows.append(row)
    write_csv("seed_results.csv", rows)

    write_csv("failure_diagnosis.csv", failure_rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_phase15_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Generate the Phase 15 markdown report (all numbers from `summary`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    base = summary.get("baseline_perf_mean", {})
    depth = summary.get("depth_results_mean", {})
    aggr = summary.get("aggregation_results_mean", {})
    cap = summary.get("capacity_results_mean", {})
    hyps = summary.get("hypotheses", {})
    fail = summary.get("failure_diagnosis", [])
    inter = summary.get("minimal_intervention", {})
    gate = summary.get("gate_info", {})
    check = summary.get("baseline_check_vs_phase14", {})

    def fam_row(name: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in FAMILIES)
        return f"| **{name}** | {cells} | {pct(perf.get('mixed_mean', perf.get('overall', 0.0)))} |"

    def hyp_row(h: str) -> str:
        d = hyps.get(h, {})
        return f"| **{h}** | {d.get('status', '?')} | {d.get('evidence', '')} |"

    agg = summary.get("aggregates", {})
    topo = agg.get("topology", {})
    ceil = agg.get("ceiling", {})
    gap = agg.get("gap", {})

    report = f"""# Phase 15 — Relational Substep Capability and Message-Passing Diagnosis

## Mandatory Scientific Disclaimer

> Phase 14 established that readout, pooling, classifier-head capacity, and fusion-path changes did not recover relational performance, and that the relational branch output is not task-sufficient for the evaluated relational prediction. Phase 15 diagnoses WHICH aspect of the relational computation is limiting (propagation → aggregation → update capacity → topology → unresolved). The router remained frozen. All verdicts below are derived programmatically from executed experiments.

---

## 1. Established by Phase 14 (not re-litigated)

- Readout alternatives did not recover relational performance.
- Classifier-head enlargement did not help.
- Fusion-path change did not help.
- Relational branch output was task-weak (rel_only R ≈ 48.6%, worst diagnostic).
- Relational information was decodable (probe ≈ 77.2%) but did not translate into prediction (gap ≈ 15pp).

## 2. Phase 15 research question

> Can a minimal, controlled increase or modification of relational computation convert the existing decodable-but-not-task-usable relational signal into stronger relational prediction?

## 3. Baseline reproduction (§3)

Phase 15 JointCo baseline vs Phase 14 reference:

| Family | Phase 15 baseline | Phase 14 reference | Delta |
|---|---:|---:|---:|
"""
    for f in list(FAMILIES) + ["mixed_mean"]:
        ref = check.get("deltas", {})
        report += f"| **{f}** | {pct(base.get(f, 0.0))} | {pct(base.get(f, 0.0) - ref.get(f, 0.0) if f in ref else 0.0)} | {ref.get(f, float('nan')) * 100:+.1f}pp |\n" if f in ref else f"| **{f}** | {pct(base.get(f, 0.0))} | — | — |\n"
    report += f"""
Baseline reproduced vs Phase 14 (tolerance 5pp on R/RC/FRC): **{check.get('reproduced', '?')}** (Phase 14 data {'available' if check.get('phase14_available') else 'NOT available — internal baseline used for all causal claims'}).

## 4. Exact current relational computation (15A audit)

WHAT: one message-passing round over a fixed ring (self + left + right). Per round: `m = Linear(H→H)(h)`; `msgs = A_norm @ m` with row-stochastic ring adjacency (mean aggregation); `rel = tanh(Linear(2H→H)([h; msgs]))`. No normalisation inside the substep; block-level residual `out = state + scaled_sum + gated_fusion`.

WHY: reuses the validated `GraphBlock`/`RingGraphAdapter` inductive bias (Phase 1-6) inside the Joint pathway.

WHEN: inside the JointCo block after the shared encoder linear, in parallel with the feature and contextual branches, before per-branch scaling, gated fusion and the residual sum.

HOW: neighbour information enters only through the single `A @ W_msg(h)` averaging step, concatenated with self features and compressed by one linear + tanh.

Relational parameter count (H=24): message 600 + update 1176 = **1776** (identical to the standalone Graph specialist's relational core).

## 5. Depth ablation (15B) — WHAT/WHY/FIXED/RESULT

- WHAT changed: `rel_depth` 1 → 2 → 3 (per-round message/update linears).
- WHY: tests insufficient relational propagation.
- WHAT stayed fixed: hidden size, aggregation (mean), topology, feature/context branches, fusion, readout, optimizer, epochs, seeds.
- RESULT:

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for cond in ("baseline", "depth2", "depth3"):
        if cond in depth:
            report += fam_row(cond, depth[cond]) + "\n"
    report += f"""
## 6. Aggregation diagnosis (15C)

Gate decision: aggregation executed = **{agg.get('aggregation', {}).get('tested', False)}**. Reasons: {"; ".join(gate.get("reasons", []))}

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for cond in ("agg_sum", "agg_max"):
        if cond in aggr:
            report += fam_row(cond, aggr[cond]) + "\n"
    report += f"""
## 7. Update-capacity diagnosis (15D)

Capacity executed = **{agg.get('capacity', {}).get('tested', False)}**. Param delta of MLP update vs linear: +{agg.get('capacity', {}).get('param_delta', 0)} relational params.

| Condition | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for cond in ("cap_mlp", "cap_control_featwide"):
        if cond in cap:
            report += fam_row(cond, cap[cond]) + "\n"
    report += f"""
Generic-capacity control (`cap_control_featwide`, ≈+588 feature params) R gain: {agg.get('capacity', {}).get('control_R_gain_mean', float('nan')) * 100:+.1f}pp.

## 8. Topology controls (15E) + marker/position controls (15F)

Destruction drop on R (normal − destroyed): baseline {topo.get('baseline_drop_R', float('nan')) * 100:+.1f}pp vs candidate ({topo.get('candidate', '?')}) {topo.get('candidate_drop_R', float('nan')) * 100:+.1f}pp. Full per-condition controls are in `causal_controls.csv`.

## 9. Branch interaction (15G) + probe reassessment (15H) + graph comparison (15I)

Probe-final gap: baseline {gap.get('baseline_gap_mean', float('nan')) * 100:.1f}pp → candidate {gap.get('candidate_gap_mean', float('nan')) * 100:.1f}pp (closes {gap.get('gap_close_mean', float('nan')) * 100:+.1f}pp). Branch matrix in `branch_ablation.csv`; standalone-Graph reference in `graph_comparison.csv`.

## 10. Portfolio re-evaluation (15K)

Old single-expert ceiling (mixed): {ceil.get('old_ceiling_mixed_mean', float('nan')) * 100:.1f}%. New ceiling with candidate: {ceil.get('new_ceiling_mixed_mean', float('nan')) * 100:.1f}% (delta {ceil.get('new_minus_old_mixed', float('nan')) * 100:+.1f}pp).

## 11. Hypotheses H1–H8 (programmatic verdicts)

| Hypothesis | Status | Evidence |
|---|---|---|
{hyp_row("H1")}
{hyp_row("H2")}
{hyp_row("H3")}
{hyp_row("H4")}
{hyp_row("H5")}
{hyp_row("H6")}
{hyp_row("H7")}
{hyp_row("H8")}

## 12. Minimal validated intervention (15J)

Selected: **{inter.get('intervention', '?')}** — {inter.get('outcome', '')}. {inter.get('detail', '')}

## 13. Final CASE classification (programmatic)

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

## 14. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
"""
    for row in fail:
        report += f"| {row.get('category')} | {row.get('failed')} | {row.get('criterion')} | {row.get('observed')} |\n"
    report += f"""
## 15. Not tested (explicitly deferred)

- Learned adjacency (no evidence yet implicates fixed topology as the sole bottleneck; §9 gate).
- Router retraining / second expert / large GNN redesign (out of scope per §2).
- Depth 4+ (only if depth 2/3 show beneficial propagation; see gate record above).
- Aggregation variants beyond sum/max and capacity variants beyond the MLP update (only if justified by §7/§8 gates).

## 16. Limitations

1. Three-seed sample (11, 23, 37); conclusions use mean ± SD, never best-seed.
2. Aggregation/capacity diagnoses are sequentially gated; ungated sections report NOT TESTED, not failure.
3. Single benchmark construction (Phase 9 mixed); claims are task-specific.
4. Small workloads: FLOPs and latency rankings may diverge (both reported).
"""
    output_path.write_text(report, encoding="utf-8")
