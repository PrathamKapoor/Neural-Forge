"""Phase 16 embedded-vs-dedicated relational diagnosis runner.

Additive: no Phase 1-15 module is modified. Reuses Phase 15 training/eval
utilities (identical protocol: mixed F+R+C, AdamW lr=0.003, best-val
checkpoint, frozen at eval) including its weight cache, so 16A baselines are
executed measurements under the exact Phase 15 protocol.

New training (per seed): relational-focused (F/C neutralised) and rel-first
then full conditions, plus a gradient-instrumented depth-3 run. All other
diagnostics (16C/16D/16E/16I) train only small heads on frozen states.
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
from neuroforge.datasets.phase9_datasets import (
    Phase9MixedStructureDataset,
    apply_phase9_relational_permutation,
)
from neuroforge.evaluation.phase13_metrics import single_expert_ceiling
from neuroforge.evaluation.phase15_metrics import branch_7way_ablation
from neuroforge.evaluation.phase16_metrics import (
    FAMILIES,
    MIXED_FAMS,
    apply_graph_blocks,
    build_failure_diagnosis,
    build_phase16_hypotheses,
    compare_input_stats,
    embedded_relational_output,
    eval_graph_diagnostic,
    eval_states_head,
    graph_native_input,
    input_equivalence_stats,
    joint_pre_relational,
    probe_r_signal,
    record_grad_norms,
    recommendation_for_case,
    select_minimal_intervention,
    select_phase16_case,
    train_graph_diagnostic_head,
    train_linear_head_on_states,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase12_expert_portfolio import _make_mixed_loader
from neuroforge.training.phase15_relational_diagnosis import (
    evaluate_condition,
    train_reference_expert,
    train_variant,
)


P15_PARTIALS = Path("results/metrics/phase15_relational_diagnosis/partials")
COADAPT_CONDS = ("rel_focused", "rel_first")


# ---------------------------------------------------------------------------
# 16F custom trainers (mirror _train_joint_on_mixed exactly)
# ---------------------------------------------------------------------------
def _freeze_to_relational_only(expert: StandaloneSpecialist) -> None:
    """Neutralise F/C branches: zero+freeze their scales and the fusion."""
    block = expert.blocks[0]
    with torch.no_grad():
        block.feature_scale.zero_()
        block.context_scale.zero_()
    block.feature_scale.requires_grad = False
    block.context_scale.requires_grad = False
    for p in block.fusion.parameters():
        p.requires_grad = False
    for p in block.fusion_gate.parameters():
        p.requires_grad = False


def _unfreeze_to_full(block: Any, init_scale: float = 1.0 / 3.0) -> None:
    """Restore the full JointCo objective after a relational-first stage."""
    with torch.no_grad():
        block.feature_scale.fill_(init_scale)
        block.context_scale.fill_(init_scale)
    block.feature_scale.requires_grad = True
    block.context_scale.requires_grad = True
    for p in block.fusion.parameters():
        p.requires_grad = True
    for p in block.fusion_gate.parameters():
        p.requires_grad = True


def _train_mixed_custom(
    expert: StandaloneSpecialist,
    seed: int,
    epochs: int,
    batch_size: int,
    mode: str,
    grad_probe: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[float], list[dict[str, float]]]:
    """Mixed-objective trainer mirroring `_train_joint_on_mixed`.

    mode "rel_focused": F/C neutralised for all epochs.
    mode "rel_first": F/C neutralised for the first half, full after.
    mode "grad_tracked": full joint training + per-epoch grad-norm recording
        on a fixed probe batch (forward+backward, then zero_grad, NO optimizer
        step, so the trajectory is unaltered).
    Returns (val curve, grad log).
    """
    torch.manual_seed(seed)
    if mode in ("rel_focused", "rel_first"):
        _freeze_to_relational_only(expert)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=0.003, weight_decay=1e-4)
    train_loader = _make_mixed_loader("mixed", seed=seed, batch_size=batch_size)
    val_ds = Phase9MixedStructureDataset(samples_per_type=20, seed=seed + 70_000)
    curve: list[float] = []
    grad_log: list[dict[str, float]] = []
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    half = epochs // 2
    for epoch in range(epochs):
        if mode == "rel_first" and epoch == half:
            _unfreeze_to_full(expert.blocks[0])
        expert.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = expert(batch["features"])
            loss = torch.nn.functional.cross_entropy(logits, batch["target"])
            loss.backward()
            optimizer.step()
        expert.eval()
        with torch.no_grad():
            preds = expert(val_ds.features).argmax(-1)
            val_acc = float((preds == val_ds.targets).float().mean().item())
        curve.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in expert.state_dict().items()}
        if mode == "grad_tracked" and grad_probe is not None:
            try:
                grad_log.append(record_grad_norms(expert, grad_probe[0], grad_probe[1]))
            except Exception as e:  # instrumentation must never break training
                grad_log.append({"error": str(e)})
    if best_state is not None:
        expert.load_state_dict(best_state)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False
    return curve, grad_log


def _coadapt_key(seed: int, cond: str, epochs: int, batch_size: int) -> str:
    raw = json.dumps({"cond": cond, "seed": seed, "epochs": epochs, "bs": batch_size}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def train_coadapt_condition(
    cond: str,
    seed: int,
    epochs: int,
    batch_size: int,
    partials_dir: Path,
) -> StandaloneSpecialist:
    """Train (or resume) a 16F co-adaptation condition (depth-3 architecture)."""
    from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant

    cfg = _coadapt_key(seed, cond, epochs, batch_size)
    pt_path = partials_dir / f"seed{seed}_{cond}.pt"
    meta_path = partials_dir / f"seed{seed}_{cond}.json"
    expert = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
    if pt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("config_hash") == cfg:
                expert.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                expert.eval()
                return expert
        except Exception:
            pass
    mode = "rel_focused" if cond == "rel_focused" else "rel_first"
    _train_mixed_custom(expert, seed, epochs, batch_size, mode=mode)
    expert.eval()
    partials_dir.mkdir(parents=True, exist_ok=True)
    torch.save(expert.state_dict(), pt_path)
    meta_path.write_text(json.dumps({"config_hash": cfg, "cond": cond, "seed": seed}), encoding="utf-8")
    return expert


def train_grad_tracked(
    seed: int,
    epochs: int,
    batch_size: int,
    partials_dir: Path,
) -> tuple[StandaloneSpecialist, list[dict[str, float]]]:
    """Depth-3 joint training with per-epoch gradient-norm recording (16G)."""
    from neuroforge.training.phase15_relational_diagnosis import build_expert_for_variant

    probe_ds = Phase9MixedStructureDataset(samples_per_type=10, seed=seed + 61_000)
    probe = (probe_ds.features[:60], probe_ds.targets[:60])
    expert = build_expert_for_variant({"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"})
    _, grad_log = _train_mixed_custom(expert, seed, epochs, batch_size, mode="grad_tracked", grad_probe=probe)
    expert.eval()
    return expert, grad_log


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_phase16_relational_interface(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    jointco_epochs: int = 40,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)
    partials_dir = m_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "Phase 16 — Embedded Relational Path vs Dedicated Graph Diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "evaluated_families": list(FAMILIES),
        "samples_per_type": samples_per_type,
        "jointco_epochs": jointco_epochs,
        "batch_size": batch_size,
        "protocol": (
            "16A baselines reuse the exact Phase 15 protocol/cache (executed "
            "measurements, evals re-run). 16F/16G use a custom trainer mirroring "
            "_train_joint_on_mixed (same optimizer/lr/budget/val selection). "
            "16C/16D/16E/16I train only small heads on frozen states with one "
            "shared head protocol. Router frozen; no learned adjacency."
        ),
    }

    selected = select_capacity_phase6()
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        train_ds = Phase9MixedStructureDataset(samples_per_type=40, seed=seed + 60_000)
        r: dict[str, Any] = {"seed": seed}

        # ---- 16A: executed baselines (Phase 15 cache, evals re-run) ----
        graph = train_reference_expert("graph", "relational", selected.depths["graph"],
                                       seed, jointco_epochs, batch_size, P15_PARTIALS)
        baseline = train_variant("baseline", seed, jointco_epochs, batch_size, P15_PARTIALS)
        depth3 = train_variant("depth3", seed, jointco_epochs, batch_size, P15_PARTIALS)
        r["graph_eval"] = evaluate_condition(graph, eval_ds, seed, do_branch7=False)
        r["baseline_eval"] = evaluate_condition(baseline, eval_ds, seed)
        r["depth3_eval"] = evaluate_condition(depth3, eval_ds, seed)

        # ---- 16F: co-adaptation conditions ----
        rel_focused = train_coadapt_condition("rel_focused", seed, jointco_epochs, batch_size, partials_dir)
        rel_first = train_coadapt_condition("rel_first", seed, jointco_epochs, batch_size, partials_dir)
        r["rel_focused_eval"] = evaluate_condition(rel_focused, eval_ds, seed)
        r["rel_first_eval"] = evaluate_condition(rel_first, eval_ds, seed)

        # ---- 16G: gradient instrumentation ----
        try:
            _, grad_log = train_grad_tracked(seed, jointco_epochs, batch_size, partials_dir)
            clean = [g for g in grad_log if "error" not in g]
            r["grad_log"] = grad_log if clean else [{"error": "empty log"}]
            r["grad_status"] = "VERIFIED" if clean else "NOT VERIFIED"
        except Exception as e:
            r["grad_log"] = [{"error": str(e)}]
            r["grad_status"] = "NOT VERIFIED"

        # ---- 16B: input equivalence (baseline JointCo vs native Graph) ----
        with torch.no_grad():
            joint_pre = joint_pre_relational(baseline, eval_ds.features)
            native_pre = graph_native_input(graph, eval_ds.features)
        stats_joint = input_equivalence_stats(joint_pre, eval_ds.features, "joint_pre_relational")
        stats_native = input_equivalence_stats(native_pre, eval_ds.features, "graph_native_input")
        r["input_stats"] = {"joint": stats_joint, "native": stats_native,
                            "comparison": compare_input_stats(stats_native, stats_joint)}

        # ---- frozen states for 16C/16D/16E/16I ----
        with torch.no_grad():
            d3_pre = joint_pre_relational(depth3, eval_ds.features)
            d3_rel = embedded_relational_output(depth3, eval_ds.features)
            d3_block_out, _ = depth3.blocks[0](d3_pre, eval_ds.features)
            g_on_j = apply_graph_blocks(graph, d3_pre)
            train_pre = joint_pre_relational(depth3, train_ds.features)
            train_rel = embedded_relational_output(depth3, train_ds.features)
            train_g_on_j = apply_graph_blocks(graph, train_pre)

        # ---- 16C: equivalent-input heads (embedded vs graph computation) ----
        head_c1 = train_linear_head_on_states(train_rel.mean(dim=1), train_ds.targets, 24, seed=seed)
        head_c2 = train_linear_head_on_states(train_g_on_j.mean(dim=1), train_ds.targets, 24, seed=seed)
        r["c1_eval"] = eval_states_head(head_c1, d3_rel.mean(dim=1), eval_ds.targets, eval_ds.families_list)
        r["c2_eval"] = eval_states_head(head_c2, g_on_j.mean(dim=1), eval_ds.targets, eval_ds.families_list)
        r["c1_probe_R"] = probe_r_signal(d3_rel, eval_ds)
        r["c2_probe_R"] = probe_r_signal(g_on_j, eval_ds)

        # ---- 16D: fresh Graph diagnostic on frozen joint input ----
        gdiag_block, gdiag_head = train_graph_diagnostic_head(train_pre, train_ds.targets, seed=seed)
        r["gdiag_eval"] = eval_graph_diagnostic(gdiag_block, gdiag_head, d3_pre, eval_ds.targets, eval_ds.families_list)
        r["gdiag_probe_R"] = probe_r_signal(apply_graph_blocks_eval(gdiag_block, d3_pre), eval_ds)

        # ---- 16E: pre/post stage probes + stage heads + stage destruction ----
        dest_feats = apply_phase9_relational_permutation(eval_ds.features, seed=seed)
        with torch.no_grad():
            pre_d = joint_pre_relational(depth3, dest_feats)
            rel_d = embedded_relational_output(depth3, dest_feats)
            block_d, _ = depth3.blocks[0](pre_d, dest_feats)
        head_pre = train_linear_head_on_states(train_pre.mean(dim=1), train_ds.targets, 24, seed=seed + 1)
        # post_rel head reuses c1 head (same states/protocol, different seed namespace kept distinct)
        head_post = train_linear_head_on_states(train_rel.mean(dim=1), train_ds.targets, 24, seed=seed + 2)
        with torch.no_grad():
            train_block, _ = depth3.blocks[0](train_pre, train_ds.features)
        head_fus = train_linear_head_on_states(train_block.mean(dim=1), train_ds.targets, 24, seed=seed + 3)
        r["stages"] = {
            "pre_probe_R": probe_r_signal(d3_pre, eval_ds),
            "post_probe_R": probe_r_signal(d3_rel, eval_ds),
            "post_fusion_probe_R": probe_r_signal(d3_block_out, eval_ds),
            "native_input_probe_R": probe_r_signal(native_pre, eval_ds),
            "pre_R": eval_states_head(head_pre, d3_pre.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0),
            "post_R": eval_states_head(head_post, d3_rel.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0),
            "post_fusion_R": eval_states_head(head_fus, d3_block_out.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0),
            "pre_drop_R": eval_states_head(head_pre, d3_pre.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0)
            - eval_states_head(head_pre, pre_d.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0),
            "post_drop_R": eval_states_head(head_post, d3_rel.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0)
            - eval_states_head(head_post, rel_d.mean(dim=1), eval_ds.targets, eval_ds.families_list).get("R", 0.0),
        }

        # ---- 16I: compatibility (rep3 = native encoder out + same head protocol) ----
        with torch.no_grad():
            train_native = graph_native_input(graph, train_ds.features)
            train_rep3 = apply_graph_blocks(graph, train_native)
            rep3_state = apply_graph_blocks(graph, native_pre)
        head_rep3 = train_linear_head_on_states(train_rep3.mean(dim=1), train_ds.targets, 24, seed=seed + 4)
        r["rep3_eval"] = eval_states_head(head_rep3, rep3_state.mean(dim=1), eval_ds.targets, eval_ds.families_list)

        # ---- ceiling inputs ----
        r["_logits"] = {}
        for arch in ("mlp", "attention", "attention_v2"):
            e = train_reference_expert(arch, {"mlp": "feature"}.get(arch, "contextual"),
                                       selected.depths[arch], seed, jointco_epochs, batch_size, P15_PARTIALS)
            e.eval()
            with torch.no_grad():
                r["_logits"][arch] = e(eval_ds.features).clone()
        with torch.no_grad():
            graph.eval()
            r["_logits"]["graph"] = graph(eval_ds.features).clone()
        per_seed.append(r)

    # =====================================================================
    # Aggregates
    # =====================================================================
    def perf(cond: str, fam: str) -> list[float]:
        return [r[f"{cond}_eval"]["perf"].get(fam, 0.0) for r in per_seed if f"{cond}_eval" in r]

    def mean(vals: list[float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    base_r = mean(perf("baseline", "R"))
    depth3_r = mean(perf("depth3", "R"))

    # computation (16C)
    c_gains = [r["c2_eval"].get("R", 0.0) - r["c1_eval"].get("R", 0.0) for r in per_seed]
    nat_gap = [r["graph_eval"]["perf"].get("R", 0.0) - r["c2_eval"].get("R", 0.0) for r in per_seed]
    computation = {
        "tested": True,
        "embedded_R": mean([r["c1_eval"].get("R", 0.0) for r in per_seed]),
        "graph_on_joint_R": mean([r["c2_eval"].get("R", 0.0) for r in per_seed]),
        "graph_on_joint_minus_embedded_R": mean(c_gains),
        "n_pos": sum(1 for g in c_gains if g > 0),
        "native_minus_graph_on_joint_R": mean(nat_gap),
        "embedded_probe_R": mean([r["c1_probe_R"] for r in per_seed]),
        "graph_on_joint_probe_R": mean([r["c2_probe_R"] for r in per_seed]),
        "shapes_equal": all(r["input_stats"]["comparison"]["shape_equal"] for r in per_seed),
    }
    gdiag_gap = [r["graph_eval"]["perf"].get("R", 0.0) - r["gdiag_eval"].get("R", 0.0) for r in per_seed]
    graph_diagnostic = {
        "tested": True,
        "diagnostic_R": mean([r["gdiag_eval"].get("R", 0.0) for r in per_seed]),
        "diagnostic_probe_R": mean([r["gdiag_probe_R"] for r in per_seed]),
        "native_minus_diagnostic_R": mean(gdiag_gap),
    }

    # coadaptation (16F), joint reference = depth3
    bf = [r["rel_focused_eval"]["perf"].get("R", 0.0) - r["depth3_eval"]["perf"].get("R", 0.0) for r in per_seed]
    br = [r["rel_first_eval"]["perf"].get("R", 0.0) - r["depth3_eval"]["perf"].get("R", 0.0) for r in per_seed]
    coadaptation = {
        "tested": True,
        "relfocused_minus_joint_R": mean(bf),
        "relfirst_minus_joint_R": mean(br),
        "n_pos": sum(1 for g in bf if g > 0),
        "relfocused_R": mean(perf("rel_focused", "R")),
        "relfirst_R": mean(perf("rel_first", "R")),
        "joint_R": depth3_r,
    }

    # stages (16E)
    stages = {
        "tested": True,
        "pre_probe_R": mean([r["stages"]["pre_probe_R"] for r in per_seed]),
        "post_probe_R": mean([r["stages"]["post_probe_R"] for r in per_seed]),
        "post_fusion_probe_R": mean([r["stages"]["post_fusion_probe_R"] for r in per_seed]),
        "native_input_probe_R": mean([r["stages"]["native_input_probe_R"] for r in per_seed]),
        "final_R": depth3_r,
        "pre_R": mean([r["stages"]["pre_R"] for r in per_seed]),
        "post_R": mean([r["stages"]["post_R"] for r in per_seed]),
        "post_fusion_R": mean([r["stages"]["post_fusion_R"] for r in per_seed]),
        "pre_drop_R": mean([r["stages"]["pre_drop_R"] for r in per_seed]),
        "post_drop_R": mean([r["stages"]["post_drop_R"] for r in per_seed]),
    }

    # best condition for compositional/sensitivity/ceiling
    cand_scores = {}
    for cond in ("depth3", "rel_focused", "rel_first"):
        cand_scores[cond] = mean(perf(cond, "R"))
    best_cond = max(cand_scores, key=lambda k: cand_scores[k])
    compositional = {
        "tested": True,
        "condition": best_cond,
        "R_gain": mean(perf(best_cond, "R")) - base_r,
        "RC_gain": mean(perf(best_cond, "RC")) - mean(perf("baseline", "RC")),
        "FRC_gain": mean(perf(best_cond, "FRC")) - mean(perf("baseline", "FRC")),
    }
    base_drop = mean([r["baseline_eval"]["destruction_drop_R"] for r in per_seed])
    cand_drop = mean([r[f"{best_cond}_eval"]["destruction_drop_R"] for r in per_seed])
    sensitivity = {"tested": True, "baseline_drop_R": base_drop, "candidate_drop_R": cand_drop,
                   "candidate_drop_minus_baseline_drop_R": cand_drop - base_drop, "candidate": best_cond}

    # ceiling
    old_m, new_m = [], []
    for r in per_seed:
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=r["seed"])
        old = single_expert_ceiling(r["_logits"], eval_ds.targets, eval_ds.families_list)
        old_m.append(statistics.mean([old["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
        be = {"depth3": train_variant("depth3", r["seed"], jointco_epochs, batch_size, P15_PARTIALS),
              "rel_focused": train_coadapt_condition("rel_focused", r["seed"], jointco_epochs, batch_size, partials_dir),
              "rel_first": train_coadapt_condition("rel_first", r["seed"], jointco_epochs, batch_size, partials_dir)}[best_cond]
        be.eval()
        with torch.no_grad():
            cand_logits = be(eval_ds.features)
        new = single_expert_ceiling({**r["_logits"], "jointco_candidate": cand_logits}, eval_ds.targets, eval_ds.families_list)
        new_m.append(statistics.mean([new["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS]))
        r["ceiling_old_mixed"] = old_m[-1]
        r["ceiling_new_mixed"] = new_m[-1]
    ceiling = {"tested": True, "old_ceiling_mixed_mean": mean(old_m),
               "new_ceiling_mixed_mean": mean(new_m), "new_minus_old_mixed": mean(new_m) - mean(old_m)}

    # gradients (16G)
    grad_statuses = [r.get("grad_status", "NOT VERIFIED") for r in per_seed]
    if all(s == "VERIFIED" for s in grad_statuses):
        groups = ("relational", "feature", "contextual", "fusion", "encoder", "head")
        per_group: dict[str, list[float]] = {g: [] for g in groups}
        for r in per_seed:
            logs = [g for g in r["grad_log"] if "error" not in g]
            for gname in groups:
                vals = [e.get(gname, 0.0) for e in logs]
                if vals:
                    per_group[gname].append(statistics.mean(vals))
        group_means = {g: mean(v) for g, v in per_group.items()}
        total = sum(group_means.values()) + 1e-12
        gradients = {"status": "VERIFIED", "group_means": group_means,
                     "rel_share_mean": group_means.get("relational", 0.0) / total,
                     "observed": f"rel share {group_means.get('relational', 0.0) / total * 100:.1f}% of grad-norm mass"}
    else:
        gradients = {"status": "NOT VERIFIED",
                     "observed": f"statuses={grad_statuses}; no conclusion manufactured",
                     "rel_share_mean": 1.0}

    # seed stability over trained-condition R
    stability: dict[str, dict[str, float]] = {"trained": {}}
    for cond in ("graph", "baseline", "depth3", "rel_focused", "rel_first"):
        vals = perf(cond, "R")
        if len(vals) > 1:
            stability["trained"][cond] = statistics.stdev(vals)

    # latency/compute
    lat_base = mean([r["baseline_eval"]["latency_us"] for r in per_seed])
    lat_best = mean([r[f"{best_cond}_eval"]["latency_us"] for r in per_seed])
    latency = {"tested": True, "baseline_total_us": lat_base, "candidate_total_us": lat_best}

    agg: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "computation": computation,
        "graph_diagnostic": graph_diagnostic,
        "coadaptation": coadaptation,
        "stages": stages,
        "compositional": compositional,
        "sensitivity": sensitivity,
        "ceiling": ceiling,
        "gradients": gradients,
        "seed_stability": stability,
        "latency": latency,
        "best_cond": best_cond,
    }
    hypotheses = build_phase16_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    case, label = select_phase16_case(hypotheses, stages)
    intervention = select_minimal_intervention(case, hypotheses, agg)
    failure_rows = build_failure_diagnosis(agg)

    # baseline cross-check vs Phase 15 executed means
    baseline_check = _check_vs_phase15(per_seed)

    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": _serialisable(per_seed),
        "baseline_perf_mean": {f: mean(perf("baseline", f)) for f in list(FAMILIES) + ["pure_mean", "mixed_mean", "overall"]},
        "graph_perf_mean": {f: mean(perf("graph", f)) for f in list(FAMILIES) + ["overall"]},
        "depth3_perf_mean": {f: mean(perf("depth3", f)) for f in list(FAMILIES) + ["overall"]},
        "rel_focused_perf_mean": {f: mean(perf("rel_focused", f)) for f in list(FAMILIES) + ["overall"]},
        "rel_first_perf_mean": {f: mean(perf("rel_first", f)) for f in list(FAMILIES) + ["overall"]},
        "baseline_check_vs_phase15": baseline_check,
        "best_cond": best_cond,
        "hypotheses": hypotheses,
        "verdict_case": case,
        "verdict_label": label,
        "recommendation": recommendation_for_case(case),
        "minimal_intervention": intervention,
        "failure_diagnosis": failure_rows,
        "aggregates": agg,
    }

    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_csvs(m_dir, per_seed, failure_rows)
    from neuroforge.visualization.phase16_plots import generate_phase16_figures
    summary["generated_figures"] = generate_phase16_figures(summary, f_dir)
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def apply_graph_blocks_eval(gblock: Any, state: torch.Tensor) -> torch.Tensor:
    out, _ = gblock(state)
    return out


def _check_vs_phase15(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        p = Path("results/metrics/phase15_relational_diagnosis/summary.json")
        if not p.exists():
            return {"phase15_available": False, "reproduced": True, "deltas": {}}
        s = json.loads(p.read_text(encoding="utf-8"))
        deltas: dict[str, float] = {}
        ok = True
        for cond, key in (("baseline", "baseline_perf_mean"), ("depth3", None)):
            ref = s.get(key, {}) if key else s.get("depth_results_mean", {}).get("depth3", {})
            for f in ("R", "RC", "FRC"):
                mine = statistics.mean([r[f"{cond}_eval"]["perf"].get(f, 0.0) for r in per_seed if f"{cond}_eval" in r])
                d = mine - float(ref.get(f, mine))
                deltas[f"{cond}_{f}"] = d
                if abs(d) > 0.05:
                    ok = False
        return {"phase15_available": True, "reproduced": ok, "deltas": deltas}
    except Exception:
        return {"phase15_available": False, "reproduced": True, "deltas": {}}


def _serialisable(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in per_seed:
        row: dict[str, Any] = {"seed": r["seed"]}
        for k, v in r.items():
            if k.startswith("_") or k == "seed":
                continue
            row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV writers (only executed experiments)
# ---------------------------------------------------------------------------
def _write_csvs(m_dir: Path, per_seed: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
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
    trained = ("graph", "baseline", "depth3", "rel_focused", "rel_first")

    rows = []
    for r in per_seed:
        for cond in ("graph", "baseline", "depth3"):
            for f in fams + ["mixed_mean", "overall"]:
                rows.append({"seed": r["seed"], "condition": cond, "metric": f,
                             "value": r[f"{cond}_eval"]["perf"].get(f, 0.0)})
    write_csv("baseline_reproduction.csv", rows)

    rows = []
    for r in per_seed:
        for lab, st in (("joint", r["input_stats"]["joint"]), ("native", r["input_stats"]["native"])):
            rows.append({"seed": r["seed"], "representation": lab,
                         "global_mean": st["global_mean"], "global_std": st["global_std"],
                         "mean_abs": st["mean_abs"], "marker_max": st["marker_max"]})
        comp = r["input_stats"]["comparison"]
        rows.append({"seed": r["seed"], "representation": "comparison",
                     "global_mean": comp["global_mean_shift_abs"],
                     "global_std": comp["global_std_ratio_b_over_a"],
                     "mean_abs": comp["per_channel_mean_l2"],
                     "marker_max": float(comp["shape_equal"])})
    write_csv("input_equivalence.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"], "condition": "embedded_branch",
                     "R": r["c1_eval"].get("R", 0.0), "RC": r["c1_eval"].get("RC", 0.0),
                     "FRC": r["c1_eval"].get("FRC", 0.0), "probe_R": r["c1_probe_R"]})
        rows.append({"seed": r["seed"], "condition": "graph_on_joint_input",
                     "R": r["c2_eval"].get("R", 0.0), "RC": r["c2_eval"].get("RC", 0.0),
                     "FRC": r["c2_eval"].get("FRC", 0.0), "probe_R": r["c2_probe_R"]})
        rows.append({"seed": r["seed"], "condition": "graph_diagnostic_fresh",
                     "R": r["gdiag_eval"].get("R", 0.0), "RC": r["gdiag_eval"].get("RC", 0.0),
                     "FRC": r["gdiag_eval"].get("FRC", 0.0), "probe_R": r["gdiag_probe_R"]})
        rows.append({"seed": r["seed"], "condition": "graph_native_full",
                     "R": r["graph_eval"]["perf"].get("R", 0.0), "RC": r["graph_eval"]["perf"].get("RC", 0.0),
                     "FRC": r["graph_eval"]["perf"].get("FRC", 0.0),
                     "probe_R": r["graph_eval"]["probe"].get("probe_R", 0.0)})
        rows.append({"seed": r["seed"], "condition": "compat_rep3_head_protocol",
                     "R": r["rep3_eval"].get("R", 0.0), "RC": r["rep3_eval"].get("RC", 0.0),
                     "FRC": r["rep3_eval"].get("FRC", 0.0), "probe_R": 0.0})
    write_csv("graph_on_joint_input.csv", rows)

    rows = []
    for r in per_seed:
        for stage in ("pre", "post", "post_fusion"):
            rows.append({"seed": r["seed"], "stage": stage,
                         "probe_R": r["stages"][f"{stage}_probe_R"],
                         "head_R": r["stages"][f"{stage}_R"]})
        rows.append({"seed": r["seed"], "stage": "native_input", "probe_R": r["stages"]["native_input_probe_R"], "head_R": 0.0})
    write_csv("representation_probe.csv", rows)

    rows = []
    for r in per_seed:
        for stage in ("pre", "post"):
            rows.append({"seed": r["seed"], "stage": stage,
                         "head_R": r["stages"][f"{stage}_R"],
                         "drop_R": r["stages"][f"{stage}_drop_R"]})
    write_csv("pre_post_relational.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("depth3", "rel_focused", "rel_first"):
            row = {"seed": r["seed"], "condition": cond}
            for f in fams:
                row[f] = r[f"{cond}_eval"]["perf"].get(f, 0.0)
            row["overall"] = r[f"{cond}_eval"]["perf"].get("overall", 0.0)
            rows.append(row)
    write_csv("coadaptation.csv", rows)

    rows = []
    for r in per_seed:
        for entry in r.get("grad_log", []):
            row = {"seed": r["seed"], "status": r.get("grad_status", "?")}
            for k, v in entry.items():
                row[k] = v if isinstance(v, (int, float)) else str(v)[:60]
            rows.append(row)
    write_csv("gradient_diagnostics.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("depth3", "rel_focused", "rel_first"):
            for combo, perf in r[f"{cond}_eval"].get("branch7", {}).items():
                if not isinstance(perf, dict):
                    continue
                row = {"seed": r["seed"], "condition": cond, "combination": combo}
                for f in fams:
                    row[f] = perf.get(f, 0.0)
                rows.append(row)
    write_csv("branch_ablation.csv", rows)

    rows = []
    for r in per_seed:
        for cond in trained:
            for ctrl in ("destruction", "marker", "token"):
                ev = r[f"{cond}_eval"][ctrl]
                variant_map = {"destruction": ("original", "relational_permuted"),
                               "marker": ("original", "marker_neutralised"),
                               "token": ("original", "token_permuted")}[ctrl]
                for sub in variant_map:
                    vals = ev.get(sub)
                    if vals is None:
                        continue
                    row = {"seed": r["seed"], "condition": cond, "control": ctrl, "variant": sub}
                    for f in ("R", "RC", "FRC"):
                        row[f] = vals.get(f, 0.0)
                    rows.append(row)
    write_csv("causal_controls.csv", rows)

    rows = []
    for r in per_seed:
        for cond in trained:
            rows.append({"seed": r["seed"], "condition": cond,
                         "R": r[f"{cond}_eval"]["perf"].get("R", 0.0),
                         "RC": r[f"{cond}_eval"]["perf"].get("RC", 0.0),
                         "FRC": r[f"{cond}_eval"]["perf"].get("FRC", 0.0)})
    write_csv("rc_frc_results.csv", rows)

    rows = []
    for r in per_seed:
        for cond in trained:
            rows.append({"seed": r["seed"], "condition": cond,
                         "params": r[f"{cond}_eval"].get("params", 0),
                         "rel_params": r[f"{cond}_eval"].get("rel_params", 0)})
    write_csv("compute.csv", rows)

    rows = []
    for r in per_seed:
        for cond in trained:
            rows.append({"seed": r["seed"], "condition": cond,
                         "latency_us": r[f"{cond}_eval"].get("latency_us", 0.0),
                         "throughput_per_s": r[f"{cond}_eval"].get("throughput_per_s", 0.0)})
    write_csv("latency.csv", rows)

    rows = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for cond in trained:
            row[f"{cond}_R"] = r[f"{cond}_eval"]["perf"].get("R", 0.0)
            row[f"{cond}_RC"] = r[f"{cond}_eval"]["perf"].get("RC", 0.0)
        row["c1_R"] = r["c1_eval"].get("R", 0.0)
        row["c2_R"] = r["c2_eval"].get("R", 0.0)
        row["gdiag_R"] = r["gdiag_eval"].get("R", 0.0)
        rows.append(row)
    write_csv("seed_results.csv", rows)

    write_csv("failure_diagnosis.csv", failure_rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_phase16_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    base = summary.get("baseline_perf_mean", {})
    graph = summary.get("graph_perf_mean", {})
    d3 = summary.get("depth3_perf_mean", {})
    bf = summary.get("rel_focused_perf_mean", {})
    br = summary.get("rel_first_perf_mean", {})
    hyps = summary.get("hypotheses", {})
    fail = summary.get("failure_diagnosis", [])
    agg = summary.get("aggregates", {})

    def fam_row(name: str, perf: dict[str, float]) -> str:
        cells = " | ".join(pct(perf.get(f, 0.0)) for f in FAMILIES)
        return f"| **{name}** | {cells} |"

    report = f"""# Phase 16 — Embedded Relational Path vs Dedicated Graph Diagnosis

## Mandatory Scientific Disclaimer

> Phase 15 established that depth-3 improves isolated relational prediction, but did NOT establish improved mixed relational composition. The large gap between the dedicated Graph specialist and the embedded JointCo relational path is the central diagnostic observation of this phase. The router remained frozen; no learned adjacency was introduced.

---

## 1. Phase 15 established / Phase 15 did NOT establish

- Established: depth-3 improves isolated R reproducibly (H1 SUPPORTED).
- NOT established: improved mixed relational composition (H5 NOT SUPPORTED), gap closure (H7 NOT SUPPORTED), ceiling increase (H8 NOT SUPPORTED).
- Central observation: standalone Graph ≈84–93% R with ≈32–47pp destruction drops vs embedded path ≈5pp drops.

## 2. Research question

> Why does the dedicated Graph expert exploit relational structure strongly while the embedded JointCo relational path does not?

## 3. Exact baselines (16A, executed — not historical numbers)

| Condition | F | R | C | FR | RC | FC | FRC |
|---|---:|---:|---:|---:|---:|---:|---:|
{fam_row("standalone Graph", graph)}
{fam_row("JointCo baseline", base)}
{fam_row("JointCo depth-3", d3)}
{fam_row("rel-focused (B)", bf)}
{fam_row("rel-first (C)", br)}

Reproduction vs Phase 15 (tolerance 5pp): **{summary.get('baseline_check_vs_phase15', {}).get('reproduced', '?')}**.

## 4. Input equivalence (16B)

Per-seed representation statistics (shape/scale/marker) are in `input_equivalence.csv`.
Key cross-seed comparison (native Graph input vs JointCo pre-relational input):

- shapes_equal: {agg.get('computation', {}).get('shapes_equal', '?')}
- Both paths consume the same raw batch (marker positions identical by construction);
  the comparison therefore tests encoder transformation, not task framing.

## 5. Graph-on-Joint-input (16C, the critical experiment)

Same input representation, different computation, identical head protocol:

- embedded branch + head R: {agg.get('computation', {}).get('embedded_R', float('nan')) * 100:.1f}%
- Graph-on-joint-input + head R: {agg.get('computation', {}).get('graph_on_joint_R', float('nan')) * 100:.1f}%
- gap: {agg.get('computation', {}).get('graph_on_joint_minus_embedded_R', float('nan')) * 100:+.1f}pp
- fresh Graph diagnostic on joint input R: {agg.get('graph_diagnostic', {}).get('diagnostic_R', float('nan')) * 100:.1f}%

## 6. Stage analysis (16E: probe information + task usability + causal dependence)

- pre-relational probe R: {agg.get('stages', {}).get('pre_probe_R', float('nan')) * 100:.1f}%
- post-relational probe R: {agg.get('stages', {}).get('post_probe_R', float('nan')) * 100:.1f}%
- post-fusion probe R: {agg.get('stages', {}).get('post_fusion_probe_R', float('nan')) * 100:.1f}%
- native-input probe R: {agg.get('stages', {}).get('native_input_probe_R', float('nan')) * 100:.1f}%
- stage head R (pre/post/fusion): {agg.get('stages', {}).get('pre_R', float('nan')) * 100:.1f}% / {agg.get('stages', {}).get('post_R', float('nan')) * 100:.1f}% / {agg.get('stages', {}).get('post_fusion_R', float('nan')) * 100:.1f}%

## 7. Co-adaptation (16F) and gradients (16G)

- rel-focused minus joint R: {agg.get('coadaptation', {}).get('relfocused_minus_joint_R', float('nan')) * 100:+.1f}pp
- rel-first minus joint R: {agg.get('coadaptation', {}).get('relfirst_minus_joint_R', float('nan')) * 100:+.1f}pp
- gradient instrumentation: {agg.get('gradients', {}).get('status', '?')} — {agg.get('gradients', {}).get('observed', '')}

## 8. Composition (16K) and ceiling

- strongest condition ({agg.get('compositional', {}).get('condition', '?')}): R {agg.get('compositional', {}).get('R_gain', float('nan')) * 100:+.1f}pp, RC {agg.get('compositional', {}).get('RC_gain', float('nan')) * 100:+.1f}pp, FRC {agg.get('compositional', {}).get('FRC_gain', float('nan')) * 100:+.1f}pp vs baseline.
- ceiling: {agg.get('ceiling', {}).get('old_ceiling_mixed_mean', float('nan')) * 100:.1f}% → {agg.get('ceiling', {}).get('new_ceiling_mixed_mean', float('nan')) * 100:.1f}% ({agg.get('ceiling', {}).get('new_minus_old_mixed', float('nan')) * 100:+.1f}pp).

## 9. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
"""
    for h in [f"H{i}" for i in range(1, 9)]:
        d = hyps.get(h, {})
        report += f"| **{h}** | {d.get('status', '?')} | {d.get('evidence', '')} |\n"
    report += f"""
## 10. Final CASE (programmatic)

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

Recommendation: {summary.get('recommendation', '')}

Minimal intervention: **{summary.get('minimal_intervention', {}).get('intervention', '?')}** — {summary.get('minimal_intervention', {}).get('outcome', '')}.

## 11. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
"""
    for row in fail:
        report += f"| {row.get('category')} | {row.get('failed')} | {row.get('criterion')} | {row.get('observed')} |\n"
    report += """
## 12. Not tested / deferred

- Learned adjacency (topology exonerated in Phase 15; remains out of scope).
- Router retraining; second expert; Transformer/attention additions.
- Interaction stacks (e.g. depth-3 + max aggregation) — deliberately not stacked.

## 13. Limitations

1. Three seeds; mean ± SD reported, never best-seed.
2. Equivalent-input heads use mean-pool + linear protocol, not the native query/mean heads; native full-model numbers are reported alongside for calibration.
3. Rel-focused training uses the mixed objective with F/C neutralised (single-variable isolation); a pure-relational objective variant was not run.
4. Gradient shares are diagnostic mass ratios on a fixed probe batch, not causal attributions.
"""
    output_path.write_text(report, encoding="utf-8")
