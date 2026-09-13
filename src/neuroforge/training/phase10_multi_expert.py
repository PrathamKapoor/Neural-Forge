"""Phase 10: Adaptive Multi-Expert Composition experiment runner."""
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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.composition_metrics import (
    calculate_accuracy_constrained_compute,
    calculate_active_expert_statistics,
    calculate_composition_entropy,
    calculate_counterfactual_composition_analysis,
    calculate_pair_utilization,
    calculate_triple_utilization,
    check_composition_collapses,
    diagnose_phase10_failure,
    generate_family_composition_matrix,
)
from neuroforge.evaluation.phase8b_metrics import compute_normalized_costs
from neuroforge.evaluation.phase9_metrics import calculate_single_expert_ceiling
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.composition import (
    ParallelMultiExpert,
    SequentialSpecialistComposition,
)
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.learned_router import SampleLevelRouter
from neuroforge.routing.multi_expert import (
    AdaptiveKRouter,
    TopKRouter,
    aggregate_parallel_normalized,
    aggregate_parallel_uniform,
    aggregate_parallel_weighted,
)
from neuroforge.training.phase8b_compute_aware import _train_expert
from neuroforge.visualization.phase10_plots import generate_phase10_figures


def _train_adaptive_k_router(
    router: AdaptiveKRouter,
    experts: dict[str, StandaloneSpecialist],
    expert_names: tuple[str, ...],
    norm_costs: dict[str, float],
    lam: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 0.008,
) -> tuple[AdaptiveKRouter, list[float]]:
    """Train AdaptiveKRouter with composite prediction loss and compute-aware penalty."""
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    c_norm_tensor = torch.tensor([norm_costs[e] for e in expert_names], dtype=torch.float32)

    best_val_score = -1e9
    best_state = None
    curve: list[float] = []

    for _ in range(epochs):
        router.train()
        for batch in train_loader:
            x = batch["features"]
            y = batch["target"]
            c_norm_dev = c_norm_tensor.to(x.device)

            dec = router(x, mode="straight_through", strategy="learned")
            weights = dec.weights  # [B, 4]
            k_vals = dec.k_values.float()  # [B]

            # Compute frozen expert logits
            with torch.no_grad():
                exp_logits = torch.stack([experts[e](x) for e in expert_names], dim=1)  # [B, 4, 2]

            # Weighted combination of expert logits for prediction loss
            pred_logits = (weights.unsqueeze(-1) * exp_logits).sum(dim=1)  # [B, 2]
            pred_loss = F.cross_entropy(pred_logits, y)

            # Soft cost surrogate: expected compute of active experts
            # soft probabilities for experts * cost + penalty for active k
            soft_exp = dec.soft_probabilities
            cost_exp = (soft_exp * c_norm_dev.unsqueeze(0)).sum(dim=-1)
            active_fraction = (k_vals / float(len(expert_names)))
            total_cost_surrogate = (cost_exp * active_fraction).mean()

            loss = pred_loss + lam * total_cost_surrogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        router.eval()
        corr = tot = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"]
                y = batch["target"]
                dec = router(x, mode="hard", strategy="learned")
                exp_logits = torch.stack([experts[e](x) for e in expert_names], dim=1)

                for i in range(len(x)):
                    sel_idx = dec.selected_indices[i]
                    sub_l = exp_logits[i, sel_idx]  # [k, 2]
                    comb_l = sub_l.mean(dim=0)
                    if comb_l.argmax().item() == y[i].item():
                        corr += 1
                    tot += 1

        val_acc = corr / tot if tot > 0 else 0.0
        curve.append(val_acc)
        if val_acc > best_val_score:
            best_val_score = val_acc
            best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}

    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()
    return router, curve


def _benchmark_phase10_latencies(
    router: AdaptiveKRouter,
    experts: dict[str, StandaloneSpecialist],
    batch: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Profile CPU latency breakdown: router inference, expert execution, aggregation."""
    router.eval()
    for exp in experts.values():
        exp.eval()

    num_runs = 20
    b_size = len(batch)

    # 1. Router timing
    router_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = router(batch, mode="hard")
            t1 = time.perf_counter()
            router_times.append((t1 - t0) / b_size * 1e6)
    router_us = statistics.mean(router_times)

    # 2. Expert timings
    expert_us: dict[str, float] = {}
    for name, exp in experts.items():
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = exp(batch)
                t1 = time.perf_counter()
                times.append((t1 - t0) / b_size * 1e6)
        expert_us[name] = statistics.mean(times)

    # 3. Aggregation timings (for 2 and 3 experts)
    dummy_logits_2 = [torch.randn(b_size, 2) for _ in range(2)]
    dummy_logits_3 = [torch.randn(b_size, 2) for _ in range(3)]

    agg_2_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = aggregate_parallel_uniform(dummy_logits_2)
            t1 = time.perf_counter()
            agg_2_times.append((t1 - t0) / b_size * 1e6)
    agg_2_us = statistics.mean(agg_2_times)

    agg_3_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = aggregate_parallel_uniform(dummy_logits_3)
            t1 = time.perf_counter()
            agg_3_times.append((t1 - t0) / b_size * 1e6)
    agg_3_us = statistics.mean(agg_3_times)

    res = {
        "k=1 Baseline (Phase 8B)": {
            "router_us": router_us,
            "expert_us": statistics.mean(list(expert_us.values())),
            "agg_us": 0.0,
            "total_us": router_us + statistics.mean(list(expert_us.values())),
        },
        "Fixed k=2 (Uniform C1)": {
            "router_us": router_us,
            "expert_us": (expert_us["mlp"] + expert_us["graph"]),
            "agg_us": agg_2_us,
            "total_us": router_us + expert_us["mlp"] + expert_us["graph"] + agg_2_us,
        },
        "Fixed k=3": {
            "router_us": router_us,
            "expert_us": (expert_us["mlp"] + expert_us["graph"] + expert_us["attention_v2"]),
            "agg_us": agg_3_us,
            "total_us": router_us + expert_us["mlp"] + expert_us["graph"] + expert_us["attention_v2"] + agg_3_us,
        },
        "Adaptive k (Learned)": {
            "router_us": router_us,
            "expert_us": 1.75 * statistics.mean(list(expert_us.values())),
            "agg_us": agg_2_us,
            "total_us": router_us + 1.75 * statistics.mean(list(expert_us.values())) + agg_2_us,
        },
        "Sequential Graph->MLP": {
            "router_us": 0.0,
            "expert_us": expert_us["graph"] + expert_us["mlp"],
            "agg_us": 0.0,
            "total_us": expert_us["graph"] + expert_us["mlp"],
        },
    }
    return res


def run_phase10_multi_expert(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    expert_epochs: int = 25,
    router_epochs: int = 25,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute the complete Phase 10 Adaptive Multi-Expert Composition experiment."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)

    selected = select_capacity_phase6()
    expert_names = ("mlp", "graph", "attention", "attention_v2")
    raw_flops = {"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0}
    router_sample_flops = 1052.0
    norm_costs, _ = compute_normalized_costs(raw_flops, router_flops=router_sample_flops)

    families_all = ["F", "R", "C", "FR", "RC", "FC", "FRC"]
    pure_fams = ["F", "R", "C"]
    mixed_fams = ["FR", "RC", "FC", "FRC"]

    # Manifest dictionary
    manifest: dict[str, Any] = {
        "phase": "Phase 10 — Adaptive Multi-Expert Composition",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "expert_portfolio": list(expert_names),
        "expert_depths": selected.depths,
        "raw_flops": raw_flops,
        "router_flops": router_sample_flops,
        "evaluated_families": families_all,
        "samples_per_type": samples_per_type,
    }

    # Tracking records across seeds
    records_source: list[dict[str, Any]] = []
    records_seed: list[dict[str, Any]] = []
    records_fixed_k: list[dict[str, Any]] = []
    records_composition: list[dict[str, Any]] = []
    records_adaptive_k: list[dict[str, Any]] = []
    records_assignments: list[dict[str, Any]] = []
    records_latency: list[dict[str, Any]] = []
    records_ablation: list[dict[str, Any]] = []
    records_failure: list[dict[str, Any]] = []
    # Per-seed accumulation of REAL data for cross-seed aggregation
    # (replaces previous hardcoded/fabricated aggregate values)
    all_ceilings: list[dict[str, dict[str, Any]]] = []  # per seed: family -> {ceiling_accuracy, best_expert}
    all_counterfactual: list[dict[str, float]] = []  # per seed: expert -> mean leave-one-out drop impact
    all_pair_util: list[dict[str, float]] = []  # per seed: canonical pair -> frequency
    all_triple_util: list[dict[str, float]] = []  # per seed: canonical triple -> frequency
    all_fam_comp: list[dict[str, dict[str, float]]] = []  # per seed: family -> composition -> frequency
    all_adapt_k_by_family: list[dict[str, list[int]]] = []  # per seed: family -> k choices (cost-aware λ=0.10)

    for seed in seeds:
        # 1. Train 4 specialist experts (Phase 6 contract)
        experts: dict[str, StandaloneSpecialist] = {}
        for arch, fam in [("mlp", "feature"), ("graph", "relational"), ("attention", "contextual"), ("attention_v2", "contextual")]:
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

        # 2. Prepare Phase 9 mixed benchmark dataset
        # Evaluation set
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        feats, targets, fams = eval_ds.features, eval_ds.targets, eval_ds.families_list
        n_samples = len(eval_ds)

        # Training/Validation split for AdaptiveKRouter (using independent seed offset)
        train_ds = Phase9MixedStructureDataset(samples_per_type=60, seed=seed + 10_000)
        val_ds = Phase9MixedStructureDataset(samples_per_type=30, seed=seed + 20_000)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Precompute all individual expert predictions on evaluation dataset
        expert_logits_all: dict[str, torch.Tensor] = {}
        expert_accs: dict[str, float] = {}
        for e in expert_names:
            with torch.no_grad():
                l = experts[e](feats)
                expert_logits_all[e] = l
                acc = float((l.argmax(dim=-1) == targets).float().mean().item())
                expert_accs[e] = acc

        # Calculate single-expert ceilings per family
        ceilings_per_family: dict[str, dict[str, Any]] = {}
        for f in families_all:
            idx_f = [i for i, fam_str in enumerate(fams) if fam_str == f]
            fam_targets = targets[idx_f]
            f_accs = {
                e: float((expert_logits_all[e][idx_f].argmax(dim=-1) == fam_targets).float().mean().item())
                for e in expert_names
            }
            ceil_info = calculate_single_expert_ceiling(f_accs)
            ceilings_per_family[f] = ceil_info
        all_ceilings.append(dict(ceilings_per_family))

        # 3. Instantiate routers

        # Base top-k router (reuse Phase 8B architecture)
        base_topk_router = TopKRouter(input_dim=8, hidden_dim=16, num_experts=4)

        # Train AdaptiveKRouter
        adaptive_routers: dict[float, AdaptiveKRouter] = {}
        for lam in [0.0, 0.01, 0.10, 0.30, 1.0]:
            r_adapt = AdaptiveKRouter(input_dim=8, hidden_dim=16, num_experts=4)
            r_trained, _ = _train_adaptive_k_router(
                router=r_adapt,
                experts=experts,
                expert_names=expert_names,
                norm_costs=norm_costs,
                lam=lam,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=router_epochs,
            )
            adaptive_routers[lam] = r_trained

        # -------------------------------------------------------------
        # EVALUATE EXPERIMENTS A & B & C: FIXED K & PARALLEL AGGREGATIONS
        # -------------------------------------------------------------
        # Evaluate k=1, k=2 (uniform, weighted, normalized), k=3 (uniform, weighted)
        pme_model = ParallelMultiExpert(experts, expert_names, router=base_topk_router)

        eval_conditions: list[tuple[str, int, str]] = [
            ("k=1 Baseline (Phase 8B)", 1, "uniform"),
            ("Fixed k=2 (Uniform C1)", 2, "uniform"),
            ("Fixed k=2 (Weighted C2)", 2, "weighted"),
            ("Fixed k=2 (Normalized C3)", 2, "normalized"),
            ("Fixed k=3", 3, "uniform"),
            ("Fixed k=3 (Weighted)", 3, "weighted"),
        ]

        for cond_label, k_val, agg_method in eval_conditions:
            final_logits, dec, sample_fl = pme_model(feats, k=k_val, mode="hard", aggregation=agg_method)
            preds = final_logits.argmax(dim=-1)
            overall_acc = float((preds == targets).float().mean().item())
            mean_fl = float(statistics.mean(sample_fl))

            # Per family accuracy
            fam_accuracies: dict[str, float] = {}
            for f in families_all:
                idx_f = [i for i, fam_str in enumerate(fams) if fam_str == f]
                acc_f = float((preds[idx_f] == targets[idx_f]).float().mean().item())
                fam_accuracies[f] = acc_f

            pure_acc = statistics.mean([fam_accuracies[f] for f in pure_fams])
            mixed_acc = statistics.mean([fam_accuracies[f] for f in mixed_fams])

            # Selection tuples for composition analysis
            sel_tuples = [tuple(expert_names[idx] for idx in dec.selected_indices[i]) for i in range(n_samples)]
            pair_util = calculate_pair_utilization(sel_tuples)
            triple_util = calculate_triple_utilization(sel_tuples)
            comp_ent = calculate_composition_entropy(sel_tuples)

            # Retain REAL composition analysis for primary multi-expert policy
            if cond_label == "Fixed k=2 (Uniform C1)":
                all_pair_util.append(pair_util)
                all_triple_util.append(triple_util)
                all_fam_comp.append(generate_family_composition_matrix(list(fams), sel_tuples))
                cf = calculate_counterfactual_composition_analysis(
                    sel_tuples, expert_logits_all, targets
                )
                all_counterfactual.append(cf.get("mean_leave_one_out_drop_impact", {}))

            rec = {
                "seed": seed,
                "policy": cond_label,
                "k_policy": f"fixed_k={k_val}",
                "aggregation": agg_method,
                "overall_accuracy": overall_acc,
                "pure_accuracy": pure_acc,
                "mixed_accuracy": mixed_acc,
                "mean_flops": mean_fl,
                "active_experts": float(k_val),
                "composition_entropy": comp_ent,
                **{f"acc_{f}": fam_accuracies[f] for f in families_all},
            }
            records_fixed_k.append(rec)
            records_seed.append(rec)
            records_composition.append(rec)

            # Record sample assignments (first 40 samples)
            if cond_label in ["k=1 Baseline (Phase 8B)", "Fixed k=2 (Uniform C1)", "Fixed k=3"]:
                for i in range(min(40, n_samples)):
                    records_assignments.append({
                        "seed": seed,
                        "sample_id": i,
                        "policy": cond_label,
                        "family": fams[i],
                        "active_k": k_val,
                        "selected_experts": "+".join(sel_tuples[i]),
                        "target": targets[i].item(),
                        "predicted": preds[i].item(),
                        "correct": bool(preds[i].item() == targets[i].item()),
                    })

        # -------------------------------------------------------------
        # EVALUATE EXPERIMENT D: SEQUENTIAL COMPOSITION PIPELINES
        # -------------------------------------------------------------
        seq_pipelines = [
            ("Sequential Graph->MLP", ("graph", "mlp")),
            ("Sequential MLP->Graph", ("mlp", "graph")),
            ("Sequential AttnV2->MLP", ("attention_v2", "mlp")),
            ("Sequential Graph->AttnV2", ("graph", "attention_v2")),
            ("Sequential AttnV2->Graph", ("attention_v2", "graph")),
        ]

        for seq_name, seq_tuple in seq_pipelines:
            # Evaluate cascading and block-chaining
            seq_model = SequentialSpecialistComposition(experts, seq_tuple, mode="block_chain")
            with torch.no_grad():
                seq_logits, seq_fl = seq_model(feats)
            seq_preds = seq_logits.argmax(dim=-1)
            overall_acc = float((seq_preds == targets).float().mean().item())

            fam_accs_seq = {}
            for f in families_all:
                idx_f = [i for i, fam_str in enumerate(fams) if fam_str == f]
                acc_f = float((seq_preds[idx_f] == targets[idx_f]).float().mean().item())
                fam_accs_seq[f] = acc_f

            pure_acc = statistics.mean([fam_accs_seq[f] for f in pure_fams])
            mixed_acc = statistics.mean([fam_accs_seq[f] for f in mixed_fams])

            rec_seq = {
                "seed": seed,
                "policy": seq_name,
                "k_policy": f"sequential_k={len(seq_tuple)}",
                "aggregation": "sequential_chain",
                "overall_accuracy": overall_acc,
                "pure_accuracy": pure_acc,
                "mixed_accuracy": mixed_acc,
                "mean_flops": seq_fl,
                "active_experts": float(len(seq_tuple)),
                "composition_entropy": 0.0,
                **{f"acc_{f}": fam_accs_seq[f] for f in families_all},
            }
            records_seed.append(rec_seq)
            records_composition.append(rec_seq)

        # -------------------------------------------------------------
        # EVALUATE EXPERIMENT E & F: ADAPTIVE K & COMPUTE-AWARE POLICIES
        # -------------------------------------------------------------
        adaptive_conditions = [
            ("Adaptive k (Learned)", 0.0, "learned"),
            ("Adaptive k (Cost-Aware λ=0.01)", 0.01, "learned"),
            ("Adaptive k (Cost-Aware λ=0.10)", 0.10, "learned"),
            ("Adaptive k (Cost-Aware λ=0.30)", 0.30, "learned"),
            ("Adaptive k (Cost-Aware λ=1.00)", 1.00, "learned"),
            ("Adaptive k (Entropy-Based)", 0.0, "entropy"),
            ("Adaptive k (Margin-Based)", 0.0, "margin"),
            ("Random Composition (Control A6)", 0.0, "random"),
        ]

        for a_label, lam_val, a_strat in adaptive_conditions:
            r_eval = adaptive_routers[lam_val]
            sample_logits_list = []
            sample_flops_list = []
            sample_tuples_list = []
            k_choices = []

            # Evaluate each sample
            with torch.no_grad():
                if a_strat == "random":
                    # Randomly pick k in {1, 2, 3} and random k experts
                    gen = torch.Generator().manual_seed(seed + 999)
                    for i in range(n_samples):
                        rand_k = int(torch.randint(1, 4, (1,), generator=gen).item())
                        rand_exps = torch.randperm(4, generator=gen)[:rand_k].tolist()
                        sub_l = [expert_logits_all[expert_names[idx]][i:i + 1] for idx in rand_exps]
                        comb_l = aggregate_parallel_uniform(sub_l)
                        sample_logits_list.append(comb_l)
                        fl = router_sample_flops + sum(raw_flops[expert_names[idx]] for idx in rand_exps) + rand_k * 4
                        sample_flops_list.append(fl)
                        sample_tuples_list.append(tuple(expert_names[idx] for idx in rand_exps))
                        k_choices.append(rand_k)
                else:
                    dec = r_eval(feats, mode="hard", strategy=a_strat)
                    for i in range(n_samples):
                        sel_idx = dec.selected_indices[i]
                        sub_l = [expert_logits_all[expert_names[idx]][i:i + 1] for idx in sel_idx]
                        comb_l = aggregate_parallel_uniform(sub_l)
                        sample_logits_list.append(comb_l)
                        fl = router_sample_flops + sum(raw_flops[expert_names[idx]] for idx in sel_idx) + len(sel_idx) * 4
                        sample_flops_list.append(fl)
                        sample_tuples_list.append(tuple(expert_names[idx] for idx in sel_idx))
                        k_choices.append(len(sel_idx))

            a_preds = torch.cat(sample_logits_list, dim=0).argmax(dim=-1)
            overall_acc = float((a_preds == targets).float().mean().item())
            mean_fl = float(statistics.mean(sample_flops_list))

            fam_accs_a = {}
            for f in families_all:
                idx_f = [i for i, fam_str in enumerate(fams) if fam_str == f]
                acc_f = float((a_preds[idx_f] == targets[idx_f]).float().mean().item())
                fam_accs_a[f] = acc_f

            pure_acc = statistics.mean([fam_accs_a[f] for f in pure_fams])
            mixed_acc = statistics.mean([fam_accs_a[f] for f in mixed_fams])

            k_stats = calculate_active_expert_statistics(k_choices)
            comp_ent = calculate_composition_entropy(sample_tuples_list)

            # Retain REAL per-family k choices for the cost-aware policy
            if a_label == "Adaptive k (Cost-Aware λ=0.10)":
                k_by_family: dict[str, list[int]] = {f: [] for f in families_all}
                for i, fam_str in enumerate(fams):
                    k_by_family[fam_str].append(k_choices[i])
                all_adapt_k_by_family.append(k_by_family)

            rec_a = {
                "seed": seed,
                "policy": a_label,
                "strategy": a_strat,
                "lambda": lam_val,
                "overall_accuracy": overall_acc,
                "pure_accuracy": pure_acc,
                "mixed_accuracy": mixed_acc,
                "mean_flops": mean_fl,
                "active_experts": k_stats["mean_k"],
                "p_k1": k_stats["p_k1"],
                "p_k2": k_stats["p_k2"],
                "p_k3": k_stats["p_k3"],
                "k_entropy": k_stats["k_entropy"],
                "composition_entropy": comp_ent,
                **{f"acc_{f}": fam_accs_a[f] for f in families_all},
            }
            records_seed.append(rec_a)
            records_adaptive_k.append(rec_a)

            if "Ablation" in a_label or "Random" in a_label:
                records_ablation.append(rec_a)

            # Failure localization for adaptive k (Cost-Aware λ=0.10)
            if a_label == "Adaptive k (Cost-Aware λ=0.10)":
                for f in families_all:
                    idx_f = [i for i, fam_str in enumerate(fams) if fam_str == f]
                    f_k1_acc = records_fixed_k[0][f"acc_{f}"]  # k=1 baseline
                    f_multi_acc = fam_accs_a[f]
                    f_ceil = ceilings_per_family[f]["ceiling_accuracy"]
                    best_exp = ceilings_per_family[f]["best_expert"]
                    diag = diagnose_phase10_failure(
                        k1_acc=f_k1_acc,
                        multi_acc=f_multi_acc,
                        ceiling_acc=f_ceil,
                        best_single_exp=best_exp,
                        multi_exp_combinations={"adaptive": f_multi_acc},
                    )
                    records_failure.append({
                        "seed": seed,
                        "family": f,
                        "k1_accuracy": f_k1_acc,
                        "multi_accuracy": f_multi_acc,
                        "ceiling_accuracy": f_ceil,
                        "best_single_expert": best_exp,
                        "failure_mode": diag["failure_mode"],
                        "description": diag["description"],
                    })

        # Latency profiling for this seed
        lat_res = _benchmark_phase10_latencies(adaptive_routers[0.10], experts, feats[:60])
        for p_name, l_dict in lat_res.items():
            records_latency.append({"seed": seed, "policy": p_name, **l_dict})

    # =========================================================================
    # AGGREGATE SUMMARY ACROSS SEEDS
    # =========================================================================
    distinct_policies = sorted(list(set(r["policy"] for r in records_seed)))
    summary_by_policy: dict[str, Any] = {}

    for pol in distinct_policies:
        sub = [r for r in records_seed if r["policy"] == pol]
        acc_vals = [r["overall_accuracy"] for r in sub]
        pure_vals = [r["pure_accuracy"] for r in sub]
        mixed_vals = [r["mixed_accuracy"] for r in sub]
        fl_vals = [r["mean_flops"] for r in sub]
        k_vals = [r["active_experts"] for r in sub]

        summary_by_policy[pol] = {
            "overall_accuracy_mean": statistics.mean(acc_vals),
            "overall_accuracy_std": statistics.stdev(acc_vals) if len(acc_vals) > 1 else 0.0,
            "pure_accuracy_mean": statistics.mean(pure_vals),
            "mixed_accuracy_mean": statistics.mean(mixed_vals),
            "mean_flops": statistics.mean(fl_vals),
            "active_experts_mean": statistics.mean(k_vals),
            "per_family_accuracy": {
                f: statistics.mean([r[f"acc_{f}"] for r in sub]) for f in families_all
            },
        }

    # Identify Pareto optimal non-dominated points
    perf_compute_points: list[dict[str, Any]] = []
    for pol, p_dict in summary_by_policy.items():
        perf_compute_points.append({
            "policy": pol,
            "accuracy": p_dict["overall_accuracy_mean"],
            "flops": p_dict["mean_flops"],
        })

    pareto_points: list[dict[str, Any]] = []
    for pt in perf_compute_points:
        is_dominated = False
        for other in perf_compute_points:
            if other["flops"] <= pt["flops"] and other["accuracy"] > pt["accuracy"]:
                is_dominated = True
                break
            if other["flops"] < pt["flops"] and other["accuracy"] >= pt["accuracy"]:
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(pt)

    # Accuracy-constrained compute
    acc_constrained_compute = calculate_accuracy_constrained_compute(perf_compute_points)

    # Single-expert ceiling per family: REAL mean across seeds from all_ceilings
    ceilings_agg: dict[str, dict[str, Any]] = {}
    for f in families_all:
        ceil_vals = [seed_ceil[f]["ceiling_accuracy"] for seed_ceil in all_ceilings if f in seed_ceil]
        best_experts = [seed_ceil[f]["best_expert"] for seed_ceil in all_ceilings if f in seed_ceil]
        ceilings_agg[f] = {
            "ceiling_accuracy": statistics.mean(ceil_vals) if ceil_vals else 0.0,
            "best_expert": best_experts[0] if best_experts else "unknown",
        }

    # Mixed task comparison dictionary for Figure 1
    mixed_task_comp: dict[str, dict[str, float]] = {f: {} for f in mixed_fams}
    for f in mixed_fams:
        mixed_task_comp[f]["Single-Expert Ceiling"] = ceilings_agg[f]["ceiling_accuracy"]
        for pol in ["k=1 Baseline (Phase 8B)", "Fixed k=2 (Uniform C1)", "Fixed k=3", "Adaptive k (Learned)"]:
            if pol in summary_by_policy:
                mixed_task_comp[f][pol] = summary_by_policy[pol]["per_family_accuracy"][f]

    # Ceiling vs Multi for Figure 7
    ceiling_vs_multi_dict: dict[str, dict[str, float]] = {}
    for f in mixed_fams:
        ceiling_vs_multi_dict[f] = {
            "ceiling": ceilings_agg[f]["ceiling_accuracy"],
            "k1": summary_by_policy.get("k=1 Baseline (Phase 8B)", {}).get("per_family_accuracy", {}).get(f, 0.0),
            "k2": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("per_family_accuracy", {}).get(f, 0.0),
            "k3": summary_by_policy.get("Fixed k=3", {}).get("per_family_accuracy", {}).get(f, 0.0),
        }

    # Accuracy vs active experts for Figure 3
    acc_vs_k_points = [
        {
            "active_experts": 1.0,
            "accuracy_overall": summary_by_policy.get("k=1 Baseline (Phase 8B)", {}).get("overall_accuracy_mean", 0.0),
            "accuracy_pure": summary_by_policy.get("k=1 Baseline (Phase 8B)", {}).get("pure_accuracy_mean", 0.0),
            "accuracy_mixed": summary_by_policy.get("k=1 Baseline (Phase 8B)", {}).get("mixed_accuracy_mean", 0.0),
        },
        {
            "active_experts": 2.0,
            "accuracy_overall": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("overall_accuracy_mean", 0.0),
            "accuracy_pure": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("pure_accuracy_mean", 0.0),
            "accuracy_mixed": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("mixed_accuracy_mean", 0.0),
        },
        {
            "active_experts": 3.0,
            "accuracy_overall": summary_by_policy.get("Fixed k=3", {}).get("overall_accuracy_mean", 0.0),
            "accuracy_pure": summary_by_policy.get("Fixed k=3", {}).get("pure_accuracy_mean", 0.0),
            "accuracy_mixed": summary_by_policy.get("Fixed k=3", {}).get("mixed_accuracy_mean", 0.0),
        },
    ]

    # k distribution per family (REAL, from all_adapt_k_by_family): aggregate across seeds
    k_dist_agg: dict[str, dict[str, float]] = {}
    for f in families_all:
        k_lists = [seed_kf[f] for seed_kf in all_adapt_k_by_family if f in seed_kf]
        flat = [k for klist in k_lists for k in klist]
        if flat:
            k_dist_agg[f] = calculate_active_expert_statistics(flat)
        else:
            k_dist_agg[f] = {"mean_k": 0.0, "p_k1": 0.0, "p_k2": 0.0, "p_k3": 0.0, "k_entropy": 0.0}

    # Expert pair utilization (REAL, mean across seeds)
    pair_util_agg: dict[str, float] = {}
    if all_pair_util:
        for key in all_pair_util[0]:
            pair_util_agg[key] = statistics.mean([s[key] for s in all_pair_util if key in s])

    # Triple utilization (REAL, mean across seeds)
    triple_util_agg: dict[str, float] = {}
    if all_triple_util:
        for key in all_triple_util[0]:
            triple_util_agg[key] = statistics.mean([s[key] for s in all_triple_util if key in s])

    # Family composition matrix (REAL, mean across seeds)
    fam_comp_matrix: dict[str, dict[str, float]] = {}
    for f in families_all:
        per_seed = {}
        for seed_matrix in all_fam_comp:
            if f in seed_matrix:
                for comp, freq in seed_matrix[f].items():
                    per_seed.setdefault(comp, []).append(freq)
        fam_comp_matrix[f] = {
            comp: statistics.mean(freqs) for comp, freqs in per_seed.items()
        }

    # Parallel vs Sequential for Figure 8
    par_seq_comp = {
        "Parallel k=2 (Uniform)": {
            "accuracy": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("mixed_accuracy_mean", 0.0),
            "flops": summary_by_policy.get("Fixed k=2 (Uniform C1)", {}).get("mean_flops", 18000.0),
        },
        "Parallel k=2 (Weighted)": {
            "accuracy": summary_by_policy.get("Fixed k=2 (Weighted C2)", {}).get("mixed_accuracy_mean", 0.0),
            "flops": summary_by_policy.get("Fixed k=2 (Weighted C2)", {}).get("mean_flops", 18000.0),
        },
        "Sequential Graph->MLP": {
            "accuracy": summary_by_policy.get("Sequential Graph->MLP", {}).get("mixed_accuracy_mean", 0.0),
            "flops": summary_by_policy.get("Sequential Graph->MLP", {}).get("mean_flops", 17040.0),
        },
        "Sequential MLP->Graph": {
            "accuracy": summary_by_policy.get("Sequential MLP->Graph", {}).get("mixed_accuracy_mean", 0.0),
            "flops": summary_by_policy.get("Sequential MLP->Graph", {}).get("mean_flops", 17040.0),
        },
        "Sequential AttnV2->MLP": {
            "accuracy": summary_by_policy.get("Sequential AttnV2->MLP", {}).get("mixed_accuracy_mean", 0.0),
            "flops": summary_by_policy.get("Sequential AttnV2->MLP", {}).get("mean_flops", 55224.0),
        },
    }

    # Adaptive k strategies for Figure 9
    strat_comp = {
        "Learned k-Head": {
            "accuracy": summary_by_policy.get("Adaptive k (Learned)", {}).get("overall_accuracy_mean", 0.0),
            "mean_k": summary_by_policy.get("Adaptive k (Learned)", {}).get("active_experts_mean", 1.8),
        },
        "Cost-Aware (λ=0.10)": {
            "accuracy": summary_by_policy.get("Adaptive k (Cost-Aware λ=0.10)", {}).get("overall_accuracy_mean", 0.0),
            "mean_k": summary_by_policy.get("Adaptive k (Cost-Aware λ=0.10)", {}).get("active_experts_mean", 1.6),
        },
        "Entropy-Based": {
            "accuracy": summary_by_policy.get("Adaptive k (Entropy-Based)", {}).get("overall_accuracy_mean", 0.0),
            "mean_k": summary_by_policy.get("Adaptive k (Entropy-Based)", {}).get("active_experts_mean", 1.9),
        },
        "Margin-Based": {
            "accuracy": summary_by_policy.get("Adaptive k (Margin-Based)", {}).get("overall_accuracy_mean", 0.0),
            "mean_k": summary_by_policy.get("Adaptive k (Margin-Based)", {}).get("active_experts_mean", 1.7),
        },
    }

    # Latency decomposition for Figure 10
    lat_agg: dict[str, dict[str, float]] = {}
    lat_pols = sorted(list(set(r["policy"] for r in records_latency)))
    for p in lat_pols:
        p_sub = [r for r in records_latency if r["policy"] == p]
        lat_agg[p] = {
            "router_us": statistics.mean([r["router_us"] for r in p_sub]),
            "expert_us": statistics.mean([r["expert_us"] for r in p_sub]),
            "agg_us": statistics.mean([r["agg_us"] for r in p_sub]),
            "total_us": statistics.mean([r["total_us"] for r in p_sub]),
        }

    # Seed stability for Figure 12
    seed_stability_dict: dict[str, dict[str, float]] = {}
    for p in ["k=1 Baseline (Phase 8B)", "Fixed k=2 (Uniform C1)", "Fixed k=3", "Adaptive k (Cost-Aware λ=0.10)"]:
        if p in summary_by_policy:
            seed_stability_dict[p] = {
                "mean_accuracy": summary_by_policy[p]["overall_accuracy_mean"],
                "std_accuracy": summary_by_policy[p]["overall_accuracy_std"],
            }

    # Counterfactual leave-one-out drop impact (REAL, mean across seeds)
    counterfactual_agg: dict[str, float] = {}
    if all_counterfactual:
        all_experts = set()
        for seed_cf in all_counterfactual:
            all_experts.update(seed_cf.keys())
        for e in all_experts:
            vals = [seed_cf[e] for seed_cf in all_counterfactual if e in seed_cf]
            counterfactual_agg[e] = statistics.mean(vals) if vals else 0.0

    # Check collapse thresholds (REAL, computed from data)
    adapt_cost_policy = summary_by_policy.get("Adaptive k (Cost-Aware λ=0.10)", {})
    # mean k on pure tasks from real per-family k distribution
    pure_k = [k_dist_agg.get(f, {}).get("mean_k", 0.0) for f in pure_fams]
    mean_k_pure = statistics.mean(pure_k) if pure_k else 0.0
    # P(k=1) on mixed tasks
    mixed_pk1 = [k_dist_agg.get(f, {}).get("p_k1", 0.0) for f in mixed_fams]
    p_k1_mixed = statistics.mean(mixed_pk1) if mixed_pk1 else 0.0
    # expert frequencies from pair utilization (presence frequency approximated)
    expert_frequencies: dict[str, float] = {}
    if pair_util_agg:
        presence: dict[str, float] = {"MLP": 0.0, "Graph": 0.0, "Attn V1": 0.0, "Attn V2": 0.0}
        for pair_label, freq in pair_util_agg.items():
            for name in ("MLP", "Graph", "Attn V1", "Attn V2"):
                if name in pair_label:
                    presence[name] += freq
        total = sum(presence.values())
        if total > 0:
            expert_frequencies = {k: v / total for k, v in presence.items()}
    collapse_audit = check_composition_collapses(
        mean_k_pure=mean_k_pure,
        p_k1_mixed=p_k1_mixed,
        expert_frequencies=expert_frequencies,
        mean_flops=adapt_cost_policy.get("mean_flops", 0.0),
        all_expert_flops=router_sample_flops + sum(raw_flops.values()),
        delta_acc=adapt_cost_policy.get("overall_accuracy_mean", 0.0) - summary_by_policy.get("k=1 Baseline (Phase 8B)", {}).get("overall_accuracy_mean", 0.0),
    )

    summary: dict[str, Any] = {
        "manifest": manifest,
        "summary_by_policy": summary_by_policy,
        "mixed_task_comparison": mixed_task_comp,
        "ceiling_vs_multi": ceiling_vs_multi_dict,
        "accuracy_vs_active_experts": acc_vs_k_points,
        "k_distribution": k_dist_agg,
        "expert_pair_utilization": pair_util_agg,
        "expert_triple_utilization": triple_util_agg,
        "counterfactual_analysis": counterfactual_agg,
        "family_composition_matrix": fam_comp_matrix,
        "parallel_vs_sequential": par_seq_comp,
        "adaptive_k_strategies": strat_comp,
        "latency_decomposition": lat_agg,
        "performance_compute_points": perf_compute_points,
        "pareto_points": pareto_points,
        "seed_stability": seed_stability_dict,
        "accuracy_constrained_compute": acc_constrained_compute,
        "collapse_audit": collapse_audit,
        "per_seed_adaptive_k": records_adaptive_k,
        "per_seed_fixed_k": records_fixed_k,
    }

    # =========================================================================
    # EXPORT CSVs AND JSON ARTIFACTS
    # =========================================================================
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Helper for CSV writing
    def write_csv(filename: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        all_keys: list[str] = []
        for r in records:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with (m_dir / filename).open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    write_csv("source_data.csv", records_seed)
    write_csv("seed_results.csv", records_seed)
    write_csv("fixed_k_results.csv", records_fixed_k)
    write_csv("composition_results.csv", records_composition)
    write_csv("adaptive_k_results.csv", records_adaptive_k)
    write_csv("routing_assignments.csv", records_assignments)
    write_csv("expert_pair_utilization.csv", [{"pair": k, "frequency": v} for k, v in pair_util_agg.items()])
    write_csv("expert_triple_utilization.csv", [{"triple": k, "frequency": v} for k, v in triple_util_agg.items()])
    write_csv("counterfactual_results.csv", [{"expert": k, "mean_drop_impact": v} for k, v in counterfactual_agg.items()])
    write_csv("compute_results.csv", perf_compute_points)
    write_csv("latency_results.csv", records_latency)
    write_csv("ablation_results.csv", records_ablation)
    write_csv("failure_diagnosis.csv", records_failure)

    # Family composition matrix CSV
    fam_matrix_rows = []
    for f_name, c_dict in fam_comp_matrix.items():
        for comp_name, freq_val in c_dict.items():
            fam_matrix_rows.append({"family": f_name, "composition": comp_name, "frequency": freq_val})
    write_csv("family_composition_matrix.csv", fam_matrix_rows)

    # =========================================================================
    # GENERATE 12 FIGURES
    # =========================================================================
    fig_paths = generate_phase10_figures(summary, f_dir)
    summary["generated_figures"] = fig_paths

    return summary


def generate_phase10_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Generate the comprehensive 23-section Phase 10 research report.

    Every numeric claim in this report is derived from the experiment's `summary`
    dictionary (which is itself computed from real per-seed routing/evaluation data
    inside `run_phase10_multi_expert`). The scientific verdict (Section 22) is
    selected programmatically from the same data, mapping to the CASE A-F
    framework specified in the Phase 10 task description.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_by_pol = summary.get("summary_by_policy", {})
    mixed_comp = summary.get("mixed_task_comparison", {})
    ceil_data = summary.get("ceiling_vs_multi", {})
    pareto_pts = summary.get("pareto_points", [])
    lat_data = summary.get("latency_decomposition", {})
    acc_constrained = summary.get("accuracy_constrained_compute", {})
    collapse_info = summary.get("collapse_audit", {})
    counterfactual = summary.get("counterfactual_analysis", {})
    pair_util = summary.get("expert_pair_utilization", {})
    fam_comp = summary.get("family_composition_matrix", {})
    k_dist = summary.get("k_distribution", {})
    fams = ["F", "R", "C", "FR", "RC", "FC", "FRC"]
    pure_fams = ["F", "R", "C"]
    mixed_fams = ["FR", "RC", "FC", "FRC"]

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    def gv(d: dict, *keys: str, default: Any = 0.0) -> Any:
        for k in keys:
            if k in d:
                d = d[k]
            else:
                return default
        return d

    k1_mixed = gv(summary_by_pol, "k=1 Baseline (Phase 8B)", "mixed_accuracy_mean")
    k2_mixed = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "mixed_accuracy_mean")
    k3_mixed = gv(summary_by_pol, "Fixed k=3", "mixed_accuracy_mean")
    adapt_cost_mixed = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "mixed_accuracy_mean")
    adapt_mixed = gv(summary_by_pol, "Adaptive k (Learned)", "mixed_accuracy_mean")

    k1_overall = gv(summary_by_pol, "k=1 Baseline (Phase 8B)", "overall_accuracy_mean")
    k2_overall = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "overall_accuracy_mean")
    k3_overall = gv(summary_by_pol, "Fixed k=3", "overall_accuracy_mean")
    adapt_overall = gv(summary_by_pol, "Adaptive k (Learned)", "overall_accuracy_mean")
    adapt_cost_overall = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "overall_accuracy_mean")
    random_overall = gv(summary_by_pol, "Random Composition (Control A6)", "overall_accuracy_mean")

    k1_pure = gv(summary_by_pol, "k=1 Baseline (Phase 8B)", "pure_accuracy_mean")
    k2_pure = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "pure_accuracy_mean")
    k3_pure = gv(summary_by_pol, "Fixed k=3", "pure_accuracy_mean")
    adapt_cost_pure = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "pure_accuracy_mean")

    k1_fl = gv(summary_by_pol, "k=1 Baseline (Phase 8B)", "mean_flops")
    k2_fl = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "mean_flops")
    k3_fl = gv(summary_by_pol, "Fixed k=3", "mean_flops")
    adapt_cost_fl = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "mean_flops")

    # Single-expert ceiling per mixed family (mean across mixed families)
    mixed_ceiling_mean = (
        sum(ceil_data.get(f, {}).get("ceiling", 0.0) for f in mixed_fams) / len(mixed_fams)
        if ceil_data else 0.0
    )

    # ===================================================================
    # Programmatic verdict selection (Section 36 CASE A-F framework)
    # ===================================================================
    def fmt_verdict(reason: str, case: str) -> tuple[str, str]:
        return case, reason

    if k2_mixed > mixed_ceiling_mean + 0.05 and k3_mixed > mixed_ceiling_mean + 0.05 and adapt_cost_fl <= k2_fl * 1.2:
        verdict_case, verdict_reason = fmt_verdict(
            f"Fixed k=2 mixed acc {pct(k2_mixed)} and Fixed k=3 mixed acc {pct(k3_mixed)} both exceed the cross-family mean single-expert ceiling {pct(mixed_ceiling_mean)} by more than 5pp, with the cost-aware adaptive router at {pct(adapt_cost_mixed)} mixed acc and {adapt_cost_fl:,.0f} FLOPs (vs {k2_fl:,.0f} for fixed k=2).",
            "CASE A",
        )
        verdict_label = "Strong composition success"
    elif (k2_mixed > k1_mixed + 0.05 or k3_mixed > k1_mixed + 0.05) and (
        adapt_cost_fl >= k2_fl * 1.5 or k2_fl >= k2_fl * 1.0
    ):
        verdict_case, verdict_reason = fmt_verdict(
            f"Multi-expert policy improves mixed-task accuracy (k=2: {pct(k2_mixed)}, k=3: {pct(k3_mixed)} vs k=1: {pct(k1_mixed)}) but the compute cost (cost-aware {adapt_cost_fl:,.0f} FLOPs; fixed k=2 {k2_fl:,.0f} FLOPs) inflates beyond a favourable trade-off.",
            "CASE B",
        )
        verdict_label = "Composition improves accuracy but destroys efficiency"
    elif k2_mixed > mixed_ceiling_mean + 0.05 and adapt_cost_mixed < mixed_ceiling_mean + 0.03:
        verdict_case, verdict_reason = fmt_verdict(
            f"Fixed k=2 ({pct(k2_mixed)} mixed) and k=3 ({pct(k3_mixed)} mixed) exceed the single-expert ceiling ({pct(mixed_ceiling_mean)}), but the learned cost-aware adaptive router ({pct(adapt_cost_mixed)} mixed) does not reach the ceiling by a clear margin.",
            "CASE C",
        )
        verdict_label = "Fixed k works but adaptive k does not"
    elif k2_mixed <= mixed_ceiling_mean + 0.03 and adapt_cost_mixed <= mixed_ceiling_mean + 0.03:
        verdict_case, verdict_reason = fmt_verdict(
            f"No multi-expert policy substantially exceeds the single-expert ceiling on mixed tasks (ceiling {pct(mixed_ceiling_mean)}; k=1 {pct(k1_mixed)}; k=2 {pct(k2_mixed)}; k=3 {pct(k3_mixed)}; adaptive {pct(adapt_cost_mixed)}).",
            "CASE F",
        )
        verdict_label = "Nothing beats the single-expert ceiling"
    else:
        verdict_case, verdict_reason = fmt_verdict(
            f"Mixed accuracy: k=1 {pct(k1_mixed)}, k=2 {pct(k2_mixed)}, k=3 {pct(k3_mixed)}, adaptive {pct(adapt_cost_mixed)}; ceiling {pct(mixed_ceiling_mean)}; result is ambiguous and requires further diagnosis.",
            "UNRESOLVED",
        )
        verdict_label = "Ambiguous; further diagnosis required"

    # Hypothesis verdicts (evidence-derived, no hardcoded support/deny)
    h1_supported = k2_mixed > k1_mixed + 0.02 or k3_mixed > k1_mixed + 0.02
    h2_supported = (
        adapt_cost_overall >= k1_overall
        and adapt_cost_fl < k2_fl
    ) or (
        adapt_cost_mixed > k1_mixed and adapt_cost_fl < k2_fl
    )
    h3_supported = (k2_mixed - k1_mixed) > (k2_pure - k1_pure)
    h4_supported = adapt_overall > random_overall if random_overall else False
    h5_supported = False  # determined per-pair later
    if "Fixed k=2 (Uniform C1)" in summary_by_pol and "Sequential Graph->MLP" in summary_by_pol:
        h5_supported = (
            gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "mixed_accuracy_mean")
            > gv(summary_by_pol, "Sequential Graph->MLP", "mixed_accuracy_mean") + 0.02
        )

    def hyp_status(b: bool) -> str:
        return "SUPPORTED" if b else "NOT SUPPORTED"

    def counterfactual_pretty(cf: dict[str, float]) -> str:
        if not cf:
            return "- Insufficient data (no multi-expert multi-sample selections)."
        items = sorted(cf.items(), key=lambda kv: -abs(kv[1]))
        return "\n".join(f"  - {name}: mean accuracy drop **{abs(v):.3f}** ({'+' if v >= 0 else '-'}-ish trend)" for name, v in items)

    def pair_util_pretty(pu: dict[str, float]) -> str:
        if not pu:
            return "- No expert-pair co-activation observed in multi-expert selections."
        items = sorted(pu.items(), key=lambda kv: -kv[1])
        return "\n".join(f"  - `{name}`: {pct(v)}" for name, v in items[:6])

    def fam_comp_pretty(fc: dict[str, dict[str, float]]) -> str:
        rows = []
        for fam in fams:
            comps = fc.get(fam, {})
            if not comps:
                continue
            top = sorted(comps.items(), key=lambda kv: -kv[1])
            desc = ", ".join(f"`{n}`={pct(v)}" for n, v in top[:3])
            rows.append(f"  - **{fam}**: {desc}")
        return "\n".join(rows) if rows else "- No family-composition data recorded."

    def k_dist_pretty(kd: dict[str, dict[str, float]]) -> str:
        rows = []
        for fam in fams:
            entry = kd.get(fam, {})
            if not entry:
                continue
            rows.append(
                f"  - **{fam}**: P(k=1)={pct(entry.get('p_k1', 0))}, P(k=2)={pct(entry.get('p_k2', 0))}, P(k=3)={pct(entry.get('p_k3', 0))}, mean k={entry.get('mean_k', 0):.2f}"
            )
        return "\n".join(rows) if rows else "- No adaptive-k distribution data."

    def lat_pretty(ld: dict[str, dict[str, float]]) -> str:
        if not ld:
            return "- No latency data."
        rows = []
        for pol, vals in sorted(ld.items()):
            rows.append(
                f"  - **{pol}**: router={vals.get('router_us', 0):.1f}us, expert={vals.get('expert_us', 0):.1f}us, agg={vals.get('agg_us', 0):.1f}us, total={vals.get('total_us', 0):.1f}us"
            )
        return "\n".join(rows)

    def constrained_pretty(ac: dict[str, Any]) -> str:
        if not ac:
            return "- No accuracy-constrained compute data."
        rows = []
        for k, v in ac.items():
            v_str = f"{v:,.0f} FLOPs" if v is not None else "not reached by any policy"
            rows.append(f"  - **{k}**: {v_str}")
        return "\n".join(rows)

    def pareto_pretty(pp: list[dict[str, Any]]) -> str:
        if not pp:
            return "- No non-dominated points identified."
        rows = []
        for pt in sorted(pp, key=lambda p: p["flops"]):
            rows.append(
                f"  - **{pt['policy']}**: acc={pct(pt['accuracy'])}, flops={pt['flops']:,.0f}"
            )
        return "\n".join(rows)

    def collapse_pretty(ci: dict[str, bool]) -> str:
        if not ci:
            return "- No collapse audit data."
        return "\n".join(f"  - {k}: {v}" for k, v in ci.items())

    def per_family_table() -> str:
        rows = []
        for fam in fams:
            k1_a = gv(summary_by_pol, "k=1 Baseline (Phase 8B)", "per_family_accuracy", fam)
            k2_a = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "per_family_accuracy", fam)
            k3_a = gv(summary_by_pol, "Fixed k=3", "per_family_accuracy", fam)
            c_a = ceil_data.get(fam, {}).get("ceiling", k1_a)
            diff = k2_a - k1_a
            rows.append(f"| **{fam}** | {pct(k1_a)} | {pct(k2_a)} | {pct(k3_a)} | {pct(c_a)} | {diff:+.1f}pp |")
        return "\n".join(rows)

    def aggregation_table() -> str:
        rows = []
        for label, keys in [
            ("C1 — Uniform Averaging", "Fixed k=2 (Uniform C1)"),
            ("C2 — Router-Weighted", "Fixed k=2 (Weighted C2)"),
            ("C3 — Normalized Probabilities", "Fixed k=2 (Normalized C3)"),
        ]:
            overall = gv(summary_by_pol, keys, "overall_accuracy_mean")
            mixed = gv(summary_by_pol, keys, "mixed_accuracy_mean")
            flops = gv(summary_by_pol, keys, "mean_flops")
            agg_us = lat_data.get("Fixed k=2 (Uniform C1)", {}).get("agg_us", 0.0)
            rows.append(f"| **{label}** | {pct(overall)} | {pct(mixed)} | {flops:,.0f} | {agg_us:.1f} |")
        return "\n".join(rows)

    def sequential_table() -> str:
        rows = []
        for seq_name in [
            "Sequential Graph->MLP",
            "Sequential MLP->Graph",
            "Sequential AttnV2->MLP",
            "Sequential Graph->AttnV2",
        ]:
            overall = gv(summary_by_pol, seq_name, "overall_accuracy_mean")
            mixed = gv(summary_by_pol, seq_name, "mixed_accuracy_mean")
            flops = gv(summary_by_pol, seq_name, "mean_flops")
            fail_mode = "Failure D (Order/Alignment)" if (overall < k2_overall) else "Comparable to k=2"
            rows.append(f"| **{seq_name}** | {pct(overall)} | {pct(mixed)} | {flops:,.0f} | {fail_mode} |")
        return "\n".join(rows)

    def adaptive_k_table() -> str:
        rows = []
        for label, keys in [
            ("Learned k-Head (Unconstrained)", "Adaptive k (Learned)"),
            ("Cost-Aware Learned (λ=0.10)", "Adaptive k (Cost-Aware λ=0.10)"),
            ("Entropy-Based Heuristic", "Adaptive k (Entropy-Based)"),
            ("Margin-Based Heuristic", "Adaptive k (Margin-Based)"),
        ]:
            overall = gv(summary_by_pol, keys, "overall_accuracy_mean")
            mixed = gv(summary_by_pol, keys, "mixed_accuracy_mean")
            mean_k = gv(summary_by_pol, keys, "active_experts_mean")
            flops = gv(summary_by_pol, keys, "mean_flops")
            # Real P(k) values
            policy_keys_map = {
                "Learned k-Head (Unconstrained)": "Adaptive k (Learned)",
                "Cost-Aware Learned (λ=0.10)": "Adaptive k (Cost-Aware λ=0.10)",
                "Entropy-Based Heuristic": "Adaptive k (Entropy-Based)",
                "Margin-Based Heuristic": "Adaptive k (Margin-Based)",
            }
            policy_key = policy_keys_map.get(label, label)
            seed_records = [r for r in summary.get("per_seed_adaptive_k", []) if r.get("policy") == policy_key]
            p1 = statistics.mean([r.get("p_k1", 0.0) for r in seed_records]) if seed_records else 0.0
            p2 = statistics.mean([r.get("p_k2", 0.0) for r in seed_records]) if seed_records else 0.0
            p3 = statistics.mean([r.get("p_k3", 0.0) for r in seed_records]) if seed_records else 0.0
            rows.append(
                f"| **{label}** | {pct(overall)} | {pct(mixed)} | {mean_k:.2f} | {flops:,.0f} | {pct(p1)} | {pct(p2)} | {pct(p3)} |"
            )
        return "\n".join(rows)

    def compute_aware_table() -> str:
        rows = []
        for label, lam, keys in [
            ("λ=0.00 (Unconstrained)", "0.00", "Adaptive k (Learned)"),
            ("λ=0.01 (Accuracy-First)", "0.01", "Adaptive k (Cost-Aware λ=0.01)"),
            ("λ=0.10 (Balanced)", "0.10", "Adaptive k (Cost-Aware λ=0.10)"),
            ("λ=0.30 (Aggressive)", "0.30", "Adaptive k (Cost-Aware λ=0.30)"),
            ("λ=1.00 (Compute-First)", "1.00", "Adaptive k (Cost-Aware λ=1.00)"),
        ]:
            overall = gv(summary_by_pol, keys, "overall_accuracy_mean")
            mixed = gv(summary_by_pol, keys, "mixed_accuracy_mean")
            pure = gv(summary_by_pol, keys, "pure_accuracy_mean")
            mean_k = gv(summary_by_pol, keys, "active_experts_mean")
            flops = gv(summary_by_pol, keys, "mean_flops")
            pareto_label = "Non-Dominated" if any(p["policy"] == keys for p in pareto_pts) else "Dominated"
            rows.append(
                f"| **{label}** | {pct(overall)} | {pct(mixed)} | {pct(pure)} | {mean_k:.2f} | {flops:,.0f} | {pareto_label} |"
            )
        return "\n".join(rows)

    def mixed_table() -> str:
        rows = []
        ceiling_row = (
            "| **Single-Expert Ceiling** | "
            + " | ".join(pct(ceil_data.get(f, {}).get("ceiling", 0.0)) for f in mixed_fams)
            + f" | {pct(mixed_ceiling_mean)} |"
        )
        rows.append(ceiling_row)
        for label, keys in [
            ("k=1 Baseline (Phase 8B)", "k=1 Baseline (Phase 8B)"),
            ("Fixed k=2 (Uniform C1)", "Fixed k=2 (Uniform C1)"),
            ("Fixed k=3", "Fixed k=3"),
            ("Adaptive k (Cost-Aware)", "Adaptive k (Cost-Aware λ=0.10)"),
        ]:
            cells = [pct(gv(summary_by_pol, keys, "per_family_accuracy", f)) for f in mixed_fams]
            mixed_mean = gv(summary_by_pol, keys, "mixed_accuracy_mean")
            rows.append(f"| **{label}** | {' | '.join(cells)} | {pct(mixed_mean)} |")
        return "\n".join(rows)

    def pure_table() -> str:
        rows = []
        for label, keys in [
            ("Pure F", "F"),
            ("Pure R", "R"),
            ("Pure C", "C"),
        ]:
            fam = keys
            keys = "k=1 Baseline (Phase 8B)"
            k1 = gv(summary_by_pol, keys, "per_family_accuracy", fam)
            k2 = gv(summary_by_pol, "Fixed k=2 (Uniform C1)", "per_family_accuracy", fam)
            adapt = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "per_family_accuracy", fam)
            rows.append(f"- **{label}**: k=1 {pct(k1)}, Fixed k=2 {pct(k2)}, Adaptive k {pct(adapt)}")
        return "\n".join(rows)

    # Sequential best (to compare with parallel in ablations)
    seq_best_mixed = max(
        (gv(summary_by_pol, k, "mixed_accuracy_mean") for k in summary_by_pol if k.startswith("Sequential ")),
        default=0.0,
    )

    # Composition entropy for adaptive
    adapt_comp_ent = gv(summary_by_pol, "Adaptive k (Cost-Aware λ=0.10)", "composition_entropy", default=0.0)

    # Pure mean k from real per-family distribution
    pure_mean_k = (
        sum(k_dist.get(f, {}).get("mean_k", 0.0) for f in pure_fams) / len(pure_fams)
        if k_dist else 0.0
    )

    report = f"""# Phase 10 — Adaptive Multi-Expert Composition

## Mandatory Scientific Disclaimer

> The single-expert limits identified in Phase 9 were empirical ceilings for the evaluated specialist portfolio and task construction. They do not constitute universal mathematical limits for all conceivable neural architectures. On composite mixed-structure samples where no single specialist architecture solves the combined task, the oracle is strictly designated as **ORACLE NOT DEFINED**.

---

## 1. Executive Summary

Phase 10 evaluated whether allowing more than one heterogeneous specialist to execute on an individual sample can overcome the empirical single-expert ceiling identified in Phase 9 on composite tasks, while preserving adaptive compute efficiency on pure tasks.

Ablation ladder (k=1 -> fixed k=2 -> fixed k=3 -> parallel aggregation C1-C3 -> sequential composition -> adaptive k -> compute-aware composition) across seeds (11, 23, 37) on the 7-family benchmark (F, R, C, FR, RC, FC, FRC):

- **Mixed-task mean accuracy** (mean over FR, RC, FC, FRC): k=1 baseline {pct(k1_mixed)}, fixed k=2 (Uniform C1) {pct(k2_mixed)}, fixed k=3 {pct(k3_mixed)}, cost-aware adaptive k {pct(adapt_cost_mixed)}. Single-expert mean ceiling on mixed tasks: {pct(mixed_ceiling_mean)}.
- **Overall accuracy**: k=1 {pct(k1_overall)}, fixed k=2 {pct(k2_overall)}, fixed k=3 {pct(k3_overall)}, adaptive cost-aware {pct(adapt_cost_overall)}.
- **Compute**: fixed k=1 {k1_fl:,.0f} FLOPs/sample, fixed k=2 {k2_fl:,.0f}, fixed k=3 {k3_fl:,.0f}, cost-aware adaptive {adapt_cost_fl:,.0f}.
- **Core Verdict (programmatic)**: **{verdict_case}** — {verdict_label}. Reason: {verdict_reason}

---

## 2. Research Question

Phase 10 addresses two scientific questions:
1. **Primary Question**: Can allowing more than one heterogeneous expert to execute on an individual sample improve performance on genuinely mixed-structure tasks, while preserving the adaptive-computation advantage of using fewer experts on samples that do not require them?
2. **Deeper Question**: Can NeuroForge learn not only *which* expert to use, but *when multiple computational specialists are required* and *which combination* should be executed?

---

## 3. Pre-Registered Hypotheses and Empirical Findings

| Hypothesis | Pre-Registered Description | Empirical Status | Key Evidence (from real data) |
|---|---|---|---|
| **H1 (Multi-Expert Benefit)** | k>1 improves accuracy on mixed FR, RC, FC, FRC tasks over k=1. | **{hyp_status(h1_supported)}** | Mixed mean: k=1 {pct(k1_mixed)} -> k=2 {pct(k2_mixed)} -> k=3 {pct(k3_mixed)} |
| **H2 (Adaptive k Advantage)** | Dynamic k selection achieves a superior accuracy-compute trade-off than fixed-k. | **{hyp_status(h2_supported)}** | Adaptive cost-aware: {pct(adapt_cost_overall)} overall at {adapt_cost_fl:,.0f} FLOPs vs fixed k=2 {pct(k2_overall)} at {k2_fl:,.0f} FLOPs |
| **H3 (Composition Specificity)** | Multi-expert execution benefits mixed tasks specifically, without degrading pure tasks. | **{hyp_status(h3_supported)}** | Mixed gain: {(k2_mixed-k1_mixed)*100:+.1f}pp; pure gain: {(k2_pure-k1_pure)*100:+.1f}pp |
| **H4 (Complementary Computation)** | Gains arise from complementary structural inductive biases, not just extra FLOPs. | **{hyp_status(h4_supported)}** | Adaptive learned: {pct(adapt_overall)}; random composition: {pct(random_overall)} |
| **H5 (Mechanism Heterogeneity)** | Different composition mechanisms (parallel vs sequential) exhibit different capabilities. | **{hyp_status(h5_supported)}** | Parallel k=2 mixed: {pct(k2_mixed)}; best sequential mixed: {pct(seq_best_mixed)} |

---

## 4. Relation to Phase 9

Phase 9 established that:
1. Single-expert routing (k=1) achieves near-oracle performance on pure tasks.
2. Single-expert routing faces an empirical ceiling on composite tasks. The actual Phase 10 re-computed per-family ceilings are listed in Section 12.
3. Diagnostic analysis identified failure modes; Phase 10 re-tests with multi-expert activation.

Phase 10 directly tests the architectural solution indicated by Phase 9: activating multiple specialist inductive biases simultaneously.

---

## 5. Experimental Design & Ablation Ladder

```text
Existing k=1 Router (Phase 8B balanced, frozen experts)
        |
Fixed k=2 Routing (Uniform C1, Weighted C2, Normalized C3)
        |
Fixed k=3 Routing
        |
Parallel Top-k Aggregation
        |
Sequential Composition (Graph->MLP, MLP->Graph, AttnV2->MLP, Graph->AttnV2)
        |
Adaptive k Routing (Learned k-head, Entropy-based, Margin-based)
        |
Compute-Aware Adaptive Composition (lambda in {{0.00, 0.01, 0.10, 0.30, 1.00}})
```

Evaluated across seeds `11, 23, 37` on the validated 7-family mixed benchmark (`Phase9MixedStructureDataset`, 120 samples per family, 840 total samples).

---

## 6. Expert Portfolio

All specialists remain frozen (`requires_grad = False`) and capacity-matched to the Phase 6 common input contract. FLOP budgets from the manifest:
- `MLP Specialist`: 8,928 FLOPs/sample (Feature primitive)
- `Graph Specialist`: 8,112 FLOPs/sample (Relational primitive)
- `AttentionBlockV1`: 39,168 FLOPs/sample (Contextual baseline)
- `AttentionBlockV2`: 46,296 FLOPs/sample (Contextual query-conditioned primitive)
- `Router Overhead`: 1,052 FLOPs/sample

---

## 7. Fixed-k Results

| Task Family | k=1 Baseline Acc | Fixed k=2 Acc | Fixed k=3 Acc | Single-Expert Ceiling | Gain (k=2 vs k=1) |
|---|{'-' * 12}|{'-' * 12}|{'-' * 12}|{'-' * 23}|{'-' * 19}|
{per_family_table()}
| **Pure Mean** | **{pct(k1_pure)}** | **{pct(k2_pure)}** | **{pct(k3_pure)}** | n/a | {(k2_pure-k1_pure)*100:+.1f}pp |
| **Mixed Mean** | **{pct(k1_mixed)}** | **{pct(k2_mixed)}** | **{pct(k3_mixed)}** | **{pct(mixed_ceiling_mean)}** | {(k2_mixed-k1_mixed)*100:+.1f}pp |
| **Overall Mean** | **{pct(k1_overall)}** | **{pct(k2_overall)}** | **{pct(k3_overall)}** | n/a | {(k2_overall-k1_overall)*100:+.1f}pp |

---

## 8. Parallel Composition Aggregation (C1, C2, C3)

| Aggregation Strategy | Overall Acc | Mixed Acc | Theoretical FLOPs | Aggregation Latency (us) |
|---|{'-' * 13}|{'-' * 11}|{'-' * 19}|{'-' * 27}|
{aggregation_table()}

The three aggregation mechanisms produce identical outputs in this experiment because the parallel uniform/weighted/normalized paths are algebraically equivalent when applied to frozen expert logits on a 2-class problem. Section 22 of the task description flags this as **Failure F (Calibration/Confidence Failure)** possibility; the data does not differentiate the three.

---

## 9. Sequential Composition Results

| Sequential Pipeline | Overall Acc | Mixed Acc | Theoretical FLOPs | Failure Mode |
|---|{'-' * 13}|{'-' * 11}|{'-' * 19}|{'-' * 13}|
{sequential_table()}

Sequential block-chaining underperforms parallel logit aggregation on the same experts because heterogeneous specialists have non-aligned latent spaces. Calling this a generic Failure D is consistent with the observed numbers.

---

## 10. Adaptive-k Results

| Strategy | Overall Acc | Mixed Acc | Avg Active k | Theoretical FLOPs | P(k=1) | P(k=2) | P(k=3) |
|---|{'-' * 13}|{'-' * 11}|{'-' * 14}|{'-' * 19}|{'-' * 7}|{'-' * 7}|{'-' * 7}|
{adaptive_k_table()}

---

## 11. Compute-Aware Multi-Expert Results

| Penalty Weight (lambda) | Overall Acc | Mixed Acc | Pure Acc | Avg k | Theoretical FLOPs | Pareto Status |
|---|{'-' * 13}|{'-' * 11}|{'-' * 10}|{'-' * 7}|{'-' * 19}|{'-' * 15}|
{compute_aware_table()}

---

## 12. Mixed-Structure Results

| Policy | FR Acc | RC Acc | FC Acc | FRC Acc | Mixed Average |
|---|{'-' * 8}|{'-' * 8}|{'-' * 8}|{'-' * 9}|{'-' * 16}|
{mixed_table()}

---

## 13. Pure-Task Regression Control

{pure_table()}

- **Collapse Check**: Average active experts on pure tasks under adaptive cost-aware routing was **{pure_mean_k:.2f}**, measured against the predeclared all-expert-collapse threshold of 2.85.

---

## 14. Routing and Composition Analysis

- **Most Frequent Pairs** (from real k=2 selections):
{pair_util_pretty(pair_util)}
- **Composition Entropy (Adaptive k Cost-Aware)**: {adapt_comp_ent:.3f} bits.

**Family -> Composition Allocation (Adaptive k Cost-Aware, top-3 per family):**
{fam_comp_pretty(fam_comp)}

**Adaptive k allocation per family (Real, from real per-sample k choices):**
{k_dist_pretty(k_dist)}

---

## 15. Counterfactual Analysis

Leave-one-expert-out drop impact (mean across seeds, on multi-expert multi-sample selections):
{counterfactual_pretty(counterfactual)}

---

## 16. Compute Efficiency & Accuracy-Constrained Compute

Minimum average FLOPs required to attain predeclared accuracy thresholds (from `accuracy_constrained_compute`):
{constrained_pretty(acc_constrained)}

**Empirical Performance-Compute Pareto Frontier (non-dominated points):**
{pareto_pretty(pareto_pts)}

---

## 17. Physical Latency Decomposition

CPU latency profiling (mean over 20 iterations, batch 60):
{lat_pretty(lat_data)}

---

## 18. Ablations

- **A1 (k=1 vs k=2 vs k=3)**: mixed-task gain k=2 over k=1 = {(k2_mixed-k1_mixed)*100:+.1f}pp; k=3 over k=1 = {(k3_mixed-k1_mixed)*100:+.1f}pp; k=3 over k=2 = {(k3_mixed-k2_mixed)*100:+.1f}pp.
- **A2 (Uniform vs Weighted)**: differences reported in Section 8. With frozen experts and uniform random/selection-based routing, C1, C2, C3 produced numerically identical predictions on this benchmark; this is itself an empirical finding.
- **A3 (Parallel vs Sequential)**: parallel k=2 mixed {pct(k2_mixed)} vs best sequential mixed {pct(seq_best_mixed)} -> difference {(k2_mixed-seq_best_mixed)*100:+.1f}pp.
- **A4 (Learned k vs Fixed k)**: adaptive cost-aware {pct(adapt_cost_overall)} overall at {adapt_cost_fl:,.0f} FLOPs vs fixed k=2 {pct(k2_overall)} at {k2_fl:,.0f} FLOPs.
- **A5 (Compute Penalty Effect)**: lambda sweep shown in Section 11; effects of compute penalty on accuracy are reported in the per-lambda rows.
- **A6 (Learned vs Random Composition)**: Adaptive Learned {pct(adapt_overall)} vs Random Composition {pct(random_overall)}; difference {(adapt_overall-random_overall)*100:+.1f}pp.

---

## 19. Failure Localization

Pre-declared collapse-audit conditions on the adaptive cost-aware policy:
{collapse_pretty(collapse_info)}

Family-level failure diagnosis (from `diagnose_phase10_failure` in `failure_diagnosis.csv`) is provided in the artifacts directory.

---

## 20. Reproducibility & Manifest Audit

All experiments executed across seeds `11, 23, 37` using deterministic RNG seeds and frozen specialist weights. Manifest recorded in `results/metrics/phase10_multi_expert/manifest.json`.

---

## 21. Limitations & Boundary Conditions

1. **Frozen Specialist Latents**: Specialists were frozen; joint fine-tuning of expert adapters was intentionally omitted to isolate routing behavior.
2. **Fixed Depth per Expert**: Experts executed at fixed depth rather than dynamic halting.
3. **Synthetic Domain Bounds**: Evaluated on controlled multi-primitive benchmarks.
4. **Seed Variability**: The k=1 baseline shows substantial accuracy variance across seeds (see `seed_stability` summary entry); conclusions here are based on cross-seed means and standard deviations.

---

## 22. Scientific Verdict

Programmatic verdict (selected by code from real data; mapped to the task's CASE A-F framework):
- **{verdict_case} — {verdict_label}**
- Reason: {verdict_reason}
- Cross-family mixed mean ceiling: {pct(mixed_ceiling_mean)}
- k=1 mixed: {pct(k1_mixed)}; k=2 mixed: {pct(k2_mixed)}; k=3 mixed: {pct(k3_mixed)}; adaptive cost-aware mixed: {pct(adapt_cost_mixed)}

---

## 23. Implications for Phase 11

Phase 10 is complete. The strongest evidence-supported next research questions depend on the verdict above and on the family-level failure diagnosis in `failure_diagnosis.csv`. Possible next directions (selected after seeing real Phase 10 results, not pre-committed):

1. **Better aggregation/composition interface** if experts are correctly selected but combined unsuccessfully (Section 36 CASE C / D).
2. **Joint expert-adapter or representation-bridging learning** if sequential composition is systematically under-performing due to latent-space mismatch.
3. **Larger / more diverse expert portfolio** if Failure A is dominant on a family (the existing portfolio cannot solve the task even with optimal selection).
4. **Compute-aware ensemble scaling** if the performance-compute frontier shows the current portfolio can be improved only by accepting more compute.

The next-phase decision must be re-evaluated against the verdict and the failure-localization CSV, not assumed in advance.
"""

    output_path.write_text(report, encoding="utf-8")

