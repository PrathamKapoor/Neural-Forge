"""Phase 7: Oracle routing and fixed-best selection experiment runner."""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neuroforge.datasets import Phase6ExpertSpecializationDataset, Phase7MixedDataset
from neuroforge.evaluation.phase7_metrics import calculate_oracle_regret, calculate_routing_statistics
from neuroforge.evaluation.specialization import parameter_count, select_capacity_phase6
from neuroforge.models import StandaloneSpecialist
from neuroforge.visualization.phase7_plots import generate_phase7_figures


def _train_expert(
    architecture: str,
    target_family: str,
    depth: int,
    seed: int,
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int,
) -> tuple[StandaloneSpecialist, list[float]]:
    """Train a specialist candidate on its designated task family."""
    torch.manual_seed(seed)
    model = StandaloneSpecialist(architecture, depth=depth)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    train_ds = Phase6ExpertSpecializationDataset(target_family, "train", train_samples, seed)
    val_ds = Phase6ExpertSpecializationDataset(target_family, "validation", val_samples, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = -1.0
    best_state = None
    curve: list[float] = []

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            loss = F.cross_entropy(model(batch["features"]), batch["target"])
            loss.backward()
            optim.step()

        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for batch in val_loader:
                corr += int((model(batch["features"]).argmax(1) == batch["target"]).sum())
                tot += len(batch["features"])
        val_acc = corr / tot
        curve.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, curve


def _measure_flops_for_expert(model: StandaloneSpecialist, sample: torch.Tensor) -> float:
    costs = [float(2 * sample.shape[1] * 8 * 24)]
    with torch.no_grad():
        state = model.encoder(sample[:1])
        for block in model.blocks:
            if model.architecture == "attention_v2":
                state, cost = block(state, sample[:1])
            else:
                state, cost = block(state)
            costs.append(float(cost))
    return sum(costs)


def _benchmark_latencies(
    experts: dict[str, StandaloneSpecialist],
    batch: torch.Tensor,
    oracle_routes: list[str],
) -> dict[str, Any]:
    """Measure isolated expert latencies and grouped oracle dispatch overhead."""
    latencies: dict[str, float] = {}

    # Isolated expert latency (batch_size = 60)
    for name, exp in experts.items():
        exp.eval()
        with torch.no_grad():
            for _ in range(3):
                exp(batch)
            started = time.perf_counter()
            for _ in range(15):
                exp(batch)
            elapsed = time.perf_counter() - started
            latencies[f"isolated_{name}_ms"] = (elapsed / 15) * 1000.0

    # Grouped oracle dispatch
    exp_order = ("mlp", "graph", "attention_v2")
    idx_groups = {e: [i for i, r in enumerate(oracle_routes) if r == e] for e in exp_order}

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            for e in exp_order:
                if idx_groups[e]:
                    experts[e](batch[idx_groups[e]])

    overhead_times = []
    forward_times = []
    total_times = []

    with torch.no_grad():
        for _ in range(15):
            t0 = time.perf_counter()
            # 1. Grouping / slicing
            sub_batches = {e: batch[idx_groups[e]] for e in exp_order if idx_groups[e]}
            t1 = time.perf_counter()

            # 2. Sub-batch expert execution
            outs = {}
            for e, sb in sub_batches.items():
                outs[e] = experts[e](sb)
            t2 = time.perf_counter()

            # 3. Reassembly
            combined = torch.empty(len(batch), 2, device=batch.device)
            for e, sb_out in outs.items():
                combined[idx_groups[e]] = sb_out
            t3 = time.perf_counter()

            overhead_times.append((t1 - t0) + (t3 - t2))
            forward_times.append(t2 - t1)
            total_times.append(t3 - t0)

    dispatch_overhead_ms = (statistics.mean(overhead_times)) * 1000.0
    oracle_exec_ms = (statistics.mean(forward_times)) * 1000.0
    total_oracle_ms = (statistics.mean(total_times)) * 1000.0

    latencies["oracle_dispatch_overhead_ms"] = dispatch_overhead_ms
    latencies["oracle_execution_latency_ms"] = oracle_exec_ms
    latencies["oracle_total_end_to_end_ms"] = total_oracle_ms

    return latencies


def run_phase7_oracle_routing(
    output_dir: str | Path,
    seeds: tuple[int, ...] = (11, 23, 37),
    train_samples_per_family: int = 360,
    val_samples_per_family: int = 120,
    test_samples_per_family: int = 240,
    epochs: int = 25,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute Phase 7 Oracle Routing and Fixed-Best Selection experiment."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    selected = select_capacity_phase6()
    condition_names = (
        "Fixed MLP",
        "Fixed Graph",
        "Fixed Attention V1",
        "Fixed Attention V2",
        "Random Router",
        "Oracle Router",
        "Oracle Without V2",
    )

    per_seed_results: dict[str, list[dict[str, Any]]] = {c: [] for c in condition_names}
    history: dict[str, Any] = {}
    routing_stats_by_seed: list[dict[str, Any]] = []
    regret_stats_by_seed: list[dict[str, Any]] = []
    timing_stats_by_seed: list[dict[str, Any]] = []

    # Track isolated FLOPs and parameters
    expert_flops: dict[str, float] = {}
    expert_params: dict[str, int] = {}

    for seed in seeds:
        torch.manual_seed(seed)

        # 1. Train the 4 candidate specialists
        mlp_exp, mlp_curve = _train_expert(
            "mlp", "feature", selected.depths["mlp"], seed,
            train_samples_per_family, val_samples_per_family, epochs, batch_size,
        )
        graph_exp, graph_curve = _train_expert(
            "graph", "relational", selected.depths["graph"], seed,
            train_samples_per_family, val_samples_per_family, epochs, batch_size,
        )
        attn_exp, attn_curve = _train_expert(
            "attention", "contextual", selected.depths["attention"], seed,
            train_samples_per_family, val_samples_per_family, epochs, batch_size,
        )
        v2_exp, v2_curve = _train_expert(
            "attention_v2", "contextual", selected.depths["attention_v2"], seed,
            train_samples_per_family, val_samples_per_family, epochs, batch_size,
        )

        history[f"mlp_expert__{seed}"] = mlp_curve
        history[f"graph_expert__{seed}"] = graph_curve
        history[f"attention_v1_expert__{seed}"] = attn_curve
        history[f"attention_v2_expert__{seed}"] = v2_curve

        experts = {
            "mlp": mlp_exp,
            "graph": graph_exp,
            "attention": attn_exp,
            "attention_v2": v2_exp,
        }

        # 2. Build mixed evaluation population (sample-level interleaved)
        mixed_test = Phase7MixedDataset(
            split="test",
            samples_per_family=test_samples_per_family,
            seed=seed,
        )
        features = mixed_test.features
        targets = mixed_test.targets
        families = mixed_test.families_list

        # Record FLOPs and parameters once
        if not expert_flops:
            for name, exp in experts.items():
                expert_flops[name] = _measure_flops_for_expert(exp, features[:1])
                expert_params[name] = parameter_count(exp)

        # 3. Model predictions across entire mixed population
        with torch.no_grad():
            preds = {name: exp(features).argmax(1) for name, exp in experts.items()}

        # 4. Routing condition predictions
        # Condition E: Random Router (seed-deterministic)
        g_rand = torch.Generator().manual_seed(seed + 77_000)
        arch_tuple = ("mlp", "graph", "attention", "attention_v2")
        rand_choices = [
            arch_tuple[int(torch.randint(4, (1,), generator=g_rand).item())]
            for _ in range(len(mixed_test))
        ]
        rand_preds = torch.tensor(
            [preds[rand_choices[i]][i].item() for i in range(len(mixed_test))],
            dtype=torch.long,
        )

        # Condition F: Oracle Router
        oracle_map = {"feature": "mlp", "relational": "graph", "contextual": "attention_v2"}
        oracle_choices = [oracle_map[f] for f in families]
        oracle_preds = torch.tensor(
            [preds[oracle_choices[i]][i].item() for i in range(len(mixed_test))],
            dtype=torch.long,
        )

        # Condition G: Oracle Without V2
        oracle_no_v2_map = {"feature": "mlp", "relational": "graph", "contextual": "attention"}
        oracle_no_v2_choices = [oracle_no_v2_map[f] for f in families]
        oracle_no_v2_preds = torch.tensor(
            [preds[oracle_no_v2_choices[i]][i].item() for i in range(len(mixed_test))],
            dtype=torch.long,
        )

        predictions = {
            "Fixed MLP": preds["mlp"],
            "Fixed Graph": preds["graph"],
            "Fixed Attention V1": preds["attention"],
            "Fixed Attention V2": preds["attention_v2"],
            "Random Router": rand_preds,
            "Oracle Router": oracle_preds,
            "Oracle Without V2": oracle_no_v2_preds,
        }

        # 5. Measure latency and dispatch overhead
        first_batch = features[:batch_size]
        first_oracle_routes = oracle_choices[:batch_size]
        latencies = _benchmark_latencies(experts, first_batch, first_oracle_routes)
        timing_stats_by_seed.append(latencies)

        # Tracemalloc memory
        tracemalloc.start()
        for _ in range(5):
            for e in ("mlp", "graph", "attention_v2"):
                idx = [i for i, r in enumerate(first_oracle_routes) if r == e]
                if idx:
                    experts[e](first_batch[idx])
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compute per-condition metrics
        for cond_name, p in predictions.items():
            overall_acc = float((p == targets).float().mean().item())
            fam_accs = {}
            for fam in Phase7MixedDataset.families:
                mask = torch.tensor([f == fam for f in families], dtype=torch.bool)
                fam_accs[fam] = float((p[mask] == targets[mask]).float().mean().item())
            macro_acc = statistics.mean(fam_accs.values())

            # Assigned FLOPs and parameters
            if cond_name == "Fixed MLP":
                c_flops = expert_flops["mlp"]
                c_params = expert_params["mlp"]
                c_lat = latencies["isolated_mlp_ms"]
            elif cond_name == "Fixed Graph":
                c_flops = expert_flops["graph"]
                c_params = expert_params["graph"]
                c_lat = latencies["isolated_graph_ms"]
            elif cond_name == "Fixed Attention V1":
                c_flops = expert_flops["attention"]
                c_params = expert_params["attention"]
                c_lat = latencies["isolated_attention_ms"]
            elif cond_name == "Fixed Attention V2":
                c_flops = expert_flops["attention_v2"]
                c_params = expert_params["attention_v2"]
                c_lat = latencies["isolated_attention_v2_ms"]
            elif cond_name == "Random Router":
                c_flops = sum(expert_flops[rc] for rc in rand_choices) / len(rand_choices)
                c_params = sum(expert_params.values()) / len(expert_params)
                c_lat = sum(latencies[f"isolated_{k}_ms"] for k in experts) / len(experts)
            elif cond_name == "Oracle Router":
                c_flops = sum(expert_flops[oc] for oc in oracle_choices) / len(oracle_choices)
                c_params = sum(expert_params[oc] for oc in oracle_choices) / len(oracle_choices)
                c_lat = latencies["oracle_total_end_to_end_ms"]
            elif cond_name == "Oracle Without V2":
                c_flops = sum(expert_flops[oc] for oc in oracle_no_v2_choices) / len(oracle_no_v2_choices)
                c_params = sum(expert_params[oc] for oc in oracle_no_v2_choices) / len(oracle_no_v2_choices)
                c_lat = (latencies["isolated_mlp_ms"] + latencies["isolated_graph_ms"] + latencies["isolated_attention_ms"]) / 3.0
            else:
                c_flops = 0.0
                c_params = 0
                c_lat = 0.0

            per_seed_results[cond_name].append({
                "seed": seed,
                "overall_accuracy": overall_acc,
                "family_accuracies": fam_accs,
                "macro_accuracy": macro_acc,
                "estimated_forward_flops": c_flops,
                "parameters": c_params,
                "batch_latency_ms": c_lat,
                "peak_python_memory_bytes": float(peak_mem),
            })

        # Routing and Regret diagnostics for this seed
        routing_stat = calculate_routing_statistics(oracle_choices, families)
        routing_stats_by_seed.append(routing_stat)

        # Regret vs Best Fixed (AttentionBlockV2 or MLP)
        regret_v2 = calculate_oracle_regret(oracle_preds, preds["attention_v2"], targets, families)
        regret_stats_by_seed.append(regret_v2)

    # 6. Aggregate across seeds
    conditions_summary: dict[str, Any] = {}
    for cond_name, rows in per_seed_results.items():
        mean_overall = statistics.mean(r["overall_accuracy"] for r in rows)
        std_overall = statistics.stdev(r["overall_accuracy"] for r in rows) if len(rows) > 1 else 0.0
        mean_macro = statistics.mean(r["macro_accuracy"] for r in rows)
        std_macro = statistics.stdev(r["macro_accuracy"] for r in rows) if len(rows) > 1 else 0.0

        fam_summaries = {}
        for fam in Phase7MixedDataset.families:
            f_vals = [r["family_accuracies"][fam] for r in rows]
            fam_summaries[fam] = {
                "mean": statistics.mean(f_vals),
                "std": statistics.stdev(f_vals) if len(f_vals) > 1 else 0.0,
            }

        conditions_summary[cond_name] = {
            "overall_accuracy": {"mean": mean_overall, "std": std_overall},
            "macro_accuracy": {"mean": mean_macro, "std": std_macro},
            "family_accuracies": fam_summaries,
            "compute": {
                "parameters": rows[0]["parameters"],
                "estimated_forward_flops": rows[0]["estimated_forward_flops"],
            },
            "latency": {
                "batch_latency_ms": statistics.mean(r["batch_latency_ms"] for r in rows),
                "samples_per_second": (batch_size * 1000.0) / statistics.mean(r["batch_latency_ms"] for r in rows),
                "peak_python_memory_bytes": statistics.mean(r["peak_python_memory_bytes"] for r in rows),
            },
            "per_seed": rows,
        }

    # 7. Identify Best Fixed Baseline
    fixed_names = ("Fixed MLP", "Fixed Graph", "Fixed Attention V1", "Fixed Attention V2")
    best_fixed_acc_name = max(fixed_names, key=lambda n: conditions_summary[n]["overall_accuracy"]["mean"])
    best_fixed_macro_name = max(fixed_names, key=lambda n: conditions_summary[n]["macro_accuracy"]["mean"])
    lowest_compute_name = min(fixed_names, key=lambda n: conditions_summary[n]["compute"]["estimated_forward_flops"])
    lowest_latency_name = min(fixed_names, key=lambda n: conditions_summary[n]["latency"]["batch_latency_ms"])

    best_fixed_summary = {
        "best_by_accuracy": {
            "condition": best_fixed_acc_name,
            "overall_accuracy": conditions_summary[best_fixed_acc_name]["overall_accuracy"]["mean"],
        },
        "best_by_macro_accuracy": {
            "condition": best_fixed_macro_name,
            "macro_accuracy": conditions_summary[best_fixed_macro_name]["macro_accuracy"]["mean"],
        },
        "lowest_compute": {
            "condition": lowest_compute_name,
            "estimated_forward_flops": conditions_summary[lowest_compute_name]["compute"]["estimated_forward_flops"],
        },
        "lowest_latency": {
            "condition": lowest_latency_name,
            "batch_latency_ms": conditions_summary[lowest_latency_name]["latency"]["batch_latency_ms"],
        },
    }

    # 8. Oracle Advantage vs every fixed expert
    oracle_summary = conditions_summary["Oracle Router"]
    oracle_mean_acc = oracle_summary["overall_accuracy"]["mean"]
    oracle_flops = oracle_summary["compute"]["estimated_forward_flops"]
    oracle_lat = oracle_summary["latency"]["batch_latency_ms"]

    advantages: dict[str, Any] = {}
    for fn in fixed_names:
        f_data = conditions_summary[fn]
        f_acc = f_data["overall_accuracy"]["mean"]
        f_flops = f_data["compute"]["estimated_forward_flops"]
        f_lat = f_data["latency"]["batch_latency_ms"]

        advantages[fn] = {
            "accuracy_advantage_delta": oracle_mean_acc - f_acc,
            "accuracy_advantage_pct": ((oracle_mean_acc - f_acc) / f_acc) * 100.0,
            "compute_advantage_flops": f_flops - oracle_flops,
            "compute_advantage_reduction_pct": ((f_flops - oracle_flops) / f_flops) * 100.0 if f_flops > oracle_flops else 0.0,
            "latency_advantage_ms": f_lat - oracle_lat,
        }

    # Average routing statistics across seeds
    avg_util = {e: statistics.mean(s["utilization"][e] for s in routing_stats_by_seed) for e in selected.depths}
    avg_sel_pct = {
        fam: {
            e: statistics.mean(s["selection_matrix_proportions"][fam][e] for s in routing_stats_by_seed)
            for e in selected.depths
        }
        for fam in Phase7MixedDataset.families
    }

    routing_diagnostics = {
        "module_utilization": avg_util,
        "selection_matrix_proportions": avg_sel_pct,
        "routing_entropy_bits": statistics.mean(s["routing_entropy_bits"] for s in routing_stats_by_seed),
        "average_active_modules_per_sample": 1.0,
        "active_modules_count": 3,  # MLP, Graph, V2 (Attention V1 has zero selection)
        "attention_v1_selection": 0.0,
    }

    # Latency breakdown
    latency_breakdown = {
        "isolated_expert_latencies_ms": {
            k.replace("isolated_", "").replace("_ms", ""): statistics.mean(s[k] for s in timing_stats_by_seed)
            for k in ("isolated_mlp_ms", "isolated_graph_ms", "isolated_attention_ms", "isolated_attention_v2_ms")
        },
        "oracle_dispatch_overhead_ms": statistics.mean(s["oracle_dispatch_overhead_ms"] for s in timing_stats_by_seed),
        "oracle_execution_latency_ms": statistics.mean(s["oracle_execution_latency_ms"] for s in timing_stats_by_seed),
        "oracle_total_end_to_end_ms": statistics.mean(s["oracle_total_end_to_end_ms"] for s in timing_stats_by_seed),
    }

    # Regret summary vs Best Fixed (AttentionBlockV2)
    regret_summary = {
        "comparison_target": best_fixed_acc_name,
        "disagreement_rate": statistics.mean(r["disagreement_rate"] for r in regret_stats_by_seed),
        "net_accuracy_advantage": statistics.mean(r["net_accuracy_advantage"] for r in regret_stats_by_seed),
        "oracle_win_count_mean": statistics.mean(r["oracle_win_count"] for r in regret_stats_by_seed),
        "fixed_win_count_mean": statistics.mean(r["fixed_win_count"] for r in regret_stats_by_seed),
        "family_breakdown": {
            fam: {
                "oracle_accuracy": statistics.mean(r["family_breakdown"][fam]["oracle_accuracy"] for r in regret_stats_by_seed),
                "best_fixed_accuracy": statistics.mean(r["family_breakdown"][fam]["best_fixed_accuracy"] for r in regret_stats_by_seed),
                "accuracy_delta": statistics.mean(r["family_breakdown"][fam]["accuracy_delta"] for r in regret_stats_by_seed),
            }
            for fam in Phase7MixedDataset.families
        },
    }

    # Scientific Verdict determination based on Section 22 logic
    # Case D: Oracle improves both accuracy and compute/latency compared with the highest accuracy fixed model (V2)
    v2_acc = conditions_summary["Fixed Attention V2"]["overall_accuracy"]["mean"]
    v2_flops = conditions_summary["Fixed Attention V2"]["compute"]["estimated_forward_flops"]
    acc_improved = (oracle_mean_acc > v2_acc + 0.05)
    compute_reduced = (oracle_flops < v2_flops)

    if acc_improved and compute_reduced:
        verdict = {
            "case": "Case D",
            "description": (
                "Oracle routing improves both accuracy (+{:.1f}%) and compute efficiency "
                "(-{:.1f}% FLOPs) compared with the strongest fixed architecture (Fixed Attention V2)."
            ).format((oracle_mean_acc - v2_acc) * 100.0, ((v2_flops - oracle_flops) / v2_flops) * 100.0),
            "routing_gate_status": "OPEN FOR LEARNED-ROUTING EXPERIMENT",
            "justification": (
                "The central research question is supported: selecting the specialized expert by construction "
                "produces a definitive performance-compute Pareto advance over the best fixed architecture. "
                "Phase 8 learned routing is now scientifically warranted."
            ),
        }
    else:
        verdict = {
            "case": "Case A/B",
            "description": "Oracle routing did not establish simultaneous performance and compute advantage.",
            "routing_gate_status": "CLOSED",
            "justification": "Routing gate remains closed until a Pareto advance is observed.",
        }

    summary: dict[str, Any] = {
        "experiment": "Phase 7 Oracle Routing and Fixed-Best Selection",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "dataset": {
            "total_test_samples": test_samples_per_family * 3,
            "samples_per_family": test_samples_per_family,
            "proportions": {"feature": 1.0 / 3.0, "relational": 1.0 / 3.0, "contextual": 1.0 / 3.0},
            "contract": "Phase 6 common input contract ([12, 8] with neutral channel-4 marker)",
        },
        "capacity": {
            "depths": selected.depths,
            "parameters": expert_params,
            "max_relative_gap": selected.max_relative_gap,
        },
        "conditions": conditions_summary,
        "best_fixed_baseline": best_fixed_summary,
        "oracle_advantage": advantages,
        "oracle_routing": routing_diagnostics,
        "latency_analysis": latency_breakdown,
        "oracle_regret": regret_summary,
        "scientific_verdict": verdict,
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "Phase 7 Oracle Routing and Fixed-Best Selection",
        "seeds": seeds,
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "git_commit": None,
        "input_contract": {
            "shape": [12, 8],
            "marker_channel": 4,
            "marker_value": 1.0,
            "families": ("feature", "relational", "contextual"),
        },
        "expert_depths": selected.depths,
        "expert_parameters": expert_params,
        "verdict": verdict,
    }

    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # CSV export
    with (destination / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "seed", "condition", "overall_accuracy", "feature_accuracy",
            "relational_accuracy", "contextual_accuracy", "macro_accuracy",
            "estimated_forward_flops", "parameters", "batch_latency_ms",
        ])
        for cond_name, rows in per_seed_results.items():
            for r in rows:
                writer.writerow([
                    r["seed"], cond_name, r["overall_accuracy"],
                    r["family_accuracies"]["feature"],
                    r["family_accuracies"]["relational"],
                    r["family_accuracies"]["contextual"],
                    r["macro_accuracy"],
                    r["estimated_forward_flops"],
                    r["parameters"],
                    r["batch_latency_ms"],
                ])

    # Generate the 6 required figures
    figures_dir = Path("figures/phase7_oracle_routing")
    generate_phase7_figures(summary, figures_dir)

    return summary
