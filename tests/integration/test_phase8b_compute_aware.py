"""Integration smoke test for Phase 8B Compute-Aware Learned Routing experiment pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase8b_compute_aware import run_phase8b_compute_aware_routing


def test_phase8b_pipeline_smoke(tmp_path: Path):
    out_dir = tmp_path / "phase8b_test_run"

    # Fast smoke run with 1 seed, 2 lambdas, 1 epoch, small sample counts
    summary = run_phase8b_compute_aware_routing(
        output_dir=out_dir,
        seeds=(11,),
        lambdas=(0.0, 0.1),
        train_samples_per_family=16,
        val_samples_per_family=10,
        test_samples_per_family=16,
        expert_epochs=1,
        router_epochs=2,
        batch_size=8,
    )
    assert summary["experiment"] == "Phase 8B Compute-Aware Learned Routing"

    # 1. Verify JSON artifacts
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "history.json").exists()

    summary_loaded = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_loaded["experiment"] == "Phase 8B Compute-Aware Learned Routing"
    assert len(summary_loaded["lambda_sweep"]) == 2
    assert "pareto_analysis" in summary_loaded
    assert "historical_accounting_audit" in summary_loaded

    # 2. Verify CSV artifacts
    assert (out_dir / "source_data.csv").exists()
    assert (out_dir / "seed_results.csv").exists()
    assert (out_dir / "routing_assignments.csv").exists()
    assert (out_dir / "latency_results.csv").exists()
    assert (out_dir / "pareto_points.csv").exists()

    # 3. Verify figures directory
    fig_dir = Path("figures/phase8b_compute_aware")
    assert fig_dir.exists()
    expected_figures = [
        "accuracy_vs_lambda.png",
        "flops_vs_lambda.png",
        "accuracy_vs_flops_pareto.png",
        "expert_utilization_vs_lambda.png",
        "family_expert_routing_matrices.png",
        "routing_entropy_vs_lambda.png",
        "accuracy_vs_latency.png",
        "frontier_comparative_overview.png",
        "family_specific_compute_vs_lambda.png",
        "representative_latency_decomposition.png",
    ]
    for fig_name in expected_figures:
        assert (fig_dir / fig_name).exists(), f"Figure {fig_name} was not generated!"
