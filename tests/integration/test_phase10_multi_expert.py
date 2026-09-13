"""Integration test for Phase 10 Adaptive Multi-Expert Composition pipeline."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from neuroforge.training.phase10_multi_expert import (
    generate_phase10_markdown_report,
    run_phase10_multi_expert,
)


def test_phase10_multi_expert_pipeline(tmp_path: Path):
    """Smoke test running Phase 10 pipeline with small sample count and fast epochs."""
    out_dir = tmp_path / "phase10_test_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase10_multi_expert.md"

    summary = run_phase10_multi_expert(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        expert_epochs=2,
        router_epochs=2,
        samples_per_type=20,
        seeds=(11,),
        batch_size=10,
    )

    # 1. Check summary JSON structure
    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded_summary = json.load(f)

    assert "summary_by_policy" in loaded_summary
    assert "mixed_task_comparison" in loaded_summary
    assert "ceiling_vs_multi" in loaded_summary
    assert "performance_compute_points" in loaded_summary
    assert "pareto_points" in loaded_summary
    assert "accuracy_constrained_compute" in loaded_summary
    assert "collapse_audit" in loaded_summary

    # 2. Check CSV files generated
    expected_csvs = [
        "source_data.csv",
        "seed_results.csv",
        "fixed_k_results.csv",
        "composition_results.csv",
        "adaptive_k_results.csv",
        "routing_assignments.csv",
        "expert_pair_utilization.csv",
        "expert_triple_utilization.csv",
        "counterfactual_results.csv",
        "compute_results.csv",
        "latency_results.csv",
        "ablation_results.csv",
        "failure_diagnosis.csv",
        "family_composition_matrix.csv",
    ]
    for csv_name in expected_csvs:
        csv_file = m_dir / csv_name
        assert csv_file.exists(), f"Expected CSV {csv_name} does not exist"
        assert csv_file.stat().st_size > 0, f"CSV {csv_name} is empty"

    # 3. Check 12 generated PNG figures
    expected_figures = [
        "mixed_task_accuracy_comparison.png",
        "accuracy_vs_flops.png",
        "accuracy_vs_active_experts.png",
        "k_distribution.png",
        "expert_pair_utilization.png",
        "family_composition_matrix.png",
        "single_expert_ceiling_vs_multi_expert.png",
        "parallel_vs_sequential_comparison.png",
        "adaptive_k_behavior.png",
        "latency_decomposition.png",
        "compute_accuracy_pareto_frontier.png",
        "seed_stability.png",
    ]
    for fig_name in expected_figures:
        fig_path = f_dir / fig_name
        assert fig_path.exists(), f"Expected figure {fig_name} does not exist"
        assert fig_path.stat().st_size > 500, f"Figure {fig_name} is too small"

    # 4. Check Markdown Report Generation
    generate_phase10_markdown_report(summary, report_path)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Phase 10 — Adaptive Multi-Expert Composition" in content
    assert "## 1. Executive Summary" in content
    assert "## 23. Implications for Phase 11" in content
    assert "ORACLE NOT DEFINED" in content
