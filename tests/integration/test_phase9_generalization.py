"""Integration test for Phase 9 Generalization and Mixed-Structure runner."""
import json
from pathlib import Path
import pytest

from neuroforge.training.phase9_generalization import (
    generate_markdown_report,
    run_phase9_generalization,
)



@pytest.mark.slow
def test_phase9_generalization_integration_pipeline(tmp_path: Path):
    """Smoke test running full Phase 9 pipeline with minimal samples and epochs."""
    out_dir = tmp_path / "phase9_test_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase9_generalization.md"

    summary = run_phase9_generalization(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        train_samples_per_family=30,
        val_samples_per_family=16,
        test_samples_per_family=30,
        expert_epochs=2,
        router_epochs=2,
        batch_size=10,
    )

    # 1. Check summary JSON
    summary_file = m_dir / "phase9_summary.json"
    assert summary_file.exists()
    with summary_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "regime_9a_fresh" in loaded
    assert "regime_9b_structural" in loaded
    assert "regime_9c_distribution_shift" in loaded
    assert "regime_9d_mixed" in loaded

    # 2. Check CSV files
    expected_csvs = [
        "phase9_in_distribution.csv",
        "phase9_structural_generalization.csv",
        "phase9_distribution_shift.csv",
        "phase9_mixed_structure.csv",
        "phase9_expert_cross_eval.csv",
        "phase9_failure_diagnosis.csv",
        "phase9_latency_decomposition.csv",
    ]
    for csv_name in expected_csvs:
        csv_file = m_dir / csv_name
        assert csv_file.exists(), f"Expected CSV {csv_name} does not exist"
        assert csv_file.stat().st_size > 0

    # 3. Check 10 generated PNG figures
    expected_figures = [
        "fresh_test_vs_phase8b_benchmark.png",
        "structural_transformation_accuracy.png",
        "distribution_shift_accuracy_curves.png",
        "distribution_shift_compute_curves.png",
        "mixed_structure_expert_cross_evaluation.png",
        "mixed_structure_router_selection.png",
        "generalization_decision_preservation.png",
        "accuracy_vs_flops_generalization.png",
        "single_expert_ceiling_mixed_tasks.png",
        "representative_latency_decomposition.png",
    ]
    for fig_name in expected_figures:
        fig_path = f_dir / fig_name
        assert fig_path.exists(), f"Expected figure {fig_name} does not exist"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is too small"

    # 4. Check Markdown Report Generation
    generate_markdown_report(summary, report_path)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Phase 9 — Generalization and Mixed-Structure Evaluation" in content
    assert "## 1. Executive Summary" in content
    assert "## 20. Research Roadmap and Transition to Phase 10" in content
    assert "ORACLE NOT DEFINED" in content
