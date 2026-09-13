"""Integration test for the Phase 14 readout/head bottleneck diagnosis pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase14_readout_diagnosis import (
    generate_phase14_markdown_report,
    run_phase14_readout_diagnosis,
)


def test_phase14_pipeline(tmp_path: Path):
    """Smoke test running the Phase 14 pipeline with minimal epochs and one seed."""
    out_dir = tmp_path / "phase14_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase14_readout_diagnosis.md"

    summary = run_phase14_readout_diagnosis(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        jointco_epochs=2,
        head_epochs=2,
        samples_per_type=10,
        batch_size=10,
    )

    # Required summary keys
    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "per_seed_results",
        "baseline_perf_mean",
        "pooling_results_mean",
        "head_results_mean",
        "branch_readout_results_mean",
        "fusion_results_mean",
        "branch_combination_results_mean",
        "relational_sensitivity_mean",
        "r_probe_mean",
        "old_ceiling_mixed_mean",
        "new_ceiling_mixed_mean",
        "new_ceiling_nl_head_mixed_mean",
        "oracle_results_mean",
        "latencies_mean",
        "causal_diagnosis_aggregate",
        "verdict_case",
        "verdict_label",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    # Required CSVs
    expected = [
        "baseline_reproduction.csv",
        "representation_extraction.csv",
        "pooling_results.csv",
        "head_results.csv",
        "relational_branch_readout.csv",
        "fusion_ablation.csv",
        "branch_combination.csv",
        "relational_destruction.csv",
        "minimal_intervention.csv",
        "compute_results.csv",
        "latency_results.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]
    for name in expected:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        first = p.read_text(encoding="utf-8").splitlines()[0]
        assert "," in first

    # Figures
    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 12, f"only {len(figs)} figures"

    # Report
    generate_phase14_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 14" in report
    for forbidden in ["CONFIRMED", "90%+", "shatters", "scientifically validated"]:
        assert forbidden not in report

    # Verdict must be valid
    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F", "CASE G",
    )
