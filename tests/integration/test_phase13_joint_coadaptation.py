"""Integration test for the Phase 13 joint co-adaptation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase13_joint_coadaptation import (
    generate_phase13_markdown_report,
    run_phase13_joint_coadaptation,
)


def test_phase13_pipeline(tmp_path: Path):
    """Smoke test running the Phase 13 pipeline with minimal epochs and one seed."""
    out_dir = tmp_path / "phase13_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase13_joint_coadaptation.md"

    summary = run_phase13_joint_coadaptation(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        baseline_epochs=2,
        extended_epochs=4,
        samples_per_type=20,
        batch_size=20,
    )

    # Required summary keys
    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "per_seed_results",
        "joint_baseline_perf_mean",
        "joint_extended_perf_mean",
        "joint_coadapted_perf_mean",
        "existing_perf_mean",
        "cross_eval_mean",
        "scales_baseline_mean",
        "scales_coadapted_mean",
        "ablation_baseline_mean",
        "ablation_coadapted_mean",
        "repr_probes_baseline_mean",
        "repr_probes_coadapted_mean",
        "joint_re_rel_mean",
        "graph_re_rel_mean",
        "baseline_re_rel_mean",
        "old_ceiling_mixed_mean",
        "new_ceiling_mixed_mean",
        "oracle_results_mean",
        "param_counts",
        "latencies_mean",
        "causal_diagnosis_aggregate",
        "verdict_case",
        "verdict_label",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    # Required CSVs
    expected = [
        "baseline_reproduction.csv",
        "training_control.csv",
        "coadaptation_results.csv",
        "branch_scales.csv",
        "branch_ablation.csv",
        "representation_probes.csv",
        "relational_diagnosis.csv",
        "portfolio_results.csv",
        "oracle_results.csv",
        "compute_results.csv",
        "latency_results.csv",
        "seed_results.csv",
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
    generate_phase13_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 13" in report
    for forbidden in ["CONFIRMED", "90%+", "shatters", "scientifically validated"]:
        assert forbidden not in report

    # Causal diagnosis has all 8 categories
    causal = loaded["causal_diagnosis_aggregate"]
    for cat in (
        "UNDERTRAINING",
        "BRANCH_INTERFERENCE",
        "RELATIONAL_CAPABILITY",
        "REPRESENTATION_FUSION",
        "RELATIONAL_SENSITIVITY",
        "PORTFOLIO_LIMITATION",
        "OPTIMIZATION",
        "BENCHMARK_LIMITATION",
    ):
        assert cat in causal
        assert causal[cat]["status"] in {
            "SUPPORTED",
            "PARTIALLY SUPPORTED",
            "NOT SUPPORTED",
            "INCONCLUSIVE",
            "NOT TESTED",
        }

    # Verdict is a valid CASE
    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F", "CASE G",
    )

    # Cross-eval includes the 4 existing experts + 3 joint conditions
    expected_experts = {"mlp", "graph", "attention", "attention_v2",
                       "joint_baseline", "joint_extended", "joint_coadapted"}
    assert set(loaded["cross_eval_mean"].keys()) == expected_experts
