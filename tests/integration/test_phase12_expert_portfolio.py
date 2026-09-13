"""Integration test for the Phase 12 expert-portfolio expansion pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase12_expert_portfolio import (
    generate_phase12_markdown_report,
    run_phase12_expert_portfolio,
)


def test_phase12_expert_portfolio_pipeline(tmp_path: Path):
    """Smoke test running the Phase 12 pipeline with minimal epochs and one seed."""
    out_dir = tmp_path / "phase12_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase12_expert_portfolio.md"

    summary = run_phase12_expert_portfolio(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        expert_epochs=2,
        samples_per_type=20,
        batch_size=20,
    )

    # 1. Required summary keys
    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "per_seed_results",
        "parameter_matching",
        "old_ceiling_per_family_mean",
        "new_ceiling_per_family_mean",
        "old_ceiling_overall_mixed_mean",
        "new_ceiling_overall_mixed_mean",
        "cross_evaluation_per_expert_mean",
        "oracle_old_per_family_mean",
        "oracle_new_per_family_mean",
        "router_per_lambda_mean",
        "capability_vs_capacity",
        "parameter_counts",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    # 2. Required CSVs present and non-empty
    expected = [
        "source_data.csv",
        "seed_results.csv",
        "expert_cross_evaluation.csv",
        "parameter_matching.csv",
        "single_expert_ceiling.csv",
        "oracle_portfolio.csv",
        "composition_results.csv",
        "router_results.csv",
        "routing_assignments.csv",
        "utilization.csv",
        "compute_results.csv",
        "latency_results.csv",
        "capability_vs_capacity.csv",
    ]
    for name in expected:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        first = p.read_text(encoding="utf-8").splitlines()[0]
        assert "," in first

    # 3. 15 figures (some may be placeholder text panels for joint-training,
    # but the file must exist on disk).
    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 12, f"only {len(figs)} figures, expected >= 12"

    # 4. Report
    generate_phase12_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 12" in report
    for forbidden in ["CONFIRMED", "90%+", "shatters", "scientifically validated"]:
        assert forbidden not in report

    # 5. Summary invariants
    assert loaded["new_ceiling_overall_mixed_mean"] >= 0.0
    assert loaded["old_ceiling_overall_mixed_mean"] >= 0.0
    # The parameter-matching gap is reported (not necessarily small)
    assert 0.0 <= loaded["parameter_matching"]["param_gap"] <= 1.0
    # cross_evaluation has the 6 experts (4 existing + joint + control)
    assert set(loaded["cross_evaluation_per_expert_mean"].keys()) == {
        "mlp", "graph", "attention", "attention_v2", "joint", "control"
    }
