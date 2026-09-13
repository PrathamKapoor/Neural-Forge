"""Integration test for the Phase 20 repaired-portfolio pipeline (tiny config)."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase20_repaired_portfolio import (
    generate_phase20_markdown_report,
    run_phase20_repaired_portfolio,
)


def test_phase20_pipeline(tmp_path: Path):
    """Smoke test with minimal epochs, one seed, few samples."""
    out_dir = tmp_path / "phase20_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase20_repaired_portfolio.md"

    summary = run_phase20_repaired_portfolio(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        expert_epochs=2,
        router_epochs=2,
        samples_per_type=8,
        batch_size=8,
    )

    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    # historical dirs must not be touched by the run (checked at repo level)
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["manifest"]["benchmark_version"] == "phase19-repaired-v1"
    for key in [
        "manifest",
        "per_seed_results",
        "cross_mean",
        "hypotheses",
        "verdict_case",
        "verdict_label",
        "recommendation",
        "minimal_intervention",
        "failure_diagnosis",
        "aggregates",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    for name in [
        "benchmark_smoke.csv",
        "dataset_manifest.csv",
        "pure_experts.csv",
        "cross_evaluation.csv",
        "empirical_ceiling.csv",
        "fixed_composition.csv",
        "oracle.csv",
        "random_routing.csv",
        "learned_routing.csv",
        "compute_aware.csv",
        "routing_diagnostics.csv",
        "component_semantics.csv",
        "rc_frc_analysis.csv",
        "representation_probe.csv",
        "causal_controls.csv",
        "compute.csv",
        "latency.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        assert "," in p.read_text(encoding="utf-8").splitlines()[0]

    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 8, f"only {len(figs)} figures"

    generate_phase20_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 20" in report
    assert "phase19-repaired-v1" in report
    for forbidden in ["solves relational reasoning", "universally superior",
                      "mathematically impossible", "general relational intelligence"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}

    # No hardcoded results: verdict must equal a fresh programmatic derivation.
    from neuroforge.evaluation.phase20_repaired_portfolio import (
        build_phase20_hypotheses,
        select_phase20_case,
    )
    hyps = build_phase20_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase20_case(hyps, loaded["aggregates"])
    assert case == loaded["verdict_case"]
