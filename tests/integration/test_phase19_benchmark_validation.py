"""Integration test for the Phase 19 benchmark-validation pipeline (no training)."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase19_benchmark_validation import (
    generate_phase19_markdown_report,
    run_phase19_benchmark_validation,
)


def test_phase19_pipeline(tmp_path: Path):
    """Smoke test with few samples (dataset construction + probes only)."""
    out_dir = tmp_path / "phase19_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase19_benchmark_validation.md"

    summary = run_phase19_benchmark_validation(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        samples_per_type=12,
    )

    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "per_seed_results",
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
        "bug_reproduction.csv",
        "observability.csv",
        "dependency_audit.csv",
        "label_consistency.csv",
        "parity_analysis.csv",
        "counterfactual_validation.csv",
        "invariance_controls.csv",
        "shortcut_audit.csv",
        "shallow_baselines.csv",
        "construction_diff.csv",
        "common_contract.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        assert "," in p.read_text(encoding="utf-8").splitlines()[0]

    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 6, f"only {len(figs)} figures"

    generate_phase19_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 19" in report
    for forbidden in ["solves relational reasoning", "universally superior",
                      "mathematically impossible", "general relational intelligence"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}

    # No hardcoded results: verdict must equal a fresh programmatic derivation.
    from neuroforge.evaluation.phase19_benchmark_validation import (
        build_phase19_hypotheses,
        select_phase19_case,
    )
    hyps = build_phase19_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase19_case(hyps, loaded["aggregates"])
    assert case == loaded["verdict_case"]
