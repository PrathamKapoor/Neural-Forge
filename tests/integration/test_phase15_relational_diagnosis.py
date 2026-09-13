"""Integration test for the Phase 15 relational-substep diagnosis pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase15_relational_diagnosis import (
    generate_phase15_markdown_report,
    run_phase15_relational_diagnosis,
)


def test_phase15_pipeline(tmp_path: Path):
    """Smoke test with minimal epochs, one seed, and few samples."""
    out_dir = tmp_path / "phase15_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase15_relational_diagnosis.md"

    summary = run_phase15_relational_diagnosis(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        jointco_epochs=2,
        samples_per_type=8,
        batch_size=8,
    )

    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "gate_info",
        "per_seed_results",
        "baseline_perf_mean",
        "depth_results_mean",
        "hypotheses",
        "verdict_case",
        "verdict_label",
        "minimal_intervention",
        "failure_diagnosis",
        "aggregates",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    # Core CSVs must exist; gated CSVs only if the gated section executed.
    for name in [
        "baseline_reproduction.csv",
        "depth_ablation.csv",
        "causal_controls.csv",
        "branch_ablation.csv",
        "representation_probe.csv",
        "compute.csv",
        "latency.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        assert "," in p.read_text(encoding="utf-8").splitlines()[0]
    if loaded["aggregates"]["aggregation"]["tested"]:
        assert (m_dir / "aggregation_ablation.csv").exists()
    if loaded["aggregates"]["capacity"]["tested"]:
        assert (m_dir / "capacity_ablation.csv").exists()

    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 6, f"only {len(figs)} figures"

    generate_phase15_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 15" in report
    for forbidden in ["proves", "solves relational reasoning", "universally superior",
                      "mathematically impossible", "general relational intelligence"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}

    # No hardcoded scientific results: verdict must equal a fresh programmatic derivation.
    from neuroforge.evaluation.phase15_metrics import (
        build_phase15_hypotheses,
        select_phase15_case,
    )
    hyps = build_phase15_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase15_case(
        hyps,
        loaded["aggregates"]["gap"].get("baseline_gap_mean", 1.0),
        loaded["aggregates"].get("compositional", {}).get("R_gain_mean", 0.0),
    )
    assert case == loaded["verdict_case"]
