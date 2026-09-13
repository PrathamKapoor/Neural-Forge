"""Integration test for the Phase 18 decision-semantics pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase18_decision_semantics import (
    generate_phase18_markdown_report,
    run_phase18_decision_semantics,
)


def test_phase18_pipeline(tmp_path: Path, monkeypatch):
    """Smoke test with minimal epochs, one seed, few samples.

    Redirects the Phase 15/16 weight caches into tmp so the smoke test never
    depends on (or pollutes) real artifacts.
    """
    import neuroforge.training.phase18_decision_semantics as p18mod

    cache15 = tmp_path / "p15cache"
    cache16 = tmp_path / "p16cache"
    cache15.mkdir(parents=True, exist_ok=True)
    cache16.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(p18mod, "P15_PARTIALS", cache15)
    monkeypatch.setattr(p18mod, "P16_PARTIALS", cache16)

    out_dir = tmp_path / "phase18_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase18_component_composition.md"

    summary = run_phase18_decision_semantics(
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
        "per_seed_results",
        "baseline_perf_mean",
        "depth3_perf_mean",
        "gate",
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
        "baseline_reproduction.csv",
        "component_targets.csv",
        "component_probe_matrix.csv",
        "joint_decodability.csv",
        "agreement_disagreement.csv",
        "logit_margin.csv",
        "counterfactuals.csv",
        "diagnostic_heads.csv",
        "objective_diagnosis.csv",
        "rc_frc_semantics.csv",
        "shuffled_controls.csv",
        "intervention.csv",
        "rc_frc_results.csv",
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

    generate_phase18_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 18" in report
    for forbidden in ["solves relational reasoning", "universally superior",
                      "mathematically impossible", "general relational intelligence"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}

    # No hardcoded results: verdict must equal a fresh programmatic derivation.
    from neuroforge.evaluation.phase18_component_composition import (
        build_phase18_hypotheses,
        select_phase18_case,
    )
    hyps = build_phase18_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase18_case(hyps, loaded["aggregates"])
    assert case == loaded["verdict_case"]
