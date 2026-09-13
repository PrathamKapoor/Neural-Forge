"""Integration test for the Phase 16 embedded-vs-dedicated diagnosis pipeline."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase16_relational_interface_diagnosis import (
    generate_phase16_markdown_report,
    run_phase16_relational_interface,
)


def test_phase16_pipeline(tmp_path: Path, monkeypatch):
    """Smoke test with minimal epochs, one seed, few samples.

    Redirects the Phase 15 weight cache into tmp so the smoke test never
    depends on (or pollutes) real Phase 15 artifacts.
    """
    import neuroforge.training.phase16_relational_interface_diagnosis as p16mod

    cache = tmp_path / "p15cache"
    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(p16mod, "P15_PARTIALS", cache)

    out_dir = tmp_path / "phase16_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase16_relational_interface.md"

    summary = run_phase16_relational_interface(
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
        "graph_perf_mean",
        "depth3_perf_mean",
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
        "input_equivalence.csv",
        "graph_on_joint_input.csv",
        "representation_probe.csv",
        "pre_post_relational.csv",
        "coadaptation.csv",
        "gradient_diagnostics.csv",
        "branch_ablation.csv",
        "causal_controls.csv",
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

    generate_phase16_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 16" in report
    for forbidden in ["solves relational reasoning", "universally superior",
                      "mathematically impossible", "general relational intelligence"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}

    # No hardcoded results: verdict must equal a fresh programmatic derivation.
    from neuroforge.evaluation.phase16_metrics import (
        build_phase16_hypotheses,
        select_phase16_case,
    )
    hyps = build_phase16_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase16_case(hyps, loaded["aggregates"]["stages"])
    assert case == loaded["verdict_case"]
