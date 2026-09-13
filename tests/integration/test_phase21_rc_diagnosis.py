"""Integration test for the Phase 21 RC diagnosis pipeline (tiny config)."""
from __future__ import annotations

import json
from pathlib import Path

from neuroforge.training.phase21_rc_diagnosis import (
    generate_phase21_markdown_report,
    run_phase21_rc_diagnosis,
)


def test_phase21_pipeline(tmp_path: Path):
    """Smoke test: 1 seed, 2 epochs. Exercises all stages end to end."""
    out_dir = tmp_path / "phase21_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase21_rc_diagnosis.md"

    summary = run_phase21_rc_diagnosis(
        output_dir=out_dir,
        metrics_dir=m_dir,
        figures_dir=f_dir,
        seeds=(11,),
        epochs=2,
        batch_size=60,
        stages=("reproduce", "capacity", "diagnose", "finalize"),
    )

    assert (m_dir / "summary.json").exists()
    assert (m_dir / "manifest.json").exists()
    with (m_dir / "summary.json").open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    for key in [
        "manifest",
        "per_seed_results",
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
        "capacity.csv",
        "stage_probes.csv",
        "location_probes.csv",
        "gradients.csv",
        "counterfactuals.csv",
        "ablation.csv",
        "heads.csv",
        "order.csv",
        "analytical.csv",
        "agreement.csv",
        "decision_sensitivity.csv",
        "compute.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]:
        p = m_dir / name
        assert p.exists(), f"missing CSV: {name}"
        assert "," in p.read_text(encoding="utf-8").splitlines()[0]

    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 6, f"only {len(figs)} figures"

    generate_phase21_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 21" in report
    for forbidden in ["CONFIRMED", "90%+", "shatters", "scientifically validated"]:
        assert forbidden not in report

    assert loaded["verdict_case"] in (
        "CASE A", "CASE B", "CASE C", "CASE D", "CASE E", "CASE F",
    )
    # Gate + verdict must be reproducible from saved aggregates (no hardcoding).
    from neuroforge.evaluation.phase21_rc_diagnosis import (
        build_phase21_hypotheses,
        h8_gate,
        select_phase21_case,
    )
    assert set(loaded["hypotheses"]) == {f"H{i}" for i in range(1, 9)}
    hyps = build_phase21_hypotheses(loaded["aggregates"])
    assert {h: hyps[h]["status"] for h in hyps} == {
        h: loaded["hypotheses"][h]["status"] for h in hyps
    }
    case, _ = select_phase21_case(hyps, loaded["aggregates"])
    assert case == loaded["verdict_case"]
    gate = h8_gate({**loaded["aggregates"], "hypotheses": hyps})
    assert gate["passed"] == (loaded["gate"].get("passed", False))
