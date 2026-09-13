"""Integration test for the Phase 11 composition-diagnosis pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroforge.training.phase11_composition_diagnosis import (
    generate_phase11_markdown_report,
    run_phase11_composition_diagnosis,
)


def test_phase11_composition_diagnosis_pipeline(tmp_path: Path):
    """Smoke test running the Phase 11 pipeline with minimal epochs and one seed."""
    out_dir = tmp_path / "phase11_out"
    m_dir = out_dir / "metrics"
    f_dir = out_dir / "figures"
    report_path = out_dir / "reports" / "phase11_composition_diagnosis.md"

    summary = run_phase11_composition_diagnosis(
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
        "oracle_ceiling_per_family",
        "oracle_ceiling_overall_mixed",
        "best_combos_per_family_aggregate",
        "representation_probe_per_expert",
        "interface_per_sequence",
        "aggregation_methods",
        "order_per_pair",
        "routing_selection_summary",
        "component_validation",
        "minimal_composition",
        "minimal_composition_with_adapter",
        "routing_representation_decodability_mean",
        "causal_diagnosis_aggregate",
        "phase10_ceiling_mean",
        "phase10_k1_mixed_mean",
    ]:
        assert key in loaded, f"missing summary key: {key}"

    # 2. All 13 required CSVs are present
    expected = [
        "oracle_composition.csv",
        "representation_probes.csv",
        "interface_ablation.csv",
        "aggregation_results.csv",
        "composition_order.csv",
        "routing_diagnosis.csv",
        "routing_representation.csv",
        "mixed_task_validation.csv",
        "counterfactual_results.csv",
        "compute_results.csv",
        "latency_results.csv",
        "seed_results.csv",
        "failure_diagnosis.csv",
    ]
    for name in expected:
        assert (m_dir / name).exists(), f"missing CSV: {name}"
        # Each must have at least the header
        first_line = (m_dir / name).read_text(encoding="utf-8").splitlines()[0]
        assert "," in first_line, f"CSV {name} has no header columns"

    # 3. 12 figures produced
    figs = list(f_dir.glob("*.png"))
    assert len(figs) >= 12, f"only {len(figs)} figures, expected >= 12"

    # 4. Report is generated
    generate_phase11_markdown_report(summary, report_path)
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Phase 11" in report
    # Must NOT contain hardcoded success language
    for forbidden in ["CONFIRMED", "90%+", "shatters", "scientifically validated"]:
        assert forbidden not in report, f"forbidden phrase in report: {forbidden!r}"

    # 5. Causal diagnosis has all 8 categories and each status is one of the allowed set
    causal = loaded["causal_diagnosis_aggregate"]
    for cat in [
        "EXPERT_CAPABILITY",
        "REPRESENTATION_TRANSFER",
        "AGGREGATION",
        "COMPOSITION_ORDER",
        "ROUTER_SELECTION",
        "BENCHMARK_SEMANTICS",
        "ROUTING_REPRESENTATION",
        "COMPUTE_ECONOMICS",
    ]:
        assert cat in causal
        assert causal[cat]["status"] in {
            "SUPPORTED",
            "PARTIALLY SUPPORTED",
            "NOT SUPPORTED",
            "INCONCLUSIVE",
            "NOT TESTED",
        }
