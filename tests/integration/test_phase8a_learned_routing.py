"""Integration test for Phase 8A Minimal Learned Router runner."""
from __future__ import annotations

from pathlib import Path

from neuroforge.training.phase8a_learned_routing import run_phase8a_learned_routing


def test_phase8a_integration_smoke(tmp_path: Path):
    output_dir = tmp_path / "phase8a_test"

    summary = run_phase8a_learned_routing(
        output_dir=output_dir,
        seeds=(11,),
        train_samples_per_family=12,
        val_samples_per_family=6,
        test_samples_per_family=12,
        expert_epochs=1,
        router_epochs=1,
        batch_size=6,
    )

    # Verify return structure
    assert "conditions" in summary
    assert "learned_routing" in summary
    assert "oracle_recovery" in summary
    assert "marker_neutrality_control" in summary
    assert "scientific_verdict" in summary

    # Verify serialization files
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "history.json").exists()
    assert (output_dir / "source_data.csv").exists()
    assert (output_dir / "routing_assignments.csv").exists()
    assert (output_dir / "seed_results.csv").exists()
    assert (output_dir / "latency_results.csv").exists()

    # Verify figure generation (8 figures)
    figures_dir = Path("figures/phase8a_learned_routing")
    assert figures_dir.exists()
    expected_figures = [
        "accuracy_comparison.png",
        "family_accuracy_comparison.png",
        "learned_selection_matrix.png",
        "expert_utilization_distribution.png",
        "accuracy_vs_flops.png",
        "oracle_vs_learned_performance.png",
        "routing_stability_across_seeds.png",
        "latency_decomposition.png",
    ]
    for fig_name in expected_figures:
        assert (figures_dir / fig_name).exists()
