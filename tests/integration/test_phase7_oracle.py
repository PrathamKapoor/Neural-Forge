import json
from pathlib import Path
from neuroforge.training import run_phase7_oracle_routing


def test_phase7_smoke_serializes_summary_and_figures(tmp_path: Path):
    summary = run_phase7_oracle_routing(
        output_dir=tmp_path,
        seeds=(11,),
        train_samples_per_family=20,
        val_samples_per_family=20,
        test_samples_per_family=20,
        epochs=1,
        batch_size=10,
    )

    # Verify JSON and CSV files exist
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "source_data.csv").exists()

    # Verify summary contents
    assert summary["experiment"] == "Phase 7 Oracle Routing and Fixed-Best Selection"
    assert "Oracle Router" in summary["conditions"]
    assert "Fixed MLP" in summary["conditions"]
    assert "Oracle Without V2" in summary["conditions"]
    assert "best_fixed_baseline" in summary
    assert "oracle_advantage" in summary

    # Verify figures generated
    figs_dir = Path("figures/phase7_oracle_routing")
    assert (figs_dir / "fixed_expert_accuracy_by_family.png").exists()
    assert (figs_dir / "oracle_vs_fixed_accuracy.png").exists()
    assert (figs_dir / "accuracy_vs_flops.png").exists()
    assert (figs_dir / "accuracy_vs_latency.png").exists()
    assert (figs_dir / "oracle_routing_selection_matrix.png").exists()
    assert (figs_dir / "compute_distribution_oracle.png").exists()

    # Verify no learned router parameters exist
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "router_parameters" not in manifest
    assert "router_training" not in manifest
