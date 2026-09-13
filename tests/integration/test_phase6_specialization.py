import json
from pathlib import Path
from neuroforge.training import run_phase6_specialization


def test_phase6_smoke_serializes_summary_and_controls(tmp_path: Path):
    result = run_phase6_specialization(
        output_dir=tmp_path,
        seeds=(11,),
        epochs=1,
        train_samples=20,
        validation_samples=20,
        test_samples=20,
        batch_size=10,
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "source_data.csv").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment"] == "Phase 6 common-input-contract expert revalidation"
    assert "mlp" in summary["matrix"]
    assert "attention_v2" in summary["matrix"]
    assert len(summary["matrix"]["attention_v2"]) == 3
    assert "marker_ablation" in summary["controls"]
    assert "v2_random_marker_interface_control" in summary["controls"]
    assert summary["routing_gate"]["status"] == "closed"
