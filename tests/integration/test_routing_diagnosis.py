def test_routing_diagnosis_smoke_writes_per_seed_and_aggregate_evidence(tmp_path):
    """Break caught: diagnosis conditions execute but do not preserve comparable evidence."""
    from neuroforge.training.routing_diagnosis import run_routing_diagnosis

    result = run_routing_diagnosis(output_dir=tmp_path, seeds=(3,), epochs=4, samples=90, batch_size=30)

    assert (tmp_path / "summary.json").exists()
    assert "learned_explicit_supervised_hard" in result["aggregate"]
    assert result["per_seed"]["3"]["oracle"]["route_accuracy"] == 1.0
