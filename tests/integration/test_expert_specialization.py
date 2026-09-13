def test_phase3_smoke_serializes_all_cross_evaluation_cells(tmp_path):
    from neuroforge.training.expert_specialization import run_expert_specialization
    result = run_expert_specialization(tmp_path, seeds=(3,), epochs=2, train_samples=60, validation_samples=30, test_samples=30)
    assert (tmp_path / "summary.json").exists()
    assert len(result["cells"]) == 9
    assert "relational_node_feature_permutation" in result["controls"]
