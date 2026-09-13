def test_smoke_experiment_writes_machine_readable_summary(tmp_path):
    """Break caught: an experiment can train but fails to leave reproducible evidence."""
    from neuroforge.config import NeuroForgeConfig, TrainingConfig
    from neuroforge.training import run_experiment

    config = NeuroForgeConfig(training=TrainingConfig(epochs=1, batch_size=12, learning_rate=0.01))
    summary = run_experiment(config, output_dir=tmp_path, include_baselines=False)

    assert (tmp_path / "summary.json").exists()
    assert 0 <= summary["adaptive_soft"]["accuracy"] <= 1
    assert "routing" in summary["adaptive_soft"]
