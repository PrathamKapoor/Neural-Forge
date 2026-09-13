import pytest


def test_default_config_is_valid_and_cpu_safe():
    """Break caught: removing a required default makes experiments unreproducible."""
    from neuroforge.config import NeuroForgeConfig

    config = NeuroForgeConfig()

    assert config.device == "cpu"
    assert config.model.modules == ("mlp", "graph", "attention")
    assert config.training.batch_size > 0


def test_invalid_router_temperature_is_rejected():
    """Break caught: accepting a non-positive temperature causes invalid softmax scaling."""
    from neuroforge.config import RouterConfig

    with pytest.raises(ValueError, match="temperature"):
        RouterConfig(temperature=0)
