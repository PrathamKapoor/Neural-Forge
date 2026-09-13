import pytest
import torch


def test_controlled_routing_generator_exposes_observable_family_descriptors_deterministically():
    """Break caught: generator route labels drift from structural motifs or a seed fails to reproduce them."""
    from neuroforge.datasets import ControlledRoutingDataset

    first = ControlledRoutingDataset(samples=30, sequence_length=8, input_dim=4, seed=17)
    second = ControlledRoutingDataset(samples=30, sequence_length=8, input_dim=4, seed=17)

    assert torch.equal(first.features, second.features)
    assert torch.equal(first.oracle_routes, second.oracle_routes)
    assert torch.equal(first.natural_descriptors, second.natural_descriptors)
    assert torch.equal(first.explicit_metadata.sum(dim=1), torch.ones(30))
    assert set(first.oracle_routes.tolist()) == {0, 1, 2}
    assert first.natural_descriptors[first.oracle_routes == 1, 2].mean() > first.natural_descriptors[first.oracle_routes == 0, 2].mean()
    assert first.natural_descriptors[first.oracle_routes == 2, 3].mean() > first.natural_descriptors[first.oracle_routes == 0, 3].mean()


def test_routing_metrics_report_confusion_and_per_route_accuracy():
    """Break caught: aggregation can hide a route-specific failure behind global accuracy."""
    from neuroforge.evaluation.routing_metrics import routing_metrics

    routes = torch.tensor([0, 1, 2, 0, 2, 1])
    oracle = torch.tensor([0, 1, 2, 1, 2, 0])
    report = routing_metrics(routes, oracle)

    assert report["route_accuracy"] == pytest.approx(4 / 6)
    assert report["confusion_matrix"] == [[1, 1, 0], [1, 1, 0], [0, 0, 2]]
    assert report["per_route_accuracy"]["attention"] == 1.0
