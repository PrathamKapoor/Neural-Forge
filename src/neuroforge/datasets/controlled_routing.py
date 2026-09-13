"""A generator-defined, input-identifiable routing benchmark."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ControlledRoutingDataset(Dataset[dict[str, torch.Tensor]]):
    """Three structural families in one input-level task distribution.

    Route 0 has independent sequence positions, route 1 has a local autoregressive
    channel-0 motif, and route 2 has a long-range endpoint-correlated channel-0
    motif. The natural descriptor is computed from the sequence itself; the explicit
    one-hot metadata is a diagnostic control, never a predictive-model input.
    """

    route_names = ("mlp", "graph", "attention")

    def __init__(self, samples: int, sequence_length: int = 8, input_dim: int = 4, seed: int = 0) -> None:
        if samples < 3 or sequence_length < 3 or input_dim < 4:
            raise ValueError("samples >= 3, sequence_length >= 3, and input_dim >= 4 are required")
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(samples, sequence_length, input_dim, generator=generator)
        self.oracle_routes = torch.arange(samples) % 3
        local = self.oracle_routes == 1
        contextual = self.oracle_routes == 2
        for position in range(1, sequence_length):
            self.features[local, position, 0] = 0.92 * self.features[local, position - 1, 0] + 0.08 * torch.randn(int(local.sum()), generator=generator)
        self.features[contextual, -1, 0] = 0.96 * self.features[contextual, 0, 0] + 0.04 * torch.randn(int(contextual.sum()), generator=generator)
        self.targets = torch.empty(samples, dtype=torch.long)
        feature_score = self.features[:, :, 1].mean(dim=1)
        relation_score = (self.features[:, :-1, 1] * self.features[:, 1:, 1]).mean(dim=1)
        context_score = self.features[:, 0, 2] - self.features[:, -1, 2]
        self.targets[self.oracle_routes == 0] = (feature_score[self.oracle_routes == 0] > 0).long()
        self.targets[local] = (relation_score[local] > 0).long()
        self.targets[contextual] = (context_score[contextual] > 0).long()
        channel = self.features[:, :, 0]
        self.natural_descriptors = torch.stack((channel.mean(1), channel.std(1), (channel[:, :-1] * channel[:, 1:]).mean(1), channel[:, 0] * channel[:, -1]), dim=1)
        self.explicit_metadata = torch.nn.functional.one_hot(self.oracle_routes, num_classes=3).to(torch.float32)

    def __len__(self) -> int: return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "target": self.targets[index], "oracle_route": self.oracle_routes[index],
                "natural_descriptor": self.natural_descriptors[index], "explicit_metadata": self.explicit_metadata[index]}
