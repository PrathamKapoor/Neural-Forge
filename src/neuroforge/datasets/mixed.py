"""Controlled data with a private oracle route used only for evaluation."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MixedStructureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """A balanced binary task with feature, local-relation, and contextual subpopulations.

    The oracle route is returned as metadata, never appended to model features. Each
    group uses a different label-generating relation, so router agreement has a
    meaningful, controlled interpretation rather than a post-hoc story.
    """

    def __init__(self, samples: int, sequence_length: int, input_dim: int, seed: int) -> None:
        if samples < 3:
            raise ValueError("samples must be at least 3 to include every structure")
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(samples, sequence_length, input_dim, generator=generator)
        self.oracle_routes = torch.arange(samples) % 3
        targets = torch.empty(samples, dtype=torch.long)
        feature_score = self.features[:, :, 0].mean(dim=1) + 0.35 * self.features[:, :, 1].mean(dim=1)
        relational_score = (self.features[:, :, 0] * torch.roll(self.features[:, :, 1], shifts=1, dims=1)).mean(dim=1)
        contextual_score = self.features[:, 0, 2] - self.features[:, -1, 2] + 0.25 * self.features[:, 2, 3]
        targets[self.oracle_routes == 0] = (feature_score[self.oracle_routes == 0] > 0).long()
        targets[self.oracle_routes == 1] = (relational_score[self.oracle_routes == 1] > 0).long()
        targets[self.oracle_routes == 2] = (contextual_score[self.oracle_routes == 2] > 0).long()
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index], self.oracle_routes[index]
