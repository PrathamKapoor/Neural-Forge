"""Deterministic Phase 3 task families and structure-preserving controls."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

_SPLIT_OFFSET = {"train": 0, "validation": 10_000, "test": 20_000}


class ExpertSpecializationDataset(Dataset[dict[str, torch.Tensor]]):
    """Balanced fixed-shape tasks without a family identifier in model inputs."""
    families = ("feature", "relational", "contextual")

    def __init__(self, family: str, split: str, samples: int, seed: int, sequence_length: int = 12, input_dim: int = 8) -> None:
        if family not in self.families or split not in _SPLIT_OFFSET or samples < 2 or samples % 2 or sequence_length != 12 or input_dim != 8:
            raise ValueError("family/split must be supported; samples must be even; Phase 3 shape is [12,8]")
        generator = torch.Generator().manual_seed(seed + _SPLIT_OFFSET[split])
        targets = torch.arange(samples) % 2
        features = torch.randn(samples, sequence_length, input_dim, generator=generator) * 0.12
        if family == "feature":
            signs_a = torch.where(torch.arange(samples) % 4 < 2, -1.0, 1.0)
            signs_b = torch.where((torch.arange(samples) % 4) % 2 == 0, -1.0, 1.0)
            features[:, :, 0] += signs_a[:, None] * 0.8
            features[:, :, 1] += signs_b[:, None] * 0.8
            targets = ((signs_a * signs_b) > 0).long()
        elif family == "relational":
            # No individual node encodes the class: only signed adjacent products do.
            base = torch.randn(samples, sequence_length, generator=generator)
            desired = torch.where(targets == 1, 1.0, -1.0)
            for index in range(samples):
                candidate = base[index]
                score = (candidate * torch.roll(candidate, -1)).mean()
                if score.sign() != desired[index]: candidate = candidate * torch.tensor([1 if j % 2 == 0 else -1 for j in range(sequence_length)])
                features[index, :, 0] += candidate
        else:
            # Continuous keys make the matching token unique without positional cues.
            for index in range(samples):
                positions = torch.randperm(sequence_length, generator=generator)
                query_pos, match_pos = int(positions[0]), int(positions[1])
                key = torch.randn(4, generator=generator)
                features[index, :, :4] = torch.randn(sequence_length, 4, generator=generator)
                features[index, match_pos, :4] = key
                features[index, query_pos, :4] = key
                features[index, match_pos, 4] = 1.0 if targets[index] else -1.0
                anti_pos = next(position for position in range(sequence_length) if position not in {query_pos, match_pos})
                features[index, anti_pos, 4] = -features[index, match_pos, 4]
                features[index, query_pos, 5] = 1.0
        self.features, self.targets = features, targets.long()
        self.family, self.split = family, split

    def __len__(self) -> int: return len(self.targets)
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: return {"features": self.features[index], "target": self.targets[index]}


def apply_structural_control(features: torch.Tensor, family: str, seed: int) -> torch.Tensor:
    """Destroy the intended correspondence without labels or primitive changes.

    Relational: a non-identity node-feature permutation changes which values are
    neighbours under GraphBlock's internally constructed fixed ring. Contextual:
    negate each query's content key, leaving all positions and values intact.
    """
    controlled = features.clone()
    if family == "relational":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(features.shape[1], generator=generator)
        if torch.equal(order, torch.arange(features.shape[1])): order = torch.roll(order, 1)
        return controlled[:, order, :]
    if family == "contextual":
        queries = controlled[:, :, 5].argmax(1)
        for row, position in enumerate(queries.tolist()):
            controlled[row, position, :4] *= -1
        return controlled
    if family == "feature": return controlled.flip(1)
    raise ValueError(f"unknown family: {family}")
