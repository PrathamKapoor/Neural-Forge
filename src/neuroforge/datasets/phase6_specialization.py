"""Phase 6 common-input-contract task families with neutral query marker."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

_SPLIT_OFFSET = {"train": 0, "validation": 10_000, "test": 20_000}


class Phase6ExpertSpecializationDataset(Dataset[dict[str, torch.Tensor]]):
    """Balanced fixed-shape tasks sharing a common neutral query marker interface.

    Input tensor shape: [sequence_length, input_dim] = [12, 8].
    Channel 4 contains a neutral query marker:
      - 1.0 at a designated query token index q ~ Uniform({0, ..., sequence_length - 1})
      - N(0, 0.03^2) Gaussian noise elsewhere
    The marker:
      - Contains no label information (q is chosen independently of target y)
      - Contains no task-family information (identical generation across all families)
      - Identifies exactly one query token for query-conditioned primitives (AttentionBlockV2)
      - Moves together with its token under arbitrary token permutation
    """
    families = ("feature", "relational", "contextual")

    def __init__(
        self,
        family: str,
        split: str,
        samples: int,
        seed: int,
        sequence_length: int = 12,
        input_dim: int = 8,
    ) -> None:
        if (
            family not in self.families
            or split not in _SPLIT_OFFSET
            or samples < 2
            or samples % 2
            or sequence_length != 12
            or input_dim != 8
        ):
            raise ValueError("family/split must be supported; samples must be even; Phase 6 shape is [12, 8]")

        generator = torch.Generator().manual_seed(seed + _SPLIT_OFFSET[split])
        targets = torch.arange(samples) % 2
        # Base background noise across all channels
        features = torch.randn(samples, sequence_length, input_dim, generator=generator) * 0.03

        for i in range(samples):
            # Designate query token position q uniformly at random, independent of target
            pos = torch.randperm(sequence_length, generator=generator)
            q = int(pos[0])
            # Common neutral query marker on channel 4
            features[i, q, 4] = 1.0

            if family == "feature":
                # Phase 3 Feature mechanism: signs in channel 0 and 1, XOR target
                sign_a = -1.0 if (i % 4 < 2) else 1.0
                sign_b = -1.0 if ((i % 4) % 2 == 0) else 1.0
                features[i, :, 0] += sign_a * 0.8
                features[i, :, 1] += sign_b * 0.8
                targets[i] = 1 if (sign_a * sign_b > 0) else 0
            elif family == "relational":
                # Phase 3 Relational mechanism: adjacent product score on channel 0
                candidate = torch.randn(sequence_length, generator=generator)
                desired = 1.0 if targets[i] == 1 else -1.0
                score = (candidate * torch.roll(candidate, -1)).mean()
                if score.sign() != desired:
                    candidate = candidate * torch.tensor([1 if j % 2 == 0 else -1 for j in range(sequence_length)])
                features[i, :, 0] += candidate
            elif family == "contextual":
                # Phase 4A / Phase 5 validated contextual mechanism
                m = int(pos[1])
                key = torch.randn(3, generator=generator)
                key = key / key.norm()
                distractors = torch.randn(sequence_length, 3, generator=generator)
                distractors = distractors / distractors.norm(dim=1, keepdim=True)
                features[i, :, 0:3] = distractors
                features[i, q, :3] = key
                features[i, m, :3] = key
                value = 1.0 if targets[i] else -1.0
                features[i, m, 3] = value
                # Three distractor values exactly cancel the matching value globally
                for p, v in zip(pos[2:5].tolist(), [-value, -value, value]):
                    features[i, p, 3] = v

        self.features = features
        self.targets = targets.long()
        self.family = family
        self.split = split

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "target": self.targets[index]}


def apply_phase6_marker_ablation(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Replace the query marker on channel 4 with neutral noise baseline (no peak)."""
    ablated = features.clone()
    generator = torch.Generator().manual_seed(seed)
    ablated[:, :, 4] = torch.randn(features.shape[0], features.shape[1], generator=generator) * 0.03
    return ablated


def apply_phase6_random_marker(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Move the query marker to a different random position (Section 15 control).

    The random position is independent of the true query key, verifying that V2's
    success is causally conditioned on the designated query rather than fixed positions.
    """
    controlled = features.clone()
    generator = torch.Generator().manual_seed(seed)
    sequence_length = features.shape[1]
    for i in range(len(controlled)):
        old_q = int(controlled[i, :, 4].argmax().item())
        other_positions = [p for p in range(sequence_length) if p != old_q]
        rand_idx = int(torch.randint(0, len(other_positions), (1,), generator=generator).item())
        new_q = other_positions[rand_idx]
        controlled[i, old_q, 4] = float(torch.randn(1, generator=generator).item() * 0.03)
        controlled[i, new_q, 4] = 1.0
    return controlled


def apply_phase6_token_permutation(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Permute sequence token order. Marker moves together with its token."""
    permuted = features.clone()
    generator = torch.Generator().manual_seed(seed)
    sequence_length = features.shape[1]
    for i in range(len(permuted)):
        perm = torch.randperm(sequence_length, generator=generator)
        permuted[i] = permuted[i, perm]
    return permuted


def apply_phase6_structural_control(features: torch.Tensor, family: str, seed: int = 42) -> torch.Tensor:
    """Structural controls preserving tensor shape and query marker."""
    controlled = features.clone()
    if family == "relational":
        # Permute node features relative to GraphBlock fixed ring
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(features.shape[1], generator=generator)
        if torch.equal(order, torch.arange(features.shape[1])):
            order = torch.roll(order, 1)
        return controlled[:, order, :]
    if family == "contextual":
        # Negate query key
        queries = controlled[:, :, 4].argmax(1)
        for row, position in enumerate(queries.tolist()):
            controlled[row, position, :3] *= -1
        return controlled
    if family == "feature":
        return controlled.flip(1)
    raise ValueError(f"unknown family: {family}")
