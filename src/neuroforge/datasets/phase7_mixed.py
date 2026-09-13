"""Phase 7 mixed evaluation population combining Phase 6 task families with common input contract."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from neuroforge.datasets.phase6_specialization import Phase6ExpertSpecializationDataset

_SPLIT_OFFSET = {"train": 0, "validation": 10_000, "test": 20_000}
_ORACLE_MAP = {"feature": "mlp", "relational": "graph", "contextual": "attention_v2"}
_FAMILY_TO_ID = {"feature": 0, "relational": 1, "contextual": 2}


class Phase7MixedDataset(Dataset[dict[str, Any]]):
    """Balanced mixed evaluation dataset operating at the sample level.

    Integrates Feature, Relational, and Contextual samples generated under the
    identical Phase 6 common input contract ([12, 8] shape, neutral channel-4 marker).
    Samples are deterministically interleaved at the sample level so batches contain
    mixed families.
    """

    families = ("feature", "relational", "contextual")

    def __init__(
        self,
        split: str,
        samples_per_family: int,
        seed: int,
        sequence_length: int = 12,
        input_dim: int = 8,
    ) -> None:
        if split not in _SPLIT_OFFSET or samples_per_family < 2 or samples_per_family % 2 != 0:
            raise ValueError(f"split must be one of {tuple(_SPLIT_OFFSET)}, samples_per_family must be even")

        self.split = split
        self.samples_per_family = samples_per_family
        self.seed = seed
        self.total_samples = samples_per_family * len(self.families)

        # Generate each family using Phase 6 common-input-contract generator
        ds_feat = Phase6ExpertSpecializationDataset(
            "feature", split, samples_per_family, seed, sequence_length, input_dim
        )
        ds_rel = Phase6ExpertSpecializationDataset(
            "relational", split, samples_per_family, seed, sequence_length, input_dim
        )
        ds_ctx = Phase6ExpertSpecializationDataset(
            "contextual", split, samples_per_family, seed, sequence_length, input_dim
        )

        all_items: list[dict[str, Any]] = []
        for i in range(samples_per_family):
            all_items.append({
                "features": ds_feat.features[i],
                "target": ds_feat.targets[i],
                "family": "feature",
                "family_id": _FAMILY_TO_ID["feature"],
                "oracle_expert": _ORACLE_MAP["feature"],
            })
            all_items.append({
                "features": ds_rel.features[i],
                "target": ds_rel.targets[i],
                "family": "relational",
                "family_id": _FAMILY_TO_ID["relational"],
                "oracle_expert": _ORACLE_MAP["relational"],
            })
            all_items.append({
                "features": ds_ctx.features[i],
                "target": ds_ctx.targets[i],
                "family": "contextual",
                "family_id": _FAMILY_TO_ID["contextual"],
                "oracle_expert": _ORACLE_MAP["contextual"],
            })

        # Deterministic sample-level interleaving
        perm_generator = torch.Generator().manual_seed(seed + _SPLIT_OFFSET[split] + 50_000)
        perm = torch.randperm(len(all_items), generator=perm_generator).tolist()

        self.items = [all_items[p] for p in perm]
        for idx, item in enumerate(self.items):
            item["sample_id"] = idx

        self.features = torch.stack([it["features"] for it in self.items])
        self.targets = torch.tensor([it["target"] for it in self.items], dtype=torch.long)
        self.families_list = [it["family"] for it in self.items]
        self.family_ids = torch.tensor([it["family_id"] for it in self.items], dtype=torch.long)
        self.oracle_experts = [it["oracle_expert"] for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]
