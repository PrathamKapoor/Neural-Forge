"""Phase 9 Datasets: Fresh in-distribution, structural transformations, distribution shifts, and mixed-structure benchmarks."""
from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import Dataset

from neuroforge.datasets.phase6_specialization import Phase6ExpertSpecializationDataset
from neuroforge.datasets.phase7_mixed import Phase7MixedDataset

_SPLIT_OFFSET = {"train": 0, "validation": 10_000, "test": 20_000, "fresh_test": 80_000}


class Phase9FreshInDistributionDataset(Dataset[dict[str, Any]]):
    """Fresh in-distribution evaluation population with independent seeds.

    Guarantees zero sample overlap with Phase 7/8B test sets while strictly
    preserving the Phase 6 common input contract ([12, 8] shape, neutral channel-4 marker),
    task semantics, and 1:1:1 family balance.
    """
    families = ("feature", "relational", "contextual")

    def __init__(
        self,
        samples_per_family: int = 240,
        seed: int = 11,
        sequence_length: int = 12,
        input_dim: int = 8,
    ) -> None:
        self.samples_per_family = samples_per_family
        self.seed = seed
        self.total_samples = samples_per_family * len(self.families)

        # Independent seed offset ensuring zero sample overlap with Phase 7/8B (which used seed + 20_000)
        fresh_seed = seed + _SPLIT_OFFSET["fresh_test"]

        ds_feat = Phase6ExpertSpecializationDataset("feature", "train", samples_per_family, fresh_seed, sequence_length, input_dim)
        ds_rel = Phase6ExpertSpecializationDataset("relational", "train", samples_per_family, fresh_seed, sequence_length, input_dim)
        ds_ctx = Phase6ExpertSpecializationDataset("contextual", "train", samples_per_family, fresh_seed, sequence_length, input_dim)

        oracle_map = {"feature": "mlp", "relational": "graph", "contextual": "attention_v2"}
        all_items: list[dict[str, Any]] = []

        for i in range(samples_per_family):
            all_items.append({
                "features": ds_feat.features[i],
                "target": ds_feat.targets[i],
                "family": "feature",
                "oracle_expert": oracle_map["feature"],
            })
            all_items.append({
                "features": ds_rel.features[i],
                "target": ds_rel.targets[i],
                "family": "relational",
                "oracle_expert": oracle_map["relational"],
            })
            all_items.append({
                "features": ds_ctx.features[i],
                "target": ds_ctx.targets[i],
                "family": "contextual",
                "oracle_expert": oracle_map["contextual"],
            })

        # Deterministic interleaving
        perm_generator = torch.Generator().manual_seed(fresh_seed + 55_000)
        perm = torch.randperm(len(all_items), generator=perm_generator).tolist()

        self.items = [all_items[p] for p in perm]
        for idx, item in enumerate(self.items):
            item["sample_id"] = idx

        self.features = torch.stack([it["features"] for it in self.items])
        self.targets = torch.tensor([it["target"] for it in self.items], dtype=torch.long)
        self.families_list = [it["family"] for it in self.items]
        self.oracle_experts = [it["oracle_expert"] for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def apply_phase9_token_permutation(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """B1 — Token Permutation: randomly permute token sequence.

    The query marker on channel 4 moves together with its token, preserving task semantics.
    """
    permuted = features.clone()
    generator = torch.Generator().manual_seed(seed)
    seq_len = features.shape[1]
    for i in range(len(permuted)):
        perm = torch.randperm(seq_len, generator=generator)
        permuted[i] = permuted[i, perm]
    return permuted


def apply_phase9_marker_variation(
    features: torch.Tensor,
    seed: int = 42,
) -> torch.Tensor:
    """B2 — Marker-Position Variation: move query marker to an unseen position.

    For contextual tasks, the designated query key vector moves with the marker
    so that query-content association semantics remain strictly valid.
    """
    transformed = features.clone()
    generator = torch.Generator().manual_seed(seed)
    seq_len = features.shape[1]

    for i in range(len(transformed)):
        old_q = int(transformed[i, :, 4].argmax().item())
        other_positions = [p for p in range(seq_len) if p != old_q]
        rand_idx = int(torch.randint(0, len(other_positions), (1,), generator=generator).item())
        new_q = other_positions[rand_idx]

        # Check if contextual (channels 0:3 carry normalized key vectors)
        # Swap tokens old_q and new_q completely to ensure marker and query key stay bound
        token_old = transformed[i, old_q].clone()
        token_new = transformed[i, new_q].clone()
        transformed[i, old_q] = token_new
        transformed[i, new_q] = token_old

    return transformed


def generate_phase9_variable_sequence_dataset(
    samples_per_family: int = 120,
    seed: int = 11,
    sequence_length: int = 16,
) -> dict[str, Any]:
    """B3 — Sequence Length Variation: generate dataset with S != 12 (e.g. S=8 or S=16)."""
    fresh_seed = seed + _SPLIT_OFFSET["fresh_test"] + sequence_length * 100
    generator = torch.Generator().manual_seed(fresh_seed)

    families = ("feature", "relational", "contextual")
    oracle_map = {"feature": "mlp", "relational": "graph", "contextual": "attention_v2"}

    features_list = []
    targets_list = []
    families_list = []
    oracles_list = []

    for fam in families:
        for i in range(samples_per_family):
            feat = torch.randn(sequence_length, 8, generator=generator) * 0.03
            pos = torch.randperm(sequence_length, generator=generator)
            q = int(pos[0])
            feat[q, 4] = 1.0  # query marker

            target_val = i % 2
            if fam == "feature":
                sign_a = -1.0 if (i % 4 < 2) else 1.0
                sign_b = -1.0 if ((i % 4) % 2 == 0) else 1.0
                feat[:, 0] += sign_a * 0.8
                feat[:, 1] += sign_b * 0.8
                target_val = 1 if (sign_a * sign_b > 0) else 0
            elif fam == "relational":
                cand = torch.randn(sequence_length, generator=generator)
                desired = 1.0 if target_val == 1 else -1.0
                score = (cand * torch.roll(cand, -1)).mean()
                if score.sign() != desired:
                    cand = cand * torch.tensor([1 if j % 2 == 0 else -1 for j in range(sequence_length)])
                feat[:, 0] += cand
            elif fam == "contextual":
                m = int(pos[1])
                key = torch.randn(3, generator=generator)
                key = key / key.norm()
                distractors = torch.randn(sequence_length, 3, generator=generator)
                distractors = distractors / distractors.norm(dim=1, keepdim=True)
                feat[:, 0:3] = distractors
                feat[q, :3] = key
                feat[m, :3] = key
                val = 1.0 if target_val == 1 else -1.0
                feat[m, 3] = val
                for p, v in zip(pos[2:min(5, sequence_length)].tolist(), [-val, -val, val][:max(1, sequence_length - 2)]):
                    feat[p, 3] = v

            features_list.append(feat)
            targets_list.append(target_val)
            families_list.append(fam)
            oracles_list.append(oracle_map[fam])

    # Interleave
    perm = torch.randperm(len(targets_list), generator=generator).tolist()
    features = torch.stack([features_list[p] for p in perm])
    targets = torch.tensor([targets_list[p] for p in perm], dtype=torch.long)
    fam_ordered = [families_list[p] for p in perm]
    ora_ordered = [oracles_list[p] for p in perm]

    return {
        "features": features,
        "targets": targets,
        "families": fam_ordered,
        "oracle_experts": ora_ordered,
        "sequence_length": sequence_length,
    }


def apply_phase9_relational_permutation(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """B4 — Destructive Control: permute node features relative to the internal ring graph.

    Breaks cyclic adjacent node correspondence without altering tensor shape or marker.
    """
    controlled = features.clone()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(features.shape[1], generator=generator)
    if torch.equal(order, torch.arange(features.shape[1])):
        order = torch.roll(order, 1)
    return controlled[:, order, :]


def apply_phase9_distribution_shift(
    features: torch.Tensor,
    shift_type: str,
    severity: str,
    seed: int = 42,
) -> torch.Tensor:
    """9C — Distribution Shift: apply non-semantic input perturbations.

    Severities:
      - 'mild', 'moderate', 'strong'
    Shift types:
      - 'magnitude_scaling': scales continuous feature channels
      - 'additive_noise': adds zero-mean Gaussian noise
      - 'variance_contrast': scales variance / contrast of feature channels
    """
    shifted = features.clone()
    generator = torch.Generator().manual_seed(seed)

    if shift_type == "magnitude_scaling":
        scale_map = {"mild": 1.15, "moderate": 1.35, "strong": 1.60}
        scale = scale_map.get(severity, 1.25)
        # Scale all feature channels except channel 4 (query marker)
        for c in (0, 1, 2, 3, 5, 6, 7):
            shifted[:, :, c] *= scale

    elif shift_type == "additive_noise":
        sigma_map = {"mild": 0.05, "moderate": 0.15, "strong": 0.30}
        sigma = sigma_map.get(severity, 0.15)
        noise = torch.randn(shifted.shape, generator=generator) * sigma
        noise[:, :, 4] = 0.0  # preserve marker
        shifted += noise

    elif shift_type == "variance_contrast":
        var_map = {"mild": 0.85, "moderate": 1.30, "strong": 1.75}
        var_scale = var_map.get(severity, 1.30)
        # Shift variance around mean
        for c in (0, 1, 2, 3, 5, 6, 7):
            ch_mean = shifted[:, :, c].mean(dim=1, keepdim=True)
            shifted[:, :, c] = ch_mean + (shifted[:, :, c] - ch_mean) * var_scale

    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")

    return shifted


class Phase9MixedStructureDataset(Dataset[dict[str, Any]]):
    """9D — Mixed-Structure Evaluation Dataset.

    Constructs samples exhibiting combinations of computational characteristics:
      - 'F' (pure Feature)
      - 'R' (pure Relational)
      - 'C' (pure Contextual)
      - 'FR' (Feature + Relational)
      - 'RC' (Relational + Contextual)
      - 'FC' (Feature + Contextual)
      - 'FRC' (Feature + Relational + Contextual)

    Target Formulation:
      Each active signal generates a binary vote s_k in {-1, +1}:
        - s_F: Feature sign XOR (sign_a * sign_b)
        - s_R: Relational adjacent product score sign
        - s_C: Contextual value retrieval sign
      Composite target:
        z = sum_{k in active} s_k
        y = 1 if z > 0 else (0 if z < 0 else (sample_id % 2))
      On FRC (3 components), sum is always non-zero in {-3, -1, +1, +3}.
      On 2-component mixtures, ties (z=0) are resolved by sample parity.

    Oracle Status:
      - Pure F -> 'mlp', Pure R -> 'graph', Pure C -> 'attention_v2'
      - Mixed FR, RC, FC, FRC: 'ORACLE NOT DEFINED' (Section 19).
    """
    mixed_families = ("F", "R", "C", "FR", "RC", "FC", "FRC")

    def __init__(
        self,
        samples_per_type: int = 120,
        seed: int = 11,
        sequence_length: int = 12,
        input_dim: int = 8,
    ) -> None:
        self.samples_per_type = samples_per_type
        self.seed = seed
        self.total_samples = samples_per_type * len(self.mixed_families)

        generator = torch.Generator().manual_seed(seed + 90_000)
        all_items: list[dict[str, Any]] = []

        for m_type in self.mixed_families:
            for i in range(samples_per_type):
                feat = torch.randn(sequence_length, input_dim, generator=generator) * 0.03
                pos = torch.randperm(sequence_length, generator=generator)
                q = int(pos[0])
                m = int(pos[1])
                feat[q, 4] = 1.0  # neutral query marker on channel 4

                # 1. Feature component
                has_f = "F" in m_type
                sign_a = -1.0 if (i % 4 < 2) else 1.0
                sign_b = -1.0 if ((i % 4) % 2 == 0) else 1.0
                sf = 1 if (sign_a * sign_b > 0) else -1
                if has_f:
                    feat[:, 0] += sign_a * 0.8
                    feat[:, 1] += sign_b * 0.8

                # 2. Relational component
                has_r = "R" in m_type
                cand = torch.randn(sequence_length, generator=generator)
                score = (cand * torch.roll(cand, -1)).mean()
                sr = 1 if score > 0 else -1
                if has_r:
                    feat[:, 0] += cand

                # 3. Contextual component
                has_c = "C" in m_type
                key = torch.randn(3, generator=generator)
                key = key / key.norm()
                distractors = torch.randn(sequence_length, 3, generator=generator)
                distractors = distractors / distractors.norm(dim=1, keepdim=True)
                sc = 1 if (i % 2 == 0) else -1
                val = float(sc)
                if has_c:
                    feat[:, 0:3] = distractors
                    feat[q, :3] = key
                    feat[m, :3] = key
                    feat[m, 3] = val
                    for p, v in zip(pos[2:5].tolist(), [-val, -val, val]):
                        feat[p, 3] = v

                # Composite Target Calculation
                active_signals = []
                if has_f:
                    active_signals.append(sf)
                if has_r:
                    active_signals.append(sr)
                if has_c:
                    active_signals.append(sc)

                z_sum = sum(active_signals)
                if z_sum > 0:
                    y = 1
                elif z_sum < 0:
                    y = 0
                else:
                    y = (i % 2)  # balanced tie-breaker on concordant discordance

                # Oracle expert assignment
                if m_type == "F":
                    oracle_exp = "mlp"
                elif m_type == "R":
                    oracle_exp = "graph"
                elif m_type == "C":
                    oracle_exp = "attention_v2"
                else:
                    oracle_exp = "ORACLE NOT DEFINED"

                all_items.append({
                    "features": feat,
                    "target": torch.tensor(y, dtype=torch.long),
                    "family": m_type,
                    "oracle_expert": oracle_exp,
                    "sf": sf,
                    "sr": sr,
                    "sc": sc,
                })

        # Interleave
        perm_gen = torch.Generator().manual_seed(seed + 95_000)
        perm = torch.randperm(len(all_items), generator=perm_gen).tolist()

        self.items = [all_items[p] for p in perm]
        for idx, it in enumerate(self.items):
            it["sample_id"] = idx

        self.features = torch.stack([it["features"] for it in self.items])
        self.targets = torch.tensor([it["target"].item() for it in self.items], dtype=torch.long)
        self.families_list = [it["family"] for it in self.items]
        self.oracle_experts = [it["oracle_expert"] for it in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]
