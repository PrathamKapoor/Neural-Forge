"""Phase 19 repaired mixed-structure benchmark (versioned, additive).

Version: ``phase19-repaired-v1``. The historical
`Phase9MixedStructureDataset` is untouched; this module reimplements the same
API with ONE minimal construction change that repairs the Phase 18 erasure
defect.

Historical defect
-----------------
Per-sample construction order was base → Feature → Relational → Contextual.
The contextual block assigns ``feat[:, 0:3]`` (distractor keys), which
overwrites the Feature offsets (channels 0/1) and the relational carrier
(channel 0) on every family containing C (RC, FC, FRC), while labels kept
depending on the erased statistics.

Minimal repair (same RNG draw order/values, only placement changes)
-------------------------------------------------------------------
Per-sample order is now base → Contextual → Feature → Relational, with the
relational carrier relocated to the previously pure-noise channel 5:

- Contextual block (if ``has_c``): identical writes (keys, votes, marker).
- Feature offsets (if ``has_f``): ``ch0 += sign_a*0.8``, ``ch1 += sign_b*0.8``
  applied AFTER the contextual block, so they survive. The offset is
  common-mode across tokens, which preserves exact key identity for
  content-addressed retrieval (verified by the 19C audit, not assumed).
- Relational carrier (if ``has_r``): ``ch5 += cand``. Channel 5 was pure
  noise; no contextual or feature write touches it, so the carrier provably
  survives. Keeps ``input_dim=8`` (no channel-count change).

Unchanged: label algebra (majority + parity tiebreak), family definitions,
class balance, sequence length, marker semantics (ch4), vote semantics (ch3),
key semantics (ch0:2), oracle-expert mapping, seed-offset scheme, item keys
(``sf``/``sr``/``sc``/``sample_id``), tensor shapes, RNG draw values.

Consequences (documented, not hidden)
-------------------------------------
- Families without C (F/R/FR) keep F identical; R moves ch0 → ch5.
- Families with C gain observable F (FC/FRC) and R (RC/FRC) carriers.
- C retrieval faces common-mode key rotation on FC/FRC (audited in 19C).
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


CONSTRUCTION_VERSION = "phase19-repaired-v1"
RELATIONAL_CHANNEL = 5


class Phase9MixedRepairedDataset(Dataset[dict[str, Any]]):
    """Repaired mixed-structure benchmark (see module docstring)."""

    mixed_families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    construction_version = CONSTRUCTION_VERSION

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

                has_f = "F" in m_type
                has_r = "R" in m_type
                has_c = "C" in m_type

                # Component statistics (same draws/algebra as the historical
                # construction; only carrier placement below differs).
                sign_a = -1.0 if (i % 4 < 2) else 1.0
                sign_b = -1.0 if ((i % 4) % 2 == 0) else 1.0
                sf = 1 if (sign_a * sign_b > 0) else -1

                cand = torch.randn(sequence_length, generator=generator)
                score = (cand * torch.roll(cand, -1)).mean()
                sr = 1 if score > 0 else -1

                key = torch.randn(3, generator=generator)
                key = key / key.norm()
                distractors = torch.randn(sequence_length, 3, generator=generator)
                distractors = distractors / distractors.norm(dim=1, keepdim=True)
                sc = 1 if (i % 2 == 0) else -1
                val = float(sc)

                # Repaired placement order: Contextual → Feature → Relational.
                if has_c:
                    feat[:, 0:3] = distractors
                    feat[q, :3] = key
                    feat[m, :3] = key
                    feat[m, 3] = val
                    for p, v in zip(pos[2:5].tolist(), [-val, -val, val]):
                        feat[p, 3] = v
                if has_f:
                    feat[:, 0] += sign_a * 0.8
                    feat[:, 1] += sign_b * 0.8
                if has_r:
                    feat[:, RELATIONAL_CHANNEL] += cand

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
                    y = (i % 2)

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
