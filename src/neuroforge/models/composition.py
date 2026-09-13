"""Phase 10 Multi-Expert Composition: Parallel and Sequential execution architectures."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.multi_expert import (
    AdaptiveKRouter,
    MultiExpertRoutingDecision,
    TopKRouter,
    aggregate_parallel_normalized,
    aggregate_parallel_uniform,
    aggregate_parallel_weighted,
)


class ParallelMultiExpert(nn.Module):
    """Executes k selected specialists in parallel and aggregates their predictions."""

    def __init__(
        self,
        experts: dict[str, StandaloneSpecialist],
        expert_names: tuple[str, ...] = ("mlp", "graph", "attention", "attention_v2"),
        router: TopKRouter | AdaptiveKRouter | None = None,
        aggregation: str = "uniform",
        router_flops: float = 1052.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleDict(experts)
        self.expert_names = expert_names
        self.router = router if router is not None else TopKRouter()
        self.aggregation = aggregation
        self.router_flops = router_flops

        # Precompute expert FLOPs for 12-token sequences
        self.raw_flops = {
            "mlp": 8928.0,
            "graph": 8112.0,
            "attention": 39168.0,
            "attention_v2": 46296.0,
        }

    def forward(
        self,
        features: torch.Tensor,
        k: int | None = None,
        mode: str = "hard",
        aggregation: str | None = None,
    ) -> tuple[torch.Tensor, MultiExpertRoutingDecision, list[float]]:
        agg_mode = aggregation if aggregation is not None else self.aggregation
        b_size = len(features)

        # Route
        if isinstance(self.router, TopKRouter):
            actual_k = k if k is not None else 2
            decision = self.router(features, k=actual_k, mode=mode)
        elif isinstance(self.router, AdaptiveKRouter):
            decision = self.router(features, mode=mode, fixed_k=k)
        else:
            raise TypeError("Unsupported router type")

        # Forward pass through selected experts
        # Compute all expert logits under no_grad since experts are frozen
        with torch.no_grad():
            expert_logits_dict = {
                name: self.experts[name](features) for name in self.expert_names
            }

        sample_final_logits = []
        sample_flops: list[float] = []

        for i in range(b_size):
            selected_idx = decision.selected_indices[i]
            selected_names = [self.expert_names[idx] for idx in selected_idx]

            sub_logits = [expert_logits_dict[name][i:i + 1] for name in selected_names]
            w = decision.weights[i, selected_idx].unsqueeze(0)  # [1, K]

            if agg_mode == "uniform":
                combined = aggregate_parallel_uniform(sub_logits)
            elif agg_mode == "weighted":
                combined = aggregate_parallel_weighted(sub_logits, w)
            elif agg_mode == "normalized":
                combined = aggregate_parallel_normalized(sub_logits, w)
            else:
                raise ValueError(f"Unknown aggregation: {agg_mode}")

            sample_final_logits.append(combined)

            # FLOPs = router + sum(selected_expert_flops) + aggregation_flops
            # Aggregation flops: ~2 flops per class per expert
            agg_flops = float(len(selected_names) * 2 * 2)
            exp_flops = sum(self.raw_flops.get(name, 10000.0) for name in selected_names)
            total_sample_fl = self.router_flops + exp_flops + agg_flops
            sample_flops.append(total_sample_fl)

        final_logits = torch.cat(sample_final_logits, dim=0)
        return final_logits, decision, sample_flops


class SequentialSpecialistComposition(nn.Module):
    """Chains two or more specialist blocks sequentially to test order of computational operations."""

    def __init__(
        self,
        experts: dict[str, StandaloneSpecialist],
        sequence: tuple[str, ...],
        mode: str = "block_chain",
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleDict({name: experts[name] for name in sequence})
        self.sequence = sequence
        self.mode = mode

        # Measure parameter count (frozen)
        self.total_parameters = sum(sum(p.numel() for p in experts[name].parameters()) for name in sequence)

        # FLOPs map
        self.raw_flops = {
            "mlp": 8928.0,
            "graph": 8112.0,
            "attention": 39168.0,
            "attention_v2": 46296.0,
        }

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Forward through the ordered sequence of specialists.
        
        Returns combined logits [B, num_classes] and total theoretical FLOPs per sample.
        """
        b_size = len(features)
        total_flops = sum(self.raw_flops.get(name, 10000.0) for name in self.sequence)

        if self.mode == "block_chain":
            # Block-level chaining:
            # First expert encodes input
            first_name = self.sequence[0]
            first_exp = self.experts[first_name]
            state = first_exp.encoder(features)

            for idx, name in enumerate(self.sequence):
                exp = self.experts[name]
                for block in exp.blocks:
                    if exp.architecture == "attention_v2":
                        state, _ = block(state, features)
                    else:
                        state, _ = block(state)

            # Last expert head classifies
            last_name = self.sequence[-1]
            last_exp = self.experts[last_name]
            if last_exp.architecture == "attention_v2":
                q = features[:, :, 4].argmax(1)
                logits = last_exp.head(state[torch.arange(b_size), q])
            else:
                logits = last_exp.head(state.mean(1))

        elif self.mode == "cascade":
            # Cascading logit refinement: each expert processes input and adds refinement
            logits_list = [self.experts[name](features) for name in self.sequence]
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return logits, float(total_flops)
