"""Standalone classifiers that reuse the Phase 1/2 computational primitives."""
from __future__ import annotations
import torch
from torch import nn
from neuroforge.blocks import (
    MLPBlock,
    GraphBlock,
    AttentionBlock,
    AttentionBlockV2,
    JointBlock,
    JointCoBlock,
)

_BLOCKS = {
    "mlp": MLPBlock,
    "graph": GraphBlock,
    "attention": AttentionBlock,
    "attention_v2": AttentionBlockV2,
    "joint": JointBlock,
    "joint_co": JointCoBlock,
}

class StandaloneSpecialist(nn.Module):
    def __init__(self, architecture: str, input_dim: int = 8, hidden_dim: int = 24, depth: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        if architecture not in _BLOCKS or depth < 1: raise ValueError("unsupported architecture or depth")
        self.architecture, self.depth = architecture, depth
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(_BLOCKS[architecture](hidden_dim) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        state = self.encoder(features)
        # Architectures that need the original raw features (channel-4 marker,
        # channels 0:3 keys) for query-conditioned retrieval.
        if self.architecture in ("attention_v2", "joint", "joint_co"):
            for block in self.blocks:
                state, _ = block(state, features)
            q = features[:, :, 4].argmax(1)
            return self.head(state[torch.arange(len(features)), q])
        for block in self.blocks: state, _ = block(state)
        return self.head(state.mean(1))
