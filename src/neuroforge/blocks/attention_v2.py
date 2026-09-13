"""Minimal query-conditioned content retrieval, separate from AttentionBlock."""
from __future__ import annotations
import torch
from torch import nn

class AttentionBlockV2(nn.Module):
    """Retrieve projected values into the marked query using raw content keys."""
    name='attention_v2'
    def __init__(self, hidden_dim:int, temperature:float=.1):
        super().__init__(); self.value=nn.Linear(hidden_dim,hidden_dim); self.output=nn.Linear(hidden_dim,hidden_dim); self.hidden_dim=hidden_dim; self.temperature=temperature
    def forward(self,state:torch.Tensor,raw_features:torch.Tensor)->tuple[torch.Tensor,float]:
        query_index=raw_features[:,:,4].argmax(1); keys=torch.nn.functional.normalize(raw_features[:,:,:3],dim=-1); q=keys[torch.arange(len(state)),query_index]
        scores=(keys*q[:,None]).sum(-1)/self.temperature; scores[torch.arange(len(state)),query_index]=-float('inf'); weights=scores.softmax(1)
        retrieved=torch.bmm(weights[:,None],self.value(state)).squeeze(1); update=torch.zeros_like(state); update[torch.arange(len(state)),query_index]=self.output(retrieved)
        return state+update,float(2*state.shape[1]*self.hidden_dim**2+2*state.shape[1]*3)
