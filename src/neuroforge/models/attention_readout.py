"""Readout-only diagnostics for unchanged AttentionBlock stacks."""
import torch
from torch import nn
from neuroforge.blocks import AttentionBlock

class AttentionReadoutClassifier(nn.Module):
    def __init__(self, readout='mean', depth=1, hidden_dim=24):
        super().__init__(); self.readout=readout; self.encoder=nn.Linear(8,hidden_dim); self.blocks=nn.ModuleList(AttentionBlock(hidden_dim) for _ in range(depth)); width=hidden_dim*(2 if readout=='query_global' else 1); self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))
    def representation(self,x):
        h=self.encoder(x)
        for b in self.blocks: h,_=b(h)
        return h
    def read(self,h,x):
        q=x[:,:,4].argmax(1); query=h[torch.arange(len(h)),q]
        if self.readout=='mean': return h.mean(1)
        if self.readout=='max': return h.max(1).values
        if self.readout=='query': return query
        if self.readout=='query_global': return torch.cat((query,h.mean(1)),1)
        raise ValueError('unsupported readout')
    def forward(self,x): return self.head(self.read(self.representation(x),x))
