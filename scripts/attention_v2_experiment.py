import json,torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from neuroforge.blocks import AttentionBlockV2
from neuroforge.datasets.contextual_attention import ContextualAssociationDataset,break_association,shuffle_values

class V2(nn.Module):
 def __init__(self,d):
  super().__init__();self.e=nn.Linear(8,24);self.b=nn.ModuleList(AttentionBlockV2(24) for _ in range(d));self.h=nn.Sequential(nn.LayerNorm(24),nn.Linear(24,2))
 def rep(self,x):
  h=self.e(x)
  for b in self.b:h,_=b(h,x)
  return h
 def forward(self,x):
  h=self.rep(x);q=x[:,:,4].argmax(1);return self.h(h[torch.arange(len(x)),q])
def xy(split,s):
 d=ContextualAssociationDataset(split,360 if split=='train' else 240,s);return d.features,d.targets
def fit(s,d):
 torch.manual_seed(s);m=V2(d);o=torch.optim.AdamW(m.parameters(),lr=.003);x,y=xy('train',s)
 for _ in range(12):o.zero_grad();l=F.cross_entropy(m(x),y);l.backward();o.step()
 return m
def sc(m,x,y,f=None):
 if f:x=f(x,7)
 return float((m(x).argmax(1)==y).float().mean())
r={'depths':{},'manifest':{'seeds':[11,23,37],'v2':'raw-key softmax retrieval into query residual; temperature .1'}}
for d in (1,2,3):
 rows=[]
 for s in (11,23,37):
  m=fit(s,d);x,y=xy('test',s);rows.append({'seed':s,'valid':sc(m,x,y),'broken':sc(m,x,y,break_association),'value_shuffle':sc(m,x,y,shuffle_values),'permuted':sc(m,x[:,torch.randperm(x.shape[1]),:],y)})
 r['depths'][str(d)]=rows
out=Path('results/metrics/phase5_attention_v2');out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(r,indent=2));(out/'manifest.json').write_text(json.dumps(r['manifest'],indent=2));print(json.dumps(r,indent=2))
