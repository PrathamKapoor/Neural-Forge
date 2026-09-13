import json,statistics,torch
from pathlib import Path
from torch.nn import functional as F
from torch.utils.data import DataLoader
from neuroforge.datasets.contextual_attention import ContextualAssociationDataset,break_association,shuffle_values
from neuroforge.models import AttentionReadoutClassifier
def load(split,seed): return DataLoader(ContextualAssociationDataset(split,360 if split=='train' else 240,seed),batch_size=60,shuffle=split=='train')
def score(m,d,c=None):
 n=z=0
 with torch.no_grad():
  for b in d:
   x=c(b['features'],7) if c else b['features']; z+=int((m(x).argmax(1)==b['target']).sum());n+=len(x)
 return z/n
def fit(kind,seed,depth=1):
 torch.manual_seed(seed);m=AttentionReadoutClassifier(kind,depth);o=torch.optim.AdamW(m.parameters(),lr=.003);best=0;state=None
 for _ in range(10):
  for b in load('train',seed):o.zero_grad();l=F.cross_entropy(m(b['features']),b['target']);l.backward();o.step()
  a=score(m,load('validation',seed));
  if a>best:best=a;state={k:v.detach().clone() for k,v in m.state_dict().items()}
 m.load_state_dict(state);return m
r={'readouts':{},'depth':{},'probes':{}}
for k in ('mean','query','max','query_global'):
 rows=[]
 for s in (11,23,37):
  m=fit(k,s);t=load('test',s);rows.append({'seed':s,'valid':score(m,t),'broken':score(m,t,break_association),'value_shuffle':score(m,t,shuffle_values),'permuted':score(m,t,lambda x,z:x[:,torch.randperm(x.shape[1]),:])})
 r['readouts'][k]=rows
for d in (1,2,3):r['depth'][str(d)]=[score(fit('query',s,d),load('test',s)) for s in (11,23,37)]
Path('results/metrics/phase4b_attention_readout').mkdir(parents=True,exist_ok=True);Path('results/metrics/phase4b_attention_readout/summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
