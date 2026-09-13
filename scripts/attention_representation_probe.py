"""Frozen R0-R3 probes over unchanged AttentionBlock outputs."""
import json, torch
from pathlib import Path
from torch.nn import functional as F
from neuroforge.datasets.contextual_attention import ContextualAssociationDataset,_query_and_match,break_association,shuffle_values
from neuroforge.models import AttentionReadoutClassifier

def data(split,seed):
 d=ContextualAssociationDataset(split,360 if split=='train' else 240,seed);return d.features,d.targets
def states(m,x):
 h=m.encoder(x); out=[h.detach()]
 with torch.no_grad():
  for b in m.blocks: h,_=b(h);out.append(h.detach())
 return out
def role(h,x,kind):
 q,match=_query_and_match(x); i=q if kind=='query' else match
 if kind=='distractor': i=(match+1)%x.shape[1];i=torch.where(i==q,(i+1)%x.shape[1],i)
 return h[torch.arange(len(h)),i] if kind!='mean' else h.mean(1)
def probe(train_x,train_y,test_x,test_y,shuffle=False):
 if shuffle: train_y=train_y[torch.randperm(len(train_y))]
 p=torch.nn.Linear(train_x.shape[1],2);o=torch.optim.AdamW(p.parameters(),lr=.02)
 for _ in range(30):o.zero_grad();l=F.cross_entropy(p(train_x),train_y);l.backward();o.step()
 return float((p(test_x).argmax(1)==test_y).float().mean())
def fit(seed):
 torch.manual_seed(seed);m=AttentionReadoutClassifier('query',depth=3);o=torch.optim.AdamW(m.parameters(),lr=.003);x,y=data('train',seed)
 for _ in range(10):o.zero_grad();l=F.cross_entropy(m(x),y);l.backward();o.step()
 return m
r={'seeds':{},'manifest':{'seeds':[11,23,37],'probe':'frozen linear 24->2, 30 epochs','controls':['random_attention','label_shuffle','broken_association','value_shuffle','permutation']}}
for s in (11,23,37):
 m=fit(s);tx,ty=data('train',s);vx,vy=data('test',s); trained=states(m,tx),states(m,vx); torch.manual_seed(s+999);random=AttentionReadoutClassifier('query',depth=3);rand=states(random,tx),states(random,vx); rows={}
 for d in range(4):
  rows[str(d)]={}
  for k in ('mean','query','match','distractor'):
   rows[str(d)][k]=probe(role(trained[0][d],tx,k),ty,role(trained[1][d],vx,k),vy)
  rows[str(d)]['random_query']=probe(role(rand[0][d],tx,'query'),ty,role(rand[1][d],vx,'query'),vy)
  rows[str(d)]['shuffled_label']=probe(role(trained[0][d],tx,'query'),ty,role(trained[1][d],vx,'query'),vy,True)
 # control only query R3
 for name,fn in [('broken',break_association),('value_shuffle',shuffle_values),('permuted',lambda x,z:x[:,torch.randperm(x.shape[1]),:])]:
  cx=fn(vx,s); cs=states(m,cx)[3]; rows['controls_'+name]={'query':probe(role(trained[0][3],tx,'query'),ty,role(cs,cx,'query'),vy)}
 r['seeds'][str(s)]=rows
out=Path('results/metrics/phase4c_attention_representation');out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(r,indent=2));(out/'manifest.json').write_text(json.dumps(r['manifest'],indent=2));print(json.dumps(r,indent=2))
