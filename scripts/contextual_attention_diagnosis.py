"""Phase 4A diagnostic using unchanged Phase 3 specialists."""
import json, statistics
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from neuroforge.datasets.contextual_attention import ContextualAssociationDataset, break_association, shuffle_values
from neuroforge.models import StandaloneSpecialist
from neuroforge.evaluation.specialization import select_capacity

def loader(split, seed, length=12): return DataLoader(ContextualAssociationDataset(split,360 if split=='train' else 240,seed,length),batch_size=60,shuffle=split=='train')
def score(model,data,control=None,seed=0):
    n=c=0
    with torch.no_grad():
      for b in data:
        x=control(b['features'],seed) if control else b['features']; c+=int((model(x).argmax(1)==b['target']).sum()); n+=len(x)
    return c/n
def train(model,seed,length=12):
    o=torch.optim.AdamW(model.parameters(),lr=.003); t,v=loader('train',seed,length),loader('validation',seed,length)
    best=-1; state=None
    for _ in range(25):
      for b in t: o.zero_grad(); loss=F.cross_entropy(model(b['features']),b['target']); loss.backward(); o.step()
      a=score(model,v)
      if a>best: best=a; state={k:x.detach().clone() for k,x in model.state_dict().items()}
    model.load_state_dict(state); return model
out=Path('results/metrics/phase4a_contextual_attention'); out.mkdir(parents=True,exist_ok=True); cap=select_capacity(); result={'capacity':cap.__dict__,'primary':{},'depth':{},'manifest':{'seeds':[11,23,37],'train_lengths':[8,12],'unseen_test_length':16,'attention_audit':'one head, no mask, no positional encoding; residual projection then mean-pooling classifier'}}
for name in ('mlp','graph','attention'):
  rows=[]
  for seed in (11,23,37):
    torch.manual_seed(seed); m=train(StandaloneSpecialist(name,depth=cap.depths[name]),seed); test=loader('test',seed)
    rows.append({'seed':seed,'valid':score(m,test),'broken_association':score(m,test,break_association,seed),'value_shuffle':score(m,test,shuffle_values,seed),'permuted':score(m,test,lambda x,s:x[:,torch.randperm(x.shape[1]),:]) ,'longer_length':score(m,loader('test',seed,16))})
  result['primary'][name]=rows
for depth in (1,2,3):
  vals=[]
  for seed in (11,23,37): torch.manual_seed(seed); vals.append(score(train(StandaloneSpecialist('attention',depth=depth),seed),loader('test',seed)))
  result['depth'][str(depth)]=vals
(out/'summary.json').write_text(json.dumps(result,indent=2)); (out/'manifest.json').write_text(json.dumps(result['manifest'],indent=2)); print(json.dumps(result,indent=2))
