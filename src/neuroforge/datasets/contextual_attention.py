"""Leak-resistant, permutation-invariant content-addressed association task."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset

_OFFSET={"train":0,"validation":10000,"test":20000}
class ContextualAssociationDataset(Dataset):
    def __init__(self, split:str, samples:int, seed:int, sequence_length:int=12):
        if split not in _OFFSET or samples%2 or sequence_length<6: raise ValueError("even samples, known split, length >= 6")
        g=torch.Generator().manual_seed(seed+_OFFSET[split]); self.targets=torch.arange(samples)%2
        x=torch.randn(samples,sequence_length,8,generator=g)*.03
        for i,y in enumerate(self.targets):
            pos=torch.randperm(sequence_length,generator=g); q,m=int(pos[0]),int(pos[1]); key=torch.randn(3,generator=g); key=key/key.norm()
            distractors=torch.randn(sequence_length,3,generator=g); distractors=distractors/distractors.norm(dim=1,keepdim=True); x[i,:,0:3]=distractors; x[i,q,:3]=key; x[i,m,:3]=key
            x[i,q,4]=1.0  # query marker, not an identifier
            value=1.0 if y else -1.0; x[i,m,3]=value
            # Three distractor values exactly cancel the relevant value globally.
            for p,v in zip(pos[2:5].tolist(),[-value,-value,value]): x[i,p,3]=v
        self.features=x; self.split=split
    def __len__(self): return len(self.targets)
    def __getitem__(self,i): return {"features":self.features[i],"target":self.targets[i]}

def _query_and_match(x):
    q=x[:,:,4].argmax(1); keys=x[:,:,:3]; query=keys[torch.arange(len(x)),q]
    score=(keys*query[:,None,:]).sum(-1); score[torch.arange(len(x)),q]=-float('inf'); return q,score.argmax(1)
def association_oracle(x):
    _,match=_query_and_match(x); return (x[torch.arange(len(x)),match,3]>0).long()
def break_association(x,seed):
    out=x.clone(); q=out[:,:,4].argmax(1); out[torch.arange(len(x)),q,:3]*=-1; return out
def shuffle_values(x,seed):
    out=x.clone(); g=torch.Generator().manual_seed(seed)
    for i in range(len(out)): out[i,:,3]=out[i,torch.randperm(out.shape[1],generator=g),3]
    return out
