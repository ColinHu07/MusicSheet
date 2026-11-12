"""Acoustic articulation head; no linguistic labels are inferred here."""
import numpy as np
import torch
from torch import nn
from .acoustic import ChannelNorm

def features(item):
    v=item['vocal'];m=v['log_mel'].astype(np.float32)
    d=np.diff(m,axis=0,prepend=m[:1])
    f=np.nan_to_num(v['rmvpe_midi']);delta=np.clip(np.diff(f,prepend=f[:1]),-12,12)/12
    return np.concatenate([m,d,np.log(np.maximum(v['rms'],1e-5))[:,None],delta[:,None],item['bb'].T],axis=1).T.astype(np.float32)


class ArticulationHead(nn.Module):
    def __init__(self,inputs=164):
        super().__init__();self.input=nn.Conv1d(inputs,48,5,padding=2)
        self.layers=nn.ModuleList([nn.Sequential(nn.Conv1d(48,48,5,padding=2*d,dilation=d),ChannelNorm(48),nn.GELU(),nn.Dropout(.12)) for d in (1,2,4,8)])
        self.output=nn.Conv1d(48,1,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,old):
        x=torch.nn.functional.gelu(self.input(x))
        for layer in self.layers:x=x+layer(x)
        correction=3*torch.tanh(self.output(x)[:,0]/3)
        return old+correction,correction


def infer(model,item,mean,scale):
    chunks=[];model.eval();x=item['ax'];old=np.log(np.clip(item['bb'][0],1e-5,1-1e-5)/(1-np.clip(item['bb'][0],1e-5,1-1e-5)))
    with torch.no_grad():
        for start in range(0,x.shape[1],2048):
            a=max(0,start-40);b=min(x.shape[1],start+2048+40)
            logits,_=model(torch.from_numpy(((x[:,a:b]-mean)/scale)[None]).to(next(model.parameters()).device),torch.from_numpy(old[None,a:b]).to(next(model.parameters()).device))
            chunks.append(torch.sigmoid(logits)[0,start-a:start-a+min(2048,x.shape[1]-start)].cpu().numpy())
    bb=item['bb'].copy();bb[0]=np.concatenate(chunks);return bb
