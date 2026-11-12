"""Audio-only articulation and event-register heads. No reference imports.

The event pitch head preserves boundaries. It learns octave and neighboring
whole-tone corrections; uncertain decisions leave the original pitch unchanged.
"""
import numpy as np
import torch
from torch import nn
from .articulation import ArticulationHead, features, infer

SHIFTS = np.array([-12, -10, -2, 0, 2, 10, 12, 14])

def event_features(item, notes):
    result=[];length=item['pp'].shape[1]
    pitches=np.array([n['midi'] for n in notes])
    spectra=[source[key] for source,key in [(item['vocal'],'log_cqt'),(item['vocal'],'harmonic_salience'),(item['mix'],'log_cqt'),(item['mix'],'harmonic_salience')]]
    f0_all=item['vocal']['rmvpe_midi'];voiced_all=item['vocal']['rmvpe_voiced']
    for j,n in enumerate(notes):
        a=max(0,int(round(n['start_s']/.02)));b=min(length,max(a+1,int(round(n['end_s']/.02))))
        midi=n['midi'];parts=[]
        for values in spectra:
            spectrum=values[a:b].mean(axis=0);idx=midi+np.arange(-24,25)-24
            parts.append(np.where((idx>=0)&(idx<len(spectrum)),spectrum[np.clip(idx,0,len(spectrum)-1)],0))
        posterior=item['pp'][:,a:b].mean(axis=1);idx=midi+np.arange(-24,25)-35
        parts.append(np.log(np.maximum(np.where((idx>0)&(idx<len(posterior)),posterior[np.clip(idx,0,len(posterior)-1)],0),1e-5)))
        f0=f0_all[a:b];f0=f0[np.isfinite(f0)]
        nearby=pitches[max(0,j-4):j+5]
        context=[(pitches[np.clip(j+d,0,len(notes)-1)]-midi)/12 for d in (-2,-1,1,2)]
        parts.append(np.array([midi/12,n['end_s']-n['start_s'],*context,(np.median(nearby)-midi)/12,
                               np.clip((np.median(f0)-midi)/12,-3,3) if len(f0) else 0,
                               float(voiced_all[a:b].mean()),posterior[0]],np.float32))
        result.append(np.concatenate(parts))
    return np.asarray(result,np.float32).reshape(len(notes),255)

class RegisterHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(255,96),nn.GELU(),nn.Dropout(.12),nn.Linear(96,48),nn.GELU(),nn.Linear(48,len(SHIFTS)))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():self.net[-1].bias[3]=2.
    def forward(self,x):return self.net(x)

def correct_register(model,item,notes,mean,scale,threshold=.99):
    if not notes:return [],[]
    model.eval()
    with torch.no_grad():
        probs=torch.softmax(model(torch.from_numpy((event_features(item,notes)-mean)/scale).to(next(model.parameters()).device)),dim=1).cpu().numpy()
    corrected=[];audit=[]
    for i,(n,p) in enumerate(zip(notes,probs)):
        choice=int(p.argmax());shift=int(SHIFTS[choice]) if p[choice]>=threshold else 0
        if not 36<=n['midi']+shift<=95:shift=0
        c=dict(n,midi=n['midi']+shift,pre_register_midi=n['midi'],register_shift=shift,register_confidence=float(p[choice]))
        corrected.append(c)
        if shift:audit.append(dict(event_index=i,start_s=n['start_s'],old_midi=n['midi'],new_midi=c['midi'],confidence=float(p[choice])))
    assert [(n['start_s'],n['end_s']) for n in notes]==[(n['start_s'],n['end_s']) for n in corrected]
    return corrected,audit
