"""Acoustic melody network, independent pitch/onset/offset outputs and decoder."""
import numpy as np
import torch
from torch import nn
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

MIN_MIDI=36
MAX_MIDI=95
PITCHES=MAX_MIDI-MIN_MIDI+1
HOP=.02
PITCH_GROUPS=7
INPUTS=PITCHES*PITCH_GROUPS+5


def acoustic_features(mix,vocal):
    """Only recording-derived arrays; no score, song ID, tempo or alignment."""
    if not np.array_equal(mix['times_s'],vocal['times_s']):raise ValueError('Audio grids differ')
    sl=slice(MIN_MIDI-24,MAX_MIDI-24+1)
    mc=mix['log_cqt'][:,sl];vc=vocal['log_cqt'][:,sl]
    mf=np.maximum(np.diff(mc,axis=0,prepend=mc[:1]),0)
    vf=np.maximum(np.diff(vc,axis=0,prepend=vc[:1]),0)
    f0=vocal['rmvpe_midi']
    support=np.exp(-.5*((f0[:,None]-np.arange(MIN_MIDI,MAX_MIDI+1)[None,:])/.65)**2)
    support=np.nan_to_num(support,nan=0)*vocal['rmvpe_voiced'][:,None]
    groups=[mc,mix['harmonic_salience'][:,sl],vc,vocal['harmonic_salience'][:,sl],mf,vf,support]
    extra=np.stack([np.log(np.maximum(mix['rms'],1e-5)),np.log(np.maximum(vocal['rms'],1e-5)),
                    mix['onset_flux'],vocal['onset_flux'],np.clip(vocal['vocal_fullmix_rms_ratio'],0,2)],axis=1)
    result=np.concatenate(groups+[extra],axis=1).T.astype(np.float32)
    if result.shape[0]!=INPUTS or not np.isfinite(result).all():raise ValueError('Bad acoustic feature input')
    return result


def training_targets(alignment,length):
    """Weak arrangement targets; unscored/uncertain spans remain unsupervised."""
    t=np.arange(length)*HOP
    pitch=np.zeros(length,np.int64);weights=np.zeros(length,np.float32);register=np.zeros(length,np.float32)
    boundary=np.zeros((2,length),np.float32);boundary_weight=np.zeros(length,np.float32)
    notes=alignment['notes']
    for i,n in enumerate(notes):
        if not n['supervise'] or not MIN_MIDI<=n['midi']<=MAX_MIDI:continue
        a,b=n['start_s'],n['end_s'];confidence=n['alignment_confidence']
        mask=(t>=a)&(t<b)
        pitch[mask]=n['midi']-MIN_MIDI+1
        weights[mask]=confidence
        register[mask]=n.get('register_supervision_weight',1.)
        # Boundaries carry alignment uncertainty; pitch supervision near a
        # boundary is weaker than in the note's stable interior.
        edge=(np.abs(t-a)<.04)|(np.abs(t-b)<.04)
        weights[mask&edge]*=.4
        for channel,event,sigma in [(0,a,.04),(1,b,.06)]:
            pulse=np.exp(-.5*((t-event)/sigma)**2);pulse[abs(t-event)>.16]=0
            boundary[channel]=np.maximum(boundary[channel],pulse)
        boundary_weight[(t>=a-.12)&(t<=b+.12)]=np.maximum(boundary_weight[(t>=a-.12)&(t<=b+.12)],confidence)
        # Internal printed rests are weak negatives only between two accepted
        # notes. No labels are fabricated for an unscored intro or later rap.
        if i+1<len(notes) and notes[i+1]['supervise'] and n.get('printed_rest_after_valid',False):
            next_start=notes[i+1]['start_s']
            rest=(t>=b)&(t<next_start)
            if 0<next_start-b<4:
                weights[rest]=.30*min(confidence,notes[i+1]['alignment_confidence'])
                register[rest]=1.
                boundary_weight[rest]=np.maximum(boundary_weight[rest],.3)
    # Gaussian boundary tails must not turn a neighbouring rejected/unknown
    # note into a negative onset example.
    boundary_weight[weights<=0]=0
    return dict(pitch=pitch,weights=weights,register=register,boundary=boundary,boundary_weight=boundary_weight)


class ChannelNorm(nn.Module):
    def __init__(self,channels):
        super().__init__();self.norm=nn.LayerNorm(channels)

    def forward(self,x):
        return self.norm(x.transpose(1,2)).transpose(1,2)


class MelodyNetwork(nn.Module):
    def __init__(self,hidden=80):
        super().__init__()
        self.input=nn.Conv1d(INPUTS,hidden,5,padding=2)
        self.layers=nn.ModuleList([nn.Sequential(
            nn.Conv1d(hidden,hidden,5,padding=2*d,dilation=d),
            ChannelNorm(hidden),nn.GELU(),nn.Dropout(.18)) for d in (1,2,4,8,16)])
        self.pitch_head=nn.Conv1d(hidden,PITCHES+1,1)
        self.boundary_head=nn.Conv1d(hidden,2,1)

    def forward(self,x):
        x=torch.nn.functional.gelu(self.input(x))
        for layer in self.layers:x=x+layer(x)
        return self.pitch_head(x),self.boundary_head(x)


def predict(model,x,mean,scale,device='cpu'):
    chunks=[];boundaries=[];context=68
    model.eval()
    with torch.no_grad():
        for start in range(0,x.shape[1],2048):
            a=max(0,start-context);b=min(x.shape[1],start+2048+context)
            value=torch.from_numpy(((x[:,a:b]-mean)/scale)[None]).to(device)
            pp,bb=model(value)
            left=start-a;right=left+min(2048,x.shape[1]-start)
            chunks.append(torch.softmax(pp,dim=1)[0,:,left:right].cpu().numpy())
            boundaries.append(torch.sigmoid(bb)[0,:,left:right].cpu().numpy())
    return np.concatenate(chunks,axis=1),np.concatenate(boundaries,axis=1)


def decode(probabilities,boundaries,onset_threshold=.45,transition=2.,minimum_duration=.08):
    """Continuity decoding and independent repeated onsets; no beat snapping."""
    probabilities=np.maximum(probabilities,1e-7)
    cost=-np.log(probabilities);states,length=cost.shape
    midi=np.r_[-1,np.arange(MIN_MIDI,MAX_MIDI+1)]
    change=np.full((states,states),transition)
    change[0,:]=transition*.65;change[:,0]=transition*.65
    change[1:,1:]+=.025*np.minimum(abs(midi[1:,None]-midi[None,1:]),12)
    np.fill_diagonal(change,0)
    previous=cost[:,0];back=np.empty((length,states),np.int16)
    for frame in range(1,length):
        local_change=change*(1-.45*boundaries[0,frame])
        # Offset confidence lowers the cost of releasing into silence. Pitch
        # evidence must still agree; an offset alone cannot force a false cut.
        local_change[:,0]*=1-.7*boundaries[1,frame]
        candidates=previous[:,None]+local_change
        ancestors=candidates.argmin(axis=0);back[frame]=ancestors
        previous=candidates[ancestors,np.arange(states)]+cost[:,frame]
    labels=np.zeros(length,int);state=int(previous.argmin())
    for frame in range(length-1,-1,-1):
        labels[frame]=state;state=back[frame,state]
    changes=np.r_[0,np.flatnonzero(np.diff(labels))+1,length]
    attacks,_=find_peaks(boundaries[0],height=onset_threshold,distance=5,prominence=.04)
    notes=[];min_frames=max(2,round(minimum_duration/HOP))
    for a,b in zip(changes[:-1],changes[1:]):
        state=labels[a]
        if state==0 or b-a<min_frames:continue
        edges=[a,b]
        for at in sorted(attacks[(attacks>=a+min_frames)&(attacks<=b-min_frames)],key=lambda t:boundaries[0,t],reverse=True):
            if min(abs(at-e) for e in edges)>=min_frames:edges.append(int(at))
        edges.sort()
        for left,right in zip(edges[:-1],edges[1:]):
            confidence=float(probabilities[state,left:right].mean())
            notes.append(dict(start_s=round(left*HOP,6),end_s=round(right*HOP,6),
                midi=int(midi[state]),velocity=82,model_confidence=confidence,
                repeated_attack=bool(left!=a),extra_review=confidence<.45))
    return notes


def baseline_probabilities(mix,vocal):
    """Frozen RMVPE/spectral evidence baseline, independent of the supplied score."""
    sl=slice(MIN_MIDI-24,MAX_MIDI-24+1)
    salience=np.maximum(np.expm1(mix['harmonic_salience'][:,sl]),0)
    vocal_salience=np.maximum(np.expm1(vocal['harmonic_salience'][:,sl]),0)
    salience/=np.maximum(salience.max(axis=1,keepdims=True),1e-5)
    vocal_salience/=np.maximum(vocal_salience.max(axis=1,keepdims=True),1e-5)
    f0=vocal['rmvpe_midi'];voiced=vocal['rmvpe_voiced'].astype(float)
    ratio=np.clip((vocal['vocal_fullmix_rms_ratio']-.1)/.25,0,1)
    pitch=np.arange(MIN_MIDI,MAX_MIDI+1)
    direct=np.nan_to_num(np.exp(-.5*((f0[:,None]-pitch[None,:])/.6)**2),nan=0)*voiced[:,None]
    notes=(.05+.9*salience)*(1-ratio[:,None])+ratio[:,None]*(.02+.7*vocal_salience+2.5*direct)
    silence=.08+.8*(1-voiced)*ratio+.8*(mix['rms']<.015)
    raw=np.concatenate([silence[:,None],notes],axis=1)
    raw/=raw.sum(axis=1,keepdims=True)
    flux=np.maximum(mix['onset_flux']*(1-ratio),vocal['onset_flux']*ratio)
    flux=np.clip(flux/max(float(np.percentile(flux,96)),1e-6),0,1)
    offset=np.maximum(-np.diff(vocal['rms'],append=vocal['rms'][-1]),0)
    offset=np.clip(offset/max(float(np.percentile(offset,95)),1e-6),0,1)
    return raw.T.astype(np.float32),np.stack([flux,offset]).astype(np.float32)
