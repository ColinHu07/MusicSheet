"""Soft aligned lyric features; no forced one-note-per-syllable rule."""
import json
import numpy as np

def lyric_input(folder,times):
    z=np.load(folder/'phonetics.npz');p=z['probabilities'].astype(np.float32);ts=z['times_s']
    aligned=json.loads((folder/'syllable_alignment.json').read_text(encoding='utf-8'))
    ppg=np.stack([np.interp(times,ts,p[:,k],left=0,right=0) for k in range(p.shape[1])]);extra=np.zeros((8,len(times)),np.float32)
    for s in aligned['syllables']:
        if not s['supported_input']:continue
        extra[0]=np.maximum(extra[0],np.exp(-.5*((times-s['nucleus_s'])/.04)**2)*s['confidence'])
        extra[1]=np.maximum(extra[1],np.exp(-.5*((times-s['attack_estimate_s'])/.04)**2)*s['confidence'])
        word=aligned['words'][s['word_index']];mask=(times>=word['asr_start_s'])&(times<word['asr_end_s'])
        extra[2,mask]=max(0.,min(1.,word['asr_confidence']));extra[3,mask]=min(word.get('estimated_syllables_per_second',0)/10,1)
        extra[4,mask]=min(word.get('syllable_count',0)/5,1);extra[5,mask]=1
    extra[6]=1-ppg[0];extra[7]=-(ppg*np.log(np.maximum(ppg,1e-7))).sum(0)/np.log(ppg.shape[0])
    return np.concatenate([ppg,extra]).astype(np.float32)
