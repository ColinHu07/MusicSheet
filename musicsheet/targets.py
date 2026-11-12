"""Conservative supervision masks: unknown audio is not silence."""
import numpy as np
from .acoustic import HOP, training_targets

def reviewed_targets(alignment,length):
    result=training_targets(alignment,length)
    times=np.arange(length)*HOP
    negatives=alignment.get('hard_negative_spans',[])
    for span in negatives:
        if not span.get('verified',False):
            continue
        if span.get('provenance') in ('unscored_intro','unscored_tail'):
            raise ValueError('Unscored audio cannot be inferred to be silence')
        a,b=float(span['start_s']),float(span['end_s'])
        mask=(times>=a)&(times<b)
        # A rest example must never overwrite a valid positive note.
        mask &= ~((result['pitch']>0)&(result['weights']>0))
        weight=min(.8,max(0.,float(span.get('confidence',span.get('weight',.5)))))
        result['pitch'][mask]=0
        result['weights'][mask]=np.maximum(result['weights'][mask],weight)
        result['register'][mask]=1.
        # Keep positive onset/offset Gaussian tails at accepted note edges.
        # Newly known rest interiors already carry zero boundary targets.
        result['boundary_weight'][mask]=np.maximum(result['boundary_weight'][mask],weight)
    return result
