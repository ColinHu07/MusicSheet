"""Original weak-label selection metrics, kept for reproducibility."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter1d
from .acoustic import MIN_MIDI, MAX_MIDI, HOP

def event_metrics(reference,predicted,tolerance=.10,offsets=False,pitch_class=False):
    count=len(reference);number=len(predicted);cost=np.full((count,number+count),1000.)
    for i,a in enumerate(reference):
        for j,b in enumerate(predicted):
            diff=abs(a['start_s']-b['start_s'])
            pitch=(a['midi']-b['midi'])%12==0 if pitch_class else a['midi']==b['midi']
            release=abs(a['end_s']-b['end_s'])<=max(.10,.20*(a['end_s']-a['start_s']))
            if diff<=tolerance and pitch and (release or not offsets):cost[i,j]=diff
    rows,columns=linear_sum_assignment(cost)
    pairs=[(int(i),int(j)) for i,j in zip(rows,columns) if j<number and cost[i,j]<999]
    matched={i for i,j in pairs}
    repeats={i for i in range(1,count) if reference[i]['midi']==reference[i-1]['midi'] and reference[i]['start_s']-reference[i-1]['end_s']<=.5}
    ratios=[(predicted[j]['end_s']-predicted[j]['start_s'])/(reference[i]['end_s']-reference[i]['start_s']) for i,j in pairs]
    return dict(reference=count,predicted=number,matched=len(pairs),precision=len(pairs)/max(number,1),
        recall=len(pairs)/max(count,1),f1=2*len(pairs)/max(count+number,1),
        repeat_reference=len(repeats),repeat_matched=len(repeats&matched),
        duration_ratio_median=float(np.median(ratios)) if ratios else None)


def evaluate(item,notes,probabilities=None):
    aligned=item['alignment'];valid=[n for n in aligned['notes'] if n['supervise']]
    # Report both all printable reference attacks and the support-filtered set.
    all_notes=[n for n in aligned['notes'] if MIN_MIDI<=n['midi']<=MAX_MIDI]
    start=all_notes[0]['start_s'];end=all_notes[-1]['end_s']
    pred=[n for n in notes if start-.1<=n['start_s']<end]
    target=item['targets'];known=maximum_filter1d((target['weights']>0).astype(int),11)>0
    accepted_pred=[n for n in pred if known[min(len(known)-1,max(0,round(n['start_s']/HOP)))]]
    reliable=[n for n in valid if n.get('register_supervision_weight',1)>.5]
    register_known=maximum_filter1d(((target['weights']>0)&(target['register']>.5)).astype(int),11)>0
    reliable_pred=[n for n in pred if register_known[min(len(known)-1,max(0,round(n['start_s']/HOP)))]]
    result=dict(all_reference=event_metrics(all_notes,pred),accepted_reference=event_metrics(reliable,reliable_pred),
        onset_offset=event_metrics(reliable,reliable_pred,offsets=True),pitch_class=event_metrics(valid,accepted_pred,pitch_class=True),
        strict_50ms=event_metrics(reliable,reliable_pred,tolerance=.05),scope_seconds=end-start,
        unknown_prediction_onsets_excluded=len(pred)-len(accepted_pred),register_reliable_notes=len(reliable),
        accepted_notes=len(valid),all_notes=len(all_notes))
    mask=(target['weights']>.35)&(target['register']>.5)&(target['pitch']>0)
    if probabilities is not None:
        argmax=probabilities.argmax(axis=0)
        result['accepted_voiced_pitch_accuracy']=float(np.mean(argmax[mask]==target['pitch'][mask])) if mask.any() else 0.
    return result


def macro(results):
    return dict(onset_f1=float(np.mean([r['accepted_reference']['f1'] for r in results])),
        onset_offset_f1=float(np.mean([r['onset_offset']['f1'] for r in results])),
        all_reference_f1=float(np.mean([r['all_reference']['f1'] for r in results])),
        pitch_class_f1=float(np.mean([r['pitch_class']['f1'] for r in results])),
        accepted_voiced_pitch_accuracy=float(np.mean([r.get('accepted_voiced_pitch_accuracy',0) for r in results])),
        repeated_matched=sum(r['accepted_reference']['repeat_matched'] for r in results),
        repeated_reference=sum(r['accepted_reference']['repeat_reference'] for r in results))
