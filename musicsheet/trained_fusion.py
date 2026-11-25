"""Apply the trained fusion model to acoustic files; no annotations/score access."""
import argparse,hashlib,json,sys
from pathlib import Path
import joblib,numpy as np
import copy
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def build_candidates(sources,tolerance=.08,median_anchor=False):
    candidates=[]
    for mi,notes in enumerate(sources):
        if not candidates:
            candidates=[dict(note=copy.deepcopy(n),members={str(mi):n}) for n in notes];continue
        cost=np.full((len(candidates),len(notes)),1e6)
        for i,c in enumerate(candidates):
            for j,n in enumerate(notes):
                delta=abs(c['note']['start_s']-n['start_s'])
                if c['note']['midi']%12==n['midi']%12 and delta<=tolerance:cost[i,j]=delta
        ii,jj=linear_sum_assignment(cost);used=set()
        for i,j in zip(ii,jj):
            if cost[i,j]>=1e5:continue
            candidates[i]['members'][str(mi)]=notes[j];used.add(j)
        candidates.extend(dict(note=copy.deepcopy(n),members={str(mi):n}) for j,n in enumerate(notes) if j not in used)
    if median_anchor:
        for c in candidates:
            for key in ['start_s','end_s']:c['note'][key]=float(np.median([n[key] for n in c['members'].values()]))
    return sorted(candidates,key=lambda c:c['note']['start_s'])

def features(candidates,sources,v,p,b,lyrics):
    t=v['times_s'];flux=np.maximum(v['onset_flux'],0);flux=flux/max(np.percentile(flux,95),1e-6)
    rms=v['rms']/max(np.percentile(v['rms'],95),1e-6)
    X=[]
    for c in candidates:
        n=c['note'];a,z,m=n['start_s'],n['end_s'],n['midi'];pc=m%12
        inside=(t>=a)&(t<z);before=(t>=a-.10)&(t<a);after=(t>=a)&(t<a+.10);near=abs(t-a)<=.06
        if not inside.any():inside[np.argmin(abs(t-a))]=True
        vals=[min(z-a,2),float(n.get('model_confidence',.5)),len(c['members'])]
        for mi,src in enumerate(sources):
            same=[q for q in src if q['midi']%12==pc]
            closest=min(same,key=lambda q:abs(a-q['start_s']),default=None)
            vals.extend([float(str(mi) in c['members']),
                min(abs(closest['start_s']-a),1) if closest else 1,
                min(abs(closest['end_s']-z),1) if closest else 1,
                sum(abs(q['start_s']-a)<.15 for q in src),
                max([max(0,min(z,q['end_s'])-max(a,q['start_s']))/max(.02,z-a) for q in same],default=0)])
        f0=v['rmvpe_midi'][inside];good=np.isfinite(f0)&(v['rmvpe_voiced'][inside]>.5)
        diff=abs((np.nan_to_num(f0,nan=0)-m+6)%12-6)
        vals.extend([float(good.mean()),float(np.mean(good&(diff<.65))),
            float(np.mean(good&(abs(np.nan_to_num(f0,nan=0)-m)<.65))),
            min(float(np.std(f0[good])) if good.any() else 12,12)/12,
            float(np.mean(v['rmvpe_confidence'][inside])),float(np.mean(v['vocal_fullmix_rms_ratio'][inside])),
            float(rms[inside].mean()),float(rms[inside].max()),float(flux[near].max()),float(flux[inside].mean())])
        states=np.arange(1,61)[np.arange(36,96)%12==pc];pcp=p[states].sum(0)
        for arr in [pcp,p[0],p[np.clip(int(m)-35,1,60)],b[0],b[1]]:
            vals.extend([float(arr[inside].mean()),float(arr[inside].max()),
                float(arr[near].max()),float(arr[before].mean()) if before.any() else 0,
                float(arr[after].mean()) if after.any() else 0])
        # Automatic lyric timing is acoustic evidence, not corrected lyric labels.
        words=[w for w in lyrics.get('words',[]) if not w.get('asr_repetition_review')]
        covering=[w for w in words if w['asr_start_s']<=a<=w['asr_end_s']]
        vals.extend([max([float(w.get('asr_confidence',0)) for w in covering],default=0),
            min([abs(a-w['asr_start_s']) for w in words]+[1]),
            min([abs(a-w['asr_end_s']) for w in words]+[1])])
        X.append(vals)
    return np.nan_to_num(np.array(X,np.float32),nan=0,posinf=0,neginf=0)

def select(candidates,scores,threshold):
    eligible=[(c,float(p)) for c,p in zip(candidates,scores) if p>=threshold]
    chosen=[]
    for c,p in sorted(eligible,key=lambda q:-q[1]):
        n=c['note']
        # Competing pitches at effectively the same onset are alternatives.
        if any(abs(n['start_s']-q[0]['note']['start_s'])<.065 for q in chosen):continue
        chosen.append((c,p))
    out=[]
    for c,p in sorted(chosen,key=lambda q:q[0]['note']['start_s']):
        n=copy.deepcopy(c['note']);n['fusion_score']=p;n['fusion_sources']=sorted(c['members']);n['quantized']=False
        if out and out[-1]['end_s']>n['start_s']:out[-1]['end_s']=n['start_s']
        out.append(n)
    return out

def infer(model_dir,baseline,rosvot,game,vocal_features,probabilities,lyrics):
    model_dir=Path(model_dir);meta=read(model_dir/'model.json');models=joblib.load(model_dir/'scorers.joblib')['models']
    sources=[read(p)['notes'] for p in [baseline,rosvot,game]]
    cs=build_candidates(sources,tolerance=meta['merge_tolerance_s'],median_anchor=meta['median_anchor'])
    with np.load(vocal_features) as z:v={k:z[k] for k in z.files}
    with np.load(probabilities) as z:p,b=z['pitch'],z['boundaries']
    if len(v['times_s'])!=p.shape[1] or b.shape[1]!=p.shape[1]:raise ValueError('Acoustic and probability clocks differ')
    X=features(cs,sources,v,p,b,read(lyrics));score=np.mean([m.predict_proba(X)[:,1] for m in models],axis=0) if cs else []
    notes=select(cs,score,meta['threshold'])
    for n in notes:
        n['uncertainty']=dict(event_score=n['fusion_score'],calibrated_probability=False,register_verified=False,
            below_violin_range=n['midi']<55,review_recommended=n['fusion_score']<.5)
    return dict(notes=notes,candidates=[dict(c,score=float(p)) for c,p in zip(cs,score)],reference_used_at_inference=False,
        model=str(model_dir),threshold=meta['threshold'],model_sha256=hashlib.sha256((model_dir/'scorers.joblib').read_bytes()).hexdigest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,default=ROOT/'checkpoints/event_fusion_20261117_v1')
    for name in ['baseline','rosvot','game','vocal-features','probabilities','lyrics','out']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous predictions; use a new output path')
    result=infer(a.model,a.baseline,a.rosvot,a.game,a.vocal_features,a.probabilities,a.lyrics);save(a.out,result);print(f'Saved {len(result["notes"])} notes to {a.out}')
if __name__=='__main__':main()
