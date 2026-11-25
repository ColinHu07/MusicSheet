"""Run released GAME ONNX model locally, without reference notes or lyrics."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch  # preload the installed CUDA/cuDNN libraries for ONNX Runtime
import onnxruntime as ort

ROOT=Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUN=ROOT/'outputs/transcription_research_20261117_v1'
MODEL=ROOT/'models/transcription/GAME_20261117_medium/GAME-1.0.3-medium-onnx'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--songs',nargs='*');ap.add_argument('--source',default='vocals',choices=['vocals','mix'])
    ap.add_argument('--model',type=Path,default=MODEL);ap.add_argument('--name',default='game_medium')
    ap.add_argument('--language-id',type=int,choices=[0,1],default=None)
    ap.add_argument('--boundaries',choices=['none','asr','rwbd'],default='none');args=ap.parse_args()
    ort.preload_dlls();so=ort.SessionOptions();so.intra_op_num_threads=2;so.log_severity_level=3
    sessions={n:ort.InferenceSession(str(args.model/(n+'.onnx')),so,providers=['CUDAExecutionProvider','CPUExecutionProvider']) for n in ['encoder','segmenter','estimator','bd2dur','dur2bd']}
    cfg=read(args.model/'config.json');sr=cfg['samplerate'];dt=cfg['timestep']
    items=read(RUN/'audio_manifest.json')
    if args.songs:items=[i for i in items if i['id'] in args.songs]
    save(RUN/(args.name+'_provenance.json'),dict(model=str(args.model),
        model_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model.glob('*.onnx')},
        reference_read=False,human_lyrics_read=False,steps=8,boundary_threshold=.3,presence_threshold=.2,
        language_policy='English=1 for wholly English songs; unknown/universal=0 for mixed/Korean (Korean absent from this checkpoint)',
        source=args.source,boundaries=args.boundaries,language_override=args.language_id))
    for item in items:
        dest=RUN/(args.name+'_'+args.source)/item['id']
        if (dest/'predictions.json').exists():continue
        raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(item['audio_start']),'-i',item[args.source],
            '-t',str(item['audio_end']-item['audio_start']),'-ar',str(sr),'-ac','1','-f','f32le','pipe:1'])
        wav=np.frombuffer(raw,dtype='<f4').copy()[None];dur=np.array([wav.shape[1]/sr],np.float32)
        x_seg,x_est,mask=sessions['encoder'].run(None,dict(waveform=wav,duration=dur))
        # One whole-clip region has no internal known boundaries. Avoid the
        # ONNX CUDA converter's empty-axis reduction bug for N=1.
        known=np.zeros_like(mask,dtype=bool)
        if args.boundaries=='asr':
            words=read(ROOT/f'outputs/syllable_note_lyrics_20261112_v2/{item["id"]}/syllable_alignment.json')['words']
            times=[w[k] for w in words if w.get('asr_confidence',0)>=.4 and not w.get('asr_repetition_review') for k in ['asr_start_s','asr_end_s']]
        elif args.boundaries=='rwbd':
            with np.load(RUN/'rosvot_vocals'/item['id']/'features.npz') as z:times=z['times_s'][z['word_boundaries']>0]
        else:times=[]
        for tm in times:
            frame=round((float(tm)-item['audio_start'])/dt)
            if 0<frame<mask.shape[1]-1:known[0,frame]=True
        print(item['id'],'encoded',mask.shape[1],'frames',flush=True)
        bd=known.copy();language=1 if item['id'] in ['i_do','gods','lifes_too_short','dynamite'] else 0
        if args.language_id is not None:
            language=args.language_id
        for step in range(8):
            bd=sessions['segmenter'].run(None,dict(x_seg=x_seg,language=np.array([language],np.int64),
                known_boundaries=known,prev_boundaries=bd,t=np.array([step/8],np.float32),
                maskT=mask,threshold=np.array(.3,np.float32),radius=np.array(2,np.int64)))[0]
        durations,nmask=sessions['bd2dur'].run(None,dict(boundaries=bd,maskT=mask))
        presence,pitch=sessions['estimator'].run(None,dict(x_est=x_est,boundaries=bd,maskT=mask,maskN=nmask,threshold=np.array(.2,np.float32)))
        ends=np.cumsum(durations[0])+item['audio_start'];starts=np.r_[item['audio_start'],ends[:-1]]
        assert abs(float(ends[-1])-item['audio_end'])<.03,(item['id'],ends[-1],item['audio_end'])
        assert np.isfinite(pitch).all()
        notes=[dict(start_s=float(a),end_s=float(z),midi=int(round(float(p))),pitch_float=float(p),velocity=82,quantized=False)
            for a,z,p,v in zip(starts,ends,pitch[0],presence[0]) if v and item['crop_start']<=a<item['crop_end'] and z>a]
        save(dest/'predictions.json',dict(track=item['id'],notes=notes,reference_read=False,human_lyrics_read=False,model=args.name,language_id=language))
        np.savez_compressed(dest/'features.npz',x_seg=x_seg[0],x_est=x_est[0],mask=mask[0],boundaries=bd[0],
            times_s=np.arange(mask.shape[1])*dt+item['audio_start'],presence=presence[0],pitch=pitch[0],durations=durations[0])
        print(item['id'],args.source,len(notes),'notes',flush=True)
if __name__=='__main__':main()
