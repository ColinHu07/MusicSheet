"""Local pretrained ROSVOT baseline. Inference reads only audio + crop metadata.

Official weights, architecture and RMVPE. Reference notes/lyrics are deliberately
not loaded by this program. Evaluation is a separate, subsequent step.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(k,'2')
import numpy as np
import torch
import yaml

ROOT=Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", Path(__file__).resolve().parents[3])).resolve()
VENDOR=ROOT/'vendor/ROSVOT_20261117'
sys.path.insert(0,str(VENDOR))
from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor, regulate_boundary
from modules.pe.rmvpe import RMVPE
from utils.audio.mel import MelNet
from utils.audio.pitch_utils import norm_interp_f0,denorm_f0,f0_to_coarse,boundary2Interval

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(name, cls,device):
    folder=VENDOR/'checkpoints'/name
    hp=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
    # Official inference defaults, frozen before observing any comparison scores.
    hp['note_bd_threshold']=.85;hp['word_bd_threshold']=.5
    net=cls(hp).to(device).eval()
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(checkpoint['state_dict']['model'])
    return net,hp

def waveform(path,start,end):
    b=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(path),
        '-t',str(end-start),'-ar','24000','-ac','1','-f','f32le','pipe:1'])
    return np.frombuffer(b,dtype='<f4').copy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'outputs/transcription_research_20261117_v1')
    ap.add_argument('--songs',nargs='*');ap.add_argument('--variants',nargs='*',default=['vocals','mix','no_word'])
    args=ap.parse_args();root=args.root
    items=read(root/'audio_manifest.json')
    if args.songs:items=[i for i in items if i['id'] in args.songs]
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    model,hp=load('rosvot',MidiExtractor,device);wmodel,whp=load('rwbd',WordbdExtractor,device)
    melnet=MelNet(hp);melnet.to(device)
    pe=RMVPE(str(VENDOR/'checkpoints/rmvpe/model.pt'),device=device)
    save(root/'rosvot_provenance.json',dict(repo_commit='3c8332bf43adae35f6e4d64971862f2f6139b310',
        model_sha256=sha(VENDOR/'checkpoints/rosvot/model.pt'),word_model_sha256=sha(VENDOR/'checkpoints/rwbd/model.pt'),
        code_sha256=sha(__file__),device=device,torch=torch.__version__,reference_read=False))
    for item in items:
        for source in ['vocals','mix']:
            variants=[v for v in args.variants if (v=='mix')==(source=='mix')]
            if not variants:continue
            if all((root/('rosvot_'+v)/item['id']/'predictions.json').exists() for v in variants):continue
            wav=waveform(item[source],item['audio_start'],item['audio_end'])
            with torch.inference_mode():
                mel=melnet(wav);realT=mel.shape[1];T=((realT+15)//16)*16
                mel=torch.nn.functional.pad(mel,(0,0,0,T-realT))
                # Match the official batch inference pitch postprocessing.
                fs,uvs=pe.get_pitch_batch([wav],sample_rate=24000,hop_size=128,lengths=[realT],fmin=hp['f0_min'],fmax=hp['f0_max'])
                f0,uv=norm_interp_f0(fs[0]);f0=torch.nn.functional.pad(torch.tensor(f0,dtype=torch.float32),(0,T-realT))
                uv=torch.nn.functional.pad(torch.tensor(uv).long(),(0,T-realT))
                coarse=f0_to_coarse(denorm_f0(f0,uv))[None].to(device);uv=uv[None].to(device)
                nonpad=mel.abs().sum(-1)>0
                wb=wmodel(mel=mel[:,:,:whp['use_mel_bins']],pitch=coarse,uv=uv,non_padding=nonpad,train=False)
                for variant in variants:
                    dest=root/('rosvot_'+variant)/item['id']
                    if (dest/'predictions.json').exists():continue
                    word_bd=None if variant=='no_word' else wb['word_bd_pred']
                    feat=model.net(model.run_encoder(mel[:,:,:hp['use_mel_bins']],word_bd,coarse,uv,nonpad))
                    logits=(model.note_bd_out(feat).squeeze(-1)/model.note_bd_temperature).clamp(-16,16)
                    bd=regulate_boundary(logits,model.note_bd_threshold,model.note_bd_min_gap,word_bd,model.note_bd_ref_min_gap,nonpad)
                    lens,nlogits,npred=model.pitch_decoder(feat,bd,False)
                    intervals=boundary2Interval(bd[0,:realT]).cpu().numpy()*128/24000+item['audio_start']
                    pitches=npred[0,:int(lens[0])].cpu().numpy();confidence=nlogits.softmax(-1)[0].amax(-1).cpu().numpy()
                    notes=[dict(start_s=float(a),end_s=float(z),midi=int(m),velocity=82,
                        model_confidence=float(c),quantized=False) for (a,z),m,c in zip(intervals,pitches,confidence)
                        if m>0 and item['crop_start']<=a<item['crop_end'] and z>a]
                    save(dest/'predictions.json',dict(track=item['id'],notes=notes,reference_read=False,
                        human_lyrics_read=False,model='official ROSVOT',variant=variant))
                    np.savez_compressed(dest/'features.npz',feat=feat[0,:realT].cpu().numpy().astype(np.float32),
                        times_s=np.arange(realT)*128/24000+item['audio_start'],
                        boundary_logits=logits[0,:realT].cpu().numpy(),word_logits=wb['word_bd_logits'][0,:realT].cpu().numpy(),
                        boundaries=bd[0,:realT].cpu().numpy(),word_boundaries=wb['word_bd_pred'][0,:realT].cpu().numpy(),
                        f0=fs[0],mel=mel[0,:realT].cpu().numpy())
                    print(item['id'],variant,len(notes),'notes',flush=True)

if __name__=='__main__':main()
