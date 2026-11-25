from pathlib import Path
import argparse
import sys
import json
import hashlib
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from torch import nn
from musicsheet.events import NoteEventNetwork, features, previous_boundaries, predict, decode_events
from musicsheet.register import ArticulationHead, RegisterHead
from musicsheet.audio import save_json as save

def load_previous():
    folder = ROOT / "checkpoints/joint_pitch_syllables_20261101_v2"
    state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    art, register = ArticulationHead().eval(), RegisterHead().eval()
    art.load_state_dict(state["articulation"])
    register.load_state_dict(state["register"])
    with np.load(folder / "normalization.npz") as norm:
        return art, register, dict(norm)

RELATIONS = ['new_syllable_attack', 'same_syllable_pitch_change',
             'same_pitch_new_syllable', 'within_note_continuation']
INPUTS = 245
HOP = .02
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class SyllableNoteNetwork(NoteEventNetwork):
    def __init__(self, inputs=INPUTS, hidden=48):
        super().__init__(inputs=inputs, hidden=hidden)
        # 61 pitch, two boundary residuals, three acoustic event heads,
        # then the four new linguistic/note-relation heads.
        self.output = nn.Conv1d(hidden, 70, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


def lyric_features(folder, times):
    """Sixteen input channels from audio-derived lyric/phone hypotheses only."""
    path = folder / 'syllable_alignment.json'
    aligned = json.loads(path.read_text(encoding='utf-8-sig'))
    asr_path = folder / 'lyrics_asr.json'
    if asr_path.exists() and aligned.get('asr_sha256'):
        if aligned['asr_sha256'] != digest(asr_path):
            raise ValueError(f'ASR changed after phone alignment: {folder}')
    extra = np.zeros((16, len(times)), np.float32)
    words = aligned.get('words', [])
    for word in words:
        if word.get('asr_repetition_review') or word.get('asr_degenerate_repetition'):
            continue
        a, b = float(word['asr_start_s']), float(word['asr_end_s'])
        conf = float(np.clip(word.get('asr_confidence', 0), 0, 1))
        mask = (times >= a) & (times < b)
        count = word.get('syllable_count') or len(word.get('syllables', []))
        extra[4, mask] = np.maximum(extra[4, mask], conf)
        extra[5, mask] = max(0, min(float(count) / 6, 1))
        extra[6, mask] = max(0, min(float(count) / max(b-a, .05) / 12, 1))
        extra[7] = np.maximum(extra[7], conf*np.exp(-.5*((times-a)/.10)**2))
        extra[8] = np.maximum(extra[8], conf*np.exp(-.5*((times-b)/.10)**2))
        extra[9, mask] = np.maximum(extra[9, mask], np.clip((times[mask]-a)/max(b-a, .02), 0, 1))
        extra[10, mask] = 1
        extra[13, mask] = max(0, min(len(word.get('phones', []))/20, 1))
    for syllable in aligned.get('syllables', []):
        if syllable.get('asr_repetition_review') or syllable.get('asr_degenerate_repetition'):
            continue
        a = syllable.get('attack_estimate_s')
        n = syllable.get('nucleus_s')
        if not isinstance(a, (int, float)) or not isinstance(n, (int, float)):
            continue
        conf = float(np.clip(syllable.get('confidence', 0), 0, 1))
        width = max(.04, float(syllable.get('timing_uncertainty_s', .12)))
        onset = conf*np.exp(-.5*((times-a)/width)**2)
        nucleus = conf*np.exp(-.5*((times-n)/width)**2)
        extra[2] = np.maximum(extra[2], onset)
        extra[3] = np.maximum(extra[3], nucleus)
        if syllable.get('supported_input', False):
            extra[0] = np.maximum(extra[0], onset)
            extra[1] = np.maximum(extra[1], nucleus)
            index = syllable.get('word_index')
            if isinstance(index, int) and 0 <= index < len(words):
                word = words[index]
                mask = (times >= word['asr_start_s']) & (times < word['asr_end_s'])
                extra[11, mask] = 1
                extra[12, mask] = np.maximum(extra[12, mask], conf)
    ppg_path = folder / 'phonetics.npz'
    if ppg_path.exists():
        with np.load(ppg_path) as z:
            p, t = z['probabilities'].astype(np.float32), z['times_s']
            extra[14] = np.interp(times, t, 1-p[:, 0], left=0, right=0)
            entropy = -(p*np.log(np.maximum(p, 1e-8))).sum(1)/np.log(p.shape[1])
            extra[15] = np.interp(times, t, entropy, left=0, right=0)
    return np.nan_to_num(extra)


def load_inputs(song, feature_dir, probability_path, lyric_dir, art, norm):
    with np.load(feature_dir/'audio_features.npz') as z:
        mix = {k:z[k] for k in z.files}
    with np.load(feature_dir/'vocal_features.npz') as z:
        vocal = {k:z[k] for k in z.files}
    with np.load(probability_path) as z:
        pp, bb = z['pitch'].copy(), z['boundaries'].copy()
        if 'times_s' in z and not np.allclose(z['times_s'], vocal['times_s'], atol=1e-6):
            raise ValueError(f'Probability time grid mismatch: {song}')
    if pp.shape[1] != len(vocal['times_s']) or not np.array_equal(vocal['times_s'], mix['times_s']):
        raise ValueError(f'Acoustic time grids disagree: {song}')
    item = dict(id=song, mix=mix, vocal=vocal, pp=pp, bb=bb)
    item['bb'] = previous_boundaries(item, art, norm)
    item['ex'] = np.concatenate([features(item), lyric_features(lyric_dir, vocal['times_s'])])
    if item['ex'].shape[0] != INPUTS:
        raise ValueError(f'Expected {INPUTS} features, got {item["ex"].shape}')
    item['input_hashes'] = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):digest(p)
        for p in (feature_dir/'audio_features.npz', feature_dir/'vocal_features.npz', probability_path,
                  lyric_dir/'syllable_alignment.json')}
    return item


def export_predictions(model, items, mean, scale, register, oldnorm, folder, model_path):
    result=[]
    for item in items:
        out=folder/item['id'];out.mkdir(parents=True,exist_ok=False)
        p,b,aux=predict(model,item,mean,scale)
        notes,audit=decode_events(item,p,b,register,oldnorm,aux[:3])
        for note in notes:
            a=max(0,round(note['start_s']/HOP));end=max(a+1,round(note['end_s']/HOP))
            note['lyric_relation_scores']={name:float(aux[j+3,max(0,a-2):min(aux.shape[1],a+3)].max())
                                          for j,name in enumerate(RELATIONS[:3])}
            note['lyric_relation_scores'][RELATIONS[3]]=float(aux[6,a:min(end,aux.shape[1])].mean())
            note['lyric_relation_status']='unverified model scores; no threshold forces a syllable or note count'
        save(out/'predictions.json',dict(track=item['id'],notes=notes,model_sha256=digest(model_path),
            unquantized=True,score_read_at_inference=False,lyrics_read_at_inference=True,
            source_pitch_preserved=True,instrument_transposition_semitones=0,tempo_used=False,
            human_listening_verified=False,model_role='experimental residual acoustic/syllable-note relation model'))
        save(out/'decoder_audit.json',audit)
        np.savez_compressed(out/'probabilities.npz',pitch=p,boundaries=b,event_heads=aux[:3],
            relation_heads=aux[3:],times_s=item['vocal']['times_s'])
        metrics=None
        result.append(dict(track=item['id'],notes=len(notes),metrics=metrics))
    return result


def main_infer(args):
    torch.set_num_threads(2)
    art,register,oldnorm=load_previous()
    item=load_inputs(args.track,args.features,args.probabilities,args.lyrics,art,oldnorm)
    folder=args.model/'all_song_refit'
    norm=np.load(folder/'normalization.npz')
    model=SyllableNoteNetwork()
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True))
    args.output.mkdir(parents=True,exist_ok=False)
    export_predictions(model,[item],norm['mean'],norm['scale'],register,oldnorm,args.output,folder/'checkpoint.pt')
    save(args.output/'inference_inputs.json',dict(score_read=False,tempo_used=False,input_hashes=item['input_hashes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["infer"])
    parser.add_argument("--track", required=True)
    for name in ("model","features","probabilities","lyrics","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    main_infer(parser.parse_args())
