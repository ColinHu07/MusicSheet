"""Local full-corpus ASR and phonetic candidates for syllable-to-note review.

No score-derived text, note labels, or tempo are inputs. Old outputs remain read-only.
All ASR and phone times remain unverified; failures and gaps are retained.
"""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", Path(__file__).resolve().parents[3])).resolve()
ENGLISH = set()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def audio16(path):
    import numpy as np
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le',
                              '-ar', '16000', '-ac', '1', 'pipe:1'], capture_output=True, check=True)
    return np.frombuffer(decoded.stdout, dtype='<f4').copy()


def asr_stage(tracks):
    import numpy as np
    from whisper_runtime import WhisperModel
    model = None
    for song in tracks:
        folder = OUT / song
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / 'lyrics_asr.json'
        if final.exists():
            print(song, 'ASR already complete', flush=True)
            continue
        cached = CACHED / song / 'lyrics_asr.json'
        if cached.exists():
            shutil.copy2(cached, final)
            save(folder/'asr_lineage.json', dict(reused_from=str(cached), sha256=digest(cached),
                 source_unchanged=True, new_inference=False))
            print(song, 'reused frozen ASR', flush=True)
            continue
        if model is None:
            model = WhisperModel('large-v3', device='cuda', compute_type='int8_float16',
                download_root=str(ROOT/'models/transcription/lyrics_asr_cache'),
                local_files_only=True, cpu_threads=2)
        source = song_source(song)
        audio = audio16(source)
        language = 'en' if song in ENGLISH else 'ko'
        rows = []
        for start in range(0, int(np.ceil(len(audio) / 16000)), 24):
            clip = audio[start*16000:(start+24)*16000]
            if len(clip) < 1600:
                continue
            segments, _ = model.transcribe(clip, language=language, beam_size=5,
                word_timestamps=True, condition_on_previous_text=False, temperature=0., vad_filter=False)
            for seg in segments:
                row = dataclasses.asdict(seg)
                row['start'] += start
                row['end'] += start
                row['window_start_s'] = start
                for word in row.get('words', []):
                    word['start'] += start
                    word['end'] += start
                rows.append(row)
            save(folder/'asr_progress.json', dict(segments=rows, status='in_progress_unverified'))
            print(json.dumps(dict(track=song, processed_s=min(start+24, len(audio)/16000),
                                  segments=len(rows))), flush=True)
        save(final, dict(segments=rows, status='unverified_audio_derived_lyrics', language=language,
             duration_s=len(audio)/16000, source=str(source), source_sha256=digest(source),
             score_read=False, notes_read=False, audio_uploaded=False,
             model='Systran/faster-whisper-large-v3',
             limits=['Short-window edges can duplicate or truncate words',
                     'ASR confidence is not correctness', 'Absence of text is not absence of singing',
                     'Sung repetitions, humming, rap and code switching can be omitted or mistranscribed']))
        save(folder/'asr_lineage.json', dict(new_inference=True, script_sha256=digest(Path(__file__)),
             source_sha256=digest(source), cached_models_only=True))


def phone_stage(tracks):
    # Keep phonemizer's temporary DLL copies in this writable experiment
    # directory instead of the app's restricted per-process temp directory.
    import tempfile
    local_temp = OUT/'_runtime_temp'
    local_temp.mkdir(exist_ok=True)
    os.environ['TEMP'] = str(local_temp)
    os.environ['TMP'] = str(local_temp)
    tempfile.tempdir = str(local_temp)
    import numpy as np
    import torch
    from transformers import Wav2Vec2ForCTC
    import phonetics as phonetic_alignment
    # eSpeak's unstressed ROSES/I# vowel is ᵻ. The older vowel whitelist
    # omitted it, losing the second nucleus of e.g. "fallin'". This local
    # process-only extension leaves the frozen prior pipeline untouched.
    phonetic_alignment.VOWELS.add('\u1d7b')
    align_lyrics = phonetic_alignment.align_lyrics
    nucleus_rule = 'espeak_reduced_vowel_1d7b_v1'
    torch.set_num_threads(2)
    modeldir = ROOT/'models/transcription/phonetic_xlsr_20261111'
    vocab = json.loads((modeldir/'vocab.json').read_text(encoding='utf-8'))
    config = json.loads((modeldir/'preprocessor_config.json').read_text())
    model = None
    for song in tracks:
        folder = OUT/song
        final = folder/'syllable_alignment.json'
        if final.exists():
            prior = json.loads(final.read_text(encoding='utf-8'))
            if prior.get('nucleus_rule') == nucleus_rule:
                print(song, 'phone alignment already complete', flush=True)
                continue
            # Preserve the first draft before applying the newly discovered
            # vowel-inventory correction inside this versioned output folder.
            archive = folder/'prior_to_reduced_vowel_fix'
            archive.mkdir(exist_ok=True)
            for name in ['syllable_alignment.json', 'syllables_enriched.json', 'chunks.json', 'phone_lineage.json']:
                if (folder/name).exists() and not (archive/name).exists():
                    shutil.copy2(folder/name, archive/name)
        cached = CACHED/song/'syllable_alignment.json'
        if cached.exists() and not (folder/'phonetics.npz').exists():
            shutil.copy2(CACHED/song/'phonetics.npz', folder/'phonetics.npz')
        asr_path = folder/'lyrics_asr.json'
        if not asr_path.exists():
            print(song, 'waiting for ASR', flush=True)
            continue
        asr = json.loads(asr_path.read_text(encoding='utf-8'))
        source = Path(asr['source'])
        assert digest(source) == asr['source_sha256']
        ppg_path = folder/'phonetics.npz'
        reused_ppg = ppg_path.exists()
        if ppg_path.exists():
            cached_ppg = np.load(ppg_path)
            ts, scores = cached_ppg['times_s'], cached_ppg['probabilities'].astype(np.float32)
        else:
            if model is None:
                model = Wav2Vec2ForCTC.from_pretrained(modeldir, local_files_only=True).eval().to('cuda')
            audio = audio16(source)
            points, probabilities = [], []
            for start in range(0, int(np.ceil(len(audio)/16000)), 12):
                left = max(0, start-1)
                right = min(len(audio)/16000, start+13)
                x = audio[int(left*16000):int(right*16000)]
                if len(x) < 400:
                    continue
                if config.get('do_normalize', True):
                    x = (x-x.mean()) / np.sqrt(x.var()+1e-7)
                with torch.no_grad():
                    logits = model(torch.from_numpy(x[None]).cuda()).logits[0]
                scores = torch.softmax(logits, dim=-1).cpu().numpy()
                ts = left+(np.arange(len(scores))*320+200)/16000
                keep = (ts >= start) & (ts < start+12)
                points.append(ts[keep])
                probabilities.append(scores[keep])
            ts, scores = np.concatenate(points), np.concatenate(probabilities)
            np.savez_compressed(ppg_path, times_s=ts, probabilities=scores.astype(np.float16))
        result = align_lyrics(asr, scores, ts, vocab)
        result.update(track=song, phonetic_model='facebook/wav2vec2-xlsr-53-espeak-cv-ft',
                      asr_sha256=digest(asr_path), audio_uploaded=False, human_verified=False,
                      nucleus_rule=nucleus_rule,
                      nucleus_rule_reference='https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages/gmw/en.md')
        save(final, result)
        save(folder/'phone_lineage.json', dict(new_acoustic_inference=not reused_ppg,
             new_ctc_alignment=True, source_sha256=digest(source), nucleus_rule=nucleus_rule,
             reused_ppg=str(CACHED/song/'phonetics.npz') if cached.exists() else (str(ppg_path) if reused_ppg else None),
             asr_sha256=digest(asr_path), script_sha256=digest(Path(__file__)), cached_models_only=True))
        if cached.exists():
            old = json.loads(cached.read_text(encoding='utf-8'))
            changes = []
            for word_index, (before, after) in enumerate(zip(old['words'], result['words'])):
                assert before['text'] == after['text']
                if len(before['syllables']) != len(after['syllables']):
                    changes.append(dict(word_index=word_index, word=after['text'],
                         old_candidates=len(before['syllables']), new_candidates=len(after['syllables']),
                         asr_start_s=after['asr_start_s'], asr_end_s=after['asr_end_s']))
            save(folder/'reduced_vowel_fix_audit.json', dict(old_syllables=len(old['syllables']),
                 new_syllables=len(result['syllables']), changes=changes,
                 fix='Recognize eSpeak unstressed vowel U+1D7B as a nucleus; not a manual pitch label.',
                 old_outputs_unchanged=True, boundaries_still_unverified=True))
        print(json.dumps(dict(track=song, words=len(result['words']), syllables=len(result['syllables']),
                              supported=sum(s['supported_input'] for s in result['syllables']))), flush=True)


