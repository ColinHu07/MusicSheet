"""Local ASR and phone evidence using separately installed pretrained models."""
from pathlib import Path
import dataclasses
import json
import numpy as np
from .audio import decode, sha256, save_json
from .phonetics import align_lyrics


def extract_lyrics(source, model_directory, output, *, language=None, device='cpu'):
    from faster_whisper import WhisperModel
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior ASR; select a fresh output directory')
    model = WhisperModel(str(model_directory), device=device,
                         compute_type='int8_float16' if device == 'cuda' else 'int8',
                         local_files_only=True, cpu_threads=2)
    audio = decode(source)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for start in range(0, int(np.ceil(len(audio) / 16000)), 24):
        clip = audio[start * 16000:(start + 24) * 16000]
        if len(clip) < 1600:
            continue
        segments, _ = model.transcribe(clip, language=language, beam_size=5, word_timestamps=True,
                                      condition_on_previous_text=False, temperature=0., vad_filter=False)
        for segment in segments:
            row = dataclasses.asdict(segment)
            row['start'] += start
            row['end'] += start
            row['window_start_s'] = start
            for word in row.get('words') or []:
                word['start'] += start
                word['end'] += start
            rows.append(row)
        save_json(output / 'asr_progress.json', dict(segments=rows, status='in_progress_unverified'))
        print(json.dumps(dict(processed_s=min(start + 24, len(audio) / 16000))), flush=True)
    save_json(output / 'lyrics_asr.json', dict(segments=rows, status='unverified_audio_derived_lyrics',
              language=language, duration_s=len(audio) / 16000, source=str(source), source_sha256=sha256(source),
              score_read=False, notes_read=False, audio_uploaded=False,
              limits=['Window edges can truncate or duplicate words', 'ASR confidence is not correctness',
                      'Missing words do not imply silence; humming and overlapping singers may be omitted']))


def extract_phonetics(directory, model_directory, *, device='cpu'):
    import torch
    from transformers import Wav2Vec2ForCTC
    directory, model_directory = Path(directory), Path(model_directory)
    if (directory / 'phonetics.npz').exists() or (directory / 'syllable_alignment.json').exists():
        raise FileExistsError('Preserve prior phonetic evidence; select a fresh lyric run')
    asr_path = directory / 'lyrics_asr.json'
    asr = json.loads(asr_path.read_text(encoding='utf-8'))
    source = Path(asr['source'])
    if sha256(source) != asr['source_sha256']:
        raise ValueError('Source changed after lyric extraction')
    vocab = json.loads((model_directory / 'vocab.json').read_text(encoding='utf-8'))
    config = json.loads((model_directory / 'preprocessor_config.json').read_text(encoding='utf-8'))
    model = Wav2Vec2ForCTC.from_pretrained(model_directory, local_files_only=True).eval().to(device)
    if int(np.prod(model.config.conv_stride)) != 320 or len(vocab) != 392:
        raise ValueError('This feature schema requires the 392-label XLSR phonetic checkpoint with 320-sample stride')
    audio = decode(source)
    points, probabilities = [], []
    for start in range(0, int(np.ceil(len(audio) / 16000)), 12):
        left, right = max(0, start - 1), min(len(audio) / 16000, start + 13)
        x = audio[int(left * 16000):int(right * 16000)]
        if len(x) < 400:
            continue
        if config.get('do_normalize', True):
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        with torch.no_grad():
            logits = model(torch.from_numpy(x[None]).to(device)).logits[0]
        scores = torch.softmax(logits, dim=-1).cpu().numpy()
        times = left + (np.arange(len(scores)) * 320 + 200) / 16000
        mask = (times >= start) & (times < start + 12)
        points.append(times[mask])
        probabilities.append(scores[mask])
    times, scores = np.concatenate(points), np.concatenate(probabilities)
    path = directory / 'phonetics.npz'
    np.savez_compressed(path, times_s=times, probabilities=scores.astype(np.float16))
    alignment = align_lyrics(asr, scores, times, vocab)
    alignment.update(asr_sha256=sha256(asr_path), phonetics_sha256=sha256(path),
                     phonetic_model='facebook/wav2vec2-xlsr-53-espeak-cv-ft')
    save_json(directory / 'syllable_alignment.json', alignment)
