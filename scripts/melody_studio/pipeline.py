"""Run the frozen three-source transcription stack using its existing local runtimes."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BACKEND = Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", ROOT)).resolve()
SCRIPTS = Path(__file__).resolve().parent / "workers"
GPU = Path(os.environ.get("MUSICSHEET_ACOUSTIC_PYTHON", BACKEND / "vendor/rvc-2.3.260718/RVC20260718Nvidia50x0/runtime/python.exe"))
RESEARCH = Path(os.environ.get("MUSICSHEET_RESEARCH_PYTHON", BACKEND / ".venv-transcription-research-20261117/Scripts/python.exe"))
LYRICS = Path(os.environ.get("MUSICSHEET_LYRICS_PYTHON", BACKEND / ".venv-lyrics/Scripts/python.exe"))

def check_backend():
    for path in (SCRIPTS / "prepare.py", GPU, RESEARCH, LYRICS):
        if not path.is_file():
            raise ValueError(f"MP3 import needs the configured transcription backend. Set MUSICSHEET_BACKEND_ROOT (missing {path.name}).")



def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def run_job(job, status):
    check_backend()
    work = ROOT / 'data/melody_studio' / job.name
    work.mkdir(parents=True, exist_ok=True)
    acoustic = work / 'acoustic'
    track = work / 'track'
    lang = read(job / 'info.json')['language']
    features = track / 'features'
    syllables = work / 'lyrics/track'
    raw = job / 'prediction.json'
    # One subprocess at a time releases GPU memory between runtimes.
    def run(stage, python, script, *args):
        status(stage)
        with (job / 'pipeline.log').open('a', encoding='utf-8') as log:
            log.write('\n' + stage + '\n')
            log.flush()
            subprocess.run([str(python), str(script), *map(str, args)], cwd=BACKEND, check=True,
                           stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1'))
    run('Separating vocals and measuring pitch', GPU, SCRIPTS/'prepare.py',
        '--audio', job/'original.mp3', '--output-dir', work, '--id', 'track')
    duration = read(track/'audio_verification.json')['duration_s']
    # Keep a usable final analysis window even when duration is nearly a multiple of 22.
    count = max(1, round(duration/22))
    cuts = [i*duration/count for i in range(count+1)]
    clips = [dict(id=f'clip{i:04}', title='Melody', crop_start=a, crop_end=b,
                  audio_start=max(0,a-1.5), audio_end=min(duration,b+1.5),
                  mix=str(track/'audio/source.mp3'), vocals=str(track/'audio/vocals.wav'))
             for i,(a,b) in enumerate(zip(cuts,cuts[1:]))]
    write(work/'audio_manifest.json', clips)
    run('Finding note starts with ROSVOT', RESEARCH, SCRIPTS/'rosvot.py',
        '--root', work, '--variants', 'vocals')
    run('Measuring vocal notes with GAME', RESEARCH, __file__, 'game', work, lang)
    run('Estimating melody and boundaries', GPU, SCRIPTS/'acoustic_stage.py',
        '--model-dir', ROOT/'checkpoints/melody_pairs_v03_20261029',
        '--features-dir', features, '--output-dir', acoustic, '--device', 'cuda', '--no-export')
    run('Recognizing sung words', LYRICS, __file__, 'asr', work, lang)
    run('Aligning syllables to the recording', GPU, __file__, 'phones', work, lang)
    run('Resolving attacks and continuations', GPU, SCRIPTS/'syllable.py',
        'infer', '--model', ROOT/'checkpoints/syllable_note_20261112_v3', '--track', 'track',
        '--features', features, '--probabilities', acoustic/'probabilities.npz',
        '--lyrics', syllables, '--output', work/'syllable')
    for kind in ['rosvot_vocals','game_medium_vocals']:
        events = [n for clip in clips for n in read(work/kind/clip['id']/'predictions.json')['notes']]
        write(work/(kind+'.json'), dict(notes=events))
    if any(read(p)['notes'] for p in [work/'syllable/track/predictions.json',work/'rosvot_vocals.json',work/'game_medium_vocals.json']):
        run('Combining note predictions', RESEARCH, ROOT/'musicsheet/trained_fusion.py',
            '--baseline', work/'syllable/track/predictions.json', '--rosvot', work/'rosvot_vocals.json',
            '--game', work/'game_medium_vocals.json', '--vocal-features', features/'vocal_features.npz',
            '--probabilities', work/'syllable/track/probabilities.npz', '--lyrics', syllables/'syllable_alignment.json',
            '--out', raw)
    else:
        write(raw, {'notes':[], 'reference_used_at_inference':False})
    status('Engraving the editable score')
    from notation import arrange
    return arrange(read(raw)['notes'], features, duration)


def stage(kind, work, lang):
    sys.path.insert(0, str(SCRIPTS))
    if kind == 'game':
        import game
        game.RUN = work
        sys.argv = [sys.argv[0], '--language-id', '1' if lang=='en' else '0']
        game.main()
    else:
        import words
        words.OUT = work/'lyrics'
        words.CACHED = work/'empty_cache'
        words.song_source = lambda _: work/'track/audio/vocals.wav'
        if lang == 'en':
            words.ENGLISH.add('track')
        (words.asr_stage if kind == 'asr' else words.phone_stage)(['track'])


if __name__ == '__main__':
    stage(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
