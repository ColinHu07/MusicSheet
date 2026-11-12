"""Command-line access to the independent transcription engine."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local note-first music transcription (private model assets required)')
    commands = parser.add_subparsers(dest='command', required=True)
    infer = commands.add_parser('infer', help='Infer unquantized notes from source-derived features')
    infer.add_argument('--assets', type=Path, required=True)
    infer.add_argument('--features', type=Path, required=True)
    infer.add_argument('--probabilities', type=Path, help='Optional frozen base-model probabilities on the identical grid')
    infer.add_argument('--mode', choices=['base', 'joint', 'note-events', 'lyric'], default='joint')
    infer.add_argument('--lyrics', type=Path)
    infer.add_argument('--vocal-audio', type=Path)
    infer.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare', help='Extract features from aligned stems plus external RMVPE evidence')
    for name in ('audio', 'vocals', 'pitch-track', 'output'):
        prepare.add_argument('--' + name, type=Path, required=True)
    lyrics = commands.add_parser('lyrics', help='Extract local lyric hypotheses, optionally align phonetic evidence')
    for name in ('vocals', 'asr-model', 'output'):
        lyrics.add_argument('--' + name, type=Path, required=True)
    lyrics.add_argument('--phone-model', type=Path)
    lyrics.add_argument('--language', choices=['en', 'ko'])
    lyrics.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    notation = commands.add_parser('score', help='Engrave a separate provisional score without changing source events')
    notation.add_argument('--predictions', type=Path, required=True)
    notation.add_argument('--output', type=Path, required=True)
    notation.add_argument('--bpm', type=float, required=True)
    notation.add_argument('--phase', type=float, default=0.)
    notation.add_argument('--key-sharps', type=int, default=0)
    notation.add_argument('--title', default='Melody draft')
    notation.add_argument('--rhythm-model', type=Path)
    notation.add_argument('--pdf', action='store_true')
    playback = commands.add_parser('render', help='Render supplied MIDI through a local SoundFont')
    for name in ('midi', 'soundfont', 'output'):
        playback.add_argument('--' + name, type=Path, required=True)
    playback.add_argument('--background', type=Path)
    playback.add_argument('--background-db', type=float, default=-10.)
    playback.add_argument('--fluidsynth', default='fluidsynth')
    train = commands.add_parser('train', help='Fit base/event/lyric models from explicit local train/validation labels')
    train.add_argument('--manifest', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--assets', type=Path)
    train.add_argument('--mode', choices=['base', 'events', 'lyric', 'lyric-control'], default='events')
    train.add_argument('--epochs', type=int, default=12)
    train.add_argument('--seed', type=int, default=20261112)
    evaluate = commands.add_parser('evaluate', help='Evaluate notes in a caller-defined common scoring scope')
    for name in ('reference', 'predictions', 'output'):
        evaluate.add_argument('--' + name, type=Path, required=True)
    evaluate.add_argument('--annotations', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'infer':
        from .engine import Engine, save_result
        if args.output.exists():
            parser.error('Use a fresh inference output directory')
        result = Engine(args.assets).infer(args.features, mode=args.mode, cached_probabilities=args.probabilities,
                                          lyric_directory=args.lyrics, vocal_audio=args.vocal_audio)
        save_result(result, args.output)
        print(json.dumps(dict(events=len(result['notes']), output=str(args.output), mode=args.mode)))
    elif args.command == 'prepare':
        from .prepare import prepare
        prepare(args.audio, args.vocals, args.pitch_track, args.output)
    elif args.command == 'lyrics':
        from .lyrics import extract_lyrics, extract_phonetics
        extract_lyrics(args.vocals, args.asr_model, args.output, language=args.language, device=args.device)
        if args.phone_model:
            extract_phonetics(args.output, args.phone_model, device=args.device)
    elif args.command == 'score':
        from .notation import export_score
        export_score(args.predictions, args.output, bpm=args.bpm, phase=args.phase, key_sharps=args.key_sharps,
                     title=args.title, rhythm_model=args.rhythm_model, pdf=args.pdf)
    elif args.command == 'render':
        from .playback import render
        render(args.midi, args.soundfont, args.output, background=args.background,
               background_db=args.background_db, fluidsynth=args.fluidsynth)
    elif args.command == 'train':
        from .training import fit
        if args.mode != 'base' and args.assets is None:
            parser.error('Residual training requires --assets for the frozen base/joint models')
        fit(args.manifest, args.output, mode=args.mode, assets=args.assets, epochs=args.epochs, seed=args.seed)
    elif args.command == 'evaluate':
        from .evaluation import evaluate_notes
        from .audio import save_json
        if args.output.exists():
            parser.error('Use a new evaluation output path')
        def read_notes(path):
            data = json.loads(path.read_text(encoding='utf-8'))
            return data['notes'] if isinstance(data, dict) else data
        annotations = json.loads(args.annotations.read_text(encoding='utf-8')) if args.annotations else None
        result = evaluate_notes(read_notes(args.reference), read_notes(args.predictions), annotations=annotations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, result)
