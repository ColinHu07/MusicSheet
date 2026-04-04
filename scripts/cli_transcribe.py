#!/usr/bin/env python
"""
Headless CLI for the transcription pipeline.

Usage examples::

    python scripts/cli_transcribe.py song.mp3
    python scripts/cli_transcribe.py song.mp3 --source-mode vocals
    python scripts/cli_transcribe.py song.mp3 --source-mode vocals --time-sig 2/2
    python scripts/cli_transcribe.py song.wav --key G --time-sig 3/4 --simplify
    python scripts/cli_transcribe.py song.mp3 --max-duration 120 --no-pdf
    python scripts/cli_transcribe.py song.mp3 --instrument flute
    python scripts/cli_transcribe.py song.mp3 --legacy   # use old pipeline
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_pipeline
from src.pipeline.utils import TranscriptionConfig, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="MP3 to Sheet Music (CLI)",
    )
    parser.add_argument("input", help="Path to MP3 / WAV / FLAC file")
    parser.add_argument("--output-dir", default="out", help="Output directory")
    parser.add_argument("--max-duration", type=float, default=45, help="Max seconds")

    parser.add_argument("--source-mode", default="mix",
                        choices=["mix", "vocals"],
                        help="Source mode: 'mix' (full audio) or 'vocals' "
                             "(separate vocals first)")
    parser.add_argument("--stems-dir", default="",
                        help="Path to directory with pre-separated stems "
                             "(e.g. from a prior demucs run)")

    parser.add_argument("--time-sig", default="auto",
                        help="Time signature: auto, 4/4, 3/4, 6/8, 2/2, 2/4")
    parser.add_argument("--key", default="auto",
                        help="Key: auto, C, G, 'D minor', etc.")
    parser.add_argument("--instrument", default="violin",
                        choices=["violin", "flute", "viola", "cello"],
                        help="Target instrument (default: violin)")
    parser.add_argument("--simplify", action="store_true",
                        help="Snap to 1/8 notes")
    parser.add_argument("--no-readable", action="store_true",
                        help="Disable rhythm simplification (raw quantization)")
    parser.add_argument("--triplets", action="store_true", help="Allow triplets")
    parser.add_argument("--slurs", action="store_true", help="Add phrase slurs")
    parser.add_argument("--min-note-ms", type=float, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--frame-threshold", type=float, default=0.3)
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy monophonic pipeline (skip melody-path resolver)")
    parser.add_argument("--pitch-floor", type=int, default=None,
                        help="Remove candidate notes below this MIDI pitch before DP "
                             "(e.g. 55 for vocal pop)")
    parser.add_argument("--reference", default="",
                        help="Path to reference_notes.json for pitch correction")
    parser.add_argument("--no-vocal-abstraction", action="store_true",
                        help="Disable vocal vibrato/phrase abstraction in vocals mode")
    parser.add_argument("--no-debug", action="store_true",
                        help="Skip debug artifacts")
    parser.add_argument("--musescore", default="", help="Path to MuseScore")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF rendering")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = TranscriptionConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        max_duration_s=args.max_duration,
        source_mode=args.source_mode,
        stems_dir=args.stems_dir,
        time_signature=args.time_sig,
        key_signature=args.key,
        instrument=args.instrument,
        simplify_rhythm=args.simplify,
        readable_rhythm=not args.no_readable,
        allow_triplets=args.triplets,
        add_phrase_slurs=args.slurs,
        min_note_duration_ms=args.min_note_ms,
        min_confidence=args.min_confidence,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        use_melody_resolver=not args.legacy,
        pitch_floor_midi=args.pitch_floor,
        reference_path=args.reference,
        vocal_abstraction=not args.no_vocal_abstraction,
        save_debug_artifacts=not args.no_debug,
        musescore_path=args.musescore if not args.no_pdf else "__SKIP__",
    )

    if args.no_pdf:
        config.musescore_path = "__SKIP_PDF__"

    result = run_pipeline(config, progress_cb=lambda msg: print(msg))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Source mode : {result.source_mode}")
    print(f"  Instrument : {args.instrument}")
    print(f"  Tempo      : {result.tempo_bpm} BPM")
    print(f"  Key        : {result.detected_key}")
    print(f"  Notes      : {result.num_notes}")
    print(f"  Measures   : {result.num_measures}")
    print(f"  MusicXML   : {result.musicxml_path or '(none)'}")
    print(f"  PDF        : {result.pdf_path or '(none)'}")
    print(f"  Debug JSON : {result.debug_json_path or '(none)'}")
    if result.melody_stats:
        print(f"  Melody stats:")
        for k, v in result.melody_stats.items():
            print(f"    {k}: {v}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    WARNING: {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
