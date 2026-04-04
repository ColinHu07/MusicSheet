"""
Pipeline orchestrator — wires all stages together.

Usage (programmatic)::

    from src.pipeline import run_pipeline
    from src.pipeline.utils import TranscriptionConfig

    cfg = TranscriptionConfig(input_path="song.mp3")
    result = run_pipeline(cfg)
"""

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from .utils import (
    TranscriptionConfig,
    TranscriptionResult,
    ensure_dir,
    note_events_to_dicts,
    quantized_notes_to_dicts,
    save_debug_json,
)

logger = logging.getLogger("mp3_to_sheet")


def run_pipeline(
    config: TranscriptionConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> TranscriptionResult:
    """
    Execute the full transcription pipeline.

    Parameters
    ----------
    config : TranscriptionConfig
        All user-selected options.
    progress_cb : callable, optional
        ``progress_cb(message)`` is called at each stage so the GUI can
        show live logs.

    Returns
    -------
    TranscriptionResult
    """

    def log(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    result = TranscriptionResult()
    song_stem = Path(config.input_path).stem

    mode_suffix = ""
    if config.source_mode != "mix":
        mode_suffix = f"__{config.source_mode}"

    ts_suffix = ""
    if config.time_signature not in ("auto", ""):
        ts_suffix = f"__{config.time_signature.replace('/', '-')}"

    file_stem = f"{song_stem}{mode_suffix}{ts_suffix}"

    # --- output dirs --------------------------------------------------------
    out_musicxml_dir = ensure_dir(os.path.join(config.output_dir, "musicxml"))
    out_pdf_dir      = ensure_dir(os.path.join(config.output_dir, "pdf"))
    out_debug_dir    = ensure_dir(os.path.join(config.output_dir, "debug"))

    # ======================================================================= #
    # Stage A -- Audio ingest
    # ======================================================================= #
    log("Stage A: Loading and pre-processing audio ...")
    from .ingest import ingest_audio

    wav_path, sr = ingest_audio(
        input_path=config.input_path,
        output_dir=config.output_dir,
        max_duration_s=config.max_duration_s,
    )
    log(f"  WAV ready ({wav_path})")

    # ======================================================================= #
    # Stage A2 -- Source separation (optional)
    # ======================================================================= #
    transcribe_wav = wav_path
    stem_metadata = {"source_mode": config.source_mode}

    if config.source_mode == "vocals":
        log("Stage A2: Source separation (vocal isolation) ...")
        from .source_separation import is_available, separate

        if is_available():
            try:
                separation_input = config.input_path
                log(f"  Feeding original audio to separator: {separation_input}")
                stems = separate(
                    separation_input,
                    output_dir=config.output_dir,
                    stems_dir_override=config.stems_dir or None,
                )
                transcribe_wav = stems.vocals
                stem_metadata.update({
                    "separator_model": stems.model_name,
                    "vocals_path": stems.vocals,
                    "accompaniment_path": stems.accompaniment,
                    "source_hash": stems.source_hash,
                    "separation_succeeded": True,
                    "fallback": False,
                })
                log(f"  Source separation SUCCEEDED")
                log(f"  Using separated stem for transcription: {stems.vocals}")
            except Exception as exc:
                msg = f"Source separation failed ({exc}), falling back to mix"
                result.warnings.append(msg)
                log(f"  WARNING: {msg}")
                log(f"  Falling back to mix because: {exc}")
                stem_metadata["fallback"] = True
                stem_metadata["separation_succeeded"] = False
                stem_metadata["error"] = str(exc)
        else:
            msg = "Source separator not installed, falling back to mix mode"
            result.warnings.append(msg)
            log(f"  WARNING: {msg}")
            stem_metadata["fallback"] = True
            stem_metadata["separation_succeeded"] = False
    else:
        log(f"  Source mode: {config.source_mode} (using original mix)")
        stem_metadata["separation_succeeded"] = None

    log(f"  Actual audio path for transcription: {transcribe_wav}")

    result.source_mode = config.source_mode

    # ======================================================================= #
    # Stage B -- Melody extraction
    # ======================================================================= #
    pyin_stats = None
    abstraction_stats = None
    pitch_stats = None
    segmentation_stats = None

    use_pyin = (config.source_mode == "vocals" and config.use_melody_resolver)

    if use_pyin:
        # ── PYIN path: direct monophonic pitch tracking ──────────────
        log("Stage B: Extracting melody (pyin monophonic tracker) ...")
        from .pyin_melody import extract_melody_pyin, PyinConfig

        pyin_cfg = PyinConfig()
        notes, pyin_stats = extract_melody_pyin(transcribe_wav, pyin_cfg)

        log(f"  pyin: {pyin_stats.voiced_frames}/{pyin_stats.total_frames} "
            f"voiced frames ({pyin_stats.voiced_pct:.0f}%)")
        log(f"  {pyin_stats.notes_raw} raw segments → "
            f"{pyin_stats.notes_after_cleanup} notes")
        log(f"  Pitch range: MIDI {pyin_stats.pitch_range_midi[0]}-"
            f"{pyin_stats.pitch_range_midi[1]}  "
            f"avg dur: {pyin_stats.avg_duration_s:.3f}s")

        if config.save_debug_artifacts:
            _save_artifact(out_debug_dir, file_stem, "pyin_notes",
                           note_events_to_dicts(notes))
            _save_artifact(out_debug_dir, file_stem, "pyin_stats",
                           pyin_stats.to_dict())

    elif config.use_melody_resolver:
        # ── Legacy Basic Pitch + DP path (mix mode) ──────────────────
        log("Stage B: Extracting candidate notes (Basic Pitch) ...")
        from .melody_basic_pitch import extract_raw_notes

        raw_events = extract_raw_notes(
            wav_path=transcribe_wav,
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
            min_note_duration_ms=config.min_note_duration_ms * 0.5,
        )
        log(f"  {len(raw_events)} raw note events from Basic Pitch")

        log("Stage B1/2: Normalizing candidates ...")
        from .candidate_notes import normalize_candidates

        candidates = normalize_candidates(
            raw_events,
            min_duration_ms=config.min_note_duration_ms * 0.5,
            min_confidence=config.min_confidence * 0.5,
        )
        log(f"  {len(candidates)} normalised candidates")

        if config.save_debug_artifacts:
            _save_artifact(out_debug_dir, file_stem, "raw_candidate_notes", [
                {"start_s": c.start_s, "end_s": c.end_s,
                 "pitch_midi": c.pitch_midi, "confidence": c.confidence}
                for c in candidates
            ])

        log("Stage B2: Running melody-path resolver (DP) ...")
        from ..config.melody_config import MelodyConfig
        from .melody_path import resolve_melody

        melody_cfg = MelodyConfig(
            instrument=config.instrument,
            min_note_duration_ms=config.min_note_duration_ms,
            min_confidence=config.min_confidence,
            save_debug_artifacts=config.save_debug_artifacts,
            pitch_floor_midi=config.pitch_floor_midi,
        )

        for k, v in config.melody_config_overrides.items():
            if hasattr(melody_cfg, k):
                setattr(melody_cfg, k, v)

        path_result = resolve_melody(candidates, melody_cfg)
        notes = path_result.selected_notes
        result.melody_stats = path_result.stats
        log(f"  Melody path: {len(notes)} notes selected "
            f"(from {path_result.stats.get('total_candidates', '?')} candidates)")

        if config.save_debug_artifacts:
            _save_artifact(out_debug_dir, file_stem, "melody_selected_notes",
                           note_events_to_dicts(notes))
            _save_artifact(out_debug_dir, file_stem, "melody_path_stats",
                           path_result.stats)
    else:
        log("Stage B: Extracting melody (Basic Pitch, legacy) ...")
        from .melody_basic_pitch import extract_melody

        notes = extract_melody(
            wav_path=transcribe_wav,
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
            min_note_duration_ms=config.min_note_duration_ms,
            min_confidence=config.min_confidence,
            clamp_to_violin_range=config.clamp_to_violin_range,
            violin_low=config.violin_low,
            violin_high=config.violin_high,
        )

    log(f"  {len(notes)} melody notes extracted")
    result.num_notes = len(notes)

    if not notes:
        result.warnings.append("No notes detected -- check audio quality / thresholds.")
        log("  WARNING: No notes detected!")
        return result

    # ======================================================================= #
    # Stage C -- Beat tracking
    # ======================================================================= #
    log("Stage C: Beat tracking (librosa) ...")
    from .beat_grid import detect_beats

    grid = detect_beats(
        wav_path=wav_path,
        time_signature=config.time_signature,
    )
    result.tempo_bpm = grid.tempo_bpm
    log(f"  Tempo: {grid.tempo_bpm} BPM   "
        f"Time sig: {grid.time_sig_num}/{grid.time_sig_den}")

    # ======================================================================= #
    # Stage D -- Key detection
    # ======================================================================= #
    log("Stage D: Detecting key ...")
    from .key_detect import detect_key, spell_pitch

    key_name, mode, sharps = detect_key(notes, config.key_signature)
    result.detected_key = f"{key_name} {mode}"
    log(f"  Key: {key_name} {mode}  (sharps={sharps})")

    # ======================================================================= #
    # Reference-guided pitch correction (optional)
    # ======================================================================= #
    ref_correction_stats = None
    if config.reference_path:
        log("Applying reference-guided pitch correction ...")
        from .reference_correction import correct_with_reference, save_alignment_report

        correction_result = correct_with_reference(
            gen_notes=notes,
            reference_path=config.reference_path,
        )
        notes = correction_result.corrected_notes
        ref_correction_stats = correction_result.stats
        log(f"  {ref_correction_stats.pitch_corrections} pitches corrected, "
            f"{ref_correction_stats.merges_applied} merges, "
            f"{ref_correction_stats.already_correct} already correct")

        if config.save_debug_artifacts:
            save_alignment_report(
                correction_result.alignment,
                correction_result.ref_notes,
                correction_result.gen_notes_original,
                ref_correction_stats,
                str(out_debug_dir),
            )

    # ======================================================================= #
    # Quantization
    # ======================================================================= #
    log("Quantizing notes to beat grid ...")
    from .quantize import quantize_notes, split_at_barlines

    quantized = quantize_notes(
        notes=notes,
        grid=grid,
        simplify=config.simplify_rhythm,
        allow_triplets=config.allow_triplets,
    )

    # Post-quantization merge: collapse adjacent same-pitch notes
    if config.use_melody_resolver:
        from .phrase_smoothing import merge_quantized_same_pitch
        pre_merge_count = sum(1 for q in quantized if not q.is_rest)
        quantized = merge_quantized_same_pitch(quantized)
        post_merge_count = sum(1 for q in quantized if not q.is_rest)
        if pre_merge_count != post_merge_count:
            log(f"  Post-quantize merge: {pre_merge_count} -> {post_merge_count} notes")

    # ======================================================================= #
    # Rhythm simplification (optional but on by default)
    # ======================================================================= #
    unusual_before = 0
    rests_before = 0
    if config.readable_rhythm:
        from .rhythm_simplify import simplify_notation, count_unusual_durations

        unusual_before = count_unusual_durations(quantized)
        rests_before = sum(1 for q in quantized if q.is_rest)
        notes_before_simpl = sum(1 for q in quantized if not q.is_rest)

        rest_threshold = 0.5 if use_pyin else 0.25
        quantized = simplify_notation(quantized, rest_threshold)

        unusual_after = count_unusual_durations(quantized)
        rests_after = sum(1 for q in quantized if q.is_rest)
        notes_after_simpl = sum(1 for q in quantized if not q.is_rest)

        log(f"  Rhythm simplification: unusual {unusual_before}->{unusual_after}, "
            f"rests {rests_before}->{rests_after}, "
            f"notes {notes_before_simpl}->{notes_after_simpl}")

    quantized = split_at_barlines(quantized, grid)
    log(f"  {len(quantized)} quantized events")

    # ======================================================================= #
    # Collect stats
    # ======================================================================= #
    from .rhythm_simplify import count_unusual_durations

    n_notes_final = sum(1 for q in quantized if not q.is_rest)
    n_rests_final = sum(1 for q in quantized if q.is_rest)
    unusual_final = count_unusual_durations(quantized)

    # ======================================================================= #
    # Save debug JSON
    # ======================================================================= #
    debug_path = os.path.join(out_debug_dir, f"{file_stem}.json")
    debug_data = {
        "config": asdict(config),
        "source_mode": config.source_mode,
        "actual_transcription_audio": transcribe_wav,
        "stem_metadata": stem_metadata,
        "tempo_bpm": grid.tempo_bpm,
        "time_signature": f"{grid.time_sig_num}/{grid.time_sig_den}",
        "detected_key": f"{key_name} {mode}",
        "raw_notes": note_events_to_dicts(notes),
        "quantized_notes": quantized_notes_to_dicts(quantized),
        "notation_stats": {
            "notes_count": n_notes_final,
            "rests_count": n_rests_final,
            "unusual_durations": unusual_final,
        },
    }
    if pyin_stats is not None:
        debug_data["pyin_stats"] = pyin_stats.to_dict()
    elif config.use_melody_resolver and not use_pyin:
        debug_data["melody_stats"] = path_result.stats
    if ref_correction_stats is not None:
        debug_data["reference_correction_stats"] = ref_correction_stats.to_dict()
    save_debug_json(debug_data, debug_path)
    result.debug_json_path = debug_path
    log(f"  Debug JSON: {debug_path}")

    # ======================================================================= #
    # Stage F -- MusicXML export
    # ======================================================================= #
    log("Stage F: Generating MusicXML ...")
    from .musicxml_export import export_musicxml

    musicxml_path = os.path.join(out_musicxml_dir, f"{file_stem}.musicxml")
    export_musicxml(
        notes=quantized,
        key_name=key_name,
        mode=mode,
        sharps=sharps,
        grid=grid,
        output_path=musicxml_path,
        add_slurs=config.add_phrase_slurs,
        instrument_name=config.instrument,
    )
    result.musicxml_path = musicxml_path
    log(f"  MusicXML: {musicxml_path}")

    try:
        result.num_measures = _count_measures(musicxml_path)
    except Exception:
        pass

    # ======================================================================= #
    # Stage G -- PDF rendering
    # ======================================================================= #
    log("Stage G: Rendering PDF ...")
    from .render_pdf import render_pdf

    pdf_path = os.path.join(out_pdf_dir, f"{file_stem}.pdf")
    rendered = render_pdf(
        musicxml_path=musicxml_path,
        pdf_path=pdf_path,
        musescore_bin=config.musescore_path,
    )

    if rendered:
        result.pdf_path = rendered
        log(f"  PDF: {rendered}")
    else:
        msg = "PDF not rendered -- install MuseScore for PDF export."
        result.warnings.append(msg)
        log(f"  WARNING: {msg}")

    log("Pipeline complete!")
    return result


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _count_measures(musicxml_path: str) -> int:
    """Quick count of <measure> tags."""
    count = 0
    with open(musicxml_path, "r", encoding="utf-8") as f:
        for line in f:
            if "<measure " in line:
                count += 1
    return count


def _save_artifact(debug_dir, song_stem: str, name: str, data):
    """Save a debug artifact as JSON."""
    path = os.path.join(debug_dir, f"{song_stem}_{name}.json")
    save_debug_json(data if isinstance(data, dict) else {"data": data}, path)
