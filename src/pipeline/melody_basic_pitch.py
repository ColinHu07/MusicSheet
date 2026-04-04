"""
Stage B — Melody extraction via Spotify Basic Pitch.

Provides two entry points:
  - ``extract_raw_notes``  — returns raw polyphonic note events (for the
    melody-path resolver)
  - ``extract_melody``     — legacy monophonic pipeline (confidence filter →
    overlap removal → range clamp → octave smooth)

The recommended flow is:
  audio → extract_raw_notes → candidate normalization → melody-path resolver
"""

import logging
from typing import List, Tuple

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.melody")


def extract_raw_notes(
    wav_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_duration_ms: float = 50.0,
) -> list:
    """
    Run Basic Pitch and return the raw note event tuples.

    Returns
    -------
    list of (start_s, end_s, pitch_midi, amplitude) tuples,
    suitable for passing to ``normalize_candidates()``.
    """
    logger.info("Running Basic Pitch inference on %s …", wav_path)

    from basic_pitch.inference import predict

    model_output, midi_data, note_events = predict(
        wav_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_duration_ms,
    )

    logger.info("Basic Pitch returned %d raw note events", len(note_events))
    return list(note_events)


def extract_melody(
    wav_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_duration_ms: float = 100.0,
    min_confidence: float = 0.3,
    clamp_to_violin_range: bool = True,
    violin_low: int = 55,    # G3
    violin_high: int = 105,  # A7
) -> List[NoteEvent]:
    """
    Legacy monophonic extraction pipeline.

    Kept for backward compatibility and as a comparison baseline.
    """
    raw = extract_raw_notes(wav_path, onset_threshold, frame_threshold, min_note_duration_ms)

    notes: List[NoteEvent] = []
    for ev in raw:
        notes.append(
            NoteEvent(
                start_s=float(ev[0]),
                end_s=float(ev[1]),
                midi_pitch=int(ev[2]),
                confidence=float(ev[3]),
            )
        )

    notes.sort(key=lambda n: (n.start_s, -n.confidence))

    # 1. Confidence filter
    before = len(notes)
    notes = [n for n in notes if n.confidence >= min_confidence]
    logger.info("Confidence filter: %d → %d", before, len(notes))

    # 2. Duration filter
    min_dur = min_note_duration_ms / 1000.0
    before = len(notes)
    notes = [n for n in notes if n.duration_s >= min_dur]
    logger.info("Duration filter: %d → %d", before, len(notes))

    # 3. Monophonic enforcement
    notes = _enforce_monophonic(notes)
    logger.info("After monophonic enforcement: %d notes", len(notes))

    # 4. Violin-range clamping
    if clamp_to_violin_range:
        notes = _clamp_violin_range(notes, violin_low, violin_high)

    # 5. Octave-error smoothing
    notes = _octave_smoothing(notes)

    logger.info("Final melody: %d notes", len(notes))
    return notes


# --------------------------------------------------------------------------- #
# Post-processing helpers (legacy)
# --------------------------------------------------------------------------- #

def _enforce_monophonic(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Keep only one note sounding at a time (highest-confidence wins)."""
    if not notes:
        return notes

    result = [notes[0]]
    for note in notes[1:]:
        prev = result[-1]
        if note.start_s < prev.end_s:
            if note.confidence > prev.confidence:
                prev.end_s = note.start_s
                if prev.duration_s < 0.01:
                    result.pop()
                result.append(note)
            else:
                if note.end_s > prev.end_s:
                    note.start_s = prev.end_s
                    if note.duration_s >= 0.05:
                        result.append(note)
        else:
            result.append(note)

    return result


def _clamp_violin_range(
    notes: List[NoteEvent], low: int, high: int
) -> List[NoteEvent]:
    """Shift notes outside the violin range to the nearest valid octave."""
    for note in notes:
        orig = note.midi_pitch
        while note.midi_pitch < low:
            note.midi_pitch += 12
        while note.midi_pitch > high:
            note.midi_pitch -= 12
        if note.midi_pitch != orig:
            logger.debug("Range clamp: MIDI %d → %d", orig, note.midi_pitch)
    return notes


def _octave_smoothing(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    If a note jumps ±12 semitones from *both* neighbours, pull it back
    to the surrounding register (likely an octave-detection error).
    """
    if len(notes) < 3:
        return notes

    pitches = [n.midi_pitch for n in notes]
    for i in range(1, len(pitches) - 1):
        dp = pitches[i] - pitches[i - 1]
        dn = pitches[i] - pitches[i + 1]
        if abs(dp) in (12, 24) and abs(dn) in (12, 24):
            if (dp > 0 and dn > 0) or (dp < 0 and dn < 0):
                direction = -1 if dp > 0 else 1
                corrected = pitches[i] + 12 * direction
                logger.debug("Octave smooth: note %d  %d → %d",
                             i, pitches[i], corrected)
                pitches[i] = corrected
                notes[i].midi_pitch = corrected

    return notes
