"""
Monophonic vocal melody extraction using librosa's pyin pitch tracker.

Replaces the Basic Pitch → candidate pool → DP melody-path chain with a
direct pitch-curve-to-notes approach.  pyin is specifically designed for
monophonic (single voice) pitch tracking and produces a continuous f0
estimate that we segment into discrete note events.

Stages:
  1. **Pitch extraction** — librosa.pyin on the vocal stem.
  2. **Pitch curve cleanup** — median filter, unvoiced gap handling.
  3. **Note segmentation** — split curve into notes at pitch jumps and
     silence boundaries.
  4. **Fragment cleanup** — drop micro-notes, merge same-pitch adjacencies.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.pyin_melody")


@dataclass
class PyinConfig:
    """Tuneable parameters for the pyin melody extraction."""

    fmin_note: str = "C3"
    fmax_note: str = "C6"
    sr: int = 22050

    # Pitch curve smoothing: median filter window (in frames).
    median_filter_frames: int = 7

    # Short unvoiced gaps below this are interpolated over (seconds).
    max_interpolate_gap_s: float = 0.08

    # --- Segmentation thresholds ---
    # Pitch jump >= this many semitones triggers a note boundary.
    pitch_change_threshold_st: float = 1.5
    # Number of consecutive frames a pitch jump must persist before splitting.
    pitch_persistence_frames: int = 3
    # Unvoiced gap >= this duration triggers a note boundary (seconds).
    silence_gap_s: float = 0.12

    # --- Fragment cleanup ---
    # Notes shorter than this are discarded (seconds).
    min_note_duration_s: float = 0.10
    # Adjacent same-pitch notes separated by < this gap are merged.
    merge_gap_s: float = 0.15
    # Pitch tolerance for merging (semitones).
    merge_pitch_tolerance_st: int = 1


@dataclass
class PyinStats:
    """Diagnostic counters from pyin extraction."""
    total_frames: int = 0
    voiced_frames: int = 0
    voiced_pct: float = 0.0
    notes_raw: int = 0
    notes_after_cleanup: int = 0
    notes_absorbed: int = 0
    pitch_range_midi: Tuple[int, int] = (0, 0)
    avg_duration_s: float = 0.0
    median_duration_s: float = 0.0
    short_notes_pct: float = 0.0
    adjacent_within_1st_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "voiced_frames": self.voiced_frames,
            "voiced_pct": round(self.voiced_pct, 1),
            "notes_raw": self.notes_raw,
            "notes_after_cleanup": self.notes_after_cleanup,
            "notes_absorbed": self.notes_absorbed,
            "pitch_range_midi": list(self.pitch_range_midi),
            "avg_duration_s": round(self.avg_duration_s, 4),
            "median_duration_s": round(self.median_duration_s, 4),
            "short_notes_pct": round(self.short_notes_pct, 1),
            "adjacent_within_1st_pct": round(self.adjacent_within_1st_pct, 1),
        }


def extract_melody_pyin(
    audio_path: str,
    config: Optional[PyinConfig] = None,
) -> Tuple[List[NoteEvent], PyinStats]:
    """
    Extract a monophonic vocal melody from audio using pyin.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (typically the separated vocal stem).
    config : PyinConfig, optional

    Returns
    -------
    (notes, stats)
        List of NoteEvent and diagnostic stats.
    """
    if config is None:
        config = PyinConfig()

    import librosa

    stats = PyinStats()

    # ── 1. Load audio ──────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=config.sr)

    # ── 2. Run pyin ────────────────────────────────────────────────────
    fmin = librosa.note_to_hz(config.fmin_note)
    fmax = librosa.note_to_hz(config.fmax_note)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
    )

    hop_length = 512  # librosa default for pyin
    frame_duration_s = hop_length / sr
    n_frames = len(f0)

    stats.total_frames = n_frames
    stats.voiced_frames = int(np.sum(voiced_flag))
    stats.voiced_pct = (
        100.0 * stats.voiced_frames / n_frames if n_frames > 0 else 0.0
    )

    logger.info(
        "pyin: %d frames, %d voiced (%.0f%%), frame_dur=%.4fs",
        n_frames, stats.voiced_frames, stats.voiced_pct, frame_duration_s,
    )

    # ── 3. Clean pitch curve ───────────────────────────────────────────
    midi_curve = _clean_pitch_curve(
        f0, voiced_flag, frame_duration_s, config)

    # ── 4. Segment into notes ──────────────────────────────────────────
    raw_notes = _segment_notes(midi_curve, frame_duration_s, config)
    stats.notes_raw = len(raw_notes)

    logger.info("pyin segmentation: %d raw note segments", len(raw_notes))

    # ── 5. Register correction ────────────────────────────────────────
    raw_notes = _correct_register(raw_notes)

    # ── 5b. Local pitch stabilization ─────────────────────────────────
    raw_notes = _stabilize_pitches(raw_notes)

    # ── 6. Cleanup ─────────────────────────────────────────────────────
    pre_cleanup = len(raw_notes)
    notes = _cleanup_notes(raw_notes, config)
    stats.notes_after_cleanup = len(notes)
    stats.notes_absorbed = pre_cleanup - len(notes)

    if notes:
        pitches = [n.midi_pitch for n in notes]
        durs = [n.duration_s for n in notes]
        stats.pitch_range_midi = (min(pitches), max(pitches))
        stats.avg_duration_s = sum(durs) / len(durs)
        stats.median_duration_s = float(np.median(durs))
        stats.short_notes_pct = 100.0 * sum(
            1 for d in durs if d < 0.20) / len(durs)
        if len(notes) > 1:
            adj_close = sum(
                1 for i in range(1, len(notes))
                if abs(notes[i].midi_pitch - notes[i-1].midi_pitch) <= 1)
            stats.adjacent_within_1st_pct = (
                100.0 * adj_close / (len(notes) - 1))

    logger.info(
        "pyin melody: %d notes  (range MIDI %d-%d, avg dur %.3fs)",
        len(notes),
        stats.pitch_range_midi[0], stats.pitch_range_midi[1],
        stats.avg_duration_s,
    )

    return notes, stats


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _clean_pitch_curve(
    f0: np.ndarray,
    voiced: np.ndarray,
    frame_dur_s: float,
    config: PyinConfig,
) -> np.ndarray:
    """
    Convert f0 (Hz) to MIDI, interpolate short gaps, median-smooth.

    Returns an array of MIDI pitch values where unvoiced frames are NaN.
    """
    import librosa

    midi = np.full(len(f0), np.nan)
    valid = voiced & np.isfinite(f0) & (f0 > 0)
    midi[valid] = librosa.hz_to_midi(f0[valid])

    # Interpolate short unvoiced gaps
    max_gap_frames = max(1, int(config.max_interpolate_gap_s / frame_dur_s))
    midi = _interpolate_short_gaps(midi, max_gap_frames)

    # Median filter for smoothing (only over voiced frames)
    k = config.median_filter_frames
    if k >= 3:
        smoothed = np.copy(midi)
        valid_mask = np.isfinite(midi)
        if np.sum(valid_mask) >= k:
            from scipy.ndimage import median_filter
            values = midi[valid_mask]
            filtered = median_filter(values, size=k)
            smoothed[valid_mask] = filtered
        midi = smoothed

    return midi


def _interpolate_short_gaps(
    midi: np.ndarray, max_gap_frames: int
) -> np.ndarray:
    """Fill NaN gaps shorter than max_gap_frames with linear interpolation."""
    result = np.copy(midi)
    n = len(result)
    i = 0

    while i < n:
        if np.isnan(result[i]):
            gap_start = i
            while i < n and np.isnan(result[i]):
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len <= max_gap_frames:
                left = result[gap_start - 1] if gap_start > 0 else np.nan
                right = result[gap_end] if gap_end < n else np.nan

                if np.isfinite(left) and np.isfinite(right):
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_len + 1)
                        result[j] = left + t * (right - left)
                elif np.isfinite(left):
                    result[gap_start:gap_end] = left
                elif np.isfinite(right):
                    result[gap_start:gap_end] = right
        else:
            i += 1

    return result


def _segment_notes(
    midi_curve: np.ndarray,
    frame_dur_s: float,
    config: PyinConfig,
) -> List[NoteEvent]:
    """
    Split the cleaned MIDI pitch curve into discrete note events.

    Boundaries are placed at:
    - pitch changes > pitch_change_threshold_st that persist for
      pitch_persistence_frames consecutive frames
    - unvoiced gaps > silence_gap_s
    """
    n = len(midi_curve)
    if n == 0:
        return []

    threshold = config.pitch_change_threshold_st
    persistence = config.pitch_persistence_frames

    notes: List[NoteEvent] = []
    seg_start: Optional[int] = None
    seg_pitches: List[float] = []

    # Tracks how many consecutive frames have deviated from the segment median.
    deviant_run: int = 0
    deviant_start: int = 0

    for i in range(n):
        val = midi_curve[i]

        if np.isnan(val):
            if seg_start is not None and seg_pitches:
                notes.append(_make_note(
                    seg_start, i, seg_pitches, frame_dur_s))
                seg_start = None
                seg_pitches = []
                deviant_run = 0
            continue

        if seg_start is None:
            seg_start = i
            seg_pitches = [val]
            deviant_run = 0
            continue

        current_median = float(
            np.median(seg_pitches[-min(8, len(seg_pitches)):]))
        jump = abs(val - current_median)

        if jump >= threshold:
            if deviant_run == 0:
                deviant_start = i
            deviant_run += 1

            if deviant_run >= persistence:
                # Confirmed pitch change — split at deviant_start.
                # Pitches up to deviant_start belong to the old segment.
                old_pitches = seg_pitches[:len(seg_pitches) - (deviant_run - 1)]
                if old_pitches:
                    notes.append(_make_note(
                        seg_start, deviant_start, old_pitches, frame_dur_s))

                # New segment starts at deviant_start and includes
                # all deviant frames plus the current one.
                seg_start = deviant_start
                seg_pitches = list(
                    midi_curve[deviant_start:i + 1][
                        np.isfinite(midi_curve[deviant_start:i + 1])])
                deviant_run = 0
            else:
                seg_pitches.append(val)
        else:
            deviant_run = 0
            seg_pitches.append(val)

    if seg_start is not None and seg_pitches:
        notes.append(_make_note(
            seg_start, n, seg_pitches, frame_dur_s))

    return notes


def _make_note(
    start_frame: int,
    end_frame: int,
    pitches: List[float],
    frame_dur_s: float,
) -> NoteEvent:
    """Create a NoteEvent from a segment of pitch values.

    Uses a two-step pitch estimator:
    1. Trim attack/release edges (~15% each end).
    2. Round each remaining frame to the nearest integer MIDI pitch
       and pick the most frequent value (mode).  Falls back to median
       if no clear mode exists.
    """
    arr = np.array(pitches)
    n = len(arr)
    if n >= 6:
        trim = max(1, n // 7)
        arr = arr[trim:-trim]

    # Histogram mode: most frequent rounded MIDI pitch
    rounded = np.round(arr).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    midi_pitch = int(values[np.argmax(counts)])

    return NoteEvent(
        start_s=start_frame * frame_dur_s,
        end_s=end_frame * frame_dur_s,
        midi_pitch=midi_pitch,
        confidence=1.0,
    )


def _correct_register(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Fix systematic octave-down tracking from pyin.

    Uses the upper quartile (75th percentile) as the anchor register.
    Any note more than 8 semitones below the UQ is shifted up by 12,
    provided the result doesn't exceed a ceiling.
    """
    if len(notes) < 3:
        return notes

    pitches = np.array([n.midi_pitch for n in notes])
    global_uq = float(np.percentile(pitches, 75))
    ceiling = 96

    corrected = 0
    for note in notes:
        if note.midi_pitch < global_uq - 8:
            shifted = note.midi_pitch + 12
            if shifted <= ceiling:
                note.midi_pitch = shifted
                corrected += 1

    if corrected:
        logger.info(
            "Register correction: shifted %d/%d notes +12 "
            "(UQ anchor=%.0f)", corrected, len(notes), global_uq)

    return notes


def _stabilize_pitches(
    notes: List[NoteEvent],
    sandwich_jump_st: int = 7,
) -> List[NoteEvent]:
    """Remove clear sandwich-outlier notes.

    A note is a sandwich outlier when it creates jumps >= threshold
    both entering and leaving, AND the neighbors are close in pitch
    (within 3 semitones).  Only the most obvious cases are removed;
    ambiguous situations are left alone to avoid making things worse.
    """
    if len(notes) < 3:
        return notes

    keep = [True] * len(notes)
    removed = 0

    for i in range(1, len(notes) - 1):
        if not keep[i]:
            continue

        note = notes[i]
        prev_pitch = notes[i - 1].midi_pitch if keep[i - 1] else None
        next_pitch = notes[i + 1].midi_pitch if keep[i + 1] else None

        if prev_pitch is None or next_pitch is None:
            continue

        jump_in = abs(note.midi_pitch - prev_pitch)
        jump_out = abs(note.midi_pitch - next_pitch)
        neighbor_diff = abs(prev_pitch - next_pitch)

        if (jump_in >= sandwich_jump_st and
                jump_out >= sandwich_jump_st and
                neighbor_diff <= 3):
            keep[i] = False
            removed += 1

    result = [n for n, k in zip(notes, keep) if k]

    if removed:
        logger.info("Pitch stabilization: removed %d/%d sandwich outliers",
                    removed, len(notes))

    return result


def _absorb_short_fragments(
    notes: List[NoteEvent],
    max_short_s: float = 0.20,
    min_neighbor_ratio: float = 1.5,
) -> List[NoteEvent]:
    """Absorb very short notes into their longer neighbors.

    A short note flanked by notes that are each at least
    ``min_neighbor_ratio`` times longer is likely an artifact.
    We extend the closer-pitch neighbor to cover it.
    """
    if len(notes) < 3:
        return notes

    keep = [True] * len(notes)
    for i in range(1, len(notes) - 1):
        dur = notes[i].duration_s
        if dur > max_short_s:
            continue

        prev_dur = notes[i - 1].duration_s
        next_dur = notes[i + 1].duration_s

        if (prev_dur >= dur * min_neighbor_ratio and
                next_dur >= dur * min_neighbor_ratio):
            # Absorb into the neighbor whose pitch is closer
            diff_prev = abs(notes[i].midi_pitch - notes[i - 1].midi_pitch)
            diff_next = abs(notes[i].midi_pitch - notes[i + 1].midi_pitch)

            if diff_prev <= diff_next:
                notes[i - 1].end_s = max(notes[i - 1].end_s, notes[i].end_s)
            else:
                notes[i + 1].start_s = min(notes[i + 1].start_s,
                                           notes[i].start_s)
            keep[i] = False

    return [n for n, k in zip(notes, keep) if k]


def _cleanup_notes(
    notes: List[NoteEvent],
    config: PyinConfig,
) -> List[NoteEvent]:
    """
    Post-processing: drop micro-fragments, merge same-pitch adjacencies.
    """
    if not notes:
        return []

    # 1. Drop notes shorter than minimum duration
    notes = [n for n in notes if n.duration_s >= config.min_note_duration_s]

    # 2. Merge adjacent notes within pitch tolerance and with small gaps
    if len(notes) < 2:
        return notes

    merged: List[NoteEvent] = [NoteEvent(
        notes[0].start_s, notes[0].end_s,
        notes[0].midi_pitch, notes[0].confidence)]

    for i in range(1, len(notes)):
        prev = merged[-1]
        curr = notes[i]
        gap = curr.start_s - prev.end_s
        pitch_diff = abs(curr.midi_pitch - prev.midi_pitch)

        if (gap <= config.merge_gap_s and
                pitch_diff <= config.merge_pitch_tolerance_st):
            # When merging, keep the pitch of the longer note
            prev_dur = prev.end_s - prev.start_s
            curr_dur = curr.end_s - curr.start_s
            if curr_dur > prev_dur:
                prev.midi_pitch = curr.midi_pitch
            prev.end_s = curr.end_s
        else:
            merged.append(NoteEvent(
                curr.start_s, curr.end_s,
                curr.midi_pitch, curr.confidence))

    # 3. Absorb short fragments into neighbors.
    # If a note is very short and its neighbors are both longer, it's
    # likely an artifact — extend the closer-pitch neighbor to cover it.
    absorbed = _absorb_short_fragments(merged, max_short_s=0.18,
                                       min_neighbor_ratio=2.0)

    # 4. Final min-duration filter after merging
    absorbed = [n for n in absorbed if n.duration_s >= config.min_note_duration_s]

    return absorbed
