"""
Audio-based note segmentation — split over-long notes using audio evidence.

After the melody-path resolver and vocal abstraction produce a note stream,
many notes are too long because they span multiple sung syllables that the
pitch-based pipeline merged together. This module uses the *audio signal*
to find split points inside notes:

  1. **Energy dips** — short-time RMS drops inside a note indicate a
     consonant, breath, or syllable boundary.
  2. **Onset detection** — librosa onset detection finds transient attacks
     inside notes.
  3. **Pitch discontinuities** — sudden pitch resets (even small ones)
     from the raw pitch track indicate re-articulation.

A note is split when audio evidence supports it and the resulting
fragments exceed a minimum duration.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.audio_segmentation")


@dataclass
class AudioSegmentationConfig:
    """Tuneable parameters for audio-based segmentation."""

    enabled: bool = True

    # Only attempt to split notes longer than this (seconds).
    min_note_duration_to_split_s: float = 0.40

    # Minimum duration of a resulting fragment (seconds).
    min_fragment_duration_s: float = 0.12

    # --- Energy dip detection ---
    energy_enabled: bool = True
    energy_frame_ms: float = 20.0
    energy_hop_ms: float = 10.0
    # A dip must drop below this fraction of the note's peak RMS.
    energy_dip_ratio: float = 0.30
    # Minimum prominence of a dip relative to surrounding energy.
    energy_min_prominence: float = 0.20

    # --- Onset detection ---
    onset_enabled: bool = True
    # Onset strength threshold (relative to max within the note).
    onset_relative_threshold: float = 0.5

    # --- Pitch discontinuity detection ---
    pitch_reset_enabled: bool = True
    # Minimum pitch jump in semitones to count as a discontinuity.
    pitch_reset_min_jump: float = 1.5
    # Frame size for pitch analysis (seconds).
    pitch_frame_s: float = 0.03

    # --- Combination logic ---
    # Require at least this many independent signals to agree for a split.
    min_evidence_count: int = 1


@dataclass
class SegmentationStats:
    """Diagnostic counters."""
    notes_before: int = 0
    notes_after: int = 0
    notes_split: int = 0
    splits_by_energy: int = 0
    splits_by_onset: int = 0
    splits_by_pitch: int = 0
    avg_duration_before: float = 0.0
    avg_duration_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "notes_before": self.notes_before,
            "notes_after": self.notes_after,
            "notes_split": self.notes_split,
            "splits_by_energy": self.splits_by_energy,
            "splits_by_onset": self.splits_by_onset,
            "splits_by_pitch": self.splits_by_pitch,
            "avg_duration_before": round(self.avg_duration_before, 4),
            "avg_duration_after": round(self.avg_duration_after, 4),
        }


def segment_notes_from_audio(
    notes: List[NoteEvent],
    audio_path: str,
    config: Optional[AudioSegmentationConfig] = None,
) -> Tuple[List[NoteEvent], SegmentationStats]:
    """
    Split over-long notes using audio evidence (energy, onsets, pitch).

    Parameters
    ----------
    notes : list[NoteEvent]
        Post-abstraction / post-refinement note stream.
    audio_path : str
        Path to the audio file used for transcription (the vocal stem).
    config : AudioSegmentationConfig, optional

    Returns
    -------
    (refined_notes, stats)
    """
    if config is None:
        config = AudioSegmentationConfig()

    stats = SegmentationStats(notes_before=len(notes))
    if notes:
        stats.avg_duration_before = sum(n.duration_s for n in notes) / len(notes)

    if not config.enabled or not notes:
        stats.notes_after = len(notes)
        stats.avg_duration_after = stats.avg_duration_before
        return list(notes), stats

    import librosa

    y, sr = librosa.load(audio_path, sr=22050)

    # Pre-compute full-track onset strength envelope (used per-note)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times_all = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr)

    result: List[NoteEvent] = []

    for note in notes:
        if note.duration_s < config.min_note_duration_to_split_s:
            result.append(note)
            continue

        split_points = _find_split_points(
            y, sr, note, onset_env, onset_times_all, config)

        if not split_points:
            result.append(note)
            continue

        fragments = _apply_splits(note, split_points, config.min_fragment_duration_s)

        if len(fragments) > 1:
            result.extend(fragments)
            stats.notes_split += 1
            for reason in split_points:
                if "energy" in reason[1]:
                    stats.splits_by_energy += 1
                if "onset" in reason[1]:
                    stats.splits_by_onset += 1
                if "pitch" in reason[1]:
                    stats.splits_by_pitch += 1
        else:
            result.append(note)

    stats.notes_after = len(result)
    if result:
        stats.avg_duration_after = sum(n.duration_s for n in result) / len(result)

    logger.info(
        "Audio segmentation: %d → %d notes (%d split, "
        "energy=%d onset=%d pitch=%d)",
        stats.notes_before, stats.notes_after, stats.notes_split,
        stats.splits_by_energy, stats.splits_by_onset,
        stats.splits_by_pitch,
    )

    return result, stats


def _find_split_points(
    y: np.ndarray,
    sr: int,
    note: NoteEvent,
    onset_env: np.ndarray,
    onset_times: np.ndarray,
    config: AudioSegmentationConfig,
) -> List[Tuple[float, str]]:
    """
    Find candidate split points inside a note using multiple signals.

    Returns list of (time_s, reason_string) for each candidate split.
    """
    import librosa

    start_sample = int(note.start_s * sr)
    end_sample = int(note.end_s * sr)
    end_sample = min(end_sample, len(y))

    if end_sample <= start_sample:
        return []

    segment = y[start_sample:end_sample]
    candidates: Dict[int, List[str]] = {}

    # Quantize split times to 10ms grid to merge nearby candidates
    grid_ms = 10

    def _to_grid(t_s: float) -> int:
        return round((t_s - note.start_s) * 1000 / grid_ms)

    def _from_grid(g: int) -> float:
        return note.start_s + g * grid_ms / 1000.0

    # --- Energy dips ---
    if config.energy_enabled and len(segment) > 0:
        frame_len = int(config.energy_frame_ms / 1000.0 * sr)
        hop_len = int(config.energy_hop_ms / 1000.0 * sr)
        frame_len = max(frame_len, 64)
        hop_len = max(hop_len, 32)

        rms = librosa.feature.rms(
            y=segment, frame_length=frame_len, hop_length=hop_len)[0]

        if len(rms) > 2:
            peak_rms = np.max(rms)
            if peak_rms > 0:
                threshold = peak_rms * config.energy_dip_ratio

                for i in range(1, len(rms) - 1):
                    if rms[i] < threshold:
                        left_max = np.max(rms[max(0, i-3):i])
                        right_max = np.max(rms[i+1:min(len(rms), i+4)])
                        prominence = min(left_max, right_max) - rms[i]

                        if prominence >= config.energy_min_prominence * peak_rms:
                            t_s = note.start_s + (i * hop_len) / sr
                            g = _to_grid(t_s)
                            candidates.setdefault(g, []).append("energy")

    # --- Onset detection ---
    if config.onset_enabled:
        mask = (onset_times >= note.start_s + config.min_fragment_duration_s) & \
               (onset_times <= note.end_s - config.min_fragment_duration_s)
        note_onset_env = onset_env.copy()
        note_onset_env[~(
            (onset_times >= note.start_s) & (onset_times <= note.end_s)
        )] = 0

        note_max = np.max(note_onset_env) if np.any(mask) else 0
        if note_max > 0:
            onset_threshold = note_max * config.onset_relative_threshold
            onset_indices = np.where(mask)[0]

            for idx in onset_indices:
                if onset_env[idx] >= onset_threshold:
                    t_s = onset_times[idx]
                    # Skip onsets too close to note boundaries
                    if (t_s - note.start_s >= config.min_fragment_duration_s and
                            note.end_s - t_s >= config.min_fragment_duration_s):
                        g = _to_grid(t_s)
                        candidates.setdefault(g, []).append("onset")

    # --- Pitch discontinuities ---
    if config.pitch_reset_enabled and len(segment) > 0:
        _find_pitch_resets(
            segment, sr, note, config, _to_grid, candidates)

    # Filter: require minimum evidence count
    valid = []
    for g, reasons in sorted(candidates.items()):
        unique_reasons = list(set(reasons))
        if len(unique_reasons) >= config.min_evidence_count:
            t_s = _from_grid(g)
            valid.append((t_s, "+".join(unique_reasons)))

    return valid


def _find_pitch_resets(
    segment: np.ndarray,
    sr: int,
    note: NoteEvent,
    config: AudioSegmentationConfig,
    to_grid_fn,
    candidates: dict,
):
    """Detect pitch discontinuities inside a note segment."""
    import librosa

    frame_len = int(config.pitch_frame_s * sr)
    hop_len = frame_len // 2
    if frame_len < 128 or len(segment) < frame_len * 2:
        return

    try:
        pitches, magnitudes = librosa.piptrack(
            y=segment, sr=sr,
            fmin=100, fmax=1200,
            n_fft=frame_len, hop_length=hop_len,
        )
    except Exception:
        return

    n_frames = pitches.shape[1]
    dominant_pitches = []
    for i in range(n_frames):
        mag_col = magnitudes[:, i]
        idx = mag_col.argmax()
        p = pitches[idx, i]
        if p > 0 and mag_col[idx] > 0.01:
            dominant_pitches.append(p)
        else:
            dominant_pitches.append(0)

    # Convert to MIDI-like values for semitone comparison
    midi_pitches = []
    for p in dominant_pitches:
        if p > 0:
            midi_pitches.append(12 * np.log2(p / 440.0) + 69)
        else:
            midi_pitches.append(None)

    # Find jumps
    for i in range(1, len(midi_pitches)):
        p_prev = midi_pitches[i - 1]
        p_curr = midi_pitches[i]
        if p_prev is None or p_curr is None:
            continue

        jump = abs(p_curr - p_prev)
        if jump >= config.pitch_reset_min_jump:
            t_s = note.start_s + (i * hop_len) / sr
            if (t_s - note.start_s >= config.min_fragment_duration_s and
                    note.end_s - t_s >= config.min_fragment_duration_s):
                g = to_grid_fn(t_s)
                candidates.setdefault(g, []).append("pitch")


def _apply_splits(
    note: NoteEvent,
    split_points: List[Tuple[float, str]],
    min_fragment_s: float,
) -> List[NoteEvent]:
    """
    Split a note at the given time points, producing fragment NoteEvents.
    Merges splits that are too close together.
    """
    times = sorted(set(sp[0] for sp in split_points))

    if not times:
        return [note]

    # Merge splits that are too close
    filtered = [times[0]]
    for t in times[1:]:
        if t - filtered[-1] >= min_fragment_s:
            filtered.append(t)
    times = filtered

    # Remove splits too close to note boundaries
    times = [t for t in times
             if (t - note.start_s >= min_fragment_s and
                 note.end_s - t >= min_fragment_s)]

    if not times:
        return [note]

    boundaries = [note.start_s] + times + [note.end_s]
    fragments = []

    for i in range(len(boundaries) - 1):
        frag = NoteEvent(
            start_s=boundaries[i],
            end_s=boundaries[i + 1],
            midi_pitch=note.midi_pitch,
            confidence=note.confidence,
        )
        if frag.duration_s >= min_fragment_s:
            fragments.append(frag)

    # If splitting produced only one viable fragment, return original
    if len(fragments) <= 1:
        return [note]

    return fragments
