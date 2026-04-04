"""
Stage C — Beat / measure grid construction.

Uses librosa beat tracking to estimate tempo and beat positions, then
builds a quantization grid at 1/16-note (or 1/8 / triplet) resolution.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger("mp3_to_sheet.beat_grid")


@dataclass
class BeatGrid:
    """Tempo, beat positions, and time-signature info."""

    tempo_bpm: float
    beat_times: np.ndarray        # seconds
    time_sig_num: int = 4
    time_sig_den: int = 4
    first_beat_s: float = 0.0     # anchor for time ↔ ql conversion

    # -- derived helpers -----------------------------------------------------

    @property
    def beat_duration_s(self) -> float:
        return 60.0 / self.tempo_bpm

    @property
    def measure_duration_ql(self) -> float:
        """Quarter-note length of one full measure."""
        # 4/4 → 4,  3/4 → 3,  6/8 → 3  (6 eighths = 3 quarters)
        return self.time_sig_num * (4.0 / self.time_sig_den)

    def time_to_ql(self, time_s: float) -> float:
        """Seconds → quarter-note offset from score start."""
        return (time_s - self.first_beat_s) * (self.tempo_bpm / 60.0)

    def ql_to_time(self, ql: float) -> float:
        """Quarter-note offset → seconds."""
        return self.first_beat_s + ql * (60.0 / self.tempo_bpm)


def detect_beats(
    wav_path: str,
    time_signature: str = "auto",
    sr: int = 22050,
) -> BeatGrid:
    """
    Estimate tempo and beat positions from audio.

    Parameters
    ----------
    wav_path : str
        Mono WAV file (from ingest stage).
    time_signature : str
        ``"auto"`` | ``"4/4"`` | ``"3/4"`` | ``"6/8"`` …
    sr : int
        Expected sample rate.

    Returns
    -------
    BeatGrid
    """
    import librosa

    logger.info("Loading audio for beat tracking: %s", wav_path)
    y, sr = librosa.load(wav_path, sr=sr)

    logger.info("Running beat tracking …")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # librosa ≥ 0.10 may return tempo as ndarray
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if tempo.size > 0 else 120.0
    tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    logger.info("Detected tempo: %.1f BPM, %d beats", tempo, len(beat_times))

    # Half-time correction: librosa often picks the double-time pulse.
    # Songs with a felt tempo of ~60-80 BPM get reported as ~120-160.
    tempo, beat_times = _correct_double_time(tempo, beat_times, y, sr)

    ts_num, ts_den = _parse_time_signature(time_signature)
    first_beat = float(beat_times[0]) if len(beat_times) > 0 else 0.0

    grid = BeatGrid(
        tempo_bpm=round(tempo, 1),
        beat_times=beat_times,
        time_sig_num=ts_num,
        time_sig_den=ts_den,
        first_beat_s=first_beat,
    )

    logger.info(
        "BeatGrid ready — tempo=%.1f BPM  ts=%d/%d  measure=%.2f ql",
        grid.tempo_bpm, ts_num, ts_den, grid.measure_duration_ql,
    )
    return grid


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _correct_double_time(
    tempo: float,
    beat_times: np.ndarray,
    y: np.ndarray,
    sr: int,
) -> tuple:
    """
    Detect whether librosa picked double-time and correct to half-time.

    Heuristic: if tempo > 100 BPM, compute the onset density (onsets per
    second). Vocal-led pop songs at a felt ~72 BPM typically have 2-4
    onsets per second. If onset density is low relative to the reported
    BPM, it's likely double-time.

    Also checks whether halved tempo falls in a more typical pop range
    (55-90 BPM) and whether the beat intervals suggest a half-time feel.
    """
    import librosa

    if tempo <= 100.0:
        return tempo, beat_times

    halved = tempo / 2.0

    # Only correct if halved tempo lands in a reasonable range
    if halved < 50.0 or halved > 100.0:
        return tempo, beat_times

    # Compute onset density as supporting evidence
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    duration_s = len(y) / sr

    if duration_s > 0:
        onset_density = len(onset_times) / duration_s
    else:
        onset_density = 0.0

    # At 143 BPM we'd expect ~2.4 beats/s; at low onset density the true
    # pulse is likely half that.
    expected_onsets_per_beat = onset_density / (tempo / 60.0)

    logger.info(
        "Double-time check: BPM=%.1f, halved=%.1f, "
        "onset_density=%.1f/s, onsets_per_beat=%.2f",
        tempo, halved, onset_density, expected_onsets_per_beat,
    )

    # If fewer than ~1.5 onsets per reported beat, the reported tempo is
    # likely double the felt pulse.
    if expected_onsets_per_beat < 1.8:
        logger.info(
            "Correcting double-time: %.1f BPM → %.1f BPM", tempo, halved)
        # Keep every other beat to match the halved tempo
        beat_times = beat_times[::2]
        return halved, beat_times

    return tempo, beat_times


_TS_MAP = {
    "4/4": (4, 4),
    "3/4": (3, 4),
    "6/8": (6, 8),
    "2/4": (2, 4),
    "2/2": (2, 2),
}


def _parse_time_signature(ts_str: str) -> Tuple[int, int]:
    if ts_str in ("auto", ""):
        return 4, 4           # sensible default

    if ts_str in _TS_MAP:
        return _TS_MAP[ts_str]

    try:
        num, den = ts_str.split("/")
        return int(num), int(den)
    except (ValueError, IndexError):
        logger.warning("Unknown time signature '%s', defaulting to 4/4", ts_str)
        return 4, 4
