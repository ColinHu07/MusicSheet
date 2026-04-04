"""
Melody-path resolver configuration.

All tuneable weights and thresholds live here, not scattered as magic numbers.
Each parameter group has a brief explanation of what it controls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class InstrumentRange:
    """MIDI pitch bounds and melodic centre for an instrument."""
    low: int
    high: int
    center: int       # most comfortable melodic register centre
    name: str = ""

    def contains(self, midi_pitch: int) -> bool:
        return self.low <= midi_pitch <= self.high


INSTRUMENT_RANGES: Dict[str, InstrumentRange] = {
    "violin":  InstrumentRange(low=55,  high=100, center=76, name="Violin"),   # G3–E7
    "flute":   InstrumentRange(low=60,  high=96,  center=79, name="Flute"),    # C4–C7
    "viola":   InstrumentRange(low=48,  high=91,  center=67, name="Viola"),    # C3–G6
    "cello":   InstrumentRange(low=36,  high=76,  center=57, name="Cello"),    # C2–E5
}


@dataclass
class MelodyConfig:
    """All parameters for the melody-path resolver."""

    # -- candidate filtering --------------------------------------------------
    min_note_duration_ms: float = 50.0
    max_note_duration_s: float = 30.0
    min_confidence: float = 0.15

    # -- intrinsic note scoring weights (higher = more influence) -------------
    confidence_weight: float = 1.0
    duration_weight: float = 3.0
    register_weight: float = 2.5
    salience_weight: float = 0.3
    instrument_plausibility_weight: float = 1.2

    # -- intrinsic note penalties ---------------------------------------------
    short_note_penalty_weight: float = 2.5
    short_note_threshold_ms: float = 150.0
    clutter_penalty_weight: float = 2.5
    clutter_window_ms: float = 250.0
    clutter_count_threshold: int = 4

    # -- transition scoring weights -------------------------------------------
    temporal_continuity_weight: float = 3.0
    leap_penalty_weight: float = 3.5
    octave_flip_penalty_weight: float = 6.0
    contour_smoothness_weight: float = 2.0
    phrase_continuity_weight: float = 3.5
    monophonic_overlap_penalty: float = 4.0

    # -- same-note continuation bonus -----------------------------------------
    same_note_continuation_bonus: float = 1.5

    # -- burst penalty (both prev and cur are very short) ---------------------
    burst_penalty_weight: float = 1.5
    burst_threshold_ms: float = 120.0

    # -- leap penalty details -------------------------------------------------
    leap_mild_threshold: int = 5      # semitones (P4)
    leap_large_threshold: int = 9     # semitones (M6)
    leap_extreme_threshold: int = 15  # semitones (> octave + m3)

    # -- gap/rest state -------------------------------------------------------
    allow_gap_state: bool = True
    gap_entry_cost: float = 1.2
    gap_continuation_cost: float = 0.15
    gap_max_duration_s: float = 4.0

    # -- register bias (global) ------------------------------------------------
    # "vocal"        — uses 75th percentile pitch; best for pop vocal extraction
    # "above_median" — uses median; moderate bias
    # "center"       — distance from instrument center; gentle
    # "none"         — no register bias
    register_bias_mode: str = "vocal"
    register_center_midi: int = 72

    # -- local register modeling (adaptive, time-windowed) ---------------------
    # Computes register targets in sliding windows instead of globally.
    # This lets the resolver adapt across verse/chorus register shifts.
    local_register_window_s: float = 2.0
    # "adaptive_vocal" — prefer notes in the local upper tier (recommended)
    # "upper_quartile" — target local 75th percentile
    # "median_plus_offset" — target local median + a 5th
    # "off" — disable local register scoring
    local_register_mode: str = "adaptive_vocal"
    local_register_weight: float = 3.0

    # -- relative register (local context comparison) --------------------------
    # Penalise notes below the local median pitch, reward those above it.
    below_local_median_penalty_weight: float = 2.0
    above_local_median_bonus_weight: float = 1.5
    high_outlier_penalty_weight: float = 0.5

    # -- stream persistence / identity -----------------------------------------
    # Rewards staying within a narrow pitch band (the same melodic stream)
    # and penalises abrupt switches to a different register band.
    same_band_semitones: int = 5
    same_band_bonus: float = 1.2
    stream_switch_penalty_weight: float = 2.5

    # -- pitch floor (optional pre-filter) ------------------------------------
    # When set, candidates below this MIDI pitch are removed before DP.
    # Useful for empirical testing on vocal-led material.
    pitch_floor_midi: Optional[int] = None

    # -- vocals-mode register enhancements ------------------------------------
    # When True, automatically compute a pitch floor from the candidate pool
    # to exclude lower-octave sub-harmonics that Basic Pitch often generates.
    # Only applies when pitch_floor_midi is not explicitly set.
    vocals_mode: bool = False
    vocals_auto_pitch_floor: bool = True
    # Floor is set at (upper_quartile - this many semitones).
    # 10 = removes only extreme sub-harmonics (> minor 7th below UQ),
    # keeping enough candidates for sections where Basic Pitch only
    # detects the lower octave of the vocal.  The global register
    # correction then fixes segments that ended up too low.
    vocals_floor_semitones_below_uq: int = 10
    # Post-DP correction: shift notes that are well below the instrument
    # center upward by an octave when evidence suggests a systematic error.
    vocals_global_register_correction: bool = True

    # -- post-selection cleanup -----------------------------------------------
    merge_gap_ms: float = 120.0
    merge_nearby_pitch_gap_ms: float = 80.0
    nearby_pitch_semitones: int = 1
    micro_note_ms: float = 80.0
    octave_smoothing: bool = True
    consolidate_repeated_pitches: bool = True
    consolidate_max_gap_ms: float = 150.0
    range_enforcement: bool = True
    range_enforcement_mode: str = "transpose"

    # -- sustain / phrase smoothing (post-DP) ---------------------------------
    pitch_stability_semitones: int = 1
    pitch_stability_max_gap_ms: float = 120.0
    phrase_smoothing_enabled: bool = True
    phrase_max_short_note_ms: float = 180.0
    phrase_max_pitch_diff: int = 2
    reattack_suppression_enabled: bool = True
    reattack_max_gap_ms: float = 120.0
    post_quantize_merge: bool = True

    # -- instrument -----------------------------------------------------------
    instrument: str = "violin"

    # -- debug ----------------------------------------------------------------
    save_debug_artifacts: bool = True

    def get_instrument_range(self) -> InstrumentRange:
        return INSTRUMENT_RANGES.get(
            self.instrument.lower(),
            INSTRUMENT_RANGES["violin"],
        )
