"""
Tests for the vocals-mode register fix.

Covers:
  A. Auto pitch floor
  B. Register preference (octave-equivalent selection)
  C. Global register correction
  D. Regression: contour, note count, non-vocals mode unchanged
"""

import pytest
from src.config.melody_config import MelodyConfig, InstrumentRange, INSTRUMENT_RANGES
from src.pipeline.candidate_notes import CandidateNote
from src.pipeline.melody_path import resolve_melody, _global_register_correction
from src.pipeline.melody_features import (
    score_note_intrinsic,
    _compute_register_bonus,
    _compute_local_register_bonus,
)
from src.pipeline.local_register import LocalRegisterContext
from src.pipeline.utils import NoteEvent


def _cand(start, end, pitch, conf=0.8, salience=0.5):
    """Helper to create a CandidateNote."""
    return CandidateNote(
        start_s=start, end_s=end, pitch_midi=pitch,
        confidence=conf, salience=salience,
    )


# ═══════════════════════════════════════════════════════════════════════
# A. AUTO PITCH FLOOR
# ═══════════════════════════════════════════════════════════════════════

class TestAutoPitchFloor:

    def test_auto_floor_computed_in_vocals_mode(self):
        """When vocals_mode is True and no explicit floor, an auto floor is computed."""
        candidates = [
            _cand(0.0, 0.5, 56),  # Ab3 — sub-harmonic
            _cand(0.0, 0.5, 58),  # Bb3 — sub-harmonic
            _cand(0.5, 1.0, 68),  # Ab4 — real vocal
            _cand(1.0, 1.5, 70),  # Bb4 — real vocal
            _cand(1.5, 2.0, 72),  # C5  — real vocal
        ]
        cfg = MelodyConfig(vocals_mode=True, vocals_auto_pitch_floor=True)
        result = resolve_melody(candidates, cfg)
        assert result.stats["auto_floor_applied"] is True
        assert result.stats["pitch_floor_midi_used"] is not None
        assert result.stats["pitch_floor_filtered"] > 0

    def test_auto_floor_removes_sub_harmonics(self):
        """Sub-harmonic candidates below the auto floor should be removed."""
        high_notes = [_cand(i * 0.3, i * 0.3 + 0.25, 68 + (i % 5)) for i in range(10)]
        low_notes = [_cand(i * 0.3, i * 0.3 + 0.25, 56 + (i % 3)) for i in range(5)]
        candidates = low_notes + high_notes

        cfg = MelodyConfig(vocals_mode=True, vocals_auto_pitch_floor=True)
        result = resolve_melody(candidates, cfg)

        selected_pitches = [n.midi_pitch for n in result.selected_notes]
        assert all(p >= 60 for p in selected_pitches), \
            f"Sub-harmonic notes should be filtered: {selected_pitches}"

    def test_auto_floor_disabled_when_explicit_floor_set(self):
        """An explicit pitch_floor_midi takes precedence over auto-floor."""
        candidates = [
            _cand(0.0, 0.5, 50),
            _cand(0.5, 1.0, 68),
        ]
        cfg = MelodyConfig(
            vocals_mode=True,
            vocals_auto_pitch_floor=True,
            pitch_floor_midi=55,
        )
        result = resolve_melody(candidates, cfg)
        assert result.stats["pitch_floor_midi_used"] == 55
        assert result.stats["auto_floor_applied"] is False

    def test_auto_floor_not_applied_in_mix_mode(self):
        """In non-vocals (mix) mode, auto-floor should not engage."""
        candidates = [
            _cand(0.0, 0.5, 48),
            _cand(0.5, 1.0, 60),
            _cand(1.0, 1.5, 72),
        ]
        cfg = MelodyConfig(vocals_mode=False, vocals_auto_pitch_floor=True)
        result = resolve_melody(candidates, cfg)
        assert result.stats["auto_floor_applied"] is False
        assert result.stats["pitch_floor_midi_used"] is None

    def test_auto_floor_can_be_disabled(self):
        """Setting vocals_auto_pitch_floor=False prevents auto floor."""
        candidates = [
            _cand(0.0, 0.5, 56),
            _cand(0.5, 1.0, 68),
        ]
        cfg = MelodyConfig(vocals_mode=True, vocals_auto_pitch_floor=False)
        result = resolve_melody(candidates, cfg)
        assert result.stats["auto_floor_applied"] is False


# ═══════════════════════════════════════════════════════════════════════
# B. REGISTER PREFERENCE
# ═══════════════════════════════════════════════════════════════════════

class TestRegisterPreference:

    def test_vocals_mode_prefers_higher_octave(self):
        """Given simultaneous candidates at Ab3 and Ab4, vocals mode should prefer Ab4."""
        candidates = []
        for i in range(10):
            t = i * 0.4
            candidates.append(_cand(t, t + 0.35, 56, conf=0.85))  # Ab3
            candidates.append(_cand(t, t + 0.35, 68, conf=0.80))  # Ab4

        cfg = MelodyConfig(vocals_mode=True, vocals_auto_pitch_floor=True)
        result = resolve_melody(candidates, cfg)

        selected_pitches = [n.midi_pitch for n in result.selected_notes]
        high_count = sum(1 for p in selected_pitches if p >= 64)
        low_count = sum(1 for p in selected_pitches if p < 64)
        assert high_count > low_count, \
            f"Should prefer higher octave: high={high_count}, low={low_count}"

    def test_vocals_register_bonus_steeper_than_default(self):
        """In vocals mode, the register bonus penalty for low notes is steeper."""
        cfg_vocals = MelodyConfig(vocals_mode=True, register_bias_mode="vocal")
        cfg_default = MelodyConfig(vocals_mode=False, register_bias_mode="vocal")
        inst = INSTRUMENT_RANGES["violin"]

        target = 70.0
        low_pitch = 58  # a full octave below target

        bonus_vocals = _compute_register_bonus(low_pitch, cfg_vocals, inst, 63.0, target)
        bonus_default = _compute_register_bonus(low_pitch, cfg_default, inst, 63.0, target)

        assert bonus_vocals < bonus_default, \
            f"Vocals mode should penalize low pitches more: vocals={bonus_vocals}, default={bonus_default}"

    def test_vocals_local_register_bonus_steeper(self):
        """In vocals mode, local register bonus penalizes low notes harder."""
        cfg_vocals = MelodyConfig(vocals_mode=True, local_register_mode="adaptive_vocal")
        cfg_default = MelodyConfig(vocals_mode=False, local_register_mode="adaptive_vocal")

        ctx = LocalRegisterContext(
            local_median=65.0, local_upper_quartile=70.0,
            local_min=55.0, local_max=80.0,
            relative_position=0.3, notes_in_window=20,
        )

        bonus_vocals = _compute_local_register_bonus(58, cfg_vocals, ctx)
        bonus_default = _compute_local_register_bonus(58, cfg_default, ctx)

        assert bonus_vocals < bonus_default, \
            f"Vocals local bonus should be steeper: vocals={bonus_vocals}, default={bonus_default}"


# ═══════════════════════════════════════════════════════════════════════
# C. GLOBAL REGISTER CORRECTION
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalRegisterCorrection:

    def test_shifts_low_segment(self):
        """A segment whose median is > 12 below center shifts up 12."""
        inst = InstrumentRange(low=55, high=100, center=76)
        # Segment median = 58, gap = 76-58 = 18 > 12 → shift
        notes = [
            NoteEvent(start_s=0.0, end_s=0.5, midi_pitch=56),
            NoteEvent(start_s=0.5, end_s=1.0, midi_pitch=58),
            NoteEvent(start_s=1.0, end_s=1.5, midi_pitch=60),
        ]

        corrected, count = _global_register_correction(notes, inst)
        assert count == 3
        assert corrected[0].midi_pitch == 68  # 56 + 12
        assert corrected[1].midi_pitch == 70  # 58 + 12
        assert corrected[2].midi_pitch == 72  # 60 + 12

    def test_preserves_contour_within_segment(self):
        """Shifting a segment preserves melodic contour (intervals)."""
        inst = InstrumentRange(low=55, high=100, center=76)
        pitches = [56, 58, 60, 63, 60, 58]
        notes = [
            NoteEvent(start_s=i * 0.5, end_s=i * 0.5 + 0.4, midi_pitch=p)
            for i, p in enumerate(pitches)
        ]

        intervals_before = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        corrected, _ = _global_register_correction(notes, inst)
        corrected_pitches = [n.midi_pitch for n in corrected]
        intervals_after = [corrected_pitches[i+1] - corrected_pitches[i]
                           for i in range(len(corrected_pitches)-1)]
        assert intervals_before == intervals_after, "Contour must be preserved"

    def test_only_low_segment_shifts(self):
        """A low segment shifts, but a high segment in the same list does not."""
        inst = InstrumentRange(low=55, high=100, center=76)
        notes = [
            # Segment 1: low (median=58, gap=76-58=18 > 12) — should shift
            NoteEvent(start_s=0.0, end_s=0.5, midi_pitch=56),
            NoteEvent(start_s=0.5, end_s=1.0, midi_pitch=58),
            NoteEvent(start_s=1.0, end_s=1.5, midi_pitch=60),
            # Gap > 1.0s → new segment
            # Segment 2: high (median=72, gap=4 ≤ 12) — should NOT shift
            NoteEvent(start_s=3.0, end_s=3.5, midi_pitch=70),
            NoteEvent(start_s=3.5, end_s=4.0, midi_pitch=72),
            NoteEvent(start_s=4.0, end_s=4.5, midi_pitch=75),
        ]

        corrected, count = _global_register_correction(notes, inst)
        assert count == 3  # only first segment
        assert corrected[0].midi_pitch == 68   # 56 + 12
        assert corrected[1].midi_pitch == 70   # 58 + 12
        assert corrected[2].midi_pitch == 72   # 60 + 12
        assert corrected[3].midi_pitch == 70   # unchanged
        assert corrected[4].midi_pitch == 72   # unchanged
        assert corrected[5].midi_pitch == 75   # unchanged

    def test_no_shift_when_median_near_center(self):
        """Segment close to the instrument center should not be shifted."""
        inst = InstrumentRange(low=55, high=100, center=76)
        notes = [
            NoteEvent(start_s=0.0, end_s=0.5, midi_pitch=68),
            NoteEvent(start_s=0.5, end_s=1.0, midi_pitch=72),
        ]

        corrected, count = _global_register_correction(notes, inst)
        assert count == 0
        assert corrected[0].midi_pitch == 68
        assert corrected[1].midi_pitch == 72

    def test_empty_notes(self):
        """Empty list should return empty with 0 corrections."""
        inst = InstrumentRange(low=55, high=100, center=76)
        corrected, count = _global_register_correction([], inst)
        assert count == 0
        assert corrected == []


# ═══════════════════════════════════════════════════════════════════════
# D. REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestRegisterRegression:

    def test_mix_mode_unchanged(self):
        """Non-vocals mode should produce same results as before (no register boost)."""
        candidates = [
            _cand(0.0, 0.5, 60, conf=0.9),
            _cand(0.5, 1.0, 67, conf=0.8),
            _cand(1.0, 1.5, 72, conf=0.85),
        ]

        cfg = MelodyConfig(vocals_mode=False)
        result = resolve_melody(candidates, cfg)

        assert result.stats["auto_floor_applied"] is False
        assert result.stats["global_register_corrections"] == 0
        assert len(result.selected_notes) > 0

    def test_no_note_explosion(self):
        """Vocals register fix should not cause note count to increase wildly."""
        candidates = []
        for i in range(20):
            t = i * 0.5
            candidates.append(_cand(t, t + 0.4, 68 + (i % 5), conf=0.8))
        candidates += [_cand(i * 0.5, i * 0.5 + 0.4, 56 + (i % 3)) for i in range(10)]

        cfg = MelodyConfig(vocals_mode=True)
        result = resolve_melody(candidates, cfg)

        assert len(result.selected_notes) <= 25, \
            f"Note explosion: {len(result.selected_notes)} notes"

    def test_contour_preserved_with_register_fix(self):
        """Ascending contour should remain ascending after register fix."""
        pitches = [65, 67, 68, 70, 72, 75]
        candidates = []
        for i, p in enumerate(pitches):
            t = i * 0.5
            candidates.append(_cand(t, t + 0.45, p, conf=0.8))
            candidates.append(_cand(t, t + 0.45, p - 12, conf=0.85))  # sub-harmonic

        cfg = MelodyConfig(vocals_mode=True)
        result = resolve_melody(candidates, cfg)

        selected = [n.midi_pitch for n in result.selected_notes]
        if len(selected) >= 3:
            ascending_pairs = sum(1 for i in range(len(selected) - 1)
                                 if selected[i + 1] >= selected[i])
            assert ascending_pairs >= len(selected) // 2, \
                f"Ascending contour broken: {selected}"

    def test_stats_include_register_fields(self):
        """Stats should contain all new register-related fields."""
        candidates = [
            _cand(0.0, 0.5, 68),
            _cand(0.5, 1.0, 70),
        ]
        cfg = MelodyConfig(vocals_mode=True)
        result = resolve_melody(candidates, cfg)

        required_keys = [
            "pitch_floor_midi_used",
            "auto_floor_applied",
            "global_register_corrections",
        ]
        for key in required_keys:
            assert key in result.stats, f"Missing stats key: {key}"
