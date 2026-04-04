"""
Tests for the improved melody scoring — verifying that the tuned weights
and new features produce the right behaviours for vocal-led pop melody
extraction.

These tests encode the key musical assumptions:
  - vocal melody sits in upper register
  - longer notes are more melodic
  - accompaniment clutter should be rejected
  - octave flips should be heavily punished
  - phrase continuity matters more than note-by-note confidence
  - burst of short notes = accompaniment, not melody
  - post-cleanup should consolidate, not fragment

Run with:  pytest tests/test_melody_scoring_v2.py -v
"""

import pytest

from src.config.melody_config import MelodyConfig, InstrumentRange, INSTRUMENT_RANGES
from src.pipeline.candidate_notes import CandidateNote, compute_clutter_density
from src.pipeline.melody_features import (
    score_note_intrinsic,
    score_transition,
    compute_median_pitch,
    compute_upper_quartile_pitch,
)
from src.pipeline.melody_path import (
    resolve_melody,
    _merge_tiny_gaps,
    _merge_nearby_pitch_gaps,
    _consolidate_repeated,
    _octave_smooth,
    _enforce_monophonic,
)
from src.pipeline.utils import NoteEvent


# =========================================================================== #
# Register bias
# =========================================================================== #

class TestRegisterBias:
    """Verify that the register bias is strong enough to separate melody
    from accompaniment in a typical vocal-led pop arrangement."""

    def test_vocal_register_prefers_upper_range(self):
        """A note at the upper quartile should score much higher than one
        an octave below, even with equal confidence/duration."""
        cfg = MelodyConfig(register_bias_mode="vocal")
        inst = cfg.get_instrument_range()
        upper = CandidateNote(0.0, 0.5, 78, confidence=0.6)
        lower = CandidateNote(0.0, 0.5, 54, confidence=0.6)
        s_upper = score_note_intrinsic(upper, cfg, inst, 0, 66, 78)
        s_lower = score_note_intrinsic(lower, cfg, inst, 0, 66, 78)
        assert s_upper > s_lower
        assert s_upper - s_lower > 2.0, \
            "Register advantage should be decisive (>2 points)"

    def test_vocal_mode_penalises_bass_notes(self):
        """Notes far below the vocal register should get negative register bonus."""
        cfg = MelodyConfig(register_bias_mode="vocal")
        inst = cfg.get_instrument_range()
        bass = CandidateNote(0.0, 0.5, 40, confidence=0.7)
        s = score_note_intrinsic(bass, cfg, inst, 0, 60, 72)
        vocal = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        s_v = score_note_intrinsic(vocal, cfg, inst, 0, 60, 72)
        assert s_v > s

    def test_above_median_still_works(self):
        """Backward-compatible mode: above_median gives some register bias."""
        cfg = MelodyConfig(register_bias_mode="above_median")
        inst = cfg.get_instrument_range()
        high = CandidateNote(0.0, 0.5, 80, confidence=0.6)
        low = CandidateNote(0.0, 0.5, 60, confidence=0.6)
        s_high = score_note_intrinsic(high, cfg, inst, 0, 70)
        s_low = score_note_intrinsic(low, cfg, inst, 0, 70)
        assert s_high > s_low


# =========================================================================== #
# Duration preference
# =========================================================================== #

class TestDurationPreference:
    """Longer notes should win decisively over short blips at equal confidence."""

    def test_half_second_beats_50ms(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        long = CandidateNote(0.0, 0.5, 72, confidence=0.6)
        short = CandidateNote(0.0, 0.05, 72, confidence=0.6)
        s_long = score_note_intrinsic(long, cfg, inst, 0, 72)
        s_short = score_note_intrinsic(short, cfg, inst, 0, 72)
        assert s_long > s_short
        assert s_long - s_short > 3.0, \
            "Duration advantage should be very strong for melody selection"

    def test_short_note_penalty_applied(self):
        """Notes below the short_note_threshold should be penalised."""
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        micro = CandidateNote(0.0, 0.06, 72, confidence=0.7)  # 60ms < 150ms threshold
        normal = CandidateNote(0.0, 0.3, 72, confidence=0.7)
        s_micro = score_note_intrinsic(micro, cfg, inst, 0, 72)
        s_normal = score_note_intrinsic(normal, cfg, inst, 0, 72)
        assert s_normal > s_micro


# =========================================================================== #
# Clutter penalty
# =========================================================================== #

class TestClutterPenalty:
    """Dense note clusters should be heavily penalised to reject
    accompaniment texture."""

    def test_high_clutter_severely_penalised(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        note = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        s_clean = score_note_intrinsic(note, cfg, inst, clutter_density=0, median_pitch=72)
        s_heavy = score_note_intrinsic(note, cfg, inst, clutter_density=12, median_pitch=72)
        diff = s_clean - s_heavy
        assert diff > 5.0, \
            f"Heavy clutter penalty should be >5 points, got {diff:.2f}"

    def test_moderate_clutter_penalised(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        note = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        s_clean = score_note_intrinsic(note, cfg, inst, 0, 72)
        s_mod = score_note_intrinsic(note, cfg, inst, 6, 72)
        assert s_clean > s_mod


# =========================================================================== #
# Transition scoring
# =========================================================================== #

class TestTransitionScoringV2:
    """Verify that transitions strongly prefer smooth melodic motion
    and heavily penalise accompaniment-like jumping."""

    def test_stepwise_greatly_preferred(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        step = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        leap = CandidateNote(0.5, 1.0, 84, confidence=0.7)  # octave
        s_step = score_transition(prev, step, cfg)
        s_leap = score_transition(prev, leap, cfg)
        advantage = s_step - s_leap
        assert advantage > 10.0, \
            f"Stepwise vs octave leap should differ by >10 pts, got {advantage:.2f}"

    def test_octave_flip_devastatingly_penalised(self):
        """An octave flip (+12) at close range should be the most expensive
        transition possible."""
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        flip = CandidateNote(0.5, 1.0, 84, confidence=0.7)  # +12
        s = score_transition(prev, flip, cfg)
        assert s < -8.0, \
            f"Octave flip penalty should produce score < -8, got {s:.2f}"

    def test_burst_of_short_notes_penalised(self):
        """Two consecutive very short notes should get burst penalty."""
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.05, 72, confidence=0.7)  # 50ms
        nxt = CandidateNote(0.05, 0.1, 74, confidence=0.7)   # 50ms
        s_burst = score_transition(prev, nxt, cfg)
        long_prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        long_nxt = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        s_normal = score_transition(long_prev, long_nxt, cfg)
        assert s_normal > s_burst

    def test_phrase_continuity_strong(self):
        """Smooth melodic continuation should receive a large bonus."""
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        cont = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        s = score_transition(prev, cont, cfg)
        assert s > 5.0, \
            f"Smooth phrase continuation should score > 5, got {s:.2f}"


# =========================================================================== #
# Upper quartile pitch
# =========================================================================== #

class TestUpperQuartilePitch:

    def test_upper_quartile_above_median(self):
        notes = [
            CandidateNote(0.0, 1.0, 48),
            CandidateNote(1.0, 2.0, 60),
            CandidateNote(2.0, 3.0, 72),
            CandidateNote(3.0, 4.0, 84),
        ]
        med = compute_median_pitch(notes)
        uq = compute_upper_quartile_pitch(notes)
        assert uq >= med

    def test_empty(self):
        assert compute_upper_quartile_pitch([]) == 78.0


# =========================================================================== #
# Post-cleanup improvements
# =========================================================================== #

class TestMergeNearbyPitchGaps:

    def test_merges_semitone_gap(self):
        """Notes differing by 1 semitone with a tiny gap: extend previous."""
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.56, 1.0, 73),
        ]
        result = _merge_nearby_pitch_gaps(notes, gap_ms=80.0, semitone_tolerance=1)
        assert result[0].end_s == pytest.approx(0.56)

    def test_does_not_merge_large_pitch_diff(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.56, 1.0, 72),
        ]
        result = _merge_nearby_pitch_gaps(notes, gap_ms=80.0, semitone_tolerance=1)
        assert result[0].end_s == pytest.approx(0.5)


class TestConsolidateRepeated:

    def test_consolidates_same_pitch(self):
        """Three repeated C4 notes with small gaps should become one."""
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.35, 0.6, 72),
            NoteEvent(0.65, 1.0, 72),
        ]
        result = _consolidate_repeated(notes, max_gap_ms=150)
        assert len(result) == 1
        assert result[0].end_s == pytest.approx(1.0)

    def test_does_not_consolidate_different_pitch(self):
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(0.35, 0.6, 74),
        ]
        result = _consolidate_repeated(notes, max_gap_ms=150)
        assert len(result) == 2

    def test_does_not_consolidate_large_gap(self):
        notes = [
            NoteEvent(0.0, 0.3, 72),
            NoteEvent(1.0, 1.3, 72),
        ]
        result = _consolidate_repeated(notes, max_gap_ms=150)
        assert len(result) == 2


# =========================================================================== #
# Full resolver — vocal melody vs. accompaniment scenario
# =========================================================================== #

class TestFullResolverVocalScenario:
    """
    Simulate a realistic vocal-led pop scenario:
    - Vocal melody at MIDI 72-78, high confidence, sustained notes
    - Bass/chord accompaniment at MIDI 48-60, moderate confidence, shorter notes
    - Some drum-like noise at various pitches, low confidence, very short

    The resolver should consistently pick the vocal line.
    """

    def _make_scenario(self):
        vocal = [
            CandidateNote(0.0, 0.8,  74, confidence=0.7),
            CandidateNote(0.8, 1.3,  76, confidence=0.65),
            CandidateNote(1.3, 2.0,  74, confidence=0.7),
            CandidateNote(2.0, 2.8,  72, confidence=0.75),
            CandidateNote(2.8, 3.5,  74, confidence=0.65),
        ]
        accompaniment = [
            CandidateNote(0.0, 0.4,  48, confidence=0.6),
            CandidateNote(0.0, 0.4,  52, confidence=0.55),
            CandidateNote(0.0, 0.4,  55, confidence=0.5),
            CandidateNote(0.4, 0.8,  48, confidence=0.6),
            CandidateNote(0.4, 0.8,  52, confidence=0.55),
            CandidateNote(0.8, 1.2,  48, confidence=0.6),
            CandidateNote(0.8, 1.2,  52, confidence=0.55),
            CandidateNote(0.8, 1.2,  55, confidence=0.5),
            CandidateNote(1.2, 1.6,  48, confidence=0.6),
            CandidateNote(1.6, 2.0,  48, confidence=0.6),
            CandidateNote(1.6, 2.0,  52, confidence=0.55),
            CandidateNote(2.0, 2.4,  48, confidence=0.6),
            CandidateNote(2.0, 2.4,  52, confidence=0.55),
            CandidateNote(2.4, 2.8,  48, confidence=0.6),
            CandidateNote(2.8, 3.2,  48, confidence=0.6),
            CandidateNote(2.8, 3.2,  52, confidence=0.55),
        ]
        noise = [
            CandidateNote(0.1, 0.13, 90, confidence=0.2),
            CandidateNote(0.5, 0.53, 35, confidence=0.25),
            CandidateNote(1.5, 1.53, 95, confidence=0.2),
            CandidateNote(2.5, 2.53, 38, confidence=0.2),
        ]
        return vocal + accompaniment + noise

    def test_selects_vocal_melody(self):
        all_notes = self._make_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        vocal_pitches_found = sum(1 for p in pitches if 70 <= p <= 80)
        accomp_pitches_found = sum(1 for p in pitches if p < 55)
        assert vocal_pitches_found >= 3, \
            f"Should find >= 3 vocal notes, got {vocal_pitches_found}: {pitches}"
        assert accomp_pitches_found == 0, \
            f"Should find 0 accompaniment notes, got {accomp_pitches_found}: {pitches}"

    def test_output_is_not_fragmented(self):
        """Average note duration should be > 0.3s (phrase-like, not choppy)."""
        all_notes = self._make_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        notes = result.selected_notes
        if notes:
            avg_dur = sum(n.duration_s for n in notes) / len(notes)
            assert avg_dur > 0.3, \
                f"Average note duration should be > 0.3s, got {avg_dur:.3f}s"

    def test_output_has_smooth_contour(self):
        """Average melodic interval should be <= 5 semitones."""
        all_notes = self._make_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        notes = result.selected_notes
        if len(notes) > 1:
            intervals = [abs(notes[i+1].midi_pitch - notes[i].midi_pitch)
                         for i in range(len(notes) - 1)]
            avg_int = sum(intervals) / len(intervals)
            assert avg_int <= 5, \
                f"Average interval should be <= 5 semitones, got {avg_int:.1f}"

    def test_stats_include_new_fields(self):
        all_notes = self._make_scenario()
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        assert "upper_quartile_pitch" in result.stats
        assert "avg_note_duration_s" in result.stats
        assert "avg_melodic_interval" in result.stats
        assert "mean_clutter_of_selected" in result.stats
