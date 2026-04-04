"""
Unit tests for the melody-path resolver: candidate normalization,
feature scoring, DP solver, and post-cleanup.

Run with:  pytest tests/test_melody_path.py -v
"""

import math
import pytest

from src.config.melody_config import MelodyConfig, InstrumentRange, INSTRUMENT_RANGES
from src.pipeline.candidate_notes import (
    CandidateNote,
    normalize_candidates,
    compute_clutter_density,
)
from src.pipeline.melody_features import (
    score_note_intrinsic,
    score_transition,
    compute_median_pitch,
    score_gap_entry,
    score_gap_continuation,
)
from src.pipeline.melody_path import (
    resolve_melody,
    _build_lattice,
    _merge_tiny_gaps,
    _octave_smooth,
    _enforce_monophonic,
    _enforce_range,
    _deduplicate_notes,
    GAP_STATE,
)
from src.pipeline.utils import NoteEvent


# --------------------------------------------------------------------------- #
# Candidate note normalization
# --------------------------------------------------------------------------- #

class TestNormalizeCandidates:

    def test_basic_conversion(self):
        raw = [(0.0, 0.5, 60, 0.8), (0.5, 1.0, 62, 0.7)]
        result = normalize_candidates(raw)
        assert len(result) == 2
        assert result[0].pitch_midi == 60
        assert result[1].pitch_midi == 62

    def test_filters_invalid_end_before_start(self):
        raw = [(0.5, 0.3, 60, 0.8)]  # end < start
        result = normalize_candidates(raw)
        assert len(result) == 0

    def test_filters_invalid_pitch(self):
        raw = [(0.0, 0.5, -1, 0.8), (0.0, 0.5, 128, 0.8)]
        result = normalize_candidates(raw)
        assert len(result) == 0

    def test_filters_nan(self):
        raw = [(float('nan'), 0.5, 60, 0.8)]
        result = normalize_candidates(raw)
        assert len(result) == 0

    def test_filters_short_notes(self):
        raw = [(0.0, 0.01, 60, 0.8)]  # 10ms, below 50ms default
        result = normalize_candidates(raw)
        assert len(result) == 0

    def test_filters_low_confidence(self):
        raw = [(0.0, 0.5, 60, 0.05)]  # below 0.15 threshold
        result = normalize_candidates(raw)
        assert len(result) == 0

    def test_deterministic_sort(self):
        raw = [
            (0.5, 1.0, 64, 0.8),
            (0.0, 0.5, 62, 0.7),
            (0.0, 0.3, 60, 0.9),
        ]
        result = normalize_candidates(raw)
        starts = [n.start_s for n in result]
        assert starts == sorted(starts)

    def test_candidate_properties(self):
        raw = [(0.0, 1.0, 72, 0.9)]
        result = normalize_candidates(raw)
        n = result[0]
        assert n.duration_s == pytest.approx(1.0)
        assert n.pitch_class == 0  # C
        assert n.octave == 5      # C5

    def test_overlap_detection(self):
        a = CandidateNote(0.0, 0.5, 60)
        b = CandidateNote(0.3, 0.8, 62)
        c = CandidateNote(0.5, 1.0, 64)
        assert a.overlaps(b)
        assert not a.overlaps(c)  # touching but not overlapping
        assert a.overlap_duration(b) == pytest.approx(0.2)


class TestClutterDensity:

    def test_single_note_zero_density(self):
        notes = [CandidateNote(0.0, 0.5, 60)]
        result = compute_clutter_density(notes, window_ms=200)
        assert result == [0]

    def test_dense_cluster(self):
        notes = [
            CandidateNote(0.0, 0.5, 60 + i, confidence=0.5)
            for i in range(8)
        ]
        result = compute_clutter_density(notes, window_ms=1000)
        assert all(d >= 5 for d in result)


# --------------------------------------------------------------------------- #
# Feature scoring
# --------------------------------------------------------------------------- #

class TestNoteScoring:

    def test_high_confidence_scores_higher(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        a = CandidateNote(0.0, 0.5, 72, confidence=0.9)
        b = CandidateNote(0.0, 0.5, 72, confidence=0.3)
        sa = score_note_intrinsic(a, cfg, inst, 0, 72)
        sb = score_note_intrinsic(b, cfg, inst, 0, 72)
        assert sa > sb

    def test_longer_note_scores_higher(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        a = CandidateNote(0.0, 1.0, 72, confidence=0.7)
        b = CandidateNote(0.0, 0.1, 72, confidence=0.7)
        sa = score_note_intrinsic(a, cfg, inst, 0, 72)
        sb = score_note_intrinsic(b, cfg, inst, 0, 72)
        assert sa > sb

    def test_clutter_penalty_reduces_score(self):
        cfg = MelodyConfig()
        inst = cfg.get_instrument_range()
        note = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        s_clean = score_note_intrinsic(note, cfg, inst, 0, 72)
        s_cluttered = score_note_intrinsic(note, cfg, inst, 10, 72)
        assert s_clean > s_cluttered

    def test_out_of_range_penalised(self):
        cfg = MelodyConfig(instrument="violin")
        inst = cfg.get_instrument_range()
        in_range = CandidateNote(0.0, 0.5, 76, confidence=0.7)  # centre
        out_range = CandidateNote(0.0, 0.5, 30, confidence=0.7)  # way below
        s_in = score_note_intrinsic(in_range, cfg, inst, 0, 72)
        s_out = score_note_intrinsic(out_range, cfg, inst, 0, 72)
        assert s_in > s_out


class TestTransitionScoring:

    def test_stepwise_better_than_leap(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        step = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        leap = CandidateNote(0.5, 1.0, 90, confidence=0.7)
        s_step = score_transition(prev, step, cfg)
        s_leap = score_transition(prev, leap, cfg)
        assert s_step > s_leap

    def test_temporal_continuity_rewarded(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        close = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        far = CandidateNote(3.0, 3.5, 74, confidence=0.7)
        s_close = score_transition(prev, close, cfg)
        s_far = score_transition(prev, far, cfg)
        assert s_close > s_far

    def test_octave_flip_penalised(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 0.5, 72, confidence=0.7)
        octave_up = CandidateNote(0.5, 1.0, 84, confidence=0.7)
        normal_step = CandidateNote(0.5, 1.0, 74, confidence=0.7)
        s_oct = score_transition(prev, octave_up, cfg)
        s_step = score_transition(prev, normal_step, cfg)
        assert s_step > s_oct

    def test_overlap_penalised(self):
        cfg = MelodyConfig()
        prev = CandidateNote(0.0, 1.0, 72, confidence=0.7)
        overlap = CandidateNote(0.3, 1.3, 74, confidence=0.7)
        no_overlap = CandidateNote(1.0, 1.5, 74, confidence=0.7)
        s_overlap = score_transition(prev, overlap, cfg)
        s_clean = score_transition(prev, no_overlap, cfg)
        assert s_clean > s_overlap


class TestGapScoring:

    def test_gap_entry_is_negative(self):
        cfg = MelodyConfig()
        s = score_gap_entry(None, 0.5, cfg)
        assert s < 0

    def test_gap_continuation_increases_cost(self):
        cfg = MelodyConfig()
        s_short = score_gap_continuation(0.1, cfg)
        s_long = score_gap_continuation(3.0, cfg)
        assert s_short > s_long

    def test_very_long_gap_heavily_penalised(self):
        cfg = MelodyConfig()
        s = score_gap_continuation(10.0, cfg)
        assert s < -3.0


class TestMedianPitch:

    def test_simple_median(self):
        notes = [
            CandidateNote(0.0, 1.0, 60),
            CandidateNote(1.0, 2.0, 72),
            CandidateNote(2.0, 3.0, 84),
        ]
        m = compute_median_pitch(notes)
        assert m == 72.0

    def test_empty(self):
        assert compute_median_pitch([]) == 72.0


# --------------------------------------------------------------------------- #
# Lattice construction
# --------------------------------------------------------------------------- #

class TestLatticeBuild:

    def test_basic_lattice(self):
        candidates = [
            CandidateNote(0.0, 0.5, 60),
            CandidateNote(0.3, 0.8, 64),
        ]
        cfg = MelodyConfig()
        lattice = _build_lattice(candidates, cfg)
        assert len(lattice) >= 3  # at least 0.0, 0.3, 0.5, 0.8

    def test_gap_states_present(self):
        candidates = [CandidateNote(0.0, 0.5, 60)]
        cfg = MelodyConfig(allow_gap_state=True)
        lattice = _build_lattice(candidates, cfg)
        assert all(col.has_gap for col in lattice)


# --------------------------------------------------------------------------- #
# Full melody resolution
# --------------------------------------------------------------------------- #

class TestResolveMelody:

    def test_selects_melody_from_candidates(self):
        candidates = [
            CandidateNote(0.0, 0.5, 72, confidence=0.9),
            CandidateNote(0.5, 1.0, 74, confidence=0.8),
            CandidateNote(1.0, 1.5, 76, confidence=0.7),
        ]
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        assert len(result.selected_notes) >= 2

    def test_prefers_melodic_line_over_clutter(self):
        """
        Given a clear melodic line (high confidence, stepwise) and
        a cluster of low-confidence accompaniment notes, the resolver
        should prefer the melody.
        """
        melody = [
            CandidateNote(0.0, 0.5, 72, confidence=0.9),
            CandidateNote(0.5, 1.0, 74, confidence=0.85),
            CandidateNote(1.0, 1.5, 76, confidence=0.8),
        ]
        clutter = [
            CandidateNote(0.0, 0.2, 48, confidence=0.3),
            CandidateNote(0.1, 0.3, 52, confidence=0.25),
            CandidateNote(0.2, 0.4, 55, confidence=0.3),
            CandidateNote(0.5, 0.7, 48, confidence=0.25),
            CandidateNote(0.6, 0.8, 52, confidence=0.3),
        ]
        all_notes = melody + clutter
        cfg = MelodyConfig()
        result = resolve_melody(all_notes, cfg)
        pitches = [n.midi_pitch for n in result.selected_notes]
        assert 72 in pitches
        assert 74 in pitches

    def test_handles_empty_input(self):
        cfg = MelodyConfig()
        result = resolve_melody([], cfg)
        assert result.selected_notes == []

    def test_single_note(self):
        candidates = [CandidateNote(0.0, 1.0, 72, confidence=0.9)]
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        assert len(result.selected_notes) == 1
        assert result.selected_notes[0].midi_pitch == 72

    def test_monophonic_output(self):
        """Result should never have overlapping notes."""
        candidates = [
            CandidateNote(0.0, 0.5, 72, confidence=0.8),
            CandidateNote(0.0, 0.5, 76, confidence=0.7),
            CandidateNote(0.3, 0.8, 74, confidence=0.9),
            CandidateNote(0.5, 1.0, 78, confidence=0.6),
        ]
        cfg = MelodyConfig()
        result = resolve_melody(candidates, cfg)
        notes = result.selected_notes
        for i in range(len(notes) - 1):
            assert notes[i].end_s <= notes[i + 1].start_s + 1e-6

    def test_gap_states_allowed(self):
        """With a large temporal gap, the resolver should insert a gap state."""
        candidates = [
            CandidateNote(0.0, 0.5, 72, confidence=0.9),
            CandidateNote(5.0, 5.5, 74, confidence=0.9),
        ]
        cfg = MelodyConfig(allow_gap_state=True)
        result = resolve_melody(candidates, cfg)
        assert result.stats["gap_states_in_path"] > 0


# --------------------------------------------------------------------------- #
# Post-cleanup helpers
# --------------------------------------------------------------------------- #

class TestMergeTinyGaps:

    def test_merges_same_pitch_tiny_gap(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.55, 1.0, 72),
        ]
        result = _merge_tiny_gaps(notes, gap_ms=60.0)
        assert len(result) == 1
        assert result[0].end_s == pytest.approx(1.0)

    def test_does_not_merge_different_pitch(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.55, 1.0, 74),
        ]
        result = _merge_tiny_gaps(notes, gap_ms=60.0)
        assert len(result) == 2


class TestOctaveSmooth:

    def test_corrects_octave_spike(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 72),  # +12 spike up
            NoteEvent(1.0, 1.5, 60),  # back to same register
        ]
        result = _octave_smooth(notes)
        assert result[1].midi_pitch == 60

    def test_preserves_real_register_change(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 1.0, 67),  # P5 up, not octave
            NoteEvent(1.0, 1.5, 72),
        ]
        result = _octave_smooth(notes)
        assert result[1].midi_pitch == 67


class TestEnforceMonophonic:

    def test_truncates_overlap(self):
        notes = [
            NoteEvent(0.0, 0.8, 72),
            NoteEvent(0.5, 1.0, 74),
        ]
        result = _enforce_monophonic(notes)
        assert result[0].end_s == pytest.approx(0.5)
        assert result[1].start_s == pytest.approx(0.5)


class TestEnforceRange:

    def test_transpose_up(self):
        inst = InstrumentRange(low=55, high=100, center=76)
        notes = [NoteEvent(0.0, 0.5, 40)]
        result = _enforce_range(notes, inst, "transpose")
        assert result[0].midi_pitch >= 55

    def test_transpose_down(self):
        inst = InstrumentRange(low=55, high=100, center=76)
        notes = [NoteEvent(0.0, 0.5, 110)]
        result = _enforce_range(notes, inst, "transpose")
        assert result[0].midi_pitch <= 100


class TestDeduplicateNotes:

    def test_removes_duplicates(self):
        notes = [
            NoteEvent(0.0, 0.5, 72),
            NoteEvent(0.0, 0.5, 72),
        ]
        result = _deduplicate_notes(notes)
        assert len(result) == 1
