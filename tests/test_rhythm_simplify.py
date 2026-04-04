"""
Tests for rhythm simplification.

Covers:
  A. Duration simplification to standard note values
  B. Micro-rest removal
  C. Unusual duration counting
  D. Full simplify_notation pipeline
  E. Tied notes are preserved
  F. Edge cases
"""

import pytest
from src.pipeline.rhythm_simplify import (
    simplify_durations,
    remove_micro_rests,
    count_unusual_durations,
    simplify_notation,
    _nearest_standard,
    STANDARD_DURATIONS,
)
from src.pipeline.utils import QuantizedNote


class TestNearestStandard:
    def test_exact_quarter(self):
        assert _nearest_standard(1.0) == 1.0

    def test_exact_half(self):
        assert _nearest_standard(2.0) == 2.0

    def test_exact_eighth(self):
        assert _nearest_standard(0.5) == 0.5

    def test_exact_sixteenth(self):
        assert _nearest_standard(0.25) == 0.25

    def test_snaps_0_3_to_quarter(self):
        assert _nearest_standard(0.3) == 0.25

    def test_snaps_0_4_to_half_or_quarter(self):
        result = _nearest_standard(0.4)
        assert result in (0.25, 0.5)

    def test_snaps_0_6_to_half_or_dotted_quarter(self):
        result = _nearest_standard(0.6)
        assert result in (0.5, 0.75)

    def test_snaps_0_9_to_quarter(self):
        assert _nearest_standard(0.9) == 1.0

    def test_snaps_1_3_to_dotted_quarter_or_half(self):
        result = _nearest_standard(1.3)
        assert result in (1.0, 1.5)

    def test_dotted_half(self):
        assert _nearest_standard(3.0) == 3.0

    def test_dotted_eighth(self):
        assert _nearest_standard(0.75) == 0.75


class TestSimplifyDurations:
    def test_standard_durations_unchanged(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.5, midi_pitch=74),
        ]
        result = simplify_durations(notes)
        assert result[0].duration_ql == 1.0
        assert result[1].duration_ql == 0.5

    def test_non_standard_gets_snapped(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.333333, midi_pitch=72),
        ]
        result = simplify_durations(notes)
        assert result[0].duration_ql in STANDARD_DURATIONS

    def test_tied_notes_untouched(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.333333, midi_pitch=72,
                          tie_type="start"),
        ]
        result = simplify_durations(notes)
        assert result[0].duration_ql == pytest.approx(0.333333)


class TestRemoveMicroRests:
    def test_removes_small_rest_between_notes(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.25, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=1.25, duration_ql=1.0, midi_pitch=74),
        ]
        result = remove_micro_rests(notes, threshold_ql=0.25)
        assert len(result) == 2
        assert result[0].duration_ql == pytest.approx(1.25)
        assert result[1].midi_pitch == 74

    def test_preserves_large_rest(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=1.0, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=2.0, duration_ql=1.0, midi_pitch=74),
        ]
        result = remove_micro_rests(notes, threshold_ql=0.25)
        assert len(result) == 3

    def test_preserves_rest_at_end(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.25, midi_pitch=0, is_rest=True),
        ]
        result = remove_micro_rests(notes, threshold_ql=0.25)
        assert len(result) == 2

    def test_preserves_rest_at_start(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.25, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=0.25, duration_ql=1.0, midi_pitch=72),
        ]
        result = remove_micro_rests(notes, threshold_ql=0.25)
        assert len(result) == 2


class TestCountUnusualDurations:
    def test_all_standard(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.5, midi_pitch=74),
        ]
        assert count_unusual_durations(notes) == 0

    def test_some_unusual(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.333, midi_pitch=74),
            QuantizedNote(start_ql=1.333, duration_ql=0.666, midi_pitch=76),
        ]
        assert count_unusual_durations(notes) == 2

    def test_tied_notes_excluded(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=0.333, midi_pitch=72,
                          tie_type="start"),
        ]
        assert count_unusual_durations(notes) == 0


class TestSimplifyNotation:
    def test_reduces_unusual_count(self):
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.15, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=1.15, duration_ql=0.85, midi_pitch=74),
            QuantizedNote(start_ql=2.0, duration_ql=0.333, midi_pitch=76),
        ]
        before = count_unusual_durations(notes)
        result = simplify_notation(notes)
        after = count_unusual_durations(result)
        assert after <= before

    def test_empty_input(self):
        assert simplify_notation([]) == []

    def test_preserves_total_note_content(self):
        """Simplification should not create or destroy notes (only rests)."""
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=72),
            QuantizedNote(start_ql=1.0, duration_ql=0.5, midi_pitch=74),
            QuantizedNote(start_ql=1.5, duration_ql=0.5, midi_pitch=76),
        ]
        result = simplify_notation(notes)
        pitches_before = [n.midi_pitch for n in notes if not n.is_rest]
        pitches_after = [n.midi_pitch for n in result if not n.is_rest]
        assert pitches_before == pitches_after
