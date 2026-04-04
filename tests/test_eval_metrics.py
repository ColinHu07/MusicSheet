"""
Tests for the evaluation metrics module.
"""

import pytest

from eval.metrics import (
    NoteForEval,
    compare_against_reference,
    compute_heuristic_metrics,
    format_report,
    notes_from_dicts,
)


class TestNotesFromDicts:

    def test_basic(self):
        dicts = [
            {"start_s": 0.0, "end_s": 0.5, "pitch_midi": 60, "confidence": 0.9},
        ]
        notes = notes_from_dicts(dicts)
        assert len(notes) == 1
        assert notes[0].pitch_midi == 60

    def test_alternate_key_names(self):
        dicts = [{"start": 0.0, "end": 0.5, "midi_pitch": 72}]
        notes = notes_from_dicts(dicts)
        assert notes[0].pitch_midi == 72


class TestHeuristicMetrics:

    def test_empty_notes(self):
        report = compute_heuristic_metrics([], "empty")
        assert report.note_count == 0

    def test_single_note(self):
        notes = [NoteForEval(0.0, 1.0, 72)]
        report = compute_heuristic_metrics(notes, "single")
        assert report.note_count == 1
        assert report.monophonic

    def test_melodic_sequence(self):
        notes = [
            NoteForEval(0.0, 0.5, 72),
            NoteForEval(0.5, 1.0, 74),
            NoteForEval(1.0, 1.5, 76),
        ]
        report = compute_heuristic_metrics(notes, "melody")
        assert report.note_count == 3
        assert report.melodic_continuity > 0.5
        assert report.monophonic

    def test_octave_jump_detected(self):
        notes = [
            NoteForEval(0.0, 0.5, 60),
            NoteForEval(0.5, 1.0, 72),
            NoteForEval(1.0, 1.5, 60),
        ]
        report = compute_heuristic_metrics(notes, "octave")
        assert report.octave_stability < 1.0

    def test_fragmentation(self):
        notes = [
            NoteForEval(0.0, 0.5, 60),
            NoteForEval(2.0, 2.5, 72),
        ]
        report = compute_heuristic_metrics(notes, "fragmented")
        assert report.fragmentation > 0.3


class TestReferenceComparison:

    def test_perfect_match(self):
        pred = [NoteForEval(0.0, 0.5, 60), NoteForEval(0.5, 1.0, 62)]
        ref = [NoteForEval(0.0, 0.5, 60), NoteForEval(0.5, 1.0, 62)]
        result = compare_against_reference(pred, ref)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_match(self):
        pred = [NoteForEval(0.0, 0.5, 60)]
        ref = [NoteForEval(5.0, 5.5, 90)]
        result = compare_against_reference(pred, ref)
        assert result["precision"] == 0.0

    def test_partial_match(self):
        pred = [NoteForEval(0.0, 0.5, 60), NoteForEval(0.5, 1.0, 62)]
        ref = [NoteForEval(0.0, 0.5, 60)]
        result = compare_against_reference(pred, ref)
        assert result["recall"] == 1.0
        assert result["precision"] == 0.5

    def test_onset_tolerance(self):
        pred = [NoteForEval(0.05, 0.5, 60)]
        ref = [NoteForEval(0.0, 0.5, 60)]
        result = compare_against_reference(pred, ref, onset_tolerance_s=0.1)
        assert result["f1"] == 1.0


class TestFormatReport:

    def test_produces_string(self):
        notes = [NoteForEval(0.0, 1.0, 72)]
        report = compute_heuristic_metrics(notes, "test")
        text = format_report(report)
        assert "test" in text
        assert "Notes" in text
