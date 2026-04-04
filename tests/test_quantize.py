"""
Unit tests for the quantization module.

Run with:  pytest tests/test_quantize.py -v
"""

import numpy as np
import pytest

from src.pipeline.beat_grid import BeatGrid
from src.pipeline.quantize import quantize_notes, split_at_barlines, _snap, _fill_rests
from src.pipeline.utils import NoteEvent, QuantizedNote


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_grid(bpm=120.0, ts_num=4, ts_den=4):
    """Helper: create a simple BeatGrid at constant tempo."""
    beat_dur = 60.0 / bpm
    beat_times = np.array([i * beat_dur for i in range(32)])
    return BeatGrid(
        tempo_bpm=bpm,
        beat_times=beat_times,
        time_sig_num=ts_num,
        time_sig_den=ts_den,
        first_beat_s=0.0,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestSnapToGrid:
    """Test the grid-snapping helper."""

    def test_snap_quarter(self):
        # 0.24 should snap to 0.25
        assert _snap(0.24, 0.25, None) == pytest.approx(0.25)

    def test_snap_eighth(self):
        assert _snap(0.74, 0.5, None) == pytest.approx(0.5)

    def test_snap_exact(self):
        assert _snap(1.0, 0.25, None) == pytest.approx(1.0)

    def test_snap_zero(self):
        assert _snap(0.0, 0.25, None) == pytest.approx(0.0)

    def test_snap_triplet_closer(self):
        # 0.33 is closer to 1/3 ≈ 0.333 than to 0.25
        result = _snap(0.33, 0.25, 1 / 3)
        assert result == pytest.approx(1 / 3, abs=0.01)


class TestQuantizeNotes:
    """Integration tests for the full quantize_notes function."""

    def test_single_note(self):
        grid = _make_grid(120)
        # At 120 BPM: 1 beat = 0.5s,  1 quarter note = 1 ql
        notes = [NoteEvent(start_s=0.0, end_s=0.5, midi_pitch=69)]
        result = quantize_notes(notes, grid)

        pitched = [n for n in result if not n.is_rest]
        assert len(pitched) == 1
        assert pitched[0].midi_pitch == 69
        assert pitched[0].duration_ql == pytest.approx(1.0, abs=0.01)

    def test_two_notes_with_gap(self):
        grid = _make_grid(120)
        notes = [
            NoteEvent(start_s=0.0, end_s=0.25, midi_pitch=60),   # eighth note
            NoteEvent(start_s=0.5, end_s=0.75, midi_pitch=62),   # eighth note after gap
        ]
        result = quantize_notes(notes, grid)

        pitched = [n for n in result if not n.is_rest]
        rests = [n for n in result if n.is_rest]
        assert len(pitched) == 2
        assert len(rests) >= 1  # at least one rest in the gap

    def test_simplify_snaps_to_eighth(self):
        grid = _make_grid(120)
        # Note that would be a sixteenth at full resolution
        notes = [NoteEvent(start_s=0.0, end_s=0.12, midi_pitch=64)]
        result = quantize_notes(notes, grid, simplify=True)

        pitched = [n for n in result if not n.is_rest]
        assert len(pitched) == 1
        # In simplified mode, minimum duration is 0.5 ql (eighth note)
        assert pitched[0].duration_ql == pytest.approx(0.5, abs=0.01)

    def test_empty_input(self):
        grid = _make_grid(120)
        assert quantize_notes([], grid) == []

    def test_overlap_resolution(self):
        grid = _make_grid(120)
        # Two overlapping notes — second should truncate first
        notes = [
            NoteEvent(start_s=0.0, end_s=1.0, midi_pitch=60),
            NoteEvent(start_s=0.25, end_s=0.75, midi_pitch=62),
        ]
        result = quantize_notes(notes, grid)
        pitched = [n for n in result if not n.is_rest]

        # Check no overlaps
        for i in range(len(pitched) - 1):
            end = pitched[i].start_ql + pitched[i].duration_ql
            assert end <= pitched[i + 1].start_ql + 1e-6


class TestSplitAtBarlines:
    """Test barline splitting and tie assignment."""

    def test_note_within_bar(self):
        grid = _make_grid(120, ts_num=4, ts_den=4)  # measure = 4 ql
        notes = [QuantizedNote(start_ql=0.0, duration_ql=2.0, midi_pitch=60)]
        result = split_at_barlines(notes, grid)
        assert len(result) == 1
        assert result[0].tie_type is None

    def test_note_crossing_barline(self):
        grid = _make_grid(120, ts_num=4, ts_den=4)  # measure = 4 ql
        # Note starts at beat 3 (ql=3) and lasts 3 ql → crosses into measure 2
        notes = [QuantizedNote(start_ql=3.0, duration_ql=3.0, midi_pitch=60)]
        result = split_at_barlines(notes, grid)
        assert len(result) == 2
        assert result[0].tie_type == "start"
        assert result[1].tie_type == "stop"
        assert result[0].duration_ql == pytest.approx(1.0)
        assert result[1].duration_ql == pytest.approx(2.0)

    def test_rest_not_split(self):
        grid = _make_grid(120, ts_num=4, ts_den=4)
        notes = [QuantizedNote(start_ql=3.0, duration_ql=3.0, midi_pitch=0, is_rest=True)]
        result = split_at_barlines(notes, grid)
        # Rests are NOT split at barlines (handled by music21 later)
        assert len(result) == 1


class TestFillRests:
    """Test rest insertion."""

    def test_leading_rest(self):
        notes = [QuantizedNote(start_ql=2.0, duration_ql=1.0, midi_pitch=60)]
        result = _fill_rests(notes, 0.25)
        assert result[0].is_rest
        assert result[0].start_ql == 0.0
        assert result[0].duration_ql == pytest.approx(2.0, abs=0.1)

    def test_no_leading_rest_when_starts_at_zero(self):
        notes = [QuantizedNote(start_ql=0.0, duration_ql=1.0, midi_pitch=60)]
        result = _fill_rests(notes, 0.25)
        assert not result[0].is_rest
