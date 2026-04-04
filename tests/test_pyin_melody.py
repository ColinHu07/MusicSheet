"""Tests for the pyin monophonic melody extraction module."""

import numpy as np
import pytest

from src.pipeline.pyin_melody import (
    PyinConfig,
    PyinStats,
    _absorb_short_fragments,
    _clean_pitch_curve,
    _correct_register,
    _interpolate_short_gaps,
    _make_note,
    _segment_notes,
    _stabilize_pitches,
    _cleanup_notes,
)
from src.pipeline.utils import NoteEvent


class TestInterpolateShortGaps:

    def test_no_gaps(self):
        midi = np.array([60.0, 61.0, 62.0])
        result = _interpolate_short_gaps(midi, max_gap_frames=3)
        np.testing.assert_array_equal(result, midi)

    def test_short_gap_interpolated(self):
        midi = np.array([60.0, np.nan, 62.0])
        result = _interpolate_short_gaps(midi, max_gap_frames=2)
        assert not np.isnan(result[1])
        assert result[1] == pytest.approx(61.0, abs=0.5)

    def test_long_gap_preserved(self):
        midi = np.array([60.0, np.nan, np.nan, np.nan, np.nan, 62.0])
        result = _interpolate_short_gaps(midi, max_gap_frames=2)
        assert np.isnan(result[2])

    def test_gap_at_start(self):
        midi = np.array([np.nan, np.nan, 60.0, 61.0])
        result = _interpolate_short_gaps(midi, max_gap_frames=3)
        assert result[0] == pytest.approx(60.0)

    def test_gap_at_end(self):
        midi = np.array([60.0, 61.0, np.nan])
        result = _interpolate_short_gaps(midi, max_gap_frames=3)
        assert result[2] == pytest.approx(61.0)


class TestMakeNote:

    def test_basic_note(self):
        pitches = [60.0, 60.5, 61.0, 60.0]
        note = _make_note(10, 14, pitches, 0.02)
        assert note.midi_pitch == 60
        assert note.start_s == pytest.approx(0.2)
        assert note.end_s == pytest.approx(0.28)

    def test_single_pitch(self):
        note = _make_note(0, 5, [72.0], 0.023)
        assert note.midi_pitch == 72


class TestSegmentNotes:

    def test_constant_pitch_one_note(self):
        midi = np.full(100, 65.0)
        cfg = PyinConfig()
        notes = _segment_notes(midi, 0.023, cfg)
        assert len(notes) == 1
        assert notes[0].midi_pitch == 65

    def test_pitch_jump_creates_boundary(self):
        midi = np.concatenate([np.full(50, 60.0), np.full(50, 67.0)])
        cfg = PyinConfig(pitch_change_threshold_st=0.8)
        notes = _segment_notes(midi, 0.023, cfg)
        assert len(notes) == 2
        assert notes[0].midi_pitch == 60
        assert notes[1].midi_pitch == 67

    def test_silence_gap_creates_boundary(self):
        midi = np.concatenate([
            np.full(50, 60.0),
            np.full(10, np.nan),
            np.full(50, 60.0),
        ])
        cfg = PyinConfig(silence_gap_s=0.05)
        notes = _segment_notes(midi, 0.023, cfg)
        assert len(notes) == 2

    def test_empty_curve(self):
        notes = _segment_notes(np.array([]), 0.023, PyinConfig())
        assert notes == []


class TestCorrectRegister:

    def test_low_notes_shifted_up(self):
        notes = [
            NoteEvent(0.0, 0.5, 48, 1.0),  # very low
            NoteEvent(0.5, 1.0, 65, 1.0),
            NoteEvent(1.0, 1.5, 67, 1.0),
            NoteEvent(1.5, 2.0, 70, 1.0),
            NoteEvent(2.0, 2.5, 68, 1.0),
        ]
        result = _correct_register(notes)
        assert result[0].midi_pitch == 60  # shifted +12

    def test_notes_in_range_unchanged(self):
        notes = [
            NoteEvent(0.0, 0.5, 65, 1.0),
            NoteEvent(0.5, 1.0, 67, 1.0),
            NoteEvent(1.0, 1.5, 70, 1.0),
        ]
        result = _correct_register(notes)
        assert [n.midi_pitch for n in result] == [65, 67, 70]

    def test_too_few_notes_unchanged(self):
        notes = [NoteEvent(0.0, 0.5, 48, 1.0)]
        result = _correct_register(notes)
        assert result[0].midi_pitch == 48


class TestCleanupNotes:

    def test_micro_notes_removed(self):
        notes = [
            NoteEvent(0.0, 0.5, 60, 1.0),
            NoteEvent(0.5, 0.52, 62, 1.0),  # 20ms — too short
            NoteEvent(0.52, 1.0, 60, 1.0),
        ]
        cfg = PyinConfig(min_note_duration_s=0.08)
        result = _cleanup_notes(notes, cfg)
        assert all(n.duration_s >= 0.08 for n in result)

    def test_same_pitch_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 60, 1.0),
            NoteEvent(0.55, 1.0, 60, 1.0),  # 50ms gap, same pitch
        ]
        cfg = PyinConfig(merge_gap_s=0.10)
        result = _cleanup_notes(notes, cfg)
        assert len(result) == 1
        assert result[0].end_s == 1.0

    def test_different_pitch_not_merged(self):
        notes = [
            NoteEvent(0.0, 0.5, 60, 1.0),
            NoteEvent(0.55, 1.0, 65, 1.0),  # different pitch
        ]
        cfg = PyinConfig(merge_gap_s=0.10)
        result = _cleanup_notes(notes, cfg)
        assert len(result) == 2


class TestPyinStats:

    def test_to_dict(self):
        s = PyinStats(
            total_frames=1000,
            voiced_frames=750,
            voiced_pct=75.0,
            notes_raw=100,
            notes_after_cleanup=80,
            notes_absorbed=20,
            pitch_range_midi=(55, 80),
            avg_duration_s=0.45,
            median_duration_s=0.40,
            short_notes_pct=5.0,
            adjacent_within_1st_pct=8.0,
        )
        d = s.to_dict()
        assert d["total_frames"] == 1000
        assert d["voiced_pct"] == 75.0
        assert d["pitch_range_midi"] == [55, 80]
        assert d["notes_absorbed"] == 20
        assert d["median_duration_s"] == 0.40
        assert d["short_notes_pct"] == 5.0
        assert d["adjacent_within_1st_pct"] == 8.0


class TestAbsorbShortFragments:

    def test_short_fragment_absorbed(self):
        notes = [
            NoteEvent(0.0, 0.6, 60, 1.0),
            NoteEvent(0.6, 0.75, 61, 1.0),  # 150ms, short
            NoteEvent(0.75, 1.5, 60, 1.0),
        ]
        result = _absorb_short_fragments(notes, max_short_s=0.20,
                                         min_neighbor_ratio=1.5)
        assert len(result) == 2

    def test_short_fragment_between_short_neighbors_kept(self):
        notes = [
            NoteEvent(0.0, 0.15, 60, 1.0),
            NoteEvent(0.15, 0.30, 62, 1.0),
            NoteEvent(0.30, 0.45, 64, 1.0),
        ]
        result = _absorb_short_fragments(notes, max_short_s=0.20,
                                         min_neighbor_ratio=2.0)
        assert len(result) == 3

    def test_long_note_not_absorbed(self):
        notes = [
            NoteEvent(0.0, 0.5, 60, 1.0),
            NoteEvent(0.5, 1.0, 65, 1.0),  # 500ms — not short
            NoteEvent(1.0, 1.5, 60, 1.0),
        ]
        result = _absorb_short_fragments(notes, max_short_s=0.20,
                                         min_neighbor_ratio=1.5)
        assert len(result) == 3

    def test_fewer_than_3_notes_unchanged(self):
        notes = [NoteEvent(0.0, 0.5, 60, 1.0), NoteEvent(0.5, 1.0, 65, 1.0)]
        result = _absorb_short_fragments(notes)
        assert len(result) == 2


class TestStabilizePitches:

    def test_sandwich_outlier_removed(self):
        """A note creating large jumps both ways with agreeing neighbors."""
        notes = [
            NoteEvent(0.0, 0.5, 68, 1.0),
            NoteEvent(0.5, 1.0, 55, 1.0),  # 13 semitones from both neighbors
            NoteEvent(1.0, 1.5, 68, 1.0),
        ]
        result = _stabilize_pitches(notes, sandwich_jump_st=7)
        assert len(result) == 2
        assert all(n.midi_pitch == 68 for n in result)

    def test_non_sandwich_kept(self):
        """A note with only one large jump is NOT removed."""
        notes = [
            NoteEvent(0.0, 0.5, 68, 1.0),
            NoteEvent(0.5, 1.0, 63, 1.0),  # 5 semitones from prev, 2 from next
            NoteEvent(1.0, 1.5, 65, 1.0),
        ]
        result = _stabilize_pitches(notes, sandwich_jump_st=7)
        assert len(result) == 3

    def test_fewer_than_3_unchanged(self):
        notes = [NoteEvent(0.0, 0.5, 68, 1.0), NoteEvent(0.5, 1.0, 55, 1.0)]
        result = _stabilize_pitches(notes)
        assert len(result) == 2


class TestMakeNoteMode:

    def test_mode_beats_median(self):
        """Mode should pick the most frequent pitch, not the median."""
        # 8 frames at 68, 2 frames at 60 — mode is 68, median might be ~66
        pitches = [68.0] * 8 + [60.0] * 2
        note = _make_note(0, 10, pitches, 0.023)
        assert note.midi_pitch == 68

    def test_uniform_pitches(self):
        pitches = [65.0, 65.1, 64.9, 65.2, 64.8]
        note = _make_note(0, 5, pitches, 0.023)
        assert note.midi_pitch == 65


class TestPersistenceSegmentation:

    def test_single_frame_spike_does_not_split(self):
        """A one-frame pitch spike should NOT create a boundary."""
        midi = np.concatenate([
            np.full(30, 65.0),
            np.array([72.0]),  # single-frame spike
            np.full(30, 65.0),
        ])
        cfg = PyinConfig(pitch_change_threshold_st=1.5,
                         pitch_persistence_frames=3)
        notes = _segment_notes(midi, 0.023, cfg)
        assert len(notes) == 1

    def test_persistent_jump_does_split(self):
        """A pitch jump lasting many frames should create a boundary."""
        midi = np.concatenate([
            np.full(30, 60.0),
            np.full(30, 72.0),  # 12 semitones, sustained
        ])
        cfg = PyinConfig(pitch_change_threshold_st=1.5,
                         pitch_persistence_frames=3)
        notes = _segment_notes(midi, 0.023, cfg)
        assert len(notes) == 2


class TestPyinConfig:

    def test_defaults(self):
        cfg = PyinConfig()
        assert cfg.fmin_note == "C3"
        assert cfg.fmax_note == "C6"
        assert cfg.pitch_change_threshold_st == 1.5
        assert cfg.pitch_persistence_frames == 3
        assert cfg.min_note_duration_s == 0.10
        assert cfg.merge_gap_s == 0.15
        assert cfg.merge_pitch_tolerance_st == 1
