"""
Round-trip tests: generate MusicXML and verify it is valid / loadable.

Run with:  pytest tests/test_musicxml_roundtrip.py -v
"""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.pipeline.beat_grid import BeatGrid
from src.pipeline.musicxml_export import export_musicxml
from src.pipeline.quantize import split_at_barlines
from src.pipeline.utils import QuantizedNote


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_grid(bpm=120.0, ts_num=4, ts_den=4):
    beat_times = np.array([i * 60.0 / bpm for i in range(32)])
    return BeatGrid(
        tempo_bpm=bpm,
        beat_times=beat_times,
        time_sig_num=ts_num,
        time_sig_den=ts_den,
        first_beat_s=0.0,
    )


def _basic_notes():
    """A simple 2-bar melody in C major for testing."""
    return [
        # Measure 1: C4 quarter, D4 quarter, E4 half
        QuantizedNote(start_ql=0.0,  duration_ql=1.0, midi_pitch=60),
        QuantizedNote(start_ql=1.0,  duration_ql=1.0, midi_pitch=62),
        QuantizedNote(start_ql=2.0,  duration_ql=2.0, midi_pitch=64),
        # Measure 2: F4 dotted quarter, G4 eighth, A4 half
        QuantizedNote(start_ql=4.0,  duration_ql=1.5, midi_pitch=65),
        QuantizedNote(start_ql=5.5,  duration_ql=0.5, midi_pitch=67),
        QuantizedNote(start_ql=6.0,  duration_ql=2.0, midi_pitch=69),
    ]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

class TestMusicXMLExport:

    def test_file_created(self, tmp_path):
        """Export should create a .musicxml file."""
        out = str(tmp_path / "test.musicxml")
        grid = _make_grid()
        notes = _basic_notes()
        result = export_musicxml(
            notes=notes,
            key_name="C",
            mode="major",
            sharps=0,
            grid=grid,
            output_path=out,
        )
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 100

    def test_valid_xml(self, tmp_path):
        """Output must be well-formed XML."""
        out = str(tmp_path / "test.musicxml")
        grid = _make_grid()
        export_musicxml(
            notes=_basic_notes(),
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        tree = ET.parse(out)
        root = tree.getroot()
        # Should have at least one <part> element
        # (namespace varies, so check local name)
        tag_names = [el.tag.split("}")[-1] if "}" in el.tag else el.tag
                     for el in root.iter()]
        assert "part" in tag_names

    def test_has_notes(self, tmp_path):
        """Output should contain <note> elements."""
        out = str(tmp_path / "test.musicxml")
        grid = _make_grid()
        export_musicxml(
            notes=_basic_notes(),
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<note" in content

    def test_has_measures(self, tmp_path):
        """Output should have at least 2 measures."""
        out = str(tmp_path / "test.musicxml")
        grid = _make_grid()
        export_musicxml(
            notes=_basic_notes(),
            key_name="G", mode="major", sharps=1,
            grid=grid, output_path=out,
        )
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.count("<measure ") >= 2

    def test_tied_note_across_barline(self, tmp_path):
        """A note crossing a barline should produce a tie."""
        grid = _make_grid(ts_num=4, ts_den=4)
        # Note starts at beat 3 of measure 1, lasts 3 beats (crosses barline)
        notes = [
            QuantizedNote(start_ql=0.0,  duration_ql=1.0, midi_pitch=60),
            QuantizedNote(start_ql=3.0,  duration_ql=3.0, midi_pitch=64, tie_type="start"),
        ]
        notes = split_at_barlines(notes, grid)

        out = str(tmp_path / "tie_test.musicxml")
        export_musicxml(
            notes=notes,
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        # music21 should produce <tie> or <tied> elements
        assert "<tie " in content or "<tied " in content

    def test_rest_included(self, tmp_path):
        """Rests should appear in output."""
        grid = _make_grid()
        notes = [
            QuantizedNote(start_ql=0.0, duration_ql=2.0, midi_pitch=0, is_rest=True),
            QuantizedNote(start_ql=2.0, duration_ql=2.0, midi_pitch=60),
        ]
        out = str(tmp_path / "rest_test.musicxml")
        export_musicxml(
            notes=notes,
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<rest" in content

    def test_key_signature_sharp(self, tmp_path):
        """D major (2 sharps) should have correct key signature."""
        grid = _make_grid()
        out = str(tmp_path / "key_test.musicxml")
        export_musicxml(
            notes=_basic_notes(),
            key_name="D", mode="major", sharps=2,
            grid=grid, output_path=out,
        )
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
        # Should contain <fifths>2</fifths>
        assert "<fifths>2</fifths>" in content

    def test_empty_notes(self, tmp_path):
        """Exporting with no notes should still produce valid XML."""
        grid = _make_grid()
        out = str(tmp_path / "empty.musicxml")
        export_musicxml(
            notes=[],
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        assert os.path.isfile(out)
        # Should be parseable
        ET.parse(out)

    def test_music21_loads_output(self, tmp_path):
        """music21 should be able to parse the generated file."""
        out = str(tmp_path / "m21_test.musicxml")
        grid = _make_grid()
        export_musicxml(
            notes=_basic_notes(),
            key_name="C", mode="major", sharps=0,
            grid=grid, output_path=out,
        )
        from music21 import converter
        score = converter.parse(out)
        parts = list(score.parts)
        assert len(parts) >= 1
