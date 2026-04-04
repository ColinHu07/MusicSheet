"""
Stage F — MusicXML export using music21.

Builds a single-part Violin score from quantized notes and writes
MusicXML.  Handles ties, rests, key/time signatures, and basic beaming.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from .beat_grid import BeatGrid
from .key_detect import spell_pitch
from .utils import QuantizedNote

logger = logging.getLogger("mp3_to_sheet.musicxml_export")


def export_musicxml(
    notes: List[QuantizedNote],
    key_name: str,
    mode: str,
    sharps: int,
    grid: BeatGrid,
    output_path: str,
    add_slurs: bool = False,
    instrument_name: str = "violin",
) -> str:
    """
    Build a music21 Score and write MusicXML.

    Parameters
    ----------
    notes : list[QuantizedNote]
        Quantized notes + rests (from quantize stage).
    key_name, mode : str
        Key information  (e.g. ``"G"``, ``"major"``).
    sharps : int
        Number of sharps (negative = flats).
    grid : BeatGrid
        Tempo / time-sig information.
    output_path : str
        Destination ``.musicxml`` file.
    add_slurs : bool
        If *True*, add heuristic phrase slurs.

    Returns
    -------
    str — the written file path.
    """
    from music21 import (
        clef,
        instrument,
        key as m21key,
        metadata,
        meter,
        note as m21note,
        stream,
        tempo as m21tempo,
        tie as m21tie,
    )

    logger.info("Building music21 score …")

    # --- Score shell --------------------------------------------------------
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = Path(output_path).stem
    score.metadata.composer = "MP3→Sheet (auto-transcribed)"

    part = stream.Part()
    inst_map = {
        "violin": (instrument.Violin, clef.TrebleClef),
        "flute":  (instrument.Flute,  clef.TrebleClef),
        "viola":  (instrument.Viola,  clef.AltoClef),
        "cello":  (instrument.Violoncello, clef.BassClef),
    }
    inst_cls, clef_cls = inst_map.get(
        instrument_name.lower(), (instrument.Violin, clef.TrebleClef)
    )
    part.partName = inst_cls().instrumentName or instrument_name.title()
    part.insert(0, inst_cls())
    part.insert(0, clef_cls())

    # time signature
    ts = meter.TimeSignature(f"{grid.time_sig_num}/{grid.time_sig_den}")
    part.insert(0, ts)

    # key signature
    ks = m21key.Key(key_name, mode)
    part.insert(0, ks)

    # tempo mark
    mm = m21tempo.MetronomeMark(number=grid.tempo_bpm)
    part.insert(0, mm)

    # --- Add notes / rests --------------------------------------------------
    for qn in notes:
        if qn.is_rest:
            r = m21note.Rest()
            r.quarterLength = qn.duration_ql
            part.insert(qn.start_ql, r)
        else:
            # Spell pitch in context of the key
            spelled = spell_pitch(qn.midi_pitch, sharps)
            n = m21note.Note(spelled)
            n.quarterLength = qn.duration_ql

            # ties
            if qn.tie_type == "start":
                n.tie = m21tie.Tie("start")
            elif qn.tie_type == "stop":
                n.tie = m21tie.Tie("stop")
            elif qn.tie_type == "continue":
                n.tie = m21tie.Tie("continue")

            part.insert(qn.start_ql, n)

    # --- Let music21 tidy up ------------------------------------------------
    logger.info("Running makeNotation (measures, beams, ties) …")
    part.makeNotation(inPlace=True)

    # Optional phrase slurs
    if add_slurs:
        _add_phrase_slurs(part)

    score.insert(0, part)

    # --- Write ---------------------------------------------------------------
    out = str(Path(output_path).resolve())
    score.write("musicxml", fp=out)
    logger.info("MusicXML written: %s", out)
    return out


# --------------------------------------------------------------------------- #
# Optional slur heuristic
# --------------------------------------------------------------------------- #

def _add_phrase_slurs(part) -> None:
    """
    Conservative heuristic: add a slur over consecutive note runs
    (no rests between them) of length ≥ 4 notes.
    """
    from music21 import spanner, note as m21note

    run: list = []
    for el in part.recurse().notesAndRests:
        if isinstance(el, m21note.Rest):
            if len(run) >= 4:
                sl = spanner.Slur(run[0], run[-1])
                part.insert(0, sl)
            run = []
        elif isinstance(el, m21note.Note):
            run.append(el)

    # flush final run
    if len(run) >= 4:
        from music21 import spanner
        sl = spanner.Slur(run[0], run[-1])
        part.insert(0, sl)
