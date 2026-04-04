"""
Stage D — Key-signature detection and pitch spelling.

Uses the Krumhansl-Schmuckler algorithm (pitch-class histogram correlated
against major / minor profiles) to estimate the most likely key.
"""

import logging
from typing import List, Tuple

import numpy as np

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet.key_detect")

# Krumhansl-Schmuckler profiles  (index 0 = C, 1 = C#, … 11 = B)
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Canonical names for display
KEY_NAMES_MAJOR = [
    "C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"
]
KEY_NAMES_MINOR = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"
]

# Sharps count for each major key (positive = sharps, negative = flats)
KEY_SHARPS = {
    "C": 0,  "G": 1,  "D": 2,  "A": 3,  "E": 4,  "B": 5,  "F#": 6,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6,
}

# Pitch-class → name depending on sharp/flat context
SHARP_SPELLING = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E",  5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
}
FLAT_SPELLING = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E",  5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect_key(
    notes: List[NoteEvent],
    override: str = "auto",
) -> Tuple[str, str, int]:
    """
    Detect or accept a key-signature override.

    Returns
    -------
    (key_name, mode, sharps)
        e.g. ``("G", "major", 1)``  or ``("A", "minor", 0)``
    """
    if override and override != "auto":
        return _parse_key_override(override)

    if not notes:
        logger.warning("No notes for key detection — defaulting to C major")
        return "C", "major", 0

    # --- pitch-class histogram (weighted by duration × confidence) ----------
    histogram = np.zeros(12)
    for n in notes:
        pc = n.midi_pitch % 12
        histogram[pc] += n.duration_s * n.confidence

    total = histogram.sum()
    if total > 0:
        histogram /= total

    # --- correlate against all keys -----------------------------------------
    best_corr = -2.0
    best_key = 0
    best_mode = "major"

    for shift in range(12):
        rotated = np.roll(histogram, -shift)

        corr_maj = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        if corr_maj > best_corr:
            best_corr, best_key, best_mode = corr_maj, shift, "major"

        corr_min = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])
        if corr_min > best_corr:
            best_corr, best_key, best_mode = corr_min, shift, "minor"

    key_name = (KEY_NAMES_MAJOR if best_mode == "major"
                else KEY_NAMES_MINOR)[best_key]

    # sharps for the key (or its relative major)
    if best_mode == "major":
        sharps = KEY_SHARPS.get(key_name, 0)
    else:
        rel_major_pc = (best_key + 3) % 12
        rel_major_name = KEY_NAMES_MAJOR[rel_major_pc]
        sharps = KEY_SHARPS.get(rel_major_name, 0)

    logger.info(
        "Detected key: %s %s  (r=%.3f, sharps=%d)",
        key_name, best_mode, best_corr, sharps,
    )
    return key_name, best_mode, sharps


def spell_pitch(midi_pitch: int, sharps: int) -> str:
    """
    Return a human-readable note name (e.g. ``"C#4"``) respecting
    the key's sharp/flat convention.
    """
    pc = midi_pitch % 12
    octave = (midi_pitch // 12) - 1
    name = SHARP_SPELLING[pc] if sharps >= 0 else FLAT_SPELLING[pc]
    return f"{name}{octave}"


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #

def _parse_key_override(key_str: str) -> Tuple[str, str, int]:
    """Parse ``"G"`` / ``"D minor"`` / ``"Bb major"`` etc."""
    parts = key_str.strip().split()
    key_name = parts[0]
    mode = parts[1].lower() if len(parts) >= 2 else "major"
    if mode not in ("major", "minor"):
        mode = "major"

    if mode == "major":
        sharps = KEY_SHARPS.get(key_name, 0)
    else:
        try:
            idx = KEY_NAMES_MINOR.index(key_name)
        except ValueError:
            idx = 0
        rel_major_name = KEY_NAMES_MAJOR[(idx + 3) % 12]
        sharps = KEY_SHARPS.get(rel_major_name, 0)

    logger.info("Key override: %s %s  (sharps=%d)", key_name, mode, sharps)
    return key_name, mode, sharps
