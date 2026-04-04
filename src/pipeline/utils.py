"""
Shared data structures and utilities for the transcription pipeline.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("mp3_to_sheet")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """Raw note event from pitch detection."""
    start_s: float
    end_s: float
    midi_pitch: int
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class QuantizedNote:
    """Note after quantization to beat grid."""
    start_ql: float        # offset in quarter-note lengths from score start
    duration_ql: float     # duration in quarter-note lengths
    midi_pitch: int        # 0 for rests
    is_rest: bool = False
    tie_type: Optional[str] = None       # 'start', 'stop', 'continue'
    spelled_name: Optional[str] = None   # e.g. "C#4"


@dataclass
class TranscriptionConfig:
    """All user-configurable parameters."""
    # --- Input ---
    input_path: str = ""
    max_duration_s: float = 45.0

    # --- Source mode ---
    # "mix"    — transcribe from the full audio mix (original behaviour)
    # "vocals" — run source separation first, transcribe from vocal stem
    source_mode: str = "mix"
    stems_dir: str = ""  # explicit path to pre-separated stems

    # --- Meter ---
    time_signature: str = "auto"

    # --- Key ---
    key_signature: str = "auto"

    # --- Quantization ---
    allow_triplets: bool = False
    simplify_rhythm: bool = False
    readable_rhythm: bool = True

    # --- Melody extraction ---
    min_note_duration_ms: float = 100.0
    min_confidence: float = 0.3
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3

    # --- Instrument ---
    instrument: str = "violin"

    # --- Instrument range (legacy, used when melody_resolver disabled) ---
    clamp_to_violin_range: bool = True
    violin_low: int = 55
    violin_high: int = 105

    # --- Melody resolver ---
    use_melody_resolver: bool = True  # use new DP melody-path resolver
    pitch_floor_midi: Optional[int] = None  # optional pre-filter for low notes

    # --- Vocal abstraction (vocals mode density reduction) ---
    vocal_abstraction: bool = True  # auto-enabled when source_mode == "vocals"

    # --- Reference-guided correction ---
    reference_path: str = ""  # path to reference_notes.json for pitch correction

    # --- Post-processing ---
    add_phrase_slurs: bool = False

    # --- Output ---
    output_dir: str = "out"
    save_debug_artifacts: bool = True

    # --- Tuning overrides (injected by the tuning loop) ---
    # Dict of MelodyConfig field name → value to override after defaults.
    melody_config_overrides: dict = field(default_factory=dict)
    # Dict of CandidateRepairConfig field name → value to override.
    repair_config_overrides: dict = field(default_factory=dict)

    # --- External tools ---
    musescore_path: str = ""


@dataclass
class TranscriptionResult:
    """Output of the full pipeline."""
    musicxml_path: Optional[str] = None
    pdf_path: Optional[str] = None
    midi_path: Optional[str] = None
    debug_json_path: Optional[str] = None
    tempo_bpm: float = 120.0
    detected_key: str = "C major"
    num_notes: int = 0
    num_measures: int = 0
    source_mode: str = "mix"
    warnings: List[str] = field(default_factory=list)
    melody_stats: dict = field(default_factory=dict)
    candidate_repair_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_hash(path: str) -> str:
    """SHA-256 hash of a file (first 10 MB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            if f.tell() > 10 * 1024 * 1024:
                break
    return h.hexdigest()[:16]


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_debug_json(data: dict, path: str):
    """Persist intermediate pipeline data for debugging."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def note_events_to_dicts(notes: List[NoteEvent]) -> List[dict]:
    """Serialize NoteEvents to plain dicts (for JSON)."""
    return [asdict(n) for n in notes]


def quantized_notes_to_dicts(notes: List[QuantizedNote]) -> List[dict]:
    """Serialize QuantizedNotes to plain dicts (for JSON)."""
    return [asdict(n) for n in notes]


def setup_logging(level=logging.INFO):
    """Configure root logger for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
