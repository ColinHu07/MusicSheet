"""
Evaluation metrics for melody transcription quality.

Provides metrics for comparing transcription outputs against references
or between pipeline stages.  Works with note-event lists (dicts or NoteEvents).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("mp3_to_sheet.eval")


@dataclass
class NoteForEval:
    """Minimal note representation for evaluation."""
    start_s: float
    end_s: float
    pitch_midi: int
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EvalReport:
    """Structured evaluation result."""
    name: str = ""
    note_count: int = 0
    total_duration_s: float = 0.0
    avg_note_duration_s: float = 0.0
    pitch_range_semitones: int = 0
    pitch_min: int = 0
    pitch_max: int = 0
    octave_stability: float = 0.0
    melodic_continuity: float = 0.0
    note_density_per_second: float = 0.0
    fragmentation: float = 0.0
    monophonic: bool = True
    pitch_accuracy: Optional[float] = None
    onset_accuracy: Optional[float] = None
    details: Dict[str, object] = field(default_factory=dict)


def notes_from_dicts(dicts: List[dict]) -> List[NoteForEval]:
    """Convert list of dicts to NoteForEval objects."""
    result = []
    for d in dicts:
        start = d.get("start_s", d.get("start", 0.0))
        end = d.get("end_s", d.get("end", 0.0))
        pitch = d.get("pitch_midi", d.get("midi_pitch", 0))
        conf = d.get("confidence", 1.0)
        result.append(NoteForEval(start_s=start, end_s=end, pitch_midi=pitch, confidence=conf))
    return result


def compute_heuristic_metrics(notes: List[NoteForEval], name: str = "") -> EvalReport:
    """
    Compute heuristic quality metrics for a note sequence.
    These don't require a ground-truth reference.
    """
    report = EvalReport(name=name)
    if not notes:
        return report

    report.note_count = len(notes)
    durations = [n.duration_s for n in notes]
    pitches = [n.pitch_midi for n in notes]

    report.total_duration_s = sum(durations)
    report.avg_note_duration_s = report.total_duration_s / len(notes)
    report.pitch_min = min(pitches)
    report.pitch_max = max(pitches)
    report.pitch_range_semitones = report.pitch_max - report.pitch_min

    time_span = max(n.end_s for n in notes) - min(n.start_s for n in notes)
    if time_span > 0:
        report.note_density_per_second = len(notes) / time_span
    else:
        report.note_density_per_second = 0.0

    # Octave stability: fraction of consecutive intervals that are NOT octave jumps
    octave_jumps = 0
    for i in range(1, len(pitches)):
        if abs(pitches[i] - pitches[i - 1]) in (12, 24):
            octave_jumps += 1
    if len(pitches) > 1:
        report.octave_stability = 1.0 - octave_jumps / (len(pitches) - 1)
    else:
        report.octave_stability = 1.0

    # Melodic continuity: fraction of consecutive note pairs that are
    # temporally adjacent (gap < 0.5s) and intervallically small (< octave)
    continuous_pairs = 0
    total_pairs = max(1, len(notes) - 1)
    for i in range(1, len(notes)):
        gap = notes[i].start_s - notes[i - 1].end_s
        interval = abs(notes[i].pitch_midi - notes[i - 1].pitch_midi)
        if gap < 0.5 and interval < 12:
            continuous_pairs += 1
    report.melodic_continuity = continuous_pairs / total_pairs if total_pairs > 0 else 1.0

    # Fragmentation: fraction of timeline covered by gaps (not notes)
    sorted_notes = sorted(notes, key=lambda n: n.start_s)
    gap_time = 0.0
    for i in range(1, len(sorted_notes)):
        gap = sorted_notes[i].start_s - sorted_notes[i - 1].end_s
        if gap > 0:
            gap_time += gap
    report.fragmentation = gap_time / time_span if time_span > 0 else 0.0

    # Monophonic check
    report.monophonic = _is_monophonic(sorted_notes)

    return report


def compare_against_reference(
    predicted: List[NoteForEval],
    reference: List[NoteForEval],
    onset_tolerance_s: float = 0.1,
    pitch_tolerance: int = 0,
) -> Dict[str, float]:
    """
    Compare predicted notes against a reference melody.

    Returns precision, recall, F1 for pitch+onset matching.
    """
    matched = 0
    ref_matched = set()

    for p in predicted:
        for j, r in enumerate(reference):
            if j in ref_matched:
                continue
            onset_ok = abs(p.start_s - r.start_s) <= onset_tolerance_s
            pitch_ok = abs(p.pitch_midi - r.pitch_midi) <= pitch_tolerance
            if onset_ok and pitch_ok:
                matched += 1
                ref_matched.add(j)
                break

    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "matched": matched,
        "predicted_count": len(predicted),
        "reference_count": len(reference),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _is_monophonic(sorted_notes: List[NoteForEval]) -> bool:
    for i in range(1, len(sorted_notes)):
        if sorted_notes[i].start_s < sorted_notes[i - 1].end_s - 1e-6:
            return False
    return True


def format_report(report: EvalReport) -> str:
    """Format an EvalReport as human-readable text."""
    lines = [
        f"=== {report.name or 'Evaluation Report'} ===",
        f"  Notes          : {report.note_count}",
        f"  Total duration : {report.total_duration_s:.2f} s",
        f"  Avg note dur   : {report.avg_note_duration_s:.3f} s",
        f"  Pitch range    : {report.pitch_min}–{report.pitch_max} ({report.pitch_range_semitones} st)",
        f"  Density        : {report.note_density_per_second:.1f} notes/s",
        f"  Octave stab.   : {report.octave_stability:.2%}",
        f"  Mel. continuity: {report.melodic_continuity:.2%}",
        f"  Fragmentation  : {report.fragmentation:.2%}",
        f"  Monophonic     : {report.monophonic}",
    ]
    if report.pitch_accuracy is not None:
        lines.append(f"  Pitch accuracy : {report.pitch_accuracy:.2%}")
    if report.onset_accuracy is not None:
        lines.append(f"  Onset accuracy : {report.onset_accuracy:.2%}")
    return "\n".join(lines)
