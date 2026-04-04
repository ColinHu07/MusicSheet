"""
Reference-guided note correction.

Uses a reference note sequence (ground truth) to correct pitch values
in the generated note sequence while preserving timing from audio.

The alignment uses Needleman-Wunsch dynamic programming on absolute time,
which avoids tempo/measure-number mapping issues between reference and
generated scores (they may use different BPM interpretations).
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .utils import NoteEvent

logger = logging.getLogger("mp3_to_sheet")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RefNote:
    """Reference note with absolute time computed from measure/position/tempo."""
    time_s: float
    end_s: float
    midi_pitch: int
    measure: int
    position_in_measure: float
    duration_quarters: float


@dataclass
class AlignedPair:
    """One entry in the alignment between reference and generated sequences."""
    ref_idx: Optional[int]
    gen_idx: Optional[int]
    ref_pitch: Optional[int]
    gen_pitch: Optional[int]
    time_distance_s: float
    pitch_distance_st: int
    confidence: str          # HIGH, MEDIUM, LOW
    action: str              # match, skip_ref, skip_gen


@dataclass
class CorrectionStats:
    total_generated: int = 0
    total_reference: int = 0
    matched_pairs: int = 0
    unmatched_gen: int = 0
    unmatched_ref: int = 0
    pitch_corrections: int = 0
    already_correct: int = 0
    corrections_by_size: Dict[int, int] = field(default_factory=dict)
    avg_correction_st: float = 0.0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    merges_applied: int = 0
    notes_deleted: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_reference_notes(ref_path: str) -> Tuple[List[RefNote], dict]:
    """Load reference notes and convert to absolute time.

    Returns (notes, metadata) where metadata contains tempo, key, etc.
    """
    with open(ref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tempo_bpm = data["tempo_bpm"]
    beat_dur_s = 60.0 / tempo_bpm
    # Assumes 4/4 time (matches reference format)
    measure_dur_s = 4.0 * beat_dur_s

    notes: List[RefNote] = []
    for n in data["notes"]:
        if n.get("is_rest", False):
            continue
        time_s = (n["measure"] - 1) * measure_dur_s + n["position_in_measure"] * beat_dur_s
        end_s = time_s + n["duration_quarters"] * beat_dur_s
        notes.append(RefNote(
            time_s=time_s,
            end_s=end_s,
            midi_pitch=n["pitch_midi"],
            measure=n["measure"],
            position_in_measure=n["position_in_measure"],
            duration_quarters=n["duration_quarters"],
        ))

    meta = {
        "title": data.get("title", ""),
        "tempo_bpm": tempo_bpm,
        "key": data.get("key", ""),
        "total_notes": len(notes),
        "time_span_s": notes[-1].end_s if notes else 0,
    }
    return notes, meta


# ---------------------------------------------------------------------------
# Alignment cost
# ---------------------------------------------------------------------------

def _match_cost(
    ref: RefNote,
    gen: NoteEvent,
    max_time_s: float = 1.5,
) -> float:
    """Cost of matching ref to gen.  Lower = better."""
    time_dist = abs(ref.time_s - gen.start_s)
    pitch_dist = abs(ref.midi_pitch - gen.midi_pitch)

    if time_dist > max_time_s:
        return 100.0   # prohibitively expensive

    # Time component dominates — we want notes at similar times to match
    time_cost = (time_dist / max_time_s) ** 1.2

    # Pitch component is secondary — we *expect* pitch errors
    pitch_cost = min(pitch_dist / 12.0, 1.0) * 0.25

    return time_cost + pitch_cost


def _match_confidence(ref: RefNote, gen: NoteEvent) -> str:
    """Classify alignment confidence for a matched pair."""
    time_dist = abs(ref.time_s - gen.start_s)
    pitch_dist = abs(ref.midi_pitch - gen.midi_pitch)

    if time_dist < 0.25:
        conf = "HIGH"
    elif time_dist < 0.6:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    # Large pitch distance downgrades confidence
    if pitch_dist > 7 and conf == "HIGH":
        conf = "MEDIUM"
    if pitch_dist > 7 and conf == "MEDIUM":
        conf = "LOW"

    return conf


# ---------------------------------------------------------------------------
# DP alignment
# ---------------------------------------------------------------------------

def align_notes(
    ref_notes: List[RefNote],
    gen_notes: List[NoteEvent],
    skip_ref_penalty: float = 0.55,
    skip_gen_penalty: float = 0.45,
    max_time_s: float = 1.5,
) -> List[AlignedPair]:
    """Global alignment of reference and generated note sequences.

    Uses Needleman-Wunsch DP.  Both input sequences must be sorted by time.
    """
    n_r = len(ref_notes)
    n_g = len(gen_notes)

    INF = float("inf")
    dp = [[INF] * (n_g + 1) for _ in range(n_r + 1)]
    bt = [[None] * (n_g + 1) for _ in range(n_r + 1)]

    dp[0][0] = 0.0
    for i in range(1, n_r + 1):
        dp[i][0] = dp[i - 1][0] + skip_ref_penalty
        bt[i][0] = "skip_ref"
    for j in range(1, n_g + 1):
        dp[0][j] = dp[0][j - 1] + skip_gen_penalty
        bt[0][j] = "skip_gen"

    for i in range(1, n_r + 1):
        for j in range(1, n_g + 1):
            c_match = dp[i - 1][j - 1] + _match_cost(
                ref_notes[i - 1], gen_notes[j - 1], max_time_s,
            )
            c_skip_r = dp[i - 1][j] + skip_ref_penalty
            c_skip_g = dp[i][j - 1] + skip_gen_penalty

            best = min(c_match, c_skip_r, c_skip_g)
            dp[i][j] = best
            if best == c_match:
                bt[i][j] = "match"
            elif best == c_skip_r:
                bt[i][j] = "skip_ref"
            else:
                bt[i][j] = "skip_gen"

    # Trace back
    pairs: List[AlignedPair] = []
    i, j = n_r, n_g
    while i > 0 or j > 0:
        action = bt[i][j]
        if action == "match":
            r, g = ref_notes[i - 1], gen_notes[j - 1]
            pairs.append(AlignedPair(
                ref_idx=i - 1,
                gen_idx=j - 1,
                ref_pitch=r.midi_pitch,
                gen_pitch=g.midi_pitch,
                time_distance_s=abs(r.time_s - g.start_s),
                pitch_distance_st=abs(r.midi_pitch - g.midi_pitch),
                confidence=_match_confidence(r, g),
                action="match",
            ))
            i -= 1
            j -= 1
        elif action == "skip_ref":
            r = ref_notes[i - 1]
            pairs.append(AlignedPair(
                ref_idx=i - 1, gen_idx=None,
                ref_pitch=r.midi_pitch, gen_pitch=None,
                time_distance_s=0, pitch_distance_st=0,
                confidence="LOW", action="skip_ref",
            ))
            i -= 1
        elif action == "skip_gen":
            g = gen_notes[j - 1]
            pairs.append(AlignedPair(
                ref_idx=None, gen_idx=j - 1,
                ref_pitch=None, gen_pitch=g.midi_pitch,
                time_distance_s=0, pitch_distance_st=0,
                confidence="LOW", action="skip_gen",
            ))
            j -= 1
        else:
            break

    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Pitch correction
# ---------------------------------------------------------------------------

def apply_corrections(
    gen_notes: List[NoteEvent],
    alignment: List[AlignedPair],
    ref_notes: List[RefNote],
    correct_high: bool = True,
    correct_medium: bool = True,
    correct_low: bool = False,
    max_correction_st: int = 12,
) -> Tuple[List[NoteEvent], CorrectionStats]:
    """Replace generated pitches with reference pitches where alignment is confident.

    Returns (corrected_notes, stats).
    """
    corrected = [NoteEvent(n.start_s, n.end_s, n.midi_pitch, n.confidence)
                 for n in gen_notes]

    corrections: List[int] = []
    already_correct = 0
    matched = 0
    high = med = low = 0

    for pair in alignment:
        if pair.action != "match":
            continue
        matched += 1

        if pair.confidence == "HIGH":
            high += 1
        elif pair.confidence == "MEDIUM":
            med += 1
        else:
            low += 1

        if pair.gen_idx is None or pair.ref_idx is None:
            continue

        should_correct = (
            (pair.confidence == "HIGH" and correct_high)
            or (pair.confidence == "MEDIUM" and correct_medium)
            or (pair.confidence == "LOW" and correct_low)
        )
        if not should_correct:
            continue

        if pair.pitch_distance_st == 0:
            already_correct += 1
            continue
        if pair.pitch_distance_st > max_correction_st:
            continue

        old = corrected[pair.gen_idx]
        corrected[pair.gen_idx] = NoteEvent(
            start_s=old.start_s,
            end_s=old.end_s,
            midi_pitch=ref_notes[pair.ref_idx].midi_pitch,
            confidence=old.confidence,
        )
        corrections.append(pair.pitch_distance_st)

    size_counts = dict(sorted(Counter(corrections).items()))

    stats = CorrectionStats(
        total_generated=len(gen_notes),
        total_reference=len(ref_notes),
        matched_pairs=matched,
        unmatched_gen=sum(1 for p in alignment if p.action == "skip_gen"),
        unmatched_ref=sum(1 for p in alignment if p.action == "skip_ref"),
        pitch_corrections=len(corrections),
        already_correct=already_correct,
        corrections_by_size=size_counts,
        avg_correction_st=(sum(corrections) / len(corrections)) if corrections else 0.0,
        high_confidence=high,
        medium_confidence=med,
        low_confidence=low,
    )

    logger.info(
        "Reference correction: %d/%d matched, %d corrected (avg %.1f st), "
        "%d already correct, %d unmatched-gen, %d unmatched-ref",
        matched, len(gen_notes), len(corrections), stats.avg_correction_st,
        already_correct, stats.unmatched_gen, stats.unmatched_ref,
    )

    return corrected, stats


# ---------------------------------------------------------------------------
# Optional merge repair
# ---------------------------------------------------------------------------

def merge_aligned_fragments(
    notes: List[NoteEvent],
    alignment: List[AlignedPair],
    ref_notes: List[RefNote],
) -> Tuple[List[NoteEvent], int]:
    """Merge adjacent generated notes that map to the same reference note.

    This handles over-segmentation: when the generator splits one held note
    into multiple fragments, and the alignment reveals they all correspond
    to a single reference note, merge them.

    Returns (merged_notes, merge_count).
    """
    # Build mapping: gen_idx -> ref_idx for matched pairs
    gen_to_ref: Dict[int, int] = {}
    for pair in alignment:
        if pair.action == "match" and pair.gen_idx is not None and pair.ref_idx is not None:
            gen_to_ref[pair.gen_idx] = pair.ref_idx

    merged: List[NoteEvent] = []
    merge_count = 0
    i = 0
    n = len(notes)

    while i < n:
        current = notes[i]
        ref_i = gen_to_ref.get(i)

        # Try to absorb subsequent notes that point to the same ref note
        j = i + 1
        while j < n and ref_i is not None:
            ref_j = gen_to_ref.get(j)
            if ref_j is not None and ref_j == ref_i:
                j += 1
            else:
                break

        if j > i + 1:
            # Merge notes[i:j] into one
            merged_note = NoteEvent(
                start_s=notes[i].start_s,
                end_s=notes[j - 1].end_s,
                midi_pitch=ref_notes[ref_i].midi_pitch,
                confidence=current.confidence,
            )
            merged.append(merged_note)
            merge_count += j - i - 1
        else:
            merged.append(current)

        i = j

    if merge_count:
        logger.info("Merge repair: absorbed %d fragments into held notes", merge_count)

    return merged, merge_count


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    """Full result of reference-guided correction."""
    corrected_notes: List[NoteEvent]
    stats: CorrectionStats
    alignment: List[AlignedPair]
    ref_notes: List[RefNote]
    gen_notes_original: List[NoteEvent]
    ref_meta: dict = field(default_factory=dict)


def correct_with_reference(
    gen_notes: List[NoteEvent],
    reference_path: str,
    correct_low_confidence: bool = False,
    enable_merge_repair: bool = True,
    max_correction_st: int = 12,
) -> CorrectionResult:
    """Full reference-guided correction pipeline.

    1. Load reference notes
    2. Align generated notes to reference using DP
    3. Apply pitch corrections
    4. Optionally merge over-segmented fragments

    Returns CorrectionResult with corrected notes, stats, and alignment.
    """
    ref_notes, meta = load_reference_notes(reference_path)
    logger.info(
        "Reference loaded: %d notes, %.1fs span, tempo=%d BPM",
        meta["total_notes"], meta["time_span_s"], meta["tempo_bpm"],
    )

    alignment = align_notes(ref_notes, gen_notes)

    corrected, stats = apply_corrections(
        gen_notes, alignment, ref_notes,
        correct_high=True,
        correct_medium=True,
        correct_low=correct_low_confidence,
        max_correction_st=max_correction_st,
    )

    if enable_merge_repair:
        corrected, n_merges = merge_aligned_fragments(
            corrected, alignment, ref_notes,
        )
        stats.merges_applied = n_merges

    return CorrectionResult(
        corrected_notes=corrected,
        stats=stats,
        alignment=alignment,
        ref_notes=ref_notes,
        gen_notes_original=list(gen_notes),
        ref_meta=meta,
    )


# ---------------------------------------------------------------------------
# Debug / reporting
# ---------------------------------------------------------------------------

def save_alignment_report(
    alignment: List[AlignedPair],
    ref_notes: List[RefNote],
    gen_notes: List[NoteEvent],
    stats: CorrectionStats,
    output_dir: str,
):
    """Save detailed alignment and correction artifacts."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Alignment detail
    alignment_data = [asdict(p) for p in alignment]
    with open(os.path.join(output_dir, "alignment.json"), "w", encoding="utf-8") as f:
        json.dump(alignment_data, f, indent=2)

    # Stats
    with open(os.path.join(output_dir, "reference_guided_corrections.json"), "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Markdown report
    md = _render_report_md(alignment, ref_notes, gen_notes, stats)
    with open(os.path.join(output_dir, "reference_guided_report.md"), "w", encoding="utf-8") as f:
        f.write(md)


NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def _render_report_md(
    alignment: List[AlignedPair],
    ref_notes: List[RefNote],
    gen_notes: List[NoteEvent],
    stats: CorrectionStats,
) -> str:
    lines = [
        "# Reference-Guided Correction Report",
        "",
        "## Summary",
        "",
        f"- Reference notes: {stats.total_reference}",
        f"- Generated notes: {stats.total_generated}",
        f"- Matched pairs: {stats.matched_pairs}",
        f"  - HIGH confidence: {stats.high_confidence}",
        f"  - MEDIUM confidence: {stats.medium_confidence}",
        f"  - LOW confidence: {stats.low_confidence}",
        f"- Unmatched generated: {stats.unmatched_gen}",
        f"- Unmatched reference: {stats.unmatched_ref}",
        f"- Pitch corrections applied: {stats.pitch_corrections}",
        f"- Already correct: {stats.already_correct}",
        f"- Average correction size: {stats.avg_correction_st:.1f} semitones",
        f"- Merges applied: {stats.merges_applied}",
        "",
        "## Corrections by Size (semitones)",
        "",
    ]

    if stats.corrections_by_size:
        lines.append("| Semitones | Count |")
        lines.append("|-----------|-------|")
        for st, count in sorted(stats.corrections_by_size.items()):
            lines.append(f"| {st} | {count} |")
    else:
        lines.append("No corrections applied.")
    lines.append("")

    # Alignment detail (first 50 matched pairs)
    lines.append("## Alignment Detail (matched pairs)")
    lines.append("")
    lines.append("| # | Ref | Gen | Ref Pitch | Gen Pitch | Corrected | Time Dist | Confidence |")
    lines.append("|---|-----|-----|-----------|-----------|-----------|-----------|------------|")

    row = 0
    for p in alignment:
        if p.action != "match":
            continue
        row += 1
        if row > 80:
            lines.append(f"| ... | ({len([x for x in alignment if x.action=='match']) - 80} more) | | | | | |")
            break
        r = ref_notes[p.ref_idx] if p.ref_idx is not None else None
        ref_label = f"m{r.measure}:{r.position_in_measure:.1f}" if r else "-"
        ref_p = _midi_name(p.ref_pitch) if p.ref_pitch is not None else "-"
        gen_p = _midi_name(p.gen_pitch) if p.gen_pitch is not None else "-"
        corrected = "yes" if p.pitch_distance_st > 0 and p.confidence in ("HIGH", "MEDIUM") else "no"
        lines.append(
            f"| {row} | {ref_label} | {p.gen_idx} | {ref_p} | {gen_p} "
            f"| {corrected} | {p.time_distance_s:.3f}s | {p.confidence} |"
        )

    lines.append("")
    return "\n".join(lines)
