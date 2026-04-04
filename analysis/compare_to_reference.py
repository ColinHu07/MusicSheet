"""
Reference vs Generated comparison tool.

Compares a reference score against generated pipeline output to identify
systematic errors, pitch mismatches, density differences, and contour
divergence.

Designed for reuse with different songs and reference sources.
"""

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .reference_score_parser import ReferenceScore, Confidence
from .generated_score_parser import GeneratedScore, parse_debug_json


# ── Comparison result types ────────────────────────────────────────────

@dataclass
class PitchMismatch:
    measure: int
    ref_pitches: List[int]
    gen_pitches: List[int]
    ref_median: float
    gen_median: float
    delta_semitones: float
    likely_octave_error: bool
    severity: str  # "minor" (1-2 st), "moderate" (3-5 st), "severe" (6+ st), "octave" (12)


@dataclass
class DensityMismatch:
    measure: int
    ref_count: int
    gen_count: int
    delta: int
    issue: str  # "too_sparse", "too_dense", "ok"


@dataclass
class ContourSegment:
    start_measure: int
    end_measure: int
    ref_direction: str  # "up", "down", "flat", "mixed"
    gen_direction: str
    agrees: bool


@dataclass
class ErrorInstance:
    category: str
    measure: int
    description: str
    ref_context: str
    gen_context: str
    likely_pipeline_stage: str
    severity: str


@dataclass
class ComparisonReport:
    title: str
    ref_summary: Dict
    gen_summary: Dict
    global_pitch_offset: float
    pitch_mismatches: List[PitchMismatch] = field(default_factory=list)
    density_mismatches: List[DensityMismatch] = field(default_factory=list)
    contour_segments: List[ContourSegment] = field(default_factory=list)
    errors: List[ErrorInstance] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "ref_summary": self.ref_summary,
            "gen_summary": self.gen_summary,
            "global_pitch_offset": self.global_pitch_offset,
            "statistics": self.statistics,
            "pitch_mismatches": [asdict(m) for m in self.pitch_mismatches],
            "density_mismatches": [asdict(m) for m in self.density_mismatches],
            "contour_segments": [asdict(s) for s in self.contour_segments],
            "errors": [asdict(e) for e in self.errors],
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_markdown(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._render_markdown())

    def _render_markdown(self) -> str:
        lines = []
        lines.append(f"# Reference vs Generated Comparison: {self.title}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Reference | Generated | Delta |")
        lines.append("|--------|-----------|-----------|-------|")
        rs, gs = self.ref_summary, self.gen_summary
        lines.append(f"| Key | {rs.get('key','-')} | {gs.get('key','-')} | {'MATCH' if rs.get('key')==gs.get('key') else 'MISMATCH'} |")
        lines.append(f"| Time Signature | {rs.get('time_sig','-')} | {gs.get('time_sig','-')} | {'MATCH' if rs.get('time_sig')==gs.get('time_sig') else 'MISMATCH'} |")
        lines.append(f"| Tempo | {rs.get('tempo','-')} | {gs.get('tempo','-')} | {gs.get('tempo',0)-rs.get('tempo',0):+.1f} |")
        lines.append(f"| Note Count (compared range) | {rs.get('note_count','-')} | {gs.get('note_count','-')} | {gs.get('note_count',0)-rs.get('note_count',0):+d} |")
        lines.append(f"| Pitch Range | {rs.get('pitch_range','-')} | {gs.get('pitch_range','-')} | |")
        lines.append(f"| Median Pitch | {rs.get('median_pitch','-')} | {gs.get('median_pitch','-')} | {self.global_pitch_offset:+.1f} st |")
        lines.append(f"| Notes/Measure (avg) | {rs.get('avg_density','?'):.1f} | {gs.get('avg_density','?'):.1f} | {gs.get('avg_density',0)-rs.get('avg_density',0):+.1f} |")
        lines.append("")

        lines.append("## Critical Finding: Global Pitch Offset")
        lines.append("")
        offset = self.global_pitch_offset
        if abs(offset) >= 10:
            lines.append(f"**SEVERE: The generated output is {offset:+.1f} semitones from the reference "
                         f"({abs(offset)/12:.1f} octaves).**")
            lines.append("")
            lines.append("This indicates a systematic octave selection error in the melody path resolver.")
        elif abs(offset) >= 5:
            lines.append(f"**MODERATE: Global pitch offset of {offset:+.1f} semitones.**")
        else:
            lines.append(f"Pitch offset is {offset:+.1f} semitones (within acceptable range).")
        lines.append("")

        # Pitch mismatch summary
        lines.append("## Pitch Mismatches by Measure")
        lines.append("")
        if self.pitch_mismatches:
            octave_errors = [m for m in self.pitch_mismatches if m.likely_octave_error]
            severe = [m for m in self.pitch_mismatches if m.severity == "severe" and not m.likely_octave_error]
            moderate = [m for m in self.pitch_mismatches if m.severity == "moderate"]
            minor = [m for m in self.pitch_mismatches if m.severity == "minor"]

            lines.append(f"- **Octave errors**: {len(octave_errors)} measures")
            lines.append(f"- **Severe (6+ st)**: {len(severe)} measures")
            lines.append(f"- **Moderate (3-5 st)**: {len(moderate)} measures")
            lines.append(f"- **Minor (1-2 st)**: {len(minor)} measures")
            lines.append("")
            lines.append("| Measure | Ref Median | Gen Median | Delta | Type |")
            lines.append("|---------|-----------|-----------|-------|------|")
            for m in self.pitch_mismatches[:30]:
                lines.append(f"| m{m.measure} | {m.ref_median:.0f} | {m.gen_median:.0f} | {m.delta_semitones:+.1f} | {m.severity} |")
        else:
            lines.append("No pitch mismatches found.")
        lines.append("")

        # Density comparison
        lines.append("## Note Density by Measure")
        lines.append("")
        lines.append("| Measure | Ref Notes | Gen Notes | Issue |")
        lines.append("|---------|-----------|-----------|-------|")
        for dm in self.density_mismatches[:30]:
            flag = "" if dm.issue == "ok" else f" **{dm.issue}**"
            lines.append(f"| m{dm.measure} | {dm.ref_count} | {dm.gen_count} | {flag} |")
        lines.append("")

        # Contour comparison
        lines.append("## Contour Comparison")
        lines.append("")
        agreements = sum(1 for s in self.contour_segments if s.agrees)
        total = len(self.contour_segments)
        if total > 0:
            lines.append(f"Contour agreement: {agreements}/{total} segments ({100*agreements/total:.0f}%)")
            lines.append("")
            for s in self.contour_segments:
                emoji = "OK" if s.agrees else "MISMATCH"
                lines.append(f"- m{s.start_measure}-{s.end_measure}: "
                             f"ref={s.ref_direction}, gen={s.gen_direction} [{emoji}]")
        lines.append("")

        # Error taxonomy
        lines.append("## Error Taxonomy")
        lines.append("")
        if self.errors:
            categories = Counter(e.category for e in self.errors)
            lines.append("### Error Counts by Category")
            lines.append("")
            for cat, count in categories.most_common():
                lines.append(f"- **{cat}**: {count}")
            lines.append("")
            lines.append("### Error Details")
            lines.append("")
            for e in self.errors:
                lines.append(f"#### m{e.measure}: {e.category} ({e.severity})")
                lines.append(f"- {e.description}")
                lines.append(f"- Reference: {e.ref_context}")
                lines.append(f"- Generated: {e.gen_context}")
                lines.append(f"- Likely stage: **{e.likely_pipeline_stage}**")
                lines.append("")
        else:
            lines.append("No errors catalogued.")
        lines.append("")

        # Statistics
        lines.append("## Aggregate Statistics")
        lines.append("")
        for k, v in self.statistics.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

        return "\n".join(lines)


# ── Comparison engine ──────────────────────────────────────────────────

def _median(values: List[float]) -> float:
    if not values:
        return 0
    return statistics.median(values)


def _direction(pitches: List[int]) -> str:
    if len(pitches) < 2:
        return "flat"
    deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
    ups = sum(1 for d in deltas if d > 0)
    downs = sum(1 for d in deltas if d < 0)
    total = len(deltas)
    if ups > 0.7 * total:
        return "up"
    if downs > 0.7 * total:
        return "down"
    if ups < 0.15 * total and downs < 0.15 * total:
        return "flat"
    return "mixed"


def compare_scores(
    ref: ReferenceScore,
    gen: GeneratedScore,
    measure_range: Optional[Tuple[int, int]] = None,
) -> ComparisonReport:
    """
    Compare a reference score against generated output.

    Args:
        ref: Reference score with ground-truth notes.
        gen: Generated score from the pipeline.
        measure_range: Optional (start, end) to limit comparison range.

    Returns:
        ComparisonReport with detailed analysis.
    """
    if measure_range:
        m_start, m_end = measure_range
    else:
        ref_measures = ref.measures_with_notes()
        gen_measures = gen.measures_with_notes()
        m_start = min(ref_measures[0] if ref_measures else 1,
                      gen_measures[0] if gen_measures else 1)
        m_end = min(ref_measures[-1] if ref_measures else 1,
                    gen_measures[-1] if gen_measures else 1)

    ref_pitches_all = []
    gen_pitches_all = []
    pitch_mismatches = []
    density_mismatches = []

    for m in range(m_start, m_end + 1):
        ref_notes = ref.notes_in_measure(m)
        gen_notes = gen.notes_in_measure(m)

        ref_p = [n.pitch_midi for n in ref_notes]
        gen_p = [n.pitch_midi for n in gen_notes]

        ref_pitches_all.extend(ref_p)
        gen_pitches_all.extend(gen_p)

        ref_count = len(ref_notes)
        gen_count = len(gen_notes)

        if ref_count > 0 or gen_count > 0:
            issue = "ok"
            if ref_count > 0 and gen_count == 0:
                issue = "missing"
            elif gen_count > 0 and ref_count == 0:
                issue = "extra"
            elif ref_count > 0 and gen_count < ref_count * 0.5:
                issue = "too_sparse"
            elif ref_count > 0 and gen_count > ref_count * 2.0:
                issue = "too_dense"

            density_mismatches.append(DensityMismatch(
                measure=m,
                ref_count=ref_count,
                gen_count=gen_count,
                delta=gen_count - ref_count,
                issue=issue,
            ))

        if ref_p and gen_p:
            ref_med = _median(ref_p)
            gen_med = _median(gen_p)
            delta = gen_med - ref_med
            is_octave = abs(abs(delta) - 12) <= 1
            if abs(delta) >= 12:
                sev = "octave"
            elif abs(delta) >= 6:
                sev = "severe"
            elif abs(delta) >= 3:
                sev = "moderate"
            elif abs(delta) >= 1:
                sev = "minor"
            else:
                sev = "match"

            if sev != "match":
                pitch_mismatches.append(PitchMismatch(
                    measure=m,
                    ref_pitches=ref_p,
                    gen_pitches=gen_p,
                    ref_median=ref_med,
                    gen_median=gen_med,
                    delta_semitones=delta,
                    likely_octave_error=is_octave,
                    severity=sev,
                ))

    global_offset = (_median(gen_pitches_all) - _median(ref_pitches_all)) if ref_pitches_all and gen_pitches_all else 0

    # Contour analysis (4-measure segments)
    contour_segments = []
    for seg_start in range(m_start, m_end, 4):
        seg_end = min(seg_start + 3, m_end)
        ref_seg_pitches = []
        gen_seg_pitches = []
        for m in range(seg_start, seg_end + 1):
            ref_seg_pitches.extend(n.pitch_midi for n in ref.notes_in_measure(m))
            gen_seg_pitches.extend(n.pitch_midi for n in gen.notes_in_measure(m))

        ref_dir = _direction(ref_seg_pitches)
        gen_dir = _direction(gen_seg_pitches)
        contour_segments.append(ContourSegment(
            start_measure=seg_start,
            end_measure=seg_end,
            ref_direction=ref_dir,
            gen_direction=gen_dir,
            agrees=(ref_dir == gen_dir),
        ))

    # Build error instances
    errors = _build_error_taxonomy(ref, gen, pitch_mismatches, density_mismatches, m_start, m_end)

    # Compute statistics
    ref_note_count = len(ref_pitches_all)
    gen_note_count = len(gen_pitches_all)
    ref_measures_active = len([m for m in range(m_start, m_end+1) if ref.notes_in_measure(m)])
    gen_measures_active = len([m for m in range(m_start, m_end+1) if gen.notes_in_measure(m)])

    ref_pitch_range = (min(ref_pitches_all), max(ref_pitches_all)) if ref_pitches_all else (0, 0)
    gen_pitch_range = (min(gen_pitches_all), max(gen_pitches_all)) if gen_pitches_all else (0, 0)

    report = ComparisonReport(
        title=f"{ref.title} comparison",
        ref_summary={
            "key": ref.key,
            "time_sig": ref.time_signature,
            "tempo": ref.tempo_bpm,
            "note_count": ref_note_count,
            "pitch_range": f"{ref_pitch_range[0]}-{ref_pitch_range[1]}",
            "median_pitch": _median(ref_pitches_all) if ref_pitches_all else 0,
            "avg_density": ref_note_count / max(1, ref_measures_active),
        },
        gen_summary={
            "key": gen.key,
            "time_sig": gen.time_signature,
            "tempo": gen.tempo_bpm,
            "note_count": gen_note_count,
            "pitch_range": f"{gen_pitch_range[0]}-{gen_pitch_range[1]}",
            "median_pitch": _median(gen_pitches_all) if gen_pitches_all else 0,
            "avg_density": gen_note_count / max(1, gen_measures_active),
        },
        global_pitch_offset=global_offset,
        pitch_mismatches=pitch_mismatches,
        density_mismatches=density_mismatches,
        contour_segments=contour_segments,
        errors=errors,
        statistics={
            "comparison_range": f"m{m_start}-m{m_end}",
            "measures_compared": m_end - m_start + 1,
            "ref_notes_in_range": ref_note_count,
            "gen_notes_in_range": gen_note_count,
            "global_pitch_offset_semitones": global_offset,
            "octave_error_measures": sum(1 for m in pitch_mismatches if m.likely_octave_error),
            "severe_mismatch_measures": sum(1 for m in pitch_mismatches if m.severity in ("severe", "octave")),
            "density_missing_measures": sum(1 for d in density_mismatches if d.issue == "missing"),
            "density_sparse_measures": sum(1 for d in density_mismatches if d.issue == "too_sparse"),
            "contour_agreement_pct": (
                100 * sum(1 for s in contour_segments if s.agrees) / max(1, len(contour_segments))
            ),
            "error_count_by_category": dict(Counter(e.category for e in errors).most_common()),
        },
    )

    return report


def _build_error_taxonomy(
    ref: ReferenceScore,
    gen: GeneratedScore,
    pitch_mismatches: List[PitchMismatch],
    density_mismatches: List[DensityMismatch],
    m_start: int,
    m_end: int,
) -> List[ErrorInstance]:
    """Identify specific error instances and classify them."""
    errors = []

    for pm in pitch_mismatches:
        if pm.likely_octave_error:
            errors.append(ErrorInstance(
                category="octave_error",
                measure=pm.measure,
                description=f"Generated pitches are ~{pm.delta_semitones:+.0f} semitones from reference (likely octave error)",
                ref_context=f"Reference median: MIDI {pm.ref_median:.0f} ({_midi_name(int(pm.ref_median))}), pitches: {[_midi_name(p) for p in pm.ref_pitches]}",
                gen_context=f"Generated median: MIDI {pm.gen_median:.0f} ({_midi_name(int(pm.gen_median))}), pitches: {[_midi_name(p) for p in pm.gen_pitches]}",
                likely_pipeline_stage="melody_path (DP resolver selecting wrong octave track)",
                severity="critical",
            ))
        elif pm.severity == "severe":
            errors.append(ErrorInstance(
                category="wrong_pitch_center",
                measure=pm.measure,
                description=f"Pitch center off by {pm.delta_semitones:+.1f} semitones",
                ref_context=f"Ref: {[_midi_name(p) for p in pm.ref_pitches]}",
                gen_context=f"Gen: {[_midi_name(p) for p in pm.gen_pitches]}",
                likely_pipeline_stage="pitch_refinement or melody_path",
                severity="major",
            ))
        elif pm.severity == "moderate":
            errors.append(ErrorInstance(
                category="pitch_drift",
                measure=pm.measure,
                description=f"Pitch drifted {pm.delta_semitones:+.1f} semitones from reference",
                ref_context=f"Ref: {[_midi_name(p) for p in pm.ref_pitches]}",
                gen_context=f"Gen: {[_midi_name(p) for p in pm.gen_pitches]}",
                likely_pipeline_stage="vocal_abstraction or pitch_refinement",
                severity="moderate",
            ))

    for dm in density_mismatches:
        if dm.issue == "missing":
            errors.append(ErrorInstance(
                category="missing_notes",
                measure=dm.measure,
                description=f"Reference has {dm.ref_count} notes but generated has none",
                ref_context=f"{dm.ref_count} notes in reference",
                gen_context="No notes generated (rest or gap)",
                likely_pipeline_stage="melody_path (gap state over-selection) or candidate_notes",
                severity="major",
            ))
        elif dm.issue == "too_sparse":
            errors.append(ErrorInstance(
                category="over_merged",
                measure=dm.measure,
                description=f"Generated has {dm.gen_count} notes vs reference's {dm.ref_count} (too few)",
                ref_context=f"{dm.ref_count} notes",
                gen_context=f"{dm.gen_count} notes",
                likely_pipeline_stage="vocal_abstraction (over-merging) or phrase_smoothing",
                severity="moderate",
            ))
        elif dm.issue == "too_dense":
            errors.append(ErrorInstance(
                category="note_explosion",
                measure=dm.measure,
                description=f"Generated has {dm.gen_count} notes vs reference's {dm.ref_count} (too many)",
                ref_context=f"{dm.ref_count} notes",
                gen_context=f"{dm.gen_count} notes",
                likely_pipeline_stage="vocal_abstraction (insufficient merging) or vibrato suppression",
                severity="minor",
            ))

    # Check for suspicious consecutive interval jumps in generated output
    gen_all_notes = []
    for m in range(m_start, m_end + 1):
        gen_all_notes.extend(gen.notes_in_measure(m))
    for i in range(1, len(gen_all_notes)):
        interval = abs(gen_all_notes[i].pitch_midi - gen_all_notes[i-1].pitch_midi)
        if interval >= 12:
            m = gen_all_notes[i].measure
            errors.append(ErrorInstance(
                category="implausible_interval",
                measure=m,
                description=f"Jump of {interval} semitones: {_midi_name(gen_all_notes[i-1].pitch_midi)} -> {_midi_name(gen_all_notes[i].pitch_midi)}",
                ref_context="Expected stepwise or small interval vocal motion",
                gen_context=f"{_midi_name(gen_all_notes[i-1].pitch_midi)} -> {_midi_name(gen_all_notes[i].pitch_midi)} ({interval} st)",
                likely_pipeline_stage="melody_path (stream switch) or pitch_refinement (octave correction failure)",
                severity="major",
            ))

    return errors


def _midi_name(midi: int) -> str:
    note_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return f"{note_names[midi % 12]}{(midi // 12) - 1}"


# ── CLI entry point ────────────────────────────────────────────────────

def main():
    import sys
    analysis_dir = Path(__file__).parent

    ref_path = analysis_dir / "reference_notes.json"
    if not ref_path.exists():
        print("ERROR: reference_notes.json not found. Run build_reference.py first.")
        sys.exit(1)

    gen_json = analysis_dir.parent / "out" / "debug" / "raise_your_glass__vocals.json"
    if not gen_json.exists():
        print(f"ERROR: Generated debug JSON not found at {gen_json}")
        sys.exit(1)

    ref = ReferenceScore.load(str(ref_path))
    gen = parse_debug_json(str(gen_json))

    print(f"Reference: {len(ref.notes)} notes, measures {ref.notes[0].measure}-{ref.notes[-1].measure}")
    print(f"Generated: {len(gen.notes)} notes")

    report = compare_scores(ref, gen, measure_range=(1, 36))

    report.save_json(str(analysis_dir / "comparison_report.json"))
    report.save_markdown(str(analysis_dir / "reference_vs_generated_report.md"))

    gen_out = GeneratedScore(
        title=gen.title,
        key=gen.key,
        time_signature=gen.time_signature,
        tempo_bpm=gen.tempo_bpm,
        source_mode=gen.source_mode,
        notes=gen.notes,
        raw_notes=gen.raw_notes,
        metadata=gen.metadata,
    )
    gen_out.save(str(analysis_dir / "generated_notes.json"))

    print(f"\nGlobal pitch offset: {report.global_pitch_offset:+.1f} semitones")
    print(f"Octave-error measures: {report.statistics['octave_error_measures']}")
    print(f"Severe mismatch measures: {report.statistics['severe_mismatch_measures']}")
    print(f"Contour agreement: {report.statistics['contour_agreement_pct']:.0f}%")
    print(f"Total errors: {len(report.errors)}")
    print(f"Error categories: {report.statistics['error_count_by_category']}")
    print(f"\nReports saved to {analysis_dir}")


if __name__ == "__main__":
    main()
