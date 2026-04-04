"""
Visualizations for reference vs generated score comparison.

Generates ASCII-art and optional matplotlib charts for:
- Pitch contour over note index
- Notes-per-measure density
- Duration-class histogram
- Pitch distribution histogram
- Mismatch heatmap by measure
"""

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .reference_score_parser import ReferenceScore
from .generated_score_parser import GeneratedScore, parse_debug_json


def _midi_name(midi: int) -> str:
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    return f"{names[midi % 12]}{(midi // 12) - 1}"


def ascii_pitch_contour(ref: ReferenceScore, gen: GeneratedScore,
                        measure_range: Tuple[int, int] = (1, 36)) -> str:
    """ASCII pitch contour comparison, measure by measure."""
    lines = []
    lines.append("=" * 80)
    lines.append("PITCH CONTOUR COMPARISON (Reference vs Generated)")
    lines.append("=" * 80)
    lines.append(f"{'Measure':>8}  {'Ref Pitches':^30}  {'Gen Pitches':^30}")
    lines.append("-" * 80)

    m_start, m_end = measure_range
    for m in range(m_start, m_end + 1):
        ref_notes = ref.notes_in_measure(m)
        gen_notes = gen.notes_in_measure(m)

        ref_str = " ".join(_midi_name(n.pitch_midi) for n in ref_notes) if ref_notes else "(rest)"
        gen_str = " ".join(n.pitch_name for n in gen_notes) if gen_notes else "(rest)"

        if len(ref_str) > 30:
            ref_str = ref_str[:27] + "..."
        if len(gen_str) > 30:
            gen_str = gen_str[:27] + "..."

        lines.append(f"  m{m:3d}    {ref_str:<30}  {gen_str:<30}")

    lines.append("")
    return "\n".join(lines)


def ascii_density_chart(ref: ReferenceScore, gen: GeneratedScore,
                        measure_range: Tuple[int, int] = (1, 36)) -> str:
    """ASCII bar chart of notes per measure."""
    lines = []
    lines.append("=" * 70)
    lines.append("NOTES PER MEASURE: Reference (R) vs Generated (G)")
    lines.append("=" * 70)

    m_start, m_end = measure_range
    for m in range(m_start, m_end + 1):
        ref_count = len(ref.notes_in_measure(m))
        gen_count = len(gen.notes_in_measure(m))
        ref_bar = "R" * ref_count
        gen_bar = "G" * gen_count
        lines.append(f"  m{m:3d}  R: {ref_bar:<12} ({ref_count})  G: {gen_bar:<12} ({gen_count})")

    lines.append("")
    return "\n".join(lines)


def ascii_pitch_histogram(ref: ReferenceScore, gen: GeneratedScore) -> str:
    """ASCII histogram of pitch distribution."""
    lines = []
    lines.append("=" * 70)
    lines.append("PITCH DISTRIBUTION HISTOGRAM")
    lines.append("=" * 70)

    ref_pitches = Counter(n.pitch_midi for n in ref.notes if not n.is_rest)
    gen_pitches = Counter(n.pitch_midi for n in gen.notes)

    all_pitches = sorted(set(ref_pitches.keys()) | set(gen_pitches.keys()))
    max_count = max(max(ref_pitches.values(), default=0),
                    max(gen_pitches.values(), default=0))
    scale = max(1, max_count // 30)

    lines.append(f"  {'Note':>6}  {'Reference':^35}  {'Generated':^35}")
    lines.append("-" * 80)

    for p in all_pitches:
        name = _midi_name(p)
        rc = ref_pitches.get(p, 0)
        gc = gen_pitches.get(p, 0)
        ref_bar = "#" * (rc // scale) if rc > 0 else ""
        gen_bar = "#" * (gc // scale) if gc > 0 else ""
        lines.append(f"  {name:>6}  {ref_bar:<30} {rc:3d}  {gen_bar:<30} {gc:3d}")

    lines.append("")
    return "\n".join(lines)


def ascii_mismatch_heatmap(ref: ReferenceScore, gen: GeneratedScore,
                           measure_range: Tuple[int, int] = (1, 36)) -> str:
    """ASCII heatmap showing severity of pitch mismatch per measure."""
    lines = []
    lines.append("=" * 70)
    lines.append("PITCH MISMATCH HEATMAP BY MEASURE")
    lines.append("  . = match, o = minor (1-2st), O = moderate (3-5st),")
    lines.append("  X = severe (6-11st), ! = octave error (12st)")
    lines.append("=" * 70)

    m_start, m_end = measure_range
    row = "  "
    for m in range(m_start, m_end + 1):
        ref_notes = ref.notes_in_measure(m)
        gen_notes = gen.notes_in_measure(m)

        if not ref_notes or not gen_notes:
            row += "-"
            continue

        ref_med = statistics.median(n.pitch_midi for n in ref_notes)
        gen_med = statistics.median(n.pitch_midi for n in gen_notes)
        delta = abs(gen_med - ref_med)

        if delta < 1:
            row += "."
        elif delta < 3:
            row += "o"
        elif delta < 6:
            row += "O"
        elif abs(delta - 12) <= 1:
            row += "!"
        else:
            row += "X"

    lines.append(row)
    lines.append("")

    measure_labels = "  "
    for m in range(m_start, m_end + 1):
        if m % 5 == 0:
            measure_labels += str(m % 10)
        else:
            measure_labels += " "
    lines.append(measure_labels)

    tick_labels = "  "
    for m in range(m_start, m_end + 1):
        if m % 10 == 0:
            tick_labels += "|"
        elif m % 5 == 0:
            tick_labels += "+"
        else:
            tick_labels += " "
    lines.append(tick_labels)
    lines.append("")

    return "\n".join(lines)


def generate_all_visualizations(
    ref_path: str = None,
    gen_json_path: str = None,
    output_path: str = None,
):
    """Generate all ASCII visualizations and save to file."""
    analysis_dir = Path(__file__).parent

    if ref_path is None:
        ref_path = str(analysis_dir / "reference_notes.json")
    if gen_json_path is None:
        gen_json_path = str(analysis_dir.parent / "out" / "debug" / "raise_your_glass__vocals.json")
    if output_path is None:
        output_path = str(analysis_dir / "visualizations.txt")

    ref = ReferenceScore.load(ref_path)
    gen = parse_debug_json(gen_json_path)

    parts = []
    parts.append("REFERENCE VS GENERATED ANALYSIS VISUALIZATIONS")
    parts.append(f"Reference: {ref.title} ({len(ref.notes)} notes)")
    parts.append(f"Generated: {gen.title} ({len(gen.notes)} notes)")
    parts.append(f"Comparison range: measures 1-36")
    parts.append("")

    parts.append(ascii_mismatch_heatmap(ref, gen))
    parts.append(ascii_pitch_contour(ref, gen))
    parts.append(ascii_density_chart(ref, gen))
    parts.append(ascii_pitch_histogram(ref, gen))

    output = "\n".join(parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Visualizations saved to {output_path}")
    print()
    print(output)


if __name__ == "__main__":
    generate_all_visualizations()
