"""
Pipeline comparison tool — compare outputs from different pipeline stages.

Usage::

    python -m eval.compare_pipeline out/debug/song.json
    python -m eval.compare_pipeline out/debug/song.json --reference eval/references/song.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.metrics import (
    EvalReport,
    compare_against_reference,
    compute_heuristic_metrics,
    format_report,
    notes_from_dicts,
)

logger = logging.getLogger("mp3_to_sheet.eval")


def load_debug_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_stages(debug_data: dict, reference_path: str = None) -> str:
    """
    Compare different pipeline stages from a debug JSON file.

    Returns a formatted text report.
    """
    sections = []

    # Raw notes (from Basic Pitch, before melody-path resolver)
    if "raw_notes" in debug_data:
        raw = notes_from_dicts(debug_data["raw_notes"])
        report = compute_heuristic_metrics(raw, "Raw Melody Notes (post-resolver)")
        sections.append(format_report(report))

    # Quantized notes
    if "quantized_notes" in debug_data:
        quant_dicts = debug_data["quantized_notes"]
        quant_notes = []
        for d in quant_dicts:
            if not d.get("is_rest", False):
                from eval.metrics import NoteForEval
                quant_notes.append(NoteForEval(
                    start_s=d.get("start_ql", 0.0),
                    end_s=d.get("start_ql", 0.0) + d.get("duration_ql", 0.0),
                    pitch_midi=d.get("midi_pitch", 0),
                ))
        report = compute_heuristic_metrics(quant_notes, "Quantized Notes")
        sections.append(format_report(report))

    # Melody stats
    if "melody_stats" in debug_data:
        stats = debug_data["melody_stats"]
        lines = [
            "=== Melody Path Stats ===",
            f"  Candidates      : {stats.get('total_candidates', '?')}",
            f"  Lattice columns : {stats.get('lattice_columns', '?')}",
            f"  Gap states      : {stats.get('gap_states_in_path', '?')}",
            f"  Selected unique : {stats.get('unique_selected', '?')}",
            f"  After cleanup   : {stats.get('after_cleanup', '?')}",
        ]
        sections.append("\n".join(lines))

    # Reference comparison
    if reference_path and os.path.isfile(reference_path):
        with open(reference_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        ref_notes = notes_from_dicts(ref_data if isinstance(ref_data, list)
                                      else ref_data.get("notes", []))
        if "raw_notes" in debug_data:
            pred = notes_from_dicts(debug_data["raw_notes"])
            comparison = compare_against_reference(pred, ref_notes)
            lines = [
                "=== Reference Comparison ===",
                f"  Precision : {comparison['precision']:.2%}",
                f"  Recall    : {comparison['recall']:.2%}",
                f"  F1        : {comparison['f1']:.2%}",
                f"  Matched   : {comparison['matched']} / {comparison['predicted_count']} predicted, {comparison['reference_count']} reference",
            ]
            sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else "No data found in debug file."


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline stages")
    parser.add_argument("debug_json", help="Path to debug JSON from pipeline")
    parser.add_argument("--reference", help="Path to reference melody JSON")
    parser.add_argument("--output", help="Save report to file")
    args = parser.parse_args()

    data = load_debug_json(args.debug_json)
    report = compare_stages(data, args.reference)

    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
