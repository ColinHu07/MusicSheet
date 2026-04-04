"""Compare corrected output against reference."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.reference_score_parser import ReferenceScore
from analysis.generated_score_parser import parse_debug_json
from analysis.compare_to_reference import compare_scores

ref = ReferenceScore.load("analysis/reference_notes.json")
gen = parse_debug_json("out/debug/raise_your_glass__vocals.json")

report = compare_scores(ref, gen, measure_range=(1, 36))

print("=" * 60)
print("COMPARISON: Reference vs Generated (with correction)")
print("=" * 60)
print(f"Global pitch offset: {report.global_pitch_offset:+.1f} semitones")
print(f"Contour agreement: {report.statistics['contour_agreement_pct']:.0f}%")
print(f"Total errors: {len(report.errors)}")
print(f"Error categories: {report.statistics['error_count_by_category']}")
print(f"Octave-error measures: {report.statistics['octave_error_measures']}")
print(f"Severe mismatch measures: {report.statistics['severe_mismatch_measures']}")

print(f"\nRef notes in range: {report.statistics['ref_notes_in_range']}")
print(f"Gen notes in range: {report.statistics['gen_notes_in_range']}")
print(f"Ref avg density: {report.ref_summary['avg_density']:.1f}")
print(f"Gen avg density: {report.gen_summary['avg_density']:.1f}")

print("\n--- Pitch Mismatches ---")
for pm in report.pitch_mismatches:
    print(f"  m{pm.measure}: ref={pm.ref_median:.0f} gen={pm.gen_median:.0f} "
          f"delta={pm.delta_semitones:+.1f} [{pm.severity}]")
