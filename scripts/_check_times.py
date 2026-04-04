"""Check time ranges of ref and gen notes."""
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.utils import NoteEvent
from src.pipeline.reference_correction import load_reference_notes

ref_notes, meta = load_reference_notes("analysis/reference_notes.json")
print(f"Reference: {len(ref_notes)} notes")
print(f"  First: {ref_notes[0].time_s:.2f}s  Last: {ref_notes[-1].end_s:.2f}s")
print(f"  Time span: {ref_notes[-1].end_s - ref_notes[0].time_s:.2f}s")

with open("out/debug/raise_your_glass__vocals.json") as f:
    d = json.load(f)

raw = d["raw_notes"]
print(f"\nGenerated: {len(raw)} notes")
print(f"  First: {raw[0]['start_s']:.2f}s  Last end: {raw[-1]['end_s']:.2f}s")
print(f"  Time span: {raw[-1]['end_s'] - raw[0]['start_s']:.2f}s")

# Show gen notes around index 60-70
for i in range(60, min(70, len(raw))):
    n = raw[i]
    print(f"  gen[{i}]: start={n['start_s']:.3f}s end={n['end_s']:.3f}s midi={n['midi_pitch']}")

# Last 5
print("\nLast 5 gen notes:")
for i in range(max(0, len(raw)-5), len(raw)):
    n = raw[i]
    print(f"  gen[{i}]: start={n['start_s']:.3f}s end={n['end_s']:.3f}s midi={n['midi_pitch']}")

# Compare time coverage
ref_time_limit = ref_notes[-1].end_s
gen_within = sum(1 for n in raw if n['start_s'] <= ref_time_limit)
print(f"\nRef time limit: {ref_time_limit:.2f}s")
print(f"Gen notes within ref range: {gen_within}/{len(raw)}")

# Check: how many ref notes are within gen time range?
gen_time_limit = raw[-1]['end_s']
ref_within = sum(1 for r in ref_notes if r.time_s <= gen_time_limit)
print(f"Gen time limit: {gen_time_limit:.2f}s")
print(f"Ref notes within gen range: {ref_within}/{len(ref_notes)}")
