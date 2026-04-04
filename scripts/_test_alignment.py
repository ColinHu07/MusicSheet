"""Quick test of reference alignment on existing debug data."""
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.utils import NoteEvent
from src.pipeline.reference_correction import (
    load_reference_notes, align_notes, apply_corrections,
    merge_aligned_fragments,
)

# Load reference notes
ref_notes, meta = load_reference_notes("analysis/reference_notes.json")
print(f"Reference: {meta['total_notes']} notes, {meta['time_span_s']:.1f}s, {meta['tempo_bpm']} BPM")

# Load generated raw notes from debug JSON
with open("out/debug/raise_your_glass__vocals.json") as f:
    d = json.load(f)

gen_notes = [
    NoteEvent(
        start_s=n["start_s"],
        end_s=n["end_s"],
        midi_pitch=n["midi_pitch"],
        confidence=n.get("confidence", 1.0),
    )
    for n in d["raw_notes"]
]
print(f"Generated: {len(gen_notes)} notes")

# Align
alignment = align_notes(ref_notes, gen_notes)

# Summary
matches = [p for p in alignment if p.action == "match"]
skip_ref = [p for p in alignment if p.action == "skip_ref"]
skip_gen = [p for p in alignment if p.action == "skip_gen"]

print(f"\nAlignment: {len(matches)} matches, {len(skip_ref)} skip_ref, {len(skip_gen)} skip_gen")

# Confidence breakdown
high = sum(1 for p in matches if p.confidence == "HIGH")
med = sum(1 for p in matches if p.confidence == "MEDIUM")
low = sum(1 for p in matches if p.confidence == "LOW")
print(f"Confidence: HIGH={high}, MEDIUM={med}, LOW={low}")

# Pitch error distribution
pitch_dists = [p.pitch_distance_st for p in matches]
correct = sum(1 for d in pitch_dists if d == 0)
off_1 = sum(1 for d in pitch_dists if d == 1)
off_2 = sum(1 for d in pitch_dists if d == 2)
off_3 = sum(1 for d in pitch_dists if d == 3)
off_big = sum(1 for d in pitch_dists if d > 3)
print(f"Pitch distances: correct={correct}, 1st={off_1}, 2st={off_2}, 3st={off_3}, >3st={off_big}")

# Average time distance for matches
time_dists = [p.time_distance_s for p in matches]
avg_time = sum(time_dists) / len(time_dists) if time_dists else 0
max_time = max(time_dists) if time_dists else 0
print(f"Time distance: avg={avg_time:.3f}s, max={max_time:.3f}s")

# Apply corrections
corrected, stats = apply_corrections(gen_notes, alignment, ref_notes)
print(f"\nCorrections: {stats.pitch_corrections} pitches corrected")
print(f"Already correct: {stats.already_correct}")
print(f"Avg correction: {stats.avg_correction_st:.1f} semitones")
print(f"By size: {stats.corrections_by_size}")

# Merge repair
merged, n_merges = merge_aligned_fragments(corrected, alignment, ref_notes)
print(f"\nMerge repair: {n_merges} merges -> {len(merged)} notes (was {len(corrected)})")

# Show first 20 alignment pairs
print("\n--- First 20 matched pairs ---")
row = 0
for p in alignment:
    if p.action != "match":
        continue
    row += 1
    if row > 20:
        break
    r = ref_notes[p.ref_idx]
    g = gen_notes[p.gen_idx]
    NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    rname = f"{NOTE_NAMES[r.midi_pitch % 12]}{r.midi_pitch // 12 - 1}"
    gname = f"{NOTE_NAMES[g.midi_pitch % 12]}{g.midi_pitch // 12 - 1}"
    delta = r.midi_pitch - g.midi_pitch
    print(f"  [{p.confidence:6s}] ref m{r.measure}:{r.position_in_measure:.1f} {rname:4s} "
          f"<-> gen {g.start_s:.2f}s {gname:4s}  "
          f"time_d={p.time_distance_s:.3f}s  pitch_d={delta:+d}st")
