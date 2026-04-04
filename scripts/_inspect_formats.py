"""Inspect reference and generated note formats."""
import json

with open("analysis/reference_notes.json") as f:
    ref = json.load(f)

print("=== REFERENCE ===")
print("Top keys:", list(ref.keys()))
notes = ref["notes"]
print(f"Notes: {len(notes)}")
print("Sample note keys:", list(notes[0].keys()))
for n in notes[:5]:
    print(f"  m{n['measure']} pos={n['position_in_measure']} "
          f"midi={n['pitch_midi']} name={n['pitch_name']} "
          f"dur={n['duration_class']} dur_q={n['duration_quarters']} "
          f"rest={n.get('is_rest', False)}")

with open("out/debug/raise_your_glass__vocals.json") as f:
    gen = json.load(f)

print("\n=== GENERATED (raw_notes) ===")
raw = gen["raw_notes"]
print(f"Count: {len(raw)}")
print("Keys:", list(raw[0].keys()))
for n in raw[:5]:
    print(f"  midi={n['midi_pitch']} start={n['start_s']:.3f} end={n['end_s']:.3f}")

print("\n=== GENERATED (quantized_notes) ===")
qn = gen["quantized_notes"]
print(f"Count: {len(qn)}")
print("Keys:", list(qn[0].keys()))
for n in qn[:8]:
    print(f"  midi={n.get('midi_pitch','?')} offset_ql={n.get('offset_ql','?')} "
          f"dur_ql={n.get('duration_ql','?')} rest={n.get('is_rest', False)}")

print(f"\ntempo_bpm: {gen['tempo_bpm']}")
print(f"time_sig: {gen['time_signature']}")

# Compute measure boundaries
tempo = gen["tempo_bpm"]
beat_dur = 60.0 / tempo
measure_dur = beat_dur * 4

print(f"\nmeasure_dur: {measure_dur:.3f}s")
print(f"beats_per_measure: 4")

# Show reference measures breakdown
ref_non_rest = [n for n in notes if not n.get("is_rest", False)]
print(f"\nReference non-rest notes: {len(ref_non_rest)}")
max_m = max(n["measure"] for n in notes)
print(f"Reference measure range: 1-{max_m}")
print(f"Total measures in reference: {max_m}")
