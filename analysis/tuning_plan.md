# Reference-Guided Tuning Plan

## Song: "Raise Your Glass" (P!nk) — Vocals Mode
## Date: 2026-04-03

---

## Executive Summary

The pipeline produces structurally plausible output (correct key, time signature, tempo,
reasonable contour shape), but suffers from two fundamental problems that render the
musical content incorrect:

1. **Systematic octave error**: The melody is extracted ~8 semitones (nearly one octave)
   too low across most of the song. The reference median pitch is Ab4 (MIDI 68), while
   the generated median is C4 (MIDI 60).

2. **Over-merging**: The generated output has roughly half the note density of the
   reference (2.4 vs 4.4 notes/measure), destroying melodic detail.

These two problems account for ~85% of all identified errors.

---

## Priority 1: Fix Octave Selection in Melody Path Resolver

### What to change

The DP melody-path resolver (`src/pipeline/melody_path.py`) is the primary source of
errors. It systematically selects notes from the wrong octave because:

- Basic Pitch generates many **lower-octave harmonic candidates** that are actually
  sub-harmonics or accompaniment artifacts
- The register scoring (`melody_features.py`) targets the **upper quartile pitch** (68),
  but this target is already biased low because the candidate pool contains so many
  low-octave notes
- The register weight (2.5) and local register weight (3.0) are insufficient to
  override the confidence/duration advantages of wrong-octave candidates

### Specific tuning recommendations

1. **Add a `pitch_floor_midi` for vocals mode.** The vocal melody for this song sits
   at Ab4-Eb5. Setting `pitch_floor_midi = 60` (C4) would eliminate the worst
   sub-harmonic candidates without losing real vocal content. This should be set
   dynamically based on analysis of the candidate pool.

2. **Increase register scoring aggressiveness in vocals mode:**
   - `register_weight`: increase from 2.5 to 4.0-5.0
   - `local_register_weight`: increase from 3.0 to 4.5-5.5
   - `below_local_median_penalty_weight`: increase from 2.0 to 3.5-4.0
   - These changes would make the resolver strongly prefer notes above the median,
     which is the correct behavior for vocal melody extraction

3. **Improve upper-quartile estimation.** Currently computed from the raw candidate
   pool, which includes many low harmonics. Consider:
   - Computing upper quartile only from candidates above the global median
   - Using a two-pass approach: first estimate the melody register, then re-score
   - Alternatively, use the instrument range center (violin center = 76) as the
     register target for vocals mode

4. **Strengthen octave-flip penalty.** Currently 6.0, which is not enough to prevent
   the resolver from switching between octave tracks. Consider increasing to 8.0-10.0
   for vocals mode, where octave jumps are rare.

5. **Consider a "vocal register detection" pre-pass.** Before running the DP:
   - Analyze candidate pitch histogram
   - Identify the likely vocal register (the peak in the upper portion)
   - Set the register target based on this analysis rather than the raw upper quartile

### What NOT to change

- Do NOT change the DP algorithm structure (lattice, Viterbi) — it works well
- Do NOT change the transition scoring logic — contour agreement is already 89%
- Do NOT remove the gap state — gap handling is functional

### Expected impact

Fixing the octave selection would eliminate ~26 of the 34 pitch mismatch errors and
should bring the median pitch from MIDI 60 to ~68, matching the reference.

---

## Priority 2: Reduce Over-Merging in Vocal Abstraction

### What to change

The vocals-mode config overrides in `src/pipeline/__init__.py` (lines 183-191) are
too aggressive:

```
Current vocals-mode overrides:
  merge_gap_ms: 200.0        (default 120.0) ← too aggressive
  merge_nearby_pitch_gap_ms: 150.0   (default 80.0) ← too aggressive
  consolidate_max_gap_ms: 300.0      (default 150.0) ← too aggressive
  pitch_stability_max_gap_ms: 250.0  (default 120.0) ← somewhat high
  phrase_max_short_note_ms: 300.0    (default 180.0) ← too aggressive
  micro_note_ms: 130.0       (default 80.0) ← reasonable
  reattack_max_gap_ms: 200.0 (default 120.0) ← somewhat high
```

### Specific tuning recommendations

1. **Reduce merge tolerances:**
   - `merge_gap_ms`: 200 → 140 (closer to default)
   - `merge_nearby_pitch_gap_ms`: 150 → 100
   - `consolidate_max_gap_ms`: 300 → 180

2. **Preserve more short notes:**
   - `phrase_max_short_note_ms`: 300 → 200 (notes up to 300ms are real melody events)

3. **Reduce reattack suppression:**
   - `reattack_max_gap_ms`: 200 → 140

4. **Target note density**: The reference has ~4.4 notes/measure. With the current
   config producing ~2.4, the tuned version should aim for 3.5-4.5 notes/measure.

### What NOT to change

- Do NOT disable vibrato suppression — it correctly handles pitch oscillation
- Do NOT disable phrase smoothing entirely — it provides useful cleanup
- Do NOT change the vibrato detection parameters — they work

### Expected impact

Reducing over-merging would recover ~13 "too sparse" measures and bring note
density closer to the reference's 4.4 notes/measure.

---

## Priority 3: Improve Pitch Refinement for Octave Awareness

### What to change

The pitch refinement stage (`src/pipeline/pitch_refinement.py`) currently:
- Applied 13 neighbor corrections
- Applied 2 interval corrections
- Applied 0 octave corrections

The octave correction function (`correct_octave_errors`) requires a note to differ by
exactly 12 semitones from BOTH neighbors. This is too strict — many octave errors occur
in sequences where adjacent notes are also wrong.

### Specific tuning recommendations

1. **Relax octave correction trigger:** Instead of requiring both neighbors to agree,
   use a windowed approach that considers 2-3 neighbors on each side.

2. **Add a "global register correction" pass:** After DP selection, compare the
   overall selected pitch distribution against the expected vocal range. If the
   median is below the instrument's comfortable center, apply a systematic octave
   shift to notes that are clearly in the wrong register.

3. **Prevent neighbor correction from propagating octave errors:** The current
   neighbor correction (13 applied) may be nudging notes toward wrong-octave
   centers. Add a guard: if two neighbors are both in the wrong octave, don't
   correct the note between them toward that wrong octave.

4. **Increase `interval_max_jump`:** Currently 7 semitones. For vocals mode, the
   chorus hook has jumps of a perfect 5th (7 semitones) that are real. Consider
   increasing to 9 for vocals mode.

### What NOT to change

- Do NOT change the segment splitting logic — it's working correctly
- Do NOT remove interval smoothing — just tune the threshold

### Expected impact

Better pitch refinement would catch the remaining octave errors that slip through
the DP resolver, reducing the 6 pitch-drift instances.

---

## Priority 4: Candidate Note Quality (Upstream)

### What to investigate

The candidate notes from Basic Pitch contain many lower-octave harmonics. This
pollutes the candidate pool and biases the DP resolver's register estimation.

### Potential approaches

1. **Harmonic filtering:** For vocals mode, identify fundamental + harmonic pairs
   in the candidate pool and suppress the sub-harmonic (lower octave) candidates.

2. **Confidence re-weighting:** Basic Pitch confidence values may favor harmonics.
   Apply a register-aware confidence adjustment before DP scoring.

3. **Candidate pool analysis:** Before DP, compute the pitch histogram of candidates.
   If there are two clear peaks an octave apart, the upper peak is likely the
   fundamental vocal melody — boost those candidates.

### Priority

This is lower priority because fixing the DP register scoring (Priority 1) should
be sufficient. Only pursue this if Priority 1 doesn't fully resolve the octave issue.

---

## What NOT to Touch

These components are working correctly and should not be modified in the next pass:

1. **Source separation** — Demucs vocals mode is working, stems are routed correctly
2. **Key detection** — Ab major is correctly identified
3. **Time signature detection** — 4/4 is correct
4. **Tempo detection** — 143.6 BPM is close to reference's 140 (2.5% off, acceptable)
5. **Beat grid / quantization** — functioning correctly
6. **MusicXML export** — structurally correct
7. **PDF rendering** — working as expected
8. **Vibrato detection** — `_is_oscillating()` is appropriately strict
9. **Contour scoring** — 89% contour agreement is good given the octave offset

---

## Validation Plan

After implementing changes, run the comparison tool to verify:

1. **Pitch offset**: Should drop from -8.0 to within ±2.0 semitones
2. **Octave errors**: Should drop from 7 to 0-1
3. **Severe mismatches**: Should drop from 26 to under 5
4. **Note density**: Should increase from 2.4 to 3.5-4.5 per measure
5. **Contour agreement**: Should remain at or above 89%
6. **Median pitch**: Should rise from MIDI 60 to ~68

Use the reference comparison tools:
```bash
python -m analysis.compare_to_reference
python -m analysis.visualize
```

---

## Estimated Effort

| Priority | Component | Estimated Effort | Expected Impact |
|----------|-----------|-----------------|-----------------|
| 1 | Melody path register scoring | 2-3 hours | Eliminates ~75% of errors |
| 2 | Vocal abstraction tuning | 1 hour | Recovers ~15% of errors |
| 3 | Pitch refinement octave awareness | 1-2 hours | Catches remaining ~10% |
| 4 | Candidate pool analysis | 2-4 hours | Insurance / marginal gain |

Total estimated: 6-10 hours for a significant quality improvement.

---

## Appendix: Key Metrics for Tracking Progress

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Global pitch offset | -8.0 st | ±2.0 st | `compare_to_reference.py` |
| Octave error measures | 7/36 | 0-1/36 | comparison report |
| Severe mismatch measures | 26/36 | <5/36 | comparison report |
| Notes per measure | 2.4 | 3.5-4.5 | comparison report |
| Contour agreement | 89% | >85% | comparison report |
| Reference median pitch | 68 (Ab4) | — | fixed |
| Generated median pitch | 60 (C4) | 66-70 | comparison report |
