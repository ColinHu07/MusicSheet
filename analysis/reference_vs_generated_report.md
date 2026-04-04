# Reference vs Generated Comparison: Raise y_our glass comparison

## Summary

| Metric | Reference | Generated | Delta |
|--------|-----------|-----------|-------|
| Key | Ab major | Ab major | MATCH |
| Time Signature | 4/4 | 4/4 | MATCH |
| Tempo | 140 | 71.8 | -68.2 |
| Note Count (compared range) | 158 | 217 | +59 |
| Pitch Range | 60-75 | 60-82 | |
| Median Pitch | 68.0 | 68 | +0.0 st |
| Notes/Measure (avg) | 4.4 | 6.0 | +1.6 |

## Critical Finding: Global Pitch Offset

Pitch offset is +0.0 semitones (within acceptable range).

## Pitch Mismatches by Measure

- **Octave errors**: 2 measures
- **Severe (6+ st)**: 8 measures
- **Moderate (3-5 st)**: 9 measures
- **Minor (1-2 st)**: 8 measures

| Measure | Ref Median | Gen Median | Delta | Type |
|---------|-----------|-----------|-------|------|
| m2 | 68 | 69 | +1.0 | minor |
| m5 | 65 | 64 | -1.0 | minor |
| m7 | 68 | 63 | -5.0 | moderate |
| m8 | 68 | 66 | -2.0 | minor |
| m10 | 63 | 68 | +5.0 | moderate |
| m12 | 68 | 63 | -5.0 | moderate |
| m13 | 68 | 67 | -1.0 | minor |
| m14 | 72 | 64 | -8.0 | severe |
| m15 | 69 | 68 | -1.5 | minor |
| m16 | 68 | 65 | -3.0 | moderate |
| m17 | 61 | 72 | +11.5 | severe |
| m18 | 68 | 72 | +4.0 | moderate |
| m20 | 70 | 63 | -7.0 | severe |
| m22 | 70 | 66 | -4.5 | moderate |
| m23 | 67 | 69 | +2.0 | minor |
| m24 | 75 | 68 | -7.0 | severe |
| m25 | 74 | 65 | -8.5 | severe |
| m26 | 68 | 63 | -5.0 | moderate |
| m27 | 72 | 63 | -9.0 | severe |
| m28 | 72 | 64 | -7.5 | severe |
| m30 | 70 | 65 | -5.0 | moderate |
| m31 | 70 | 64 | -6.0 | severe |
| m32 | 70 | 65 | -5.0 | moderate |
| m33 | 68 | 79 | +11.5 | severe |
| m34 | 72 | 66 | -6.5 | severe |
| m35 | 72 | 70 | -2.0 | minor |
| m36 | 68 | 67 | -1.0 | minor |

## Note Density by Measure

| Measure | Ref Notes | Gen Notes | Issue |
|---------|-----------|-----------|-------|
| m1 | 6 | 7 |  |
| m2 | 7 | 6 |  |
| m3 | 6 | 6 |  |
| m4 | 6 | 6 |  |
| m5 | 5 | 4 |  |
| m6 | 6 | 7 |  |
| m7 | 7 | 5 |  |
| m8 | 5 | 4 |  |
| m9 | 6 | 8 |  |
| m10 | 3 | 7 |  **too_dense** |
| m11 | 4 | 5 |  |
| m12 | 6 | 5 |  |
| m13 | 5 | 1 |  **too_sparse** |
| m14 | 3 | 4 |  |
| m15 | 4 | 4 |  |
| m16 | 7 | 6 |  |
| m17 | 3 | 8 |  **too_dense** |
| m18 | 4 | 9 |  **too_dense** |
| m19 | 4 | 5 |  |
| m20 | 6 | 5 |  |
| m21 | 4 | 6 |  |
| m22 | 4 | 10 |  **too_dense** |
| m23 | 3 | 8 |  **too_dense** |
| m24 | 1 | 9 |  **too_dense** |
| m25 | 2 | 4 |  |
| m26 | 1 | 9 |  **too_dense** |
| m27 | 6 | 7 |  |
| m28 | 6 | 4 |  |
| m29 | 2 | 3 |  |
| m30 | 4 | 6 |  |

## Contour Comparison

Contour agreement: 9/9 segments (100%)

- m1-4: ref=mixed, gen=mixed [OK]
- m5-8: ref=mixed, gen=mixed [OK]
- m9-12: ref=mixed, gen=mixed [OK]
- m13-16: ref=mixed, gen=mixed [OK]
- m17-20: ref=mixed, gen=mixed [OK]
- m21-24: ref=mixed, gen=mixed [OK]
- m25-28: ref=mixed, gen=mixed [OK]
- m29-32: ref=mixed, gen=mixed [OK]
- m33-36: ref=mixed, gen=mixed [OK]

## Error Taxonomy

### Error Counts by Category

- **note_explosion**: 10
- **pitch_drift**: 9
- **wrong_pitch_center**: 8
- **octave_error**: 2
- **over_merged**: 1
- **implausible_interval**: 1

### Error Details

#### m7: pitch_drift (moderate)
- Pitch drifted -5.0 semitones from reference
- Reference: Ref: ['F4', 'Eb4', 'Ab4', 'Ab4', 'Ab4', 'Bb4', 'Ab4']
- Generated: Gen: ['Ab4', 'D4', 'F4', 'Eb4', 'Db4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m10: pitch_drift (moderate)
- Pitch drifted +5.0 semitones from reference
- Reference: Ref: ['Eb4', 'F4', 'Eb4']
- Generated: Gen: ['C4', 'Ab4', 'Gb4', 'Ab4', 'Bb4', 'Db4', 'Ab4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m12: pitch_drift (moderate)
- Pitch drifted -5.0 semitones from reference
- Reference: Ref: ['Bb4', 'Ab4', 'Eb4', 'F4', 'Ab4', 'Ab4']
- Generated: Gen: ['C4', 'G4', 'Eb4', 'Db4', 'Eb4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m14: wrong_pitch_center (major)
- Pitch center off by -8.0 semitones
- Reference: Ref: ['C5', 'Ab4', 'C5']
- Generated: Gen: ['G4', 'F4', 'Eb4', 'C4']
- Likely stage: **pitch_refinement or melody_path**

#### m16: pitch_drift (moderate)
- Pitch drifted -3.0 semitones from reference
- Reference: Ref: ['C5', 'Bb4', 'Ab4', 'Bb4', 'Ab4', 'F4', 'Eb4']
- Generated: Gen: ['G4', 'F4', 'Eb4', 'F4', 'C4', 'Bb4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m17: octave_error (critical)
- Generated pitches are ~+12 semitones from reference (likely octave error)
- Reference: Reference median: MIDI 61 (Db4), pitches: ['Eb4', 'Db4', 'C4']
- Generated: Generated median: MIDI 72 (C5), pitches: ['Ab4', 'Ab4', 'Db5', 'Eb5', 'C5', 'Ab4', 'Db5', 'Eb5']
- Likely stage: **melody_path (DP resolver selecting wrong octave track)**

#### m18: pitch_drift (moderate)
- Pitch drifted +4.0 semitones from reference
- Reference: Ref: ['Ab4', 'Ab4', 'Bb4', 'Ab4']
- Generated: Gen: ['C5', 'Bb4', 'Ab4', 'Db5', 'Eb5', 'C5', 'Ab4', 'Db5', 'Eb5']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m20: wrong_pitch_center (major)
- Pitch center off by -7.0 semitones
- Reference: Ref: ['Bb4', 'Bb4', 'Bb4', 'Ab4', 'Bb4', 'Ab4']
- Generated: Gen: ['Ab4', 'Eb4', 'C4', 'Bb4', 'C4']
- Likely stage: **pitch_refinement or melody_path**

#### m22: pitch_drift (moderate)
- Pitch drifted -4.5 semitones from reference
- Reference: Ref: ['Bb4', 'Bb4', 'Bb4', 'Bb4']
- Generated: Gen: ['C4', 'Bb4', 'C4', 'Bb4', 'C4', 'Eb4', 'Db4', 'Bb4', 'Ab4', 'B4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m24: wrong_pitch_center (major)
- Pitch center off by -7.0 semitones
- Reference: Ref: ['Eb5']
- Generated: Gen: ['Bb4', 'C4', 'Bb4', 'G4', 'Bb4', 'Ab4', 'Bb4', 'Ab4', 'F4']
- Likely stage: **pitch_refinement or melody_path**

#### m25: wrong_pitch_center (major)
- Pitch center off by -8.5 semitones
- Reference: Ref: ['Eb5', 'C5']
- Generated: Gen: ['F4', 'Gb4', 'F4', 'D4']
- Likely stage: **pitch_refinement or melody_path**

#### m26: pitch_drift (moderate)
- Pitch drifted -5.0 semitones from reference
- Reference: Ref: ['Ab4']
- Generated: Gen: ['Eb4', 'Ab4', 'F4', 'Eb4', 'C4', 'Eb4', 'Db4', 'A4', 'Eb4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m27: wrong_pitch_center (major)
- Pitch center off by -9.0 semitones
- Reference: Ref: ['Ab4', 'Eb5', 'C5', 'Ab4', 'Eb5', 'C5']
- Generated: Gen: ['F4', 'Eb4', 'C4', 'D4', 'C4', 'Bb4', 'Bb4']
- Likely stage: **pitch_refinement or melody_path**

#### m28: wrong_pitch_center (major)
- Pitch center off by -7.5 semitones
- Reference: Ref: ['Bb4', 'Eb5', 'C5', 'Ab4', 'Eb5', 'C5']
- Generated: Gen: ['Ab4', 'A4', 'Db4', 'Db4']
- Likely stage: **pitch_refinement or melody_path**

#### m30: pitch_drift (moderate)
- Pitch drifted -5.0 semitones from reference
- Reference: Ref: ['Ab4', 'Ab4', 'C5', 'Eb5']
- Generated: Gen: ['D4', 'C4', 'Bb4', 'Ab4', 'Bb4', 'D4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m31: wrong_pitch_center (major)
- Pitch center off by -6.0 semitones
- Reference: Ref: ['C5', 'Ab4']
- Generated: Gen: ['C4', 'Bb4', 'Eb4', 'F4']
- Likely stage: **pitch_refinement or melody_path**

#### m32: pitch_drift (moderate)
- Pitch drifted -5.0 semitones from reference
- Reference: Ref: ['Ab4', 'Bb4', 'C5', 'Bb4']
- Generated: Gen: ['F4', 'Eb4', 'F4', 'Eb4', 'Bb4', 'C4', 'Bb4', 'Db4', 'F4', 'G4']
- Likely stage: **vocal_abstraction or pitch_refinement**

#### m33: octave_error (critical)
- Generated pitches are ~+12 semitones from reference (likely octave error)
- Reference: Reference median: MIDI 68 (G4), pitches: ['Ab4', 'G4']
- Generated: Generated median: MIDI 79 (G5), pitches: ['G4', 'Bb4', 'G5', 'Bb5', 'Ab5']
- Likely stage: **melody_path (DP resolver selecting wrong octave track)**

#### m34: wrong_pitch_center (major)
- Pitch center off by -6.5 semitones
- Reference: Ref: ['Ab4', 'Eb5', 'C5', 'Ab4', 'Eb5', 'C5']
- Generated: Gen: ['Gb4', 'Ab4', 'F4', 'G4', 'F4', 'F4']
- Likely stage: **pitch_refinement or melody_path**

#### m10: note_explosion (minor)
- Generated has 7 notes vs reference's 3 (too many)
- Reference: 3 notes
- Generated: 7 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m13: over_merged (moderate)
- Generated has 1 notes vs reference's 5 (too few)
- Reference: 5 notes
- Generated: 1 notes
- Likely stage: **vocal_abstraction (over-merging) or phrase_smoothing**

#### m17: note_explosion (minor)
- Generated has 8 notes vs reference's 3 (too many)
- Reference: 3 notes
- Generated: 8 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m18: note_explosion (minor)
- Generated has 9 notes vs reference's 4 (too many)
- Reference: 4 notes
- Generated: 9 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m22: note_explosion (minor)
- Generated has 10 notes vs reference's 4 (too many)
- Reference: 4 notes
- Generated: 10 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m23: note_explosion (minor)
- Generated has 8 notes vs reference's 3 (too many)
- Reference: 3 notes
- Generated: 8 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m24: note_explosion (minor)
- Generated has 9 notes vs reference's 1 (too many)
- Reference: 1 notes
- Generated: 9 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m26: note_explosion (minor)
- Generated has 9 notes vs reference's 1 (too many)
- Reference: 1 notes
- Generated: 9 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m32: note_explosion (minor)
- Generated has 10 notes vs reference's 4 (too many)
- Reference: 4 notes
- Generated: 10 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m33: note_explosion (minor)
- Generated has 5 notes vs reference's 2 (too many)
- Reference: 2 notes
- Generated: 5 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m36: note_explosion (minor)
- Generated has 7 notes vs reference's 2 (too many)
- Reference: 2 notes
- Generated: 7 notes
- Likely stage: **vocal_abstraction (insufficient merging) or vibrato suppression**

#### m34: implausible_interval (major)
- Jump of 14 semitones: Ab5 -> Gb4
- Reference: Expected stepwise or small interval vocal motion
- Generated: Ab5 -> Gb4 (14 st)
- Likely stage: **melody_path (stream switch) or pitch_refinement (octave correction failure)**


## Aggregate Statistics

- **comparison_range**: m1-m36
- **measures_compared**: 36
- **ref_notes_in_range**: 158
- **gen_notes_in_range**: 217
- **global_pitch_offset_semitones**: 0.0
- **octave_error_measures**: 2
- **severe_mismatch_measures**: 10
- **density_missing_measures**: 0
- **density_sparse_measures**: 1
- **contour_agreement_pct**: 100.0
- **error_count_by_category**: {'note_explosion': 10, 'pitch_drift': 9, 'wrong_pitch_center': 8, 'octave_error': 2, 'over_merged': 1, 'implausible_interval': 1}
