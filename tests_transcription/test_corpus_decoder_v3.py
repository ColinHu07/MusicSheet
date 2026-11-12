"""Synthetic audio-decoder invariants; no corpus audio, labels or checkpoints.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -p test_corpus_decoder_v3.py -v
"""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from musicsheet.decoder import decode, HOP, MIN_MIDI


def state(midi):
    return midi - MIN_MIDI + 1


def fixture(length=120, midi=60, confidence=.92):
    pp = np.full((61, length), (1 - confidence) / 60)
    pp[state(midi)] = confidence
    return pp, np.zeros((2, length))


def region(pp, a, b, alternatives):
    """Assign exact distributions while leaving the unlisted classes nonzero."""
    pp[:, a:b] = (1 - sum(alternatives.values())) / (61 - len(alternatives))
    for midi, probability in alternatives.items():
        pp[0 if midi is None else state(midi), a:b] = probability


def f0(length, midi=60, confidence=.9):
    return dict(rmvpe_midi=np.full(length, float(midi)),
                rmvpe_voiced=np.ones(length), rmvpe_confidence=np.full(length, confidence),
                vocal_fullmix_rms_ratio=np.full(length, .5))


class DecoderV3Audit(unittest.TestCase):
    def test_independent_repeats_do_not_require_silence(self):
        pp, bb = fixture()
        bb[0, [40, 80]] = .94
        notes = decode(pp, bb)
        self.assertEqual([n['midi'] for n in notes], [60, 60, 60])
        self.assertEqual([n['start_s'] for n in notes], [0., .8, 1.6])
        self.assertEqual([n['repeated_attack'] for n in notes], [False, True, True])
        self.assertEqual(notes[-1]['end_s'], 2.4)

    def test_held_vibrato_is_not_a_series_of_notes(self):
        pp, bb = fixture()
        # Periodic semitone ambiguity and low attack response during vibrato.
        for a in range(8, 110, 10):
            region(pp, a, a + 2, {60: .42, 61: .50})
            bb[0, a] = .3
        notes = decode(pp, bb)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['midi'], 60)
        self.assertEqual(notes[0]['end_s'], 2.4)

    def test_real_sustained_octave_leap_without_attack_survives(self):
        pp, bb = fixture()
        region(pp, 60, 120, {60: .10, 72: .86})
        notes = decode(pp, bb)
        self.assertEqual([n['midi'] for n in notes], [60, 72])
        self.assertAlmostEqual(notes[1]['start_s'], 1.2)
        self.assertIn('stable_pitch_transition', notes[1]['decoder_reasons'])

    def test_brief_unsupported_octave_glitch_is_bridged(self):
        pp, bb = fixture()
        region(pp, 50, 56, {60: .40, 72: .56})
        notes, audit = decode(pp, bb, return_audit=True)
        self.assertEqual([n['midi'] for n in notes], [60])
        self.assertEqual(notes[0]['end_s'], 2.4)
        self.assertEqual(audit['summary']['octave_changed_frames'], 6)
        self.assertTrue(audit['changes'])

    def test_legato_step_does_not_require_onset(self):
        pp, bb = fixture()
        region(pp, 60, 120, {60: .12, 62: .84})
        notes = decode(pp, bb)
        self.assertEqual([n['midi'] for n in notes], [60, 62])
        self.assertEqual(notes[0]['end_s'], notes[1]['start_s'])

    def test_short_strong_note_is_not_removed_by_duration_floor(self):
        pp, bb = fixture()
        region(pp, 50, 52, {60: .01, 64: .98})
        bb[0, 50] = .97
        evidence = f0(120)
        evidence['rmvpe_midi'][50:52] = 64
        evidence['onset_flux'] = np.full(120, .1)
        evidence['onset_flux'][50] = 1.
        notes = decode(pp, bb, vocal=evidence)
        self.assertEqual([n['midi'] for n in notes], [60, 64, 60])
        short = notes[1]
        self.assertAlmostEqual(short['end_s'] - short['start_s'], .04)
        self.assertIn('short_onset_acoustic_attack_and_pitch', short['decoder_reasons'])

    def test_short_strong_octave_ornament_survives(self):
        pp, bb = fixture()
        region(pp, 50, 53, {60: .005, 72: .99})
        bb[0, 50] = .96
        evidence = f0(120)
        evidence['rmvpe_midi'][50:53] = 72
        evidence['onset_flux'] = np.full(120, .1)
        evidence['onset_flux'][50] = 1.
        notes = decode(pp, bb, vocal=evidence)
        self.assertEqual([n['midi'] for n in notes], [60, 72, 60])
        self.assertAlmostEqual(notes[1]['end_s'] - notes[1]['start_s'], .06)

    def test_missing_acoustics_does_not_remove_uncertain_sustained_note(self):
        pp, bb = fixture()
        region(pp, 0, 120, {60: .22, 64: .20, 67: .18, None: .15})
        notes = decode(pp, None)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['midi'], 60)
        self.assertTrue(notes[0]['extra_review'])
        self.assertIn('sustained_uncertain_pitch', notes[0]['decoder_reasons'])

    def test_missing_frames_preserve_context_without_inventing_rest(self):
        pp, bb = fixture()
        pp[:, 40:50] = np.nan
        notes, audit = decode(pp, bb, return_audit=True)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]['end_s'], 2.4)
        self.assertTrue(notes[0]['extra_review'])
        self.assertEqual(audit['input_quality']['missing_probability_frames'], 10)

    def test_unsupported_short_island_is_rejected_with_reason(self):
        pp, bb = fixture()
        region(pp, 0, 120, {None: .9})
        region(pp, 50, 56, {60: .23, 64: .21, 67: .20, None: .01})
        notes, audit = decode(pp, bb, transition=.2, return_audit=True)
        self.assertEqual(notes, [])
        rejected = [e for e in audit['events'] if e['status'] == 'rejected']
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['midi'], 60)
        self.assertEqual(audit['summary']['removed_events'], 1)

    def test_repeats_survive_nearby_octave_glitch(self):
        pp, bb = fixture()
        region(pp, 45, 51, {60: .40, 72: .56})
        bb[0, [40, 80]] = .94
        notes = decode(pp, bb)
        self.assertEqual([n['midi'] for n in notes], [60, 60, 60])
        self.assertEqual([n['start_s'] for n in notes], [0., .8, 1.6])

    def test_fast_repeated_sixteenths_can_be_distinct(self):
        pp, bb = fixture(20)
        bb[0, [4, 8, 12, 16]] = .95
        notes = decode(pp, bb)
        self.assertEqual(len(notes), 5)
        self.assertTrue(all(abs(n['end_s'] - n['start_s'] - .08) < 1e-9 for n in notes))

    def test_soft_f0_register_evidence_breaks_only_ambiguous_octave(self):
        pp, bb = fixture()
        region(pp, 0, 120, {60: .44, 72: .50})
        before = decode(pp, bb)
        after, audit = decode(pp, bb, vocal=f0(120), return_audit=True)
        self.assertEqual(before[0]['midi'], 72)
        self.assertEqual(after[0]['midi'], 60)
        self.assertEqual(audit['input_quality']['reliable_f0_frames'], 120)
        self.assertEqual(audit['summary']['octave_changed_frames'], 120)

    def test_wrong_f0_cannot_veto_strong_true_octave_leap(self):
        pp, bb = fixture()
        region(pp, 60, 120, {60: .07, 72: .90})
        notes = decode(pp, bb, vocal=f0(120))
        self.assertEqual([n['midi'] for n in notes], [60, 72])

    def test_unreliable_or_missing_f0_confidence_has_no_register_vote(self):
        pp, bb = fixture()
        region(pp, 0, 120, {60: .44, 72: .50})
        unreliable = f0(120, confidence=.4)
        missing = f0(120)
        missing.pop('rmvpe_confidence')
        for evidence in (unreliable, missing):
            with self.subTest(evidence=list(evidence)):
                notes, audit = decode(pp, bb, vocal=evidence, return_audit=True)
                self.assertEqual(notes[0]['midi'], 72)
                self.assertEqual(audit['input_quality']['reliable_f0_frames'], 0)

    def test_low_vocal_energy_f0_is_not_trusted(self):
        pp, bb = fixture()
        region(pp, 0, 120, {60: .44, 72: .50})
        evidence = f0(120)
        evidence['vocal_fullmix_rms_ratio'][:] = .01
        notes = decode(pp, bb, vocal=evidence)
        self.assertEqual(notes[0]['midi'], 72)

    def test_range_is_not_folded_to_violin(self):
        pp, bb = fixture(midi=40)
        self.assertEqual(decode(pp, bb)[0]['midi'], 40)

    def test_real_rest_is_not_bridged(self):
        pp, bb = fixture()
        region(pp, 50, 56, {None: .97})
        bb[1, 50] = .95
        notes = decode(pp, bb)
        self.assertEqual(len(notes), 2)
        self.assertAlmostEqual(notes[1]['start_s'] - notes[0]['end_s'], .12)

    def test_no_mutation_and_json_audit(self):
        pp, bb = fixture()
        evidence = f0(120)
        originals = [pp.copy(), bb.copy(), evidence['rmvpe_midi'].copy()]
        notes, audit = decode(pp, bb, vocal=evidence, return_audit=True)
        np.testing.assert_array_equal(pp, originals[0])
        np.testing.assert_array_equal(bb, originals[1])
        np.testing.assert_array_equal(evidence['rmvpe_midi'], originals[2])
        json.dumps(dict(notes=notes, audit=audit), allow_nan=False)

    def test_empty_and_all_missing_inputs(self):
        self.assertEqual(decode(np.zeros((61, 0)), np.zeros((2, 0))), [])
        self.assertEqual(decode(np.full((61, 30), np.nan), None), [])

    def test_bad_shapes_and_negative_probabilities_fail_clearly(self):
        pp, bb = fixture()
        for bad in (pp.T, pp[:60]):
            with self.assertRaises(ValueError):
                decode(bad, bb)
        with self.assertRaises(ValueError):
            decode(-pp, bb)
        with self.assertRaises(ValueError):
            decode(pp, bb[:, :-1])
        with self.assertRaises(ValueError):
            decode(pp, bb, vocal=f0(5))

    def test_short_posterior_and_model_onset_without_acoustic_support_are_rejected(self):
        pp, bb = fixture()
        region(pp, 50, 52, {60: .01, 64: .98})
        bb[0, 50] = .97
        notes, audit = decode(pp, bb, return_audit=True)
        self.assertNotIn(64, [n['midi'] for n in notes])
        rejected = [e for e in audit['events'] if 'unsupported_short_candidate' in e.get('reasons', [])]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]['midi'], 64)

    def test_short_note_needs_pitch_corroboration_besides_acoustic_attack(self):
        pp, bb = fixture()
        region(pp, 50, 52, {60: .01, 64: .98})
        bb[0, 50] = .97
        evidence = {'onset_flux': np.full(120, .1)}
        evidence['onset_flux'][50] = 1.
        notes = decode(pp, bb, vocal=evidence)
        self.assertNotIn(64, [n['midi'] for n in notes])

    def test_near_edge_model_peak_does_not_create_40ms_repeat(self):
        pp, bb = fixture()
        bb[0, 2] = .97
        notes, audit = decode(pp, bb, vocal=f0(120), return_audit=True)
        self.assertEqual(len(notes), 1)
        self.assertEqual(audit['summary']['rejected_repeat_splits'], 1)

    def test_near_edge_repeat_requires_independent_transient(self):
        pp, bb = fixture(confidence=.98)
        bb[0, 2] = .97
        evidence = f0(120)
        evidence['onset_flux'] = np.full(120, .1)
        evidence['onset_flux'][2] = 1.
        notes = decode(pp, bb, vocal=evidence)
        self.assertEqual([n['midi'] for n in notes], [60, 60])
        self.assertEqual(notes[1]['start_s'], .04)

    def test_sustained_loud_flux_is_not_an_independent_attack(self):
        pp, bb = fixture()
        bb[0, 2] = .97
        evidence = {'onset_flux': np.full(120, 1.)}
        notes = decode(pp, bb, vocal=evidence)
        self.assertEqual(len(notes), 1)

    def test_120ms_sixteenths_are_not_suppressed(self):
        pp, bb = fixture(30)
        bb[0, [6, 12, 18, 24]] = .94
        notes = decode(pp, bb)
        self.assertEqual(len(notes), 5)
        self.assertTrue(all(abs(n['end_s'] - n['start_s'] - .12) < 1e-8 for n in notes))


if __name__ == '__main__':
    unittest.main()
