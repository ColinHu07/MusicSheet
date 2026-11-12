import json
import subprocess
import sys
from pathlib import Path

import pytest

from musicsheet.evaluation import compare_notes, evaluate_notes


def note(start, end, midi=60):
    return {"start_s": start, "end_s": end, "midi": midi}


def first(reference, predicted, **kwargs):
    return evaluate_notes(reference, predicted, tolerances=(.1,), **kwargs)["sensitivity"][0]


def test_duplicate_predictions_cannot_claim_the_same_reference_twice():
    result = first([note(1, 2)], [note(1, 2), note(1.02, 2)])
    assert result["exact_pitch_onset"]["matched"] == 1
    assert result["exact_pitch_onset"]["precision"] == .5
    assert result["exact_pitch_onset"]["unmatched_predictions"] == 1


def test_competing_near_boundary_matches_maximize_cardinality_before_distance():
    # Greedily assigning ref .10 -> pred .19 would strand ref .28. The optimal
    # assignment must reserve .19 for .28 and use .01 for .10.
    result = first([note(.10, .2), note(.28, .4)], [note(.19, .27), note(.01, .09)])
    pairs = result["exact_pitch_onset"]["pairs"]
    assert pairs == [{"reference_index": 0, "prediction_index": 1},
                     {"reference_index": 1, "prediction_index": 0}]


def test_octave_and_other_pitch_errors_are_separate_from_missed_attack_metrics():
    result = first([note(1, 1.4, 60), note(2, 2.4, 62), note(3, 3.4, 64)],
                   [note(1, 1.4, 72), note(2, 2.4, 63), note(3, 3.4, 64)])
    assert result["exact_pitch_onset"]["matched"] == 1
    assert result["pitch_class_onset"]["matched"] == 2
    diagnostics = result["pitch_diagnostics_on_time_only_matches"]
    assert diagnostics["time_matched"] == 3
    assert diagnostics["octave_errors"] == 1
    assert diagnostics["other_pitch_errors"] == 1
    assert diagnostics["exact_pitch"] == 1


def test_contiguous_identical_pitches_are_repeated_attacks_even_without_silence():
    reference = [note(1, 1.5), note(1.5, 2), note(2, 2.5)]
    result = first(reference, [note(1, 2.5)])
    repeated = result["repeated_same_pitch_attacks"]
    assert repeated["reference"] == 2
    assert repeated["missed_exact_pitch"] == 2
    assert repeated["exact_pitch_recall"] == 0


def test_unsorted_reference_events_still_find_adjacent_repeat_attacks():
    result = first([note(2, 2.5), note(1, 1.5), note(1.5, 2)], [note(1, 2.5)])
    assert result["repeated_same_pitch_attacks"]["reference_indices"] == [0, 2]


def test_sustain_duration_alone_does_not_create_melisma_or_false_split_labels():
    result = first([note(1, 4)], [note(1, 2), note(2, 4)])
    assert result["melisma"]["available"] is False
    assert result["melisma"]["sequence_pitch_recall"] is None
    assert result["sustained_false_splits"]["available"] is False
    assert result["sustained_false_splits"]["false_extra_attacks"] is None


def test_unverified_annotations_are_rejected_with_reason():
    annotations = {"regions": [{"kind": "melisma", "verified": False,
                                "start_s": 1, "end_s": 3, "reference_indices": [0, 1]}]}
    result = evaluate_notes([note(1, 2), note(2, 3, 62)], [], annotations=annotations)
    assert result["annotation_rejections"][0]["reason"] == "human_pattern_confirmation_and_provenance_required"
    assert result["sensitivity"][0]["melisma"]["available"] is False


def test_human_melisma_pattern_can_be_evaluated_without_claiming_verified_timing():
    reference = [note(1, 1.5, 68), note(1.5, 2, 66), note(2, 3, 64)]
    annotations = {"reference_provenance": "human pitch pattern / approximate score-aligned times",
                   "regions": [{"id": "held_off", "kind": "melisma", "verified": True,
                                "provenance": "i_do_review_008.json", "timing_verified": False,
                                "start_s": 1, "end_s": 3, "reference_indices": [0, 1, 2]}]}
    # Correct order and all notes, but materially different internal onset times.
    result = first(reference, [note(1, 1.8, 68), note(1.8, 2.3, 66), note(2.3, 3, 64)], annotations=annotations)
    melisma = result["melisma"]
    assert melisma["available"] is True
    assert melisma["sequence_pitch_recall"] == 1
    assert melisma["onset_pitch_recall"] == pytest.approx(1/3)
    assert melisma["all_internal_times_human_verified"] is False
    assert melisma["regions"][0]["provenance"] == "i_do_review_008.json"


def test_missing_run_note_and_octave_shift_are_not_hidden_by_pitch_class_score():
    reference = [note(1, 1.5, 68), note(1.5, 2, 66), note(2, 3, 64)]
    annotations = {"regions": [{"kind": "melisma", "verified": True, "provenance": "review",
                                "start_s": 1, "end_s": 3, "reference_indices": [0, 1, 2]}]}
    result = first(reference, [note(1, 2, 80), note(2, 3, 76)], annotations=annotations)["melisma"]
    assert result["sequence_pitch_recall"] == 0
    assert result["sequence_pitch_class_recall"] == pytest.approx(2/3)


def test_confirmed_sustain_detects_extra_attacks_but_not_one_late_onset():
    annotations = {"regions": [{"kind": "sustain", "verified": True, "provenance": "review",
                                "start_s": 1, "end_s": 3, "reference_indices": [0]}]}
    split = first([note(1, 3)], [note(1, 1.5), note(1.5, 2), note(2, 3)], annotations=annotations)
    late = first([note(1, 3)], [note(1.5, 3)], annotations=annotations)
    assert split["sustained_false_splits"]["false_extra_attacks"] == 2
    assert late["sustained_false_splits"]["false_extra_attacks"] == 0


def test_three_hums_cannot_be_mislabeled_as_one_sustain():
    reference = [note(1, 1.5), note(1.5, 2), note(2, 2.5)]
    annotations = {"regions": [{"kind": "sustain", "verified": True, "provenance": "review",
                                "start_s": 1, "end_s": 3, "reference_indices": [0, 1, 2]}]}
    result = evaluate_notes(reference, reference, annotations=annotations)
    assert result["annotation_rejections"][0]["reason"] == "sustain_must_describe_one_note_not_a_phrase"


def test_duplicate_sustain_annotations_cannot_double_count_errors():
    region = {"kind": "sustain", "verified": True, "provenance": "review",
              "start_s": 1, "end_s": 3, "reference_indices": [0]}
    result = evaluate_notes([note(1, 3)], [note(1, 2), note(2, 3)],
                            annotations={"regions": [region, dict(region)]})
    assert result["sensitivity"][0]["sustained_false_splits"]["false_extra_attacks"] == 1
    assert result["annotation_rejections"][0]["reason"] == "overlapping_sustain_regions_would_double_count_attacks"


def test_sensitivity_and_release_metrics_are_distinct():
    result = evaluate_notes([note(1, 2)], [note(1.08, 1.3)], tolerances=(.05, .1, .15))
    assert [row["exact_pitch_onset"]["matched"] for row in result["sensitivity"]] == [0, 1, 1]
    assert result["sensitivity"][1]["exact_pitch_onset_offset"]["matched"] == 0


def test_empty_reference_does_not_report_perfect_accuracy():
    result = first([], [])
    assert result["exact_pitch_onset"]["f1"] is None
    assert result["repeated_same_pitch_attacks"]["exact_pitch_recall"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [note(2, 1), note(float("nan"), 2), note(-1, 1), note(1, 2, 60.3)])
def test_invalid_events_raise_instead_of_silently_changing_reference(bad):
    with pytest.raises(ValueError):
        evaluate_notes([bad], [])


def test_comparison_deltas_and_cli_use_the_same_reference(tmp_path):
    reference = [note(1, 1.5), note(1.5, 2)]
    baseline = [note(1, 2)]
    result = compare_notes(reference, baseline, reference, tolerances=(.1,))
    assert result["deltas"][0]["repeated_exact_pitch_recall"] == 1
    paths = [tmp_path / f"{name}.json" for name in ("reference", "baseline", "candidate")]
    for path, data in zip(paths, (reference, {"notes": baseline}, {"events": reference})):
        path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "comparison.json"
    script = Path(__file__).resolve().parents[1] / "musicsheet/evaluation.py"
    subprocess.run([sys.executable, str(script), "--reference", str(paths[0]), "--baseline", str(paths[1]),
                    "--candidate", str(paths[2]), "--tolerances", ".1", "--output", str(output)], check=True)
    assert json.loads(output.read_text(encoding="utf-8")) == result
