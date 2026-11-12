"""Evaluate performed note events without notation or beat quantization.

This module does not load model weights, audio, lyrics, or score references. Its
results are only as authoritative as the supplied reference/provenance. In
particular, score-aligned events with approximate times are silver labels, not
independent listening verification.

Canonical events: {"start_s": float, "end_s": float, "midi": int}.
Optional annotations: {"reference_provenance": ..., "regions": [
    {"id": "...", "kind": "melisma" | "sustain", "verified": true,
     "provenance": "human-review-file", "timing_verified": false,
     "start_s": ..., "end_s": ..., "reference_indices": [0, 1, 2]}]}.
Here verified means the human-confirmed *pattern*, not accurate frame timing.
Sustain regions must name exactly one sustained note; a three-syllable humming
phrase must not be marked as one sustain. Melisma recall is also reported as
ordered pitch-sequence recall, which does not assume exact internal onsets.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _events(events: list[dict], name: str) -> list[dict]:
    result = []
    for i, event in enumerate(events):
        try:
            start, end, midi = (float(event[k]) for k in ("start_s", "end_s", "midi"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{i}] needs start_s, end_s, and midi") from exc
        if not all(math.isfinite(v) for v in (start, end, midi)):
            raise ValueError(f"{name}[{i}] contains a non-finite value")
        if start < 0 or end <= start or midi != int(midi) or not 0 <= midi <= 127:
            raise ValueError(f"{name}[{i}] needs 0 <= start < end and integer MIDI 0..127")
        result.append(dict(event, start_s=start, end_s=end, midi=int(midi)))
    return result


def _match(reference: list[dict], predicted: list[dict], tolerance: float,
           pitch_mode: str = "exact", offsets: bool = False) -> list[tuple[int, int]]:
    """Maximum-cardinality, then minimum-onset-error one-to-one assignment."""
    n, m = len(reference), len(predicted)
    if not n or not m:
        return []
    # A lost match costs more than every possible timing improvement together.
    # Per-reference dummy columns also let all events remain unmatched.
    penalty = float(n + m + 1)
    cost = np.full((n, m + n), penalty)
    cost[:, :m] = penalty * 3
    valid = np.zeros((n, m), dtype=bool)
    for i, ref in enumerate(reference):
        for j, pred in enumerate(predicted):
            delta = abs(ref["start_s"] - pred["start_s"])
            pitch_delta = pred["midi"] - ref["midi"]
            pitch_ok = (pitch_mode == "any" or pitch_delta == 0 or
                        (pitch_mode == "class" and pitch_delta % 12 == 0))
            release_ok = abs(ref["end_s"] - pred["end_s"]) <= max(
                .10, .20 * (ref["end_s"] - ref["start_s"]))
            if delta <= tolerance + 1e-10 and pitch_ok and (not offsets or release_ok):
                valid[i, j] = True
                cost[i, j] = delta / (tolerance + 1)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if j < m and valid[i, j]]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _scores(reference: list[dict], predicted: list[dict], pairs: list[tuple[int, int]]) -> dict:
    n, m, k = len(reference), len(predicted), len(pairs)
    onset_errors = [predicted[j]["start_s"] - reference[i]["start_s"] for i, j in pairs]
    offset_errors = [predicted[j]["end_s"] - reference[i]["end_s"] for i, j in pairs]
    return {
        "reference": n, "predicted": m, "matched": k,
        "unmatched_reference": n-k, "unmatched_predictions": m-k,
        "precision": _ratio(k, m), "recall": _ratio(k, n),
        "f1": _ratio(2*k, n+m),
        "mean_absolute_onset_error_s": float(np.mean(np.abs(onset_errors))) if k else None,
        "mean_absolute_offset_error_s": float(np.mean(np.abs(offset_errors))) if k else None,
        "pairs": [{"reference_index": i, "prediction_index": j} for i, j in pairs],
    }


def _repeat_indices(reference: list[dict], max_gap_s: float) -> list[int]:
    ordered = sorted(range(len(reference)), key=lambda i: (reference[i]["start_s"], i))
    return [i for previous, i in zip(ordered, ordered[1:])
            if reference[previous]["midi"] == reference[i]["midi"]
            and reference[i]["start_s"] - reference[previous]["end_s"] <= max_gap_s]


def _lcs(a: list[int], b: list[int], pitch_class: bool = False) -> int:
    """Ordered, one-to-one pitch matches without invented internal boundaries."""
    row = [0] * (len(b) + 1)
    for left in a:
        previous = row[:]
        for j, right in enumerate(b, 1):
            equal = (left-right) % 12 == 0 if pitch_class else left == right
            row[j] = previous[j-1] + 1 if equal else max(previous[j], row[j-1])
    return row[-1]


def _regions(annotations: dict | None, reference: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for index, region in enumerate((annotations or {}).get("regions", [])):
        ident = region.get("id", str(index))
        reason = None
        if region.get("kind") not in ("sustain", "melisma"):
            reason = "unsupported_annotation_kind"
        elif region.get("verified") is not True or not region.get("provenance"):
            reason = "human_pattern_confirmation_and_provenance_required"
        else:
            indices = region.get("reference_indices", [])
            if (not isinstance(indices, (list, tuple)) or not indices or
                    any(not isinstance(i, int) or isinstance(i, bool) or
                        not 0 <= i < len(reference) for i in indices) or
                    len(set(indices)) != len(indices)):
                reason = "valid_unique_reference_indices_required"
            elif region["kind"] == "sustain" and len(indices) != 1:
                reason = "sustain_must_describe_one_note_not_a_phrase"
            elif region["kind"] == "melisma" and len(indices) < 2:
                reason = "melisma_requires_multiple_annotated_notes"
            else:
                try:
                    start, end = float(region["start_s"]), float(region["end_s"])
                    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                        reason = "invalid_region_times"
                    elif any(not start-1e-8 <= reference[i]["start_s"] < end+1e-8 for i in indices):
                        reason = "reference_onsets_outside_region"
                except (KeyError, TypeError, ValueError):
                    reason = "invalid_region_times"
        if not reason and region["kind"] == "sustain" and any(
                previous["kind"] == "sustain" and
                max(start, previous["start_s"]) < min(end, previous["end_s"])
                for previous in accepted):
            reason = "overlapping_sustain_regions_would_double_count_attacks"
        if reason:
            rejected.append({"id": ident, "reason": reason})
        else:
            accepted.append(dict(region, id=ident, start_s=start, end_s=end,
                                 timing_verified=region.get("timing_verified") is True))
    return accepted, rejected


def _annotated_metrics(reference: list[dict], predicted: list[dict], regions: list[dict],
                       exact: list[tuple[int, int]], tolerance: float) -> dict:
    exact_refs = {i for i, _ in exact}
    melismas, sustains = [], []
    for region in regions:
        indices = sorted(region["reference_indices"], key=lambda i: reference[i]["start_s"])
        # Keep the nominal region fixed at every tolerance. Widening scope with
        # tolerance would itself introduce attacks and confound sensitivity.
        candidates = sorted((j for j, p in enumerate(predicted)
                             if region["start_s"] <= p["start_s"] < region["end_s"]),
                            key=lambda j: predicted[j]["start_s"])
        info = {k: region[k] for k in ("id", "provenance", "timing_verified", "start_s", "end_s")}
        info.update(reference_indices=indices, prediction_indices=candidates)
        if region["kind"] == "melisma":
            pitches = [reference[i]["midi"] for i in indices]
            guesses = [predicted[j]["midi"] for j in candidates]
            info.update(reference=len(indices), onset_pitch_matched=len(set(indices) & exact_refs),
                        sequence_pitch_matched=_lcs(pitches, guesses),
                        sequence_pitch_class_matched=_lcs(pitches, guesses, True))
            melismas.append(info)
        else:
            # Confirmed one-note regions cannot justify multiple attacks. This
            # count deliberately does not call a single late note a false split.
            # Predicted notes starting just before the region may carry its
            # legitimate first attack, so count that one separately if present.
            carry = any(region["start_s"]-tolerance <= p["start_s"] < region["start_s"]
                        and p["end_s"] > region["start_s"] for p in predicted)
            info.update(predicted_attacks=len(candidates) + int(carry),
                        false_extra_attacks=max(0, len(candidates) + int(carry) - 1))
            sustains.append(info)
    # Aggregate unique reference events for onset recall. Regions should normally
    # be disjoint, but overlapping annotations must not inflate that denominator.
    melisma_indices = {i for r in melismas for i in r["reference_indices"]}
    ordered_melismas = sorted(melismas, key=lambda r: r["start_s"])
    nonoverlap = (len(melisma_indices) == sum(r["reference"] for r in melismas) and
                  all(left["end_s"] <= right["start_s"] for left, right in
                      zip(ordered_melismas, ordered_melismas[1:])))
    total_sequence = sum(r["sequence_pitch_matched"] for r in melismas)
    total_class = sum(r["sequence_pitch_class_matched"] for r in melismas)
    return {
        "melisma": {
            "available": bool(melismas),
            "unavailable_reason": None if melismas else "no_verified_melisma_patterns_with_provenance",
            "reference": len(melisma_indices),
            "onset_pitch_recall": _ratio(len(melisma_indices & exact_refs), len(melisma_indices)),
            "sequence_pitch_recall": _ratio(total_sequence, len(melisma_indices)) if nonoverlap else None,
            "sequence_pitch_class_recall": _ratio(total_class, len(melisma_indices)) if nonoverlap else None,
            "sequence_aggregate_unavailable_reason": None if nonoverlap else "overlapping_regions",
            "all_internal_times_human_verified": all(r["timing_verified"] for r in melismas) if melismas else None,
            "regions": melismas,
        },
        "sustained_false_splits": {
            "available": bool(sustains),
            "unavailable_reason": None if sustains else "no_verified_single_note_sustains_with_provenance",
            "confirmed_regions": len(sustains),
            "false_extra_attacks": sum(r["false_extra_attacks"] for r in sustains) if sustains else None,
            "regions_with_false_splits": sum(r["false_extra_attacks"] > 0 for r in sustains) if sustains else None,
            "regions": sustains,
        },
    }


def evaluate_notes(reference: list[dict], predicted: list[dict], *,
                   tolerances: tuple[float, ...] = (.05, .10, .15),
                   annotations: dict | None = None, repeat_max_gap_s: float = .50) -> dict[str, Any]:
    """Return reproducible diagnostics, with no training or quantization side effects.

    References and predictions must cover the same scoring scope. Unknown/rap/
    excluded regions must be filtered by the caller for *both* lists. Never use
    a full-song prediction against a partial reference and call it precision.
    MIDI values must use the same sounding/arrangement register convention.
    """
    reference, predicted = _events(reference, "reference"), _events(predicted, "predicted")
    tolerances = tuple(sorted(set(float(t) for t in tolerances)))
    if not tolerances or any(not math.isfinite(t) or t < 0 for t in tolerances):
        raise ValueError("tolerances must be finite non-negative seconds")
    if not math.isfinite(repeat_max_gap_s) or repeat_max_gap_s < 0:
        raise ValueError("repeat_max_gap_s must be finite and non-negative")
    regions, rejected = _regions(annotations, reference)
    repeats = set(_repeat_indices(reference, repeat_max_gap_s))
    results = []
    for tolerance in tolerances:
        exact = _match(reference, predicted, tolerance)
        by_class = _match(reference, predicted, tolerance, "class")
        timed = _match(reference, predicted, tolerance, "any")
        offsets = _match(reference, predicted, tolerance, offsets=True)
        exact_refs, class_refs = {i for i, _ in exact}, {i for i, _ in by_class}
        differences = [predicted[j]["midi"]-reference[i]["midi"] for i, j in timed]
        octaves = sum(delta != 0 and delta % 12 == 0 for delta in differences)
        other = sum(delta % 12 != 0 for delta in differences)
        item = {
            "onset_tolerance_s": tolerance,
            "exact_pitch_onset": _scores(reference, predicted, exact),
            "pitch_class_onset": _scores(reference, predicted, by_class),
            "exact_pitch_onset_offset": _scores(reference, predicted, offsets),
            "pitch_diagnostics_on_time_only_matches": {
                "matching_basis": "one_to_one_onset_only_minimum_time_error; correspondence_can_be_ambiguous",
                "time_matched": len(timed), "exact_pitch": differences.count(0),
                "octave_errors": octaves, "other_pitch_errors": other,
                "octave_error_rate": _ratio(octaves, len(timed)),
                "other_pitch_error_rate": _ratio(other, len(timed)),
                "semitone_error_histogram": dict(sorted(Counter(str(d) for d in differences).items())),
                "pairs": [{"reference_index": i, "prediction_index": j,
                           "semitone_error": predicted[j]["midi"]-reference[i]["midi"]} for i, j in timed],
            },
            "repeated_same_pitch_attacks": {
                "basis": "consecutive_reference_events_at_identical_pitch_not_phoneme_ground_truth",
                "max_gap_s": repeat_max_gap_s, "reference": len(repeats),
                "exact_pitch_matched": len(repeats & exact_refs),
                "missed_exact_pitch": len(repeats - exact_refs),
                "exact_pitch_recall": _ratio(len(repeats & exact_refs), len(repeats)),
                "pitch_class_matched": len(repeats & class_refs),
                "pitch_class_recall": _ratio(len(repeats & class_refs), len(repeats)),
                "reference_indices": sorted(repeats),
            },
        }
        item.update(_annotated_metrics(reference, predicted, regions, exact, tolerance))
        results.append(item)
    return {
        "schema_version": 1,
        "reference_provenance": (annotations or {}).get("reference_provenance", "unspecified_not_verified_ground_truth"),
        "scope_policy": "caller_supplies_common_scoring_scope_and_same_pitch_register_convention",
        "timing_policy": "fixed_nominal_onsets; tolerance_sensitivity_does_not_validate_reference_times",
        "offset_policy": "max_100ms_or_20_percent_of_reference_duration",
        "annotation_rejections": rejected,
        "sensitivity": results,
    }


def compare_notes(reference: list[dict], baseline: list[dict], candidate: list[dict], **kwargs: Any) -> dict:
    """Evaluate both systems against identical references and tolerances."""
    before = evaluate_notes(reference, baseline, **kwargs)
    after = evaluate_notes(reference, candidate, **kwargs)
    deltas = []
    for b, c in zip(before["sensitivity"], after["sensitivity"]):
        def difference(x: Any, y: Any) -> Any:
            return y-x if x is not None and y is not None else None
        deltas.append({
            "onset_tolerance_s": b["onset_tolerance_s"],
            "exact_pitch_onset_f1": difference(b["exact_pitch_onset"]["f1"], c["exact_pitch_onset"]["f1"]),
            "pitch_class_onset_f1": difference(b["pitch_class_onset"]["f1"], c["pitch_class_onset"]["f1"]),
            "repeated_exact_pitch_recall": difference(b["repeated_same_pitch_attacks"]["exact_pitch_recall"], c["repeated_same_pitch_attacks"]["exact_pitch_recall"]),
            "melisma_sequence_pitch_recall": difference(b["melisma"]["sequence_pitch_recall"], c["melisma"]["sequence_pitch_recall"]),
            "sustained_false_extra_attacks": difference(b["sustained_false_splits"]["false_extra_attacks"], c["sustained_false_splits"]["false_extra_attacks"]),
        })
    return {"schema_version": 1, "baseline": before, "candidate": after,
            "delta_policy": "candidate_minus_baseline; positive_false_extra_attacks_is_worse",
            "deltas": deltas}


def _load_notes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("notes", data.get("events"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list or an object containing notes/events")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--tolerances", default="0.05,0.10,0.15")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8-sig")) if args.annotations else None
    result = compare_notes(_load_notes(args.reference), _load_notes(args.baseline), _load_notes(args.candidate),
                           tolerances=tuple(float(t) for t in args.tolerances.split(",")), annotations=annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
