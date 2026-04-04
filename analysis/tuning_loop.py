"""
Reference-guided parameter tuning loop.

Runs the transcription pipeline with different parameter configurations,
compares each result to the reference score, and ranks configurations
by a composite quality score.

Usage:
    python -m analysis.tuning_loop [--runs N] [--strategy random|grid]
"""

import itertools
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("tuning_loop")


# ── Tunable parameter definitions ──────────────────────────────────────

@dataclass
class TunableParam:
    name: str
    target: str            # "melody" or "repair"
    values: List[Any]
    description: str = ""


TUNABLE_PARAMS: List[TunableParam] = [
    # --- Melody-path register scoring ---
    TunableParam("register_weight", "melody",
                 [3.0, 4.0, 5.0, 6.0, 8.0],
                 "Global register bias strength"),
    TunableParam("local_register_weight", "melody",
                 [3.0, 4.0, 5.0, 6.0, 8.0],
                 "Local (windowed) register bias"),
    TunableParam("below_local_median_penalty_weight", "melody",
                 [2.0, 3.0, 4.0, 6.0],
                 "Penalty for notes below local median"),
    TunableParam("octave_flip_penalty_weight", "melody",
                 [6.0, 8.0, 10.0, 14.0],
                 "Penalty for ±12 semitone jumps"),

    # --- Candidate repair ---
    TunableParam("low_band_threshold", "repair",
                 [5, 6, 7, 8, 9],
                 "Semitones below local UQ to flag as subharmonic"),
    TunableParam("octave_recovery_confidence_scale", "repair",
                 [0.70, 0.80, 0.90, 1.00],
                 "Confidence of synthetic upper-octave candidates"),
    TunableParam("suppress_confidence_scale", "repair",
                 [0.30, 0.50, 0.70],
                 "How much to reduce suppressed subharmonic confidence"),

    # --- Post-selection cleanup / abstraction ---
    TunableParam("merge_gap_ms", "melody",
                 [120.0, 160.0, 200.0, 250.0],
                 "Max gap to merge same-pitch notes (ms)"),
    TunableParam("consolidate_max_gap_ms", "melody",
                 [150.0, 200.0, 300.0, 400.0],
                 "Max gap for consolidating repeated pitches (ms)"),
    TunableParam("phrase_max_short_note_ms", "melody",
                 [150.0, 200.0, 300.0, 400.0],
                 "Max note length to absorb in phrase smoothing (ms)"),
]


# ── Scoring function ──────────────────────────────────────────────────

def compute_score(metrics: Dict[str, Any]) -> float:
    """
    Composite quality score.  Lower is better.

    Weights emphasise correctness (wrong_pitch_center, severe mismatches)
    while rewarding contour agreement.
    """
    return (
        5.0 * metrics.get("total_errors", 0)
        + 3.0 * metrics.get("wrong_pitch_center", 0)
        + 2.0 * metrics.get("pitch_drift", 0)
        + 4.0 * metrics.get("severe_mismatch_measures", 0)
        + 6.0 * metrics.get("octave_error_measures", 0)
        - 2.0 * metrics.get("contour_agreement_pct", 0)
    )


# ── Config generation ────────────────────────────────────────────────

def _baseline_config() -> Dict[str, Any]:
    """Current production defaults for vocals mode."""
    return {
        "melody": {
            "register_weight": 5.0,
            "local_register_weight": 5.0,
            "below_local_median_penalty_weight": 4.0,
            "octave_flip_penalty_weight": 10.0,
            "merge_gap_ms": 200.0,
            "consolidate_max_gap_ms": 300.0,
            "phrase_max_short_note_ms": 300.0,
        },
        "repair": {
            "low_band_threshold": 7,
            "octave_recovery_confidence_scale": 0.90,
            "suppress_confidence_scale": 0.50,
        },
    }


def generate_random_configs(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate n random configs by sampling each parameter independently."""
    rng = random.Random(seed)
    configs: List[Dict[str, Any]] = []
    for _ in range(n):
        cfg: Dict[str, Any] = {"melody": {}, "repair": {}}
        for p in TUNABLE_PARAMS:
            cfg[p.target][p.name] = rng.choice(p.values)
        configs.append(cfg)
    return configs


def generate_one_at_a_time_configs() -> List[Dict[str, Any]]:
    """
    Vary one parameter at a time from baseline.
    Useful for sensitivity analysis.
    """
    baseline = _baseline_config()
    configs: List[Dict[str, Any]] = []
    for p in TUNABLE_PARAMS:
        current_val = baseline[p.target].get(p.name)
        for v in p.values:
            if v == current_val:
                continue
            cfg = {
                "melody": dict(baseline["melody"]),
                "repair": dict(baseline["repair"]),
            }
            cfg[p.target][p.name] = v
            configs.append(cfg)
    return configs


# ── Single trial execution ───────────────────────────────────────────

def run_trial(
    config_overrides: Dict[str, Any],
    input_path: str,
    ref_path: str,
    output_dir: str,
    max_duration_s: float = 45.0,
) -> Dict[str, Any]:
    """
    Run one pipeline trial and compare against reference.

    Returns dict with 'config', 'metrics', 'score', 'duration_s'.
    """
    from src.pipeline import run_pipeline
    from src.pipeline.utils import TranscriptionConfig
    from analysis.reference_score_parser import ReferenceScore
    from analysis.generated_score_parser import parse_debug_json
    from analysis.compare_to_reference import compare_scores

    t0 = time.time()

    tc = TranscriptionConfig(
        input_path=input_path,
        max_duration_s=max_duration_s,
        source_mode="vocals",
        instrument="violin",
        clamp_to_violin_range=True,
        use_melody_resolver=True,
        vocal_abstraction=True,
        readable_rhythm=True,
        save_debug_artifacts=True,
        output_dir=output_dir,
        melody_config_overrides=config_overrides.get("melody", {}),
        repair_config_overrides=config_overrides.get("repair", {}),
    )

    result = run_pipeline(tc)
    duration_s = time.time() - t0

    # Find debug JSON
    stem = Path(input_path).stem
    debug_json = Path(output_dir) / "debug" / f"{stem}__vocals.json"
    if not debug_json.exists():
        debug_json = Path(output_dir) / "debug" / f"{stem}.json"

    ref = ReferenceScore.load(ref_path)
    gen = parse_debug_json(str(debug_json))
    report = compare_scores(ref, gen, measure_range=(1, 36))

    stats = report.statistics
    error_cats = stats.get("error_count_by_category", {})

    metrics = {
        "total_errors": len(report.errors),
        "wrong_pitch_center": error_cats.get("wrong_pitch_center", 0),
        "pitch_drift": error_cats.get("pitch_drift", 0),
        "contour_agreement_pct": stats.get("contour_agreement_pct", 0),
        "severe_mismatch_measures": stats.get("severe_mismatch_measures", 0),
        "octave_error_measures": stats.get("octave_error_measures", 0),
        "missing_notes": error_cats.get("missing_notes", 0),
        "over_merged": error_cats.get("over_merged", 0),
        "global_pitch_offset": report.global_pitch_offset,
        "num_notes": result.num_notes,
    }

    score = compute_score(metrics)

    return {
        "config": config_overrides,
        "metrics": metrics,
        "score": score,
        "duration_s": round(duration_s, 1),
    }


# ── Full tuning loop ────────────────────────────────────────────────

def run_tuning(
    input_path: str,
    ref_path: str,
    output_dir: str,
    n_random: int = 25,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run the full tuning loop: baseline + one-at-a-time + random configs.
    Returns all results sorted by score (best first).
    """
    all_configs: List[Tuple[str, Dict[str, Any]]] = []

    # Always include baseline
    all_configs.append(("baseline", _baseline_config()))

    # One-at-a-time sensitivity sweep
    oaat = generate_one_at_a_time_configs()
    for i, cfg in enumerate(oaat):
        all_configs.append((f"oaat_{i}", cfg))

    # Random exploration
    rand_cfgs = generate_random_configs(n_random, seed=seed)
    for i, cfg in enumerate(rand_cfgs):
        all_configs.append((f"random_{i}", cfg))

    total = len(all_configs)
    print(f"Tuning loop: {total} configurations to evaluate")
    print(f"  1 baseline + {len(oaat)} one-at-a-time + {n_random} random")
    print()

    results: List[Dict[str, Any]] = []
    for idx, (label, cfg) in enumerate(all_configs):
        print(f"[{idx+1}/{total}] {label}...", end=" ", flush=True)
        try:
            trial = run_trial(cfg, input_path, ref_path, output_dir)
            trial["label"] = label
            results.append(trial)
            m = trial["metrics"]
            print(f"score={trial['score']:.1f}  "
                  f"err={m['total_errors']}  "
                  f"wpc={m['wrong_pitch_center']}  "
                  f"drift={m['pitch_drift']}  "
                  f"contour={m['contour_agreement_pct']:.0f}%  "
                  f"severe={m['severe_mismatch_measures']}  "
                  f"({trial['duration_s']}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "label": label,
                "config": cfg,
                "metrics": {},
                "score": 9999.0,
                "error": str(e),
            })

    results.sort(key=lambda r: r["score"])
    return results


# ── Report generation ────────────────────────────────────────────────

def generate_reports(results: List[Dict[str, Any]], output_dir: str):
    """Save JSON results and a Markdown summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full results
    with open(out / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Top 5
    top5 = results[:5]
    with open(out / "top_configs.json", "w", encoding="utf-8") as f:
        json.dump(top5, f, indent=2, default=str)

    # Markdown summary
    baseline = next((r for r in results if r.get("label") == "baseline"), None)
    best = results[0] if results else None

    lines = ["# Tuning Results\n"]
    lines.append(f"Total configurations tested: {len(results)}\n")

    if baseline and best:
        lines.append("## Baseline vs Best\n")
        lines.append("| Metric | Baseline | Best | Delta |")
        lines.append("|--------|----------|------|-------|")
        bm = baseline.get("metrics", {})
        xm = best.get("metrics", {})
        for key in ["total_errors", "wrong_pitch_center", "pitch_drift",
                     "contour_agreement_pct", "severe_mismatch_measures",
                     "octave_error_measures", "missing_notes", "over_merged",
                     "global_pitch_offset", "num_notes"]:
            bv = bm.get(key, "?")
            xv = xm.get(key, "?")
            if isinstance(bv, (int, float)) and isinstance(xv, (int, float)):
                delta = xv - bv
                lines.append(f"| {key} | {bv} | {xv} | {delta:+.1f} |")
            else:
                lines.append(f"| {key} | {bv} | {xv} | |")
        lines.append(f"\nBaseline score: **{baseline['score']:.1f}**")
        lines.append(f"Best score: **{best['score']:.1f}**")
        lines.append(f"Improvement: **{baseline['score'] - best['score']:.1f}** points\n")

    lines.append("## Best Configuration\n")
    if best:
        lines.append(f"Label: `{best.get('label')}`\n")
        lines.append("```json")
        lines.append(json.dumps(best.get("config", {}), indent=2))
        lines.append("```\n")

    lines.append("## Top 5 Configurations\n")
    lines.append("| Rank | Label | Score | Errors | WPC | Drift | Contour | Severe |")
    lines.append("|------|-------|-------|--------|-----|-------|---------|--------|")
    for i, r in enumerate(top5):
        m = r.get("metrics", {})
        lines.append(
            f"| {i+1} | {r.get('label','')} | {r['score']:.1f} "
            f"| {m.get('total_errors','?')} | {m.get('wrong_pitch_center','?')} "
            f"| {m.get('pitch_drift','?')} | {m.get('contour_agreement_pct','?')} "
            f"| {m.get('severe_mismatch_measures','?')} |"
        )

    # Sensitivity analysis: which parameters differ between top-5 and bottom-5?
    if len(results) >= 10:
        lines.append("\n## Parameter Sensitivity\n")
        lines.append("Parameters that differ most between top-5 and bottom-5:\n")

        top5_cfgs = [r.get("config", {}) for r in results[:5]]
        bot5_cfgs = [r.get("config", {}) for r in results[-5:]]

        for p in TUNABLE_PARAMS:
            top_vals = [c.get(p.target, {}).get(p.name) for c in top5_cfgs]
            bot_vals = [c.get(p.target, {}).get(p.name) for c in bot5_cfgs]
            top_vals = [v for v in top_vals if v is not None]
            bot_vals = [v for v in bot_vals if v is not None]
            if top_vals and bot_vals:
                top_avg = sum(float(v) for v in top_vals) / len(top_vals)
                bot_avg = sum(float(v) for v in bot_vals) / len(bot_vals)
                diff = top_avg - bot_avg
                if abs(diff) > 0.01:
                    direction = "higher in top-5" if diff > 0 else "lower in top-5"
                    lines.append(
                        f"- **{p.name}** ({p.target}): "
                        f"top-5 avg={top_avg:.2f}, bottom-5 avg={bot_avg:.2f} "
                        f"({direction}, delta={diff:+.2f})"
                    )

    lines.append("\n## All Results\n")
    lines.append("| Rank | Label | Score | Errors | WPC | Drift | Contour | Severe | Octave |")
    lines.append("|------|-------|-------|--------|-----|-------|---------|--------|--------|")
    for i, r in enumerate(results):
        m = r.get("metrics", {})
        lines.append(
            f"| {i+1} | {r.get('label','')} | {r['score']:.1f} "
            f"| {m.get('total_errors','?')} | {m.get('wrong_pitch_center','?')} "
            f"| {m.get('pitch_drift','?')} | {m.get('contour_agreement_pct','?')} "
            f"| {m.get('severe_mismatch_measures','?')} "
            f"| {m.get('octave_error_measures','?')} |"
        )

    with open(out / "tuning_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReports saved to {out}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reference-guided parameter tuning")
    parser.add_argument("--input", default=r"C:\Users\Colin\Downloads\raise_your_glass.mp3")
    parser.add_argument("--ref", default=str(Path(__file__).parent / "reference_notes.json"))
    parser.add_argument("--output-dir", default="out")
    parser.add_argument("--report-dir", default=str(Path(__file__).parent))
    parser.add_argument("--runs", type=int, default=25, help="Number of random configs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_tuning(
        input_path=args.input,
        ref_path=args.ref,
        output_dir=args.output_dir,
        n_random=args.runs,
        seed=args.seed,
    )

    generate_reports(results, args.report_dir)

    if results:
        best = results[0]
        baseline = next((r for r in results if r.get("label") == "baseline"), None)
        print(f"\nBest config: {best.get('label')} (score={best['score']:.1f})")
        if baseline:
            print(f"Baseline score: {baseline['score']:.1f}")
            print(f"Improvement: {baseline['score'] - best['score']:.1f} points")


if __name__ == "__main__":
    main()
