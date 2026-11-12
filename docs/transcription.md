# Transcription engine setup

Python 3.10+ is required; the portable package is checked with synthetic tests
and private compatibility runs. All recordings, reference scores, lyric outputs,
feature archives, learned weights and SoundFonts remain local. None are bundled.

## Install

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[ml,notation,test]"
python -m musicsheet --help
python -m pytest tests_transcription -q
```

Install an appropriate PyTorch build for your platform. Core inference works on
CPU. FFmpeg is required for decoding recordings. Lyrics are optional:
`python -m pip install -e ".[lyrics]"`. The legacy GUI dependencies remain in
`requirements.txt`; they are not required by the new package.

## Try notation without private models

```sh
python -m musicsheet score --predictions examples/synthetic_notes.json --bpm 120 --key-sharps 0 --pdf --output outputs/synthetic-score-v1
```

This creates MusicXML, PDF, per-page SVG, raw MIDI, notated MIDI and an audit.
The sample is hand-created, not an extracted song. The exporter currently writes
a single violin part in **4/4**; the caller supplies tempo, score origin and key.
Pitches and attack count are preserved. Below-G3 pitches are reported instead
of silently raised. Rhythm remains a separate, provisional interpretation.

## Configure local model assets

Copy `docs/assets.example.json` to **`local/assets.json`**. Set each path relative
to that manifest (absolute local paths also work), and replace every placeholder
with the file's actual SHA-256. Inference verifies those hashes before loading
weights with `weights_only=True`. Use checkpoints you trust.

| Section | Compatible workspace bundle | Purpose |
| --- | --- | --- |
| `base` | `melody_pairs_v03_20261029` | 425 acoustic inputs; pitch/onset/offset network |
| `joint` | `joint_pitch_syllables_20261101_v2` | Acoustic articulation and event-register correction |
| `events` | `note_events_20261111_v1` | Experimental 229-input residual note-event network |
| `lyric` | `lyric_aware_pilot_20261111_v1/lyric_conditioned` | Experimental 629-input lyric/phone-conditioned network |

The base/event/lyric sections need `checkpoint.pt` and `normalization.npz`.
Joint additionally needs `decoder_config.json`. The optional event `decoder`
entry accepts the calibrated `attack_mix`/`continuation_strength` JSON from the
v2 experiment. Do not mix normalization files from different trained bundles.

Example checksum command:

```sh
python -c "import hashlib,pathlib; p=pathlib.Path('models/base/checkpoint.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Weights are intentionally **not downloadable from this repo**. A fresh clone
can run tests and the notation example, but trained song inference requires
the local checkpoints and features described here. No weights are fabricated
or silently replaced with a different pitch estimator.

## Run inference on existing features

```sh
python -m musicsheet infer --assets local/assets.json --features data/my-track/features --mode joint --output outputs/my-track-joint-v1
python -m musicsheet infer --assets local/assets.json --features data/my-track/features --mode note-events --output outputs/my-track-events-v1
```

Existing base probability caches can be supplied with
`--probabilities data/my-track/base-probabilities.npz`. They must contain
`pitch` (61,T), `boundaries` (2,T), and the identical `times_s` grid. They must
come from the same recording; equal lengths alone do not establish provenance.

Every new run needs a fresh output directory. Outputs include:

- `predictions.json`: raw MIDI pitch, source-second start/end, evidence and provenance.
- `raw.mid`: the same note events; its 120 BPM serialization scale does not quantize time.
- `probabilities.npz` and `decoder_audit.json`: evidence and rejected/changed decisions.
- `beat_candidates.json`: independent rhythm diagnostics that never rewrite events.

No reference score, annotation file, beat grid or violin octave shift is an
input to note inference. `joint` is the prior review candidate; `note-events`
and `lyric` are experimental and have not been promoted as accuracy improvements.

## Prepare a new recording

The repository contains the source-independent acoustic feature extractor.
It deliberately does **not** include the old project's voice-conversion runtime
or third-party separation/pitch model assets.

Supply a full-length vocal stem aligned to the original MP3 and an RMVPE pitch
archive from your local pitch-estimation workflow. Keep leading silence and
source gain. Do not peak-normalize the stems independently. The CLI validates
sample count/time grids; it cannot prove that an arbitrary stem belongs to a mix.

```sh
python -m musicsheet prepare --audio data/source.mp3 --vocals data/vocals.wav --pitch-track data/rmvpe-track.npz --output data/my-track/features
```

The pitch archive contains one-dimensional `times_s`, `rmvpe_f0_hz`,
`rmvpe_midi`, `rmvpe_confidence`, `rmvpe_voiced` on the exact 20ms grid starting
at zero. The frame count is `ceil(decoded_16kHz_samples / 320)`. Unknown MIDI
may be NaN; remaining evidence must be finite. An existing compatible vocal
feature archive also contains these arrays. The extractor does not manufacture
pitch evidence if a model or stem is missing. An automatic MP3-to-stems/RMVPE
installer is not included in this source release.

## Lyrics and syllable hypotheses

Install locally cached faster-whisper large-v3 and the 392-label
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` model separately. The command uses
`local_files_only=True`; no audio or text is sent to a remote API.

```sh
python -m musicsheet lyrics --vocals data/vocals.wav --asr-model models/whisper --phone-model models/phonetics --language ko --device cpu --output outputs/my-track-lyrics-v1
python -m musicsheet infer --assets local/assets.json --features data/my-track/features --mode lyric --lyrics outputs/my-track-lyrics-v1 --vocal-audio data/vocals.wav --output outputs/my-track-lyric-notes-v1
```

Outputs contain recognized words, phonetic evidence, vowel-nucleus candidates,
confidence gates and unresolved cases. English/Korean pronunciation is supported.
CTC peaks are **not vowel duration ground truth**. Missing ASR words are not rests.
One syllable can contain multiple pitches, and repeated syllables can share a
pitch. Automatic hypotheses are model inputs, not verified training labels.
Word-window edges, humming, code-switching and overlapping vocals remain difficult.

## Training and evaluation

Create a local manifest following `docs/training.example.json` and the schema
below. Explicit source hashes and disjoint song splits are required. The trainer
never opens test features, lyrics or labels, and fits normalization on training
songs only. Unknown regions are masked instead of treated as silence.

```sh
python -m musicsheet train --manifest local/training.json --mode base --epochs 12 --output models/base-run-v1
python -m musicsheet train --manifest local/training.json --assets local/assets.json --mode events --epochs 12 --output models/event-run-v1
python -m musicsheet train --manifest local/training.json --assets local/assets.json --mode lyric-control --epochs 12 --seed 42 --output models/control-v1
python -m musicsheet train --manifest local/training.json --assets local/assets.json --mode lyric --epochs 12 --seed 42 --output models/lyric-v1
```

`lyric-control` has the same 629-input architecture and zeroes the 400 added
lyric/phone inputs. Compare it to `lyric` with identical seeds and splits.
For residual training, base/joint models remain frozen. Joint-head training is
not exposed by this portable runner; it uses the existing local joint bundle.
The runner ports the masked loss and conservative targets, but **does not
reproduce the historical reviewed-region oversampling schedule**. New runs
must establish their own results. Selection saves the best validation checkpoint,
optimizer-step count, weight-change magnitude and `promoted=false`.

Each alignment JSON contains `source_sha256` and ordered `notes` with
`start_s`, `end_s`, `midi`, `supervise`, `alignment_confidence`, and optionally
`register_supervision_weight`. Set `supervise=false` for unresolved events.
Only independently verified quiet spans belong in `hard_negative_spans`.
Feature metadata JSON must match both the feature archive SHA-256 and source
SHA-256. Optional `lyrics` and `vocal_audio` paths are needed for lyric modes.

Evaluate only a common, explicitly aligned scoring region and pitch convention:

```sh
python -m musicsheet evaluate --reference local/verified-notes.json --predictions local/scoped-predictions.json --annotations local/review-regions.json --output outputs/evaluation-v1.json
```

The evaluator reports repeats, pitch-class versus exact-pitch errors, octave
errors, onsets/endings and tolerance sensitivity. Verified annotations are
required for sustained-note false splits and melisma recall; unavailable
metrics remain null. A violin arrangement is not automatically performed-vocal
ground truth. Passing software tests does not establish musical accuracy.

## Score playback with quieter background

Install FluidSynth and supply a local SoundFont you have permission to use:

```sh
python -m musicsheet render --midi outputs/synthetic-score-v1/notated.mid --soundfont local/instruments.sf2 --background data/source.mp3 --background-db -10 --output outputs/playback-v1
```

Use `raw.mid` to hear unquantized note timing or `notated.mid` to hear the score.
The renderer uses the MIDI's instrument program (violin by default), preserves
the source-time origin and writes `transcription.wav` plus `overlay.wav`.
Backing level is based on whole-clip RMS, followed by a shared peak reduction
if necessary. SoundFont quality and articulation affect musicality independently
of transcription accuracy. No random timing jitter or automatic pedal is added.
