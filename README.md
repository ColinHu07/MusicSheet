# MusicSheet

Local music-transcription research: recover **pitches, repeated attacks, endings and uncertainty first**, then create an editable score.

The new `musicsheet` package contains the current acoustic note models, register correction, experimental lyric/phonetic conditioning, training/evaluation tools, MusicXML/PDF export and local MIDI playback. The earlier desktop app is preserved as legacy code.

## Start here

```sh
python -m venv .venv
# Activate the virtual environment for your platform.
python -m pip install -e ".[ml,notation,test]"
python -m musicsheet --help
python -m pytest tests_transcription -q
```

Try score generation without private models:

```sh
python -m musicsheet score --predictions examples/synthetic_notes.json --bpm 120 --pdf --output outputs/example-v1
```

- [Setup, local assets, inference, training and playback](docs/transcription.md)
- [Model status, evaluation limits and research](docs/model-status.md)
- [Local model manifest template](docs/assets.example.json)
- [Training manifest template](docs/training.example.json)
- [Legacy desktop app instructions](docs/legacy-desktop.md)

## What is included

| Component | Behavior |
| --- | --- |
| Acoustic models and decoder | Independent pitch/attack/ending predictions; repeated notes remain possible at unchanged pitch |
| Register correction | Learned acoustic/contextual pitch correction; no blanket violin octave shift |
| Note-event experiment | Residual predictions for reattacks, transitions and continuation |
| Lyric experiment | Local ASR plus phonetic alignment; syllables are soft evidence, not a fixed note quota |
| Training | Explicit train/validation/test manifest, train-only normalization, checkpoint selection and weight-change reporting |
| Notation | Contextual triplets/binary grids, source attack/pitch preservation, separate raw and notated MIDI |
| Playback | Local SoundFont rendering and optional quieter background overlay |
| Evaluation | Separate repeat, pitch/octave, boundary, sustain and melisma diagnostics where labels support them |

## Current limits

**This is a source release, not a pretrained one-click MP3 converter.** Recordings, reference scores, lyric outputs, learned checkpoints and instrument banks remain local. A fresh clone can run the synthetic tests and score example; song inference requires compatible local weights and features. New-audio feature preparation expects an aligned vocal stem and independently computed RMVPE evidence. See the setup guide for the exact schema.

The latest note-event and lyric-conditioned models are experiments. The lyric pilot has **not demonstrated better note accuracy** than its matched acoustic control. Most historical labels are weakly aligned arrangements. Passing software tests or producing smoother playback does not establish musical accuracy.

The new CLI does not call the legacy GUI pipeline. `python -m src.app` still runs the old desktop app, with its earlier limitations. No voice-conversion or live-microphone code is part of the new package.
