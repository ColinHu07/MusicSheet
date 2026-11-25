# MusicSheet

Local music-transcription research: recover **pitches, repeated attacks, endings and uncertainty first**, then create an editable score.

## Contributor notebook

[Open the annotation notebook](https://colinhu07.github.io/MusicSheet/). Import several MP3s at once to get one random excerpt per song. Each excerpt includes playback, a clickable estimated pitch map and note-to-syllable review rows. The browser estimates pitches from the mix, so instrumental notes and missed attacks need review. Imports and edits persist on this device. Export the annotation backup to share your work.

## Melody Studio

Import an MP3, follow the engraved melody, click or drag notes to correct pitch and timing, and export PDF, MusicXML or MIDI. The gallery saves edits and song details locally.

From a clone, install the editor and launch it with:

```sh
python -m venv .venv
# Activate .venv first
python -m pip install -e ".[notation,fusion]"
python scripts/melody_studio/server.py --open
```

**MP3 transcription requires the local audio workers, GPU runtimes and third-party acoustic assets described in [Studio setup](docs/studio.md).** The trained MusicSheet checkpoints are included. The local server listens on loopback and keeps uploads and edits on disk.

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

The earlier note-first CLI uses its own asset manifest. Studio uses the bundled fusion, acoustic, syllable, register and rhythm checkpoints. See [Studio setup](docs/studio.md).

The latest note-event and lyric-conditioned models are experiments. The lyric pilot has **not demonstrated better note accuracy** than its matched acoustic control. Most historical labels are weakly aligned arrangements. Passing software tests or producing smoother playback does not establish musical accuracy.

The new CLI does not call the legacy GUI pipeline. `python -m src.app` still runs the old desktop app, with its earlier limitations. No voice-conversion or live-microphone code is part of the new package.
