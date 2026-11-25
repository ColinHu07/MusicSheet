# Running Melody Studio locally

Studio runs on loopback. Install the editor as shown in the README, then run `python scripts/melody_studio/server.py --open`. On Windows the launcher in that directory uses the checkout's `.venv`. Use `--port 8791` to open a separate instance. Use a separate `--store` for each concurrent instance.

## Importing MP3s

Full transcription uses a CUDA-capable NVIDIA GPU and FFmpeg (`ffmpeg` and `ffprobe` on PATH). The checkout includes all Studio worker scripts and MusicSheet-trained weights. The third-party models, separated Python environments, piano samples and user library are external. The app reads cached models offline. A fresh installation must provision these dependencies before importing a song.

If you already have the working music-transcription workspace, point Studio at its asset directory before starting the server. The new checkout supplies the app, workers and first-party checkpoints.

```powershell
$env:MUSICSHEET_BACKEND_ROOT = 'D:\path\to\your\transcription-workspace'
python scripts/melody_studio/server.py --open
```

By default the asset root is the checkout. Studio expects these paths underneath it:

| Path | Contents |
| --- | --- |
| `vendor/rvc-2.3.260718/RVC20260718Nvidia50x0/` | CUDA PyTorch runtime, BS-Roformer separation code and checkpoint, RMVPE extractor and checkpoint. Studio calls only separation and pitch extraction. The expected weight hashes are in `workers/audio_stems.py`. |
| `vendor/ROSVOT_20261117/` | [ROSVOT](https://github.com/RickyL-2000/ROSVOT) checkout at `3c8332bf43adae35f6e4d64971862f2f6139b310` with the upstream ROSVOT, RWBD and RMVPE checkpoints. |
| `models/transcription/GAME_20261117_medium/GAME-1.0.3-medium-onnx/` | The five [GAME](https://github.com/openvpi/GAME) ONNX graphs and config. |
| `models/transcription/lyrics_asr_cache/` | Local faster-whisper cache for `Systran/faster-whisper-large-v3`. |
| `models/transcription/phonetic_xlsr_20261111/` | Local `facebook/wav2vec2-xlsr-53-espeak-cv-ft` model, vocabulary and preprocessing config. |
| `data/instruments/salamander_subset/` (optional) | Local sample manifest and recordings for piano playback. Without a bank the browser synthesizes the note preview. |

The worker Python paths can be set independently:

| Variable | Required environment |
| --- | --- |
| `MUSICSHEET_ACOUSTIC_PYTHON` | PyTorch CUDA, the separation and pitch dependencies, librosa, SciPy, soundfile, transformers, phonemizer and espeakng-loader. Defaults to the bundled Windows acoustic runtime. |
| `MUSICSHEET_RESEARCH_PYTHON` | ROSVOT dependencies, torch, ONNX Runtime GPU, NumPy, SciPy, joblib and scikit-learn 1.7.2. Defaults to `.venv-transcription-research-20261117/Scripts/python.exe`. |
| `MUSICSHEET_LYRICS_PYTHON` | faster-whisper and CTranslate2 configured for CUDA. Defaults to `.venv-lyrics/Scripts/python.exe`. |

Uploaded audio, score revisions and logs are stored in `outputs/melody_studio`. Intermediate stems, feature arrays and automatically recognized words are in `data/melody_studio`. Both directories are ignored by Git.

## Reusing the trained model

The frozen three-source ensemble in `checkpoints/event_fusion_20261117_v1` combines the syllable-aware baseline, ROSVOT and GAME. It uses 56 features and an event selection threshold of 0.30. For existing local inference files, run `python -m musicsheet.trained_fusion --help` for the input paths. This step works on CPU. Only load trusted joblib checkpoints.

Raw predictions contain source-time notes and model scores. Studio then estimates the beat grid, applies the rhythm model and fits the melody to violin range. Score position and written octave can therefore differ from the raw prediction. The model saw the 17 annotated songs during development. Its scores are uncalibrated and its octave accuracy on new singers remains unverified.
