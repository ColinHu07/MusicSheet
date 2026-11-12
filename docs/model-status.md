# Model status and provenance

This source release imports the local transcription work through November 12,
2026. `import_manifest.json` records original component hashes and extracted
definitions. Device/path adapters make inference portable; they do not retrain
weights or establish a new musical-accuracy result.

The earlier acoustic articulation head was sometimes called a "syllable" head.
It consumes acoustic features, not recognized lyrics. The later lyric pilot
really adds ASR, phonetic posteriors and aligned vowel-nucleus hypotheses.
Neither supplies verified syllable-versus-melisma/vibrato class labels yet.

## Recorded experiments

The separate note-event experiment used nine training songs, two validation
songs, and two previously evaluated holdouts. Validation exact-pitch/onset F1
changed from 73.2288% to 73.2225%; holdout F1 changed from 67.6262% to 67.9442%.
Those small mixed changes did not establish a useful improvement.

The lyric-conditioned pilot used two training songs, one validation song and
one previously evaluated holdout. Matched acoustic-control versus lyric F1:

| Role/song | Control | Lyric-conditioned |
| --- | ---: | ---: |
| Training: I DO | 79.54% | 79.54% |
| Training: What Is Love | 76.47% | 76.47% |
| Validation: Dynamite | 76.06% | 75.95% |
| Existing holdout: Life's Too Short | 68.27% | 68.27% |

Repeated-note matches were unchanged. The pilot is not promoted. These figures
use existing weak, score-aligned note labels and 100ms onset tolerance. They
are not fresh blind-test results or human-verified accuracy percentages. The
recordings, arrangements, labels and weights remain private and are not shipped
to make these results independently reproducible from a fresh clone.

## Packaging verification

Private compatibility checks reproduced all 329 I DO pitch/onset/ending triples
from both the existing note-event and lyric-conditioned outputs. This proves
compatibility for that case, not that the notes are correct. Public CI uses
synthetic fixtures without song audio, reference scores or trained checkpoints.
The portable training runner has its own recorded sampling/selection recipe;
historical experiment metrics must not be attached to new runs automatically.

## Architectural differences from the legacy app

- Raw notes and musical notation are separate artifacts.
- Supported same-pitch attacks are split after continuity repairs.
- Register correction uses acoustic/contextual evidence and a learned head.
- Unknown intervals are masked; missing lyrics or omitted score sections are not silence.
- Contextual binary/triplet notation preserves source pitches and attack count.
- A source-time MIDI is available separately from score-timed MIDI.

The legacy `src/`, `analysis/`, `eval/`, `scripts/cli_transcribe.py` and `tests/`
are retained for history. Their earlier tuning reports and behavior do not
describe the new `musicsheet` package. Launching `python -m src.app` still runs
the old GUI; use the new CLI for this engine.

## Research and external assets

- [Phoneme-informed note-level singing transcription](https://arxiv.org/abs/2304.05917)
- [Official phoneme-informed implementation](https://github.com/seyong92/phoneme-informed-note-level-singing-transcription)
- [Meta XLSR phonetic checkpoint](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [SOFA singing alignment research](https://github.com/qiuqiao/SOFA)

External models and instrument banks have their own licenses. No third-party
weights, copyrighted song assets, voice models or microphone-control code are
included in the new package. The ASR/phonetic integrations load local models;
the repository does not upload audio or extracted text.
