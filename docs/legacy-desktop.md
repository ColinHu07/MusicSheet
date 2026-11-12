# MP3 → Violin Sheet Music

A desktop application that transcribes audio files (MP3/WAV/FLAC) into
violin melody sheet music, outputting **MusicXML** and **PDF**.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Quick Start

### 1. Prerequisites

| Tool | Required? | Notes |
|------|-----------|-------|
| **Python 3.10+** | Yes | 3.11 or 3.12 recommended |
| **ffmpeg** | Recommended | Needed for MP3 decoding. [Download](https://ffmpeg.org/download.html) |
| **MuseScore 4** | Optional | Needed for PDF rendering. [Download](https://musescore.org/en/download) |

### 2. Install

```bash
# Clone the repo
git clone <repo-url> mp3-to-sheet
cd mp3-to-sheet

# Create virtual environment
python -m venv .venv

# Activate  (Windows)
.venv\Scripts\activate
# Activate  (macOS / Linux)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Basic Pitch backend:** By default, Basic Pitch pulls TensorFlow.
> For a lighter install, replace `basic-pitch` with `basic-pitch[onnx]` in
> `requirements.txt` (uses ONNX Runtime instead).

### 3. Run the GUI

```bash
python -m src.app
```

### 4. Run via CLI (headless)

```bash
python scripts/cli_transcribe.py song.mp3
python scripts/cli_transcribe.py song.wav --key G --time-sig 3/4 --simplify
python scripts/cli_transcribe.py song.mp3 --max-duration 120 --no-pdf -v
```

---

## Demo Walkthrough

1. **Launch** the app: `python -m src.app`
2. **Click "Browse"** and select an MP3 or WAV file.
3. **Choose settings:**
   - Instrument: *Violin (melody)* (only option for MVP)
   - Time signature: *Auto (4/4)* or pick manually
   - Key: *Auto* or select a specific key
   - Optionally check *Simplify rhythm* for a cleaner lead-sheet style
4. **Click "Generate Sheet Music"** and watch the log window.
5. **When done:**
   - Click **"Open MusicXML"** → opens in MuseScore / notation editor
   - Click **"Open PDF"** → opens the rendered PDF (requires MuseScore)
   - Click **"Open output folder"** → browse all outputs

### Output Structure

```
out/
├── musicxml/
│   └── <songname>.musicxml
├── pdf/
│   └── <songname>.pdf        (if MuseScore installed)
├── debug/
│   └── <songname>.json       (intermediate note data)
└── cache/
    └── <hash>.wav             (cached processed audio)
```

---

## Architecture

```
Audio file
    │
    ▼
┌──────────────────┐
│  Stage A: Ingest │   Load → mono WAV → normalize → truncate
└────────┬─────────┘
         ▼
┌──────────────────────────┐
│  Stage B: Basic Pitch    │   Audio → raw note events (polyphonic)
│  + monophonic cleanup    │   → confidence filter → overlap removal
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  Stage C: Beat tracking  │   librosa → tempo + beat times
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  Stage D: Key detection  │   Krumhansl-Schmuckler → key + pitch spelling
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  Quantize to grid        │   Snap to 1/16 (or 1/8) → rests → barline splits
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  Stage F: MusicXML       │   music21 → .musicxml
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  Stage G: PDF render     │   MuseScore CLI → .pdf
└──────────────────────────┘
```

---

## Configuration Options

### GUI / CLI Flags

| Option | GUI control | CLI flag | Default | Description |
|--------|-------------|----------|---------|-------------|
| Max duration | Spin box | `--max-duration` | 45 s | Limit transcription length |
| Time signature | Dropdown | `--time-sig` | auto (4/4) | Force meter |
| Key | Dropdown | `--key` | auto | Force key signature |
| Simplify rhythm | Checkbox | `--simplify` | off | Snap to 1/8 notes |
| Allow triplets | Checkbox | `--triplets` | off | Include triplet grid |
| Phrase slurs | Checkbox | `--slurs` | off | Add heuristic slurs |
| Min note (ms) | Spin box | `--min-note-ms` | 100 | Filter short notes |
| Min confidence | Spin box | `--min-confidence` | 0.3 | Filter weak notes |
| Violin range | Checkbox | `--no-clamp` | on | Transpose out-of-range |
| MuseScore path | Text field | `--musescore` | auto-detect | Path to mscore |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure

```
mp3-to-sheet/
├── README.md
├── requirements.txt
├── docs/
│   └── research_notes.md        # Tool selection rationale + citations
├── src/
│   ├── __init__.py
│   ├── app.py                   # PySide6 GUI entry point
│   └── pipeline/
│       ├── __init__.py          # Pipeline orchestrator
│       ├── utils.py             # Data classes, helpers
│       ├── ingest.py            # Stage A: audio loading
│       ├── melody_basic_pitch.py # Stage B: melody extraction
│       ├── beat_grid.py         # Stage C: beat tracking
│       ├── key_detect.py        # Stage D: key detection
│       ├── quantize.py          # Quantization + rest insertion
│       ├── musicxml_export.py   # Stage F: MusicXML generation
│       └── render_pdf.py        # Stage G: PDF via MuseScore
├── scripts/
│   └── cli_transcribe.py       # Headless CLI
└── tests/
    ├── test_quantize.py
    └── test_musicxml_roundtrip.py
```

---

## Known Limitations (MVP)

1. **Monophonic only** — extracts the predominant melody line. Chords, harmony,
   and accompaniment are intentionally discarded.
2. **Constant tempo assumed** — tempo changes within a song are not tracked.
   The single BPM estimate may be inaccurate for rubato or tempo-varying music.
3. **Key detection is heuristic** — works well for tonal music in common keys.
   Atonal or modal music may get a poor key estimate (use manual override).
4. **Pickup measures** — the first detected beat is treated as beat 1 of
   measure 1. Anacrusis (pickup) bars are not automatically detected.
5. **PDF requires MuseScore** — without MuseScore installed, only MusicXML
   is produced.
6. **First 45 seconds default** — longer songs can be transcribed by increasing
   the "Max duration" setting, but processing time scales accordingly.
7. **Basic Pitch dependency** — pulls TensorFlow (large). Use `basic-pitch[onnx]`
   for a lighter alternative.

---

## License

MIT (application code). See `docs/research_notes.md` for dependency licenses.
