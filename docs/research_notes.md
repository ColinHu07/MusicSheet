# Research Notes — MP3 → Violin Sheet Music

## Tool Selection & Rationale

### 1. Basic Pitch (Audio → MIDI / Note Events)

**Source:** Spotify / [GitHub](https://github.com/spotify/basic-pitch) — Bittner et al., 2022

Basic Pitch is a lightweight neural audio-to-MIDI converter developed by Spotify.
It is designed to work on "almost any instrument" and accepts common formats
(MP3, WAV, FLAC, OGG). Its strength for our use case:

- Works best **one instrument at a time**, which aligns with our "violin melody"
  MVP — we feed it audio and extract the predominant pitch [1].
- Returns note events with start/end times, MIDI pitch, and amplitude (confidence),
  enabling downstream filtering.
- Relatively lightweight compared to full polyphonic transcription models.

**Limitation:** Basic Pitch may produce polyphonic output (chords detected in
the audio). We add a "monophonic enforcement" post-processing step that
resolves overlapping notes by keeping the highest-confidence event.

> [1] Bittner, R. M., et al. "A Lightweight Instrument-Agnostic Model for
> Polyphonic Note Transcription and Multipitch Estimation." ICASSP 2022.
> https://github.com/spotify/basic-pitch

---

### 2. Librosa (Beat Tracking + Tempo Estimation)

**Source:** [librosa.org](https://librosa.org/) — McFee et al., 2015

Librosa provides `librosa.beat.beat_track()` which returns estimated tempo (BPM)
and beat frame positions.  Key features for our pipeline:

- **Tempo estimation** gives us BPM to convert note times → quarter-note positions.
- **Beat times** provide anchors for our quantization grid.
- Supports dynamic tempo tracking via `librosa.beat.beat_track(units='time')` [2].

For MVP we assume constant tempo (single BPM value), which is a reasonable
default for most pop/classical recordings.

> [2] McFee, B., et al. "librosa: Audio and Music Signal Analysis in Python."
> Proceedings of the 14th Python in Science Conference, 2015.
> https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html

---

### 3. MuseScore CLI (Score → PDF Rendering)

**Source:** [MuseScore 4](https://musescore.org/)

MuseScore's command-line interface converts notation files to rendered output:

```
mscore -o output.pdf input.musicxml
```

This gives us publication-quality PDF rendering from our generated MusicXML [3].
On Windows the binary is typically at `C:\Program Files\MuseScore 4\bin\MuseScore4.exe`.

**Fallback:** If MuseScore is not installed, the app still produces a valid
MusicXML file that can be opened in any notation editor (Finale, Sibelius,
Dorico, etc.).

> [3] MuseScore Handbook: Command line usage.
> https://musescore.org/en/handbook/command-line-usage

---

### 4. MusicXML (Interchange Format)

**Source:** W3C Music Notation Community Group / [musicxml.com](https://www.musicxml.com/)

MusicXML is the de facto interchange format for Western music notation.
Key structural elements we use:

- `<note>` elements represent pitched notes with `<pitch>`, `<duration>`,
  and `<type>` (e.g. "quarter", "eighth").
- Rests are represented as `<note>` elements containing `<rest/>` instead of `<pitch>` [4].
- **Ties** use both `<tie type="start"/>` (sound) and `<tied type="start"/>`
  (notation) sub-elements — both are needed for correct playback and display.
- `<key>` and `<time>` elements define key/time signatures per measure.

We rely on **music21** to handle MusicXML generation, which correctly manages
these details.

> [4] MusicXML 4.0 Specification.
> https://www.w3.org/2021/06/musicxml40/

---

### 5. music21 (Score Construction + MusicXML Export)

**Source:** MIT — Cuthbert & Ariza, 2010
[music21.mit.edu](https://web.mit.edu/music21/)

music21 is used to:

- Construct a `Score` → `Part` → `Measure` hierarchy programmatically.
- Handle complex notation: splitting notes at barlines, creating tied note
  groups, beaming, rest decomposition.
- Export valid MusicXML via `score.write('musicxml')` [5].

> [5] Cuthbert, M. S. & Ariza, C. "music21: A Toolkit for Computer-Aided
> Musicology." 11th ISMIR, 2010.
> https://music21.readthedocs.io/

---

### 6. Key Detection — Krumhansl-Schmuckler Algorithm

We implement the Krumhansl-Schmuckler key-finding algorithm, which correlates
a pitch-class histogram (weighted by note duration) against established
major and minor tonal profiles.

- 24 candidate keys (12 major + 12 minor) are scored by Pearson correlation.
- The key with the highest correlation is chosen.
- Users can override with a manual key selection.

> Krumhansl, C. L. "Cognitive Foundations of Musical Pitch." Oxford, 1990.

---

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| GUI framework | PySide6 (Qt) | Cross-platform, mature, easy to implement |
| Audio→notes | Basic Pitch | Lightweight, instrument-agnostic, good Python API |
| Beat tracking | librosa | Industry standard, well-documented |
| Notation export | music21 → MusicXML | Handles complex notation logic automatically |
| PDF rendering | MuseScore CLI | Publication quality, free, widely installed |
| Quantization grid | 1/16 note (default) | Good balance of accuracy vs. readability |

## Licensing Notes

- **Basic Pitch:** Apache 2.0
- **librosa:** ISC License
- **music21:** BSD (or LGPL — check version)
- **PySide6:** LGPL v3
- **MuseScore 4:** GPL v3 (external tool, not bundled)

All licenses are compatible with an open-source project.
