"""Local SoundFont playback of an actual exported MIDI, with optional quiet backing."""
from pathlib import Path
import math
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from .audio import decode, save_json, sha256


def render(midi, soundfont, output, *, background=None, background_db=-10., fluidsynth='fluidsynth'):
    if not math.isfinite(background_db) or background_db > 0:
        raise ValueError('Background level must be finite and <= 0dB relative to the transcription')
    executable = shutil.which(fluidsynth)
    if executable is None:
        raise FileNotFoundError('Install FluidSynth or pass --fluidsynth with its executable path')
    midi, soundfont, output = Path(midi).resolve(), Path(soundfont).resolve(), Path(output)
    if not midi.is_file() or not soundfont.is_file():
        raise FileNotFoundError('MIDI and a local licensed SoundFont are required')
    output.mkdir(parents=True, exist_ok=False)
    piano = output / 'transcription.wav'
    subprocess.run([executable, '-ni', '-F', str(piano), '-r', '44100', str(soundfont), str(midi)], check=True)
    metadata = dict(midi_sha256=sha256(midi), soundfont_sha256=sha256(soundfont),
                    timing_source='Supplied MIDI; no random timing changes or added notes', background_db=background_db)
    if background:
        rate, samples = wavfile.read(piano)
        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float32) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        if samples.ndim == 1:
            samples = np.repeat(samples[:, None], 2, axis=1)
        backing = resample_poly(decode(Path(background)), rate // math.gcd(rate, 16000), 16000 // math.gcd(rate, 16000))
        backing = np.pad(backing[:len(samples)], (0, max(0, len(samples) - len(backing))))
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
        gain = rms(samples) * 10 ** (background_db / 20) / max(rms(backing), 1e-12)
        mixed = samples + backing[:, None] * gain
        peak_gain = min(1., .98 / max(float(np.max(abs(mixed))), 1e-12))
        wavfile.write(output / 'overlay.wav', rate, (mixed * peak_gain).astype(np.float32))
        metadata.update(background_sha256=sha256(Path(background)), background_gain=gain, shared_peak_gain=peak_gain)
    save_json(output / 'playback.json', metadata)
