"""
Source separation — isolate vocal stem from mixed audio.

Wraps Demucs (or any future separator) behind a clean interface so
the rest of the pipeline only sees ``separate(audio_path) → StemPaths``.

Key design decisions:
  - Lazy import: demucs is only imported if vocals mode is actually used.
  - MP3 output: Demucs is invoked with ``--mp3`` because the default WAV
    save path fails on some platforms (TorchCodec issue).  This is the
    confirmed-working invocation on the developer's machine.
  - Module invocation: uses ``python -m demucs`` when the ``demucs``
    executable is not on PATH, which is the common case when installed
    via pip in a virtual environment.
  - Caching: separated stems are stored in ``{output_dir}/stems/{hash}/``
    so re-runs of the same file skip the expensive separation step.
  - Graceful fallback: if demucs is unavailable, ``is_available()``
    returns False and the pipeline falls back to mix mode.
  - Modular: swap separator backend by implementing a new ``_run_*``
    function and wiring it in ``separate()``.
"""

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import ensure_dir, file_hash

logger = logging.getLogger("mp3_to_sheet.source_separation")

STEM_EXTENSIONS = (".mp3", ".wav", ".flac")


@dataclass
class StemPaths:
    """Paths to separated audio stems."""
    vocals: Optional[str] = None
    accompaniment: Optional[str] = None
    drums: Optional[str] = None
    bass: Optional[str] = None
    other: Optional[str] = None
    model_name: str = ""
    source_hash: str = ""


def is_available() -> bool:
    """Check whether a usable source separator is installed."""
    return (
        _demucs_cli_available()
        or _demucs_module_available()
        or _demucs_library_available()
    )


def build_demucs_command(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs",
    mp3_output: bool = True,
    use_module: bool = False,
) -> List[str]:
    """
    Build the Demucs CLI command list.

    Parameters
    ----------
    use_module : bool
        If True, use ``python -m demucs.separate`` instead of the
        bare ``demucs`` executable.  Required when demucs is installed
        in a venv but not on PATH.
    """
    if use_module:
        cmd = [sys.executable, "-m", "demucs.separate"]
    else:
        cmd = ["demucs"]

    cmd.extend(["--two-stems", "vocals", "-n", model, "-o", output_dir])
    if mp3_output:
        cmd.append("--mp3")
    cmd.append(audio_path)
    return cmd


def separate(
    audio_path: str,
    output_dir: str = "out",
    model: str = "htdemucs",
    stems_dir_override: Optional[str] = None,
) -> StemPaths:
    """
    Separate *audio_path* into vocal and accompaniment stems.

    Results are cached in ``{output_dir}/stems/{file_hash}/``.

    Parameters
    ----------
    audio_path : str
        Path to input audio file (WAV, MP3, etc.).
    output_dir : str
        Root output directory (stems go into a subdirectory).
    model : str
        Demucs model name.
    stems_dir_override : str, optional
        Explicit path to a directory containing pre-separated stems
        (e.g. from a prior manual demucs run).

    Returns
    -------
    StemPaths with at least ``vocals`` populated.
    """
    audio_path = os.path.abspath(audio_path)
    fhash = file_hash(audio_path)
    stems_dir = ensure_dir(os.path.join(output_dir, "stems", fhash))

    # 1. Check explicit override
    if stems_dir_override and os.path.isdir(stems_dir_override):
        override_result = _check_cache(stems_dir_override, model)
        if override_result is not None:
            logger.info("Using stems from override dir: %s",
                        stems_dir_override)
            return override_result

    # 2. Check our own cache
    cached = _check_cache(stems_dir, model)
    if cached is not None:
        logger.info("Using cached stems from %s", stems_dir)
        return cached

    # 3. Auto-discover pre-existing stems from default demucs output
    discovered = _discover_preexisting_stems(audio_path, model)
    if discovered is not None:
        logger.info("Discovered pre-existing stems")
        return discovered

    # 4. Run demucs
    logger.info("Running source separation on %s (model=%s) ...",
                audio_path, model)

    backend, use_module = _select_backend()

    if backend == "cli":
        return _run_demucs_cli(
            audio_path, stems_dir, model, fhash,
            use_module=use_module,
        )
    elif backend == "library":
        return _run_demucs_library(audio_path, stems_dir, model, fhash)
    else:
        raise RuntimeError(
            "No source separator available.  Install demucs:\n"
            "  pip install demucs\n"
            "Then retry with --source-mode vocals"
        )


# --------------------------------------------------------------------------- #
# Availability checks
# --------------------------------------------------------------------------- #

def _demucs_cli_available() -> bool:
    """Check if ``demucs`` binary is on PATH."""
    return shutil.which("demucs") is not None


def _demucs_module_available() -> bool:
    """Check if ``python -m demucs.separate`` works (common in venvs)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "demucs.separate", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _demucs_library_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def _discover_preexisting_stems(
    audio_path: str,
    model: str = "htdemucs",
) -> Optional[StemPaths]:
    """
    Look for pre-existing separated stems in common locations.

    Demucs by default outputs to ``./separated/{model}/{stem_name}/``.
    If the user has already run demucs manually, the stems will be there.
    """
    src_stem = Path(audio_path).stem
    search_dirs = [
        Path(".") / "separated" / model / src_stem,
        Path(audio_path).parent / "separated" / model / src_stem,
    ]

    for candidate_dir in search_dirs:
        if candidate_dir.is_dir():
            vocals = _find_stem_file(candidate_dir, "vocals")
            if vocals is not None:
                accomp = (
                    _find_stem_file(candidate_dir, "no_vocals")
                    or _find_stem_file(candidate_dir, "accompaniment")
                )
                logger.info(
                    "Found pre-existing stems at %s", candidate_dir)
                return StemPaths(
                    vocals=vocals,
                    accompaniment=accomp,
                    drums=_find_stem_file(candidate_dir, "drums"),
                    bass=_find_stem_file(candidate_dir, "bass"),
                    other=_find_stem_file(candidate_dir, "other"),
                    model_name=model,
                    source_hash="preexisting",
                )
    return None


def _select_backend() -> Tuple[str, bool]:
    """
    Choose the best available Demucs backend.

    Returns (backend_type, use_module) where backend_type is one of
    "cli", "library", or "none".

    Preference order:
    1. ``demucs`` on PATH (CLI, use_module=False)
    2. ``python -m demucs.separate`` (CLI via module, use_module=True)
    3. Demucs Python API (library — may fail with TorchCodec on some platforms)
    4. None
    """
    if _demucs_cli_available():
        logger.info("Using demucs CLI backend (executable on PATH)")
        return ("cli", False)
    if _demucs_module_available():
        logger.info("Using demucs module backend (python -m demucs.separate)")
        return ("cli", True)
    if _demucs_library_available():
        logger.info("Using demucs library backend (Python API)")
        return ("library", False)
    return ("none", False)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def _check_cache(stems_dir: Path, model: str) -> Optional[StemPaths]:
    """Return StemPaths if cached vocal stem exists, else None."""
    vocals_path = _find_stem_file(stems_dir, "vocals")
    if vocals_path is not None:
        accomp_path = (
            _find_stem_file(stems_dir, "no_vocals")
            or _find_stem_file(stems_dir, "accompaniment")
        )
        return StemPaths(
            vocals=vocals_path,
            accompaniment=accomp_path,
            drums=_find_stem_file(stems_dir, "drums"),
            bass=_find_stem_file(stems_dir, "bass"),
            other=_find_stem_file(stems_dir, "other"),
            model_name=model,
            source_hash=os.path.basename(str(stems_dir)),
        )
    return None


def _find_stem_file(directory, stem_name: str) -> Optional[str]:
    """Find a stem file with any supported extension."""
    for ext in STEM_EXTENSIONS:
        p = os.path.join(directory, f"{stem_name}{ext}")
        if os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# Demucs CLI backend (works with bare executable or python -m module)
# --------------------------------------------------------------------------- #

def _run_demucs_cli(
    audio_path: str,
    stems_dir: Path,
    model: str,
    fhash: str,
    use_module: bool = False,
) -> StemPaths:
    """Run demucs via CLI (or python -m module) and copy stems to cache."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = build_demucs_command(
            audio_path, tmpdir, model,
            mp3_output=True, use_module=use_module,
        )
        logger.info("Running: %s", " ".join(cmd))

        env = _build_env_with_ffmpeg()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env,
        )

        if proc.returncode != 0:
            logger.error("demucs failed (exit %d):\nSTDOUT: %s\nSTDERR: %s",
                         proc.returncode,
                         proc.stdout[:1000],
                         proc.stderr[:2000])
            raise RuntimeError(
                f"demucs separation failed (exit {proc.returncode})")

        logger.info("Demucs completed successfully, locating output stems...")

        src_stem = Path(audio_path).stem
        demucs_out = Path(tmpdir) / model / src_stem

        if not demucs_out.is_dir():
            for candidate in Path(tmpdir).rglob("vocals.*"):
                if candidate.suffix in STEM_EXTENSIONS:
                    demucs_out = candidate.parent
                    logger.info("Found stems via search at: %s", demucs_out)
                    break

        if not demucs_out.is_dir():
            raise RuntimeError(
                f"Demucs output directory not found. "
                f"Searched in {tmpdir} for model={model}, stem={src_stem}")

        vocals_src = _find_stem_file(demucs_out, "vocals")
        if vocals_src is None:
            raise RuntimeError(
                f"Expected vocal stem not found in {demucs_out}")

        vocals_ext = Path(vocals_src).suffix
        vocals_dst = os.path.join(stems_dir, f"vocals{vocals_ext}")
        shutil.copy2(vocals_src, vocals_dst)
        logger.info("Vocal stem saved: %s", vocals_dst)

        accomp_src = (
            _find_stem_file(demucs_out, "no_vocals")
            or _find_stem_file(demucs_out, "accompaniment")
        )
        accomp_dst = None
        if accomp_src is not None:
            accomp_ext = Path(accomp_src).suffix
            accomp_dst = os.path.join(
                stems_dir, f"no_vocals{accomp_ext}")
            shutil.copy2(accomp_src, accomp_dst)

    logger.info("Stems cached at %s", stems_dir)
    return StemPaths(
        vocals=vocals_dst,
        accompaniment=accomp_dst,
        model_name=model,
        source_hash=fhash,
    )


# --------------------------------------------------------------------------- #
# FFmpeg environment helper
# --------------------------------------------------------------------------- #

_ffmpeg_shim_dir: Optional[str] = None


def _build_env_with_ffmpeg() -> Optional[dict]:
    """
    Build an environment dict that puts a working ``ffmpeg`` binary on PATH.

    imageio-ffmpeg bundles an FFmpeg binary but with a versioned name
    (e.g. ``ffmpeg-win-x86_64-v7.1.exe``).  Demucs expects a bare
    ``ffmpeg`` / ``ffmpeg.exe`` on PATH, so we create a shim directory
    containing a copy (or hard link) with the canonical name.
    """
    global _ffmpeg_shim_dir

    if _ffmpeg_shim_dir is not None and os.path.isdir(_ffmpeg_shim_dir):
        env = os.environ.copy()
        env["PATH"] = _ffmpeg_shim_dir + os.pathsep + env.get("PATH", "")
        return env

    if shutil.which("ffmpeg") is not None:
        logger.debug("System ffmpeg found on PATH")
        return None

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, Exception) as exc:
        logger.debug("imageio-ffmpeg not available: %s", exc)
        return None

    import tempfile
    shim_dir = tempfile.mkdtemp(prefix="ffmpeg_shim_")
    target_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    shim_path = os.path.join(shim_dir, target_name)

    try:
        shutil.copy2(ffmpeg_exe, shim_path)
        logger.info(
            "Created ffmpeg shim: %s -> %s", ffmpeg_exe, shim_path)
    except Exception as exc:
        logger.warning("Could not create ffmpeg shim: %s", exc)
        return None

    _ffmpeg_shim_dir = shim_dir
    env = os.environ.copy()
    env["PATH"] = shim_dir + os.pathsep + env.get("PATH", "")
    return env


# --------------------------------------------------------------------------- #
# Demucs library backend (fallback — may fail with TorchCodec on some systems)
# --------------------------------------------------------------------------- #

def _run_demucs_library(
    audio_path: str,
    stems_dir: Path,
    model: str,
    fhash: str,
) -> StemPaths:
    """Use demucs Python API directly (heavier but no PATH needed)."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import save_audio
    import torch
    import torchaudio

    logger.info("Loading demucs model '%s' via Python API ...", model)
    separator = get_model(model)
    separator.eval()

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        sources = apply_model(separator, wav, device="cpu")

    source_names = separator.sources
    vocals_idx = source_names.index("vocals")

    vocals_audio = sources[0, vocals_idx]
    vocals_dst = os.path.join(stems_dir, "vocals.wav")
    save_audio(vocals_audio, vocals_dst, sr)

    accomp_dst = None
    non_vocal_indices = [
        i for i, s in enumerate(source_names) if s != "vocals"]
    if non_vocal_indices:
        accomp_audio = sum(sources[0, i] for i in non_vocal_indices)
        accomp_dst = os.path.join(stems_dir, "no_vocals.wav")
        save_audio(accomp_audio, accomp_dst, sr)

    logger.info("Stems cached at %s", stems_dir)
    return StemPaths(
        vocals=vocals_dst,
        accompaniment=accomp_dst,
        model_name=model,
        source_hash=fhash,
    )
