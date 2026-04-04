"""
Stage G — PDF (and PNG) rendering via MuseScore CLI.

Searches for a MuseScore installation and invokes it in headless mode
to convert MusicXML → PDF.
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mp3_to_sheet.render_pdf")


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #

def render_pdf(
    musicxml_path: str,
    pdf_path: str,
    musescore_bin: str = "",
) -> Optional[str]:
    """
    Convert *musicxml_path* → *pdf_path* using MuseScore CLI.

    Returns the PDF path on success, or ``None`` (with a warning) if
    MuseScore is not available.
    """
    exe = musescore_bin or find_musescore()
    if exe is None:
        logger.warning(
            "MuseScore not found.  Install MuseScore 4 to enable PDF export.\n"
            "https://musescore.org/en/download"
        )
        return None

    logger.info("Using MuseScore: %s", exe)
    logger.info("Rendering %s → %s", musicxml_path, pdf_path)

    cmd = [exe, "-o", pdf_path, musicxml_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("MuseScore error (rc=%d):\n%s",
                         result.returncode, result.stderr[:1000])
            return None

        if os.path.isfile(pdf_path):
            logger.info("PDF rendered: %s", pdf_path)
            return pdf_path
        else:
            logger.error("MuseScore exited OK but PDF was not created")
            return None

    except FileNotFoundError:
        logger.error("MuseScore binary not found at '%s'", exe)
        return None
    except subprocess.TimeoutExpired:
        logger.error("MuseScore timed out after 120 s")
        return None
    except Exception as exc:
        logger.error("MuseScore failed: %s", exc)
        return None


def render_png(
    musicxml_path: str,
    png_path: str,
    musescore_bin: str = "",
) -> Optional[str]:
    """Render a PNG preview image (bonus)."""
    exe = musescore_bin or find_musescore()
    if exe is None:
        return None

    cmd = [exe, "-o", png_path, musicxml_path]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # MuseScore appends page numbers: foo-1.png, foo-2.png …
        base = Path(png_path).stem
        parent = Path(png_path).parent
        candidates = sorted(parent.glob(f"{base}*.png"))
        if candidates:
            logger.info("PNG preview: %s", candidates[0])
            return str(candidates[0])
    except Exception as exc:
        logger.warning("PNG render failed: %s", exc)
    return None


# --------------------------------------------------------------------------- #
# MuseScore discovery
# --------------------------------------------------------------------------- #

def find_musescore() -> Optional[str]:
    """
    Try to locate the MuseScore executable on this system.

    Search order:
      1. ``mscore`` / ``musescore`` / ``MuseScore4`` on PATH
      2. Common install directories per OS
    """
    # 1. PATH search
    for name in ("mscore", "musescore", "MuseScore4", "MuseScore 4", "musescore4"):
        found = shutil.which(name)
        if found:
            return found

    system = platform.system()

    # 2. OS-specific well-known paths
    if system == "Windows":
        candidates = [
            r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
            r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
            r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
            r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
            "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/mscore",
            "/usr/local/bin/mscore",
            "/snap/bin/musescore",
        ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    return None
