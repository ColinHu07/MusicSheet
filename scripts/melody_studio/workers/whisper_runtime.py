"""Local audio-derived lyric drafts. No audio upload or melody-weight changes."""
from pathlib import Path
import os,json,time,hashlib,dataclasses,argparse,subprocess
import numpy as np
ROOT=Path(os.environ.get("MUSICSHEET_BACKEND_ROOT", Path(__file__).resolve().parents[3])).resolve()
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
# Reuse installed CUDA DLLs read-only. Dependencies live in .venv-lyrics.
dlls=ROOT/'vendor/rvc-2.3.260718/RVC20260718Nvidia50x0/runtime/Lib/site-packages/torch/lib'
if dlls.exists():
    dll_handle=os.add_dll_directory(str(dlls))
    os.environ['PATH']=str(dlls)+os.pathsep+os.environ['PATH']
from faster_whisper import WhisperModel
import ctranslate2

