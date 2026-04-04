# conftest.py — ensures pytest can find `src` package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
