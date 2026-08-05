"""Make `src/` importable so `pytest` works from a clean checkout with no setup.

Without this, running pytest requires PYTHONPATH=src, which is an easy thing to
forget and produces a confusing ModuleNotFoundError rather than a useful one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
