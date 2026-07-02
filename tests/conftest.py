"""Shared test fixtures and path setup.

The package imports itself as ``src.<module>`` (it runs as ``python -m src``), so the
repository root must be importable. When pytest is invoked from the repo root that is
already the case, but we add it explicitly so the suite also works from other CWDs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
