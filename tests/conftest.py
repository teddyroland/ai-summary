"""Pytest configuration for the ai-summary test suite.

Adds the project's `code/` directory to sys.path so tests can import the
modules with plain `import ids` / `import io_utils` / etc., matching the way
the modules import each other at runtime.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"

# Prepend so our `code/` modules win over anything else with the same name.
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
