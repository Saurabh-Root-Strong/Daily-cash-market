"""Entry point for dashboard.bat — adds project root to sys.path then delegates to app.py."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.app import main  # noqa: E402
main()
