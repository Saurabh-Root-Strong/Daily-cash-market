"""
Analytics-layer gateway to data health checks.

Dashboard and CLI import from here — never from src.data.health directly.
This preserves the dashboard → analytics → data layering rule.
"""
from src.data.health import DataHealth, SourceStatus, run_health_check

__all__ = ["DataHealth", "SourceStatus", "run_health_check"]
