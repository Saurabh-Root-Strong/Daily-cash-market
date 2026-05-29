"""
Cloud deployment support — Streamlit Community Cloud mode.

When CLOUD_MODE=true (set in Streamlit secrets or env):
  1. Download compressed DuckDB snapshot from GitHub Releases
  2. Set DATABASE_PATH env var so get_config() finds it
  3. App runs in read-only mode (no data fetching, no writes)

The laptop's run_daily.bat uploads a fresh snapshot after each successful
data fetch via scripts/upload_snapshot.py.

Environment variables required (set in Streamlit Cloud secrets):
  CLOUD_MODE    = "true"
  GITHUB_TOKEN  = "ghp_..." (PAT with contents:read for private repos;
                              not required if repo is public)
  GITHUB_REPO   = "username/repo-name"   e.g. "saurabh/Daily_Cash_Market_Data"
                  (can be a separate public data repo — keeps code repo private)
"""
from __future__ import annotations

import gzip
import io
import os
import shutil
import time
from pathlib import Path

__all__ = ["is_cloud", "ensure_database", "get_snapshot_info"]

_CLOUD_DB_PATH = Path("/tmp/market.duckdb")    # ephemeral cloud storage
_RELEASE_TAG   = "latest-data"


def is_cloud() -> bool:
    """True when running on Streamlit Community Cloud (or any cloud env)."""
    return os.environ.get("CLOUD_MODE", "").lower() == "true"


def ensure_database() -> bool:
    """
    If in cloud mode, download the DuckDB snapshot from GitHub Releases.
    Sets DATABASE_PATH so get_config() uses the downloaded file.
    Returns True if database is ready, False if download failed.

    Called once at app startup in app.py.
    """
    if not is_cloud():
        return True

    os.environ["DATABASE_PATH"] = str(_CLOUD_DB_PATH)

    if _CLOUD_DB_PATH.exists() and _CLOUD_DB_PATH.stat().st_size > 1_000:
        return True   # already downloaded this session

    return _download_snapshot()


def _download_snapshot() -> bool:
    """Download and decompress market.duckdb.gz from GitHub Releases."""
    try:
        import requests
    except ImportError:
        print("[cloud] requests not available — cannot download snapshot")
        return False

    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")

    if not repo:
        print("[cloud] GITHUB_REPO not set — cannot download snapshot")
        return False

    headers: dict = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # Get release metadata
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{_RELEASE_TAG}"
    try:
        r = requests.get(api_url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"[cloud] GitHub release not found ({r.status_code})")
            return False
        release = r.json()
    except Exception as exc:
        print(f"[cloud] Failed to fetch release metadata: {exc}")
        return False

    # Find the .duckdb.gz asset
    asset = next(
        (a for a in release.get("assets", []) if a["name"] == "market.duckdb.gz"),
        None,
    )
    if not asset:
        print("[cloud] market.duckdb.gz not found in release assets")
        return False

    size_mb = asset["size"] / 1024 / 1024
    print(f"[cloud] Downloading snapshot ({size_mb:.1f} MB)...")
    t0 = time.time()

    # For private repos: use the API download URL with auth
    # For public repos: browser_download_url works without auth
    download_url = asset["url"]   # API URL — requires auth header for private repos
    dl_headers   = {**headers, "Accept": "application/octet-stream"}

    try:
        resp = requests.get(download_url, headers=dl_headers, timeout=120, stream=True)
        if resp.status_code != 200:
            # Fall back to browser_download_url (public repos only)
            resp = requests.get(asset["browser_download_url"], timeout=120, stream=True)
        if resp.status_code != 200:
            print(f"[cloud] Download failed: HTTP {resp.status_code}")
            return False

        compressed = io.BytesIO(resp.content)
    except Exception as exc:
        print(f"[cloud] Download error: {exc}")
        return False

    # Decompress
    try:
        _CLOUD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(compressed) as gz_in:
            with open(_CLOUD_DB_PATH, "wb") as db_out:
                shutil.copyfileobj(gz_in, db_out)
    except Exception as exc:
        print(f"[cloud] Decompress error: {exc}")
        return False

    elapsed = time.time() - t0
    db_mb   = _CLOUD_DB_PATH.stat().st_size / 1024 / 1024
    print(f"[cloud] Database ready: {db_mb:.1f} MB in {elapsed:.1f}s")
    return True


def get_snapshot_info() -> dict:
    """Return metadata about the current cloud snapshot (for dashboard display)."""
    info = {"cloud_mode": is_cloud(), "db_ready": False, "db_size_mb": 0.0,
            "last_updated": None, "error": None}

    if not is_cloud():
        return info

    if _CLOUD_DB_PATH.exists():
        info["db_ready"]    = True
        info["db_size_mb"]  = round(_CLOUD_DB_PATH.stat().st_size / 1024 / 1024, 1)

    # Read last_updated from database
    try:
        from src.data.repository import query_dataframe
        df = query_dataframe("SELECT MAX(trade_date) AS d FROM daily_data", [])
        info["last_updated"] = df["d"].iloc[0]
    except Exception:
        pass

    return info
