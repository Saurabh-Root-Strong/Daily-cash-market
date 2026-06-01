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

_CLOUD_DB_PATH      = Path("/tmp/market.duckdb")   # ephemeral cloud storage
_CLOUD_VER_PATH     = Path("/tmp/market.version")  # last-downloaded version timestamp
_RELEASE_TAG        = "latest-data"
_VERSION_ASSET_NAME = "version.txt"


def is_cloud() -> bool:
    """True when running on Streamlit Community Cloud (or any cloud env)."""
    return os.environ.get("CLOUD_MODE", "").lower() == "true"


def _get_remote_version() -> str | None:
    """
    Fetch the remote version.txt from GitHub Releases — ~20 bytes, very fast.
    Returns the ISO-8601 timestamp string, or None on any failure.
    Used to detect whether a newer snapshot is available without downloading 27 MB.
    """
    try:
        import requests
    except ImportError:
        return None

    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")
    if not repo:
        return None

    headers: dict = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/releases/tags/{_RELEASE_TAG}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return None
        assets = r.json().get("assets", [])
        ver_asset = next((a for a in assets if a["name"] == _VERSION_ASSET_NAME), None)
        if not ver_asset:
            return None
        # Download the tiny version.txt content
        dl = requests.get(
            ver_asset["url"],
            headers={**headers, "Accept": "application/octet-stream"},
            timeout=10,
        )
        return dl.text.strip() if dl.status_code == 200 else None
    except Exception:
        return None


def ensure_database() -> tuple[bool, bool]:
    """
    Ensure the Cloud DuckDB snapshot is downloaded and up to date.
    Returns (success, newly_downloaded).

    Logic:
    1. If DB doesn't exist → download.
    2. If DB exists → check remote version.txt (lightweight, ~10 ms).
       If remote timestamp > local version → re-download (new data available).
       If same or can't reach GitHub → use existing DB.

    Called on app startup and periodically from app.py.
    """
    if not is_cloud():
        return True, False

    os.environ["DATABASE_PATH"] = str(_CLOUD_DB_PATH)

    db_exists = _CLOUD_DB_PATH.exists() and _CLOUD_DB_PATH.stat().st_size > 1_000

    if db_exists:
        # Read local version we downloaded last time
        local_ver = _CLOUD_VER_PATH.read_text().strip() if _CLOUD_VER_PATH.exists() else ""
        remote_ver = _get_remote_version()

        if remote_ver and remote_ver != local_ver:
            print(f"[cloud] New snapshot detected (remote={remote_ver}, local={local_ver}) — re-downloading")
        elif remote_ver:
            return True, False   # already up to date
        else:
            return True, False   # can't reach GitHub, use existing DB

    ok = _download_snapshot()
    return ok, ok


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

    # Save the remote version so the next check can compare without re-downloading
    try:
        ver_asset = next(
            (a for a in release.get("assets", []) if a["name"] == _VERSION_ASSET_NAME),
            None,
        )
        if ver_asset:
            vr = requests.get(
                ver_asset["url"],
                headers={**headers, "Accept": "application/octet-stream"},
                timeout=10,
            )
            if vr.status_code == 200:
                _CLOUD_VER_PATH.write_text(vr.text.strip())
    except Exception:
        pass   # version save failing doesn't affect the DB

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
