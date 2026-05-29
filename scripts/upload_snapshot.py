"""
Upload compressed DuckDB snapshot to GitHub Releases after each daily fetch.

Called automatically from run_daily.bat after a successful data fetch.
The Streamlit Community Cloud app downloads this snapshot on startup.

Environment variables (set in Windows Environment Variables or .env):
  GITHUB_TOKEN  — Personal Access Token with 'contents:write' scope
  GITHUB_REPO   — "username/repo-name"  (can be a separate public data repo)

Run manually:
  python scripts/upload_snapshot.py
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RELEASE_TAG = "latest-data"
_ASSET_NAME  = "market.duckdb.gz"


def _find_db() -> Path | None:
    """Locate the DuckDB file from settings.yaml."""
    try:
        import yaml
        cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
        db_rel = cfg["database"]["path"]
        db_path = PROJECT_ROOT / db_rel
        if db_path.exists():
            return db_path
    except Exception:
        pass

    # Fallback: search common locations
    for candidate in [
        PROJECT_ROOT / "data" / "market.duckdb",
        PROJECT_ROOT / "data" / "nse_market.duckdb",
        PROJECT_ROOT / "market.duckdb",
    ]:
        if candidate.exists():
            return candidate

    return None


def upload_snapshot(quiet: bool = False) -> bool:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")

    if not token or not repo:
        if not quiet:
            print("[snapshot] GITHUB_TOKEN or GITHUB_REPO not set — skipping upload")
        return False

    db_path = _find_db()
    if not db_path:
        if not quiet:
            print("[snapshot] DuckDB file not found — skipping upload")
        return False

    db_mb = db_path.stat().st_size / 1024 / 1024
    if not quiet:
        print(f"[snapshot] Compressing {db_mb:.1f} MB database...")

    # Compress in memory
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        with open(db_path, "rb") as fh:
            gz.write(fh.read())
    compressed = buf.getvalue()
    comp_mb = len(compressed) / 1024 / 1024

    if not quiet:
        print(f"[snapshot] Compressed to {comp_mb:.1f} MB "
              f"({comp_mb/db_mb*100:.0f}% of original)")

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get or create the 'latest-data' release
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{_RELEASE_TAG}"
    r = requests.get(release_url, headers=headers, timeout=30)

    if r.status_code == 404:
        # Create the release
        if not quiet:
            print(f"[snapshot] Creating GitHub release '{_RELEASE_TAG}'...")
        r = requests.post(
            f"https://api.github.com/repos/{repo}/releases",
            headers=headers,
            json={
                "tag_name":         _RELEASE_TAG,
                "name":             "Latest Market Data",
                "body":             "Auto-updated after each daily NSE data fetch.",
                "draft":            False,
                "prerelease":       False,
            },
            timeout=30,
        )
        if r.status_code not in (200, 201):
            print(f"[snapshot] Failed to create release: {r.status_code} {r.text[:200]}")
            return False

    release    = r.json()
    release_id = release["id"]

    # Delete existing asset (if any) before uploading new one
    for asset in release.get("assets", []):
        if asset["name"] == _ASSET_NAME:
            if not quiet:
                print(f"[snapshot] Replacing existing asset...")
            requests.delete(
                f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}",
                headers=headers,
                timeout=30,
            )
            break

    # Upload
    upload_url = (
        f"https://uploads.github.com/repos/{repo}/releases"
        f"/{release_id}/assets?name={_ASSET_NAME}"
    )
    upload_headers = {**headers, "Content-Type": "application/gzip"}

    if not quiet:
        print(f"[snapshot] Uploading to GitHub Releases...")
    t0 = time.time()

    r = requests.post(
        upload_url,
        headers=upload_headers,
        data=compressed,
        timeout=300,   # 5 min max for upload
    )

    if r.status_code == 201:
        elapsed = time.time() - t0
        if not quiet:
            print(f"[snapshot] Upload complete in {elapsed:.1f}s — "
                  f"mobile dashboard will use this data.")
        return True
    else:
        print(f"[snapshot] Upload failed: {r.status_code} {r.text[:300]}")
        return False


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    success = upload_snapshot(quiet=quiet)
    sys.exit(0 if success else 1)
