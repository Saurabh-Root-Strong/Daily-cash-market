"""
Upload compressed DuckDB snapshot to GitHub Releases after each daily fetch.

Called automatically from run_daily.bat after a successful data fetch.
The Streamlit Community Cloud app downloads this snapshot on startup.

Environment variables (set in Windows Environment Variables or .env):
  GITHUB_TOKEN  — Personal Access Token with 'contents:write' scope
  GITHUB_REPO   — "username/repo-name"  (can be a separate public data repo)

Run manually:
  python scripts/upload_snapshot.py                 # full DB
  python scripts/upload_snapshot.py --days 120      # slim: last 120 days only

SLIM MODE (--days N): the full DuckDB is ~500 MB and would OOM Streamlit
Community Cloud's ~1 GB RAM container. --days builds a compact copy containing
only the last N days of each time-series table (reference/log tables kept whole),
which compresses to well under 100 MB. Recommended for cloud deployment.
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RELEASE_TAG = "latest-data"
_ASSET_NAME  = "market.duckdb.gz"

# Tables copied in FULL even in slim mode (small reference/learning data).
# prediction_log is the memory engine's history — tiny but must stay complete.
_KEEP_FULL_TABLES = {"sector_master", "prediction_log", "run_log"}


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


def _build_slim_db(src_path: Path, days: int, quiet: bool) -> Path | None:
    """
    Build a compact copy of the DuckDB holding only the last `days` calendar days
    of each time-series table (per-table cutoff = that table's own MAX(trade_date)
    − days). Reference/log tables in _KEEP_FULL_TABLES are copied whole.

    Source is attached READ_ONLY so this is safe to run alongside the dashboard.
    Returns the slim file path, or None on failure.
    """
    try:
        import duckdb
    except ImportError:
        if not quiet:
            print("[snapshot] duckdb not available — cannot build slim snapshot")
        return None

    slim_path = Path(tempfile.gettempdir()) / "market_slim.duckdb"
    if slim_path.exists():
        slim_path.unlink()

    con = None
    try:
        con = duckdb.connect(str(slim_path))
        con.execute(f"ATTACH '{src_path.as_posix()}' AS src (READ_ONLY)")

        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = 'src' AND table_schema = 'main'"
        ).fetchall()]

        for t in tables:
            cols = [r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_catalog = 'src' AND table_schema = 'main' "
                "AND table_name = ?", [t],
            ).fetchall()]

            if t in _KEEP_FULL_TABLES or "trade_date" not in cols:
                con.execute(f'CREATE TABLE main."{t}" AS SELECT * FROM src.main."{t}"')
                kept = "full"
            else:
                con.execute(
                    f'CREATE TABLE main."{t}" AS SELECT * FROM src.main."{t}" '
                    f'WHERE trade_date >= '
                    f'(SELECT MAX(trade_date) FROM src.main."{t}") - INTERVAL {int(days)} DAY'
                )
                kept = f"last {days}d"
            if not quiet:
                n = con.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
                print(f"[snapshot]   {t:<26} {kept:<10} {n:>10,} rows")

        con.execute("DETACH src")
        con.close()
        con = None

        # Reclaim free pages so the file on disk is actually compact
        con2 = duckdb.connect(str(slim_path))
        con2.execute("CHECKPOINT")
        con2.close()

        return slim_path
    except Exception as exc:
        if not quiet:
            print(f"[snapshot] Slim build failed: {exc}")
        if con is not None:
            try: con.close()
            except Exception: pass
        return None


def upload_snapshot(quiet: bool = False, days: int | None = None) -> bool:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPO", "")

    if not token or not repo:
        if not quiet:
            print("[snapshot] GITHUB_TOKEN or GITHUB_REPO not set — skipping upload")
        return False

    source_db = _find_db()
    if not source_db:
        if not quiet:
            print("[snapshot] DuckDB file not found — skipping upload")
        return False

    # Slim mode: build a compact last-N-days copy to fit cloud RAM limits.
    slim_tmp: Path | None = None
    if days is not None:
        if not quiet:
            print(f"[snapshot] Building slim snapshot (last {days} days)...")
        slim_tmp = _build_slim_db(source_db, days, quiet)
        if slim_tmp is None:
            print("[snapshot] Slim build failed — aborting (not uploading full DB).")
            return False
        db_path = slim_tmp
    else:
        db_path = source_db

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

    # Slim temp file no longer needed once compressed into memory
    if slim_tmp is not None:
        try: slim_tmp.unlink()
        except Exception: pass

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


def _parse_days(argv: list[str]) -> int | None:
    """Parse --days N (or --days=N). Returns None if absent (full upload)."""
    for i, a in enumerate(argv):
        if a == "--days" and i + 1 < len(argv):
            try: return int(argv[i + 1])
            except ValueError: return None
        if a.startswith("--days="):
            try: return int(a.split("=", 1)[1])
            except ValueError: return None
    return None


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    days  = _parse_days(sys.argv)
    success = upload_snapshot(quiet=quiet, days=days)
    sys.exit(0 if success else 1)
