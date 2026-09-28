"""Failure-mode tests for scripts/fetch_stock_classification.py (no network)."""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_stock_classification as f  # noqa: E402

COMPANY_PAGE = (
    "<title>Sun Pharmaceutical Industries Ltd share price | Screener</title>"
    '<a title="Broad Sector">Healthcare</a><a title="Sector">Healthcare</a>'
    '<a title="Broad Industry">Pharmaceuticals &amp; Biotechnology</a>'
    '<a title="Industry">Pharmaceuticals</a>'
)
CAPTCHA_PAGE = "<title>Just a moment...</title><div>verify you are human</div>"


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text


class _Session:
    def __init__(self, responses):
        self._responses = responses

    def get(self, url, **_):
        return self._responses.pop(0)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "RAW_DIR", tmp_path)
    monkeypatch.setattr(f.time, "sleep", lambda *_: None)
    return tmp_path


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def _patch_session(monkeypatch, responses):
    import requests
    monkeypatch.setattr(requests, "Session", lambda: _Session(responses))


def test_block_page_is_not_cached(raw_dir, monkeypatch):
    _patch_session(monkeypatch, [_Resp(200, CAPTCHA_PAGE)])
    f.fetch(["SUNPHARMA"], delay=0)
    assert not f._cache_path("SUNPHARMA").exists()     # retried next run


def test_company_page_is_cached_and_parsed(raw_dir, monkeypatch):
    _patch_session(monkeypatch, [_Resp(200, COMPANY_PAGE)])
    f.fetch(["SUNPHARMA"], delay=0)
    row = f.parse("SUNPHARMA")
    assert row["status"] == "ok"
    assert row["industry"] == "Pharmaceuticals & Biotechnology"
    assert row["basic_industry"] == "Pharmaceuticals"


def test_consecutive_block_pages_stop_the_run(raw_dir, monkeypatch):
    syms = [f"S{i}" for i in range(10)]
    _patch_session(monkeypatch, [_Resp(200, CAPTCHA_PAGE) for _ in syms])
    f.fetch(syms, delay=0)
    assert not any(f._cache_path(s).exists() for s in syms)


def test_worklist_order_and_expiry(raw_dir):
    fresh = f._cache_path("FRESH");  fresh.write_text(COMPANY_PAGE)
    stale = f._cache_path("STALE");  stale.write_text(COMPANY_PAGE); _age(stale, 200)
    new404 = f._cache_path("NEW404"); new404.write_text(f.NOT_FOUND); _age(new404, 2)
    old404 = f._cache_path("OLD404"); old404.write_text(f.NOT_FOUND); _age(old404, 10)
    work = f._worklist(["FRESH", "STALE", "NEW404", "OLD404", "MISSING"], refresh_days=180)
    assert work == ["MISSING", "OLD404", "STALE"]
    assert f._worklist(["STALE"], refresh_days=None) == []


def test_404_does_not_overwrite_a_good_page(raw_dir, monkeypatch):
    p = f._cache_path("SUNPHARMA"); p.write_text(COMPANY_PAGE); _age(p, 200)
    _patch_session(monkeypatch, [_Resp(404, "")])
    f.fetch(["SUNPHARMA"], delay=0, refresh_days=180)
    assert f.parse("SUNPHARMA")["status"] == "ok"


def test_etf_detection_word_boundaries():
    assert f._ETF_RE.search("Motilal Oswal Nifty Realty ETF")
    assert f._ETF_RE.search("Nippon India ETF Nifty BeES")
    assert not f._ETF_RE.search("Brainbees Solutions Ltd")
    assert not f._ETF_RE.search("SBI Funds Management Ltd")
