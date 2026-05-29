"""
NSEHttpClient — typed, config-injected HTTP client for NSE endpoints.

Responsibilities:
  - Cookie priming (NSE blocks cold requests without session cookies)
  - Retry with exponential back-off
  - Polite delay between requests
  - Returns None when a 404 is acceptable (holiday / no data)
  - Raises FetchError after all retries exhausted

Inject a custom IngestionConfig in tests to avoid real network calls.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import requests

from src.core.exceptions import FetchError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.core.config import IngestionConfig

__all__ = ["NSEHttpClient"]

log = get_logger(__name__)

_PRIME_URLS = (
    "https://www.nseindia.com/all-reports-derivatives",
    "https://www.nseindia.com/all-reports",
)


class NSEHttpClient:
    """
    Stateful HTTP session toward NSE India.

    Parameters
    ----------
    config:
        IngestionConfig to use.  Omit to read from the global AppConfig.
        Pass an override in tests to avoid touching the real network.
    """

    def __init__(self, config: "IngestionConfig | None" = None) -> None:
        if config is None:
            from src.core.config import get_config
            config = get_config().ingestion
        self._cfg = config
        self._session: requests.Session | None = None

    # ── Session management ────────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent":      self._cfg.user_agent,
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         "https://www.nseindia.com/",
        })
        for url in _PRIME_URLS:
            try:
                s.get(url, timeout=self._cfg.timeout)
                time.sleep(self._cfg.polite_delay)
            except Exception as exc:
                log.debug("Cookie priming skipped for %s: %s", url, exc)
        return s

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def _reset(self) -> None:
        self._session = None

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, url: str, *, expect_404_ok: bool = False) -> requests.Response | None:
        """
        GET url with retries.

        Parameters
        ----------
        expect_404_ok:
            If True, a 404 is treated as "no data" and returns None instead
            of raising.  Use for holiday/weekend data files.

        Returns
        -------
        Response or None (only when expect_404_ok=True and server 404'd).

        Raises
        ------
        FetchError
            After all retries are exhausted.
        """
        for attempt in range(1, self._cfg.retries + 1):
            try:
                resp = self._get_session().get(url, timeout=self._cfg.timeout)
                if resp.status_code == 404 and expect_404_ok:
                    return None
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                if code == 404 and expect_404_ok:
                    return None
                log.warning("HTTP %s on attempt %d/%d: %s", code, attempt, self._cfg.retries, url)
            except Exception as exc:
                log.warning("Request error attempt %d/%d for %s: %s", attempt, self._cfg.retries, url, exc)
            self._reset()
            time.sleep(self._cfg.polite_delay * attempt)

        raise FetchError(url=url, reason=f"all {self._cfg.retries} attempts failed")

    def get_bytes(self, url: str, *, expect_404_ok: bool = False) -> bytes | None:
        resp = self.get(url, expect_404_ok=expect_404_ok)
        return resp.content if resp is not None else None

    def get_text(self, url: str, *, expect_404_ok: bool = False) -> str | None:
        resp = self.get(url, expect_404_ok=expect_404_ok)
        return resp.text if resp is not None else None

    def __repr__(self) -> str:
        return f"NSEHttpClient(retries={self._cfg.retries}, timeout={self._cfg.timeout})"
