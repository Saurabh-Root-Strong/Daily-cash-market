import time
import requests
from src.logging_setup import get_logger

log = get_logger(__name__)

_PRIME_URLS = [
    "https://www.nseindia.com/",
    "https://www.nseindia.com/all-reports",
]


class NSEClient:
    def __init__(self):
        from src.config_loader import load_config
        cfg = load_config()["ingestion"]
        self._user_agent = cfg["user_agent"]
        self._retries = cfg["retries"]
        self._timeout = cfg["timeout"]
        self._polite_delay = cfg["polite_delay"]
        self._session: requests.Session | None = None

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/",
        })
        for url in _PRIME_URLS:
            try:
                s.get(url, timeout=self._timeout)
                time.sleep(self._polite_delay)
            except Exception as exc:
                log.debug("Cookie priming failed for %s: %s", url, exc)
        return s

    def _session_or_new(self) -> requests.Session:
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def _reset_session(self) -> None:
        self._session = None

    def get(self, url: str, expect_404_ok: bool = False) -> requests.Response | None:
        for attempt in range(1, self._retries + 1):
            try:
                s = self._session_or_new()
                resp = s.get(url, timeout=self._timeout)
                if resp.status_code == 404 and expect_404_ok:
                    return None
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404 and expect_404_ok:
                    return None
                log.warning("HTTP error on attempt %d/%d for %s: %s", attempt, self._retries, url, exc)
            except Exception as exc:
                log.warning("Request error on attempt %d/%d for %s: %s", attempt, self._retries, url, exc)
            self._reset_session()
            time.sleep(self._polite_delay * attempt)
        log.error("All %d attempts failed for %s", self._retries, url)
        return None

    def get_text(self, url: str, expect_404_ok: bool = False) -> str | None:
        resp = self.get(url, expect_404_ok=expect_404_ok)
        if resp is None:
            return None
        return resp.text
