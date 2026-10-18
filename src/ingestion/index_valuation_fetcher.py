"""
Index valuation history: P/E, P/B, dividend yield, price and total-return index.

Sources (every URL string for this feature lives in this file):
  niftyindices.com /BackPage/getpepbHistoricaldataDBtoString   P/E, P/B, DY  (1999+)
  niftyindices.com /BackPage/getTotalReturnIndexString         TRI           (1999+)
  niftyindices.com /BackPage/getHistoricaldatatabletoString    OHLC          (1990s+)
  nseindia.com     /api/historicalOR/indicesYield              P/E, P/B, DY  (~2016+, cross-check only)
  nseindia.com     /api/historicalOR/vixhistory                India VIX     (~2011+)
  api.db.nomics.world  IMF IFS + OECD FINMARK                  10y G-sec yield, monthly

Traps found while building this:
  * The niftyindices endpoints serve at most ~1 year per request (the site's own JS
    enforces 365 days since 2025-08-28), so history is pulled year by year.
  * An index whose constituents' combined earnings are negative has its P/E
    published as '-' (e.g. Nifty Midcap 50 in 2005). That is "not computed", never 0.
  * Dates come as '01 Jan 1999' (niftyindices) and '01-JAN-2020' (NSE). Month names
    are mapped explicitly: pandas' %b is locale dependent.
  * Both NSE endpoints silently cap a response at 70 rows (the same cap as the
    bulk-deals JSON). A one-year VIX request returns Jan..mid-April and nothing
    else, with no error. Requests are chunked into 90-day windows (<= 66 sessions).
  * The index names the endpoints accept are NOT the display names: Smallcap 100 is
    'NIFTY SMLCAP 100', Financial Services is 'NIFTY FIN SERVICE', Infrastructure is
    'NIFTY INFRA', India Consumption is 'NIFTY CONSUMPTION'.
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.core.logging import get_logger

__all__ = ["IndexValuationFetcher", "VALUATION_INDICES", "parse_day"]

log = get_logger(__name__)

_NI = "https://www.niftyindices.com"
_NI_REFERER = _NI + "/reports/historical-data"
_NI_PEPB = _NI + "/BackPage/getpepbHistoricaldataDBtoString"
_NI_TRI = _NI + "/BackPage/getTotalReturnIndexString"
_NI_OHLC = _NI + "/BackPage/getHistoricaldatatabletoString"

_NSE = "https://www.nseindia.com"
_NSE_YIELD = _NSE + "/api/historicalOR/indicesYield"
_NSE_VIX = _NSE + "/api/historicalOR/vixhistory"

_DBNOMICS = "https://api.db.nomics.world/v22/series/{code}?observations=1&format=json"
# IMF IFS 'Government bond yield' for India runs 1960-01..2017-05; OECD's
# FINMARK long-term (10y benchmark) series starts 2011-12 and is current. The
# overlap (2011-12..2017-05) is used to verify the splice before trusting it.
GSEC_SERIES = {
    "IMF_IFS": "IMF/IFS/M.IN.FIGB_PA",
    "OECD_FINMARK": "OECD/DSD_STES@DF_FINMARK/IND.M.IRLT.PA._Z._Z._Z._Z.N",
}

# endpoint name -> label used everywhere else (matches index_data.index_name style)
VALUATION_INDICES: dict[str, str] = {
    "NIFTY 50": "Nifty 50",
    "NIFTY NEXT 50": "Nifty Next 50",
    "NIFTY 100": "Nifty 100",
    "NIFTY 200": "Nifty 200",
    "NIFTY 500": "Nifty 500",
    "NIFTY MIDCAP 100": "Nifty Midcap 100",
    "NIFTY MIDCAP 150": "Nifty Midcap 150",
    "NIFTY SMLCAP 100": "Nifty Smallcap 100",
    "NIFTY SMLCAP 250": "Nifty Smallcap 250",
    "NIFTY BANK": "Nifty Bank",
    "NIFTY FIN SERVICE": "Nifty Financial Services",
    "NIFTY PSU BANK": "Nifty PSU Bank",
    "NIFTY IT": "Nifty IT",
    "NIFTY FMCG": "Nifty FMCG",
    "NIFTY PHARMA": "Nifty Pharma",
    "NIFTY AUTO": "Nifty Auto",
    "NIFTY METAL": "Nifty Metal",
    "NIFTY REALTY": "Nifty Realty",
    "NIFTY ENERGY": "Nifty Energy",
    "NIFTY INFRA": "Nifty Infrastructure",
    "NIFTY MEDIA": "Nifty Media",
    "NIFTY CONSUMPTION": "Nifty India Consumption",
    "NIFTY COMMODITIES": "Nifty Commodities",
    "NIFTY PSE": "Nifty PSE",
    "NIFTY MNC": "Nifty MNC",
    "NIFTY SERV SECTOR": "Nifty Services Sector",
    # Added 2026-09-24. Endpoint names come from niftyindices' own
    # assets/json/IndexMapping.json (Trading_Index_Name); labels match index_data so
    # the as-published P/B override in the analytics layer can join on them.
    # Broad market
    "Nifty Midcap 50": "Nifty Midcap 50",
    "NIFTY SMLCAP 50": "Nifty Smallcap 50",
    "NIFTY MIDSML 400": "Nifty MidSmallcap 400",
    "NIFTY LARGEMID250": "Nifty LargeMidcap 250",
    "NIFTY TOTAL MKT": "Nifty Total Market",
    "NIFTY MICROCAP250": "Nifty Microcap 250",
    "Nifty Smallcap 500": "Nifty Smallcap 500",
    "Nifty Next 100": "Nifty Next 100",
    # Sectoral
    "Nifty Pvt Bank": "Nifty Private Bank",
    "NIFTY HEALTHCARE": "Nifty Healthcare Index",
    "NIFTY CONSR DURBL": "Nifty Consumer Durables",
    "NIFTY OIL AND GAS": "Nifty Oil & Gas",
    "NIFTY FINSRV25 50": "Nifty Financial Services 25/50",
    "Nifty FinSerExBnk": "Nifty Financial Services Ex-Bank",
    "Nifty MS Fin Serv": "Nifty MidSmall Financial Services",
    "NIFTY MIDSML HLTH": "Nifty MidSmall Healthcare",
    "Nifty MS IT Telcm": "Nifty MidSmall IT & Telecom",
    "Nifty Chemicals": "Nifty Chemicals",
    "Nifty Cement": "Nifty Cement",
    "Nifty Power": "Nifty Power",
    "Nifty NBFC": "Nifty NBFC",
    "Nifty Insurance": "Nifty Insurance",
    "Nifty Capital Goods": "Nifty Capital Goods",
    "Nifty Hospitals": "Nifty Hospitals",
    "Nifty Retail": "Nifty Retail",
    "Nifty Telecommunications": "Nifty Telecommunications",
    "Nifty Construction": "Nifty Construction",
    "Nifty Housing Finance": "Nifty Housing Finance",
    "Nifty Consumer Services": "Nifty Consumer Services",
    "Nifty Commercial & Transport Services": "Nifty Commercial & Transport Services",
    # Thematic
    "Nifty CPSE": "Nifty CPSE",
    "Nifty Capital Mkt": "Nifty Capital Markets",
    "Nifty Ind Defence": "Nifty India Defence",
    "Nifty Housing": "Nifty Housing",
    "Nifty CoreHousing": "Nifty Core Housing",
    "Nifty Mobility": "Nifty Mobility",
    "NIFTY IND DIGITAL": "Nifty India Digital",
    "NIFTY INDIA MFG": "Nifty India Manufacturing",
    "Nifty Trans Logis": "Nifty Transportation & Logistics",
    "Nifty Ind Tourism": "Nifty India Tourism",
    "Nifty India Railways PSU": "Nifty India Railways PSU",
    "Nifty EV": "Nifty EV & New Age Automotive",
    "Nifty MS Ind Cons": "Nifty MidSmall India Consumption",
    "Nifty New Consump": "Nifty India New Age Consumption",
    "Nifty REITs & Realty": "Nifty REITs & Realty",
}


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def parse_day(s: pd.Series) -> pd.Series:
    """'01 Jan 1999' or '01-JAN-2020' -> datetime64, independent of locale."""
    parts = (s.astype(str).str.strip().str.upper()
             .str.replace("-", " ", regex=False).str.split(r"\s+", n=2, expand=True))
    if parts.shape[1] < 3:
        return pd.Series(pd.NaT, index=s.index)
    return pd.to_datetime({"year": pd.to_numeric(parts[2], errors="coerce"),
                           "month": parts[1].str[:3].map(_MONTHS),
                           "day": pd.to_numeric(parts[0], errors="coerce")},
                          errors="coerce")


def _num(s: pd.Series) -> pd.Series:
    """'-' / '' / 'NA' -> NaN, '1,234.5' -> 1234.5, and non-positive -> NaN.

    Every quantity here (P/E, P/B, yield, index level) is strictly positive when
    it is published at all; a 0 is a missing value, not a reading.
    """
    v = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                      errors="coerce")
    return v.where(v > 0)


class IndexValuationFetcher:
    """Year-at-a-time pulls. Every method returns a DataFrame keyed by trade_date."""

    def __init__(self, polite_delay: float = 0.4) -> None:
        self._delay = polite_delay
        self._ni: requests.Session | None = None
        self._nse: requests.Session | None = None

    # ── sessions ────────────────────────────────────────────────────────────
    def _ni_session(self) -> requests.Session:
        if self._ni is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": _UA,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Content-Type": "application/json; charset=UTF-8",
                "Origin": _NI, "Referer": _NI_REFERER,
                "X-Requested-With": "XMLHttpRequest"})
            s.get(_NI_REFERER, timeout=30)
            self._ni = s
        return self._ni

    def _nse_session(self, referer_page: str) -> requests.Session:
        if self._nse is None:
            s = requests.Session()
            s.headers.update({"User-Agent": _UA, "Accept": "*/*",
                              "Accept-Language": "en-US,en;q=0.9"})
            s.get(_NSE + "/", timeout=30)
            self._nse = s
        self._nse.headers["Referer"] = f"{_NSE}/{referer_page}"
        return self._nse

    def _ni_post(self, url: str, name: str, start: date, end: date) -> list[dict]:
        ci = "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (
            name, start.strftime("%d-%b-%Y"), end.strftime("%d-%b-%Y"), name)
        for attempt in range(3):
            try:
                r = self._ni_session().post(url, json={"cinfo": ci}, timeout=60)
                r.raise_for_status()
                d = r.json()
                time.sleep(self._delay)
                if isinstance(d, dict) and "d" in d:
                    d = json.loads(d["d"]) if isinstance(d["d"], str) else d["d"]
                return d or []
            except Exception as exc:          # transient: back off, fresh cookies
                log.warning("niftyindices %s %s %s..%s attempt %d: %s",
                            url.rsplit("/", 1)[-1], name, start, end, attempt + 1, exc)
                self._ni = None
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"niftyindices failed 3x: {name} {start}..{end}")

    # ── niftyindices ────────────────────────────────────────────────────────
    def fetch_pepb(self, name: str, start: date, end: date) -> pd.DataFrame:
        rows = self._ni_post(_NI_PEPB, name, start, end)
        if not rows:
            return pd.DataFrame(columns=["trade_date", "pe", "pb", "dy"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"trade_date": parse_day(df["DATE"]).dt.date,
                             "pe": _num(df["pe"]), "pb": _num(df["pb"]),
                             "dy": _num(df["divYield"])}).dropna(subset=["trade_date"])

    def fetch_tri(self, name: str, start: date, end: date) -> pd.DataFrame:
        rows = self._ni_post(_NI_TRI, name, start, end)
        if not rows:
            return pd.DataFrame(columns=["trade_date", "tri"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"trade_date": parse_day(df["Date"]).dt.date,
                             "tri": _num(df["TotalReturnsIndex"])}
                            ).dropna(subset=["trade_date"])

    def fetch_close(self, name: str, start: date, end: date) -> pd.DataFrame:
        rows = self._ni_post(_NI_OHLC, name, start, end)
        if not rows:
            return pd.DataFrame(columns=["trade_date", "close_val"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"trade_date": parse_day(df["HistoricalDate"]).dt.date,
                             "close_val": _num(df["CLOSE"])}
                            ).dropna(subset=["trade_date"])

    def fetch_year(self, name: str, year: int, today: date | None = None) -> pd.DataFrame:
        """P/E, P/B, DY, close and TRI for one index and one calendar year."""
        today = today or date.today()
        start, end = date(year, 1, 1), min(date(year, 12, 31), today)
        v = self.fetch_pepb(name, start, end)
        if v.empty:
            return pd.DataFrame()
        c = self.fetch_close(name, start, end)
        t = self.fetch_tri(name, start, end)
        out = (v.merge(c, on="trade_date", how="left")
                .merge(t, on="trade_date", how="left"))
        out.insert(1, "index_name", VALUATION_INDICES.get(name, name))
        # the site returns newest-first and can repeat a date across a year edge
        return (out.drop_duplicates(["trade_date", "index_name"])
                   .sort_values("trade_date").reset_index(drop=True))

    # ── NSE (cross-check + VIX) ─────────────────────────────────────────────
    @staticmethod
    def _windows(start: date, end: date, days: int = 90):
        from datetime import timedelta
        a = start
        while a <= end:
            b = min(a + timedelta(days=days - 1), end)
            yield a, b
            a = b + timedelta(days=1)

    def fetch_nse_yield(self, name: str, start: date, end: date) -> pd.DataFrame:
        parts = [self._nse_yield_once(name, a, b) for a, b in self._windows(start, end)]
        return pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")

    def fetch_vix(self, start: date, end: date) -> pd.DataFrame:
        parts = [self._vix_once(a, b) for a, b in self._windows(start, end)]
        return pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")

    def _nse_yield_once(self, name: str, start: date, end: date) -> pd.DataFrame:
        s = self._nse_session("reports-indices-yield")
        r = s.get(_NSE_YIELD, params={"indexType": name.upper(),
                                      "from": start.strftime("%d-%m-%Y"),
                                      "to": end.strftime("%d-%m-%Y")}, timeout=60)
        r.raise_for_status()
        time.sleep(self._delay)
        rows = r.json().get("data") or []
        if len(rows) >= 70:
            raise RuntimeError(f"indicesYield {name} {start}..{end}: 70-row cap hit")
        if not rows:
            return pd.DataFrame(columns=["trade_date", "pe", "pb", "dy"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"trade_date": parse_day(df["IY_DT"]).dt.date,
                             "pe": _num(df["IY_PE"]), "pb": _num(df["IY_PB"]),
                             "dy": _num(df["IY_DY"])})

    def _vix_once(self, start: date, end: date) -> pd.DataFrame:
        s = self._nse_session("reports-indices-historical-vix")
        r = s.get(_NSE_VIX, params={"from": start.strftime("%d-%m-%Y"),
                                    "to": end.strftime("%d-%m-%Y")}, timeout=60)
        r.raise_for_status()
        time.sleep(self._delay)
        rows = r.json().get("data") or []
        if len(rows) >= 70:
            raise RuntimeError(f"VIX {start}..{end}: 70-row cap hit, window too wide")
        if not rows:
            return pd.DataFrame(columns=["trade_date", "vix"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"trade_date": parse_day(df["EOD_TIMESTAMP"]).dt.date,
                             "vix": _num(df["EOD_CLOSE_INDEX_VAL"])})

    # ── 10y G-sec ───────────────────────────────────────────────────────────
    @staticmethod
    def fetch_gsec_monthly() -> pd.DataFrame:
        """Monthly 10y G-sec yield (%), one row per (source, month)."""
        frames = []
        for src, code in GSEC_SERIES.items():
            r = requests.get(_DBNOMICS.format(code=code),
                             headers={"User-Agent": _UA}, timeout=(15, 150))
            r.raise_for_status()
            doc = r.json()["series"]["docs"][0]
            df = pd.DataFrame({"period": doc["period"], "value": doc["value"]})
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["month"] = pd.to_datetime(df["period"] + "-01", errors="coerce").dt.date
            df["source"] = src
            frames.append(df.dropna(subset=["month", "value"])[["source", "month", "value"]])
        out = pd.concat(frames, ignore_index=True)
        out["value"] = out["value"].astype(float).replace([np.inf, -np.inf], np.nan)
        return out.dropna()
