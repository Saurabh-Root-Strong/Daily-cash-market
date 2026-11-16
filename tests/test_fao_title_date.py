"""Guards against storing another date's NSE file (found 2026-11-15: 298 copied days)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.ingestion.fao_fetcher import FAOParticipantFetcher, title_date

_CSV = ('"Participant wise Open Interest (no. of contracts) in Equity Derivatives as on {d}",,,,,,,,,,,,,,,\n'
        "Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short,"
        "Option Index Call Long,Option Index Put Long,Option Index Call Short,Option Index Put Short,"
        "Option Stock Call Long,Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
        "Total Long Contracts,Total Short Contracts\n"
        "FII,100,200,1,1,1,1,1,1,1,1,1,1,1,1\n")


@pytest.mark.parametrize("line,expected", [
    ('""Participant wise Open Interest ... as on Feb 09, 2026"",,,', date(2026, 2, 9)),
    ('"Participant wise Open Interest ... as on January 03, 2012",,', date(2012, 1, 3)),
    ('"... as on Sept 30, 2024",', date(2024, 9, 30)),
    ("no date here", None),
])
def test_title_date_parses_both_archive_formats(line, expected):
    assert title_date(line) == expected


class _Client:
    """Serves a fixed body per URL substring, like NSEHttpClient.get_text."""
    def __init__(self, bodies):
        self.bodies = bodies

    def get_text(self, url, expect_404_ok=False):
        for k, v in self.bodies.items():
            if k in url:
                return v
        return None

    def get(self, *a, **k):
        return None


def test_fetcher_rejects_file_for_another_date_and_uses_next_source():
    d = date(2026, 2, 9)
    wrong = _CSV.format(d="Jan 09, 2026")
    right = _CSV.format(d="Feb 09, 2026").replace("FII,100,200", "FII,444,555")
    f = FAOParticipantFetcher(_Client({"nsearchives": wrong, "archives.nse": right, "api/reports": wrong}))
    text = f._fetch_csv(["https://nsearchives.x/oi", "https://archives.nseindia.com/oi"], "OI", d)
    assert text == right


def test_fetcher_returns_nothing_when_every_source_is_the_wrong_date():
    d = date(2026, 4, 9)
    wrong = _CSV.format(d="Jan 09, 2026")
    f = FAOParticipantFetcher(_Client({"": wrong}))
    assert f._fetch_csv(["https://nsearchives.x/oi", "https://api/reports"], "OI", d) is None


def test_fii_stats_parser_rejects_other_date():
    from src.core.exceptions import ParseError
    from src.ingestion import fii_stats_fetcher as m
    raw = pd.DataFrame([["FII DERIVATIVES STATISTICS FOR 09-Jan-2026", None, None, None, None, None, None]])
    assert m._date_from_title_row(raw) == date(2026, 1, 9)
    with pytest.raises(ParseError):
        # monkeypatch the Excel read so the date check runs on our frame
        orig = pd.ExcelFile
        class _X:
            sheet_names = ["s"]
            def __init__(self, *a, **k): pass
            def parse(self, *a, **k): return raw
        pd.ExcelFile = _X
        try:
            m._parse_fii_stats_xls(m._XLS_MAGIC + b"0" * 10, date(2026, 4, 9))
        finally:
            pd.ExcelFile = orig
