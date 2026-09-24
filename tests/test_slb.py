"""Offline tests for the SLB + corporate-action parsers (no network)."""
from datetime import date

from src.ingestion.corp_actions_fetcher import parse_dividend_amount
from src.ingestion.slb_fetcher import (parse_bhavcopy, parse_foreclosure,
                                       parse_open_positions, series_kind)

import pandas as pd

BHAV = (
    "ALOK INDUSTRIES LIMITED       ,ALOKINDS  ,O1,06-OCT-2026,N,0000000.15,0000000.15,0000000.20,"
    "0000000.15,0000000.20,    ,        834257,         165776.40,0000000.20,0000000.05,22-SEP-2026,      56\n"
    "ACC LIMITED                   ,ACC       ,XN,03-NOV-2026,N,0000001.00,0000001.00,0000001.00,"
    "0000001.00,0000001.00,    ,         11755,          11755.00,0000001.00,0000001.00,22-SEP-2026,      14\n"
    "BAJAJ AUTO LIMITED            ,BAJAJ-AUTO,02,01-FEB-2018,N,0000001.00,0000001.25,0000002.00,"
    "0000001.25,0000002.00,    ,          4952,           7602.20,0000003.00,0000001.00,15-JAN-2018,     6\n"
)


def test_bhavcopy_headerless_columns():
    b = parse_bhavcopy(BHAV)
    assert len(b) == 3
    r = b.set_index("symbol").loc["ALOKINDS"]
    assert r["trade_date"] == date(2026, 9, 22)
    assert r["expiry_date"] == date(2026, 10, 6)          # first TUESDAY, read from the file
    assert r["quantity"] == 834257 and r["trades"] == 56
    assert r["low_fee"] <= r["value"] / r["quantity"] <= r["high_fee"]


def test_series_stays_a_string():
    b = parse_bhavcopy(BHAV)
    assert b.set_index("symbol").loc["BAJAJ-AUTO", "series"] == "02"   # never the int 2


def test_series_kind():
    s = pd.Series(["01", "12", "10", "X1", "XO", "XN", "XD", "O1", "OD", "F1", "13"])
    assert series_kind(s).tolist() == [
        "REGULAR", "REGULAR", "REGULAR", "NO_AGM", "NO_AGM", "NO_AGM", "NO_AGM",
        "ROLLOVER", "ROLLOVER", "ROLLOVER", "ROLLOVER"]


def test_open_positions():
    text = ("Sr no,Security,Series,Outstanding Quantity at the end of the day\n"
            "1,360ONE,X1,2\n2,360ONE,XN,7587\n3,ASHOKLEY,10,1\n")
    o = parse_open_positions(text, date(2026, 9, 22))
    assert o.groupby("symbol")["open_qty"].sum().to_dict() == {"360ONE": 7589, "ASHOKLEY": 1}
    assert set(o["series"]) == {"X1", "XN", "10"}


def test_foreclosure_drops_legend():
    text = ("SECURITY,SERIES**,ISIN,ANNOUNCEMENT DATE,BOOK CLOSURE START DATE,RECORD DATE,EX DATE,"
            "FORECLOSURE DATE,SHUT PERIOD START DATE,SHUT PERIOD END DATE,NEXT TRADE DATE,"
            "FORECLOSURE SETTLEMENT NO,CORPORATE ACTION DESCRIPTION\n"
            "PGIL,ALL,INE940H01022,08-SEP-2026,,11-SEP-2026,11-SEP-2026,11-SEP-2026,02-SEP-2026,"
            "11-SEP-2026,15-SEP-2026,2026860,BONUS 1:1                 BONUS 1:1\n"
            "\n\nSERIES**\nOLD,Series-01-12\nNEW,Series-X1-XD\n")
    f = parse_foreclosure(text)
    assert f["symbol"].tolist() == ["PGIL"]
    assert f.loc[0, "record_date"] == date(2026, 9, 11)
    assert f.loc[0, "description"] == "BONUS 1:1"


def test_dividend_amount_ignores_rights_and_face_value():
    assert parse_dividend_amount("Interim Dividend- Rs 3 Per Share / Rights 1:16 @ Premium Rs 554 Per Share") == 3
    assert parse_dividend_amount("Annual General Meeting/Dividend - Rs 10 Per Share/Face Value Split "
                                 "(Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share") == 10
    assert parse_dividend_amount("Dividend - Rs 525 Per Share & Special Dividend Rs 131 Per Share") == 656
    assert parse_dividend_amount("Annual General Meeting/Dividend - Re 1/- Per Share") == 1
    assert parse_dividend_amount("Dividend - 0.40 Per Share") == 0.4
    assert parse_dividend_amount("Bonus 1:1") == 0
