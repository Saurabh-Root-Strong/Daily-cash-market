"""
F&O parser + signal regression tests.

Pure-function locks for the bugs fixed during the UDiFF migration. Every test
here corresponds to a real defect that shipped because there was no test:
  - UDiFF settle_price was mapped to same-day settlement (→ 0% price change →
    everything mis-classified Neutral). Must map to PrvsClsgPric (prev close).
  - Options put-leg sentiment was coloured with call-side substring logic
    (P.Writing showed red / bearish). Puts invert.
  - feat_fii_delta divisor 100K saturated the feature at the -1 clip.
None of these need a database — they exercise the classifiers directly.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest


# ── UDiFF parser ─────────────────────────────────────────────────────────────

_UDIFF_COLS = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)


def _udiff_row(**ov) -> str:
    d = dict(
        TradDt="2026-06-12", BizDt="2026-06-12", Sgmt="FO", Src="NSE",
        FinInstrmTp="STF", FinInstrmId="1", ISIN="X", TckrSymb="ICICIBANK",
        SctySrs="FUT", XpryDt="2026-06-30", FininstrmActlXpryDt="2026-06-30",
        StrkPric="0", OptnTp="", FinInstrmNm="ICICIBANK", OpnPric="1320",
        HghPric="1350", LwPric="1318", ClsPric="1344.2", LastPric="1344",
        PrvsClsgPric="1319.2", UndrlygPric="1344", SttlmPric="1344.2",
        OpnIntrst="137114600", ChngInOpnIntrst="-3502100", TtlTradgVol="100",
        TtlTrfVal="1000", TtlNbOfTxsExctd="10", SsnId="", NewBrdLotQty="700",
        Rmks="", Rsvd1="", Rsvd2="", Rsvd3="", Rsvd4="",
    )
    d.update(ov)
    return ",".join(str(d[k]) for k in _UDIFF_COLS.split(","))


def _udiff_csv(rows) -> str:
    return _UDIFF_COLS + "\n" + "\n".join(rows) + "\n"


def test_udiff_detected_and_settle_is_previous_close():
    from src.ingestion.fno_bhavcopy_fetcher import _is_udiff, _parse_fno_udiff

    csv = _udiff_csv([_udiff_row()])
    assert _is_udiff(csv)

    df = _parse_fno_udiff(csv, date(2026, 6, 12))
    assert len(df) == 1
    row = df.iloc[0]
    # THE regression: settle_price must be PrvsClsgPric (1319.2), NOT SttlmPric (1344.2).
    assert row["settle_price"] == pytest.approx(1319.2)
    assert row["close_price"] == pytest.approx(1344.2)
    # So (close - settle)/settle is the real +1.9% move, not 0%.
    pc = (row["close_price"] - row["settle_price"]) / row["settle_price"] * 100
    assert pc == pytest.approx(1.9, abs=0.1)


def test_udiff_instrument_type_mapping():
    from src.ingestion.fno_bhavcopy_fetcher import _parse_fno_udiff

    rows = [
        _udiff_row(FinInstrmTp="STF", TckrSymb="A"),
        _udiff_row(FinInstrmTp="IDF", TckrSymb="NIFTY"),
        _udiff_row(FinInstrmTp="STO", TckrSymb="B", OptnTp="CE", StrkPric="1300"),
        _udiff_row(FinInstrmTp="IDO", TckrSymb="NIFTY", OptnTp="PE", StrkPric="25000"),
    ]
    df = _parse_fno_udiff(_udiff_csv(rows), date(2026, 6, 12))
    assert set(df["instrument"]) == {"FUTSTK", "FUTIDX", "OPTSTK", "OPTIDX"}
    # futures get option_type XX; options keep CE/PE
    fut = df[df["instrument"] == "FUTSTK"].iloc[0]
    assert fut["option_type"] == "XX"


# ── Futures OI × price matrix ────────────────────────────────────────────────

def test_fut_signal_quadrants_and_deadzones():
    from src.analytics.fno_stocks import _fut_signal

    assert _fut_signal(1.0, 5.0) == "Long Buildup"      # price↑ OI↑
    assert _fut_signal(-1.0, 5.0) == "Short Buildup"    # price↓ OI↑
    assert _fut_signal(1.0, -5.0) == "Short Covering"   # price↑ OI↓
    assert _fut_signal(-1.0, -5.0) == "Long Unwinding"  # price↓ OI↓
    # dead zones (|price|<=0.1 or |OI|<=0.5) collapse to Neutral
    assert _fut_signal(0.05, 5.0) == "Neutral"
    assert _fut_signal(1.0, 0.2) == "Neutral"
    assert _fut_signal(None, 5.0) == "Neutral"


# ── Options OI-premium matrix + sentiment ────────────────────────────────────

def test_opt_oi_prem_matrix():
    from src.analytics.fno_stocks import _opt_oi_prem_signal

    assert _opt_oi_prem_signal(5.0, 1.0, "CE") == "C.Buying"    # OI↑ prem↑
    assert _opt_oi_prem_signal(5.0, -1.0, "CE") == "C.Writing"  # OI↑ prem↓
    assert _opt_oi_prem_signal(-5.0, 1.0, "PE") == "P.SC"       # OI↓ prem↑
    assert _opt_oi_prem_signal(-5.0, -1.0, "PE") == "P.LE"      # OI↓ prem↓
    assert _opt_oi_prem_signal(None, 1.0, "CE") == "—"


def test_opt_sentiment_puts_invert_calls():
    from src.analytics.fno_stocks import _opt_sentiment

    # calls: buying/short-cover bullish, writing/long-exit bearish
    assert _opt_sentiment("C.Buying") == 1
    assert _opt_sentiment("C.SC") == 1
    assert _opt_sentiment("C.Writing") == -1
    assert _opt_sentiment("C.LE") == -1
    # puts INVERT: writing/long-exit bullish, buying/short-cover bearish
    assert _opt_sentiment("P.Writing") == 1
    assert _opt_sentiment("P.LE") == 1
    assert _opt_sentiment("P.Buying") == -1
    assert _opt_sentiment("P.SC") == -1
    assert _opt_sentiment("C.Neutral") == 0


def test_combined_opt_label_colour_and_buckets():
    from src.analytics.fno_stocks import _combined_opt_label

    # The reported bug: a lone put-writing leg must read bullish (green).
    assert _combined_opt_label("C.Neutral", "P.Writing", 0.6).startswith("\U0001F7E2")  # 🟢
    assert _combined_opt_label("C.Neutral", "P.Buying", 1.4).startswith("\U0001F534")   # 🔴
    assert _combined_opt_label("C.Neutral", "P.SC", 0.9).startswith("\U0001F534")       # 🔴
    assert _combined_opt_label("C.Neutral", "P.LE", 0.8).startswith("\U0001F7E2")       # 🟢
    # combined directional / structure buckets
    assert "Bull" in _combined_opt_label("C.Buying", "P.Writing", 0.4)
    assert "Bear" in _combined_opt_label("C.Writing", "P.Buying", 1.6)
    assert "Vol Bet" in _combined_opt_label("C.Buying", "P.Buying", 1.0)
    assert "Range" in _combined_opt_label("C.Writing", "P.Writing", 0.9)


# ── Memory fingerprint feature scaling ───────────────────────────────────────

def _mk_pred(call_net: int, put_net: int):
    ctx = SimpleNamespace(
        fii_fut_idx_net=0, fii_cumul_flows_5d={},
        fii_opt_call_net=call_net, fii_opt_put_net=put_net,
        vix_close=15.0, vix_5d_chg_pct=0.0, breadth_pct_advancing=50.0,
    )
    r = SimpleNamespace(error=False, memory_avg=0.5, perm_entropy=0.9)
    return SimpleNamespace(
        market_context=ctx, regime=r, pcr=0.9, spot_close=23000.0,
        levels=SimpleNamespace(max_pain=23000.0), carry_pct_ann=0.0,
        fno_symbol="NIFTY", composite_score=0.0,
    )


def test_feat_fii_delta_not_clipped_at_realistic_magnitude():
    from src.analytics.memory_engine import extract_features

    # call_net - put_net = -751,632 (a realistic structurally put-heavy book)
    feats = extract_features(_mk_pred(call_net=-209_020, put_net=542_612))
    # -751632 / 800000 = -0.9395 — must NOT be pinned at the -1.0 clip (the bug).
    assert feats["feat_fii_delta"] == pytest.approx(-0.9395, abs=0.01)
    assert feats["feat_fii_delta"] > -1.0
