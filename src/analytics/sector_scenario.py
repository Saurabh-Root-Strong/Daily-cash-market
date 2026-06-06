"""
Market-scenario engine + sector F&O OI-buildup — the regime PLAYBOOK.

Different factors win in different market scenarios. Momentum leads in an uptrend
but is a trap in a downtrend; delivery-accumulation shines in a range; the biggest
reward is buying beaten-down sectors that smart money accumulates at a bottom.

classify_scenario() detects WHERE the market is AND where it is HEADED (the
transition) from trend + acceleration + breadth-trend + volatility + FII flow,
maps it to one of 7 scenarios, and returns the ranking FACTOR the dashboard
should use ("momentum" | "defense" | "accumulation" | "reversal").

get_sector_fno_buildup() classifies each sector's stock-futures positioning
(long/short buildup, short covering, long unwinding) — the F&O confirmation layer.

Everything is point-in-time: only data <= as_of_date is used.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["classify_scenario", "get_sector_fno_buildup", "SCENARIO_FACTOR"]

# Scenario → the factor the sector ranking should optimise for.
SCENARIO_FACTOR = {
    "STRONG_UPTREND":       "momentum",
    "UPTREND_TOPPING":      "defense",
    "SIDEWAYS":             "accumulation",
    "SIDEWAYS_BREAKDOWN":   "defense",
    "STRONG_DOWNTREND":     "defense",
    "DOWNTREND_BOTTOMING":  "reversal",
    "SIDEWAYS_BREAKOUT":    "momentum",
}

_SCENARIO_PLAY = {
    "STRONG_UPTREND": ("📈 Strong Uptrend",
        "Trend + breadth + flows aligned up. Buy the momentum leaders — highest "
        "relative strength with institutional accumulation. Ride winners, full size."),
    "UPTREND_TOPPING": ("⚠️ Uptrend Losing Steam",
        "Price still up but momentum fading / breadth narrowing / VIX ticking up. "
        "Rotate from high-beta into quality + defensives, book profits, tighten stops."),
    "SIDEWAYS": ("🔵 Sideways / Range",
        "No market trend — sector picking IS the edge. Favour sectors with sustained "
        "DELIVERY ACCUMULATION (Secret/Confirmed) and positive relative strength; "
        "long-leaders / short-laggards pairs."),
    "SIDEWAYS_BREAKDOWN": ("🟠 Range Breaking Down",
        "Range support breaking, breadth deteriorating, FII selling. Shift to defense "
        "early — exit weak sectors before the downtrend confirms."),
    "STRONG_DOWNTREND": ("🔴 Strong Downtrend",
        "Trend down, flows out. Capital protection mode — only low-downside-capture "
        "defensive sectors; everything else amplifies the fall. Pairs or cash."),
    "DOWNTREND_BOTTOMING": ("🟢 Downtrend Bottoming (highest reward)",
        "Selling is exhausting — VIX peaking, breadth improving, FII covering, and "
        "smart money is taking DELIVERY into the weakness. Buy the beaten-down sectors "
        "showing Secret Accumulation; this is the best risk/reward entry."),
    "SIDEWAYS_BREAKOUT": ("🚀 Range Breaking Out",
        "Range resolving up with expanding breadth + FII longs. Buy the breakout "
        "leaders with volume + delivery confirmation."),
}


def _nifty_trend(as_of_date: date) -> dict:
    df = query_dataframe("""
        SELECT trade_date, close_val FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 60
    """, [as_of_date])
    if df.empty or len(df) < 22:
        return {}
    df = df.sort_values("trade_date").reset_index(drop=True)
    c = df["close_val"].astype(float)
    latest = float(c.iloc[-1])
    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1]) if len(c) >= 50 else ema20

    def _ret(n):  # trailing n-trading-day % return
        return (latest / float(c.iloc[-1 - n]) - 1) * 100 if len(c) > n else 0.0
    r5, r10, r20 = _ret(5), _ret(10), _ret(20)
    # Acceleration: recent 5d pace vs the 20d average pace.
    accel = r5 - (r20 / 4.0)
    return {
        "price_vs_ema20": latest > ema20,
        "ema_golden":     ema20 > ema50,
        "r5": round(r5, 2), "r10": round(r10, 2), "r20": round(r20, 2),
        "accel": round(accel, 2),
    }


def _breadth_trend(as_of_date: date) -> dict:
    df = query_dataframe("""
        SELECT trade_date,
               SUM(CASE WHEN close_price > prev_close THEN 1 ELSE 0 END) * 1.0
               / NULLIF(COUNT(*), 0) AS breadth
        FROM daily_data
        WHERE series = 'EQ' AND prev_close > 0 AND turnover_lacs >= 100
          AND trade_date <= ? AND trade_date > (? - INTERVAL 20 DAY)
        GROUP BY trade_date ORDER BY trade_date DESC
    """, [as_of_date, as_of_date])
    if df.empty or len(df) < 6:
        return {}
    b = df["breadth"].astype(float)
    recent = float(b.iloc[:5].mean())                       # last 5 days
    prior  = float(b.iloc[5:10].mean()) if len(b) >= 10 else recent
    return {"breadth": round(recent, 3), "breadth_trend": round(recent - prior, 3)}


def classify_scenario(as_of_date: date) -> dict:
    """
    Classify the market into one of 7 scenarios and return the recommended ranking
    factor + playbook. Reuses get_market_regime for VIX / FII / regime context.
    """
    from src.analytics.sector_rotation import get_market_regime
    reg = get_market_regime(as_of_date)
    t = _nifty_trend(as_of_date)
    b = _breadth_trend(as_of_date)

    vix       = reg.get("vix")
    vix_trend = reg.get("vix_trend", "STABLE")
    fii_5d    = reg.get("fii_5d_cr") or 0.0
    breadth   = b.get("breadth", 0.5)
    brd_trend = b.get("breadth_trend", 0.0)
    up_ema    = t.get("price_vs_ema20", False)
    golden    = t.get("ema_golden", False)
    r5        = t.get("r5", 0.0); r20 = t.get("r20", 0.0); accel = t.get("accel", 0.0)

    # ── Trend state ───────────────────────────────────────────────────────────
    if up_ema and golden:
        trend = "UP"
    elif (not up_ema) and (not golden):
        trend = "DOWN"
    else:
        trend = "FLAT"

    # ── Scenario decision (state + transition) ────────────────────────────────
    if trend == "UP":
        topping = (r5 < 0) or (brd_trend < -0.05) or (vix_trend == "RISING") or (fii_5d < -3000)
        scenario = "UPTREND_TOPPING" if topping else "STRONG_UPTREND"
    elif trend == "DOWN":
        # Bottoming = price still below EMAs but momentum + flow + fear turning up.
        bottoming = (r5 > 0.5 and (brd_trend > 0.03 or vix_trend == "FALLING")) or \
                    (r5 > 1.0 and fii_5d > 0)
        scenario = "DOWNTREND_BOTTOMING" if bottoming else "STRONG_DOWNTREND"
    else:  # FLAT / range
        if r5 > 1.0 and brd_trend > 0.03 and fii_5d > 0:
            scenario = "SIDEWAYS_BREAKOUT"
        elif r5 < -1.0 and (brd_trend < -0.03 or fii_5d < -2000):
            scenario = "SIDEWAYS_BREAKDOWN"
        else:
            scenario = "SIDEWAYS"

    label, play = _SCENARIO_PLAY[scenario]
    return {
        "scenario":       scenario,
        "label":          label,
        "playbook":       play,
        "ranking_factor": SCENARIO_FACTOR[scenario],
        "trend":          trend,
        "regime":         reg.get("regime"),
        "nifty_5d":       r5, "nifty_20d": r20, "accel": accel,
        "breadth":        breadth, "breadth_trend": brd_trend,
        "vix":            vix, "vix_trend": vix_trend, "fii_5d_cr": fii_5d,
    }


def get_sector_fno_buildup(
    as_of_date: date, min_turnover_lacs: float = 1.0,
) -> pd.DataFrame:
    """
    Per-sector stock-futures OI buildup vs the prior session.

    Classifies the dominant F&O stance per sector by aggregating constituent stock
    futures (price change × OI change), matched on SAME expiry to avoid rollover
    noise. Returns DataFrame[sector, fno_stance, fut_oi_chg_pct, price_chg_pct,
    n_fut]. Stance: Long Buildup / Short Buildup / Short Covering / Long Unwinding.
    """
    # TOTAL OI per symbol (all expiries) — robust to the early June→July rollover
    # that drains the near-month and to per-contract data gaps. Per-stock OI change
    # clipped ±50% and prev-OI-weighted; near-month settle drives the price move.
    # Only sectors with >=5 futures names get a stance (small samples are noise).
    df = query_dataframe("""
        WITH td  AS (SELECT MAX(trade_date) d FROM fno_bhavcopy WHERE trade_date <= ?),
        ptd AS (SELECT MAX(trade_date) d FROM fno_bhavcopy WHERE trade_date < (SELECT d FROM td)),
        near AS (
            SELECT MIN(expiry_date) e FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND trade_date = (SELECT d FROM td)
              AND expiry_date > (SELECT d FROM td)
        ),
        toi AS (SELECT symbol, SUM(open_interest) oi FROM fno_bhavcopy
                WHERE instrument='FUTSTK' AND trade_date=(SELECT d FROM td)
                GROUP BY symbol HAVING SUM(open_interest) > 0),
        poi AS (SELECT symbol, SUM(open_interest) prev_oi FROM fno_bhavcopy
                WHERE instrument='FUTSTK' AND trade_date=(SELECT d FROM ptd)
                GROUP BY symbol HAVING SUM(open_interest) > 0),
        tpr AS (SELECT symbol, settle_price FROM fno_bhavcopy
                WHERE instrument='FUTSTK' AND trade_date=(SELECT d FROM td) AND expiry_date=(SELECT e FROM near)),
        ppr AS (SELECT symbol, settle_price AS prev_price FROM fno_bhavcopy
                WHERE instrument='FUTSTK' AND trade_date=(SELECT d FROM ptd) AND expiry_date=(SELECT e FROM near))
        SELECT sm.sector,
               SUM(GREATEST(-50, LEAST(50, (toi.oi - poi.prev_oi) / poi.prev_oi * 100)) * poi.prev_oi)
                   / NULLIF(SUM(poi.prev_oi), 0)                 AS fut_oi_chg_pct,
               SUM(GREATEST(-25, LEAST(25, (tpr.settle_price - ppr.prev_price) / ppr.prev_price * 100)) * poi.prev_oi)
                   / NULLIF(SUM(poi.prev_oi), 0)                 AS price_chg_pct,
               COUNT(*)                                          AS n_fut
        FROM toi
        JOIN poi ON toi.symbol = poi.symbol
        JOIN tpr ON toi.symbol = tpr.symbol
        JOIN ppr ON toi.symbol = ppr.symbol
        JOIN v_sector_master sm ON toi.symbol = sm.symbol
        WHERE sm.sector NOT IN ('ETF', 'Others') AND ppr.prev_price > 0
        GROUP BY sm.sector
        HAVING COUNT(*) >= 5
    """, [as_of_date])
    if df.empty:
        return pd.DataFrame()

    df["fut_oi_chg_pct"] = df["fut_oi_chg_pct"].round(1)
    df["price_chg_pct"]  = df["price_chg_pct"].round(2)

    def _stance(r):
        oi_up = r["fut_oi_chg_pct"] is not None and r["fut_oi_chg_pct"] > 1.0
        oi_dn = r["fut_oi_chg_pct"] is not None and r["fut_oi_chg_pct"] < -1.0
        p_up  = r["price_chg_pct"] is not None and r["price_chg_pct"] > 0.1
        p_dn  = r["price_chg_pct"] is not None and r["price_chg_pct"] < -0.1
        if oi_up and p_up:  return "🟢 Long Buildup"          # new longs — bullish
        if oi_up and p_dn:  return "🔴 Short Buildup"         # new shorts — bearish
        if oi_dn and p_up:  return "🟡 Short Covering"        # shorts exiting — squeeze up
        if oi_dn and p_dn:  return "⚪ Long Unwinding"        # longs exiting — weak
        return "⚖️ Neutral"

    df["fno_stance"] = df.apply(_stance, axis=1)
    return df[["sector", "fno_stance", "fut_oi_chg_pct", "price_chg_pct", "n_fut"]]
