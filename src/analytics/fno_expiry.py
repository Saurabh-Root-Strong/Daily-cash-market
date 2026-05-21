"""
F&O Expiry Structure Analytics.

NSE expiry architecture (as of 2026):
  FUTURES (FUTSTK / FUTIDX):
    - 3 monthly expiries for all symbols (near / next / far month)
    - Expire on last Tuesday (NIFTY) / Wednesday (BANKNIFTY) / Thursday (stocks, others)

  OPTIONS — Stocks (OPTSTK):
    - 3 monthly expiries only (same dates as stock futures)

  OPTIONS — Indices (OPTIDX):
    - NIFTY       : weekly (every Tuesday) + 3 monthly + quarterly + long-dated
    - BANKNIFTY   : monthly + quarterly
    - FINNIFTY    : monthly only
    - MIDCPNIFTY  : monthly only
    - NIFTYNXT50  : monthly only

Expiry classification tiers:
  Weekly     — not last-of-weekday-in-month (NIFTY only)
  Near Month — nearest monthly expiry
  Next Month — 2nd monthly expiry
  Far Month  — 3rd monthly expiry
  Quarterly  — 4th or later monthly expiry within ~90 days
  Long Term  — expiry > 90 days out
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_stock_monthly_expiries",
    "get_stock_expiry_matrix",
    "get_index_full_structure",
    "get_options_chain",
    "get_index_options_chain",
]

_EXPIRY_TIER_ORDER = ["Near Month", "Next Month", "Far Month", "Quarterly", "Long Term", "Weekly"]


def _as_date(d) -> date:
    if isinstance(d, date) and not hasattr(d, "hour"):
        return d
    if hasattr(d, "date"):
        return d.date()
    return date.fromisoformat(str(d)[:10])


def _is_monthly_expiry(d: date) -> bool:
    """True when d is the last occurrence of its weekday in its calendar month."""
    return (d + timedelta(days=7)).month != d.month


def _classify_expiry_tier(
    d: date,
    trade_date: date,
    monthly_expiries: list[date],
) -> str:
    """
    Assign a human-readable tier label to a given expiry date.
    monthly_expiries must be sorted ascending.
    """
    if not _is_monthly_expiry(d):
        return "Weekly"
    dte = (d - trade_date).days
    try:
        rank = monthly_expiries.index(d)
    except ValueError:
        rank = len(monthly_expiries)

    if rank == 0:
        return "Near Month"
    if rank == 1:
        return "Next Month"
    if rank == 2:
        return "Far Month"
    if dte <= 120:
        return "Quarterly"
    return "Long Term"


def _compute_max_pain(opts: pd.DataFrame) -> float:
    """Max pain = strike where total option-writer payout is minimised."""
    calls = opts[opts["option_type"] == "CE"][["strike_price", "open_interest"]]
    puts  = opts[opts["option_type"] == "PE"][["strike_price", "open_interest"]]
    if calls.empty or puts.empty:
        return float("nan")
    candidates = np.unique(opts["strike_price"].values)
    best, min_pay = candidates[0], np.inf
    for p in candidates:
        c_pay  = ((p - calls["strike_price"]).clip(lower=0) * calls["open_interest"]).sum()
        pu_pay = ((puts["strike_price"] - p).clip(lower=0) * puts["open_interest"]).sum()
        total  = c_pay + pu_pay
        if total < min_pay:
            min_pay = total
            best = p
    return float(best)


def _opt_stats(opts: pd.DataFrame, spot: float) -> dict:
    """Compute PCR, max pain, mp_dist%, top OI strikes from a slice of opt rows."""
    call_oi = float(opts[opts["option_type"] == "CE"]["open_interest"].sum())
    put_oi  = float(opts[opts["option_type"] == "PE"]["open_interest"].sum())
    pcr     = round(put_oi / call_oi, 2) if call_oi > 0 else None
    mp      = _compute_max_pain(opts) if len(opts) >= 4 else float("nan")
    mp_dist = round((spot - mp) / mp * 100, 1) if (not np.isnan(mp) and mp > 0 and spot > 0) else None

    ce_top = opts[opts["option_type"] == "CE"].nlargest(1, "open_interest")
    pe_top = opts[opts["option_type"] == "PE"].nlargest(1, "open_interest")
    return {
        "call_oi":     int(call_oi),
        "put_oi":      int(put_oi),
        "pcr":         pcr,
        "max_pain":    round(mp, 0) if not np.isnan(mp) else None,
        "mp_dist_pct": mp_dist,
        "top_ce_strike": int(ce_top["strike_price"].iloc[0]) if not ce_top.empty else None,
        "top_pe_strike": int(pe_top["strike_price"].iloc[0]) if not pe_top.empty else None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_stock_monthly_expiries(trade_date: date) -> list[date]:
    """
    The 3 nearest monthly FUTSTK expiry dates as of trade_date.
    Returns [near_month, next_month, far_month] sorted ascending.
    """
    trade_date = _as_date(trade_date)
    df = query_dataframe(
        """
        SELECT DISTINCT expiry_date
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND instrument  = 'FUTSTK'
          AND expiry_date >= ?
        ORDER BY expiry_date
        LIMIT 3
        """,
        [trade_date, trade_date],
    )
    return [_as_date(d) for d in df["expiry_date"].tolist()] if not df.empty else []


def get_stock_expiry_matrix(
    trade_date: date,
    min_fut_oi: int = 50_000,
) -> pd.DataFrame:
    """
    One row per F&O stock with Near / Next / Far month stats side-by-side.

    Futures columns (prefix = near_ / next_ / far_):
        {x}_exp          expiry date
        {x}_fut_oi       open interest
        {x}_chg_oi       change in OI vs previous session
        {x}_settle       settlement price
        {x}_basis_pct    (settle - spot) / spot × 100

    Options columns (prefix = near_ / next_ / far_):
        {x}_call_oi      total call open interest
        {x}_put_oi       total put open interest
        {x}_pcr          put/call ratio
        {x}_max_pain     max pain strike
        {x}_mp_dist_pct  (spot - max_pain) / max_pain × 100

    Common columns:
        symbol, company_name, sector, industry, spot_price,
        roll_signal      Rolling Fwd / Building / Unwinding / Neutral
    """
    trade_date = _as_date(trade_date)
    expiries   = get_stock_monthly_expiries(trade_date)
    if not expiries:
        return pd.DataFrame()

    labels = ["near", "next", "far"]
    # Pad to exactly 3 so IN (?,?,?) always has 3 params
    exp_args = expiries + [date(2099, 1, 1)] * (3 - len(expiries))

    # ── Futures for all 3 expiries ────────────────────────────────────────────
    # Fetch ALL expiries without OI filter — next/far month naturally has lower OI
    # and would be wiped out if we applied min_fut_oi here.
    # We apply the min_fut_oi filter only to near month to decide which symbols qualify.
    fut = query_dataframe(
        """
        SELECT symbol, expiry_date,
               SUM(open_interest)  AS fut_oi,
               SUM(chg_in_oi)      AS chg_oi,
               MAX(settle_price)   AS settle,
               MAX(close_price)    AS fut_close
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND instrument  = 'FUTSTK'
          AND expiry_date IN (?, ?, ?)
        GROUP BY symbol, expiry_date
        """,
        [trade_date, *exp_args],
    )
    if fut.empty:
        return pd.DataFrame()
    fut["expiry_date"] = pd.to_datetime(fut["expiry_date"]).dt.date

    # Filter qualifying symbols by near-month OI only
    near_oi = fut[fut["expiry_date"] == expiries[0]]
    qualifying = near_oi[near_oi["fut_oi"] >= min_fut_oi]["symbol"].unique()
    fut = fut[fut["symbol"].isin(qualifying)]
    if fut.empty:
        return pd.DataFrame()

    # ── Spot prices ───────────────────────────────────────────────────────────
    cash = query_dataframe(
        """
        SELECT symbol, close_price AS spot_price
        FROM daily_data
        WHERE trade_date = (SELECT MAX(trade_date) FROM daily_data WHERE trade_date <= ?)
        """,
        [trade_date],
    )

    # ── Options for all 3 expiries ────────────────────────────────────────────
    opts = query_dataframe(
        """
        SELECT symbol, expiry_date, strike_price, option_type, open_interest
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND instrument  = 'OPTSTK'
          AND expiry_date IN (?, ?, ?)
          AND open_interest > 0
        """,
        [trade_date, *exp_args],
    )
    if not opts.empty:
        opts["expiry_date"] = pd.to_datetime(opts["expiry_date"]).dt.date

    # ── Sector master ─────────────────────────────────────────────────────────
    sec = query_dataframe("SELECT symbol, company_name, sector, industry FROM sector_master", [])
    sec_map = sec.set_index("symbol").to_dict("index")

    # ── Build per-stock rows ──────────────────────────────────────────────────
    symbols = fut[fut["expiry_date"] == expiries[0]]["symbol"].unique()

    records = []
    for sym in symbols:
        sf   = fut[fut["symbol"] == sym]
        so   = opts[opts["symbol"] == sym] if not opts.empty else pd.DataFrame()
        sr   = cash[cash["symbol"] == sym]
        spot = float(sr["spot_price"].iloc[0]) if not sr.empty else float("nan")
        meta = sec_map.get(sym, {})

        rec: dict = {
            "symbol":       sym,
            "company_name": meta.get("company_name", sym),
            "sector":       meta.get("sector",       "Unknown"),
            "industry":     meta.get("industry",     "Unknown"),
            "spot_price":   round(spot, 2) if not np.isnan(spot) else None,
        }

        for label, exp in zip(labels, expiries):
            frow = sf[sf["expiry_date"] == exp]
            orow = so[so["expiry_date"] == exp] if not so.empty else pd.DataFrame()

            rec[f"{label}_exp"] = exp

            if not frow.empty:
                settle = float(frow["settle"].iloc[0] or 0)
                rec[f"{label}_fut_oi"]    = int(frow["fut_oi"].iloc[0]  or 0)
                rec[f"{label}_chg_oi"]    = int(frow["chg_oi"].iloc[0]  or 0)
                rec[f"{label}_settle"]    = round(settle, 2)
                rec[f"{label}_basis_pct"] = round((settle - spot) / spot * 100, 2) if spot > 0 else None
            else:
                for k in ("fut_oi", "chg_oi", "settle", "basis_pct"):
                    rec[f"{label}_{k}"] = None

            if not orow.empty:
                stats = _opt_stats(orow, spot)
                for k, v in stats.items():
                    rec[f"{label}_{k}"] = v
            else:
                for k in ("call_oi", "put_oi", "pcr", "max_pain", "mp_dist_pct",
                          "top_ce_strike", "top_pe_strike"):
                    rec[f"{label}_{k}"] = None

        # Roll signal: compare near vs next OI change.
        # Threshold = 3% of near-month OI (min 2K) — relative so it works across
        # large-cap (multi-million OI) and mid-cap stocks alike.
        nc       = rec.get("near_chg_oi") or 0
        nk       = rec.get("next_chg_oi") or 0
        near_oi  = rec.get("near_fut_oi")  or 0
        thresh   = max(2_000, int(near_oi * 0.03))

        if nc < -thresh and nk > thresh * 0.5:
            rec["roll_signal"] = "Rolling Fwd"
        elif nc < -thresh and nk < -thresh * 0.5:
            rec["roll_signal"] = "Unwinding"
        elif nc > thresh:
            rec["roll_signal"] = "Building"
        else:
            rec["roll_signal"] = "Neutral"

        records.append(rec)

    result = pd.DataFrame(records)
    return result.sort_values("near_fut_oi", ascending=False, na_position="last").reset_index(drop=True)


def get_index_full_structure(
    trade_date: date,
    symbol: str = "NIFTY",
) -> pd.DataFrame:
    """
    Complete per-expiry breakdown for a single index symbol.

    One row per expiry:
        expiry_date, expiry_label, expiry_tier, days_to_expiry,
        fut_oi, fut_chg_oi, fut_settle, fut_basis_pct,
        call_oi, put_oi, pcr, max_pain, mp_dist_pct,
        top_ce_strike, top_pe_strike, total_oi, spot_price
    """
    trade_date = _as_date(trade_date)

    # Spot price (try daily_data first, then index name lookup)
    cash = query_dataframe(
        """
        SELECT close_price AS spot
        FROM daily_data
        WHERE symbol = ?
          AND trade_date = (SELECT MAX(trade_date) FROM daily_data WHERE trade_date <= ?)
        """,
        [symbol, trade_date],
    )
    spot = float(cash["spot"].iloc[0]) if not cash.empty else float("nan")

    # Futures
    fut = query_dataframe(
        """
        SELECT expiry_date,
               SUM(open_interest)  AS fut_oi,
               SUM(chg_in_oi)      AS fut_chg,
               MAX(settle_price)   AS settle
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND symbol      = ?
          AND instrument  = 'FUTIDX'
          AND expiry_date >= ?
        GROUP BY expiry_date
        """,
        [trade_date, symbol, trade_date],
    )
    if not fut.empty:
        fut["expiry_date"] = pd.to_datetime(fut["expiry_date"]).dt.date

    # Options
    opts = query_dataframe(
        """
        SELECT expiry_date, strike_price, option_type, open_interest
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND symbol      = ?
          AND instrument  = 'OPTIDX'
          AND expiry_date >= ?
          AND open_interest > 0
        """,
        [trade_date, symbol, trade_date],
    )
    if not opts.empty:
        opts["expiry_date"] = pd.to_datetime(opts["expiry_date"]).dt.date

    if fut.empty and opts.empty:
        return pd.DataFrame()

    # Collect all expiry dates and classify
    all_exps = set()
    if not fut.empty:
        all_exps.update(fut["expiry_date"].tolist())
    if not opts.empty:
        all_exps.update(opts["expiry_date"].tolist())
    all_exps = sorted(e for e in all_exps if e >= trade_date)

    monthly_exps = sorted(e for e in all_exps if _is_monthly_expiry(e))

    records = []
    for exp in all_exps:
        tier = _classify_expiry_tier(exp, trade_date, monthly_exps)
        rec: dict = {
            "expiry_date":    exp,
            "expiry_label":   exp.strftime("%d %b '%y"),
            "expiry_tier":    tier,
            "days_to_expiry": (exp - trade_date).days,
            "spot_price":     round(spot, 2) if not np.isnan(spot) else None,
        }

        # Futures
        if not fut.empty:
            frow = fut[fut["expiry_date"] == exp]
            if not frow.empty:
                settle = float(frow["settle"].iloc[0] or 0)
                rec["fut_oi"]        = int(frow["fut_oi"].iloc[0] or 0)
                rec["fut_chg_oi"]    = int(frow["fut_chg"].iloc[0] or 0)
                rec["fut_settle"]    = round(settle, 2)
                rec["fut_basis_pct"] = round((settle - spot) / spot * 100, 2) if spot > 0 else None
            else:
                rec.update({"fut_oi": 0, "fut_chg_oi": 0, "fut_settle": None, "fut_basis_pct": None})
        else:
            rec.update({"fut_oi": 0, "fut_chg_oi": 0, "fut_settle": None, "fut_basis_pct": None})

        # Options
        if not opts.empty:
            orow = opts[opts["expiry_date"] == exp]
            if not orow.empty:
                stats = _opt_stats(orow, spot)
                rec.update(stats)
                rec["total_oi"] = rec["fut_oi"] + stats["call_oi"] + stats["put_oi"]
            else:
                rec.update({"call_oi": 0, "put_oi": 0, "pcr": None, "max_pain": None,
                            "mp_dist_pct": None, "top_ce_strike": None, "top_pe_strike": None,
                            "total_oi": rec["fut_oi"]})
        else:
            rec.update({"call_oi": 0, "put_oi": 0, "pcr": None, "max_pain": None,
                        "mp_dist_pct": None, "top_ce_strike": None, "top_pe_strike": None,
                        "total_oi": rec["fut_oi"]})

        records.append(rec)

    return pd.DataFrame(records).sort_values("expiry_date").reset_index(drop=True)


def get_options_chain(
    trade_date: date,
    symbol: str,
    expiry_date: date,
    instrument: str = "OPTSTK",
) -> pd.DataFrame:
    """
    Full options chain for symbol + expiry, pivoted to one row per strike.

    Columns:
        strike_price,
        ce_oi, ce_chg_oi, ce_close,   (call side)
        pe_oi, pe_chg_oi, pe_close,   (put side)
        total_oi, pcr_at_strike,
        is_max_pain                    (True for the max-pain strike)
    """
    trade_date  = _as_date(trade_date)
    expiry_date = _as_date(expiry_date)

    raw = query_dataframe(
        """
        SELECT strike_price, option_type,
               open_interest, chg_in_oi, close_price, settle_price
        FROM fno_bhavcopy
        WHERE trade_date  = ?
          AND symbol      = ?
          AND expiry_date = ?
          AND instrument  = ?
          AND open_interest >= 0
        ORDER BY strike_price, option_type
        """,
        [trade_date, symbol, expiry_date, instrument],
    )
    if raw.empty:
        return raw

    ce = (raw[raw["option_type"] == "CE"]
          .rename(columns={"open_interest": "ce_oi", "chg_in_oi": "ce_chg_oi",
                           "close_price": "ce_close", "settle_price": "ce_settle"})
          [["strike_price", "ce_oi", "ce_chg_oi", "ce_close"]])

    pe = (raw[raw["option_type"] == "PE"]
          .rename(columns={"open_interest": "pe_oi", "chg_in_oi": "pe_chg_oi",
                           "close_price": "pe_close", "settle_price": "pe_settle"})
          [["strike_price", "pe_oi", "pe_chg_oi", "pe_close"]])

    chain = ce.merge(pe, on="strike_price", how="outer").sort_values("strike_price").fillna(0)
    chain["total_oi"]      = chain["ce_oi"] + chain["pe_oi"]
    chain["pcr_at_strike"] = chain.apply(
        lambda r: round(r["pe_oi"] / r["ce_oi"], 2) if r["ce_oi"] > 0 else None, axis=1
    )

    mp = _compute_max_pain(raw)
    chain["is_max_pain"] = chain["strike_price"].apply(
        lambda s: not np.isnan(mp) and abs(s - mp) < 0.5
    )

    return chain.reset_index(drop=True)


def get_index_options_chain(
    trade_date: date,
    symbol: str,
    expiry_date: date,
    n_strikes: int = 15,
) -> pd.DataFrame:
    """
    Options chain for an index (OPTIDX), centered ATM ± n_strikes.

    Spot proxied from near-month FUTIDX settle price (most reliable for indices).
    Max Pain computed from ALL available strikes before windowing.

    Returns one row per strike with columns:
        strike_price,
        ce_oi, ce_chg_oi, ce_vol, ce_close,
        pe_oi, pe_chg_oi, pe_vol, pe_close,
        total_oi, pcr_at_strike,
        is_atm, is_max_pain,
        spot_price, atm_strike, max_pain   (constant metadata on every row)
    """
    trade_date  = _as_date(trade_date)
    expiry_date = _as_date(expiry_date)

    # Spot from near-month futures settle (nearest expiry >= trade_date)
    spot_df = query_dataframe(
        """
        SELECT settle_price AS spot
        FROM fno_bhavcopy
        WHERE trade_date = ? AND symbol = ? AND instrument = 'FUTIDX'
          AND expiry_date >= ? AND settle_price > 0
        ORDER BY expiry_date ASC LIMIT 1
        """,
        [trade_date, symbol, trade_date],
    )
    spot = float(spot_df["spot"].iloc[0]) if not spot_df.empty else float("nan")

    # All options for max-pain (before strike windowing)
    all_raw = query_dataframe(
        """
        SELECT strike_price, option_type, open_interest, chg_in_oi,
               contracts, close_price
        FROM fno_bhavcopy
        WHERE trade_date = ? AND symbol = ? AND expiry_date = ?
          AND instrument = 'OPTIDX' AND open_interest >= 0
        ORDER BY strike_price, option_type
        """,
        [trade_date, symbol, expiry_date],
    )
    if all_raw.empty:
        return pd.DataFrame()

    # Max pain from all strikes (accurate — not limited to ATM window)
    mp = _compute_max_pain(all_raw)

    # ATM strike
    all_strikes = sorted(all_raw["strike_price"].unique())
    if np.isnan(spot):
        atm_strike = float(all_strikes[len(all_strikes) // 2])
        spot = atm_strike
    else:
        atm_strike = float(min(all_strikes, key=lambda k: abs(k - spot)))

    # Filter to ATM ± n_strikes window
    atm_idx    = all_strikes.index(atm_strike)
    lo, hi     = max(0, atm_idx - n_strikes), min(len(all_strikes), atm_idx + n_strikes + 1)
    window_set = set(all_strikes[lo:hi])
    raw        = all_raw[all_raw["strike_price"].isin(window_set)].copy()

    # Pivot to one row per strike
    ce = (raw[raw["option_type"] == "CE"]
          .rename(columns={"open_interest": "ce_oi", "chg_in_oi": "ce_chg_oi",
                           "contracts": "ce_vol", "close_price": "ce_close"})
          [["strike_price", "ce_oi", "ce_chg_oi", "ce_vol", "ce_close"]])

    pe = (raw[raw["option_type"] == "PE"]
          .rename(columns={"open_interest": "pe_oi", "chg_in_oi": "pe_chg_oi",
                           "contracts": "pe_vol", "close_price": "pe_close"})
          [["strike_price", "pe_oi", "pe_chg_oi", "pe_vol", "pe_close"]])

    chain = (ce.merge(pe, on="strike_price", how="outer")
               .sort_values("strike_price")
               .fillna(0)
               .reset_index(drop=True))

    chain["total_oi"]      = (chain["ce_oi"] + chain["pe_oi"]).astype(int)
    chain["pcr_at_strike"] = chain.apply(
        lambda r: round(r["pe_oi"] / r["ce_oi"], 2) if r["ce_oi"] > 0 else None, axis=1
    )
    chain["is_atm"]      = chain["strike_price"].apply(lambda s: abs(s - atm_strike) < 0.5)
    chain["is_max_pain"] = chain["strike_price"].apply(
        lambda s: not np.isnan(mp) and abs(s - mp) < 0.5
    )
    # Metadata embedded as constant columns for easy extraction in view layer
    chain["spot_price"] = round(spot, 2)
    chain["atm_strike"] = atm_strike
    chain["max_pain"]   = round(mp, 0) if not np.isnan(mp) else None

    return chain
