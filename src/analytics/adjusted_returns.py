"""
Corporate-action adjusted stock returns -- ONE definition for every panel.

THE PROBLEM
    `daily_data.prev_close` is the RAW previous close. NSE's file is not re-based
    for any corporate action (see ingestion/corp_actions_fetcher.py), so on an
    ex-date the plain  (close - prev_close) / prev_close  books the re-basing as a
    return. Measured on series EQ, 2022 to Oct-2026:
        392 split/bonus ex-days, median raw "return" -58%
        a +-40% guard drops 337 of them; 55 pass as real falls, 23 pass +-25%
        60 more drops beyond -40% are demergers or ETF splits with NO factor
    Small in aggregate (0.017% of stock-days) and total on the day it happens: a
    1:2 bonus is a -33% "crash" in that stock's row, its sector's average, and
    every window return that spans it.

THE RULE
    split / bonus   adjusted previous close = prev_close / share_factor, where
                    share_factor is NSE's own (corp_actions), compounded when two
                    actions share an ex-date (BAJFINANCE split + bonus = x10).
    demerger        no factor exists -- value is split, not the share count. The
                    ex-day OPENING gap is the demerged business, so the adjusted
                    previous close is that day's open and the return is
                    open-to-close (RELIANCE 20 Jul 2023: raw -7.8%, real +1.5%).
    dividends       NOT adjusted. These are price returns, the same convention as
                    the indices they are compared with.
    Never inferred from the tape: POLICYBZR fell 36% on 24 Sep 2026 with OI +54%
    (price x OI ~ 1, the exact shape of a re-basing) and it was a real crash.

RENAMED TICKERS
    NSE files a corporate action under the company's CURRENT symbol even when the
    ex-date predates the rename: Tata Motors' demerger is on TMPV, ex 14 Oct 2025,
    when the price row still says TATAMOTORS. 23 actions since 2018 were being
    missed for that reason alone (MOTHERSUMI bonus, MCDOWELL-N split, MINDAIND
    bonuses, KPIT scheme ...). config/symbol_changes.csv is NSE's official list
    (nsearchives.nseindia.com/content/equities/symbolchange.csv); the lookup uses
    only the pairs that have such an action, so the SQL stays small.

USE
    Every helper returns a SQL expression for a `daily_data` row under the given
    table alias. Interpolate at call time:
        f"SELECT {ret_sql('b')} AS r FROM daily_data b ..."
    On a day with no corporate action each expression equals the raw one exactly
    (verified over every EQ row, scripts/adjusted_returns_audit.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["ret_sql", "prev_close_sql", "up_sql", "factor_sql", "demerger_sql",
           "symbol_sql", "bridge_sql", "ca_symbol_map", "back_adjust"]

# used when the config file or the database cannot be read
_FALLBACK = {"ZOMATO": "ETERNAL", "TATAMOTORS": "TMPV"}
_DEMERGER_KINDS = "('DEMERGER', 'SCHEME')"


_MAP_CACHE: dict = {}


def ca_symbol_map() -> dict:
    """{old symbol: current symbol} for renames that hide a corporate action.

    Cached only once it has been built from the config file AND the database. A
    failure (database locked by the nightly job, file missing) returns the small
    fallback for that call and is retried on the next one -- caching the failure
    would pin a freshly started dashboard to two renames until it was restarted.
    """
    if _MAP_CACHE:
        return _MAP_CACHE
    out = dict(_FALLBACK)
    try:
        from src.core.config import PROJECT_ROOT
        from src.data.repository import query_dataframe
        sc = pd.read_csv(Path(PROJECT_ROOT) / "config" / "symbol_changes.csv")
        sc["changed_on"] = pd.to_datetime(sc["changed_on"], errors="coerce")
        ca = query_dataframe(f"""
            SELECT DISTINCT symbol, ex_date FROM corp_actions
            WHERE (share_factor > 0 AND share_factor <> 1) OR kind IN {_DEMERGER_KINDS}
        """)
        ca["ex_date"] = pd.to_datetime(ca["ex_date"])
        m = ca.merge(sc, left_on="symbol", right_on="new_symbol")
        m = m[m["ex_date"] < m["changed_on"]]
        out.update({str(o): str(n) for o, n in zip(m["old_symbol"], m["new_symbol"])
                    if "'" not in str(o) and "'" not in str(n)})
        _MAP_CACHE.update(out)
    except Exception:                        # never let a config problem break a page
        pass
    return out


def _sym(a: str) -> str:
    m = ca_symbol_map()
    if not m:
        return f"{a}.symbol"
    w = " ".join(f"WHEN '{o}' THEN '{n}'" for o, n in sorted(m.items()))
    return f"CASE {a}.symbol {w} ELSE {a}.symbol END"


def symbol_sql(a: str = "daily_data") -> str:
    """The row's CURRENT ticker -- the one corp_actions is keyed on."""
    return _sym(a)


def factor_sql(a: str = "daily_data") -> str:
    """Share factor in force on the row's date (1 on an ordinary day)."""
    return ("COALESCE((SELECT EXP(SUM(LN(ca.share_factor))) FROM corp_actions ca "
            f"WHERE ca.symbol = {_sym(a)} AND ca.ex_date = {a}.trade_date "
            "AND ca.share_factor > 0 AND ca.share_factor <> 1), 1)")


def demerger_sql(a: str = "daily_data") -> str:
    """TRUE on a demerger / scheme ex-date for the row's company."""
    return ("EXISTS (SELECT 1 FROM corp_actions cd "
            f"WHERE cd.symbol = {_sym(a)} AND cd.ex_date = {a}.trade_date "
            f"AND cd.kind IN {_DEMERGER_KINDS})")


def prev_close_sql(a: str = "daily_data") -> str:
    """The previous close, on today's share basis."""
    return (f"CASE WHEN {a}.open_price > 0 AND {demerger_sql(a)} THEN {a}.open_price "
            f"ELSE {a}.prev_close / {factor_sql(a)} END")


def ret_sql(a: str = "daily_data") -> str:
    """Adjusted one-day % return. NULL when prev_close is 0 or missing."""
    pc = prev_close_sql(a)
    return f"(({a}.close_price - ({pc})) / NULLIF(({pc}), 0) * 100)"


def up_sql(a: str = "daily_data") -> str:
    """TRUE when the stock closed above its adjusted previous close."""
    return f"({a}.close_price > ({prev_close_sql(a)}))"


def bridge_sql(sym: str, lag_date: str, row_date: str, lag_close: str,
               adj_pc: str) -> str:
    """Factor that carries a price across the gap between two CONSECUTIVE rows.

    For returns measured between two price LEVELS (a window's start and end).
    Multiply the later price by the product of this over every row after the
    start, and the ratio to the start price is free of corporate actions.

        ordinary day                      1
        split / bonus / demerger day      previous row's close / adjusted prev close
        action during a SUSPENSION        the same -- and this is the case a
                                          per-day adjustment cannot see

    India Glycols demerged on 02 Sep 2026 and did not trade for two weeks. There
    is no price row on the ex-date, and when it resumed NSE's prev_close was
    already on the new basis: daily returns are right, but 1,111 -> 258 across the
    gap reads -72% to anything comparing levels. The previous ROW's close over
    the resumption day's adjusted prev_close bridges it.

    Applied ONLY when corp_actions has a split, bonus or demerger dated inside the
    gap. A stock that simply spent sessions in another series has a gap too, and
    its move across that gap is real and must be kept.

    Arguments are SQL expressions for one row: the current ticker, the previous
    row's date and close, this row's date, and this row's prev_close_sql().
    """
    return (f"CASE WHEN {lag_close} > 0 AND {adj_pc} > 0 AND EXISTS ("
            f"SELECT 1 FROM corp_actions cb WHERE cb.symbol = {sym} "
            f"AND cb.ex_date > {lag_date} AND cb.ex_date <= {row_date} "
            f"AND ((cb.share_factor > 0 AND cb.share_factor <> 1) "
            f"OR cb.kind IN {_DEMERGER_KINDS})) "
            f"THEN {lag_close} / {adj_pc} ELSE 1 END")


def back_adjust(df: pd.DataFrame, price_cols=("o", "h", "l", "c", "pc"),
                symbol_col: str = "symbol", date_col: str = "trade_date") -> pd.DataFrame:
    """Put a multi-day OHLC panel on ONE share basis: the latest date's.

    For level-based analysis (bases, breakouts, gaps) a daily adjusted return is
    not enough -- the highs of last month must be comparable with today's. Every
    price before an ex-date is divided by the factors still to come. `pc` on the
    ex-date itself is the raw previous close, so it takes that day's factor too.
    Demergers are handled the same way using the ex-day open gap.
    """
    if df.empty:
        return df
    from src.data.repository import query_dataframe
    syms = sorted(df[symbol_col].unique())
    amap = ca_symbol_map()
    lookup = sorted(set(syms) | {amap[s] for s in syms if s in amap})
    ph = ", ".join("?" * len(lookup))
    d0, d1 = df[date_col].min(), df[date_col].max()
    ca = query_dataframe(f"""
        SELECT symbol, ex_date, kind, share_factor FROM corp_actions
        WHERE symbol IN ({ph}) AND ex_date > ? AND ex_date <= ?
          AND ((share_factor > 0 AND share_factor <> 1) OR kind IN {_DEMERGER_KINDS})
    """, [*lookup, d0, d1])
    if ca.empty:
        return df
    out = df.copy()
    out["_d"] = pd.to_datetime(out[date_col])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    cols = [c for c in price_cols if c in out.columns]
    for sym, g in out.groupby(symbol_col):
        ev = ca[ca["symbol"].isin({sym, amap.get(sym, sym)})]
        for ex, e in ev.groupby("ex_date"):
            f = float(e.loc[e["share_factor"] > 0, "share_factor"].prod()) or 1.0
            if abs(f - 1) < 1e-9:
                # demerger: the ex-day open over the raw previous close
                row = g[g["_d"] == ex]
                pc_col = "pc" if "pc" in out.columns else None
                o_col = "o" if "o" in out.columns else None
                if row.empty or not pc_col or not o_col:
                    continue
                o, pc = float(row[o_col].iloc[0]), float(row[pc_col].iloc[0])
                if not (o > 0 and pc > 0):
                    continue
                f = pc / o
            before = out.index[(out[symbol_col] == sym) & (out["_d"] < ex)]
            out.loc[before, cols] = out.loc[before, cols] / f
            if "pc" in cols:
                on = out.index[(out[symbol_col] == sym) & (out["_d"] == ex)]
                out.loc[on, "pc"] = out.loc[on, "pc"] / f
    return out.drop(columns="_d")
