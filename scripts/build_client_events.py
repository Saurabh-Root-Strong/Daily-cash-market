"""Build `deal_client_events` — one row per (client, symbol, day) with forward
excess returns at 1wk / 2wk / 1mo / 2mo / 3mo.

Same matched-control scoring as scripts/build_deal_outcomes.py (read its header
first). Two additions:

  1. CLIENT-LEVEL netting. A client that buys and sells the same name the same day
     has done a round trip, and that is not a buy. Netting is per client per
     symbol per day, so "BUY event" means that client ended the day long.

  2. NAME NORMALISATION. NSE types the client name free-hand, so the same entity
     appears as "QE SECURITIES" and "QE SECURITIES LLP", "MANSI SHARE AND STOCK
     ADVISORS PVT LTD" and "MANSI SHARES & STOCK ADVISORS PVT LTD". Left raw, one
     entity is split across several rows and every per-entity sample is understated.
     `client_key` collapses legal suffixes and punctuation; `client_name` keeps the
     most common raw spelling for display.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.repository import get_repository

HORIZONS = {"m5": 5, "m10": 10, "m21": 21, "m42": 42, "m63": 63}

_SUFFIX = re.compile(
    r"\b(PRIVATE LIMITED|PVT LTD|PVT\. LTD|PVT LIMITED|PRIVATE LTD|LIMITED|LTD|"
    r"LLP|LLC|INC|CO|COMPANY|AND|THE)\b"
)
_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalise(name: str) -> str:
    """Collapse the spellings NSE uses for one entity into a single key.

    Spaces are removed at the end, not just squeezed: 34 entity pairs differed ONLY
    by internal spacing ("ADITYA BIRLA SUN LIFE MUTUAL FUND" with 34 events vs
    "ADITYA BIRLA SUNLIFE MUTUAL FUND" with 3, "BIRLA SUN LIFE" vs "BIRLA SUNLIFE"),
    which split one firm's record in two and understated both halves. Dropping spaces
    cannot merge two genuinely different names that are otherwise identical.
    """
    u = (name or "").upper().replace("&", " AND ")
    u = _PUNCT.sub(" ", u)
    u = _SUFFIX.sub(" ", u)
    return _SPACE.sub(" ", u).strip().replace(" ", "")


UNIVERSE_SQL = """
CREATE OR REPLACE TEMP TABLE px AS
SELECT trade_date, symbol, open_price, close_price, prev_close, ttl_trd_qnty, turnover_lacs,
       deliv_qty,
-- prev_close is NOT always trustworthy: on 2026-04-20 NIRLON carries prev_close 1.40
-- against an open of 540.00 and a close of 539.79 (8 symbols corrupted that session,
-- 439 such rows across 2018-2026). Compounding that raw ratio injects a fake +38,000%
-- into the stock's return series. Where prev_close is contradicted by the day's own
-- open, use close/open; otherwise clamp. A real circuit-limit move stays inside 0.5-2.0.
       CASE WHEN close_price / prev_close BETWEEN 0.5 AND 2.0 THEN close_price / prev_close
            WHEN open_price / prev_close NOT BETWEEN 0.5 AND 2.0
                 AND close_price / open_price BETWEEN 0.5 AND 2.0
              THEN close_price / open_price
            ELSE greatest(least(close_price / prev_close, 2.0), 0.5) END AS ratio_clean
FROM daily_data
WHERE series = 'EQ' AND close_price > 0 AND open_price > 0 AND prev_close > 0
  AND trade_date >= DATE '2017-11-01';

CREATE OR REPLACE TEMP TABLE cum AS
SELECT *,
  exp(sum(ln(ratio_clean)) OVER (PARTITION BY symbol ORDER BY trade_date)) AS cf,
  avg(turnover_lacs) OVER (PARTITION BY symbol ORDER BY trade_date
                           ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS adv20,
  ratio_clean - 1 AS r0
FROM px;

CREATE OR REPLACE TEMP TABLE fwd AS
SELECT trade_date, symbol, turnover_lacs, adv20, r0, deliv_qty,
       lead(open_price, 1) OVER w AS o1, lead(close_price, 1) OVER w AS c1,
       lead(cf, 1)  OVER w AS cf1,
       lead(cf, 5)  OVER w AS cf5,
       lead(cf, 10) OVER w AS cf10,
       lead(cf, 21) OVER w AS cf21,
       lead(cf, 42) OVER w AS cf42,
       lead(cf, 63) OVER w AS cf63
FROM cum WINDOW w AS (PARTITION BY symbol ORDER BY trade_date);

CREATE OR REPLACE TEMP TABLE ret AS
SELECT trade_date, symbol, turnover_lacs, adv20, r0, deliv_qty,
       turnover_lacs / nullif(adv20, 0) AS spike,
       (c1 / o1) * (cf5  / cf1) - 1 AS r5,
       (c1 / o1) * (cf10 / cf1) - 1 AS r10,
       (c1 / o1) * (cf21 / cf1) - 1 AS r21,
       (c1 / o1) * (cf42 / cf1) - 1 AS r42,
       (c1 / o1) * (cf63 / cf1) - 1 AS r63
FROM fwd
-- Keep events that cannot be SCORED yet. The newest session has no next-day open to
-- enter at, so o1/c1 are NULL and every forward return is NULL -- but the deal still
-- happened, and "who bought" is exactly what a recent window is for. Excluding the row
-- entirely made the To-date picker grey out the latest session and left the page blind
-- to the most recent day, while the sidebar showed that day as current.
WHERE trade_date >= DATE '2018-01-01' AND turnover_lacs > 0 AND adv20 > 0;

CREATE OR REPLACE TEMP TABLE u AS
SELECT *,
  ntile(5)  OVER (PARTITION BY trade_date ORDER BY turnover_lacs) AS liq_q,
  ntile(10) OVER (PARTITION BY trade_date ORDER BY spike)         AS spike_d,
  ntile(10) OVER (PARTITION BY trade_date ORDER BY r0)            AS r0_d
FROM ret;

CREATE OR REPLACE TEMP TABLE evday AS
SELECT DISTINCT trade_date, symbol FROM bulk_block_deals
WHERE trade_date >= DATE '2018-01-01';

CREATE OR REPLACE TEMP TABLE cellmean AS
SELECT u.trade_date, u.liq_q, u.spike_d, u.r0_d,
       avg(u.r5) AS c5, avg(u.r10) AS c10, avg(u.r21) AS c21,
       avg(u.r42) AS c42, avg(u.r63) AS c63, count(*) AS n_ctrl
FROM u LEFT JOIN evday e ON e.symbol = u.symbol AND e.trade_date = u.trade_date
WHERE e.symbol IS NULL
GROUP BY 1, 2, 3, 4;
"""

CLIENT_SQL = """
-- Client-level netting: BUY minus SELL by that client in that name that day.
-- DISTINCT first: a block-window trade that also crosses the 0.5% bulk threshold is
-- filed identically in BOTH feeds, which doubled qty_net, value_cr and net_pct_adv on
-- 895 buy events (5.7%). Sign was never affected, but size was.
CREATE OR REPLACE TEMP TABLE dedup AS
SELECT DISTINCT trade_date, symbol, client_name, k.client_key, buy_sell, quantity, trade_price,
-- If the trade appears in the BLOCK feed at all, it went through the block window.
-- max(deal_type) got this backwards: 'BULK' sorts after 'BLOCK', so all 893 twin
-- events were labelled BULK and then judged by the delivery test -- which block
-- trades cannot pass, because they settle outside the regular market and never
-- enter deliv_qty. That made institutions look like they were faking purchases.
       CASE WHEN bool_or(deal_type = 'BLOCK') OVER (
                PARTITION BY trade_date, symbol, client_name, buy_sell,
                             quantity, trade_price)
            THEN 'BLOCK' ELSE 'BULK' END AS deal_type
FROM bulk_block_deals JOIN _keymap k USING (client_name)
WHERE trade_date >= DATE '2018-01-01';

-- Net by client_key, not by the raw spelling.
--
-- NSE types the name by hand, so one firm arrives as "SMALL CAP WORLD FUND INC." and
-- "SMALL CAP WORLD FUND, INC" in the same file. Grouping by client_name left 29
-- (day, stock, entity) groups split across two rows: the leaderboard then counted one
-- decision as two buys, and -- the part that actually distorts -- a buy under one
-- spelling was NOT netted against a sell under the other, so a flat round trip could
-- be published as a purchase. Keying the grouping fixes both at the source.
CREATE OR REPLACE TEMP TABLE cev AS
SELECT trade_date, symbol, client_key,
       mode(client_name) AS client_name,
       sum(CASE WHEN buy_sell = 'BUY' THEN quantity ELSE -quantity END) AS qty_net,
       sum(quantity) AS qty_gross,
       sum(quantity * trade_price) / nullif(sum(quantity), 0) AS vwap,
       sum(quantity * trade_price) / 1e7 AS value_cr,
       -- same rule at client-day level: any block leg makes the delivery test
       -- inapplicable to this row, so it must not be scored as a bulk purchase.
       CASE WHEN bool_or(deal_type = 'BLOCK') THEN 'BLOCK' ELSE 'BULK' END AS deal_type,
       count(*) AS n_lines
FROM dedup
GROUP BY 1, 2, 3;

CREATE OR REPLACE TABLE deal_client_events AS
SELECT
    c.trade_date, c.symbol, c.client_name, c.client_key,
    CASE WHEN c.qty_net > 0 THEN 'BUY'
         WHEN c.qty_net < 0 THEN 'SELL' ELSE 'FLAT' END AS side,
    c.qty_net, c.qty_gross, c.value_cr, c.vwap, c.deal_type, c.n_lines,
    u.turnover_lacs, u.spike, u.r0, u.liq_q, u.deliv_qty,
    100.0 * c.qty_net / nullif(u.deliv_qty, 0) AS pct_of_delivery,
    (c.qty_net * c.vwap / 100000.0) / nullif(u.adv20, 0) AS net_pct_adv,
    u.r5 - m.c5   AS m5,
    u.r10 - m.c10 AS m10,
    u.r21 - m.c21 AS m21,
    u.r42 - m.c42 AS m42,
    u.r63 - m.c63 AS m63,
    m.n_ctrl,
    now() AS built_at
FROM cev c
JOIN u          ON u.symbol = c.symbol AND u.trade_date = c.trade_date
JOIN cellmean m ON m.trade_date = u.trade_date AND m.liq_q = u.liq_q
                AND m.spike_d = u.spike_d AND m.r0_d = u.r0_d
WHERE m.n_ctrl >= 5;
"""


def _add_walk_forward(con) -> None:
    """Attach each BUY event's PRIOR track record for that entity.

    This is the column that decides whether "follow the smart buyer" is a strategy
    or a backtest illusion. The prior record uses only that entity's earlier buys
    whose own 1-month window had already CLOSED (trade_date + 45 calendar days) as
    of the current event - an entity's still-open positions are not knowable at the
    time you would have had to act, and including them is the standard way this
    kind of study fools itself.
    """
    d = con.sql("""
        SELECT trade_date, symbol, client_key, m21
        FROM deal_client_events WHERE side = 'BUY'
    """).df()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d["resolved_on"] = d["trade_date"] + pd.Timedelta(days=45)

    prior_mean = np.full(len(d), np.nan)
    prior_n = np.zeros(len(d), dtype=int)
    for _, idx in d.groupby("client_key").indices.items():
        sub = d.iloc[idx]
        td = sub["trade_date"].to_numpy()
        order = np.argsort(sub["resolved_on"].to_numpy())
        rs_s = sub["resolved_on"].to_numpy()[order]
        v_s = sub["m21"].to_numpy()[order]
        csum = np.nancumsum(np.nan_to_num(v_s))
        cn = np.cumsum(~np.isnan(v_s))
        pos = np.searchsorted(rs_s, td, side="left")
        prev = np.clip(pos - 1, 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = csum[prev] / np.maximum(cn[prev], 1)
        prior_mean[idx] = np.where(pos > 0, pm, np.nan)
        prior_n[idx] = np.where(pos > 0, cn[prev], 0)

    d["prior_mean"] = prior_mean
    d["prior_n"] = prior_n
    con.register("_wf", d[["trade_date", "symbol", "client_key", "prior_mean", "prior_n"]])
    con.execute("ALTER TABLE deal_client_events ADD COLUMN IF NOT EXISTS prior_mean DOUBLE")
    con.execute("ALTER TABLE deal_client_events ADD COLUMN IF NOT EXISTS prior_n INTEGER")
    con.execute("""
        UPDATE deal_client_events e
        SET prior_mean = w.prior_mean, prior_n = w.prior_n
        FROM _wf w
        WHERE w.trade_date = e.trade_date AND w.symbol = e.symbol
          AND w.client_key = e.client_key
    """)


def _add_settlement_flags(con) -> None:
    """Attach the two checks that separate a real purchase from a printed one.

    pct_of_delivery -- the net purchase as a share of the WHOLE market's delivered
    shares in that stock that day. A genuine accumulation settles, so it has to fit
    inside the day's delivery. Above 100% it cannot have been delivered at all.
    Only meaningful for BULK: a BLOCK trade is negotiated outside the regular market
    and never enters deliv_qty, so block rows are left out of the judgement.

    exit_days -- calendar days until the SAME entity next appears as a net seller in
    the SAME stock. Beware the asymmetry: NSE only publishes a leg once it crosses
    0.5% of listed shares, so an exit spread over several days, or simply smaller,
    is INVISIBLE here. "No disclosed exit" therefore means "never sold 0.5% in one
    day again" -- it is not proof the position was held.
    """
    d = con.sql("""
        SELECT trade_date, symbol, client_key, side, qty_net
        FROM deal_client_events
    """).df()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    buys = d[d.side == "BUY"][["trade_date", "symbol", "client_key"]].copy()
    sells = d[d.side == "SELL"][["trade_date", "symbol", "client_key"]].rename(
        columns={"trade_date": "sell_date"})
    j = buys.merge(sells, on=["symbol", "client_key"], how="left")
    gap = (j["sell_date"] - j["trade_date"]).dt.days
    j["gap"] = gap.where(gap > 0)
    first = (j.groupby(["trade_date", "symbol", "client_key"])["gap"]
               .min().rename("exit_days").reset_index())
    con.register("_exits", first)
    con.execute("ALTER TABLE deal_client_events ADD COLUMN IF NOT EXISTS exit_days DOUBLE")
    con.execute("""
        UPDATE deal_client_events e SET exit_days = x.exit_days
        FROM _exits x
        WHERE x.trade_date = e.trade_date AND x.symbol = e.symbol
          AND x.client_key = e.client_key
    """)



SINCE_SQL = """
-- MOVE SINCE THE BUY, measured to the latest close we hold. Unlike m5..m63 this is
-- never blank, so a 10-session window still says how the purchases are doing.
--
-- Split-adjusted through cf (the compounded ratio_clean), never raw last_close/vwap:
-- a 1:10 split between the deal and today would print -90% on a stock that went up.
-- The deal-day close is carried forward by cf_last/cf_event, then compared with the
-- client's own average deal price, which is what they actually paid.
--
-- NOT the EQ-only `cum` chain. A stock pushed into the BE (trade-to-trade) or BZ
-- series drops out of EQ for weeks or months, and an EQ-only chain treats that whole
-- stretch as a 0% move. APOLLO spent 169 sessions in BE across four spells: the EQ
-- chain printed +7,257% since a 2019 buy against a true ~+4,700%. The price never
-- stopped existing, it just moved series, so follow it there. EQ wins a date that
-- has both (the transition day).
CREATE OR REPLACE TEMP TABLE pxall AS
SELECT trade_date, symbol, close_price,
       CASE WHEN close_price / prev_close BETWEEN 0.5 AND 2.0 THEN close_price / prev_close
            WHEN open_price / prev_close NOT BETWEEN 0.5 AND 2.0
                 AND close_price / open_price BETWEEN 0.5 AND 2.0
              THEN close_price / open_price
            ELSE greatest(least(close_price / prev_close, 2.0), 0.5) END AS ratio_clean
FROM daily_data
WHERE series IN ('EQ', 'BE', 'BZ') AND close_price > 0 AND open_price > 0 AND prev_close > 0
  AND trade_date >= DATE '2017-11-01'
QUALIFY row_number() OVER (PARTITION BY symbol, trade_date
                           ORDER BY CASE series WHEN 'EQ' THEN 0 WHEN 'BE' THEN 1 ELSE 2 END) = 1;

CREATE OR REPLACE TEMP TABLE evpx AS
SELECT symbol, trade_date, close_price,
       exp(sum(ln(ratio_clean)) OVER (PARTITION BY symbol ORDER BY trade_date)) AS cf,
       row_number() OVER (PARTITION BY symbol ORDER BY trade_date) AS rn
FROM pxall;

CREATE OR REPLACE TEMP TABLE lastpx AS
SELECT symbol,
       max(trade_date)                 AS px_date,
       arg_max(cf, trade_date)          AS cf_last,
       arg_max(close_price, trade_date) AS close_last,
       max(rn)                          AS n_rows
FROM evpx GROUP BY symbol;

CREATE OR REPLACE TEMP TABLE nifty AS
SELECT trade_date, close_val FROM index_data
WHERE index_name = 'Nifty 50' AND close_val > 0;

CREATE OR REPLACE TEMP TABLE since AS
SELECT e.trade_date, e.symbol, e.client_key,
       -- A deal price far from that day's close is a data error or a different
       -- series' print, not a real 50%+ discount; refuse to score it.
       CASE WHEN e.vwap / p.close_price BETWEEN 0.5 AND 2.0
            THEN p.close_price * (l.cf_last / p.cf) / e.vwap - 1 END AS since_pct,
       l.cf_last / p.cf - 1                AS since_close_pct,
       l.close_last                        AS last_close,
       l.px_date                           AS px_date,
       l.n_rows - p.rn                     AS sessions_since
FROM deal_client_events e
JOIN evpx p   ON p.symbol = e.symbol AND p.trade_date = e.trade_date
JOIN lastpx l ON l.symbol = e.symbol;

-- Nifty 50 over the SAME span: deal-day close to the stock's own last price date.
-- Ending at px_date rather than today keeps a suspended stock from being compared
-- with months of index movement it never experienced. ASOF tolerates index_data's
-- known session holes by taking the nearest earlier close.
CREATE OR REPLACE TEMP TABLE since_nifty AS
SELECT s.*, n1.close_val AS n_start, n2.close_val AS n_end
FROM since s
ASOF LEFT JOIN nifty n1 ON s.trade_date >= n1.trade_date
ASOF LEFT JOIN nifty n2 ON s.px_date   >= n2.trade_date;
"""


def _add_since_move(con) -> None:
    """Attach how far each bought stock has moved since the client's purchase."""
    for stmt in _statements(SINCE_SQL):
        con.execute(stmt)
    for col, typ in (("since_pct", "DOUBLE"), ("since_close_pct", "DOUBLE"),
                     ("nifty_since_pct", "DOUBLE"), ("last_close", "DOUBLE"),
                     ("px_date", "DATE"), ("sessions_since", "INTEGER")):
        con.execute(f"ALTER TABLE deal_client_events ADD COLUMN IF NOT EXISTS {col} {typ}")
    con.execute("""
        UPDATE deal_client_events e
        SET since_pct = s.since_pct, since_close_pct = s.since_close_pct,
            nifty_since_pct = s.n_end / s.n_start - 1,
            last_close = s.last_close, px_date = s.px_date,
            sessions_since = s.sessions_since
        FROM since_nifty s
        WHERE s.trade_date = e.trade_date AND s.symbol = e.symbol
          AND s.client_key = e.client_key
    """)


def _statements(script: str) -> list[str]:
    """Split a multi-statement SQL script on semicolons, ignoring `--` comments.

    A plain script.split(";") breaks the moment a comment contains a semicolon --
    which silently truncated the px SELECT list mid-expression and produced a
    baffling "FROM clause is missing". Comment text is stripped for splitting
    purposes; it stays in the source for whoever reads it next.
    """
    cleaned = chr(10).join(line.split("--")[0] for line in script.splitlines())
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def main() -> None:
    # Through ConnectionManager, not duckdb.connect(): DuckDB gives ONE process the
    # write lock, and this build holds it for minutes. The manager's backoff turns a
    # collision with the running dashboard into a wait instead of a hard IO error.
    with get_repository()._cm.connect() as con:
        con.execute("SET threads=4")
        for stmt in _statements(UNIVERSE_SQL):
            con.execute(stmt)
        # The entity key must exist BEFORE netting, so the grouping can use it.
        names = con.sql(
            "SELECT DISTINCT client_name FROM bulk_block_deals "
            "WHERE trade_date >= DATE '2018-01-01'").df()
        names["client_key"] = names["client_name"].map(normalise)
        con.register("_keymap", names)

        for stmt in _statements(CLIENT_SQL):
            con.execute(stmt)

        _add_walk_forward(con)
        _add_settlement_flags(con)
        _add_since_move(con)

        n = con.sql("SELECT count(*) FROM deal_client_events").fetchone()[0]
        k = con.sql("SELECT count(DISTINCT client_key) FROM deal_client_events").fetchone()[0]
        raw = con.sql("SELECT count(DISTINCT client_name) FROM deal_client_events").fetchone()[0]
        print(f"deal_client_events: {n:,} client-events, {raw:,} raw names -> {k:,} entities")
        print(con.sql("""
            SELECT side, count(*) AS events,
                   round(100 * avg(m5), 2)  AS x_1wk,
                   round(100 * avg(m10), 2) AS x_2wk,
                   round(100 * avg(m21), 2) AS x_1mo,
                   round(100 * avg(m42), 2) AS x_2mo,
                   round(100 * avg(m63), 2) AS x_3mo
            FROM deal_client_events GROUP BY 1 ORDER BY 1
        """).df().to_string(index=False))


if __name__ == "__main__":
    main()
