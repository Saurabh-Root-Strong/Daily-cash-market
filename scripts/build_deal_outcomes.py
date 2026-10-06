"""Build `deal_event_outcomes` — what actually happened after every bulk/block deal.

    python scripts/build_deal_outcomes.py

WHY A MATCHED CONTROL, NOT AN INDEX BENCHMARK
    A bulk-deal day is, by construction, a high-volume day with a big move in a
    mostly small, illiquid name. All three of those revert on their own. Scoring
    the deal against Nifty would credit the deal with small-cap beta and with
    ordinary post-spike reversal.

    So each event is scored against the mean forward return of NON-event stock-days
    in the SAME session, the same liquidity quintile, the same volume-spike decile
    and the same same-day-move decile (>= 5 controls in the cell, else the event is
    dropped). A placebo drawn from that control pool prints -0.00% at 21 sessions
    (t = -0.16), so the machinery itself is unbiased.

CORPORATE ACTIONS
    Forward returns compound close / ADJUSTED prev_close, never close_t+n / close_t.
    NSE's prev_close is NOT ex-adjusted (this docstring said the opposite until
    Oct-2026). The 0.5-2.0 guard below caught the large re-basings by accident --
    a 5:1 split is a 0.2 ratio, so it fell through to close/open -- and missed
    every small one: a 1:2 bonus is 0.67, inside the band, and went into the
    return series as a real -33% day. Measured 2018-2026: 571 split / bonus /
    demerger days on EQ, 339 caught by the guard, 232 leaked. prev_close is now
    put on the day's share basis first (src/analytics/adjusted_returns.py).
    THE STORED TABLE MUST BE REBUILT for this to take effect:
        python scripts/build_deal_outcomes.py

ENTRY
    Deals are disclosed after the close, so entry is the NEXT session's OPEN.
    Scoring from the event-day close would be a lookahead.

KNOWN LIMITS (do not quietly forget these when reading the numbers)
    - Series EQ only. SME-platform deals are excluded; they are a large share of
      raw bulk-deal rows and are near-untradeable.
    - Delisted names drop out of the join (~28% of raw events also covers non-EQ
      series and windows still open). Survivors-only biases returns UP, so the
      measured negative is if anything understated.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.data.repository import get_repository
from src.analytics.deal_signals import classify_client
from src.analytics.adjusted_returns import prev_close_sql

UNIVERSE_SQL = """
CREATE OR REPLACE TEMP TABLE px AS
SELECT trade_date, symbol, open_price, close_price, prev_close, ttl_trd_qnty, turnover_lacs,
-- prev_close is NOT always trustworthy: on 2026-04-20 NIRLON carries prev_close 1.40
-- against an open of 540.00 and a close of 539.79 (8 symbols corrupted that session,
-- 439 such rows across 2018-2026). Compounding that raw ratio injects a fake +38,000%
-- into the stock's return series. Where prev_close is contradicted by the day's own
-- open, use close/open; otherwise clamp. A real circuit-limit move stays inside 0.5-2.0.
-- __APC__ is prev_close on the day's own share basis (split / bonus / demerger
-- adjusted). The guard stays for the corrupted-prev_close rows it was written for.
       CASE WHEN close_price / (__APC__) BETWEEN 0.5 AND 2.0 THEN close_price / (__APC__)
            WHEN open_price / (__APC__) NOT BETWEEN 0.5 AND 2.0
                 AND close_price / open_price BETWEEN 0.5 AND 2.0
              THEN close_price / open_price
            ELSE greatest(least(close_price / (__APC__), 2.0), 0.5) END AS ratio_clean
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
SELECT trade_date, symbol, turnover_lacs, adv20, r0,
       lead(open_price, 1) OVER w AS o1, lead(close_price, 1) OVER w AS c1,
       lead(cf, 1) OVER w AS cf1, lead(cf, 21) OVER w AS cf21, lead(cf, 42) OVER w AS cf42
FROM cum WINDOW w AS (PARTITION BY symbol ORDER BY trade_date);

CREATE OR REPLACE TEMP TABLE ret AS
SELECT trade_date, symbol, turnover_lacs, adv20, r0,
       turnover_lacs / nullif(adv20, 0) AS spike,
       (c1 / o1) * (cf21 / cf1) - 1 AS r21,
       (c1 / o1) * (cf42 / cf1) - 1 AS r42
FROM fwd
-- Demanding cf21 here once dropped every event of the last ~21 sessions, so the table
-- stopped a month short of the raw feed and "Recent deals" had no recent deals.
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

CREATE OR REPLACE TEMP TABLE ev AS
SELECT trade_date, symbol,
       max(CASE WHEN deal_type = 'BULK'  THEN 1 ELSE 0 END) AS is_bulk,
       max(CASE WHEN deal_type = 'BLOCK' THEN 1 ELSE 0 END) AS is_block,
       sum(CASE WHEN deal_type = 'BULK' AND buy_sell = 'BUY'  THEN quantity
                WHEN deal_type = 'BULK' AND buy_sell = 'SELL' THEN -quantity
                ELSE 0 END) AS qty_net,
       sum(quantity) AS qty_gross,
       sum(quantity * trade_price) / 1e7 AS value_cr,
       count(*) AS n_deals,
       count(DISTINCT client_name) AS n_clients,
       string_agg(DISTINCT buy_sell || '::' || client_name, '||') AS client_rows
FROM bulk_block_deals
WHERE trade_date >= DATE '2018-01-01'
GROUP BY 1, 2;

-- The control cohort: non-event days only, so an event never benchmarks itself.
CREATE OR REPLACE TEMP TABLE cellmean AS
SELECT u.trade_date, u.liq_q, u.spike_d, u.r0_d,
       avg(u.r21) AS c21, avg(u.r42) AS c42, count(*) AS n_ctrl
FROM u LEFT JOIN ev e ON e.symbol = u.symbol AND e.trade_date = u.trade_date
WHERE e.symbol IS NULL
GROUP BY 1, 2, 3, 4;
""".replace("__APC__", prev_close_sql("daily_data"))

OUTCOME_SQL = """
CREATE OR REPLACE TABLE deal_event_outcomes AS
SELECT
    u.trade_date, u.symbol,
    e.is_bulk, e.is_block,
    CASE WHEN e.is_bulk = 0 THEN 'NONE'
         WHEN e.qty_net > 0 THEN 'BUY'
         WHEN e.qty_net < 0 THEN 'SELL'
         ELSE 'FLAT' END              AS side,
    e.qty_net, e.qty_gross, e.value_cr, e.n_deals, e.n_clients, e.client_rows,
    u.turnover_lacs, u.spike, u.r0, u.liq_q, u.spike_d, u.r0_d,
    u.r21, u.r42, m.c21, m.c42, m.n_ctrl,
    u.r21 - m.c21 AS m21,
    u.r42 - m.c42 AS m42,
    now() AS built_at
FROM u
JOIN ev e       ON e.symbol = u.symbol AND e.trade_date = u.trade_date
JOIN cellmean m ON m.trade_date = u.trade_date AND m.liq_q = u.liq_q
                AND m.spike_d = u.spike_d AND m.r0_d = u.r0_d
WHERE m.n_ctrl >= 5;
"""


def _statements(script: str) -> list[str]:
    """Split a multi-statement SQL script on semicolons, ignoring `--` comments.

    A plain script.split(";") breaks the moment a comment contains a semicolon --
    which silently truncated the px table's SELECT list mid-expression and produced
    a baffling "FROM clause is missing". Comment text is stripped for splitting
    purposes; it stays in the source for whoever reads it next.
    """
    cleaned = chr(10).join(line.split("--")[0] for line in script.splitlines())
    return [s.strip() for s in cleaned.split(";") if s.strip()]

def main() -> None:
    with get_repository()._cm.connect() as con:
        con.execute("SET threads=4")
        for stmt in _statements(UNIVERSE_SQL):
            con.execute(stmt)
        con.execute(OUTCOME_SQL)

        # Which client TYPES were on each side. Done in pandas: the bucket rules are
        # keyword logic that belongs in one place (src/analytics/deal_signals.py),
        # not duplicated as a 60-branch CASE expression.
        df = con.sql("SELECT trade_date, symbol, client_rows FROM deal_event_outcomes").df()

        def buckets(rows: str, want: str) -> str:
            if not rows:
                return ""
            out = set()
            for part in rows.split("||"):
                side, _, name = part.partition("::")
                if side == want:
                    out.add(classify_client(name))
            return ",".join(sorted(out))

        df["buy_buckets"] = df["client_rows"].fillna("").map(lambda r: buckets(r, "BUY"))
        df["sell_buckets"] = df["client_rows"].fillna("").map(lambda r: buckets(r, "SELL"))
        con.register("_bkt", df[["trade_date", "symbol", "buy_buckets", "sell_buckets"]])
        con.execute("ALTER TABLE deal_event_outcomes ADD COLUMN IF NOT EXISTS buy_buckets VARCHAR")
        con.execute("ALTER TABLE deal_event_outcomes ADD COLUMN IF NOT EXISTS sell_buckets VARCHAR")
        con.execute("""
            UPDATE deal_event_outcomes o
            SET buy_buckets = b.buy_buckets, sell_buckets = b.sell_buckets
            FROM _bkt b
            WHERE b.trade_date = o.trade_date AND b.symbol = o.symbol
        """)

        print(con.sql("""
            SELECT CASE WHEN is_bulk = 1 THEN 'BULK' ELSE 'BLOCK' END AS kind,
                   side, count(*) AS events,
                   round(100 * avg(m21), 2) AS excess_1m_pct,
                   round(100 * avg(m42), 2) AS excess_2m_pct
            FROM deal_event_outcomes GROUP BY 1, 2 ORDER BY 1, 2
        """).df().to_string(index=False))
        n = con.sql("SELECT count(*) FROM deal_event_outcomes").fetchone()[0]
        print(f"\ndeal_event_outcomes: {n:,} scored events")
        con.close()


if __name__ == "__main__":
    main()
