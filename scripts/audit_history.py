"""
Read-only deep audit of the ~348-day downloaded history.
Run AFTER the F&O re-ingest completes. Checks coverage, duplicates, completeness,
post-fix consistency (settle = prev close, OI scale), and sanity ranges.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duckdb

DB = str(Path(__file__).resolve().parent.parent / "data" / "market_data.duckdb")
c = duckdb.connect(DB, read_only=True)


def hdr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ── 1. COVERAGE & RANGE per table ────────────────────────────────────────────
hdr("1. COVERAGE & DATE RANGE")
for t in ["daily_data", "index_data", "fno_bhavcopy", "fao_participant",
          "fii_derivatives_stats"]:
    try:
        mn, mx, n = c.execute(
            f"SELECT min(trade_date), max(trade_date), count(DISTINCT trade_date) FROM {t}"
        ).fetchone()
        print(f"  {t:24} {n:4} days   {mn} -> {mx}")
    except Exception as e:
        print(f"  {t:24} ERROR {e}")

# ── 2. GAPS: trading days in daily_data missing from other tables ────────────
hdr("2. COVERAGE GAPS vs daily_data (last 348 trading days)")
base = [r[0] for r in c.execute(
    "SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date DESC LIMIT 348"
).fetchall()]
lo = min(base)
for t in ["index_data", "fno_bhavcopy", "fao_participant", "fii_derivatives_stats"]:
    present = {r[0] for r in c.execute(
        f"SELECT DISTINCT trade_date FROM {t} WHERE trade_date >= ?", [lo]
    ).fetchall()}
    missing = [d for d in base if d not in present]
    print(f"  {t:24} missing {len(missing):3} of {len(base)} days"
          + (f"  e.g. {missing[:5]}" if missing else "  ✓"))

# ── 3. WEEKEND / PHANTOM rows ────────────────────────────────────────────────
hdr("3. WEEKEND / NON-TRADING DATE ROWS (should be 0)")
for t in ["daily_data", "index_data", "fno_bhavcopy"]:
    n = c.execute(
        f"SELECT count(*) FROM {t} WHERE dayofweek(trade_date) IN (0,6)"
    ).fetchone()[0]
    print(f"  {t:24} {n} weekend rows")

# ── 4. DUPLICATES ────────────────────────────────────────────────────────────
hdr("4. DUPLICATE KEYS (should be 0)")
dups = {
    "daily_data": "SELECT count(*) FROM (SELECT symbol,trade_date FROM daily_data GROUP BY 1,2 HAVING count(*)>1)",
    "index_data": "SELECT count(*) FROM (SELECT index_name,trade_date FROM index_data GROUP BY 1,2 HAVING count(*)>1)",
    "fno_bhavcopy": "SELECT count(*) FROM (SELECT trade_date,instrument,symbol,expiry_date,strike_price,option_type FROM fno_bhavcopy GROUP BY 1,2,3,4,5,6 HAVING count(*)>1)",
}
for t, q in dups.items():
    print(f"  {t:24} {c.execute(q).fetchone()[0]} duplicate keys")

# ── 5. CRITICAL FIELD COMPLETENESS ───────────────────────────────────────────
hdr("5. NULL / ZERO RATES on critical fields (% of rows)")
checks = [
    ("daily_data", "deliv_per",   "deliv_per IS NULL OR deliv_per=0"),
    ("daily_data", "close_price", "close_price IS NULL OR close_price=0"),
    ("daily_data", "prev_close",  "prev_close IS NULL OR prev_close=0"),
    ("index_data", "pct_chg",     "pct_chg IS NULL"),
    ("index_data", "prev_close",  "prev_close IS NULL OR prev_close=0"),
    ("fno_bhavcopy", "open_interest", "open_interest IS NULL OR open_interest=0"),
    ("fno_bhavcopy", "settle_price",  "settle_price IS NULL OR settle_price=0"),
    ("fno_bhavcopy", "close_price",   "close_price IS NULL OR close_price=0"),
]
for t, f, cond in checks:
    tot = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    bad = c.execute(f"SELECT count(*) FROM {t} WHERE {cond}").fetchone()[0]
    print(f"  {t}.{f:16} {bad/tot*100:5.1f}%  ({bad:,}/{tot:,})")

# ── 6. SETTLE = PREV-CLOSE consistency (the bug we fixed) ─────────────────────
hdr("6. settle_price == previous-day close?  (FUTSTK, should be ~0 diff)")
df = c.execute("""
    WITH f AS (
        SELECT symbol, expiry_date, trade_date, close_price, settle_price,
               LAG(close_price) OVER (PARTITION BY symbol,expiry_date ORDER BY trade_date) AS prev_close
        FROM fno_bhavcopy WHERE instrument='FUTSTK'
    )
    SELECT
      median(abs(settle_price - prev_close)) AS med_abs_diff,
      avg(CASE WHEN abs(settle_price-prev_close) < 0.5 THEN 1 ELSE 0 END)*100 AS pct_match
    FROM f WHERE prev_close IS NOT NULL AND prev_close>0
""").fetchone()
print(f"  median |settle(t) - close(t-1)| = {df[0]:.3f}   (~0 = correct prev-close)")
print(f"  rows where they match (<0.5)    = {df[1]:.1f}%")

# ── 7. OI SCALE CONSISTENCY across dates (near-month, day-over-day) ───────────
hdr("7. NEAR-MONTH FUTSTK day-over-day OI change distribution")
df = c.execute("""
    WITH near AS (
        SELECT symbol, trade_date, MIN(expiry_date) AS exp
        FROM fno_bhavcopy WHERE instrument='FUTSTK' AND expiry_date>=trade_date
        GROUP BY 1,2
    ),
    oi AS (
        SELECT f.symbol, f.trade_date, SUM(f.open_interest) AS o
        FROM fno_bhavcopy f JOIN near n
          ON f.symbol=n.symbol AND f.trade_date=n.trade_date AND f.expiry_date=n.exp
        WHERE f.instrument='FUTSTK' GROUP BY 1,2
    ),
    chg AS (
        SELECT symbol, trade_date,
          (o - LAG(o) OVER (PARTITION BY symbol ORDER BY trade_date))
            / NULLIF(LAG(o) OVER (PARTITION BY symbol ORDER BY trade_date),0) * 100 AS pct
        FROM oi
    )
    SELECT
      quantile_cont(abs(pct),0.5)  AS p50,
      quantile_cont(abs(pct),0.95) AS p95,
      max(abs(pct))                AS mx,
      avg(CASE WHEN abs(pct)>200 THEN 1 ELSE 0 END)*100 AS pct_over_200
    FROM chg WHERE pct IS NOT NULL
""").fetchone()
print(f"  |OI chg %| median={df[0]:.1f}  p95={df[1]:.1f}  max={df[2]:.0f}  | >200%: {df[3]:.1f}% of rows")
print("  (sane near-month OI changes are single/low-double digits)")

# ── 8. SANITY RANGES ─────────────────────────────────────────────────────────
hdr("8. SANITY RANGES")
print("  index_data |pct_chg|>15% rows:",
      c.execute("SELECT count(*) FROM index_data WHERE abs(pct_chg)>15").fetchone()[0])
print("  daily_data deliv_per>100 or <0 rows:",
      c.execute("SELECT count(*) FROM daily_data WHERE deliv_per>100 OR deliv_per<0").fetchone()[0])
# prev_close continuity in index_data
pc = c.execute("""
    WITH x AS (SELECT index_name,trade_date,prev_close,
                 LAG(close_val) OVER (PARTITION BY index_name ORDER BY trade_date) AS lag_close
               FROM index_data WHERE index_name='Nifty 50')
    SELECT avg(CASE WHEN abs(prev_close-lag_close)<1 THEN 1 ELSE 0 END)*100
    FROM x WHERE lag_close IS NOT NULL
""").fetchone()[0]
print(f"  Nifty prev_close == prior close_val: {pc:.1f}% (should be ~100%)")

# ── 9. F&O ROWS PER DAY (partial-fetch detector) ─────────────────────────────
hdr("9. F&O ROWS PER DAY (low outliers = partial fetch)")
df = c.execute("""
    SELECT trade_date, count(*) n FROM fno_bhavcopy GROUP BY 1 ORDER BY n ASC LIMIT 5
""").fetchall()
print("  lowest-row days:", [(str(d), n) for d, n in df])
df = c.execute("SELECT avg(n),min(n),max(n) FROM (SELECT trade_date,count(*) n FROM fno_bhavcopy GROUP BY 1)").fetchone()
print(f"  rows/day  avg={df[0]:,.0f}  min={df[1]:,}  max={df[2]:,}")

c.close()
print("\nAUDIT COMPLETE")
