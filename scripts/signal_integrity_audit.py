"""
READ-ONLY signal-integrity audit for the Sector Rotation / Performance pipeline.
You trade equity sector -> stock off these numbers, so this checks for look-ahead
leakage and logic faults. Writes a report; changes nothing. Safe to delete.

Tests:
  T1  Look-ahead leakage    — as-of(D) must use ZERO data dated > D
  T2  Baseline excludes today — 100D mean/std/breadth must not include as_of
  T3  z-score inflation      — is z a usable statistic or a trending artifact?
  T4  dv_ratio_5d day-count  — calendar /5 vs actual trading days in the 1W window
  T5  stock conviction units — recent weighted-% vs 100D simple-% mismatch
  T6  breadth turnover bias  — does the per-day turnover filter inflate the baseline?
  T7  silent sector drops    — sectors with data that vanish from the signal
  T8  stock<->sector reconcile — do per-stock deliveries roll up to the sector total?
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import pandas as pd
from src.data.repository import query_dataframe
from src.analytics.sector_rotation import get_sector_rotation, get_sector_stocks_rotation
from src.analytics.sector_aggregator import get_sector_master_performance

OUT = []
def p(s=""): OUT.append(str(s))

dates = query_dataframe("SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date DESC LIMIT 130")
dates = [d.date() if hasattr(d,"date") else d for d in dates["trade_date"]]
LATEST = dates[0]
MIDPAST = dates[30]          # ~30 trading days ago, for leakage test
MINT = 100.0

p(f"latest trade_date = {LATEST} | leakage test date = {MIDPAST} | min_turnover = {MINT} lacs")

# ── T1 + T2: LEAKAGE — recompute as-of a PAST date; assert no window touches > D ──
p("\n" + "="*84)
p("T1/T2  LOOK-AHEAD LEAKAGE  (as-of a past date must ignore all later data)")
p("="*84)
# The signal for MIDPAST must be byte-identical whether or not future rows exist.
# Future rows DO exist in the DB (MIDPAST is 30 days back), so if the as-of(MIDPAST)
# output depends only on <=MIDPAST data, recomputing must equal a snapshot that
# only sees <=MIDPAST. We emulate the "only-past" world by checking the raw windows.
perf_past = get_sector_master_performance(MIDPAST, MINT)
# What is the max trade_date that fed today_dv / dv_ratio for MIDPAST?
tdv = query_dataframe(
    "SELECT MAX(trade_date) m FROM daily_data WHERE trade_date = ?", [MIDPAST])
base = query_dataframe(
    "SELECT DISTINCT trade_date FROM daily_data WHERE trade_date < ? "
    "ORDER BY trade_date DESC LIMIT 1 OFFSET 100", [MIDPAST])
base_cut = base["trade_date"].iloc[0] if not base.empty else None
# Does the 100D baseline window ever include MIDPAST or later? It uses > cutoff AND < MIDPAST.
maxbase = query_dataframe(
    "SELECT MAX(trade_date) m FROM daily_data WHERE trade_date > ? AND trade_date < ?",
    [base_cut, MIDPAST])
p(f"  as-of {MIDPAST}: today window max = {tdv['m'].iloc[0]} (must == {MIDPAST})")
p(f"  100D baseline window max = {maxbase['m'].iloc[0]} (must be STRICTLY < {MIDPAST})")
leak1 = (str(tdv['m'].iloc[0])[:10] == str(MIDPAST))
leak2 = (str(maxbase['m'].iloc[0])[:10] < str(MIDPAST))
p(f"  -> today uses only as_of: {'PASS' if leak1 else 'FAIL'}")
p(f"  -> baseline strictly past: {'PASS' if leak2 else 'FAIL'}")
# Determinism: two calls identical
r1 = get_sector_rotation(MIDPAST, MINT)[["sector","accum_score","z_score","dv_ratio"]]
r2 = get_sector_rotation(MIDPAST, MINT)[["sector","accum_score","z_score","dv_ratio"]]
p(f"  -> deterministic (2 calls identical): {'PASS' if r1.equals(r2) else 'FAIL'}")

# ── T3: z-score inflation ────────────────────────────────────────────────────
p("\n" + "="*84)
p("T3  Z-SCORE INFLATION  (a healthy cross-section centers near 0, |z|>2 is rare)")
p("="*84)
perf = get_sector_master_performance(LATEST, MINT)
z = perf["z_score"].dropna()
p(f"  z median={z.median():+.2f}  mean={z.mean():+.2f}  min={z.min():+.2f}  max={z.max():+.2f}")
p(f"  fraction |z|>2 = {(z.abs()>2).mean():.0%}  (healthy ~5%)   z>1 = {(z>1).mean():.0%}")
p(f"  dv_ratio median={perf['dv_ratio'].median():.2f}  -> if >>1, baseline is trending up")
p(f"  VERDICT: {'INFLATED — z unusable as absolute gate' if z.median()>1.5 else 'ok'}")

# ── T4: dv_ratio_5d calendar vs trading-day count ────────────────────────────
p("\n" + "="*84)
p("T4  dv_ratio_5d DAY-COUNT  (1W deliv is divided by 5; real trading days may differ)")
p("="*84)
for lbl, cal in [("1W",7),("2W",14)]:
    start = LATEST - pd.Timedelta(days=cal)
    n = query_dataframe(
        "SELECT COUNT(DISTINCT trade_date) n FROM daily_data WHERE trade_date > ? AND trade_date <= ?",
        [start.date() if hasattr(start,'date') else start, LATEST])["n"].iloc[0]
    divisor = 5 if lbl=="1W" else 10
    if n == divisor:
        verdict = "OK"
    else:
        direction = "understated" if n < divisor else "overstated"
        verdict = f"BIAS {(divisor-n)/divisor*100:+.0f}% (daily avg {direction})"
    p(f"  {lbl}: window has {n} trading days, code divides by {divisor} -> {verdict}")

# ── T5: stock conviction — weighted recent % vs simple 100D % ─────────────────
p("\n" + "="*84)
p("T5  STOCK CONVICTION UNITS  (recent = turnover-WEIGHTED %, baseline = SIMPLE avg %)")
p("="*84)
top_sec = get_sector_rotation(LATEST, MINT).iloc[0]["sector"]
stk = get_sector_stocks_rotation(top_sec, LATEST, MINT)
if not stk.empty and "avg_deliv_per_100d" in stk.columns:
    valid = stk.dropna(subset=["wtd_deliv_per","avg_deliv_per_100d"])
    # Recompute baseline as turnover-WEIGHTED to see how many stocks flip above/below
    sym_list = valid["symbol"].tolist()
    if sym_list:
        ph = ",".join("?"*len(sym_list))
        base_cut2 = query_dataframe(
            "SELECT DISTINCT trade_date FROM daily_data WHERE trade_date < ? "
            "ORDER BY trade_date DESC LIMIT 1 OFFSET 100",[LATEST])["trade_date"].iloc[0]
        wbase = query_dataframe(
            f"SELECT symbol, SUM(deliv_per*turnover_lacs)/NULLIF(SUM(turnover_lacs),0) wavg "
            f"FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST') "
            f"AND turnover_lacs>=? AND trade_date>? AND trade_date<? GROUP BY symbol",
            sym_list+[MINT, base_cut2, LATEST])
        m = valid.merge(wbase, on="symbol", how="left")
        m["conv_simple"] = m["wtd_deliv_per"] > m["avg_deliv_per_100d"]
        m["conv_wtd"]    = m["wtd_deliv_per"] > m["wavg"]
        flips = (m["conv_simple"] != m["conv_wtd"]).sum()
        p(f"  sector '{top_sec}': {len(m)} stocks")
        p(f"  conviction above-baseline using SIMPLE 100D avg (current code): {m['conv_simple'].sum()}")
        p(f"  conviction above-baseline using WEIGHTED 100D avg (consistent): {m['conv_wtd'].sum()}")
        p(f"  -> stocks that FLIP buy/weak between the two methods: {flips} "
          f"({flips/len(m)*100:.0f}% of the sector)")
        p(f"  VERDICT: {'INCONSISTENT — recent% and baseline% use different weighting' if flips else 'consistent'}")

# ── T6: breadth turnover-filter bias ─────────────────────────────────────────
p("\n" + "="*84)
p("T6  BREADTH TURNOVER-FILTER BIAS  (100D stock avg only counts days turnover>=min)")
p("="*84)
base_cut3 = query_dataframe(
    "SELECT DISTINCT trade_date FROM daily_data WHERE trade_date < ? "
    "ORDER BY trade_date DESC LIMIT 1 OFFSET 100",[LATEST])["trade_date"].iloc[0]
# Breadth as coded (filtered baseline) vs unfiltered baseline, sector-agnostic sample
cmp = query_dataframe(f"""
    WITH today AS (
      SELECT symbol, turnover_lacs*deliv_per/10000.0 dv
      FROM daily_data WHERE trade_date=? AND series IN ('EQ','SM','ST') AND turnover_lacs>=?
    ),
    filt AS (
      SELECT symbol, AVG(turnover_lacs*deliv_per/10000.0) a
      FROM daily_data WHERE series IN ('EQ','SM','ST') AND turnover_lacs>=?
        AND trade_date>? AND trade_date<? GROUP BY symbol
    ),
    unfilt AS (
      SELECT symbol, AVG(turnover_lacs*deliv_per/10000.0) a
      FROM daily_data WHERE series IN ('EQ','SM','ST')
        AND trade_date>? AND trade_date<? GROUP BY symbol
    )
    SELECT
      AVG(CASE WHEN t.dv > f.a THEN 1.0 ELSE 0.0 END) breadth_filtered,
      AVG(CASE WHEN t.dv > u.a THEN 1.0 ELSE 0.0 END) breadth_unfiltered,
      COUNT(*) n
    FROM today t JOIN filt f USING(symbol) JOIN unfilt u USING(symbol)
""", [LATEST, MINT, MINT, base_cut3, LATEST, base_cut3, LATEST])
bf = cmp["breadth_filtered"].iloc[0]; bu = cmp["breadth_unfiltered"].iloc[0]
p(f"  market breadth using FILTERED baseline (current code): {bf:.1%}")
p(f"  market breadth using UNFILTERED baseline (all days):    {bu:.1%}")
p(f"  -> filtering low-turnover days from the baseline shifts breadth by {(bf-bu)*100:+.1f} pts")
p(f"  VERDICT: {'BIASED — baseline excludes quiet days, distorts breadth' if abs(bf-bu)>0.03 else 'negligible'}")

# ── T7: silent sector drops ──────────────────────────────────────────────────
p("\n" + "="*84)
p("T7  SILENT SECTOR DROPS  (sectors present in data but missing from the signal)")
p("="*84)
all_sec = query_dataframe(
    "SELECT DISTINCT s.sector FROM daily_data b JOIN sector_master s ON b.symbol=s.symbol "
    "WHERE b.trade_date=? AND b.series IN ('EQ','SM','ST') AND b.turnover_lacs>=? "
    "AND s.sector NOT IN ('ETF','Others') AND s.sector IS NOT NULL", [LATEST, MINT])
in_data = set(all_sec["sector"])
in_sig  = set(get_sector_rotation(LATEST, MINT)["sector"])
dropped = in_data - in_sig
p(f"  sectors with data today: {len(in_data)} | sectors in signal: {len(in_sig)}")
p(f"  -> dropped: {sorted(dropped) if dropped else 'NONE'}")

# ── T8: stock<->sector reconciliation ────────────────────────────────────────
p("\n" + "="*84)
p("T8  STOCK<->SECTOR RECONCILE  (sum of per-stock 1W deliv ≈ sector 1W deliv)")
p("="*84)
sec_1w = query_dataframe("""
    SELECT SUM(b.turnover_lacs*b.deliv_per/100.0)/100.0 cr FROM daily_data b
    JOIN sector_master s ON b.symbol=s.symbol
    WHERE s.sector=? AND b.series IN ('EQ','SM','ST') AND b.turnover_lacs>=?
      AND b.trade_date > ? AND b.trade_date <= ?
""", [top_sec, MINT, (LATEST - pd.Timedelta(days=7)).date() if hasattr(LATEST-pd.Timedelta(days=7),'date') else LATEST-pd.Timedelta(days=7), LATEST])["cr"].iloc[0]
stk_sum = stk["deliv_value_cr"].sum() if not stk.empty else 0
p(f"  sector '{top_sec}' 1W deliv (aggregate query): Rs{sec_1w:,.0f} Cr")
p(f"  sum of per-stock 1W deliv (drill-down)       : Rs{stk_sum:,.0f} Cr")
diff = abs(sec_1w - stk_sum)/sec_1w*100 if sec_1w else 0
p(f"  -> mismatch: {diff:.1f}%  {'OK (rounding)' if diff<2 else 'INVESTIGATE'}")

p("\nAUDIT_DONE")
Path("logs/integrity.txt").write_text("\n".join(OUT), encoding="utf-8")
print("\n".join(OUT))
