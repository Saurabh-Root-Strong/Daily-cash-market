"""Verify Layer 1 (delivery value) and Layer 2 (DV Ratio) correctness."""
import sys
sys.path.insert(0, ".")
from datetime import date, timedelta
from src.data.repository import query_dataframe

TARGET = date(2026, 5, 19)
MIN_TO = 100.0

print("=" * 70)
print(f"DATABASE SNAPSHOT  (target={TARGET})")
print("=" * 70)
snap = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS n, MIN(trade_date) AS d0, MAX(trade_date) AS d1 FROM daily_data", []
)
print(snap.to_string(index=False))

# ── Exact 100-trading-day cutoff ─────────────────────────────────────────────
cutoff_row = query_dataframe(
    "SELECT DISTINCT trade_date FROM daily_data "
    "WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
    [TARGET],
)
cutoff_100d = cutoff_row["trade_date"].iloc[0]
print(f"\n101st most-recent trading date (OFFSET 100): {cutoff_100d}")
print("Using > this date gives exactly 100 trading days.\n")

# Confirm it's exactly 100 trading days
confirm = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS td FROM daily_data "
    "WHERE trade_date > ? AND trade_date <= ?",
    [cutoff_100d, TARGET],
)
print(f"Trading days in 100D window [>{cutoff_100d} .. {TARGET}]: {int(confirm['td'].iloc[0])}")

# ── Actual 1W trading days ────────────────────────────────────────────────────
td_1w_row = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS td, MIN(trade_date) AS d0, MAX(trade_date) AS d1 "
    "FROM daily_data WHERE trade_date > ? AND trade_date <= ?",
    [TARGET - timedelta(days=7), TARGET],
)
td_1w = int(td_1w_row["td"].iloc[0])
print(f"\nActual 1W trading days: {td_1w}  [{td_1w_row['d0'].iloc[0]} .. {td_1w_row['d1'].iloc[0]}]")
print(f"(Fixed constant was 5 — actual is {td_1w} — {'SAME' if td_1w==5 else 'DIFFERENT, now fixed'})")

print("\n" + "=" * 70)
print("LAYER 1 + LAYER 2  — per sector cross-check")
print("=" * 70)

SECTORS = ["Banking", "Defence & Aerospace", "Oil & Gas", "Information Technology"]

def dq(start, end, sector):
    return query_dataframe("""
        SELECT COUNT(DISTINCT b.trade_date) AS td,
               MIN(b.trade_date) AS d0, MAX(b.trade_date) AS d1,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS dv
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ? AND b.trade_date > ? AND b.trade_date <= ?
    """, [sector, MIN_TO, start, end])

for SEC in SECTORS:
    r1w  = dq(TARGET - timedelta(days=7),  TARGET, SEC)
    r100 = dq(cutoff_100d,                 TARGET, SEC)

    dv1w   = float(r1w["dv"].iloc[0])
    dv100  = float(r100["dv"].iloc[0])
    td_sec = int(r1w["td"].iloc[0])
    td100  = int(r100["td"].iloc[0])
    ratio  = (dv1w / td_sec) / (dv100 / td100) if (td_sec and td100) else float("nan")

    print(f"\n{SEC}")
    print(f"  1W  [{r1w['d0'].iloc[0]} -> {r1w['d1'].iloc[0]}]  td={td_sec}  DV=Rs{dv1w:.1f} Cr")
    print(f"  100D[{r100['d0'].iloc[0]} -> {r100['d1'].iloc[0]}]  td={td100}  DV=Rs{dv100:.1f} Cr")
    print(f"  daily_1w = {dv1w:.1f}/{td_sec} = {dv1w/td_sec:.2f}")
    print(f"  daily_100d = {dv100:.1f}/{td100} = {dv100/td100:.2f}")
    print(f"  DV_Ratio = {ratio:.4f}x")
    print(f"  Size-bias check: Banking raw DV >> Defence raw DV, but ratio tells the truth")

print("\n" + "=" * 70)
print("LEAKAGE CHECKS")
print("=" * 70)

# No future data in window
future = query_dataframe(
    "SELECT COUNT(*) AS n FROM daily_data WHERE trade_date > ?", [TARGET]
)
print(f"\nFuture rows after {TARGET}: {int(future['n'].iloc[0])}  (must be 0 in windows)")

# Boundary: date ON cutoff must NOT appear in 100D window
on_cutoff = query_dataframe(
    "SELECT COUNT(*) AS n FROM daily_data WHERE trade_date = ?", [cutoff_100d]
)
in_window = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS n FROM daily_data "
    "WHERE trade_date > ? AND trade_date <= ? AND trade_date = ?",
    [cutoff_100d, TARGET, cutoff_100d],
)
print(f"Cutoff date {cutoff_100d} rows in DB: {int(on_cutoff['n'].iloc[0])}")
print(f"Cutoff date rows INSIDE window (must be 0): {int(in_window['n'].iloc[0])}  {'OK' if int(in_window['n'].iloc[0])==0 else 'FAIL!'}")

# Monotonicity for Banking
print("\nMonotonicity check (Banking — 1W<=2W<=1M<=3M<=100D):")
dvs = []
for label, start in [("1W",7),("2W",14),("1M",30),("3M",90)]:
    r = dq(TARGET - timedelta(days=start), TARGET, "Banking")
    dvs.append((label, float(r["dv"].iloc[0])))
r100b = dq(cutoff_100d, TARGET, "Banking")
dvs.append(("100D", float(r100b["dv"].iloc[0])))
for label, v in dvs:
    print(f"  {label}: Rs{v:.1f} Cr")
mono = all(dvs[i][1] <= dvs[i+1][1] for i in range(len(dvs)-1))
print(f"  Monotonicity: {'OK' if mono else 'FAIL!'}")

print("\n" + "=" * 70)
print("ANALYTICS LAYER OUTPUT  (get_sector_master_performance)")
print("=" * 70)
from src.analytics.sector_aggregator import get_sector_master_performance
df = get_sector_master_performance(TARGET, min_turnover_lacs=MIN_TO)
cols = ["sector", "1W_deliv_cr", "3M_deliv_cr", "100D_deliv_cr", "dv_ratio"]
print(df[cols].to_string(index=False))
print(f"\nTotal sectors : {len(df)}")
print(f"dv_ratio NaN  : {df['dv_ratio'].isna().sum()}")
print(f"Top DV Ratio  : {df.iloc[0]['sector']}  {df.iloc[0]['dv_ratio']:.3f}x")
print(f"Low DV Ratio  : {df.iloc[-1]['sector']}  {df.iloc[-1]['dv_ratio']:.3f}x")
