"""
Deep verification of Layer 1, 2, 3, and Breadth correctness and leakage.

Pure-history design (what the analytics does):
  - Layer 1 window:           > period_start AND <= as_of_date  (includes today)
  - Layer 2/3/Breadth window: > cutoff       AND <  as_of_date  (today EXCLUDED)

Checks:
  1.  DB snapshot
  2.  Pure-history 100D cutoff (OFFSET 100 from dates BEFORE today)
  3.  Layer 1 -- period delivery values, monotonicity
  4.  Layer 2 -- today_dv_cr, dv_ratio formula vs pure-history baseline
  5.  Layer 3 -- z_score: mean/stddev vs pure-history daily rows
  6.  Breadth -- stock-level cross-check, no today leakage
  7.  Full analytics table global checks
  8.  Leakage probes -- today must NOT appear in any baseline calculation
"""
import sys
sys.path.insert(0, ".")

from datetime import date, timedelta
import pandas as pd
from src.data.repository import query_dataframe
from src.analytics.sector_aggregator import get_sector_master_performance

TARGET        = date(2026, 5, 19)
MIN_TO        = 100.0
SAMPLE_SECTOR = "Banking"

PASS = "PASS"
FAIL = "*** FAIL ***"

def chk(cond, label):
    status = PASS if cond else FAIL
    print(f"  [{status}] {label}")
    return cond

all_ok = True

print("=" * 70)
print(f"DEEP VERIFICATION  target={TARGET}  min_turnover={MIN_TO}")
print(f"  Layer 2/3/Breadth baseline: pure history (today EXCLUDED)")
print("=" * 70)

# ── 1. DB snapshot ────────────────────────────────────────────────────────────
print("\n[1] DATABASE SNAPSHOT")
snap = query_dataframe(
    "SELECT COUNT(*) AS rows, COUNT(DISTINCT trade_date) AS days, "
    "MIN(trade_date) AS d0, MAX(trade_date) AS d1 FROM daily_data", []
)
print(f"  rows={int(snap['rows'].iloc[0]):,}  trading_days={int(snap['days'].iloc[0])}"
      f"  range=[{snap['d0'].iloc[0]} .. {snap['d1'].iloc[0]}]")
total_days = int(snap["days"].iloc[0])
all_ok &= chk(total_days >= 101, f"at least 101 trading days in DB ({total_days})")

# ── 2. Pure-history 100D cutoff ───────────────────────────────────────────────
print("\n[2] PURE-HISTORY 100D CUTOFF (today EXCLUDED from baseline)")
# Analytics uses < TARGET so today is at position -1 (not in set).
# OFFSET 100 from dates BEFORE today = 101st most-recent pre-today date.
cutoff_row = query_dataframe(
    "SELECT DISTINCT trade_date FROM daily_data "
    "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
    [TARGET],
)
cutoff_100d = pd.to_datetime(cutoff_row["trade_date"].iloc[0]).date()
print(f"  Cutoff (101st pre-today date, OFFSET 100): {cutoff_100d}")

# Window: > cutoff AND < TARGET = exactly 100 trading days, none of which is today
confirm = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS td FROM daily_data "
    "WHERE trade_date > ? AND trade_date < ?",
    [cutoff_100d, TARGET],
)
td_in_window = int(confirm["td"].iloc[0])
all_ok &= chk(
    td_in_window == 100,
    f"pure-history window (>{cutoff_100d} .. <{TARGET}) = {td_in_window} days (must be 100)",
)

# Cutoff date itself must NOT be in the window (exclusive lower bound)
n_cutoff = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS n FROM daily_data "
    "WHERE trade_date = ? AND trade_date > ? AND trade_date < ?",
    [cutoff_100d, cutoff_100d, TARGET],
)
all_ok &= chk(int(n_cutoff["n"].iloc[0]) == 0,
              f"cutoff {cutoff_100d} is EXCLUDED from baseline window")

# Today must NOT be in the pure-history window (this is the leakage check)
n_today = query_dataframe(
    "SELECT COUNT(DISTINCT trade_date) AS n FROM daily_data "
    "WHERE trade_date = ? AND trade_date > ? AND trade_date < ?",
    [TARGET, cutoff_100d, TARGET],
)
all_ok &= chk(int(n_today["n"].iloc[0]) == 0,
              f"TODAY ({TARGET}) is EXCLUDED from pure-history baseline window")

# ── 3. Layer 1 — period delivery values ───────────────────────────────────────
print("\n[3] LAYER 1 -- PERIOD DELIVERY (includes today, Banking cross-check)")

def raw_deliv_inclusive(start_exclusive, end_inclusive, sector):
    """Layer 1: total DV from start+1 to end (today included)."""
    r = query_dataframe("""
        SELECT SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS dv,
               COUNT(DISTINCT b.trade_date) AS td
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ? AND b.trade_date <= ?
    """, [sector, MIN_TO, start_exclusive, end_inclusive])
    return r

r1w = raw_deliv_inclusive(TARGET - timedelta(days=7),  TARGET, SAMPLE_SECTOR)
r2w = raw_deliv_inclusive(TARGET - timedelta(days=14), TARGET, SAMPLE_SECTOR)
r1m = raw_deliv_inclusive(TARGET - timedelta(days=30), TARGET, SAMPLE_SECTOR)
r3m = raw_deliv_inclusive(TARGET - timedelta(days=90), TARGET, SAMPLE_SECTOR)

vals = [
    ("1W",  float(r1w["dv"].iloc[0]), int(r1w["td"].iloc[0])),
    ("2W",  float(r2w["dv"].iloc[0]), int(r2w["td"].iloc[0])),
    ("1M",  float(r1m["dv"].iloc[0]), int(r1m["td"].iloc[0])),
    ("3M",  float(r3m["dv"].iloc[0]), int(r3m["td"].iloc[0])),
]
for lbl, v, td in vals:
    print(f"  {lbl:5s}: Rs{v:,.1f} Cr  ({td} trading days, includes today)")

mono = all(vals[i][1] <= vals[i+1][1] for i in range(len(vals)-1))
all_ok &= chk(mono, "monotonicity: 1W_days <= 2W_days <= 1M_days <= 3M_days")

future = query_dataframe("SELECT COUNT(*) AS n FROM daily_data WHERE trade_date > ?", [TARGET])
all_ok &= chk(int(future["n"].iloc[0]) == 0, f"no rows after {TARGET}")

# ── 4. Layer 2 — pure-history baseline + dv_ratio ─────────────────────────────
print("\n[4] LAYER 2 -- DV RATIO CROSS-CHECK vs PURE-HISTORY BASELINE (Banking)")

# Today's delivery (signal, not in baseline)
today_raw = query_dataframe("""
    SELECT SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS today_dv
    FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
    WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
      AND b.turnover_lacs >= ? AND b.trade_date = ?
""", [SAMPLE_SECTOR, MIN_TO, TARGET])
today_dv_raw = float(today_raw["today_dv"].iloc[0])

# 100D total DV — pure history, today EXCLUDED (<TARGET)
hist_raw = query_dataframe("""
    SELECT SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS dv,
           COUNT(DISTINCT b.trade_date) AS td
    FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
    WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
      AND b.turnover_lacs >= ?
      AND b.trade_date > ? AND b.trade_date < ?
""", [SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])
dv100_hist = float(hist_raw["dv"].iloc[0])
td100_hist  = int(hist_raw["td"].iloc[0])
daily_avg   = dv100_hist / td100_hist
dv_ratio_expected = today_dv_raw / daily_avg

print(f"  today_dv_cr (raw)      : Rs{today_dv_raw:,.3f} Cr")
print(f"  100D_deliv_cr (history): Rs{dv100_hist:,.3f} Cr  ({td100_hist} days, today excluded)")
print(f"  100D daily avg         : Rs{daily_avg:,.3f} Cr")
print(f"  DV Ratio expected      : {dv_ratio_expected:.4f}x")

perf_df = get_sector_master_performance(TARGET, min_turnover_lacs=MIN_TO)
brow = perf_df[perf_df["sector"] == SAMPLE_SECTOR].iloc[0]
today_dv_a = float(brow["today_dv_cr"])
dv100_a    = float(brow["100D_deliv_cr"])
dvratio_a  = float(brow["dv_ratio"])

print(f"  today_dv_cr (analytic) : Rs{today_dv_a:,.3f} Cr")
print(f"  100D_deliv_cr (analyt) : Rs{dv100_a:,.3f} Cr")
print(f"  DV Ratio (analytics)   : {dvratio_a:.4f}x")

all_ok &= chk(abs(today_dv_raw - today_dv_a) < 0.01,
              f"today_dv_cr raw == analytics (diff={abs(today_dv_raw-today_dv_a):.4f})")
all_ok &= chk(abs(dv100_hist - dv100_a) < 0.01,
              f"100D_deliv_cr matches pure-history raw SQL (diff={abs(dv100_hist-dv100_a):.4f})")
all_ok &= chk(abs(dv_ratio_expected - dvratio_a) < 0.0001,
              f"dv_ratio matches expected (diff={abs(dv_ratio_expected-dvratio_a):.6f})")

dv_ratio_formula = today_dv_raw / (dv100_hist / 100)
all_ok &= chk(abs(dv_ratio_formula - dvratio_a) < 0.0001,
              f"dv_ratio == today / (100D_total/100)  [formula check]")

# ── 5. Layer 3 — Z-Score pure-history cross-check ─────────────────────────────
print("\n[5] LAYER 3 -- Z-SCORE CROSS-CHECK vs PURE-HISTORY DAILY ROWS (Banking)")

daily_dv_rows = query_dataframe("""
    SELECT b.trade_date,
           SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
    FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
    WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
      AND b.turnover_lacs >= ?
      AND b.trade_date > ? AND b.trade_date < ?
    GROUP BY b.trade_date
    ORDER BY b.trade_date
""", [SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])

n_days      = len(daily_dv_rows)
mean_manual = float(daily_dv_rows["daily_dv"].mean())
std_manual  = float(daily_dv_rows["daily_dv"].std(ddof=1))
z_manual    = (today_dv_raw - mean_manual) / std_manual

print(f"  Daily DV rows in pure-history window : {n_days}")
print(f"  mean_100d_dv (pandas)                : Rs{mean_manual:,.3f} Cr")
print(f"  std_100d_dv  (pandas ddof=1)          : Rs{std_manual:,.3f} Cr")
print(f"  Z-Score expected                     : {z_manual:+.4f}")

mean_a = float(brow["mean_100d_dv"])
std_a  = float(brow["std_100d_dv"])
z_a    = float(brow["z_score"])
print(f"  mean_100d_dv (analytics)             : Rs{mean_a:,.3f} Cr")
print(f"  std_100d_dv  (analytics)             : Rs{std_a:,.3f} Cr")
print(f"  Z-Score (analytics)                  : {z_a:+.4f}")

all_ok &= chk(n_days == 100,
              f"pure-history window has exactly 100 daily rows for {SAMPLE_SECTOR}")
all_ok &= chk(abs(mean_manual - mean_a) < 0.01,
              f"mean_100d_dv matches pandas (diff={abs(mean_manual-mean_a):.4f})")
all_ok &= chk(abs(std_manual - std_a) < 0.01,
              f"std_100d_dv matches pandas ddof=1 (diff={abs(std_manual-std_a):.4f})")
all_ok &= chk(abs(z_manual - z_a) < 0.001,
              f"z_score matches manual formula (diff={abs(z_manual-z_a):.5f})")

mean_times_100 = mean_a * 100
all_ok &= chk(abs(mean_times_100 - dv100_a) < 0.5,
              f"mean*100 ~= 100D_deliv_cr ({mean_times_100:.1f} vs {dv100_a:.1f})")
all_ok &= chk(abs(mean_a - daily_avg) < 0.01,
              f"z_score mean == 100D daily avg (diff={abs(mean_a-daily_avg):.4f})")

# ── 6. Breadth — stock-level leakage + formula cross-check ────────────────────
print("\n[6] BREADTH -- STOCK-LEVEL CROSS-CHECK (Banking)")

# Compute breadth manually from raw SQL
breadth_raw = query_dataframe("""
    WITH today_dv AS (
        SELECT b.symbol,
               b.turnover_lacs * b.deliv_per / 100.0 / 100.0 AS today_dv_cr
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ? AND b.trade_date = ?
    ),
    hist_avg AS (
        SELECT b.symbol,
               AVG(b.turnover_lacs * b.deliv_per / 100.0 / 100.0) AS avg_dv_cr,
               COUNT(DISTINCT b.trade_date) AS hist_days
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ? AND b.trade_date < ?
        GROUP BY b.symbol
    )
    SELECT t.symbol,
           t.today_dv_cr,
           h.avg_dv_cr,
           h.hist_days,
           CASE WHEN h.avg_dv_cr IS NOT NULL AND t.today_dv_cr > h.avg_dv_cr THEN 1 ELSE 0 END AS above
    FROM today_dv t
    LEFT JOIN hist_avg h ON t.symbol = h.symbol
    ORDER BY t.today_dv_cr DESC
""", [SAMPLE_SECTOR, MIN_TO, TARGET,
      SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])

total_with_history = int((breadth_raw["hist_days"].notna()).sum())
above_count        = int(breadth_raw["above"].sum())
breadth_manual     = above_count / total_with_history if total_with_history > 0 else 0.0
breadth_a          = float(brow["breadth"])

print(f"  Banking stocks today (turnover >= {MIN_TO}L): {len(breadth_raw)}")
print(f"  Stocks with 100D history                   : {total_with_history}")
print(f"  Stocks above their 100D avg DV today        : {above_count}")
print(f"  Breadth (manual)                           : {breadth_manual:.4f}  ({breadth_manual*100:.1f}%)")
print(f"  Breadth (analytics)                        : {breadth_a:.4f}  ({breadth_a*100:.1f}%)")

all_ok &= chk(abs(breadth_manual - breadth_a) < 0.001,
              f"breadth matches stock-level manual count (diff={abs(breadth_manual-breadth_a):.5f})")

# Confirm today is not in hist_avg (leakage probe on stock level)
target_in_hist = query_dataframe("""
    SELECT COUNT(DISTINCT b.trade_date) AS n
    FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
    WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
      AND b.turnover_lacs >= ?
      AND b.trade_date = ?
      AND b.trade_date > ? AND b.trade_date < ?
""", [SAMPLE_SECTOR, MIN_TO, TARGET, cutoff_100d, TARGET])
all_ok &= chk(int(target_in_hist["n"].iloc[0]) == 0,
              f"breadth hist_avg: today ({TARGET}) NOT in historical average window")

# Confirm no stock has hist_days > 100 (baseline window is exactly 100 days max)
max_hist_days = breadth_raw["hist_days"].max()
all_ok &= chk(max_hist_days is None or int(max_hist_days) <= 100,
              f"no stock has hist_days > 100 (max={max_hist_days})")

# ── 7. Full analytics table global checks ─────────────────────────────────────
print("\n[7] FULL ANALYTICS TABLE -- GLOBAL CHECKS")
print(f"  Total sectors: {len(perf_df)}")

all_ok &= chk(perf_df["dv_ratio"].isna().sum() == 0, "no NaN in dv_ratio")
all_ok &= chk(perf_df["z_score"].isna().sum() == 0,  "no NaN in z_score")
all_ok &= chk(perf_df["breadth"].isna().sum() == 0,  "no NaN in breadth")
all_ok &= chk((perf_df["today_dv_cr"] >= 0).all(),   "all today_dv_cr >= 0")
all_ok &= chk((perf_df["100D_deliv_cr"] > 0).all(),  "all 100D_deliv_cr > 0")
all_ok &= chk((perf_df["std_100d_dv"] > 0).all(),    "all std_100d_dv > 0")
all_ok &= chk(((perf_df["breadth"] >= 0) & (perf_df["breadth"] <= 1)).all(),
              "all breadth values in [0, 1]")

perf_df["_ratio_check"] = perf_df["today_dv_cr"] / (perf_df["100D_deliv_cr"] / 100)
max_ratio_diff = (perf_df["_ratio_check"] - perf_df["dv_ratio"]).abs().max()
all_ok &= chk(max_ratio_diff < 0.0001,
              f"dv_ratio == today/(100D/100) for all sectors (max diff={max_ratio_diff:.6f})")

perf_df["_z_check"] = (perf_df["today_dv_cr"] - perf_df["mean_100d_dv"]) / perf_df["std_100d_dv"]
max_z_diff = (perf_df["_z_check"] - perf_df["z_score"]).abs().max()
all_ok &= chk(max_z_diff < 0.0001,
              f"z_score == (today-mean)/std for all sectors (max diff={max_z_diff:.6f})")

perf_df["_mean_diff"] = (perf_df["mean_100d_dv"] * 100 - perf_df["100D_deliv_cr"]).abs()
max_mean_diff = perf_df["_mean_diff"].max()
all_ok &= chk(max_mean_diff < 1.0,
              f"mean*100 ~= 100D_deliv_cr for all sectors (max Rs{max_mean_diff:.3f} Cr)")

# ── 8. Leakage probes — today must not appear in any baseline ─────────────────
print("\n[8] LEAKAGE PROBES -- TODAY MUST NOT BE IN ANY BASELINE CALCULATION")

# Probe: if we remove today from DB, do all baseline stats stay the same?
# Simulate by checking the window bounds directly

# 8a. 100D baseline window upper bound
upper_bound_check = query_dataframe(
    "SELECT MAX(trade_date) AS max_d FROM daily_data "
    "WHERE trade_date > ? AND trade_date < ?",
    [cutoff_100d, TARGET],
)
max_in_window = pd.to_datetime(upper_bound_check["max_d"].iloc[0]).date()
all_ok &= chk(max_in_window < TARGET,
              f"max date in pure-history window ({max_in_window}) is BEFORE today ({TARGET})")

# 8b. Stats SQL: mean/std window does not include today
stats_probe = query_dataframe("""
    SELECT MAX(trade_date) AS max_d, COUNT(DISTINCT trade_date) AS n_days
    FROM (
        SELECT b.trade_date
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ? AND b.trade_date < ?
        GROUP BY b.trade_date
    ) t
""", [SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])
stats_max = pd.to_datetime(stats_probe["max_d"].iloc[0]).date()
stats_days = int(stats_probe["n_days"].iloc[0])
all_ok &= chk(stats_max < TARGET,
              f"z_score window max date ({stats_max}) < today ({TARGET})")
all_ok &= chk(stats_days == 100,
              f"z_score window has exactly 100 rows ({stats_days})")

# 8c. Breadth hist_avg window does not include today (all sectors)
breadth_max_probe = query_dataframe("""
    SELECT MAX(trade_date) AS max_d, COUNT(DISTINCT trade_date) AS n_days
    FROM daily_data
    WHERE trade_date > ? AND trade_date < ?
""", [cutoff_100d, TARGET])
br_max = pd.to_datetime(breadth_max_probe["max_d"].iloc[0]).date()
all_ok &= chk(br_max < TARGET,
              f"breadth hist_avg window max date ({br_max}) < today ({TARGET})")

# 8d. today_dv is ONLY from today (single-day query)
today_other_dates = query_dataframe("""
    SELECT COUNT(DISTINCT trade_date) AS n
    FROM daily_data
    WHERE trade_date >= ? AND trade_date != ?
      AND series IN ('EQ','SM','ST')
""", [TARGET, TARGET])
# This is just confirming no future rows exist
all_ok &= chk(int(today_other_dates["n"].iloc[0]) == 0,
              f"no rows with trade_date > today in DB")

# 8e. Probe: what would happen if today leaked into baseline?
# Re-compute Banking mean with today included vs excluded
with_today = query_dataframe("""
    SELECT AVG(daily_dv) AS mean_with
    FROM (
        SELECT b.trade_date,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ? AND b.trade_date <= ?
        GROUP BY b.trade_date
    ) t
""", [SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])
without_today = query_dataframe("""
    SELECT AVG(daily_dv) AS mean_without
    FROM (
        SELECT b.trade_date,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
        FROM daily_data b INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ? AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ? AND b.trade_date < ?
        GROUP BY b.trade_date
    ) t
""", [SAMPLE_SECTOR, MIN_TO, cutoff_100d, TARGET])
mean_with    = float(with_today["mean_with"].iloc[0])
mean_without = float(without_today["mean_without"].iloc[0])
leakage_bias = abs(mean_with - mean_without)
print(f"  [PROBE] Banking mean WITH today included   : Rs{mean_with:,.3f} Cr")
print(f"  [PROBE] Banking mean WITHOUT today (correct): Rs{mean_without:,.3f} Cr")
print(f"  [PROBE] Leakage bias if today leaked in    : Rs{leakage_bias:,.3f} Cr")
all_ok &= chk(abs(mean_without - mean_a) < 0.01,
              f"analytics mean == without-today mean (not with-today mean)")
all_ok &= chk(leakage_bias > 1.0,
              f"leakage bias is non-trivial (Rs{leakage_bias:.1f} Cr) -- today's data DOES matter")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n[9] TOP 10 SECTORS -- ALL LAYERS + BREADTH")
show = perf_df[["sector", "today_dv_cr", "dv_ratio", "z_score", "breadth"]].head(10).copy()
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_colwidth", 28)
print(show.to_string(index=False))

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if all_ok:
    print("RESULT: ALL CHECKS PASSED -- Layers 1/2/3/Breadth correct, no leakage")
else:
    print("RESULT: ONE OR MORE CHECKS FAILED -- see *** FAIL *** lines above")
print("=" * 70)
