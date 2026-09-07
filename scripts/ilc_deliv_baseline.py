"""Should the delivery baseline be 21 sessions instead of 100?

THE TRAP. Delivery % is PERSISTENT -- it drifts in regimes rather than oscillating
around a fixed level. Normalising against a short trailing window makes the
baseline chase the very move you are trying to detect: if delivery has been
elevated for three weeks, a 21-session mean has already absorbed it and the z
prints ~0 exactly when the signal is strongest. A long window keeps the reference
still, at the cost of being slow to notice a genuine regime change.

Which effect dominates is an empirical question about the autocorrelation of the
series, not a matter of taste. Measured here:

  Q1 How persistent is bucket delivery %? If the 21-day autocorrelation is high,
     a 21-day baseline is largely subtracting the signal from itself.
  Q2 Do the two baselines actually disagree? Correlation, sign flips, and whether
     the 21d version is DEGENERATE (compressed toward zero).
  Q3 Noise cost. A 21-sample sd has ~16% relative error against ~7% at 100, so a
     short window produces more extreme z's from estimation error alone.
  Q4 The special sessions now in the data (Budget days, Muhurat) are 6 sessions.
     They are 1/200th of a 100-day window but up to 1/21st of a short one, and
     Muhurat ran 64% delivery against a ~53% norm. How far does one such session
     move a 21-day baseline?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

from src.analytics.index_largecap import INDEX_BUCKETS
from src.data.repository import query_dataframe

pd.set_option("display.width", 240)
B = INDEX_BUCKETS["NIFTY"]["buckets"]
ALL50 = tuple(s for v in B.values() for s in v)
ph = ",".join("?" * len(ALL50))

df = query_dataframe(f"""
    SELECT trade_date, symbol, deliv_per
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND deliv_per IS NOT NULL AND trade_date >= '2021-06-01'
""", list(ALL50))
df["trade_date"] = pd.to_datetime(df["trade_date"])
DL = df.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
print(f"panel {DL.shape[0]} sessions x {DL.shape[1]} symbols")


def persym_z(M, cols, base):
    S = M[[s for s in cols if s in M.columns]]
    mp = max(10, base // 2)
    mu = S.rolling(base, min_periods=mp).mean().shift(1)
    sd = S.rolling(base, min_periods=mp).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


print("\n" + "=" * 96)
print("Q1  PERSISTENCE — how autocorrelated is bucket delivery %?")
print("=" * 96)
for bn, mem in B.items():
    v = DL[[s for s in mem if s in DL.columns]].mean(axis=1).dropna()
    acs = [v.autocorr(l) for l in (1, 5, 10, 21, 42, 100)]
    print(f"  {bn:8s} lag1 {acs[0]:+.3f}  lag5 {acs[1]:+.3f}  lag10 {acs[2]:+.3f}  "
          f"lag21 {acs[3]:+.3f}  lag42 {acs[4]:+.3f}  lag100 {acs[5]:+.3f}")
print("  A high lag-21 number means a 21-session mean already contains the move.")

print("\n" + "=" * 96)
print("Q2  DO THEY DISAGREE? 21-session vs 100-session baseline")
print("=" * 96)
for bn, mem in B.items():
    cols = [s for s in mem if s in DL.columns]
    z21, z100 = persym_z(DL, cols, 21), persym_z(DL, cols, 100)
    j = pd.DataFrame({"z21": z21, "z100": z100}).dropna()
    flip = (np.sign(j.z21) != np.sign(j.z100)).mean() * 100
    # how often does each cross the +-0.3 band the panel uses for heavy/light?
    fire21 = (j.z21.abs() > 0.3).mean() * 100
    fire100 = (j.z100.abs() > 0.3).mean() * 100
    print(f"  {bn:8s} n={len(j):5d}  corr {j.z21.corr(j.z100):+.3f}  "
          f"sign flips {flip:4.1f}%   sd21 {j.z21.std():.2f} vs sd100 {j.z100.std():.2f}   "
          f"|z|>0.3 fires: 21d {fire21:4.1f}%  100d {fire100:4.1f}%")

print("\n" + "=" * 96)
print("Q3  IS THE SHORT WINDOW DEGENERATE? z magnitude when delivery is trending")
print("=" * 96)
print("  Take sessions where the bucket's delivery has risen for 5 straight days")
print("  -- the case where a chasing baseline would erase the signal.")
for bn, mem in B.items():
    cols = [s for s in mem if s in DL.columns]
    lvl = DL[cols].mean(axis=1)
    rising = (lvl.diff() > 0).rolling(5).sum() == 5
    z21, z100 = persym_z(DL, cols, 21), persym_z(DL, cols, 100)
    j = pd.DataFrame({"z21": z21, "z100": z100, "rise": rising}).dropna()
    r = j[j["rise"]]
    if len(r) < 20:
        continue
    print(f"  {bn:8s} {len(r):4d} such sessions   mean z21 {r.z21.mean():+.2f}   "
          f"mean z100 {r.z100.mean():+.2f}   "
          f"ratio {abs(r.z21.mean()) / max(abs(r.z100.mean()), 1e-9):.2f}x")
print("  ratio < 1 means the SHORT window is muting a real, sustained move.")

print("\n" + "=" * 96)
print("Q4  SPECIAL SESSIONS inside a short window")
print("=" * 96)
specials = ["2020-02-01", "2022-08-08", "2023-11-12", "2024-03-02",
            "2025-02-01", "2026-02-01"]
for bn, mem in B.items():
    cols = [s for s in mem if s in DL.columns]
    lvl = DL[cols].mean(axis=1)
    out = []
    for sd in specials:
        ts = pd.Timestamp(sd)
        if ts not in lvl.index:
            continue
        prior = lvl[lvl.index < ts].tail(21)
        if len(prior) < 10:
            continue
        with21 = pd.concat([prior.iloc[1:], pd.Series([lvl[ts]], index=[ts])])
        out.append(abs(with21.mean() - prior.mean()))
    if out:
        print(f"  {bn:8s} one special session shifts a 21-day baseline by "
              f"{np.mean(out):.3f}pp on average, worst {max(out):.3f}pp")
print("  (a 100-day window dilutes the same session by roughly 5x)")
