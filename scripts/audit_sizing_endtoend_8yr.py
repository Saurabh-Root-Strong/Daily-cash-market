"""
END-TO-END SIZING AUDIT (10x sharp) — walks the REAL shipped _market_regime over 8yr Nifty
and tests whether the composite size_hint (regime + confirmation-gate + trend-quality nudge +
downtrend cut + divergence overlays) actually improves risk-adjusted outcomes vs flat/constant.

Faithful: calls src.analytics.sector_forward_tilt._market_regime(nifty[:D]) per day (it needs
only the index series). Backtests a long-only book that scales next-day Nifty exposure by
today's size_hint (the BETA-timing component; relative alpha isn't in Nifty but the risk-
management claim is). Compares vs flat(1.0) and vs constant(mean size) to separate TIMING
from just-being-smaller. Also: forward-return monotonicity by size bucket; and the crux —
does the downtrend size-cut help or hurt (downtrends mark bottoms).
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q
from src.analytics.sector_forward_tilt import _market_regime

def sharpe(r): return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
def maxdd(equity):
    peak=equity.cummax(); return ((equity-peak)/peak).min()*100

if __name__=="__main__":
    n=q("SELECT trade_date, close_val, pct_chg FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"]=pd.to_datetime(n["trade_date"]); n["nret"]=n["pct_chg"].astype(float)
    n=n.reset_index(drop=True)
    print(f"walking real _market_regime over {len(n)} days ({n.trade_date.iloc[0].date()}->{n.trade_date.iloc[-1].date()})...")
    rows=[]
    for D in range(200, len(n)):
        sub=n.iloc[:D+1][["trade_date","close_val","nret"]]
        reg=_market_regime(sub)
        rows.append((n.trade_date.iloc[D], reg["state"], reg["verdict"], reg["size_hint"],
                     reg.get("trend_strength"), reg.get("momentum_inverts")))
    R=pd.DataFrame(rows, columns=["date","state","verdict","size","tq","inv"]).set_index("date")
    R["nret"]=n.set_index("trade_date")["nret"].reindex(R.index)
    R["fwd1"]=R["nret"].shift(-1)                    # next-day return (exposure set today, earned tomorrow)
    R=R.dropna(subset=["fwd1"])

    print(f"\nsize distribution: {R['size'].value_counts().sort_index().to_dict()}")
    print(f"mean size {R['size'].mean():.2f} | verdict mix {R['verdict'].value_counts().to_dict()}")

    print("\n"+"="*88); print("1) STRATEGY COMPARISON — long-only Nifty scaled by size_hint vs baselines"); print("="*88)
    strat={
        "FLAT 1.0 (always full)":        pd.Series(1.0, index=R.index),
        "CONSTANT mean-size":           pd.Series(R["size"].mean(), index=R.index),
        "SIZE-SCALED (shipped logic)":  R["size"],
    }
    print(f"  {'strategy':30s} {'ann.ret':>8s} {'ann.vol':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'return/DD':>9s}")
    for name,w in strat.items():
        r=(w*R["fwd1"]/100)                          # daily strat return (fraction)
        eq=(1+r).cumprod()
        ann=(eq.iloc[-1]**(252/len(r))-1)*100
        print(f"  {name:30s} {ann:+7.1f}% {r.std()*np.sqrt(252)*100:7.1f}% {sharpe(r):+6.2f} {maxdd(eq):+6.1f}% {ann/abs(maxdd(eq)):8.2f}")

    print("\n"+"="*88); print("2) MONOTONICITY — next-day Nifty return by size bucket (should rise with size)"); print("="*88)
    R["sb"]=pd.cut(R["size"], [0,0.35,0.55,0.85,1.01], labels=["0-.35","≤.55","≤.85","≤1.0"])
    for b,g in R.groupby("sb", observed=True):
        print(f"  size {b:>6s}: fwd1 {g['fwd1'].mean():+.3f}%/day  ann {g['fwd1'].mean()*252:+5.1f}%  "
              f"hit {(g['fwd1']>0).mean()*100:.0f}%  n{len(g)}")

    print("\n"+"="*88); print("3) CRUX — the DOWNTREND size-cut: does reducing in DOWN help or hurt?"); print("="*88)
    dn=R[R["state"]=="TRENDING_DOWN"]
    print(f"  DOWN days n{len(dn)} | shipped avg size {dn['size'].mean():.2f}")
    print(f"  next-day Nifty on DOWN days: {dn['fwd1'].mean():+.3f}%/day (ann {dn['fwd1'].mean()*252:+.1f}%)")
    print(f"  => if POSITIVE, cutting size in DOWN gives up return (downtrends bounce); if NEGATIVE, the cut avoids losses.")
    # what if we DIDN'T cut in downtrends (size=1 there instead of 0.4)?
    alt=R["size"].copy(); alt[R["state"]=="TRENDING_DOWN"]=1.0
    for name,w in [("shipped (cut DOWN to ~0.4)",R["size"]),("no-cut (DOWN=1.0)",alt)]:
        r=(w*R["fwd1"]/100); eq=(1+r).cumprod(); ann=(eq.iloc[-1]**(252/len(r))-1)*100
        print(f"    {name:28s} ann {ann:+.1f}%  Sharpe {sharpe(r):+.2f}  maxDD {maxdd(eq):+.1f}%")

    print("\n"+"="*88); print("4) CONFIRMATION-GATE value — does debounced regime capture forward better than raw?"); print("="*88)
    # compare verdict stability: how often does shipped size change day-to-day?
    flips=(R["size"].round(2)!=R["size"].round(2).shift()).sum()
    print(f"  shipped size changes {flips} times over {len(R)} days = {flips/len(R)*100:.0f}% of days (lower=stable, good)")
    print(f"  (pre-gate raw regime flipped ~every 3d = ~33%/days; confirmation-gate should be well below)")
