"""
ADJUDICATE the downtrend overweight-suppression, now that real bears are loaded.

The conflict: DCM broad sectors show OW-UW +2.26% in DOWN regime (momentum HOLDS),
Tradebot F&O sectors show -0.97% (mild invert). Concern: DCM's positive number may be
the 2020/2022 RECOVERY leg sitting inside the lagging EMA-DOWN label (a day is 'DOWN' but
the next 10d actually bounce, so momentum leaders lead the bounce).

Clean split (within DOWN regime): condition on what the market ACTUALLY did next.
  DOWN & fwd-Nifty < 0  = sustained fall  -> the case the suppression is meant to protect
  DOWN & fwd-Nifty >= 0 = bounce/recovery -> where momentum legitimately resumes
If OW-UW is NEGATIVE in sustained-fall but POSITIVE in bounce, the suppression is RIGHT in
spirit and the pooled +2.26% is a recovery artifact. If OW-UW is POSITIVE even in
sustained falls, the suppression is WRONG and should be lifted for the sector product.
(fwd-conditioning is lookahead — used only to UNDERSTAND, never as a live signal.)

Also reports ABSOLUTE OW return (long-only P&L) per split, and a purely-causal
'active-crash' split (trailing Nifty 10d deeply negative) as a live-usable cross-check.
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scripts.audit_tilt_realbears import sector_panel, nifty

def build():
    P = sector_panel(); N = nifty()
    ret = P.pivot_table("wret","trade_date","sector").sort_index().clip(-15,15)
    cr = np.log1p(ret/100).cumsum()
    mom = np.expm1(cr - cr.shift(10))*100
    fwd = np.expm1(cr.shift(-10) - cr)*100                       # sector ABSOLUTE fwd10
    rs = mom.sub(mom.median(axis=1), axis=0)
    relfwd = fwd.sub(fwd.median(axis=1), axis=0)                 # rel-to-median-sector
    rank = rs.rank(axis=1, pct=True)
    nc = N.set_index("trade_date")["close_val"].astype(float).reindex(ret.index)
    nret = N.set_index("trade_date")["pct_chg"].astype(float).reindex(ret.index)
    ncr = np.log1p(nret/100).cumsum()
    nfwd = np.expm1(ncr.shift(-10) - ncr)*100                    # Nifty fwd10 (for the split)
    ntrail = np.expm1(ncr - ncr.shift(10))*100                   # Nifty trailing 10d (causal)
    e20 = nc.ewm(span=20,adjust=False).mean(); e50 = nc.ewm(span=50,adjust=False).mean()
    reg = pd.Series("CHOP", index=nc.index)
    reg[(nc>e20)&(e20>e50)] = "UP"; reg[(nc<e20)&(e20<e50)] = "DOWN"
    return dict(rank=rank, relfwd=relfwd, absfwd=fwd, reg=reg, nfwd=nfwd, ntrail=ntrail)

def owuw_stats(rank, relfwd, absfwd, mask):
    r = rank.where(mask); rf = relfwd.where(mask); af = absfwd.where(mask)
    ow = rf.where(r>=0.75); uw = rf.where(r<=0.25)
    owa = af.where(r>=0.75)
    d_ow = ow.mean(axis=1).dropna(); d_uw = uw.mean(axis=1).dropna()
    sp = (d_ow - d_uw).dropna()
    t = sp.iloc[::10].mean()/(sp.iloc[::10].std()/np.sqrt(len(sp.iloc[::10]))) if len(sp)>=40 else np.nan
    return dict(owuw=sp.mean(), t=t, ow_abs=owa.mean(axis=1).mean(),
                ndays=int(mask.any(axis=1).sum()) if hasattr(mask,'any') else int(mask.sum()))

if __name__ == "__main__":
    B = build(); rank,relfwd,absfwd,reg,nfwd,ntrail = (B[k] for k in ["rank","relfwd","absfwd","reg","nfwd","ntrail"])
    down = (reg=="DOWN")
    # broadcast day-level conditions to the (date x sector) mask
    dmask = pd.DataFrame(np.repeat(down.values[:,None], rank.shape[1], axis=1), index=rank.index, columns=rank.columns)
    fall = pd.DataFrame(np.repeat((down & (nfwd<0)).values[:,None], rank.shape[1], axis=1), index=rank.index, columns=rank.columns)
    bounce = pd.DataFrame(np.repeat((down & (nfwd>=0)).values[:,None], rank.shape[1], axis=1), index=rank.index, columns=rank.columns)
    crash = pd.DataFrame(np.repeat((down & (ntrail<=-5)).values[:,None], rank.shape[1], axis=1), index=rank.index, columns=rank.columns)

    print(f"DCM sector panel | DOWN calendar days {int(down.sum())} "
          f"(sustained-fall {int((down&(nfwd<0)).sum())} / bounce {int((down&(nfwd>=0)).sum())})\n")
    print("="*92)
    print("ADJUDICATION — within DOWN regime, conditioned on what the market ACTUALLY did next")
    print("  OW-UW = momentum spread (rel-to-median-sector, fwd10). OW abs = long-only leader P&L.")
    print("="*92)
    print(f"  {'sub-case':34s} {'OW-UW':>8s} {'t':>6s} {'OW abs':>8s} {'days':>6s}")
    for lab, m in [("DOWN — all", dmask),
                   ("DOWN & market kept FALLING (fwd<0)", fall),
                   ("DOWN & market BOUNCED (fwd>=0)", bounce),
                   ("DOWN & active crash (trail 10d<=-5%)", crash)]:
        s = owuw_stats(rank, relfwd, absfwd, m)
        tt = f"{s['t']:+.1f}" if np.isfinite(s['t']) else "  . "
        print(f"  {lab:34s} {s['owuw']:+7.2f}% {tt:>6s} {s['ow_abs']:+7.2f}% {s['ndays']:6d}")

    print("\n" + "="*92)
    print("VERDICT LOGIC: if OW-UW is NEGATIVE in 'kept falling' but positive pooled -> the")
    print("suppression is right in spirit (protects real sustained falls); the pooled + is a")
    print("recovery artifact. If OW-UW is POSITIVE even in 'kept falling' -> suppression wrong.")
    print("="*92)
