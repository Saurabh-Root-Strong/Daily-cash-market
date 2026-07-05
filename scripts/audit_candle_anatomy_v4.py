"""
CANDLE ANATOMY v4 — the two gaps v3 left open: WEEKLY patterns + BREAKOUT-bar anatomy.

v3 killed DAILY candlestick patterns (marubozu/hammer/engulfing: win 43-52%, sign
flips across panels) and the alignment horse-race (EMA proxy > true weekly candles).
NOT yet tested:
  A. WEEKLY candlestick patterns — weekly marubozu / hammer / shooting-star / engulfing
     / close-in-range. Weekly bars are slower & less noisy; maybe patterns survive there.
  B. The BREAKOUT candle's OWN anatomy — v3 did range-expansion & volume of the break
     bar (fade), but NOT body% / close-position-in-range / upper-wick rejection. Does a
     clean marubozu breakout follow through better than a wick-heavy / doji breakout?
  C. DAILY×WEEKLY candle CONFLUENCE — weekly strong body + daily breakout.

Dual-panel (DCM broad + 4yr F&O OOS), forward-RELATIVE to universe median, honest
non-overlapping t (step=horizon). Reuses v2 loaders/build.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market/scripts")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from audit_rotation_filters_v2 import load_dcm, load_fno, build, sample, HS, COST, DON

def relret(C, h):
    r = C.shift(-h) / C - 1.0
    return r.sub(r.median(axis=1), axis=0)

def nonov_t(series_by_date, h):
    """series_by_date: DataFrame date-indexed of per-date mean rel-ret. Take every h-th."""
    g = series_by_date.dropna()
    if len(g) < 6: return None
    gn = g.iloc[::max(1, h // 5)]
    t = gn.mean() / (gn.std() / np.sqrt(len(gn))) if len(gn) >= 5 else np.nan
    return len(g), g.mean() * 100, t

def eventcol(C, flag, h, step):
    """mean fwd-rel return of flagged cells, per sampled date, non-overlapping t."""
    fr = relret(C, h)
    dates = C.index[75:len(C.index) - h - 1:step]
    per = []
    n_ev = 0
    for d in dates:
        m = flag.loc[d]
        if m.sum() < 3: continue
        v = fr.loc[d][m.values]
        per.append(v.mean()); n_ev += int(m.sum())
    if len(per) < 6: return None
    per = pd.Series(per)
    t = per.mean() / (per.std() / np.sqrt(len(per)))
    win = None
    return n_ev, per.mean() * 100, t, len(per)

def show_ev(C, flag, label, h, step=None):
    step = step or h
    r = eventcol(C, flag, h, step)
    if r is None: print(f"    {label:40s} f{h}: thin"); return
    n_ev, m, t, nd = r
    print(f"    {label:40s} f{h:<2d}: events={n_ev:6d} dates={nd:4d}  {m:+.2f}%  t{t:+.1f}")

# ---------- WEEKLY candle features ----------
def weekly_candles(C, H, L, O):
    wc = C.resample("W-FRI").last(); wo = O.resample("W-FRI").first()
    wh = H.resample("W-FRI").max(); wl = L.resample("W-FRI").min()
    rng = (wh - wl).replace(0, np.nan)
    body = (wc - wo)
    body_pct = body.abs() / rng
    up_wick = (wh - wc.where(wc >= wo, wo)) / rng
    lo_wick = (wc.where(wc <= wo, wo) - wl) / rng
    close_pos = (wc - wl) / rng                    # 1 = close at high
    # patterns (cross-sectional body cut per week to control the tiny abs spread)
    marubozu_up = (body > 0) & (body_pct >= 0.80)  # strong green, tiny wicks
    marubozu_dn = (body < 0) & (body_pct >= 0.80)
    hammer = (body_pct <= 0.40) & (lo_wick >= 0.5) & (up_wick <= 0.15)   # long lower wick
    shoot  = (body_pct <= 0.40) & (up_wick >= 0.5) & (lo_wick <= 0.15)   # long upper wick
    engulf_up = (body > 0) & (body.abs() > body.shift(1).abs()) & (wc > wo.shift(1)) & (wo < wc.shift(1)) & (body.shift(1) < 0)
    def asof(w): return w.reindex(C.index, method="ffill")
    return {k: asof(v) for k, v in dict(
        maru_up=marubozu_up, maru_dn=marubozu_dn, hammer=hammer, shoot=shoot,
        engulf_up=engulf_up, body_pct=body_pct, close_pos=close_pos).items()}

def panel(name, C, H, L, O, TV):
    print("\n" + "#"*100); print(f"# {name}: {C.shape[0]}d x {C.shape[1]} syms"); print("#"*100)
    rng = (H - L).replace(0, np.nan)

    # ---- A. WEEKLY candlestick patterns ----
    print("\nA) WEEKLY candle PATTERNS → fwd-rel (do slower bars rescue what daily can't?)")
    W = weekly_candles(C, H, L, O)
    for h in [10, 20]:
        show_ev(C, W["maru_up"], "weekly Marubozu-UP (body>=80%)", h)
        show_ev(C, W["maru_dn"], "weekly Marubozu-DOWN", h)
        show_ev(C, W["hammer"], "weekly Hammer (long lower wick)", h)
        show_ev(C, W["shoot"],  "weekly Shooting-star (upper wick)", h)
        show_ev(C, W["engulf_up"], "weekly Bullish-engulfing", h)
        print("    " + "-"*70)

    # ---- B. BREAKOUT candle anatomy ----
    print("\nB) BREAKOUT-BAR ANATOMY — split 🚀 Breakout by the break candle's own shape")
    X = build(C, H, L, O, TV); S = sample(X)
    # attach the break-bar anatomy of the AS-OF bar to each sampled row
    body_pct_d = (C - O).abs() / rng
    close_pos_d = (C - L) / rng
    up_wick_d = (H - C.where(C >= O, O)) / rng
    def attach(mat, nm):
        vals = []
        for d, g in S.groupby("date"):
            row = mat.loc[d] if d in mat.index else pd.Series(dtype=float)
            vals.append(g["symbol"].map(row))
        return pd.concat(vals).reindex(S.index)
    S = S.copy()
    S["b_body"] = attach(body_pct_d, "body")
    S["b_close"] = attach(close_pos_d, "close")
    S["b_upwick"] = attach(up_wick_d, "upwick")
    # shipped breakout with weekly gate
    bo = (S["brk"] == "Breakout") & (S["w_dir"] > 0)
    print(f"    breakouts n={int(bo.sum())}  median break-bar body% {S.loc[bo,'b_body'].median():.2f}")
    def bkt(mask, label, h, cost=0.0):
        m = bo & mask
        if m.sum() < 25: print(f"    {label:40s} f{h}: thin (n={int(m.sum())})"); return
        g = S[m].groupby("date")[f"f{h}"].mean().dropna() - cost
        t = g.mean() / (g.std() / np.sqrt(len(g))) if len(g) >= 6 else np.nan
        w = (S[m][f"f{h}"] > 0).mean() * 100
        print(f"    {label:40s} f{h:<2d}: n={int(m.sum()):5d}  {g.mean()*100:+.2f}%  t{t:+.1f}  win {w:.0f}%")
    for h in [10, 20]:
        thr_b = S.loc[bo, "b_body"].median()
        bkt(S["b_body"] >= thr_b, "strong-body break (marubozu-like)", h)
        bkt(S["b_body"] <  thr_b, "weak-body break (doji/wick-y)", h)
        bkt(S["b_close"] >= 0.75, "close in top-25% of bar (no rejection)", h)
        bkt(S["b_close"] <  0.75, "close mid/low of bar (upper rejection)", h)
        bkt(S["b_upwick"] >= 0.35, "big upper wick (intraday fade)", h)
        print("    " + "-"*70)

    # ---- C. Daily×Weekly candle confluence ----
    print("\nC) DAILY×WEEKLY CONFLUENCE — breakout + weekly strong body / close-high")
    S["w_body"] = attach(W["body_pct"], "wbody")
    S["w_closepos"] = attach(W["close_pos"], "wclose")
    for h in [10, 20]:
        bkt(S["w_body"] >= 0.6, "breakout + weekly strong body (>=0.6)", h)
        bkt(S["w_closepos"] >= 0.7, "breakout + weekly close near high", h)
        print("    " + "-"*70)

if __name__ == "__main__":
    CA, HA, LA, OA, TA = load_dcm()
    panel("PANEL A — DCM broad", CA, HA, LA, OA, TA)
    CB, HB, LB, OB, TB = load_fno()
    panel("PANEL B — 4yr F&O OOS", CB, HB, LB, OB, TB)
