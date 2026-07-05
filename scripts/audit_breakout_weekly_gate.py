"""
BREAKOUT x WEEKLY-TREND GATE — the "daily breakout but weekly downtrend" case.

User scenario: a stock breaks the 20d high on the DAILY chart, but the WEEKLY
trend is DOWN. Textbook "false breakout". Currently the 🚀 Breakout state ignores
weekly direction (pure Donchian-break-from-tight-base). Question: should Breakout
be GATED by weekly-up, or does weekly-down break still carry (or invert)?

Splits the Breakout basket by weekly trend and measures forward-RELATIVE returns:
  - 3 weekly reads: EMA20/50 proxy (shipped), true weekly close>SMA10w, weekly HH/HL.
  - horizons 5/10/15/20d, non-overlapping honest t, BOTH panels.
  - decay curves (does weekly-down break fade / invert over 2-3 weeks).
  - regime split, net-of-cost.
  - the actionable spread: Breakout|weekly-up  minus  Breakout|weekly-down.
Reuses loaders/build from audit_rotation_filters_v2 (import-safe).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market/scripts")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from audit_rotation_filters_v2 import (load_dcm, load_fno, build, sample, bt, show,
                                        HS, COST, LOOK, DON)

# ---- extra weekly-trend reads, added onto the sample frame -------------------
def add_weekly_reads(X, S):
    """Attach true-weekly-candle trend flags per (date,symbol) row of S."""
    C = X["C"]
    # true weekly candles: resample W-FRI on close/high/low
    wk_c = C.resample("W-FRI").last()
    wk_h = X.get("_H", C).resample("W-FRI").max() if "_H" in X else None
    # weekly SMA10 (~10 weeks ~ 50 trading days)
    w_sma = wk_c.rolling(10, min_periods=5).mean()
    w_close_gt_sma = (wk_c > w_sma)                       # weekly close above 10w SMA
    # weekly higher-high & higher-low structure (2-week)
    w_hh = (wk_c > wk_c.shift(1)) & (wk_c.shift(1) > wk_c.shift(2))
    w_ll = (wk_c < wk_c.shift(1)) & (wk_c.shift(1) < wk_c.shift(2))
    # map weekly flag back to each daily date via as-of (last completed week <= date)
    def asof_daily(wframe):
        # reindex weekly frame onto daily index, forward-fill last known week
        return wframe.reindex(C.index, method="ffill")
    d_sma = asof_daily(w_close_gt_sma)
    d_hh  = asof_daily(w_hh); d_ll = asof_daily(w_ll)
    def pull(dframe, name):
        vals = []
        for d, g in S.groupby("date"):
            row = dframe.loc[d] if d in dframe.index else pd.Series(dtype=float)
            vals.append(g["symbol"].map(row))
        return pd.concat(vals).reindex(S.index)
    S = S.copy()
    S["w_sma_up"] = pull(d_sma, "sma").fillna(False).astype(bool)
    S["w_struct_up"] = pull(d_hh, "hh").fillna(False).astype(bool)
    S["w_struct_dn"] = pull(d_ll, "ll").fillna(False).astype(bool)
    return S

def spread(S, base, up, dn, h, label):
    ru = bt(S, base & up, h); rd = bt(S, base & dn, h)
    if not ru or not rd:
        print(f"  {label:34s} f{h}: thin (up n={int((base&up).sum())}, dn n={int((base&dn).sum())})"); return
    print(f"  {label:34s} f{h:<2d}:  UP {ru[1]:+.2f}% (t {ru[3]:+.1f} win {ru[4]:.0f} n{ru[0]})"
          f"   DN {rd[1]:+.2f}% (t {rd[3]:+.1f} win {rd[4]:.0f} n{rd[0]})"
          f"   SPREAD {ru[1]-rd[1]:+.2f}pp")

def panel(name, C, H, L, O, TV):
    print("\n" + "#"*104); print(f"# {name}: {C.shape[0]}d x {C.shape[1]} syms  {C.index.min().date()}..{C.index.max().date()}"); print("#"*104)
    X = build(C, H, L, O, TV); X["_H"] = H
    S = sample(X)
    S = add_weekly_reads(X, S)
    bo = S["brk"] == "Breakout"
    wu = S["w_dir"] > 0; wd = S["w_dir"] < 0; wf = S["w_dir"] == 0
    print(f"pooled {len(S):,}  breakouts n={int(bo.sum())}  of which weekly-up {int((bo&wu).sum())}"
          f" / flat {int((bo&wf).sum())} / down {int((bo&wd).sum())}")

    print("\n1) BREAKOUT split by weekly trend (EMA20/50 proxy = shipped read)")
    for h in HS:
        spread(S, bo, wu, wd, h, "Breakout up-vs-DOWN weekly")
    print("   weekly-FLAT breakouts (no weekly signal):")
    for h in [10, 20]:
        show(S, bo & wf, "Breakout | weekly-flat", h)

    print("\n2) same split — TRUE weekly candle reads (does resampling agree?)")
    for h in [10, 20]:
        spread(S, bo, S["w_sma_up"], ~S["w_sma_up"], h, "Breakout | wk close>10wSMA")
    for h in [10, 20]:
        spread(S, bo, S["w_struct_up"], S["w_struct_dn"], h, "Breakout | wk HH-HL vs LL-LH")

    print("\n3) is a weekly-DOWN breakout an actual FALSE break (neg / worse than universe)?")
    for h in HS:
        show(S, bo & wd, "Breakout + weekly-DOWN", h)

    print("\n4) REGIME split of weekly-down breakout (f10/f20)")
    for rg in ["BULL", "CHOP", "BEAR"]:
        rm = S["regime"] == rg
        for h in [10, 20]:
            show(S, bo & wd & rm, f"Brk+wk-DOWN | {rg}", h)

    print("\n5) NET of cost — weekly-up vs weekly-down breakout")
    for h in [10, 20]:
        show(S, bo & wu, f"Brk+wk-UP  net{COST*100:.1f}%", h, cost=COST)
        show(S, bo & wd, f"Brk+wk-DN  net{COST*100:.1f}%", h, cost=COST)
    return S

if __name__ == "__main__":
    CA, HA, LA, OA, TA = load_dcm()
    panel("PANEL A — DCM broad", CA, HA, LA, OA, TA)
    CB, HB, LB, OB, TB = load_fno()
    panel("PANEL B — 4yr F&O OOS", CB, HB, LB, OB, TB)
