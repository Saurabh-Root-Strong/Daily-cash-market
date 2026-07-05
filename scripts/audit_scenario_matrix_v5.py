"""
SCENARIO MATRIX v5 — the COMPLETE daily×weekly price-action state space, one map.

Every prior audit tested one slice. This enumerates the whole finite grid and
measures forward-RELATIVE return per cell, both panels, honest non-overlapping t,
so we have a single definitive scenario table.

State axes
  WEEKLY backdrop : W-Up / W-Down            (EMA20/50 of daily close — shipped read)
  DAILY action    : D-Up / D-Range / D-Down  (Kaufman ER10<0.25 = Range/chop, else sign r10)
  COMPRESSION     : tight / loose            (bottom-third cross-sectional ATR20)
  EVENT           : range-breakout (break 20d-hi FROM tight) / trend-breakout (break, loose)
                    / breakdown (break 20d-lo) / coiling (tight, no break) / none

Named scenarios mapped to cells
  breakout           = range-breakout (🚀, the shipped edge) + trend-breakout (↗)
  uptrend            = W-Up  & D-Up
  downtrend          = W-Down & D-Down
  consolidation range= tight & no break (🧊 Coiling)
  range breakout     = range-breakout (tight-base 20d-hi break)
  false breakout     = range-breakout & W-Down (⚠️ shipped)
  pullback / false-pop = W-Up&D-Down / W-Down&D-Up

Dual-panel (DCM broad + 4yr F&O OOS). Reuses v2 loaders/build/sample.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market/scripts")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from audit_rotation_filters_v2 import load_dcm, load_fno, build, sample

ER10_RANGE = 0.25

def attach(S, mat):
    vals = []
    for d, g in S.groupby("date"):
        row = mat.loc[d] if d in mat.index else pd.Series(dtype=float)
        vals.append(g["symbol"].map(row))
    return pd.concat(vals).reindex(S.index)

def cell(S, mask, h):
    n = int(mask.sum())
    if n < 25: return (n, None, None, None)
    g = S[mask].groupby("date")[f"f{h}"].mean().dropna()
    if len(g) < 6: return (n, None, None, None)
    gn = g.iloc[::max(1, h // 5)]
    t = gn.mean() / (gn.std() / np.sqrt(len(gn))) if len(gn) >= 5 else np.nan
    win = (S[mask][f"f{h}"] > 0).mean() * 100
    return (n, g.mean() * 100, t, win)

def row(SA, SB, maskA, maskB, label, h=10):
    a = cell(SA, maskA, h); b = cell(SB, maskB, h)
    def fmt(c):
        n, m, t, w = c
        if m is None: return f"n={n:<5d} thin       "
        return f"n={n:<5d} {m:+5.2f}% t{t:+4.1f} w{w:2.0f}"
    print(f"  {label:34s} | DCM {fmt(a)} | F&O {fmt(b)}")

def classify(S, X):
    S = S.copy()
    C = X["C"]
    er10 = (C - C.shift(10)).abs() / C.diff().abs().rolling(10).sum().replace(0, np.nan)
    S["er10"] = attach(S, er10)
    # daily action state
    S["dstate"] = np.where(S["er10"] < ER10_RANGE, "Range",
                    np.where(S["d_dir"] > 0, "Up", np.where(S["d_dir"] < 0, "Down", "Range")))
    S["wstate"] = np.where(S["w_dir"] > 0, "WUp", np.where(S["w_dir"] < 0, "WDown", "WFlat"))
    return S

def panel_pair(SA, SB):
    print("\n" + "="*104)
    print("TABLE 1 — WEEKLY × DAILY backdrop grid (no event filter): fwd-10d relative")
    print("="*104)
    for w in ["WUp", "WDown"]:
        for d in ["Up", "Range", "Down"]:
            mA = (SA["wstate"] == w) & (SA["dstate"] == d)
            mB = (SB["wstate"] == w) & (SB["dstate"] == d)
            row(SA, SB, mA, mB, f"{w:5s} × D-{d}")
        print("  " + "-"*100)

    print("\n" + "="*104)
    print("TABLE 2 — EVENT overlay by weekly backdrop: fwd-10d relative")
    print("="*104)
    events = {
        "range-breakout 🚀 (tight-base hi)": lambda S: S["brk"] == "Breakout",
        "trend-breakout ↗ (loose hi)":       lambda S: S["brk"] == "Extended",
        "breakdown 💥 (20d-lo break)":        lambda S: S["brk"] == "Bounce",
        "consolidation 🧊 (tight, no break)": lambda S: S["brk"] == "Coiling",
        "no-event —":                         lambda S: S["brk"] == "None",
    }
    for name, fn in events.items():
        for w in ["WUp", "WDown"]:
            mA = fn(SA) & (SA["wstate"] == w)
            mB = fn(SB) & (SB["wstate"] == w)
            row(SA, SB, mA, mB, f"{name:32s} | {w}")
        print("  " + "-"*100)

    print("\n" + "="*104)
    print("TABLE 3 — the NAMED scenarios head-to-head: fwd-10d AND fwd-20d relative")
    print("="*104)
    named = {
        "UPTREND (WUp×DUp)":            lambda S: (S["wstate"]=="WUp") & (S["dstate"]=="Up"),
        "DOWNTREND (WDn×DDn)":          lambda S: (S["wstate"]=="WDown") & (S["dstate"]=="Down"),
        "CONSOL range (tight coil)":    lambda S: S["brk"]=="Coiling",
        "RANGE-BREAKOUT 🚀 (WUp)":       lambda S: (S["brk"]=="Breakout") & (S["wstate"]=="WUp"),
        "FALSE break (WDn)":            lambda S: (S["brk"]=="Breakout") & (S["wstate"]=="WDown"),
        "TREND-breakout ↗ (loose)":      lambda S: S["brk"]=="Extended",
        "PULLBACK (WUp×DDn)":           lambda S: (S["wstate"]=="WUp") & (S["dstate"]=="Down"),
        "FALSE-POP (WDn×DUp)":          lambda S: (S["wstate"]=="WDown") & (S["dstate"]=="Up"),
        "BREAKDOWN bounce 💥":           lambda S: S["brk"]=="Bounce",
    }
    for name, fn in named.items():
        for h in [10, 20]:
            row(SA, SB, fn(SA), fn(SB), f"{name}" + (f"  [{h}d]"), h)
        print("  " + "-"*100)

def base_rates(S, tag):
    n = len(S)
    print(f"\n{tag} base rates (share of stock-days):")
    print("  weekly:", (S["wstate"].value_counts()/n*100).round(1).to_dict())
    print("  daily :", (S["dstate"].value_counts()/n*100).round(1).to_dict())
    print("  event :", (S["brk"].value_counts()/n*100).round(1).to_dict())

if __name__ == "__main__":
    CA, HA, LA, OA, TA = load_dcm()
    XA = build(CA, HA, LA, OA, TA); SA = classify(sample(XA), XA)
    CB, HB, LB, OB, TB = load_fno()
    XB = build(CB, HB, LB, OB, TB); SB = classify(sample(XB), XB)
    print(f"\nPANEL A DCM broad: {len(SA):,} stock-days | PANEL B 4yr F&O: {len(SB):,}")
    base_rates(SA, "DCM"); base_rates(SB, "F&O")
    panel_pair(SA, SB)
