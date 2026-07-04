"""
Follow-up: is a single "efficient movers" toggle (ER>=0.25 = Clean∪Volatile) robust
enough to ship at the 2-3wk horizon? Halves, regimes, within-ConfirmUp, both panels.
Also quantify the veto x breakout interaction cleanly (win-rate vs mean trade-off).
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v2", r"d:/Python Projects/Daily_Cash_Market/scripts/audit_rotation_filters_v2.py")
# import only the helpers by executing up to run_panel — simpler: copy the small bits
from scripts.audit_rotation_filters_v2 import load_dcm, load_fno, build, sample, bt  # type: ignore

def show(S, mask, label, h):
    r = bt(S, mask, h)
    if r is None: print(f"  {label:44s} f{h}: thin"); return
    n, m, to, tn, w = r
    print(f"  {label:44s} f{h:<2d}: n={n:6d}  {m:+.2f}%  t_ov {to:+.1f}  t_nonov {tn:+.1f}  win {w:4.1f}%")

for name, loader in [("A — DCM broad", load_dcm), ("B — 4yr F&O OOS", load_fno)]:
    C, H, L, O, TV = loader()
    S = sample(build(C, H, L, O, TV))
    print("\n" + "#" * 96); print(f"# PANEL {name}"); print("#" * 96)
    eff = S["er"] >= 0.25
    mid = S["date"].median()
    print("ER>=0.25 (efficient movers) vs rest:")
    for h in [10, 15, 20]:
        show(S, eff, "efficient", h); show(S, ~eff, "  rest", h)
    print("halves (f20):")
    show(S, eff & (S["date"] <= mid), "efficient H1", 20)
    show(S, eff & (S["date"] > mid), "efficient H2", 20)
    show(S, ~eff & (S["date"] <= mid), "  rest H1", 20)
    show(S, ~eff & (S["date"] > mid), "  rest H2", 20)
    print("regimes (f20):")
    for rg in ["BULL", "CHOP", "BEAR"]:
        show(S, eff & (S["regime"] == rg), f"efficient | {rg}", 20)
        show(S, ~eff & (S["regime"] == rg), f"  rest | {rg}", 20)
    print("stacked on kept filters (f20):")
    cu = S["align"] == "ConfirmUp"
    show(S, cu & eff, "ConfirmUp + efficient", 20)
    show(S, cu & ~eff, "ConfirmUp + rest", 20)
    print("per-date long-short spread (efficient minus rest, f20):")
    sp = (S[eff].groupby("date")["f20"].mean() - S[~eff].groupby("date")["f20"].mean()).dropna()
    print(f"  mean {sp.mean()*100:+.2f}pp  t {sp.mean()/(sp.std()/np.sqrt(len(sp))):+.1f}  "
          f"pos-dates {(sp>0).mean()*100:.0f}%  (n_dates {len(sp)})")
    print("veto x breakout interaction (risk-vs-return trade, f20):")
    bo = S["brk"] == "Breakout"
    for lab, m in [("Breakout ALL", bo), ("Breakout not-risky", bo & ~S["risky"]),
                   ("Breakout risky", bo & S["risky"])]:
        x = S[m]["f20"]
        if len(x) < 30: print(f"  {lab:24s} thin"); continue
        print(f"  {lab:24s} n={len(x):5d}  mean {x.mean()*100:+.2f}%  med {x.median()*100:+.2f}%  "
              f"win {(x>0).mean()*100:4.1f}%  p10 {x.quantile(.1)*100:+.1f}%  p90 {x.quantile(.9)*100:+.1f}%")
