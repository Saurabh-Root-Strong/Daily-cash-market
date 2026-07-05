"""
ARE THE TWO DROPDOWNS REDUNDANT?  🚀 Setup (breakout state)  vs  🧭 Alignment (D×W quadrant)

Both now use the WEEKLY trend (the False-break split gated Breakout on weekly-up,
same leg that defines ConfirmUp/FalsePop). So: does 🧭 add anything over 🚀, or is
  • 🚀 Breakout   already ⊆  🧭 ConfirmUp   (weekly-up + daily-up), and
  • ⚠️ False break already ⊆  ⚠️ False Pop   (weekly-down + daily-up) — i.e. did the
    new False-break label just re-create False Pop under another name?

Measures on both panels (reusing v2 sample, but re-deriving brk with the SHIPPED
weekly gate so the audit matches the live engine):
  1. Cross-tab counts brk × align + Cramér's V (axis overlap).
  2. Conditional P(align | brk) for the key cells (subset test).
  3. Incremental forward return — does 🧭 sharpen 🚀, or is it already implied?
  4. Does ⚠️ False break carry info BEYOND generic ⚠️ False Pop (is it a worse trap)?
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market/scripts")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from audit_rotation_filters_v2 import load_dcm, load_fno, build, sample, bt, HS, COST

def cramers_v(ct):
    ct = ct.values.astype(float)
    chi2 = ((ct - ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()) ** 2
            / (ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()).clip(1e-9)).sum()
    n = ct.sum(); r, k = ct.shape
    return np.sqrt((chi2 / n) / max(1, min(r - 1, k - 1)))

def rederive_brk(S):
    """Match the SHIPPED engine: tight-base up-break gated on weekly trend."""
    wu = S["w_dir"] > 0
    out = np.where(S["brk"] == "Breakout",
                   np.where(wu, "Breakout", "FalseBreak"), S["brk"])
    return pd.Series(out, index=S.index)

def basket(S, mask, h, cost=0.0):
    if mask.sum() < 30: return None
    g = S[mask].groupby("date")[f"f{h}"].mean().dropna() - cost
    if len(g) < 6: return None
    t = g.mean() / (g.std() / np.sqrt(len(g)))
    return int(mask.sum()), g.mean() * 100, t, (S[mask][f"f{h}"] > 0).mean() * 100

def line(S, mask, label, h, cost=0.0):
    r = basket(S, mask, h, cost)
    if r is None: print(f"    {label:44s} f{h}: thin (n={int(mask.sum())})"); return
    n, m, t, w = r
    print(f"    {label:44s} f{h:<2d}: n={n:6d}  {m:+.2f}%  t{t:+.1f}  win {w:.0f}%")

def panel(name, C, H, L, O, TV):
    print("\n" + "#"*100); print(f"# {name}"); print("#"*100)
    X = build(C, H, L, O, TV); S = sample(X)
    S["brk2"] = rederive_brk(S)
    bo = S["brk2"] == "Breakout"; fb = S["brk2"] == "FalseBreak"
    cu = S["align"] == "ConfirmUp"; fp = S["align"] == "FalsePop"

    print("\n1) CROSS-TAB  brk2 (rows) × align (cols) — how the two axes co-occur")
    ct = pd.crosstab(S["brk2"], S["align"])
    order_r = [x for x in ["Breakout","FalseBreak","Extended","Bounce","Coiling","None"] if x in ct.index]
    order_c = [x for x in ["ConfirmUp","FalsePop","Pullback","DownAlign","Neutral"] if x in ct.columns]
    print(ct.loc[order_r, order_c].to_string())
    print(f"   Cramér's V(brk2, align) = {cramers_v(ct):.3f}   (0=independent, 1=identical)")

    print("\n2) SUBSET test — is each setup cell trapped inside one alignment cell?")
    def cond(m_from, m_to, lbl):
        if m_from.sum() == 0: print(f"    {lbl}: n=0"); return
        print(f"    P({lbl}) = {(m_from & m_to).sum()/m_from.sum()*100:5.1f}%   (n_from={int(m_from.sum())})")
    cond(bo, cu, "align=ConfirmUp | brk=Breakout")
    cond(bo, S["d_dir"] > 0, "daily-up      | brk=Breakout")
    cond(fb, fp, "align=FalsePop  | brk=FalseBreak")
    cond(fb, S["d_dir"] > 0, "daily-up      | brk=FalseBreak")

    print("\n3) DOES 🧭 SHARPEN 🚀? — breakout basket with/without the ConfirmUp gate")
    for h in [10, 20]:
        line(S, bo, "Breakout (all)", h)
        line(S, bo & cu, "Breakout ∩ ConfirmUp", h)
        line(S, bo & ~cu, "Breakout ∩ NOT-ConfirmUp", h)

    print("\n4) DOES 🚀 SHARPEN 🧭? — ConfirmUp basket with/without a breakout event")
    for h in [10, 20]:
        line(S, cu, "ConfirmUp (all)", h)
        line(S, cu & bo, "ConfirmUp ∩ Breakout", h)
        line(S, cu & ~bo & (S["brk2"] != "None"), "ConfirmUp ∩ other-setup", h)
        line(S, cu & (S["brk2"] == "None"), "ConfirmUp ∩ no-setup (grind)", h)

    print("\n5) IS ⚠️ FalseBreak A WORSE TRAP THAN GENERIC ⚠️ FalsePop? (incremental info)")
    for h in [10, 20]:
        line(S, fp, "FalsePop (all)", h)
        line(S, fb, "FalseBreak (the Donchian-break subset)", h)
        line(S, fp & ~fb, "FalsePop ∩ NOT-FalseBreak (rest)", h)
    return S

if __name__ == "__main__":
    CA, HA, LA, OA, TA = load_dcm()
    panel("PANEL A — DCM broad", CA, HA, LA, OA, TA)
    CB, HB, LB, OB, TB = load_fno()
    panel("PANEL B — 4yr F&O OOS", CB, HB, LB, OB, TB)
