"""
PRICE-ACTION AUDIT v3 — textbook pattern library, TRUE weekly candles, S/R levels,
breakout-candle quality. Dual panel, causal, forward-relative, 5/10/20d.

Sections:
  0. DATA SANITY — candle math integrity on the raw panel (zero-range bars, split
     artifacts, body+wick accounting) — "is the price action reading candles right?"
  1. DAILY CANDLE PATTERNS (the books): engulfing, hammer, shooting star, doji,
     marubozu, inside-bar, NR7 compression, 3 soldiers — each context-gated, tested
     at the horizons the dashboard trades (1-2-3 weeks), not the 1-3 day folklore.
  2. SUPPORT / RESISTANCE — multi-touch levels (>=3 touches of the rolling 20d
     extreme): fade at resistance? bounce at support? and the USER SCENARIO —
     many candles pressing a level then a BIG breakout through it.
  3. BREAKOUT QUALITY — does the breakout CANDLE matter? strong close, range
     expansion, volume surge, prior touches, tight base — each + stacked.
  4. WEEKLY-TREND HORSE RACE — current EMA20/50 daily proxy vs TRUE weekly-candle
     trends (completed weeks only, no lookahead): weekly SMA10, weekly HH/HL.
     Which gives the biggest ConfirmUp-vs-FalsePop spread?
  5. PULLBACK GRADING — weekly-up + daily-down: shallow (near EMA20) vs deep;
     with/without a reversal candle trigger.
All masks use info <= t. fwd returns relative to same-day universe median.
Basket stats = per-date basket mean -> t over dates; t_nonov = every-H-th date.
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from scripts.audit_rotation_filters_v2 import load_dcm, load_fno, regime_series  # type: ignore

HS = [5, 10, 20]
MIN_EVENTS_DATE = 3     # need >=3 names on a date to form a basket

def basket(fwdH, M, dates_step):
    m = M.fillna(False)
    cnt = m.sum(axis=1)
    ok = cnt >= MIN_EVENTS_DATE
    if ok.sum() < 8: return None
    s = (fwdH.where(m).mean(axis=1))[ok]
    wins = ((fwdH > 0) & m).sum(axis=1)[ok] / cnt[ok]
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    sn = s.iloc[::dates_step]
    tn = sn.mean() / (sn.std() / np.sqrt(len(sn))) if len(sn) >= 6 else np.nan
    return int(m.values.sum()), s.mean() * 100, t, tn, wins.mean() * 100, len(s)

def show(fwd, M, label, reg=None, regs=None):
    out = f"  {label:46s}"
    n0 = int(M.fillna(False).values.sum())
    if n0 < 60:
        print(out + f" n={n0} (thin)"); return
    for h in HS:
        r = basket(fwd[h], M, max(1, h))
        if r is None: out += f"  f{h}: thin"; continue
        n, m, t, tn, w, nd = r
        out += f"  f{h}: {m:+.2f}% t{t:+.1f}/{tn:+.1f} w{w:.0f}"
    print(out + f"   (n={n0})")
    if reg is not None and regs:
        for rg in regs:
            Mm = M.mul(reg == rg, axis=0)
            r = basket(fwd[10], Mm, 2)
            if r: print(f"      | {rg:4s} f10: {r[1]:+.2f}% t{r[2]:+.1f} w{r[4]:.0f} (n={r[0]})")

def spread(fwdH, Ma, Mb, la, lb):
    """paired per-date basket diff t between two masks"""
    a = fwdH.where(Ma.fillna(False)).mean(axis=1)
    b = fwdH.where(Mb.fillna(False)).mean(axis=1)
    d = (a - b).dropna()
    d = d[(Ma.fillna(False).sum(axis=1) >= MIN_EVENTS_DATE) & (Mb.fillna(False).sum(axis=1) >= MIN_EVENTS_DATE)]
    if len(d) < 10: print(f"  {la} vs {lb}: thin"); return
    print(f"  {la:34s} - {lb:22s}: {d.mean()*100:+.2f}pp  t {d.mean()/(d.std()/np.sqrt(len(d))):+.1f}  pos {(d>0).mean()*100:.0f}%  (dates {len(d)})")

def run(name, C, H, L, O, V):
    print("\n" + "#" * 108); print(f"# PANEL {name}: {C.shape[0]}d x {C.shape[1]}s  {C.index.min().date()}..{C.index.max().date()}"); print("#" * 108)
    rng = H - L
    body = (C - O).abs()
    up_w = H - np.maximum(O, C)
    lo_w = np.minimum(O, C) - L
    green = C > O; red = C < O
    atr20 = rng.rolling(20, min_periods=15).mean()
    ret5 = C / C.shift(5) - 1; ret10 = C / C.shift(10) - 1
    clv = ((C - L) / rng.replace(0, np.nan))            # close location 0..1
    vol_surge = (V / V.rolling(20, min_periods=10).mean().shift(1))

    # ── 0. DATA SANITY ─────────────────────────────────────────────────────────
    tot = C.notna().values.sum()
    zr = ((rng <= 0) & C.notna()).values.sum()
    bad_ohlc = ((H < np.maximum(O, C)) | (L > np.minimum(O, C))).values.sum()
    jump = ((C / C.shift(1) - 1).abs() > 0.25).values.sum()
    acc = ((body + up_w + lo_w - rng).abs() > 1e-6 * C).values.sum()
    print(f"0) SANITY: bars {tot:,} | zero-range {zr/tot*100:.2f}% | OHLC-violations {bad_ohlc} | "
          f">25% jumps (splits?) {jump} ({jump/tot*100:.3f}%) | body+wick!=range {acc}")

    fwd = {}
    for h in HS:
        f = C.shift(-h) / C - 1
        fwd[h] = f.sub(f.median(axis=1), axis=0)
    reg = regime_series(C.index)
    REGS = ["BULL", "CHOP", "BEAR"]

    # ── 1. DAILY CANDLE PATTERNS ───────────────────────────────────────────────
    print("\n1) DAILY CANDLE PATTERNS (context-gated) — f5/f10/f20: mean t_ov/t_nonov win%")
    meaningful = body >= 0.5 * atr20
    bull_eng = (green & red.shift(1) & (O <= C.shift(1)) & (C >= O.shift(1)) & meaningful & (ret5.shift(1) < 0))
    bear_eng = (red & green.shift(1) & (O >= C.shift(1)) & (C <= O.shift(1)) & meaningful & (ret5.shift(1) > 0))
    hammer = ((lo_w >= 2 * body) & (up_w <= body) & (rng >= atr20) & (ret5.shift(1) < -0.02))
    sstar  = ((up_w >= 2 * body) & (lo_w <= body) & (rng >= atr20) & (ret5.shift(1) > 0.02))
    doji_t = ((body <= 0.1 * rng) & (ret10.abs() > 0.04) & (rng >= atr20))
    maru_up = (green & (body >= 0.7 * rng) & (rng >= 1.5 * atr20))
    maru_dn = (red & (body >= 0.7 * rng) & (rng >= 1.5 * atr20))
    inside = ((H < H.shift(1)) & (L > L.shift(1)))
    nr7 = (rng == rng.rolling(7).min()) & (rng > 0)
    soldiers = (green & green.shift(1) & green.shift(2) & (C > C.shift(1)) & (C.shift(1) > C.shift(2)))
    show(fwd, bull_eng, "bullish engulfing (after 5d decline)", reg, REGS)
    show(fwd, bear_eng, "bearish engulfing (after 5d rise)")
    show(fwd, hammer, "hammer (after decline)", reg, REGS)
    show(fwd, sstar, "shooting star (after rise)")
    show(fwd, doji_t, "doji after trend (|10d|>4%)")
    show(fwd, maru_up, "marubozu UP (big conviction candle)")
    show(fwd, maru_dn, "marubozu DOWN")
    show(fwd, inside, "inside bar")
    show(fwd, nr7, "NR7 compression")
    show(fwd, soldiers, "three white soldiers")

    # ── 2. SUPPORT / RESISTANCE multi-touch ───────────────────────────────────
    print("\n2) SUPPORT/RESISTANCE — multi-touch rolling 20d levels")
    don_hi = H.shift(1).rolling(20).max(); don_lo = L.shift(1).rolling(20).min()
    near_res_day = H >= don_hi * 0.99
    near_sup_day = L <= don_lo * 1.01
    touches_res = near_res_day.shift(1).rolling(40).sum()
    touches_sup = near_sup_day.shift(1).rolling(40).sum()
    at_res = (H >= don_hi * 0.99) & (C < don_hi * 1.01)     # pressing, not broken
    at_sup = (L <= don_lo * 1.01) & (C > don_lo * 0.99)
    show(fwd, at_res & (touches_res >= 3), "AT resistance, >=3 prior touches (fade?)")
    show(fwd, at_res & (touches_res < 3), "AT resistance, <3 touches")
    show(fwd, at_sup & (touches_sup >= 3), "AT support, >=3 prior touches (bounce?)")
    show(fwd, at_sup & (touches_sup < 3), "AT support, <3 touches")

    # ── 3. BREAKOUT QUALITY (user scenario: consolidation at level -> BIG break) ──
    print("\n3) BREAKOUT QUALITY — does the breakout candle/level matter? (base = plain Donchian break)")
    brk = C > don_hi * 1.01
    tight = atr20.div(C).rank(axis=1, pct=True) <= 0.33
    strong_close = clv >= 0.8
    range_exp = rng >= 1.5 * atr20
    vsurge = vol_surge >= 2.0
    many_touch = touches_res >= 3
    show(fwd, brk, "breakout (plain)", reg, REGS)
    show(fwd, brk & tight, "  + tight base (shipped def)")
    show(fwd, brk & many_touch, "  + >=3 level touches (user scenario)")
    show(fwd, brk & strong_close, "  + strong close (clv>=0.8)")
    show(fwd, brk & range_exp, "  + range expansion (>=1.5x ATR)")
    show(fwd, brk & vsurge, "  + volume surge (>=2x 20d)")
    show(fwd, brk & tight & strong_close, "  + tight + strong close")
    show(fwd, brk & tight & vsurge, "  + tight + volume surge")
    show(fwd, brk & many_touch & strong_close & range_exp, "  + touches + strong close + range exp")
    q = (tight.astype(int) + strong_close.astype(int) + range_exp.astype(int)
         + vsurge.astype(int) + many_touch.astype(int))
    show(fwd, brk & (q >= 3), "  QUALITY SCORE >=3 of 5", reg, REGS)
    show(fwd, brk & (q <= 1), "  QUALITY SCORE <=1 of 5")
    spread(fwd[10], brk & (q >= 3), brk & (q <= 1), "breakout q>=3", "q<=1 (f10)")
    spread(fwd[20], brk & (q >= 3), brk & (q <= 1), "breakout q>=3", "q<=1 (f20)")

    # ── 4. WEEKLY-TREND HORSE RACE (true weekly candles, completed weeks only) ──
    print("\n4) WEEKLY-TREND DEFINITION HORSE RACE — ConfirmUp-vs-FalsePop f10 spread")
    d_dir = np.sign(C / C.shift(10) - 1)
    ema_proxy = np.sign(C.ewm(span=20, adjust=False).mean() - C.ewm(span=50, adjust=False).mean())
    WC = C.resample("W-FRI").last(); WH = H.resample("W-FRI").max(); WL = L.resample("W-FRI").min()
    w_sma10 = np.sign(WC.shift(1) - WC.rolling(10).mean().shift(1)).reindex(C.index, method="ffill")
    w_hhhl = (((WH > WH.shift(1)) & (WL > WL.shift(1))).shift(1)).reindex(C.index, method="ffill")
    w_llhl = (((WH < WH.shift(1)) & (WL < WL.shift(1))).shift(1)).reindex(C.index, method="ffill")
    dup = d_dir > 0
    for lab, up_m, dn_m in [
        ("EMA20/50 daily proxy (shipped)", ema_proxy > 0, ema_proxy < 0),
        ("TRUE weekly close>SMA10w", w_sma10 > 0, w_sma10 < 0),
        ("TRUE weekly HH+HL vs LL+LH", w_hhhl == True, w_llhl == True),
    ]:
        spread(fwd[10], dup & up_m, dup & dn_m, f"D-up & {lab}", "D-up & weekly-dn")

    # ── 5. PULLBACK GRADING (daily-down inside weekly uptrend) ────────────────
    print("\n5) PULLBACK GRADING — weekly-up + daily-down")
    ema20 = C.ewm(span=20, adjust=False).mean()
    pull = (ema_proxy > 0) & (d_dir < 0)
    near20 = (C / ema20 - 1).abs() <= 0.02
    deep = (C / ema20 - 1) < -0.06
    trigger = hammer | bull_eng
    show(fwd, pull, "pullback (all)", reg, REGS)
    show(fwd, pull & near20, "  shallow: at EMA20 (+-2%)")
    show(fwd, pull & deep, "  deep: >6% below EMA20")
    show(fwd, pull & trigger, "  + reversal candle trigger (hammer/engulf)")
    spread(fwd[10], pull & near20, pull & deep, "shallow pullback", "deep pullback (f10)")

CA, HA, LA, OA, TA = load_dcm()
run("A — DCM broad", CA, HA, LA, OA, TA)
CB, HB, LB, OB, TB = load_fno()
run("B — 4yr F&O OOS", CB, HB, LB, OB, TB)
