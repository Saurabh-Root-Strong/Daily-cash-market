"""
FILTER AUDIT v2 — ADVERSARIAL, all scenarios, 2-3 WEEK reaction.

Attacks the pruned filter set (setup / alignment / hide-risky) + re-checks the
REMOVED pa_class axis at longer horizons, on TWO panels:

  PANEL A: DCM broad universe (DuckDB, ~3000 liquid names, 372d) — where it runs live.
  PANEL B: Tradebot 4yr F&O daily (211 large-caps, 2022-05..2026-06) — OOS regimes.

Scenarios:
  1. HORIZONS 5/10/15/20d — overlapping basket-t AND non-overlapping (step=H) t.
  2. EVENT-TIME DECAY — breakout / ConfirmUp / FalsePop relative cum-return day 1..20
     (how it reacts within 2-3 weeks; optimal hold).
  3. REGIME split (Nifty SMA20/50 BULL/CHOP/BEAR) per bucket.
  4. UNIVERSE split (rupee-volume terciles) — is edge small/mid-cap only?
  5. INTERACTIONS — breakout ± risky veto, ConfirmUp ± risky veto, breakout × weekly.
  6. REMOVED-AXIS re-check — pa_class at 15/20d inside kept filters (removal safe?).
  7. FalsePop fade depth at 15/20d (does avoid-signal persist?).
  8. COILING conversion — watchlist value: P(breakout within 10d), fwd return then.
  9. MC null for breakout at 10/20d + net-of-cost (0.3% rt).
All causal (features <= t), forward returns RELATIVE to same-day universe median.
"""
from __future__ import annotations
import sys, glob, os
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

D_MOM, DON, LOOK = 10, 20, 60
ER_TREND, ER_RANGE, HIVOL, GAPPY_F, GAP_PCT, CONSOL_PCTL = 0.25, 0.12, 4.5, 0.35, 1.0, 0.33
HS = [5, 10, 15, 20]
COST = 0.003
DDIR = r"d:/Python Projects/Tradebot/data/historical/daily"

nif = pd.read_parquet(os.path.join(DDIR, "NSE_NIFTY50_INDEX_daily.parquet"))
nif["ts"] = pd.to_datetime(nif["ts"]).dt.normalize()
nif = nif.set_index("ts")["close"]

def regime_series(idx):
    n = nif.reindex(idx).ffill()
    s20, s50, r20 = n.rolling(20).mean(), n.rolling(50).mean(), n / n.shift(20) - 1
    out = pd.Series("CHOP", index=idx)
    out[(n > s20) & (s20 > s50) & (r20 > 0)] = "BULL"
    out[(n < s20) & (s20 < s50) & (r20 < 0)] = "BEAR"
    out[s50.isna()] = "NA"
    return out

def load_dcm():
    from src.data.repository import query_dataframe as q
    P = q("""
        WITH liq AS (
          SELECT DISTINCT symbol FROM daily_data
          WHERE series IN ('EQ','SM','ST') AND turnover_lacs>=? )
        SELECT b.symbol, b.trade_date, b.open_price o, b.high_price h,
               b.low_price l, b.close_price c, b.turnover_lacs tv
        FROM daily_data b INNER JOIN liq ON liq.symbol=b.symbol
        WHERE b.series IN ('EQ','SM','ST') AND b.close_price>0
        ORDER BY b.trade_date, b.symbol
    """, [100.0])
    P["trade_date"] = pd.to_datetime(P["trade_date"])
    piv = lambda v: P.pivot_table(v, "trade_date", "symbol")
    return piv("c"), piv("h"), piv("l"), piv("o"), piv("tv")

def load_fno():
    frames = {}
    for f in glob.glob(os.path.join(DDIR, "*_EQ_daily.parquet")):
        s = os.path.basename(f).replace("NSE_", "").replace("_EQ_daily.parquet", "")
        d = pd.read_parquet(f); d["ts"] = pd.to_datetime(d["ts"]).dt.normalize()
        frames[s] = d.set_index("ts")
    col = lambda k: pd.DataFrame({s: d[k] for s, d in frames.items()}).sort_index()
    C, Hh, Ll, O, V = col("close"), col("high"), col("low"), col("open"), col("volume")
    return C, Hh, Ll, O, (C * V) / 1e5   # rupee-vol as pseudo turnover

def build(C, H, L, O, TV):
    rng = H - L
    body = ((C - O).abs() / rng.replace(0, np.nan)).rolling(LOOK, min_periods=40).mean()
    gapf = (((O - C.shift(1)).abs() / C.shift(1) * 100) > GAP_PCT).rolling(LOOK, min_periods=40).mean()
    atr60 = (rng / C * 100).rolling(LOOK, min_periods=40).mean()
    er = (C - C.shift(LOOK)).abs() / C.diff().abs().rolling(LOOK).sum().replace(0, np.nan)
    d_dir = np.sign(C / C.shift(D_MOM) - 1)
    w_dir = np.sign(C.ewm(span=20, adjust=False).mean() - C.ewm(span=50, adjust=False).mean())
    don_hi = H.shift(1).rolling(DON).max(); don_lo = L.shift(1).rolling(DON).min()
    brk_u = C > don_hi * 1.01; brk_d = C < don_lo * 0.99
    atr20 = (rng / C * 100).rolling(DON, min_periods=15).mean()
    fwd = {h: (C.shift(-h) / C - 1).sub((C.shift(-h) / C - 1).median(axis=1), axis=0) for h in HS}
    # day-by-day relative cum return for the decay curve
    day = {k: (C.shift(-k) / C - 1).sub((C.shift(-k) / C - 1).median(axis=1), axis=0) for k in range(1, 21)}
    reg = regime_series(C.index)
    tvr = TV.rolling(20, min_periods=10).mean().rank(axis=1, pct=True)  # size/liquidity rank
    return dict(C=C, body=body, gapf=gapf, atr60=atr60, er=er, d_dir=d_dir, w_dir=w_dir,
                brk_u=brk_u, brk_d=brk_d, atr20=atr20, fwd=fwd, day=day, reg=reg, tvr=tvr)

def sample(X, step=5, warm=75):
    C = X["C"]; dates = C.index; rows = []
    for i in range(warm, len(dates) - max(HS) - 1, step):
        d = dates[i]
        rec = pd.DataFrame({
            "er": X["er"].iloc[i], "body": X["body"].iloc[i], "atr60": X["atr60"].iloc[i],
            "gapf": X["gapf"].iloc[i], "atr20": X["atr20"].iloc[i],
            "d_dir": X["d_dir"].iloc[i], "w_dir": X["w_dir"].iloc[i],
            "brk_u": X["brk_u"].iloc[i], "brk_d": X["brk_d"].iloc[i],
            "tvr": X["tvr"].iloc[i],
        })
        for h in HS: rec[f"f{h}"] = X["fwd"][h].iloc[i]
        rec = rec.dropna(subset=["er", "body", "atr60", "atr20", "f10", "f20"])
        if len(rec) < 60: continue
        med = rec["body"].median()
        hv = rec["atr60"] >= HIVOL
        rec["pa_class"] = np.where(rec["er"] >= ER_TREND,
                                   np.where(hv | (rec["body"] < med), "Volatile", "Clean"),
                                   np.where(rec["body"] >= med, "Choppy", "Quiet"))
        rec["risky"] = (rec["gapf"] >= GAPPY_F) | hv
        tight = rec["atr20"] <= rec["atr20"].quantile(CONSOL_PCTL)
        rec["brk"] = np.where(rec["brk_u"], np.where(tight, "Breakout", "Extended"),
                      np.where(rec["brk_d"], "Bounce", np.where(tight, "Coiling", "None")))
        rec["align"] = np.select(
            [(rec.d_dir > 0) & (rec.w_dir > 0), (rec.d_dir > 0) & (rec.w_dir < 0),
             (rec.d_dir < 0) & (rec.w_dir > 0), (rec.d_dir < 0) & (rec.w_dir < 0)],
            ["ConfirmUp", "FalsePop", "Pullback", "DownAlign"], "Neutral")
        rec["date"] = d; rec["regime"] = X["reg"].loc[d]
        rows.append(rec.reset_index(names="symbol"))
    return pd.concat(rows, ignore_index=True)

def bt(S, mask, h, cost=0.0):
    """per-date basket t (overlapping) + non-overlapping t (every h-th sampled date)"""
    g = S[mask].groupby("date")[f"f{h}"].mean().dropna() - cost
    if len(g) < 6: return None
    t_ov = g.mean() / (g.std() / np.sqrt(len(g)))
    gn = g.iloc[::max(1, h // 5)]
    t_no = gn.mean() / (gn.std() / np.sqrt(len(gn))) if len(gn) >= 5 else np.nan
    win = (S[mask][f"f{h}"] - cost > 0).mean() * 100
    return int(mask.sum()), g.mean() * 100, t_ov, t_no, win

def show(S, mask, label, h, cost=0.0):
    r = bt(S, mask, h, cost)
    if r is None: print(f"  {label:40s} f{h}: thin (n={int(mask.sum())})"); return
    n, m, to, tn, w = r
    print(f"  {label:40s} f{h:<2d}: n={n:6d}  {m:+.2f}%  t_ov {to:+.1f}  t_nonov {tn:+.1f}  win {w:4.1f}%")

def decay_curve(X, mask_fn, label, step=20, warm=75):
    """non-overlapping event-time mean relative cum return, day 1..20"""
    C = X["C"]; dates = C.index
    curves = []
    for i in range(warm, len(dates) - 21, step):
        m = mask_fn(i)
        if m.sum() < 3: continue
        curves.append([X["day"][k].iloc[i][m].mean() for k in range(1, 21)])
    if not curves: print(f"  {label}: no events"); return
    A = np.array(curves) * 100
    mu = np.nanmean(A, axis=0); se = np.nanstd(A, axis=0) / np.sqrt(A.shape[0])
    peak = int(np.nanargmax(mu)) + 1
    print(f"  {label} (n_dates {A.shape[0]}):  peak day {peak} ({mu[peak-1]:+.2f}%)")
    for k in [1, 3, 5, 10, 15, 20]:
        print(f"    day {k:2d}: {mu[k-1]:+.2f}%  (t {mu[k-1]/se[k-1] if se[k-1]>0 else 0:+.1f})")

def run_panel(name, C, H, L, O, TV):
    print("\n" + "#" * 104); print(f"# PANEL {name}: {C.shape[0]} days x {C.shape[1]} syms  {C.index.min().date()}..{C.index.max().date()}"); print("#" * 104)
    X = build(C, H, L, O, TV)
    S = sample(X)
    print(f"pooled obs {len(S):,}  dates {S['date'].nunique()}  regimes {S.groupby('regime')['date'].nunique().to_dict()}")

    bo = S["brk"] == "Breakout"; cu = S["align"] == "ConfirmUp"; fp = S["align"] == "FalsePop"

    print("\n1) HORIZON SWEEP — kept buckets, 5/10/15/20d (t_ov overlapping, t_nonov honest)")
    for h in HS:
        show(S, bo, "brk=Breakout", h)
    for h in [10, 20]:
        show(S, cu, "align=ConfirmUp", h); show(S, fp, "align=FalsePop", h)
        show(S, S["risky"], "risky flag=True", h); show(S, ~S["risky"], "risky flag=False", h)

    print("\n2) EVENT-TIME DECAY (non-overlap step 20) — reaction over 2-3 weeks")
    idx_of = {d: i for i, d in enumerate(X["C"].index)}
    def mfn(cond_cols):
        def f(i):
            d = X["C"].index[i]
            g = S[S["date"] == d]
            if g.empty: return pd.Series(False, index=X["C"].columns)
            syms = g.loc[cond_cols(g), "symbol"]
            return pd.Series(X["C"].columns.isin(syms), index=X["C"].columns)
        return f
    decay_curve(X, mfn(lambda g: g["brk"] == "Breakout"), "Breakout")
    decay_curve(X, mfn(lambda g: (g["brk"] == "Breakout") & (g["w_dir"] > 0)), "Breakout + weekly-up")
    decay_curve(X, mfn(lambda g: g["align"] == "ConfirmUp"), "ConfirmUp")
    decay_curve(X, mfn(lambda g: g["align"] == "FalsePop"), "FalsePop")

    print("\n3) REGIME SPLIT (f10 / f20)")
    for rg in ["BULL", "CHOP", "BEAR"]:
        rm = S["regime"] == rg
        for h in [10, 20]:
            show(S, bo & rm, f"Breakout | {rg}", h)
        r = bt(S, cu & rm, 10); r2 = bt(S, fp & rm, 10)
        if r and r2:
            print(f"  ConfirmUp-vs-FalsePop spread | {rg:5s} f10: {r[1]-r2[1]:+.2f}pp  (win {r[4]:.1f} vs {r2[4]:.1f})")

    print("\n4) UNIVERSE SPLIT (20d-avg rupee-turnover terciles)")
    for lab, lo, hi in [("small", 0.0, 0.33), ("mid", 0.33, 0.67), ("large", 0.67, 1.01)]:
        um = (S["tvr"] > lo) & (S["tvr"] <= hi)
        show(S, bo & um, f"Breakout | {lab}", 10)
        show(S, bo & um, f"Breakout | {lab}", 20)

    print("\n5) INTERACTIONS — does the 🛡️ risky veto help or hurt the kept baskets?")
    for h in [10, 20]:
        show(S, bo & ~S["risky"], "Breakout + veto ON (not risky)", h)
        show(S, bo & S["risky"], "Breakout risky-only (what veto drops)", h)
    for h in [10, 20]:
        show(S, cu & ~S["risky"], "ConfirmUp + veto ON", h)
        show(S, cu & S["risky"], "ConfirmUp risky-only", h)
    show(S, bo & (S["w_dir"] > 0), "Breakout + weekly-up", 10)
    show(S, bo & (S["w_dir"] > 0), "Breakout + weekly-up", 20)
    show(S, bo & (S["w_dir"] <= 0), "Breakout + weekly-dn/flat", 10)

    print("\n6) REMOVED AXIS re-check — pa_class at 15/20d (was removal safe at 2-3wk?)")
    for h in [15, 20]:
        for c in ["Clean", "Volatile", "Choppy", "Quiet"]:
            show(S, S["pa_class"] == c, f"pa_class={c}", h)
    for c in ["Clean", "Volatile", "Choppy", "Quiet"]:
        show(S, cu & (S["pa_class"] == c), f"ConfirmUp + {c}", 20)

    print("\n7) COILING conversion — watchlist value")
    coil = S["brk"] == "Coiling"
    # among today's coilers: did they break out within 10 trading days? use raw brk_u matrix
    conv = []
    for d, g in S[coil].groupby("date"):
        i = idx_of.get(d)
        if i is None or i + 10 >= len(X["C"].index): continue
        win_next = X["brk_u"].iloc[i + 1:i + 11]
        syms = [s for s in g["symbol"] if s in win_next.columns]
        if syms: conv.append(win_next[syms].any().mean())
    if conv: print(f"  P(coiler breaks out within 10d) = {np.mean(conv)*100:.1f}%  (dates {len(conv)})")
    show(S, coil, "Coiling (hold as-is)", 10); show(S, coil, "Coiling (hold as-is)", 20)

    print("\n8) NET-OF-COST + MC null (breakout)")
    for h in [10, 20]:
        show(S, bo, f"Breakout NET of {COST*100:.1f}% rt", h, cost=COST)
    rng_ = np.random.default_rng(11)
    for h in [10, 20]:
        real = S[bo].groupby("date")[f"f{h}"].mean().dropna().mean()
        nulls = []
        for _ in range(300):
            tot = cnt = 0
            for d, g in S.groupby("date"):
                k = int((g["brk"] == "Breakout").sum())
                if k: tot += g[f"f{h}"].sample(k, random_state=int(rng_.integers(1e9))).mean(); cnt += 1
            nulls.append(tot / cnt)
        nulls = np.array(nulls)
        print(f"  MC f{h}: real {real*100:+.2f}% vs null {nulls.mean()*100:+.2f}%  p(real>null) {(nulls < real).mean():.3f}")
    return S

CA, HA, LA, OA, TA = load_dcm()
run_panel("A — DCM broad", CA, HA, LA, OA, TA)
CB, HB, LB, OB, TB = load_fno()
run_panel("B — 4yr F&O OOS", CB, HB, LB, OB, TB)
