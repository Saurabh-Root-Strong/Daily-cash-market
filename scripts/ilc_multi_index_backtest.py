"""Index & Large Cap on Bank Nifty, Fin Nifty and Midcap Select -- re-tested, not
inherited.

The Nifty 50 verdict (scripts/nifty_bucket_backtest_v2.py + stages 2-4,
nifty_bucket_composite.py) was measured on Nifty's own 50 names. None of its
numbers transfer: a 14-stock bank index has different concentration, different
flow and a regime change in its own weights. So every Nifty test is re-run here
per index, on the SAME construction the live panel uses (src/analytics/
index_largecap.py: 21-session per-symbol delivery z, total-forward OI matched
contract-to-itself, settlement sessions dropped, +-50% winsorised), plus the
tests only these indices need.

  1 COMPOSITE    net flow score -> gap / next day / next 5, IC + hit vs base,
                 Newey-West t at lag = horizon
  2 CONTROL      does it survive the index's own close-strength (CLR) + momentum?
  3 WALK-FORWARD sign must hold in the second half of the panel
  4 BREADTH      heavyweight breadth -> gap, alone and beside CLR
  5 CARRY        does a top-heavy day (index - equal weight) predict anything?
  6 DISJOINT     5-day result on non-overlapping weeks, all 5 phase offsets
  7 REALITY      max |t| over the whole candidate family against a circular-shift
                 null that keeps the candidates' cross-correlation intact
  8 BAND         the per-index carry_band: same percentile of |spread| as Nifty's
                 0.15pp
  9 WEIGHT CAP   rolling implied weight of the heavyweight bucket -- the SEBI
                 concentration cap (top <= 20%, top 3 <= 45%) should show up as a
                 step down in 2025-26 for the bank indices
 10 MIDCAP BSE   how much of Midcap Select's carry spread is one stock

Usage:  python scripts/ilc_multi_index_backtest.py [--only BANKNIFTY] [--shifts 500]
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

from src.analytics.index_largecap import (INDEX_BUCKETS, index_meta,
                                          _DELIV_BASE, _DELIV_MINP,
                                          _opt_label)
from src.analytics.fno_stocks import _sessions_until
from src.analytics.adjusted_returns import ret_sql, factor_sql
_R_SQL = ret_sql('daily_data')

pd.set_option("display.width", 250)
ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2022-01-01")
ap.add_argument("--only", default=None)
ap.add_argument("--shifts", type=int, default=500)
ap.add_argument("--asof", default=None,
                help="basket in force on this date (default today); 2026-09-30 = post-review")
a = ap.parse_args()
rng = np.random.default_rng(7)
c = duckdb.connect("data/market_data.duckdb", read_only=True)
SETTLE = set(pd.to_datetime(c.execute(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date").df()["trade_date"]))


def nw_t(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 40:
        return float("nan")
    e = x - x.mean()
    v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float("nan")


def ols_nw(y, X, lag):
    """OLS with Newey-West standard errors. Returns t-stats (constant first)."""
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = X * e[:, None]
    meat = S.T @ S
    for L in range(1, lag + 1):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b / se


ASOF = pd.Timestamp(a.asof).date() if a.asof else None


def build(sym):
    meta = index_meta(sym, ASOF)
    B = meta["buckets"]
    W = meta["weights"]
    ALL = [s for v in B.values() for s in v]
    ph = ",".join("?" * len(ALL))
    start = pd.Timestamp(a.start)
    if meta.get("history_from"):
        start = max(start, pd.Timestamp(meta["history_from"]))
    warm = (start - pd.Timedelta(days=420)).date()

    cash = c.execute(f"""SELECT trade_date, symbol, deliv_per,
            {_R_SQL} r
        FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
          AND close_price>0 AND prev_close>0 AND trade_date>=?""", [*ALL, warm]).df()
    cash["trade_date"] = pd.to_datetime(cash["trade_date"])
    cash = cash[cash["r"].abs() < 40]
    R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
    DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()

    oi = c.execute(f"""
        WITH f AS (SELECT trade_date, symbol, expiry_date, instrument, option_type,
                          SUM(open_interest) oi FROM fno_bhavcopy
                   WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest>0
                     AND symbol IN ({ph}) AND expiry_date>trade_date AND trade_date>=?
                   GROUP BY 1,2,3,4,5),
        l AS (SELECT *, LAG(oi) OVER (PARTITION BY symbol, expiry_date, instrument,
                                      option_type ORDER BY trade_date) poi FROM f)
        SELECT trade_date, symbol, instrument, option_type,
               SUM(oi / COALESCE((SELECT EXP(SUM(LN(ca.share_factor))) FROM corp_actions ca
                    WHERE ca.symbol = l.symbol AND ca.ex_date = l.trade_date
                      AND ca.share_factor > 0 AND ca.share_factor <> 1), 1)) oi,
               SUM(poi) poi
        FROM l WHERE poi>0 GROUP BY 1,2,3,4""", [*ALL, warm]).df()
    oi["trade_date"] = pd.to_datetime(oi["trade_date"])
    oi = oi[~oi["trade_date"].isin(SETTLE)].copy()
    oi["pct"] = ((oi["oi"] - oi["poi"]) / oi["poi"] * 100).clip(-50, 50)
    FOI = oi[oi.instrument == "FUTSTK"].pivot_table("pct", "trade_date", "symbol")
    o = oi[oi.instrument == "OPTSTK"]
    CE = o[o.option_type == "CE"].pivot_table("pct", "trade_date", "symbol")
    PE = o[o.option_type == "PE"].pivot_table("pct", "trade_date", "symbol")

    ix = c.execute("""SELECT trade_date, open_val, high_val, low_val, close_val, pct_chg
        FROM index_data WHERE index_name=? AND trade_date>=?""",
                   [meta["index_name"], warm]).df()
    ix["trade_date"] = pd.to_datetime(ix["trade_date"])
    ix = ix.set_index("trade_date").sort_index().astype(float)

    IX = R.index.intersection(ix.index)
    y = ix.loc[IX, "pct_chg"]
    lg = np.log1p(y / 100)
    d5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100
    d1 = y.shift(-1)
    # gap needs consecutive rows; a hole in index_data would make a 2-day "gap"
    gap = (ix["open_val"] / ix["close_val"].shift(1) - 1).reindex(IX).shift(-1) * 100
    clr = ((ix["close_val"] - ix["low_val"]) /
           (ix["high_val"] - ix["low_val"]).replace(0, np.nan)).reindex(IX) - 0.5

    fe = c.execute("""SELECT trade_date, MIN(expiry_date) e FROM fno_bhavcopy
        WHERE instrument='FUTSTK' AND expiry_date>trade_date AND trade_date>=? GROUP BY 1""",
                   [warm]).df()
    DTE = {pd.Timestamp(t): _sessions_until(pd.Timestamp(t).date(), pd.Timestamp(e).date())
           for t, e in zip(fe["trade_date"], fe["e"])}
    legs, feats = {}, {}
    for i, (bn, mem) in enumerate(B.items()):
        cols = [s for s in mem if s in R.columns]
        sub = R[cols].reindex(IX)
        nmin = max(2, min(4, -(-len(mem) * 3 // 5)))      # live _min_valid
        D = DL[cols]
        mu = D.rolling(_DELIV_BASE, min_periods=_DELIV_MINP).mean().shift(1)
        sd = D.rolling(_DELIV_BASE, min_periods=_DELIV_MINP).std().shift(1)
        dz = ((D - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan) \
            .mean(axis=1).reindex(IX)
        dleg = pd.Series(np.where(dz > .3, 1., np.where(dz < -.3, -1., 0.)),
                         index=IX).where(dz.notna())
        f = FOI.reindex(columns=cols).reindex(IX)
        valid = f.notna() & sub.notna()
        bull = (((sub > 0) & (f > .5)) | ((sub > 0) & (f < -.5))) & valid
        bear = (((sub < 0) & (f > .5)) | ((sub < 0) & (f < -.5))) & valid
        nb, ns, nv = bull.sum(axis=1), bear.sum(axis=1), valid.sum(axis=1)
        # live rule: a lean needs the quorum among names that MOVED OI
        fleg = pd.Series(np.where(nb + ns < nmin, 0.,
                                  np.where(nb > ns * 1.5, 1.,
                                           np.where(ns > nb * 1.5, -1., 0.))),
                         index=IX).where(nv >= nmin)
        ce = CE.reindex(columns=cols).reindex(IX)
        pe = PE.reindex(columns=cols).reindex(IX)
        both = ce.notna() & pe.notna()
        cm, pm = ce.where(both).mean(axis=1), pe.where(both).mean(axis=1)
        # the LIVE label: gap net of its expiry-cycle norm (_opt_cycle_norm)
        _num = {"put writing": 1., "put side heavier": .5, "balanced": 0.,
                "call side heavier": -.5, "call writing": -1., None: np.nan}
        _sc = meta.get("opt_cycle_scale", 1.0)
        oleg = pd.Series([_num[_opt_label(c_, p_, DTE.get(t), _sc)]
                          for t, c_, p_ in zip(IX, cm, pm)],
                         index=IX).where(both.sum(axis=1) >= nmin)
        L = pd.DataFrame({"deliv": dleg, "fut": fleg, "opt": oleg})
        L["score"] = L.mean(axis=1)            # only the legs that exist, as live
        legs[bn] = L
        n = sub.notna().sum(axis=1)
        feats[f"b{i}_adv"] = ((sub > 0).sum(axis=1) / n.where(n > 0) * 100 - 50)
        feats[f"b{i}_ret"] = sub.mean(axis=1)
    sc = pd.DataFrame({bn: legs[bn]["score"] for bn in B})
    wv = pd.Series(W)
    net = (sc * wv).sum(axis=1) / sc.notna().mul(wv).sum(axis=1).replace(0, np.nan)
    ew = R.reindex(IX).mean(axis=1)
    K = pd.DataFrame({"net": net, "top": legs[list(B)[0]]["score"],
                      "spread": y - ew, "clr": clr, "mom": y,
                      "gap": gap, "d1": d1, "d5": d5, "idx": y, **feats})
    K = K[K.index >= start]
    return meta, B, legs, K, R.reindex(IX), ix


def run(sym):
    meta, B, legs, K, R, ix = build(sym)
    bl = list(B)
    print("\n" + "#" * 110)
    print(f"#  {meta['display']}  basket: {meta['source']}"
          f"{'  (PROVISIONAL)' if meta['provisional'] else ''}")
    print(f"#  buckets " +
          " / ".join(f"{b}({len(v)})" for b, v in B.items()) +
          f"   panel {K.index.min():%Y-%m-%d}..{K.index.max():%Y-%m-%d}")
    print("#" * 110)
    k = K.dropna(subset=["net", "clr", "gap", "d1"])
    print(f"sessions with a score {len(k)}   base: gap {k.gap.mean()*100:+.1f}bps "
          f"(up {(k.gap>0).mean()*100:.1f}%)  d1 {k.d1.mean()*100:+.1f}bps "
          f"(up {(k.d1>0).mean()*100:.1f}%)  d5 {k.d5.mean()*100:+.1f}bps "
          f"(up {(k.d5>0).mean()*100:.1f}%)")

    print("\n[1] COMPOSITE -- signed return, NW t at lag = horizon")
    res = {}
    for nm in ("net", "top"):
        for h, lag in (("gap", 1), ("d1", 1), ("d5", 5)):
            kk = k.dropna(subset=[h])
            sg = np.sign(kk[nm]).replace(0, np.nan) * kk[h]
            sg = sg.dropna()
            ic = kk[nm].rank().corr(kk[h].rank())
            base_up = (kk[h] > 0).mean() * 100
            res[(nm, h)] = dict(ic=ic, hit=(sg > 0).mean() * 100, base=base_up,
                                t=nw_t(sg, lag), n=len(sg))
            print(f"   {nm:4s} {h:3s} IC {ic:+.3f}  hit {(sg>0).mean()*100:5.1f}% "
                  f"(base up {base_up:5.1f}%)  mean {sg.mean()*100:+6.1f}bps  "
                  f"t={nw_t(sg, lag):+5.2f}  n={len(sg)}")

    print("\n[2] CONTROL -- net score beside momentum + CLR")
    for h, lag in (("gap", 1), ("d1", 1), ("d5", 5)):
        kk = k.dropna(subset=[h])
        t = ols_nw(kk[h].values, kk[["mom", "clr", "net"]].values, lag)
        print(f"   {h:3s} n={len(kk)}  mom t={t[1]:+5.2f}  CLR t={t[2]:+5.2f}  "
              f"NET t={t[3]:+5.2f}")

    print("\n[3] WALK-FORWARD -- first half vs second half")
    cut = k.index[len(k) // 2]
    for h in ("gap", "d1", "d5"):
        sg = (np.sign(k["net"]).replace(0, np.nan) * k[h]).dropna()
        i_, o_ = sg[sg.index < cut], sg[sg.index >= cut]
        print(f"   {h:3s} IN  {i_.mean()*100:+6.1f}bps hit {(i_>0).mean()*100:5.1f}%   "
              f"OUT (from {cut:%Y-%m-%d}) {o_.mean()*100:+6.1f}bps hit {(o_>0).mean()*100:5.1f}%")

    print(f"\n[4] BREADTH -- heavyweight ({bl[0]}) advance share -> next gap")
    kk = K.dropna(subset=["b0_adv", "clr", "gap", "mom"])
    t_alone = ols_nw(kk["gap"].values, kk[["b0_adv"]].values, 1)[1]
    t_all = ols_nw(kk["gap"].values, kk[["mom", "clr", "b0_adv"]].values, 1)
    rho = kk["b0_adv"].corr(kk["clr"])
    print(f"   alone t={t_alone:+5.2f}   with CLR+mom: breadth t={t_all[3]:+5.2f}  "
          f"CLR t={t_all[2]:+5.2f}   corr(breadth, CLR) {rho:+.3f}")
    kt = K.dropna(subset=[f"b{len(bl)-1}_adv", "gap"])
    print(f"   tail ({bl[-1]}) breadth alone t="
          f"{ols_nw(kt['gap'].values, kt[[f'b{len(bl)-1}_adv']].values, 1)[1]:+5.2f}")

    print("\n[5] CARRY SPREAD (index - equal weight) -> next gap / day, beside CLR")
    for h in ("gap", "d1"):
        kk = K.dropna(subset=["spread", "clr", "mom", h])
        t1 = ols_nw(kk[h].values, kk[["spread"]].values, 1)[1]
        tt = ols_nw(kk[h].values, kk[["mom", "clr", "spread"]].values, 1)
        print(f"   {h:3s} alone t={t1:+5.2f}   with CLR+mom t={tt[3]:+5.2f}")

    print("\n[6] DISJOINT WEEKS -- net score -> 5d on non-overlapping windows")
    ts_, ms_ = [], []
    for off in range(5):
        kk = k.iloc[off::5].dropna(subset=["d5"])
        sg = (np.sign(kk["net"]).replace(0, np.nan) * kk["d5"]).dropna()
        ts_.append(nw_t(sg, 0) if len(sg) > 40 else float("nan"))
        ms_.append(sg.mean() * 100)
    print("   phase offsets: " + "  ".join(f"{m:+.1f}bps(t{t:+.2f})"
                                          for m, t in zip(ms_, ts_)))

    print("\n[7] REALITY CHECK -- max |t| over the family vs circular-shift null")
    cand = {}
    for bn in bl:
        for leg in ("deliv", "fut", "opt", "score"):
            cand[f"{bn}:{leg}"] = legs[bn][leg].reindex(K.index)
    for i in range(len(bl)):
        cand[f"b{i}_adv"] = K[f"b{i}_adv"]
    cand["net"], cand["spread"] = K["net"], K["spread"]
    S = pd.DataFrame(cand)
    outs = {"gap": 1, "d1": 1, "d5": 5}

    def family_t(Sm):
        best = []
        for h, lag in outs.items():
            yv = K[h]
            for col in Sm.columns:
                sg = (np.sign(Sm[col]).replace(0, np.nan) * yv).dropna()
                best.append((abs(nw_t(sg, lag)), col, h))
        return best

    obs = family_t(S)
    obs = [o for o in obs if not math.isnan(o[0])]
    obs.sort(reverse=True)
    null = []
    n = len(S)
    for _ in range(a.shifts):
        sh = int(rng.integers(20, n - 20))
        Sm = pd.DataFrame(np.roll(S.values, sh, axis=0), index=S.index, columns=S.columns)
        v = [x[0] for x in family_t(Sm) if not math.isnan(x[0])]
        null.append(max(v) if v else 0)
    null = np.array(null)
    p = (null >= obs[0][0]).mean()
    print(f"   {len(obs)} candidates. best |t|={obs[0][0]:.2f} ({obs[0][1]} -> {obs[0][2]})  "
          f"null median {np.median(null):.2f}  95th {np.percentile(null,95):.2f}  p={p:.3f}")
    print("   next best: " + ", ".join(f"{c_}->{h}|t|{t:.2f}" for t, c_, h in obs[1:5]))
    # A family survivor is only interesting if it is not close-strength again:
    # the futures leg is SIGNED BY PRICE (long build / short cover = price up), so
    # it can be a price proxy wearing an OI label. Re-test each beside CLR + mom.
    print("   beside CLR + momentum (the control every Nifty survivor died under):")
    for t0, col, h in obs[:6]:
        kk = pd.DataFrame({"x": np.sign(S[col]), "clr": K["clr"], "mom": K["mom"],
                           "y": K[h]}).dropna()
        tt = ols_nw(kk["y"].values, kk[["mom", "clr", "x"]].values, outs[h])
        print(f"     {col:15s}->{h:3s} alone |t| {t0:.2f}   controlled t={tt[3]:+5.2f}"
              f"   CLR t={tt[2]:+5.2f}   corr(x, CLR) {kk.x.corr(kk.clr):+.2f}")

    print("\n[8] CARRY BAND -- |spread| percentiles")
    sp = K["spread"].dropna().abs()
    qs = sp.quantile([.5, .6, .7, .8, .9])
    print("   " + "  ".join(f"p{int(q*100)} {v:.3f}" for q, v in qs.items()))

    print(f"\n[9] IMPLIED WEIGHT of {bl[0]} (rolling 120d OLS of index on bucket returns)")
    Rb = K[[f"b{i}_ret" for i in range(len(bl))]]
    yy = K["idx"]
    rows = []
    for end in pd.date_range(K.index.min() + pd.Timedelta(days=200), K.index.max(), freq="QE"):
        w = (Rb.index <= end) & (Rb.index > end - pd.Timedelta(days=175))
        X_, y_ = Rb[w].dropna(), yy[w]
        y_ = y_.reindex(X_.index).dropna()
        X_ = X_.reindex(y_.index)
        if len(y_) < 60:
            continue
        b, *_ = np.linalg.lstsq(X_.values, y_.values, rcond=None)
        rows.append(f"{end:%Y-%m} {b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}")
    print("   " + "  ".join(rows))

    if sym == "MIDCPNIFTY" and "BSE" in R.columns:      # BSE left on 30 Sep 2026
        print("\n[10] How much of Midcap Select's carry spread is BSE alone?")
        ew = R.mean(axis=1).reindex(K.index)
        bse = (R["BSE"].reindex(K.index) - ew)
        kk = pd.DataFrame({"s": K["spread"], "b": bse}).dropna()
        print(f"   corr(spread, BSE - equal weight) {kk.s.corr(kk.b):+.3f}   "
              f"R2 {kk.s.corr(kk.b)**2:.3f}  n={len(kk)}")
    return res, qs


out = {}
for s in ([a.only] if a.only else list(INDEX_BUCKETS)):
    out[s] = run(s)

# carry-band calibration: Nifty's 0.15pp as a percentile of its own |spread|
if "NIFTY" in out:
    meta, B, legs, K, R, ix = build("NIFTY")
    q = (K["spread"].dropna().abs() < 0.15).mean()
    print(f"\n[BAND] Nifty 0.15pp sits at p{q*100:.1f} of its |spread|. Same percentile:")
    for s in out:
        _, _, _, Ks, _, _ = build(s)
        print(f"   {s:11s} {Ks['spread'].dropna().abs().quantile(q):.3f} pp")
