"""
EARLY WARNING of Nifty swing turns from FII / participant data -- an honest backtest.

Question: on the evening of day D, using only what is published by then, can we say a
Nifty swing TOP (or BOTTOM) is coming in the next 1-10 sessions -- better than price
alone can?

Targets (labels use the future, features never do):
  top_ahead[D]    = a confirmed 3% zigzag TOP falls on D+1 .. D+10
  bottom_ahead[D] = a confirmed 3% zigzag BOTTOM falls on D+1 .. D+10
  (5% versions reported too)

Features (all as of D's close; the participant / FII-stats files publish after the close,
so the earliest action is the D+1 open, which the target window respects):
  POSITIONS  FII/Client/Pro/DII index-futures long-share percentile (regime-aware),
             FII options bullish-share percentile, 5- and 20-session changes,
             5-session buildup/unwinding flows (% of own book, lot-adjusted)
  VOLUME     what FII TRADED that day (participant Vol file): futures buy share of its
             volume, options bullish-leg share of its option volume, churn (volume/OI),
             FII share of all index-futures volume -- 5-session averages, z-scored
  RUPEES     FII index-futures net buy value (FII derivatives statistics), 5-session sum,
             and 5-session % change in its OI value
  MARKET     total put/call OI ratio across all participants, z-scored
  PRICE      5/20-session return, distance from 20-session high/low, close vs 50-day
             average, 20-session realised vol, RSI-14, sessions since the 20-session extreme

Tests:
  1. single-feature AUC (does the feature rank turn-days above other days?), circular-shift
     permutation p, Benjamini-Hochberg across the whole screen
  2. walk-forward logistic regression, refit every calendar year on all prior years,
     scored on the next year only: PRICE-ONLY vs FII-ONLY vs COMBINED -- the combined
     model must beat price-only, otherwise FII data adds nothing
  3. practical overlay: hold Nifty, step aside for 10 sessions when the walk-forward model's
     top-risk is in its top decile (threshold from training data only), costs included
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from scipy.optimize import minimize                        # noqa: E402

from src.analytics import big_players as bp, positioning_cycle as pc   # noqa: E402
from src.data.repository import query_dataframe             # noqa: E402

AS = date(2026, 11, 13)
H = 10
RNG = np.random.default_rng(21)


# ───────────────────────────────────────────────────────────── features
def zt(s: pd.Series, win: int = 250) -> pd.Series:
    """Causal z-score, restarted at regime breaks like the percentiles."""
    parts = []
    for seg in pc._segments(s):
        m = seg.rolling(win, min_periods=60).mean()
        sd = seg.rolling(win, min_periods=60).std().replace(0, np.nan)
        parts.append((seg - m) / sd)
    return pd.concat(parts).reindex(s.index)


def build() -> tuple[pd.DataFrame, dict]:
    wide, px, exp = bp._load(AS)
    f = wide["FII"].dropna(subset=["fut_idx_long"])
    idx = f.index
    X = pd.DataFrame(index=idx)
    groups: dict[str, list[str]] = {"POSITIONS": [], "VOLUME": [], "RUPEES": [], "MARKET": [], "PRICE": []}

    def add(g, name, s):
        X[name] = s.reindex(idx).astype(float)
        groups[g].append(name)

    # positions
    for ct in ("FII", "Client", "Pro", "DII"):
        w = wide[ct].reindex(idx)
        ls = w["fut_idx_long"] / (w["fut_idx_long"] + w["fut_idx_short"]) * 100
        add("POSITIONS", f"{ct}_ls_pct", pc._pct_rank(ls))
        if ct == "FII":
            add("POSITIONS", "FII_ls_d5", ls.diff(5))
            add("POSITIONS", "FII_ls_d20", ls.diff(20))
    ob = f["opt_idx_call_long"] + f["opt_idx_put_short"]
    obr = f["opt_idx_call_short"] + f["opt_idx_put_long"]
    osh = ob / (ob + obr) * 100
    add("POSITIONS", "FII_opt_pct", pc._pct_rank(osh))
    add("POSITIONS", "FII_opt_d5", osh.rolling(5).median().diff(5))
    for nm, L, S in (("fut", f["fut_idx_long"], f["fut_idx_short"]), ("opt", ob, obr)):
        dl5, ds5, _ = bp._side_flows(bp._lot_adjust(L.astype(float)), bp._lot_adjust(S.astype(float)))
        add("POSITIONS", f"FII_{nm}_net_flow5", pd.Series(dl5 - ds5, index=idx))

    # volume (what FII traded)
    v = query_dataframe("""SELECT * FROM fao_participant WHERE upper(data_type)='VOL' AND trade_date <= ?""", [AS])
    v["trade_date"] = pd.to_datetime(v["trade_date"])
    v = v[v["trade_date"].isin(idx)]
    vf = v[v.client_type == "FII"].drop_duplicates("trade_date").set_index("trade_date").reindex(idx).astype(
        {c: float for c in v.columns if c not in ("trade_date", "client_type", "data_type")}, errors="ignore")
    vb = (vf["fut_idx_long"] / (vf["fut_idx_long"] + vf["fut_idx_short"]) * 100)
    add("VOLUME", "FII_fut_buyshare_vol5", zt(vb.rolling(5).mean()))
    vo = (vf["opt_idx_call_long"] + vf["opt_idx_put_short"])
    vot = vo + vf["opt_idx_call_short"] + vf["opt_idx_put_long"]
    add("VOLUME", "FII_opt_bullshare_vol5", zt((vo / vot * 100).rolling(5).mean()))
    churn = (vf["fut_idx_long"] + vf["fut_idx_short"]) / (f["fut_idx_long"] + f["fut_idx_short"])
    add("VOLUME", "FII_fut_churn5", zt(churn.rolling(5).mean()))
    allv = v.groupby("trade_date")[["fut_idx_long", "fut_idx_short"]].sum().sum(axis=1).reindex(idx)
    add("VOLUME", "FII_fut_volshare5", zt(((vf["fut_idx_long"] + vf["fut_idx_short"]) / allv).rolling(5).mean()))

    # rupees
    st = query_dataframe("""SELECT trade_date, buy_value_cr, sell_value_cr, oi_value_cr FROM fii_derivatives_stats
                            WHERE category='INDEX FUTURES' AND trade_date <= ?""", [AS])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st.drop_duplicates("trade_date").set_index("trade_date").reindex(idx)
    add("RUPEES", "FII_rupee_net5", zt((st["buy_value_cr"] - st["sell_value_cr"]).rolling(5).sum()))
    add("RUPEES", "FII_oi_value_chg5", st["oi_value_cr"].pct_change(5) * 100)

    # market-wide
    o = query_dataframe("""SELECT trade_date, SUM(opt_idx_put_long) pl, SUM(opt_idx_call_long) cl FROM fao_participant
                           WHERE upper(data_type)='OI' AND trade_date <= ? GROUP BY 1""", [AS])
    o["trade_date"] = pd.to_datetime(o["trade_date"])
    o = o.set_index("trade_date").reindex(idx)
    add("MARKET", "pcr_z", zt(o["pl"] / o["cl"]))

    # price
    c = px["close_val"].reindex(idx).astype(float)
    r = c.pct_change()
    add("PRICE", "ret5", c.pct_change(5) * 100)
    add("PRICE", "ret20", c.pct_change(20) * 100)
    add("PRICE", "dist_hi20", (c / c.rolling(20).max() - 1) * 100)
    add("PRICE", "dist_lo20", (c / c.rolling(20).min() - 1) * 100)
    add("PRICE", "vs_ma50", (c / c.rolling(50).mean() - 1) * 100)
    add("PRICE", "rvol20", r.rolling(20).std() * np.sqrt(250) * 100)
    up, dn = r.clip(lower=0).rolling(14).mean(), (-r.clip(upper=0)).rolling(14).mean()
    add("PRICE", "rsi14", 100 - 100 / (1 + up / dn.replace(0, np.nan)))
    hi_age = c.rolling(20).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True)
    lo_age = c.rolling(20).apply(lambda a: len(a) - 1 - int(np.argmin(a)), raw=True)
    add("PRICE", "days_since_hi20", hi_age)
    add("PRICE", "days_since_lo20", lo_age)

    # targets (future-looking labels only)
    cv = c.ffill().values
    n = len(cv)
    for thr in (3, 5):
        zz = pc._zigzag(cv, thr / 100)
        for kind, col in (("TOP", f"top{thr}"), ("BOTTOM", f"bot{thr}")):
            ti = np.array([i for i, k in zz if k == kind])
            lab = np.zeros(n)
            for i in range(n):
                d = ti - i
                lab[i] = float(((d >= 1) & (d <= H)).any())
            X[col] = lab
    # the last H sessions (and anything after the last confirmed turn) cannot be labelled honestly
    last_turn = zz[-1][0] if zz else n
    X.loc[X.index[max(0, min(n - H, last_turn + 1)):], [c_ for c_ in X if c_[:3] in ("top", "bot")]] = np.nan
    X["fwd10"] = ((px["close_val"].shift(-10) / px["open_val"].shift(-1) - 1) * 100).reindex(idx)
    X["oc1"] = ((px["close_val"].shift(-1) / px["open_val"].shift(-1) - 1)).reindex(idx)
    # a HELD position earns close-to-close; decided on D's evening, it applies from D+1
    X["cc_next"] = (px["close_val"].shift(-1) / px["close_val"] - 1).reindex(idx)
    return X, groups


# ───────────────────────────────────────────────────────────── tools
def auc(score: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(score) & np.isfinite(y)
    s, t = score[m], y[m]
    pos, neg = s[t == 1], s[t == 0]
    if len(pos) < 5 or len(neg) < 5:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logit_fit(Xtr: np.ndarray, ytr: np.ndarray, lam: float = 2.0):
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    sd[sd == 0] = 1
    Z = np.nan_to_num((Xtr - mu) / sd)
    Z = np.c_[np.ones(len(Z)), Z]

    def f(w):
        p = Z @ w
        ll = np.sum(np.logaddexp(0, p) - ytr * p)
        return ll + lam * np.sum(w[1:] ** 2), Z.T @ (1 / (1 + np.exp(-p)) - ytr) + 2 * lam * np.r_[0, w[1:]]
    w = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return w, mu, sd


def logit_pred(model, X_: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    Z = np.c_[np.ones(len(X_)), np.nan_to_num((X_ - mu) / sd)]
    return 1 / (1 + np.exp(-(Z @ w)))


def walk_forward(X: pd.DataFrame, feats: list[str], target: str) -> pd.Series:
    out = pd.Series(np.nan, index=X.index)
    years = sorted(set(X.index.year))
    for yr in years:
        if yr < 2020:
            continue
        tr = X[(X.index.year < yr)].dropna(subset=[target])
        te = X[X.index.year == yr]
        if len(tr) < 300 or te.empty:
            continue
        model = logit_fit(tr[feats].values, tr[target].values)
        out[te.index] = logit_pred(model, te[feats].values)
    return out


# ───────────────────────────────────────────────────────────── main
def main() -> None:
    X, G = build()
    feats = [f for g in G.values() for f in g]
    print(f"sessions {len(X)}  features {len(feats)}  ({', '.join(f'{k} {len(v)}' for k, v in G.items())})")
    for t in ("top3", "bot3", "top5", "bot5"):
        print(f"  base rate {t}: {100 * np.nanmean(X[t]):.1f}% of days have one in the next {H} sessions")

    # 1. single-feature screen
    print("\n1) SINGLE FEATURES — AUC (0.5 = useless; >0.5 higher value -> turn ahead; <0.5 lower value)")
    rows = []
    for t in ("top3", "bot3"):
        y = X[t].values
        for fe in feats:
            s = X[fe].values
            a = auc(s, y)
            if not np.isfinite(a):
                continue
            null = []
            m = np.isfinite(s) & np.isfinite(y)
            yy = y[m]
            ss = s[m]
            for _ in range(400):
                null.append(auc(np.roll(ss, RNG.integers(60, len(ss) - 60)), yy))
            null = np.abs(np.array(null) - 0.5)
            p = (null >= abs(a - 0.5)).mean()
            grp = next(g for g, v in G.items() if fe in v)
            rows.append((t, grp, fe, a, p))
    R = pd.DataFrame(rows, columns=["target", "group", "feature", "auc", "p"])
    R = R.sort_values("p")
    m = len(R)
    R["bh_ok"] = R["p"].values <= 0.05 * (np.arange(1, m + 1) / m)
    R["bh_ok"] = R["bh_ok"][::-1].cummax()[::-1]
    for t in ("top3", "bot3"):
        sub = R[R.target == t].copy()
        sub["edge"] = (sub["auc"] - 0.5).abs()
        print(f"\n  {t}: top 10 by |AUC-0.5|")
        print(sub.sort_values("edge", ascending=False).head(10)[["group", "feature", "auc", "p", "bh_ok"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  survivors of Benjamini-Hochberg (5%) across {m} tests: {int(R['bh_ok'].sum())}")
    print(R[R.bh_ok].groupby(["target", "group"]).size().to_string())

    # 2. walk-forward models
    print("\n2) WALK-FORWARD LOGISTIC (refit yearly on prior years, scored on the next year) — out-of-sample AUC")
    sets = {"PRICE only": G["PRICE"], "FII/participant only": G["POSITIONS"] + G["VOLUME"] + G["RUPEES"] + G["MARKET"],
            "COMBINED": feats}
    preds = {}
    for t in ("top3", "bot3", "top5", "bot5"):
        line = f"  {t}: "
        for nm, fs in sets.items():
            pr = walk_forward(X, fs, t)
            preds[(t, nm)] = pr
            line += f"{nm} {auc(pr.values, X[t].values):.3f}   "
        print(line)
    print("\n  by year (top3 / bot3), COMBINED minus PRICE-only AUC:")
    for t in ("top3", "bot3"):
        yrs = []
        for yr in range(2020, 2027):
            msk = (X.index.year == yr)
            a_c = auc(preds[(t, "COMBINED")][msk].values, X[t][msk].values)
            a_p = auc(preds[(t, "PRICE only")][msk].values, X[t][msk].values)
            yrs.append(f"{yr}: {a_c - a_p:+.3f}")
        print(f"   {t}  " + "  ".join(yrs))
    # block bootstrap of the combined-minus-price AUC gap (months as blocks)
    print("\n  is COMBINED better than PRICE-only? month-block bootstrap of the AUC gap (2020+):")
    for t in ("top3", "bot3"):
        d = pd.DataFrame({"c": preds[(t, "COMBINED")], "p": preds[(t, "PRICE only")], "y": X[t]}).dropna()
        months = d.index.to_period("M")
        um = months.unique()
        gaps = []
        for _ in range(500):
            pick = RNG.choice(um, len(um))
            dd = pd.concat([d[months == mm] for mm in pick])
            gaps.append(auc(dd["c"].values, dd["y"].values) - auc(dd["p"].values, dd["y"].values))
        g = np.array(gaps)
        print(f"   {t}: gap {auc(d['c'].values, d['y'].values) - auc(d['p'].values, d['y'].values):+.3f}  "
              f"95% CI [{np.percentile(g, 2.5):+.3f}, {np.percentile(g, 97.5):+.3f}]  P(gap<=0) {np.mean(g <= 0):.2f}")
    # precision in the top decile of predicted risk
    print("\n  precision when the model's risk is in its top 10% (2020+), vs the base rate:")
    for t in ("top3", "bot3"):
        for nm in sets:
            d = pd.DataFrame({"p": preds[(t, nm)], "y": X[t]}).dropna()
            thr = d["p"].quantile(0.9)
            print(f"   {t} {nm:<22} hit {100 * d.loc[d.p >= thr, 'y'].mean():5.1f}%  vs base {100 * d['y'].mean():5.1f}%")

    # 3. overlay
    print("\n3) PRACTICAL OVERLAY (2020+): long Nifty, step aside 10 sessions when walk-forward top-risk is in its"
          " TRAINING top decile; 5 bps per switch")
    for nm in sets:
        risk = preds[("top3", nm)]
        thr_by_year = {}
        for yr in range(2020, 2027):
            tr = preds[("top3", nm)][X.index.year < yr].dropna()
            thr_by_year[yr] = tr.quantile(0.9) if len(tr) > 100 else np.inf
        flag = np.array([r_ >= thr_by_year.get(d_.year, np.inf) if np.isfinite(r_) else False
                         for d_, r_ in zip(X.index, risk.values)])
        expo = np.ones(len(X))
        for i in np.flatnonzero(flag):
            expo[i + 1:i + 1 + H] = 0.0                   # act from the next session
        ret = X["cc_next"].fillna(0).values                  # held overnight: close-to-close
        d = pd.DataFrame({"r": ret, "e": expo}, index=X.index)
        d = d[d.index.year >= 2020]
        sw = np.abs(np.diff(np.r_[1, d["e"].values])).sum()
        strat = (1 + d["r"] * d["e"]).prod() * (1 - 0.0005) ** sw - 1
        bh = (1 + d["r"]).prod() - 1
        eq = (1 + d["r"] * d["e"]).cumprod()
        dd_s = (eq / eq.cummax() - 1).min()
        eqb = (1 + d["r"]).cumprod()
        dd_b = (eqb / eqb.cummax() - 1).min()
        print(f"   {nm:<22} in market {100 * d['e'].mean():4.0f}%  total {100 * strat:+6.1f}% (max DD {100 * dd_s:5.1f}%)"
              f"  vs always-in {100 * bh:+6.1f}% (max DD {100 * dd_b:5.1f}%)  switches {int(sw)}")
    print("\n   (close-to-close returns: a held position earns the overnight gap. Exposure decided on D's evening"
          " applies from D+1's close; the D->D+1 gap still belongs to the previous exposure.)")

    # 4. the fair version: warn of the TOP inside an uptrend, of the BOTTOM inside a downtrend
    print("\n4) FAIR TEST - top warnings judged only on uptrend days (ret20 > 0), bottom warnings only on downtrend"
          " days (ret20 < 0). Walk-forward, refit yearly on that subset.")
    for t, cond, lab in (("top3", X["ret20"] > 0, "uptrend"), ("bot3", X["ret20"] < 0, "downtrend"),
                         ("top5", X["ret20"] > 0, "uptrend"), ("bot5", X["ret20"] < 0, "downtrend")):
        Xs = X[cond.fillna(False)]
        line = f"  {t} on {lab} days (base {100 * np.nanmean(Xs[t]):.0f}%): "
        res = {}
        for nm, fs in sets.items():
            pr = walk_forward(Xs, fs, t)
            res[nm] = pr
            d = pd.DataFrame({"p": pr, "y": Xs[t]}).dropna()
            hit = 100 * d.loc[d.p >= d.p.quantile(0.9), "y"].mean() if len(d) > 50 else np.nan
            line += f"{nm} AUC {auc(pr.values, Xs[t].values):.3f} (top-decile hit {hit:.0f}%)  "
        print(line)
        d = pd.DataFrame({"c": res["COMBINED"], "p": res["PRICE only"], "y": Xs[t]}).dropna()
        months = d.index.to_period("M")
        um = months.unique()
        gaps = []
        for _ in range(400):
            pick = RNG.choice(um, len(um))
            dd = pd.concat([d[months == mm] for mm in pick])
            gaps.append(auc(dd["c"].values, dd["y"].values) - auc(dd["p"].values, dd["y"].values))
        g = np.array(gaps)
        print(f"     COMBINED minus PRICE-only: {auc(d['c'].values, d['y'].values) - auc(d['p'].values, d['y'].values):+.3f}"
              f"  95% CI [{np.percentile(g, 2.5):+.3f}, {np.percentile(g, 97.5):+.3f}]")
    print("\n   single FII features on the fair subsets (AUC, circular-shift p):")
    for t, cond in (("top3", X["ret20"] > 0), ("bot3", X["ret20"] < 0)):
        Xs = X[cond.fillna(False)]
        out = []
        for fe in G["POSITIONS"] + G["VOLUME"] + G["RUPEES"] + G["MARKET"]:
            s_, y_ = Xs[fe].values, Xs[t].values
            m_ = np.isfinite(s_) & np.isfinite(y_)
            a = auc(s_[m_], y_[m_])
            null = np.abs(np.array([auc(np.roll(s_[m_], RNG.integers(40, m_.sum() - 40)), y_[m_])
                                    for _ in range(300)]) - 0.5)
            out.append((fe, a, (null >= abs(a - 0.5)).mean()))
        out.sort(key=lambda r: -abs(r[1] - 0.5))
        print(f"   {t}: " + " | ".join(f"{fe} {a:.3f} (p {p_:.3f})" for fe, a, p_ in out[:6]))


if __name__ == "__main__":
    main()
