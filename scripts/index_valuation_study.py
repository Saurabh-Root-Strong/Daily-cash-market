"""Does index valuation (P/E, P/B, dividend yield) tell you when a crash is coming,
or when it is the best time to invest?

    python scripts/index_valuation_study.py

Writes data/index_valuation_study/summary.json, which the "Index P/E & P/B" panel
reads. Inputs: index_valuation (scripts/backfill_index_valuation.py), 1999+.

PRE-REGISTERED (written before any result was looked at)
  Primary test : Nifty 50, point-in-time percentile of the SPLICED P/E, against the
                 next 12 months' total-return (TRI) log return. Expected sign: negative.
  Family       : Nifty 50 x {P/E, P/B, DY, earnings-yield-minus-G-sec, composite,
                 5y z-scores of P/E, P/B, DY} x {1,3,6,12,24,36 months} forward return,
                 plus next-12-month max drawdown. One family-wise p-value.
  Inference    : month-end samples only. Three views, because overlapping windows
                 make the naive t useless: (a) Newey-West t with lag 1.5x horizon,
                 (b) OLS t on DISJOINT windows, median over all start phases,
                 (c) circular-shift null: the signal series is rotated against the
                 return series by a random offset (keeps both autocorrelations, kills
                 the alignment), 2,000 draws; the family p uses the max standardised
                 |IC| across all tests in each draw.
  Base rate    : every zone is scored against the unconditional forward return,
                 because Nifty rose in most 12-month windows regardless.
  Honesty      : 27 years hold ~27 independent 12-month windows and ~9 independent
                 36-month windows. Five >=20% crashes. That is the sample.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analytics.index_valuation import (STUDY_PATH, add_features,
                                           drawdown_episodes, index_names,
                                           load_valuation, return_level,
                                           valuation_zone)
from src.data.repository import query_dataframe

H = {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "24m": 504, "36m": 756}
H_MONTHS = {"1m": 1, "3m": 3, "6m": 6, "12m": 12, "24m": 24, "36m": 36}
SIGNALS = {                 # name -> (column, expected sign of IC with fwd return)
    "P/E percentile": ("pe_pct", -1),
    "P/B percentile": ("pb_pct", -1),
    "Div yield percentile": ("dy_pct", +1),
    "Earnings yield − G-sec percentile": ("ey_gap_pct", +1),
    "Composite (P/E+P/B+DY)": ("val_score", -1),
    "P/E 5y z-score": ("pe_z5y", -1),
    "P/B 5y z-score": ("pb_z5y", -1),
    "DY 5y z-score": ("dy_z5y", +1),
}
ZONES = [(0.0, 0.10, "Extreme cheap"), (0.10, 0.30, "Cheap"), (0.30, 0.70, "Fair"),
         (0.70, 0.90, "Expensive"), (0.90, 1.0001, "Extreme expensive")]
BROAD = {"Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty 200", "Nifty 500",
         "Nifty Midcap 100", "Nifty Midcap 150", "Nifty Smallcap 100",
         "Nifty Smallcap 250"}
N_SHIFT = 2000
RNG = np.random.default_rng(20260924)


# ── statistics ──────────────────────────────────────────────────────────────
def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 12:
        return np.nan
    return float(np.corrcoef(_rank(x[m]), _rank(y[m]))[0, 1])


def nw_t(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 12:
        return np.nan
    X = np.column_stack([np.ones(n), (x - x.mean()) / (x.std() or 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = (X * (y - X @ beta)[:, None])
    S = u.T @ u
    for l in range(1, min(lag, n - 1) + 1):
        w = 1 - l / (lag + 1)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    return float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else np.nan


def disjoint_t(x: np.ndarray, y: np.ndarray, step: int) -> tuple[float, int]:
    """OLS t on non-overlapping windows, median across the `step` start phases."""
    ts, ns = [], []
    for ph in range(step):
        xs, ys = x[ph::step], y[ph::step]
        m = ~(np.isnan(xs) | np.isnan(ys))
        if m.sum() < 6:
            continue
        xs, ys = xs[m], ys[m]
        n = len(xs)
        xc = xs - xs.mean()
        if (xc ** 2).sum() == 0:
            continue
        b = (xc * (ys - ys.mean())).sum() / (xc ** 2).sum()
        e = ys - ys.mean() - b * xc
        se = np.sqrt((e ** 2).sum() / (n - 2) / (xc ** 2).sum())
        ts.append(b / se)
        ns.append(n)
    if not ts:
        return np.nan, 0
    return float(np.median(ts)), int(np.median(ns))


def shift_null(sig: np.ndarray, outs: list[np.ndarray], draws: np.ndarray) -> np.ndarray:
    """IC of rotated signal vs each outcome: shape (len(draws), len(outs))."""
    res = np.empty((len(draws), len(outs)))
    for i, k in enumerate(draws):
        s = np.roll(sig, int(k))
        for j, y in enumerate(outs):
            res[i, j] = spearman(s, y)
    return res


# ── data ────────────────────────────────────────────────────────────────────
def outcomes(f: pd.DataFrame) -> pd.DataFrame:
    ll = np.log(return_level(f).to_numpy(dtype=float))
    c = f["close_val"].to_numpy(dtype=float)
    n = len(f)
    for k, h in H.items():
        fr = np.full(n, np.nan)
        if h < n:                      # a young index has no 36-month outcome yet
            fr[: n - h] = ll[h:] - ll[: n - h]
        f["fwd_" + k] = fr
    # worst close within the next 252 sessions, relative to today's close
    mdd = np.full(n, np.nan)
    s = pd.Series(c[::-1]).rolling(252, min_periods=252).min().to_numpy()[::-1]
    # s[i] = min(c[i..i+251]); we need min(c[i+1..i+252]) -> s[i+1]
    if n > 252:
        mdd[: n - 252] = s[1: n - 251] / c[: n - 252] - 1
    f["mdd_12m"] = mdd
    return f


def month_ends(f: pd.DataFrame) -> pd.DataFrame:
    g = f.assign(ym=f["trade_date"].dt.to_period("M"))
    return g.groupby("ym").tail(1).reset_index(drop=True)


def zone_of(p: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=p.index, dtype="object")
    for lo, hi, lab in ZONES:
        out[(p >= lo) & (p < hi)] = lab
    return out


def zone_table(m: pd.DataFrame, col: str, horizons=("3m", "12m", "24m", "36m"),
               invert: bool = False) -> list[dict]:
    """Zones keyed so that HIGH = EXPENSIVE. Yield signals pass invert=True (a high
    dividend or earnings yield is cheap)."""
    rows = []
    z = zone_of(1 - m[col] if invert else m[col])
    for lo, hi, lab in ZONES:
        sel = m[z == lab]
        row = {"zone": lab, "months": int(len(sel))}
        for k in horizons:
            r = np.expm1(sel["fwd_" + k].dropna())
            allr = np.expm1(m.loc[m[col].notna(), "fwd_" + k].dropna())
            row[k] = {
                "n": int(len(r)), "indep": round(len(r) / H_MONTHS[k], 1),
                "mean": _f(r.mean()), "median": _f(r.median()),
                "hit": _f((r > 0).mean()), "worst": _f(r.min()), "best": _f(r.max()),
                "excess_mean": _f(r.mean() - allr.mean()) if len(r) else None,
                "cagr_median": _f((1 + r.median()) ** (12 / H_MONTHS[k]) - 1)
                if len(r) and H_MONTHS[k] >= 12 else None,
            }
        d = sel["mdd_12m"].dropna()
        row["p_dd10"] = _f((d <= -0.10).mean())
        row["p_dd20"] = _f((d <= -0.20).mean())
        row["median_mdd"] = _f(d.median())
        row["mdd_n"] = int(len(d))
        rows.append(row)
    base = m.loc[m[col].notna()]
    rows.append({"zone": "ALL months (base rate)", "months": int(len(base)),
                 **{k: {"n": int(base["fwd_" + k].notna().sum()),
                        "indep": round(base["fwd_" + k].notna().sum() / H_MONTHS[k], 1),
                        "mean": _f(np.expm1(base["fwd_" + k]).mean()),
                        "median": _f(np.expm1(base["fwd_" + k]).median()),
                        "hit": _f((base["fwd_" + k].dropna() > 0).mean()),
                        "worst": _f(np.expm1(base["fwd_" + k]).min()),
                        "best": _f(np.expm1(base["fwd_" + k]).max()),
                        "excess_mean": 0.0,
                        "cagr_median": _f((1 + np.expm1(base["fwd_" + k]).median())
                                          ** (12 / H_MONTHS[k]) - 1)
                        if H_MONTHS[k] >= 12 else None}
                    for k in horizons},
                 "p_dd10": _f((base["mdd_12m"].dropna() <= -0.10).mean()),
                 "p_dd20": _f((base["mdd_12m"].dropna() <= -0.20).mean()),
                 "median_mdd": _f(base["mdd_12m"].median()),
                 "mdd_n": int(base["mdd_12m"].notna().sum())})
    return rows


def _f(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _d(x):
    return None if x is None or pd.isna(x) else str(pd.Timestamp(x).date())


# ── the study ───────────────────────────────────────────────────────────────
def family_tests(m: pd.DataFrame, draws: np.ndarray) -> dict:
    tests, null_cols = [], []
    for sname, (col, sign) in SIGNALS.items():
        x = m[col].to_numpy(dtype=float)
        outs, labels = [], []
        for k in H:
            outs.append(m["fwd_" + k].to_numpy(dtype=float))
            labels.append(("fwd", k))
        outs.append(m["mdd_12m"].to_numpy(dtype=float))
        labels.append(("mdd", "12m"))
        nul = shift_null(x, outs, draws)
        for j, (kind, k) in enumerate(labels):
            y = outs[j]
            ic = spearman(x, y)
            hm = H_MONTHS[k]
            dt, dn = disjoint_t(x, y, hm)
            nsd = np.nanstd(nul[:, j])
            p = float(np.mean(np.abs(nul[:, j]) >= abs(ic))) if np.isfinite(ic) else None
            tests.append({"signal": sname, "col": col, "expected_sign": sign,
                          "outcome": ("fwd return " if kind == "fwd" else "max drawdown ") + k,
                          "ic": _f(ic, 3), "nw_t": _f(nw_t(x, y, int(np.ceil(1.5 * hm))), 2),
                          "disjoint_t": _f(dt, 2), "disjoint_n": dn,
                          "p_shift": _f(p, 3), "null_sd": _f(nsd, 3),
                          "z_null": _f(ic / nsd if nsd else np.nan, 2),
                          "months": int((~(np.isnan(x) | np.isnan(y))).sum())})
            null_cols.append(nul[:, j] / (nsd or 1))
    N = np.abs(np.column_stack(null_cols))
    fam_max = np.nanmax(N, axis=1)
    obs = [abs(t["z_null"]) if t["z_null"] is not None else 0 for t in tests]
    for t, o in zip(tests, obs):
        t["p_family"] = _f(float(np.mean(fam_max >= o)), 3)
    best = int(np.argmax(obs))
    return {"tests": tests, "n_tests": len(tests), "best": tests[best],
            "family_p_best": tests[best]["p_family"]}


def eras(m: pd.DataFrame) -> list[dict]:
    spans = [("1999–2012", "1999-01-01", "2012-12-31"),
             ("2013–2026", "2013-01-01", "2026-12-31"),
             ("2018–2026 (your window)", "2018-01-01", "2026-12-31")]
    out = []
    for lab, a, b in spans:
        s = m[(m["trade_date"] >= a) & (m["trade_date"] <= b)]
        row = {"era": lab, "months": int(len(s))}
        for sname in ("P/E percentile", "P/B percentile", "Div yield percentile",
                      "Composite (P/E+P/B+DY)", "P/E 5y z-score"):
            col = SIGNALS[sname][0]
            for k in ("12m", "36m"):
                x, y = s[col].to_numpy(float), s["fwd_" + k].to_numpy(float)
                row[f"{sname}|{k}"] = {"ic": _f(spearman(x, y), 3),
                                       "disjoint_t": _f(disjoint_t(x, y, H_MONTHS[k])[0], 2),
                                       "n": int((~(np.isnan(x) | np.isnan(y))).sum())}
        out.append(row)
    return out


def robustness(nf: pd.DataFrame) -> dict:
    """Checks added AFTER the first look at results; none is in the family p.

    splice : P/E percentile built on the RAW (unspliced) P/E, i.e. treating the
             2021 accounting change as if it were a real de-rating.
    ex_0309: the same tests with every month from 2003 through 2009 removed. The
             2003 cheap-to-2008 expensive cycle is the single biggest contributor.
    pbdy   : P/B + DY composite. Post-hoc: chosen because those two did not drift
             between 1999-2012 and 2013-2026 while P/E did.
    """
    from src.analytics.index_valuation import pit_percentile
    f = nf.copy()
    f["pe_raw_pct"] = pit_percentile(f["pe"])
    parts = pd.concat([f["pb_pct"], 1 - f["dy_pct"]], axis=1)
    f["pbdy_raw"] = parts.mean(axis=1, skipna=False)
    f["pbdy_score"] = pit_percentile(f["pbdy_raw"], min_obs=250)
    m = month_ends(f)
    ex = m[(m["trade_date"] < "2003-01-01") | (m["trade_date"] > "2009-12-31")]
    post = m[m["trade_date"] >= "2013-01-01"]
    since18 = m[m["trade_date"] >= "2018-01-01"]
    out = {}
    for lab, col in (("P/E percentile (spliced)", "pe_pct"),
                     ("P/E percentile (RAW, unspliced)", "pe_raw_pct"),
                     ("P/B percentile", "pb_pct"), ("Div yield percentile", "dy_pct"),
                     ("Composite (P/E+P/B+DY)", "val_score"),
                     ("P/B+DY composite (post-hoc)", "pbdy_score")):
        row = {}
        for tag, s in (("all", m), ("ex_2003_2009", ex), ("2013+", post), ("2018+", since18)):
            for k in ("12m", "36m"):
                x, y = s[col].to_numpy(float), s["fwd_" + k].to_numpy(float)
                row[f"{tag}|{k}"] = {"ic": _f(spearman(x, y), 3),
                                     "disjoint_t": _f(disjoint_t(x, y, H_MONTHS[k])[0], 2),
                                     "n": int((~(np.isnan(x) | np.isnan(y))).sum())}
        row["share_months_ge70"] = _f((m[col] >= 0.7).mean(), 3)
        row["share_months_ge70_since_2013"] = _f((post[col] >= 0.7).mean(), 3)
        out[lab] = row
    out["pbdy_zones"] = zone_table(m, "pbdy_score")
    out["pbdy_now"] = _f(f["pbdy_score"].iloc[-1], 3)
    return out


def threshold_drift(f: pd.DataFrame) -> dict:
    """A fixed 'expensive' line learned on 1999-2012, applied afterwards."""
    a = f[f["trade_date"] <= "2012-12-31"]
    b = f[f["trade_date"] > "2012-12-31"]
    out = {}
    for c in ("pe_adj", "pb_adj", "dy_adj"):
        q80 = a[c].quantile(0.8)
        q20 = a[c].quantile(0.2)
        out[c] = {"train_p80": _f(q80, 2), "train_p20": _f(q20, 2),
                  "share_after_above_p80": _f((b[c] > q80).mean(), 3),
                  "share_after_below_p20": _f((b[c] < q20).mean(), 3),
                  "median_1999_2012": _f(a[c].median(), 2),
                  "median_2013_2026": _f(b[c].median(), 2)}
    return out


def episode_audit(f: pd.DataFrame, vix: pd.Series, thr: float, cap: float | None) -> list[dict]:
    ep = drawdown_episodes(f["close_val"], f["trade_date"], thr)
    rows = []
    for _, e in ep.iterrows():
        if cap is not None and e["depth"] <= -cap:
            continue
        p, t = int(e["peak_i"]), int(e["trough_i"])
        pre = f.iloc[max(0, p - 126): p + 1]
        pk, tr = f.iloc[p], f.iloc[t]
        # decomposition of the fall: earnings vs multiple
        eps_chg = tr["eps"] / pk["eps"] - 1 if pk["eps"] and tr["eps"] else np.nan
        pe_chg = tr["pe_adj"] / pk["pe_adj"] - 1 if pk["pe_adj"] and tr["pe_adj"] else np.nan
        lvl = return_level(f).to_numpy(float)
        fw = {}
        for k in ("12m", "36m"):
            h = H[k]
            fw[k] = _f(lvl[t + h] / lvl[t] - 1) if t + h < len(f) else None
        # when did the composite first say "cheap" (<=0.30) during the fall, and how
        # much further did the index fall after that?
        fall = f.iloc[p: t + 1]
        cheap = fall.index[fall["val_score"] <= 0.30]
        first_cheap = None
        if len(cheap):
            ci = int(cheap[0])
            first_cheap = {"date": _d(f.loc[ci, "trade_date"]),
                           "further_fall": _f(f["close_val"].iloc[ci: t + 1].min()
                                              / f.loc[ci, "close_val"] - 1)}
        rows.append({
            "peak": _d(e["peak_date"]), "trough": _d(e["trough_date"]),
            "recovered": _d(e["recover_date"]), "depth": _f(e["depth"]),
            "fall_sessions": int(e["fall_sessions"]),
            "peak_level": _f(pk["close_val"], 1), "trough_level": _f(tr["close_val"], 1),
            "pe_at_peak": _f(pk["pe"], 2), "pe_adj_at_peak": _f(pk["pe_adj"], 2),
            "pe_pct_at_peak": _f(pk["pe_pct"], 3), "pb_pct_at_peak": _f(pk["pb_pct"], 3),
            "dy_pct_at_peak": _f(pk["dy_pct"], 3), "val_at_peak": _f(pk["val_score"], 3),
            "val_max_6m_before": _f(pre["val_score"].max(), 3),
            "zone_at_peak": valuation_zone(pk["val_score"]),
            "ey_gap_at_peak": _f(pk["ey_gap"], 2),
            "pe_at_trough": _f(tr["pe"], 2), "pe_pct_at_trough": _f(tr["pe_pct"], 3),
            "pb_pct_at_trough": _f(tr["pb_pct"], 3), "val_at_trough": _f(tr["val_score"], 3),
            "zone_at_trough": valuation_zone(tr["val_score"]),
            "vix_at_peak": _f(vix.get(pk["trade_date"]), 1),
            "vix_at_trough": _f(vix.get(tr["trade_date"]), 1),
            "eps_change": _f(eps_chg), "pe_change": _f(pe_chg),
            "fwd_from_trough": fw, "first_cheap_signal": first_cheap,
        })
    return rows


def warning_stats(m: pd.DataFrame) -> dict:
    """Was 'expensive' followed by a crash more often than any month? And how many
    crashes came WITHOUT the warning?"""
    out = {}
    for col in ("val_score", "pe_pct", "pb_pct"):
        s = m[[col, "mdd_12m"]].dropna()
        crash = s["mdd_12m"] <= -0.20
        hi = s[col] >= 0.90
        mid = s[col] >= 0.70
        out[col] = {
            "months": int(len(s)), "base_p_crash": _f(crash.mean(), 3),
            "p_crash_if_ge90": _f(crash[hi].mean(), 3), "months_ge90": int(hi.sum()),
            "p_crash_if_ge70": _f(crash[mid].mean(), 3), "months_ge70": int(mid.sum()),
            "p_crash_if_lt30": _f(crash[s[col] < 0.30].mean(), 3),
            "months_lt30": int((s[col] < 0.30).sum()),
            "share_of_crash_months_flagged_ge70": _f(mid[crash].mean(), 3),
            "crash_months": int(crash.sum()),
        }
    return out


def sector_cross_section(feats: dict[str, pd.DataFrame], draws_n: int = 1000) -> dict:
    """Among SECTOR indices, does the one that is cheap vs its OWN history beat
    Nifty 50 over the next 12 months? Cross-sectional rank IC each month-end."""
    nifty = feats["Nifty 50"][["trade_date", "fwd_12m"]].rename(columns={"fwd_12m": "n12"})
    panel = []
    for name, f in feats.items():
        if name in BROAD:
            continue
        m = month_ends(f)[["trade_date", "pe_pct", "pb_pct", "dy_pct", "val_score", "fwd_12m"]]
        m = m.merge(nifty, on="trade_date", how="inner")
        m["rel12"] = m["fwd_12m"] - m["n12"]
        m["index_name"] = name
        panel.append(m)
    if not panel:
        return {}
    P = pd.concat(panel, ignore_index=True)
    P["ym"] = P["trade_date"].dt.to_period("M")
    res = {}
    months = sorted(P["ym"].unique())
    for col, sign in (("pb_pct", -1), ("pe_pct", -1), ("dy_pct", +1), ("val_score", -1)):
        ics, used = [], []
        for ym in months:
            s = P[P["ym"] == ym][[col, "rel12"]].dropna()
            if len(s) < 6:
                continue
            # ~10 sectors a month: rank correlation directly, not spearman()'s
            # 12-point floor (which silently returned NaN for every month)
            ics.append(float(np.corrcoef(s[col].rank(), s["rel12"].rank())[0, 1]))
            used.append(ym)
        ics = np.array(ics)
        if len(ics) < 24:
            res[col] = None
            continue
        # NW t of the mean monthly IC, lag 18 (12m windows overlap)
        x = ics - ics.mean()
        L = 18
        S = (x ** 2).sum() + 2 * sum((1 - l / (L + 1)) * (x[l:] * x[:-l]).sum()
                                     for l in range(1, L + 1))
        t = ics.mean() / np.sqrt(S / len(ics) ** 2)
        # disjoint: every 12th month, median over phases
        dts = []
        for ph in range(12):
            v = ics[ph::12]
            if len(v) >= 5:
                dts.append(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        # top-minus-bottom tercile relative return per month
        spread = []
        for ym in used:
            s = P[P["ym"] == ym][[col, "rel12"]].dropna()
            q = s[col].rank(pct=True)
            spread.append(np.expm1(s.loc[q <= 1 / 3, "rel12"]).mean()
                          - np.expm1(s.loc[q > 2 / 3, "rel12"]).mean())
        res[col] = {"months": int(len(ics)), "mean_ic": _f(ics.mean(), 3),
                    "nw_t": _f(t, 2), "disjoint_t_median": _f(np.median(dts), 2) if dts else None,
                    "expected_sign": sign,
                    "cheap_minus_rich_12m": _f(np.nanmean(spread)),
                    "first": str(used[0]), "last": str(used[-1]),
                    "sectors_median": int(P.groupby("ym")[col].count().median())}
    return res


def vix_study(nifty_m: pd.DataFrame) -> dict:
    s = nifty_m.dropna(subset=["vix_pct"])
    out = {"months": int(len(s)), "from": _d(s["trade_date"].min()) if len(s) else None}
    for k in ("3m", "12m"):
        x, y = s["vix_pct"].to_numpy(float), s["fwd_" + k].to_numpy(float)
        out[k] = {"ic": _f(spearman(x, y), 3), "disjoint_t": _f(disjoint_t(x, y, H_MONTHS[k])[0], 2)}
    both = s[(s["vix_pct"] >= 0.8) & (s["val_score"] <= 0.3)]
    out["fear_and_cheap_months"] = int(len(both))
    out["fear_and_cheap_fwd12_median"] = _f(np.expm1(both["fwd_12m"]).median())
    out["all_fwd12_median"] = _f(np.expm1(s["fwd_12m"]).median())
    out["fear_and_cheap_dates"] = [_d(d) for d in both["trade_date"]]
    return out


def data_audit(f: pd.DataFrame) -> dict:
    d = f["trade_date"]
    gaps = d.diff().dt.days
    big = f.loc[gaps > 6, "trade_date"]
    px = np.log(f["close_val"]).diff()
    resid = (np.log(f["pe"]).diff() - px).abs()
    jumps = f.loc[resid > 0.08, ["trade_date"]].assign(r=resid[resid > 0.08])
    return {"sessions": int(len(f)), "first": _d(d.min()), "last": _d(d.max()),
            "gaps_over_6_days": [_d(x) for x in big][:20],
            "pe_null_sessions": int(f["pe"].isna().sum()),
            "tri_null_sessions": int(f["tri"].isna().sum()),
            "close_null_sessions": int(f["close_val"].isna().sum()),
            "pe_jumps_unexplained_by_price_gt8pct": [
                {"date": _d(r.trade_date), "log_jump": _f(r.r, 3)} for r in jumps.itertuples()][:15]}


def main() -> None:
    t0 = time.time()
    vix = query_dataframe("SELECT trade_date, vix FROM india_vix_history ORDER BY 1")
    vix["trade_date"] = pd.to_datetime(vix["trade_date"])
    vix_s = vix.set_index("trade_date")["vix"]

    feats, breaks, audits = {}, {}, {}
    for name in index_names():
        raw = load_valuation(name)
        if len(raw) < 1500:
            continue
        f, info = add_features(raw)
        f = outcomes(f)
        feats[name], breaks[name], audits[name] = f, info, data_audit(f)
        print(f"{name:26s} {len(f):>6,} sessions  break={info}", flush=True)

    nf = feats["Nifty 50"].merge(vix.rename(columns={"vix": "vix_lvl"}), on="trade_date", how="left")
    from src.analytics.index_valuation import pit_percentile
    nf["vix_pct"] = pit_percentile(nf["vix_lvl"], min_obs=500)
    feats["Nifty 50"] = nf
    nm = month_ends(nf)
    draws = RNG.integers(24, len(nm) - 24, N_SHIFT)
    fam = family_tests(nm, draws)
    print("family done", round(time.time() - t0), "s", flush=True)

    nifty = {
        "family": fam,
        "zones": {s: zone_table(nm, c, invert=sign > 0) for s, (c, sign) in SIGNALS.items()
                  if s in ("P/E percentile", "P/B percentile", "Div yield percentile",
                           "Composite (P/E+P/B+DY)", "Earnings yield − G-sec percentile")},
        "eras": eras(nm),
        "drift": threshold_drift(nf),
        "robustness": robustness(nf),
        "crashes": episode_audit(nf, vix_s, 0.20, None),
        "corrections": episode_audit(nf, vix_s, 0.10, 0.20),
        "warning": warning_stats(nm),
        "vix": vix_study(nm),
        "months": int(len(nm)),
        "cheap_months": [_d(x) for x in nm.loc[nm["val_score"] <= 0.30, "trade_date"]],
        "first_signal_month": _d(nm.loc[nm["val_score"].notna(), "trade_date"].min()),
    }

    per_index = {}
    for name, f in feats.items():
        m = month_ends(f)
        dr = RNG.integers(24, max(25, len(m) - 24), 500)
        row = {"months": int(len(m)), "first": _d(f["trade_date"].min())}
        for col in ("val_score", "pe_pct", "pb_pct", "dy_pct"):
            x = m[col].to_numpy(float)
            for k in ("12m", "36m"):
                y = m["fwd_" + k].to_numpy(float)
                ic = spearman(x, y)
                nul = shift_null(x, [y], dr)[:, 0]
                row[f"{col}|{k}"] = {"ic": _f(ic, 3),
                                     "disjoint_t": _f(disjoint_t(x, y, H_MONTHS[k])[0], 2),
                                     "p_shift": _f(np.mean(np.abs(nul) >= abs(ic)), 3)
                                     if np.isfinite(ic) else None,
                                     "n": int((~(np.isnan(x) | np.isnan(y))).sum())}
        row["zones_val"] = zone_table(m, "val_score", horizons=("12m", "36m"))
        row["crashes"] = episode_audit(f, vix_s if name == "Nifty 50" else pd.Series(dtype=float),
                                       0.20, None)
        per_index[name] = row
        print(f"per-index {name}", flush=True)

    xs = sector_cross_section(feats)

    gs = query_dataframe("SELECT source, month, value FROM gsec_yield_monthly")
    gw = gs.pivot_table(index="month", columns="source", values="value").dropna()
    splice = {"overlap_months": int(len(gw)),
              "mean_abs_diff": _f((gw["IMF_IFS"] - gw["OECD_FINMARK"]).abs().mean(), 3)
              if len(gw) else None,
              "max_abs_diff": _f((gw["IMF_IFS"] - gw["OECD_FINMARK"]).abs().max(), 3)
              if len(gw) else None,
              "last_month": _d(gs["month"].max())}

    out = {"generated": str(date.today()), "runtime_s": round(time.time() - t0),
           "breaks": breaks, "audit": audits, "gsec_splice": splice,
           "nifty": nifty, "indices": per_index, "sector_cross_section": xs,
           "n_shift": N_SHIFT}
    os.makedirs(os.path.dirname(STUDY_PATH), exist_ok=True)
    with open(STUDY_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", STUDY_PATH, round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
