"""
Measurements behind the FII-only panel (Index Prediction page), computed from data.

WHY THIS MODULE EXISTS (2026-11-16). Every number the FII-only panel quoted was a
hand-pasted constant in fii_only.py, measured once in Aug-2026 by scratch scripts
that were never kept. Two things then went wrong:
  1. The participant table turned out to hold ~300 days of ANOTHER date's NSE file
     (repaired 2026-11-16, see scripts/repair_participant_copies.py), so every one of
     those constants was measured on partly corrupt data.
  2. The zone cut-points (_ZONE_CUTS) CLASSIFY today's reading and name the stance,
     yet were frozen — a classification constant measured on old data silently
     mislabels new data (the project's own Round-7 rule says: compute it live).

So: the zone table and its cut-points are computed here at call time from data on
or before as_of. The descriptive track records are computed by the same functions,
run by scripts/remeasure_fii_only.py, and pasted back into fii_only.py with a date.

Conventions (identical to the original measurements, stated once):
  * The participant file publishes after the close, so every forward return starts
    at the NEXT session's open:  next day = close[t+1]/open[t+1]-1,
    h sessions = close[t+h]/open[t+1]-1.
  * Close-to-close (cc) next day is reported only as the non-tradeable reference.
  * "hit" = share of directional calls whose sign matched the next return.
  * "naive" = the better of always-up / always-down over the same days.
  * t against a base = (mean_cell - mean_all) / (sd_cell / sqrt(n)), days not episodes
    (as originally measured; overlapping multi-day windows inflate it — noted).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["outcomes", "measure_zones", "live_cuts", "measure_era_split", "measure_track_record",
           "leg_series", "measure_leg_ic", "measure_action_track", "measure_leg_backtest",
           "measure_hedge_test"]

_DIRS = ("cash_net", "fut_share", "fut_net_chg", "opt_bull", "stats_net")


# ───────────────────────────────────────────────────────────── shared inputs
def outcomes(as_of: date) -> pd.DataFrame:
    """Nifty forward returns (%) indexed by session, all starting at the next open."""
    px = query_dataframe("""SELECT trade_date, open_val, close_val FROM index_data
                            WHERE index_name = 'Nifty 50' ORDER BY trade_date""")
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates("trade_date").set_index("trade_date").astype(float)
    o, c = px["open_val"], px["close_val"]
    out = pd.DataFrame(index=px.index)
    out["oc1"] = (c.shift(-1) / o.shift(-1) - 1) * 100
    out["cc1"] = (c.shift(-1) / c - 1) * 100
    for h in (5, 10):
        out[f"w{h}"] = (c.shift(-h) / o.shift(-1) - 1) * 100
    # as_of bounds the SIGNAL date; outcomes after it are allowed only for measurement
    return out


def _composite(as_of: date) -> pd.DataFrame:
    from src.analytics.fii_only import _components_uncached
    Z = _components_uncached(as_of)
    if Z.empty:
        return Z
    dirs = [c for c in _DIRS if c in Z.columns]
    Z["n_live"] = Z[dirs].notna().sum(axis=1)
    Z["raw"] = Z[dirs].mean(axis=1).where(Z["n_live"] >= 3)
    m = Z["raw"].rolling(250, min_periods=120).mean()
    sd = Z["raw"].rolling(250, min_periods=120).std().replace(0.0, np.nan)
    Z["norm"] = (Z["raw"] - m) / sd
    return Z


def _cell(x: pd.Series, base_mean: float) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 3:
        return {"n": n, "mean": float("nan"), "up": float("nan"), "t": float("nan")}
    sd = x.std(ddof=1)
    return {"n": int(n), "mean": round(float(x.mean()), 3), "up": round(float(100 * (x > 0).mean()), 1),
            "t": round(float((x.mean() - base_mean) / (sd / np.sqrt(n))), 2) if sd > 0 else 0.0}


def _ic(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    j = pd.concat([a, b], axis=1).dropna()
    n = len(j)
    if n < 30:
        return float("nan"), float("nan"), n
    r = float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()))
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, t, n


def _hit(sig: pd.Series, ret: pd.Series) -> tuple[float, float, int]:
    j = pd.concat([sig, ret], axis=1).dropna()
    j = j[j.iloc[:, 0] != 0]
    if j.empty:
        return float("nan"), float("nan"), 0
    hit = float(100 * (np.sign(j.iloc[:, 0]) == np.sign(j.iloc[:, 1])).mean())
    up = float(100 * (j.iloc[:, 1] > 0).mean())
    return hit, max(up, 100 - up), len(j)


# ───────────────────────────────────────────────────────────── zones (LIVE)
def live_cuts(as_of: date) -> list[float] | None:
    """Quintile edges of the composition-stable score, from every scored day <= as_of."""
    Z = _composite(as_of)
    if Z.empty:
        return None
    nv = Z["norm"].dropna()
    if len(nv) < 500:
        return None
    return [round(float(v), 4) for v in np.quantile(nv.values, [0.2, 0.4, 0.6, 0.8])]


def measure_zones(as_of: date) -> dict | None:
    """Same shape as fii_only.FII_ZONES, measured walk-forward on data <= as_of.

    Zone labels come from fii_only._zone_series (cuts from strictly earlier data);
    outcomes must have happened on or before as_of to be counted."""
    from src.analytics.fii_only import _zone_series
    zs = _zone_series(as_of)
    if zs.empty or zs["zone"].notna().sum() < 500:
        return None
    oc = outcomes(as_of)
    last = pd.Timestamp(as_of)
    d = zs.join(oc, how="left")
    sess = oc.index[oc.index <= last]
    # an outcome h sessions ahead is only known if that session is <= as_of
    pos = {t: i for i, t in enumerate(sess)}
    known = np.array([pos.get(t, -1) for t in d.index])
    d.loc[known > len(sess) - 2, "oc1"] = np.nan
    d.loc[known > len(sess) - 6, "w5"] = np.nan
    d = d.dropna(subset=["zone"])
    out = {"cuts": live_cuts(as_of)}
    for h, col in (("d1", "oc1"), ("d5", "w5")):
        allx = d[col].dropna()
        out[f"{h}_base"] = {"mean": round(float(allx.mean()), 4), "up": round(float(100 * (allx > 0).mean()), 1)}
        out[h] = {int(z): _cell(d.loc[d["zone"] == z, col], float(allx.mean())) for z in range(1, 6)}
    out["n_days"] = int(d["oc1"].notna().sum())
    out["measured_to"] = str(sess[-1].date()) if len(sess) else None
    out["basis"] = "live"
    return out


# ───────────────────────────────────────────────────────────── era split / track record
def measure_era_split(as_of: date) -> dict:
    Z = _composite(as_of)
    oc = outcomes(as_of)
    d = Z.join(oc, how="left")
    d = d[d.index <= pd.Timestamp(as_of)]
    stance = np.sign(d["raw"].where(d["raw"].abs() >= 0.5, 0.0))
    cash_start = d["cash_net"].first_valid_index() if "cash_net" in d else None
    out = {}
    for key, m, lab in (("era4", d.index < cash_start, "4 components (before FII cash data)"),
                        ("era5", d.index >= cash_start, "5 components (with FII cash data)")):
        sub = d[m & d["raw"].notna()]
        st_ = stance[sub.index]
        hit, _, _ = _hit(st_, sub["oc1"])
        bear = sub.loc[st_ < 0, "oc1"].dropna()
        out[key] = {"label": lab, "n": int(sub["oc1"].notna().sum()),
                    "fires_pct": round(float(100 * (st_ != 0).mean()), 0),
                    "dir_hit": round(hit, 1), "bear_down": round(float(100 * (bear < 0).mean()), 1),
                    "bear_mean": round(float(bear.mean()), 3)}
    out["cash_start"] = str(cash_start.date()) if cash_start is not None else None
    return out


def measure_track_record(as_of: date) -> dict:
    Z = _composite(as_of)
    oc = outcomes(as_of)
    d = Z.join(oc, how="left")
    d = d[(d.index <= pd.Timestamp(as_of)) & d["raw"].notna()]
    st_ = np.sign(d["raw"].where(d["raw"].abs() >= 0.5, 0.0))
    hit_cc, _, _ = _hit(st_, d["cc1"])
    hit_oc, naive, n = _hit(st_, d["oc1"])
    dirn = d[st_ != 0]
    bear, bull = d.loc[st_ < 0, "oc1"].dropna(), d.loc[st_ > 0, "oc1"].dropna()
    w5h, w5n, _ = _hit(st_, d["w5"])
    w10h, w10n, _ = _hit(st_, d["w10"])
    return {
        "n": int(len(d)), "since": str(d.index[0].date()),
        "next_day_cc_hit": round(hit_cc, 1), "next_day_cc_base": round(float(100 * (dirn["cc1"] > 0).mean()), 1),
        "next_day_cc_ic": round(_ic(d["raw"], d["cc1"])[0], 3),
        "next_day_oc_hit": round(hit_oc, 1), "next_day_oc_base": round(float(100 * (dirn["oc1"] > 0).mean()), 1),
        "next_day_oc_ic": round(_ic(d["raw"], d["oc1"])[0], 3),
        "next_day_naive": round(naive, 1),
        "bear_down_rate": round(float(100 * (bear < 0).mean()), 1), "bear_mean_pct": round(float(bear.mean()), 3),
        "bull_up_rate": round(float(100 * (bull > 0).mean()), 1), "bull_mean_pct": round(float(bull.mean()), 3),
        "wk1_hit": round(w5h, 1), "wk1_naive": round(w5n, 1),
        "wk2_hit": round(w10h, 1), "wk2_naive": round(w10n, 1),
        "cost_floor_bps": 50,
    }


# ───────────────────────────────────────────────────────────── six legs
def leg_series(as_of: date) -> pd.DataFrame:
    """net_1d / net_5d exactly as fii_only.get_fii_actions builds them, as series."""
    from src.analytics.fii_only import _LEG_COL, _LEG_SIGN
    p = query_dataframe("""SELECT * FROM fao_participant WHERE client_type='FII' AND data_type='OI'
                           AND trade_date <= ? ORDER BY trade_date""", [as_of])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")
    out = pd.DataFrame(index=p.index)
    out["net_1d"] = 0.0
    out["net_5d"] = 0.0
    for leg, col in _LEG_COL.items():
        raw = p[col].astype(float)
        scale = raw.rolling(60, min_periods=20).mean()
        out[leg] = _LEG_SIGN[leg] * raw.diff() / scale * 100
        out["net_1d"] += out[leg]
        out["net_5d"] += _LEG_SIGN[leg] * raw.diff(5) / scale * 100
    return out


def _option_expiries(as_of: date) -> set:
    ex = query_dataframe("""SELECT DISTINCT expiry_date FROM fno_bhavcopy WHERE instrument='OPTIDX'
                            AND symbol='NIFTY' AND expiry_date <= ?""", [as_of])
    return set(pd.to_datetime(ex["expiry_date"])) if not ex.empty else set()


def measure_leg_ic(as_of: date) -> dict:
    from src.analytics.fii_only import _LEG_COL
    L = leg_series(as_of).join(outcomes(as_of), how="left")
    L = L[L.index <= pd.Timestamp(as_of)]
    ic, tt = {}, {}
    for leg in _LEG_COL:
        r, t, _ = _ic(L[leg], L["oc1"])
        ic[leg], tt[leg] = round(r, 4), round(abs(t), 2)
    r, t, n = _ic(L["net_1d"], L["oc1"])
    Z = _composite(as_of)
    j = pd.concat([L["net_1d"], Z["raw"]], axis=1).dropna()
    rho = float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()))
    s1 = np.sign(j["net_1d"].where(j["net_1d"].abs() >= 5, 0))
    s2 = np.sign(j["raw"].where(j["raw"].abs() >= 0.5, 0))
    disagree = float(100 * ((s1 * s2) < 0).mean())
    return {**ic, "_t": tt, "net_ic": round(r, 4), "net_t": round(abs(t), 2), "n": n,
            "vs_composite_spearman": round(rho, 3), "disagree_pct": round(disagree, 1)}


def measure_action_track(as_of: date) -> dict:
    from src.analytics.fii_only import _LEG_COL
    L = leg_series(as_of).join(outcomes(as_of), how="left")
    L = L[(L.index <= pd.Timestamp(as_of)) & L["net_1d"].notna()]
    s = np.sign(L["net_1d"].where(L["net_1d"].abs() >= 5, 0))
    h1, n1, _ = _hit(s, L["oc1"])
    h5, n5, _ = _hit(s, L["w5"])
    h10, n10, _ = _hit(s, L["w10"])
    bear, bull = L[s < 0], L[s > 0]
    ics = [_ic(L[leg], L["oc1"])[0] for leg in _LEG_COL]
    return {"n": int(len(L)), "d1_oc_hit": round(h1, 1), "d1_oc_naive": round(n1, 1),
            "d1_oc_ic": round(_ic(L["net_1d"], L["oc1"])[0], 3),
            "wk1_hit": round(h5, 1), "wk1_naive": round(n5, 1), "wk2_hit": round(h10, 1), "wk2_naive": round(n10, 1),
            "bear_down_1wk": round(float(100 * (bear["w5"].dropna() < 0).mean()), 1),
            "bear_down_2wk": round(float(100 * (bear["w10"].dropna() < 0).mean()), 1),
            "bull_up_2wk": round(float(100 * (bull["w10"].dropna() > 0).mean()), 1),
            "base_up_2wk": round(float(100 * (L["w10"].dropna() > 0).mean()), 1),
            "leg_ic_range": f"{min(ics):+.3f} to {max(ics):+.3f}"}


def _arm(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    up = float(100 * (x > 0).mean()) if n else float("nan")
    se = np.sqrt(up / 100 * (1 - up / 100) / n) * 100 if n else float("nan")
    return {"n": n, "mean": round(float(x.mean()), 3), "up": round(up, 1),
            "ci": (round(up - 1.96 * se, 1), round(up + 1.96 * se, 1))}


def _welch(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def measure_leg_backtest(as_of: date, years: int = 2) -> dict:
    L = leg_series(as_of).join(outcomes(as_of), how="left")
    end = pd.Timestamp(as_of)
    start = end - pd.DateOffset(years=years)
    full = L[(L.index <= end) & L["net_1d"].notna()]
    W = full[full.index >= start]
    exp = _option_expiries(as_of)
    W = W.assign(expiry=W.index.isin(exp))

    def block(D, thr=5):
        s = np.sign(D["net_1d"].where(D["net_1d"].abs() >= thr, 0))
        bull, bear = D.loc[s > 0, "oc1"], D.loc[s < 0, "oc1"]
        hit, naive, _ = _hit(s, D["oc1"])
        return s, bull, bear, hit, naive

    s, bull, bear, hit, naive = block(W)
    base = W["oc1"].dropna()
    ex_, eo_ = W[~W["expiry"]], W[W["expiry"]]
    _, b2, r2, h2, _ = block(ex_)
    _, b3, r3, h3, _ = block(eo_)
    _, bf, rf, hf, nf = block(full)
    s5 = np.sign(ex_["net_5d"].where(ex_["net_5d"].abs() >= 10, 0))
    return {
        "start": str(W.index[0].date()), "end": str(W.index[-1].date()), "n": int(len(W)),
        "expiry_n": int(W["expiry"].sum()), "expiry_pct": round(float(100 * W["expiry"].mean()), 1),
        "base_mean": round(float(base.mean()), 3), "base_up": round(float(100 * (base > 0).mean()), 1),
        "naive": round(naive, 1), "naive_side": "up" if (base > 0).mean() >= 0.5 else "down",
        "bull": _arm(bull), "bear": _arm(bear),
        "diff": round(float(bull.mean() - bear.mean()), 3), "welch_t": round(_welch(bull, bear), 2), "hit": round(hit, 1),
        "exq": {"n": int(len(ex_)), "diff": round(float(b2.mean() - r2.mean()), 3), "t": round(_welch(b2, r2), 2),
                "hit": round(h2, 1), "bull_mean": round(float(b2.mean()), 3), "bear_mean": round(float(r2.mean()), 3)},
        "expiry_only": {"n": int(len(eo_)), "diff": round(float(b3.mean() - r3.mean()), 3),
                        "t": round(_welch(b3, r3), 2), "hit": round(h3, 1)},
        "thresholds": {k: round(block(ex_, k)[3], 1) for k in (3, 5, 10, 15)},
        "full_hist": {"n": int(len(full)), "diff": round(float(bf.mean() - rf.mean()), 3),
                      "t": round(_welch(bf, rf), 2), "hit": round(hf, 1), "naive": round(nf, 1)},
        "week": {"bull_mean": round(float(W.loc[s > 0, "w5"].mean()), 3),
                 "bull_up": round(float(100 * (W.loc[s > 0, "w5"].dropna() > 0).mean()), 1),
                 "bear_mean": round(float(W.loc[s < 0, "w5"].mean()), 3),
                 "bear_up": round(float(100 * (W.loc[s < 0, "w5"].dropna() > 0).mean()), 1),
                 "t": round(_welch(W.loc[s > 0, "w5"], W.loc[s < 0, "w5"]), 2),
                 "hit": round(_hit(s, W["w5"])[0], 1), "naive": round(_hit(s, W["w5"])[1], 1),
                 "base_mean": round(float(W["w5"].mean()), 3),
                 "base_up": round(float(100 * (W["w5"].dropna() > 0).mean()), 1)},
        "net5": {"bull_n": int((s5 > 0).sum()), "bull_mean": round(float(ex_.loc[s5 > 0, "w5"].mean()), 3),
                 "bull_up": round(float(100 * (ex_.loc[s5 > 0, "w5"].dropna() > 0).mean()), 1),
                 "bear_n": int((s5 < 0).sum()), "bear_mean": round(float(ex_.loc[s5 < 0, "w5"].mean()), 3),
                 "bear_up": round(float(100 * (ex_.loc[s5 < 0, "w5"].dropna() > 0).mean()), 1),
                 "base_mean": round(float(ex_["w5"].mean()), 3),
                 "base_up": round(float(100 * (ex_["w5"].dropna() > 0).mean()), 1)},
    }


def measure_hedge_test(as_of: date) -> dict:
    p = query_dataframe("""SELECT trade_date, fut_idx_short FROM fao_participant WHERE client_type='FII'
                           AND data_type='OI' AND trade_date <= ? ORDER BY trade_date""", [as_of])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    sh = p.drop_duplicates("trade_date").set_index("trade_date")["fut_idx_short"].astype(float)
    short_pct = sh.diff() / sh.rolling(60, min_periods=20).mean() * 100
    c = query_dataframe("SELECT trade_date, fii_net, dii_net FROM fii_dii_cash WHERE trade_date <= ?", [as_of])
    c["trade_date"] = pd.to_datetime(c["trade_date"])
    c = c.drop_duplicates("trade_date").set_index("trade_date")
    d = pd.DataFrame({"short_pct": short_pct}).join(c, how="inner").join(outcomes(as_of), how="left").dropna(
        subset=["short_pct", "fii_net", "oc1"])
    base = d["oc1"]
    states = {}
    for key, m in (("build_sell", (d.short_pct > 0) & (d.fii_net < 0)), ("build_buy", (d.short_pct > 0) & (d.fii_net >= 0)),
                   ("cover_buy", (d.short_pct <= 0) & (d.fii_net >= 0)), ("cover_sell", (d.short_pct <= 0) & (d.fii_net < 0))):
        x = d.loc[m, "oc1"]
        cell = _cell(x, float(base.mean()))
        states[key] = {"n": cell["n"], "oc": cell["mean"], "up": cell["up"], "t": cell["t"],
                       "wk": round(float(d.loc[m, "w5"].mean()), 3)}
    alone = d.loc[d.short_pct > 0, "oc1"]
    both = d.loc[(d.short_pct > 0) & (d.fii_net < 0), "oc1"]
    alone_w = d.loc[d.short_pct > 0, "w5"]
    both_w = d.loc[(d.short_pct > 0) & (d.fii_net < 0), "w5"]
    rho_dii = float(d["fii_net"].rank().corr(d["dii_net"].rank()))
    return {"start": str(d.index[0].date()), "end": str(d.index[-1].date()), "n": int(len(d)),
            "base_oc": round(float(base.mean()), 3), "base_up": round(float(100 * (base > 0).mean()), 1),
            "states": states, "alone_oc": round(float(alone.mean()), 3), "alone_n": int(len(alone)),
            "adds_oc": round(float(both.mean() - alone.mean()), 3), "adds_t": round(_welch(both, alone), 2),
            "adds_wk": round(float(both_w.mean() - alone_w.mean()), 3), "adds_wk_t": round(_welch(both_w, alone_w), 2),
            "rho_dii": round(rho_dii, 3)}


# ───────────────────────────────────────────────────────────── downside tails (LIVE)
def _sigma_moves(as_of: date) -> pd.DataFrame:
    """Forward close-to-close moves in units of the causal EWMA sigma (lambda 0.94)."""
    px = query_dataframe("""SELECT trade_date, close_val FROM index_data
                            WHERE index_name = 'Nifty 50' ORDER BY trade_date""")
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    c = px.drop_duplicates("trade_date").set_index("trade_date")["close_val"].astype(float)
    ret = c.pct_change() * 100
    sg = ret.ewm(alpha=1 - 0.94).std().shift(1)
    last = len(c[c.index <= pd.Timestamp(as_of)]) - 1
    pos = np.arange(len(c))
    z1 = ((c.shift(-1) / c - 1) * 100) / sg
    z5 = ((c.shift(-5) / c - 1) * 100) / sg
    z1[pos > last - 1] = np.nan          # outcome after as_of: unknown
    z5[pos > last - 5] = np.nan
    return pd.DataFrame({"z1": z1, "z5": z5, "sig": sg})


def measure_tails(as_of: date, n_perm: int = 2000, seed: int = 7) -> dict | None:
    """P(forward move below -1 sigma next day / -2 sigma over 5 sessions) by zone,
    with a circular-shift permutation p-value for the Zone-1 excess.
    Same shape as fii_only._FII_TAIL plus the test."""
    from src.analytics.fii_only import _zone_series
    zs = _zone_series(as_of)
    if zs.empty or zs["zone"].notna().sum() < 500:
        return None
    d = zs.join(_sigma_moves(as_of), how="left").dropna(subset=["zone"])
    rng = np.random.default_rng(seed)
    out = {}
    for h, col, thr in (("d1", "z1", -1.0), ("d5", "z5", -2.0)):
        sub = d[["zone", col]].dropna()
        below = (sub[col] < thr).to_numpy()
        zone = sub["zone"].to_numpy()
        row = {"thr": thr, 0: round(float(100 * below.mean()), 1)}
        for z in range(1, 6):
            row[z] = round(float(100 * below[zone == z].mean()), 1)
        obs = below[zone == 1].mean() - below.mean()
        row["q1_excess_pp"] = round(float(100 * obs), 2)
        if n_perm:
            null = np.array([below[np.roll(zone, int(rng.integers(60, len(zone) - 60))) == 1].mean()
                             - below.mean() for _ in range(n_perm)])
            row["q1_perm_p"] = round(float((null >= obs).mean()), 4)
        row["n"] = int(len(sub))
        out[h] = row
    out["measured_to"] = str(pd.Timestamp(as_of).date())
    return out
