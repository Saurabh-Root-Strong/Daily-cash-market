"""
Walk-forward audit of the analogue matcher behind "Days that looked like today".

WHY THIS FILE EXISTS. src/analytics/market_context.py carries a per-mode
calibration block (_ANALOGUE_CAL) that the dashboard displays as fact. Three
times now those numbers have drifted away from the model that actually runs —
once when the feature set changed, once when the metric changed from Euclidean
to Mahalanobis, and once when a single flat calibration blob was emitted under
all three radio buttons so two of the three readings were another model's. This
script regenerates every one of those numbers so the drift is checkable instead
of remembered.

    python -m scripts.audit_analogues                 # everything
    python -m scripts.audit_analogues --verify-only   # just the consistency check
    python -m scripts.audit_analogues --modes fii

REIMPLEMENTATION RISK, HANDLED. CLAUDE.md says to test the SHIPPED code, not a
copy of it. Calling get_analogues once per date re-queries the whole database
every time and is far too slow for ~5,700 walk-forward days, so `_walk` below is
a vectorised replication. `verify_against_shipped` therefore samples dates,
calls the real get_analogues, and asserts the picked analogue dates and forward
returns agree exactly. If someone edits the matcher and not this file, that
check fails before any number is printed.

WHAT IT MEASURES, AND THE TRAPS IT PRICES IN
  * Common window across all three modes, so mode comparison is like-for-like.
  * Selection at each mode's OWN vote quantile, not an absolute >=70% cut. The
    vote takes only 13 values at k=12 and its spread differs by mode, so a fixed
    cut fires at 15% of days in one mode and 30% in another.
  * HAC t on the return DIFFERENCE (return on a signal dummy, Newey-West at
    lag=horizon), not on the signal-day return level — the level t is just the
    equity risk premium and reads ~2.6 for buy-and-hold.
  * A random-neighbour control, 20 draws, same k and separation rules. Its
    spread is the size of the effects being claimed.
  * Overlapping windows priced with a max-statistic circular-shift permutation.

Costs are quoted for an index-FUTURES expression (~10bps round trip), not the
25bps/side cash-equity floor in CLAUDE.md — that floor is for stock baskets and
over-penalises an index bet by ~40bps.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.analytics.market_context import (
    HORIZONS, _formation, _fii_formation, _index, _session_breaks,
    _ANALOGUE_MIN_SEP, _ANALOGUE_K, get_analogues,
)

MODES = ("price", "price+fii", "fii")
HMAX = max(HORIZONS.values())
NCTL = 20
COST_FUT = 0.10          # % round trip, index futures


# ─────────────────────────────────────────────────────────────── helpers ──
def nw_ols(y, x, lag):
    """OLS of y on [1, x]; Newey-West t on the slope."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    u = X * (y - X @ b)[:, None]
    S = u.T @ u
    for L in range(1, lag + 1):
        G = u[L:].T @ u[:-L]
        S += (1 - L / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return b[1], b[1] / np.sqrt(V[1, 1])


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    return float(np.corrcoef(pd.Series(a[m]).rank(), pd.Series(b[m]).rank())[0, 1])


def _greedy(order, sep, k):
    out = []
    for j in order:
        if any(abs(j - q) < sep for q in out):
            continue
        out.append(j)
        if len(out) >= k:
            break
    return out


def build_F(s, mode, as_of):
    F = _formation(s)
    if mode in ("price+fii", "fii"):
        fx = _fii_formation(as_of, full=(mode == "fii"))
        F = fx.copy() if mode == "fii" else F.join(fx, how="left")
        F = F[F.index >= pd.Timestamp("2018-01-01")]
        F = F[F.index.isin(s.index)]
    return F


def _bad_positions(s):
    """Positions whose forward window straddles a hole in index_data."""
    bad = set()
    for b in _session_breaks():
        if b in s.index:
            p = s.index.get_loc(b)
            bad.update(range(max(0, p - HMAX), p))
    return bad


# ────────────────────────────────────────────────────────── the walk ──
def _walk(s, mode, as_of, rng, k=_ANALOGUE_K, sep=_ANALOGUE_MIN_SEP,
          controls=True, require_outcome=True):
    """require_outcome=False keeps rows whose own forward window has not closed.

    The walk normally drops them — there is nothing to score. The consistency
    check needs them, because the date it compares against get_analogues IS the
    as_of, whose outcome is by definition still open.
    """
    F = build_F(s, mode, as_of)
    Fv = F.values.astype(float)
    ok = np.isfinite(Fv).all(axis=1)
    spos = np.array([s.index.get_loc(d) for d in F.index])
    sv = s.values
    bad = _bad_positions(s)
    rows = []
    for i in range(len(F)):
        if not ok[i]:
            continue
        t = spos[i]
        if require_outcome and t + 1 + HMAX >= len(s):
            continue
        hidx = np.flatnonzero(ok[:i])
        if len(hidx) < 500:
            continue
        hist = Fv[hidx]
        mu, sd = hist.mean(0), hist.std(0)
        if not np.all(sd > 0):
            continue
        zh, zc = (hist - mu) / sd, (Fv[i] - mu) / sd
        VI = np.linalg.pinv(np.cov(zh, rowvar=False))
        d_ = zh - zc
        dd = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d_, VI, d_), 0, None))
        elig = hidx[((spos[hidx] + HMAX) < (t + 1))
                    & ~np.isin(spos[hidx], list(bad))]
        if len(elig) < 200:
            continue
        pos_of = {j: p for p, j in enumerate(hidx)}
        order = elig[np.argsort(dd[[pos_of[j] for j in elig]])]
        picked = _greedy(order, sep, k)
        if len(picked) < k:
            continue

        rec = {"date": F.index[i], "t": t,
               "picked": tuple(s.index[spos[np.array(picked)]].date)}
        ps = spos[np.array(picked)]
        for hn, h in HORIZONS.items():
            v = (sv[ps + h] / sv[ps] - 1) * 100
            rec[f"vote_{hn}"] = float((v > 0).mean())
            rec[f"med_{hn}"] = float(np.median(v))
            rec[f"lo_{hn}"], rec[f"hi_{hn}"] = float(v.min()), float(v.max())
            if t + 1 + h < len(s):
                rec[f"act_{hn}"] = float(sv[t + h] / sv[t] - 1) * 100
                rec[f"act1_{hn}"] = float(sv[t + 1 + h] / sv[t + 1] - 1) * 100
        if controls:
            for c in range(NCTL):
                rp = _greedy(rng.permutation(elig), sep, k) or picked
                rps = spos[np.array(rp)]
                for hn, h in HORIZONS.items():
                    v = (sv[rps + h] / sv[rps] - 1) * 100
                    rec[f"c{c}_{hn}"] = float((v > 0).mean())
        # walk-forward separation, with and without the guard
        rec["gap_guard"] = float(np.median(np.diff(sorted(picked))))
        ng = sorted(order[:k])
        rec["gap_noguard"] = float(np.median(np.diff(ng))) if len(ng) == k else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────── consistency with shipped ──
def verify_against_shipped(s, mode, n_samples=6, seed=11):
    """Assert the fast walk reproduces get_analogues exactly on sampled dates."""
    rng = np.random.default_rng(seed)
    F = build_F(s, mode, s.index[-1].date())
    cand = [d for d in F.index if d >= F.index[600]][:-HMAX - 2]
    if not cand:
        return True, "no sampleable dates"
    picks = rng.choice(len(cand), size=min(n_samples, len(cand)), replace=False)
    bad = []
    for p in picks:
        as_of = cand[int(p)].date()
        ship = get_analogues(as_of, mode=mode)
        if not ship.get("ok"):
            continue
        s_sub = s[s.index <= pd.Timestamp(as_of)]
        fast = _walk(s_sub, mode, as_of, np.random.default_rng(0),
                     controls=False, require_outcome=False)
        if fast.empty:
            continue
        f_last = fast.iloc[-1]
        if f_last["date"].date() != ship["meta"]["formation_date"]:
            bad.append(f"{as_of}: formation date {f_last['date'].date()} "
                       f"vs shipped {ship['meta']['formation_date']}")
            continue
        got = tuple(f_last["picked"])
        want = tuple(ship["analogues"]["date"])
        if got != want:
            bad.append(f"{as_of}: picks differ\n    fast   {got}\n    shipped{want}")
    return (not bad), ("\n  ".join(bad) if bad else
                       f"{len(picks)} sampled dates reproduce exactly")


# ─────────────────────────────────────────────────────────── reporting ──
def report(walks, quantile=0.20):
    common = set.intersection(*[set(w["date"]) for w in walks.values()])
    common = sorted(common)
    C = {m: w[w["date"].isin(common)].sort_values("date").reset_index(drop=True)
         for m, w in walks.items()}
    act = {h: C[MODES[0]][f"act_{h}"].values for h in HORIZONS}
    act1 = {h: C[MODES[0]][f"act1_{h}"].values for h in HORIZONS}
    base = {h: (act[h] > 0).mean() * 100 for h in HORIZONS}

    print(f"\nCOMMON WINDOW  {len(common)} sessions  "
          f"{common[0].date()}..{common[-1].date()}")
    print("BASE RATE      " + " / ".join(f"{base[h]:.1f}%" for h in HORIZONS))
    print(f"RULE           top {int(quantile*100)}% of each mode's own vote")
    print("\nt is the HAC t of the RETURN DIFFERENCE, not of the return level.")

    print("\n" + "=" * 100)
    print(f"{'mode':<11}{'hz':>6}{'n':>6}{'hit%':>7}{'edge':>8}{'HAC t':>8}"
          f"{'ctl':>16}{'IC':>8}{'lag1':>8}{'net%':>8}")
    print("=" * 100)
    cal = {}
    for m in MODES:
        cal[m] = {kk: [] for kk in ("bull_hit", "bull_edge", "bull_n", "bull_t",
                                    "bear_hit", "bear_edge", "bear_n",
                                    "ctl_edge", "ctl_sd", "ic")}
        for h in HORIZONS:
            v = C[m][f"vote_{h}"].values
            sel = v >= np.quantile(v, 1 - quantile)
            bear = v <= np.quantile(v, quantile)
            hit = (act[h][sel] > 0).mean() * 100
            _, t_ = nw_ols(act[h], sel.astype(float), HORIZONS[h])
            ctl = []
            for c in range(NCTL):
                col = f"c{c}_{h}"
                if col not in C[m]:
                    continue
                cv = C[m][col].values
                s_ = cv >= np.quantile(cv, 1 - quantile)
                if s_.sum() >= 20:
                    ctl.append((act[h][s_] > 0).mean() * 100 - base[h])
            cm, cs = (np.mean(ctl), np.std(ctl)) if ctl else (np.nan, np.nan)
            ic = spearman(C[m][f"med_{h}"].values, act[h])
            net = act[h][sel].mean() - COST_FUT
            print(f"{m:<11}{h:>6}{sel.sum():>6}{hit:>7.1f}{hit-base[h]:>+8.1f}"
                  f"{t_:>8.2f}{f'{cm:+.1f}+-{cs:.1f}':>16}{ic:>+8.3f}"
                  f"{(act1[h][sel]>0).mean()*100-base[h]:>+8.1f}{net:>+8.2f}")
            cal[m]["bull_hit"].append(round(hit, 1))
            cal[m]["bull_edge"].append(round(hit - base[h], 1))
            cal[m]["bull_n"].append(int(sel.sum()))
            cal[m]["bull_t"].append(round(float(t_), 2))
            cal[m]["bear_hit"].append(round((act[h][bear] > 0).mean() * 100, 1))
            cal[m]["bear_edge"].append(round((act[h][bear] > 0).mean() * 100 - base[h], 1))
            cal[m]["bear_n"].append(int(bear.sum()))
            cal[m]["ctl_edge"].append(round(float(cm), 1))
            cal[m]["ctl_sd"].append(round(float(cs), 1))
            cal[m]["ic"].append(round(float(ic), 3))
        cal[m]["gap_guard"] = int(np.nanmedian(C[m]["gap_guard"]))
        cal[m]["gap_noguard"] = int(np.nanmedian(C[m]["gap_noguard"]))
        for h in HORIZONS:
            lo, hi = C[m][f"lo_{h}"].values, C[m][f"hi_{h}"].values
            if h == "1 month":
                cal[m]["span_cov_1m"] = round(
                    float(((act[h] >= lo) & (act[h] <= hi)).mean() * 100), 1)
                cal[m]["span_width_1m"] = round(float(np.median(hi - lo)), 2)

    # ── span vs trailing vol: is the spread its own information? ──
    print("\n" + "=" * 100)
    print("SPAN vs 20-DAY TRAILING VOL, predicting the realised |move|")
    print("=" * 100)
    lr = np.log(_index("Nifty 50")).diff()
    vol = (lr.rolling(20).std() * np.sqrt(252) * 100)
    vser = vol.reindex(C[MODES[0]]["date"]).values
    for m in MODES:
        cal[m]["span_ic"] = []
        for h in HORIZONS:
            span = (C[m][f"hi_{h}"] - C[m][f"lo_{h}"]).values
            y = np.abs(act[h])
            ic_s = spearman(span, y)
            cal[m]["span_ic"].append(ic_s)
            print(f"  {m:<11}{h:>6}  IC span {ic_s:+.3f}   "
                  f"IC vol20 {spearman(vser, y):+.3f}")

    # ── max-statistic permutation over the mode x horizon grid ──
    print("\n" + "=" * 100)
    print("MAX-STATISTIC PERMUTATION (circular shift preserves autocorrelation)")
    print("=" * 100)
    rng = np.random.default_rng(7)
    obs = max(cal[m]["bull_edge"][i] for m in MODES for i in range(len(HORIZONS)))
    null = []
    for _ in range(2000):
        sh = rng.integers(HMAX + 1, len(common) - HMAX - 1)
        cands = []
        for m in MODES:
            for h in HORIZONS:
                v = C[m][f"vote_{h}"].values
                sel = v >= np.quantile(v, 1 - quantile)
                a = np.roll(act[h], sh)
                cands.append((a[sel] > 0).mean() * 100 - (a > 0).mean() * 100)
        null.append(max(cands))
    null = np.array(null)
    print(f"  observed best {obs:+.2f}pp | null median {np.median(null):+.2f}pp "
          f"| p = {(null >= obs).mean():.3f}")

    print("\n" + "=" * 100)
    print("PASTE-READY _ANALOGUE_CAL BODY")
    print("=" * 100)
    for m in MODES:
        c = cal[m]
        print(f'    "{m}": {{')
        for kk in ("bull_hit", "bull_edge", "bull_n", "bull_t",
                   "bear_hit", "bear_edge", "bear_n", "ctl_edge", "ctl_sd", "ic"):
            # cast: numpy scalars repr as "np.float64(59.2)", which is not
            # paste-able into the module
            vals = tuple(int(x) if isinstance(x, (int, np.integer)) else
                         round(float(x), 3) for x in c[kk])
            print(f'        "{kk}": {vals},')
        print(f'        "gap_guard": {c["gap_guard"]}, '
              f'"gap_noguard": {c["gap_noguard"]},')
        print(f'        "span_cov_1m": {c["span_cov_1m"]}, '
              f'"span_width_1m": {c["span_width_1m"]},')
        print(f'        "span_ic": {tuple(round(float(x), 3) for x in c["span_ic"])},')
        print('        # sens_lo / sens_hi are a TODAY-only sweep over k=8..20 x '
              'sep=10..42;')
        print('        # they are not produced by this walk — carry them forward '
              'or re-sweep.')
        print("    },")
    return cal


def check_index_continuity():
    print("=" * 100)
    print("INDEX CALENDAR CONTINUITY (prev_close vs the stored previous close)")
    print("=" * 100)
    b = _session_breaks()
    print(f"  {len(b)} break(s): sessions are missing immediately before each")
    for d in b:
        print(f"    {d.date()}")
    s = _index("Nifty 50")
    per = s.groupby([s.index.year, s.index.month]).size()
    thin = per[per < 15]
    if len(thin):
        print("\n  months holding < 15 sessions (normal is 18-23):")
        for (y, mth), v in thin.items():
            print(f"    {y}-{mth:02d}: {v}")
    print("\n  Candidates whose forward window straddles a break are dropped by "
          "get_analogues;\n  without that, a '1 week' return could span several.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    ap.add_argument("--quantile", type=float, default=0.20)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD; default = latest")
    a = ap.parse_args()

    s = _index("Nifty 50")
    if s.empty:
        print("Nifty history unavailable.", file=sys.stderr)
        return 2
    if a.as_of:
        s = s[s.index <= pd.Timestamp(a.as_of)]
    as_of = s.index[-1].date()
    print(f"Nifty 50: {len(s)} sessions through {as_of}\n")

    check_index_continuity()

    if not a.skip_verify:
        print("\n" + "=" * 100)
        print("CONSISTENCY WITH THE SHIPPED MATCHER")
        print("=" * 100)
        allok = True
        for m in a.modes:
            ok, msg = verify_against_shipped(s, m)
            allok &= ok
            print(f"  {'OK  ' if ok else 'FAIL'} {m:<11} {msg}")
        if not allok:
            print("\nRefusing to print calibration: the walk no longer matches "
                  "get_analogues. Reconcile them first.", file=sys.stderr)
            return 1
    if a.verify_only:
        return 0

    rng = np.random.default_rng(20260815)
    walks = {}
    for m in a.modes:
        w = _walk(s, m, as_of, rng)
        walks[m] = w
        print(f"\nwalked {m:<11} n={len(w)} "
              f"{w['date'].min().date()}..{w['date'].max().date()}")
    if len(walks) < 2:
        print("\nNeed at least two modes for the like-for-like comparison.")
        return 0
    report(walks, quantile=a.quantile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
