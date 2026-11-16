"""
Do participant positioning RATIOS mark Nifty TURNING POINTS?  (2018-2026)

The question (user, 2026-11-14): FIIs "rule the market" and trade against Clients, so
some ratio -- long/short, put/call, option posture -- should hit a recognisable level
or reverse BEFORE the index tops or bottoms (e.g. 3-Aug-2026 top, 8-Oct-2026 bottom).
Find that ratio and its threshold.

This is a different test from the earlier direction studies (study_fii_positioning.py):
those asked "does the level forecast the next N days"; this asks "is there a
recognisable state AT / JUST BEFORE swing turns, and can it be acted on".

Guards carried over from every previous positioning study in this repo:
  * ratios only (lot size changed 5 times; contract counts are not comparable)
  * every percentile is CAUSAL (trailing 250 sessions)
  * the file publishes after the close -> a D-dated reading is tradable at D+1 OPEN
  * signals are counted as EPISODES (first fire, then a cooldown), not days
  * the whole family is priced with one max-statistic permutation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src.data.repository import query_dataframe             # noqa: E402

PCT_WIN, PCT_MIN = 250, 120
COOLDOWN = 20
HOR = (5, 10, 20)
ZZ = (0.05, 0.03, 0.08)          # zigzag reversal thresholds; first is the headline
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    fao = query_dataframe("""
        SELECT trade_date, client_type,
               fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short,
               opt_idx_call_long, opt_idx_call_short,
               opt_idx_put_long,  opt_idx_put_short
        FROM fao_participant WHERE upper(data_type) = 'OI'""")
    fao["trade_date"] = pd.to_datetime(fao["trade_date"])
    f = {}
    for ct, g in fao.groupby("client_type"):
        k = ct.lower()
        g = (g.drop(columns="client_type").drop_duplicates("trade_date")
              .set_index("trade_date").sort_index().astype(float))
        fl, fs = g["fut_idx_long"], g["fut_idx_short"]
        cl_, cs = g["opt_idx_call_long"], g["opt_idx_call_short"]
        pl, ps = g["opt_idx_put_long"], g["opt_idx_put_short"]
        f[f"{k}_fut_ls"] = fl / (fl + fs).replace(0, np.nan)            # long share
        f[f"{k}_opt_dir"] = (cl_ + ps) / (cl_ + cs + pl + ps).replace(0, np.nan)
        if k == "fii":
            f["fii_call_vs_put_long"] = cl_ / (cl_ + pl).replace(0, np.nan)   # 1-PCR(long)
            f["fii_put_write_share"] = ps / (ps + cs).replace(0, np.nan)
            f["fii_stk_ls"] = g["fut_stk_long"] / (g["fut_stk_long"]
                                                   + g["fut_stk_short"]).replace(0, np.nan)
            f["fii_all_bull"] = (fl + cl_ + ps) / (fl + fs + cl_ + cs + pl + ps).replace(0, np.nan)
    F = pd.DataFrame(f).sort_index()
    F["fii_minus_client"] = F["fii_fut_ls"] - F["client_fut_ls"]
    F = F.drop(columns=["dii_opt_dir"], errors="ignore")    # DII barely trades index options

    n = query_dataframe("""SELECT trade_date, open_val, high_val, low_val, close_val
                           FROM index_data WHERE index_name = 'Nifty 50'
                           ORDER BY trade_date""")
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    n = n.set_index("trade_date").sort_index().astype(float)
    # fao_participant carries rows dated on market HOLIDAYS (82 of them, 2018-2026) whose
    # contents are junk -- e.g. 2022-10-05: FII long share 73.8% between 19.3% and 19.7%.
    # Keep real sessions only, or every diff / percentile / extreme is contaminated.
    F = F[F.index.isin(n.index)]
    return F, n


def causal_pct(s: pd.Series) -> pd.Series:
    return s.rolling(PCT_WIN, min_periods=PCT_MIN).apply(
        lambda x: (x[-1] >= x).mean() * 100.0, raw=True)


def zigzag(close: pd.Series, thr: float) -> pd.DataFrame:
    """Hindsight swing turns on closes: a turn is a close extreme followed by >= thr reversal."""
    c = close.values
    turns, direction = [], 0
    hi_i = lo_i = 0
    for i in range(1, len(c)):
        if c[i] > c[hi_i]:
            hi_i = i
        if c[i] < c[lo_i]:
            lo_i = i
        if direction >= 0 and c[i] <= c[hi_i] * (1 - thr) and hi_i > (turns[-1][0] if turns else -1):
            turns.append((hi_i, "TOP"))
            direction, lo_i = -1, i
        elif direction <= 0 and c[i] >= c[lo_i] * (1 + thr) and lo_i > (turns[-1][0] if turns else -1):
            turns.append((lo_i, "BOTTOM"))
            direction, hi_i = 1, i
    out = pd.DataFrame(turns, columns=["i", "kind"])
    out["date"] = close.index[out["i"]]
    out["close"] = c[out["i"]]
    return out


def episodes(mask: np.ndarray, cooldown: int = COOLDOWN) -> np.ndarray:
    out = np.zeros(len(mask), bool)
    last = -10 ** 9
    for i in np.flatnonzero(mask):
        if i - last >= cooldown:
            out[i] = True
            last = i
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    F, n = load()
    cl, op = n["close_val"], n["open_val"]
    feats = list(F.columns)
    P = pd.DataFrame({c: causal_pct(F[c]) for c in feats})
    D5 = F.diff(5)
    idx = F.index
    close_f = cl.reindex(idx)

    fwd = {h: ((cl.shift(-h) / op.shift(-1) - 1) * 100).reindex(idx) for h in HOR}
    past5 = ((cl / cl.shift(5) - 1) * 100).reindex(idx)
    past20 = ((cl / cl.shift(20) - 1) * 100).reindex(idx)

    print(f"sample {idx[0].date()} -> {idx[-1].date()}  n={len(idx)}  features={len(feats)}")
    for h in HOR:
        r = fwd[h].dropna()
        print(f"  base tradable fwd {h:>2}d: mean {r.mean():+.3f}%  up {100*(r>0).mean():.1f}%")

    # ---------------------------------------------------------------- A. state AT turns
    for thr in ZZ:
        T = zigzag(cl.loc[idx[0]:], thr)
        T = T[T["date"].isin(idx)]
        tops, bots = T[T.kind == "TOP"], T[T.kind == "BOTTOM"]
        print(f"\n{'='*100}\nA. STATE AT HINDSIGHT TURNS  (zigzag {thr:.0%}: {len(tops)} tops, {len(bots)} bottoms)")
        print("   mean causal 250d percentile of each ratio on the turn day (50 = nothing special).")
        print("   p = circular-shift permutation of the turn dates, two-sided.\n")
        print(f"   {'ratio':<22}{'TOPS pct':>9}{'p':>7}{'%>=80':>7} |{'BOTS pct':>9}{'p':>7}{'%<=20':>7} | top-bot")
        pos = {d: i for i, d in enumerate(idx)}
        for c in feats:
            p = P[c].values
            row = []
            for sub in (tops, bots):
                ii = np.array([pos[d] for d in sub["date"]])
                obs = np.nanmean(p[ii])
                null = []
                for _ in range(2000):
                    k = RNG.integers(60, len(p) - 60)
                    null.append(np.nanmean(p[(ii + k) % len(p)]))
                null = np.array(null)
                pv = (np.abs(null - np.nanmean(null)) >= abs(obs - np.nanmean(null))).mean()
                row.append((obs, pv, p[ii]))
            t80 = 100 * np.nanmean(row[0][2] >= 80)
            b20 = 100 * np.nanmean(row[1][2] <= 20)
            print(f"   {c:<22}{row[0][0]:>9.1f}{row[0][1]:>7.3f}{t80:>6.0f}% |"
                  f"{row[1][0]:>9.1f}{row[1][1]:>7.3f}{b20:>6.0f}% | {row[0][0]-row[1][0]:+6.1f}")
        if thr == ZZ[0]:
            T5 = T.copy()

    # ---------------------------------------------------------------- B. lead or lag?
    print(f"\n{'='*100}\nB. DOES THE RATIO MOVE BEFORE PRICE OR AFTER IT?")
    print("   spearman of the ratio's 5-day change with the PAST 5d Nifty move vs the NEXT 5d (tradable).\n")
    print(f"   {'ratio':<22}{'vs PAST 5d':>11}{'vs NEXT 5d':>11}{'level-pct vs PAST 20d':>23}")
    for c in feats:
        a = pd.concat([D5[c], past5, fwd[5], P[c], past20], axis=1).dropna()
        a.columns = ["d", "p", "f", "lv", "p20"]
        a = a.rank()                                        # spearman without scipy
        print(f"   {c:<22}{a['d'].corr(a['p']):>+11.3f}"
              f"{a['d'].corr(a['f']):>+11.3f}"
              f"{a['lv'].corr(a['p20']):>+23.3f}")

    print("\n   event profile: mean percentile at offsets around 5% turns (sessions; - = before the turn)")
    offs = (-20, -10, -5, -2, 0, 2, 5, 10, 20)
    pos = {d: i for i, d in enumerate(idx)}
    for kind in ("TOP", "BOTTOM"):
        ii = np.array([pos[d] for d in T5[T5.kind == kind]["date"]])
        print(f"\n   {kind}S (n={len(ii)})   " + "".join(f"{o:>+7d}" for o in offs))
        for c in feats:
            p = P[c].values
            vals = []
            for o in offs:
                j = ii + o
                j = j[(j >= 0) & (j < len(p))]
                vals.append(np.nanmean(p[j]))
            print(f"   {c:<22}" + "".join(f"{v:>7.1f}" for v in vals))

    # ---------------------------------------------------------------- C. turn RULES
    print(f"\n{'='*100}\nC. TURN RULES -- can any state be ACTED on?  (entry D+1 open, episodes, cooldown {COOLDOWN})")
    sigs: dict[str, np.ndarray] = {}
    for c in feats:
        p = P[c]
        d5 = D5[c]
        lo10 = p.rolling(10).min()
        hi10 = p.rolling(10).max()
        sigs[f"{c}|LOW<=10"] = episodes((p <= 10).fillna(False).values)
        sigs[f"{c}|HIGH>=90"] = episodes((p >= 90).fillna(False).values)
        sigs[f"{c}|LOW&TURNING-UP"] = episodes(((lo10 <= 15) & (d5 > 0) & (p >= lo10 + 10)).fillna(False).values)
        sigs[f"{c}|HIGH&TURNING-DN"] = episodes(((hi10 >= 85) & (d5 < 0) & (p <= hi10 - 10)).fillna(False).values)

    R = {h: fwd[h].values for h in HOR}
    base = {h: np.nanmean(R[h]) for h in HOR}
    sd = {h: np.nanstd(R[h]) for h in HOR}

    def cell(mask: np.ndarray, r: np.ndarray, b: float):
        x = r[mask]
        x = x[~np.isnan(x)]
        if len(x) < 8:
            return None
        ex = x.mean() - b
        t = ex / (x.std(ddof=1) / np.sqrt(len(x)))
        return len(x), x.mean(), 100 * (x > 0).mean(), ex, t

    rows = []
    for name, m in sigs.items():
        for h in (10, 20):
            c_ = cell(m, R[h], base[h])
            if c_:
                rows.append((name, h, *c_))
    res = pd.DataFrame(rows, columns=["signal", "h", "n", "mean", "up%", "excess", "t"])
    res["abs_t"] = res["t"].abs()
    print(f"   cells tested: {len(res)}   base 10d {base[10]:+.2f}%  20d {base[20]:+.2f}%")
    print("\n   TOP 15 by |t| (excess = mean fwd return minus the unconditional mean):")
    print(res.sort_values("abs_t", ascending=False).head(15)
          .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    # family-wise: shift returns against the signal matrix
    obs_best = res["abs_t"].max()
    names = list(sigs)
    M = np.array([sigs[k] for k in names])
    null_best = []
    for _ in range(1000):
        k = RNG.integers(60, len(idx) - 60)
        best = 0.0
        for h in (10, 20):
            r = np.roll(R[h], k)
            for m in M:
                c_ = cell(m, r, base[h])
                if c_ and abs(c_[4]) > best:
                    best = abs(c_[4])
        null_best.append(best)
    null_best = np.array(null_best)
    print(f"\n   FAMILY-WISE: observed best |t| {obs_best:.2f}   null best median "
          f"{np.median(null_best):.2f}, 95th {np.percentile(null_best, 95):.2f}   "
          f"p = {(null_best >= obs_best).mean():.3f}")

    # turn capture: does a signal sit near a real turn more often than chance?
    print("\n   TURN CAPTURE (5% zigzag): share of signal episodes with a matching turn within "
          "-3..+10 sessions,\n   vs the same signals circularly shifted (chance).")
    pos = {d: i for i, d in enumerate(idx)}
    top_i = np.array([pos[d] for d in T5[T5.kind == "TOP"]["date"]])
    bot_i = np.array([pos[d] for d in T5[T5.kind == "BOTTOM"]["date"]])

    def capture(sig_i: np.ndarray, turn_i: np.ndarray) -> float:
        if len(sig_i) == 0:
            return np.nan
        d = turn_i[None, :] - sig_i[:, None]
        return 100 * ((d >= -3) & (d <= 10)).any(axis=1).mean()

    print(f"   {'signal':<40}{'n':>4}{'->BOTTOM':>10}{'chance':>8}{'p':>7}{'->TOP':>8}{'chance':>8}{'p':>7}")
    cap_rows = []
    for name, m in sigs.items():
        si = np.flatnonzero(m)
        if len(si) < 8:
            continue
        ob, ot = capture(si, bot_i), capture(si, top_i)
        nb, nt = [], []
        for _ in range(1000):
            k = RNG.integers(60, len(idx) - 60)
            sh = (si + k) % len(idx)
            nb.append(capture(sh, bot_i))
            nt.append(capture(sh, top_i))
        nb, nt = np.array(nb), np.array(nt)
        cap_rows.append((name, len(si), ob, nb.mean(), (nb >= ob).mean(), ot, nt.mean(), (nt >= ot).mean()))
    cap = pd.DataFrame(cap_rows, columns=["signal", "n", "bot", "bot_ch", "bot_p", "top", "top_ch", "top_p"])
    cap["best_p"] = cap[["bot_p", "top_p"]].min(axis=1)
    for r in cap.sort_values("best_p").head(14).itertuples():
        print(f"   {r.signal:<40}{r.n:>4}{r.bot:>9.0f}%{r.bot_ch:>7.0f}%{r.bot_p:>7.3f}"
              f"{r.top:>7.0f}%{r.top_ch:>7.0f}%{r.top_p:>7.3f}")
    print(f"   ({len(cap)} signals x 2 turn types = {2*len(cap)} tests; Bonferroni p < {0.05/(2*len(cap)):.4f})")

    # ---------------------------------------------------------------- D. the 2026 case
    print(f"\n{'='*100}\nD. THE USER'S CASE: 3-Aug-2026 top -> 8-Oct-2026 bottom -> rally")
    show = ["fii_fut_ls", "client_fut_ls", "pro_fut_ls", "fii_opt_dir", "fii_call_vs_put_long",
            "fii_all_bull", "fii_minus_client"]
    w = pd.DataFrame({"nifty": close_f})
    for c in show:
        w[c] = F[c] * 100
        w[c + "_p"] = P[c]
    w = w.loc["2026-07-01":]
    key = [d for d in ("2026-08-03", "2026-10-08") if pd.Timestamp(d) in w.index]
    pick = sorted(set(list(w.index[::4]) + [pd.Timestamp(k) for k in key] + [w.index[-1]]))
    print("   value% (causal 250d percentile)")
    print("   date        nifty   " + "".join(f"{c[:17]:>19}" for c in show))
    for d in pick:
        r = w.loc[d]
        tag = "  <-- TOP" if str(d.date()) == "2026-08-03" else ("  <-- BOTTOM" if str(d.date()) == "2026-10-08" else "")
        print(f"   {d.date()} {r['nifty']:>8.0f} " +
              "".join(f"{r[c]:>11.1f} ({r[c+'_p']:>3.0f})" for c in show) + tag)

    print("\n   signals that fired 2026-07-01 onward:")
    for name, m in sigs.items():
        ds = [str(idx[i].date()) for i in np.flatnonzero(m) if idx[i] >= pd.Timestamp("2026-07-01")]
        if ds:
            print(f"   {name:<40} {', '.join(ds)}")

    print(f"\n   all 5% turns since 2018 with the three headline percentiles on the turn day:")
    print("   date        kind    nifty   fii_fut_ls  client_fut_ls  fii_opt_dir  fii_all_bull")
    for r in T5.itertuples():
        if r.date in P.index:
            q = P.loc[r.date]
            print(f"   {r.date.date()}  {r.kind:<7}{r.close:>7.0f}{q['fii_fut_ls']:>11.0f}"
                  f"{q['client_fut_ls']:>14.0f}{q['fii_opt_dir']:>13.0f}{q['fii_all_bull']:>14.0f}")




# ============================================================================ E / F / G
def extra() -> None:
    """Divergence test, the full scenario matrix, and the expiry-roll artefact check."""
    F, n = load()
    cl, op = n["close_val"], n["open_val"]
    idx = F.index
    c = cl.reindex(idx)
    fwd = {h: ((cl.shift(-h) / op.shift(-1) - 1) * 100).reindex(idx) for h in (10, 20)}
    base = {h: fwd[h].mean() for h in fwd}
    T = zigzag(cl.loc[idx[0]:], 0.05)
    pos = {d: i for i, d in enumerate(idx)}
    bot_i = np.array([pos[d] for d in T[T.kind == "BOTTOM"]["date"] if d in pos])
    top_i = np.array([pos[d] for d in T[T.kind == "TOP"]["date"] if d in pos])

    def near(i_arr, turn_i):
        if len(i_arr) == 0:
            return np.nan
        d = turn_i[None, :] - i_arr[:, None]
        return 100 * ((d >= -3) & (d <= 10)).any(axis=1).mean()

    def stats(mask, h=10):
        e = episodes(np.asarray(mask, bool), 10)
        x = fwd[h].values[e]
        x = x[~np.isnan(x)]
        ii = np.flatnonzero(e)
        if len(x) < 5:
            return None
        return dict(days=int(np.sum(mask)), n=len(x), mean=x.mean(), up=100 * (x > 0).mean(),
                    ex=x.mean() - base[h], t=(x.mean() - base[h]) / (x.std(ddof=1) / np.sqrt(len(x))),
                    bot=near(ii, bot_i), top=near(ii, top_i), x=x)

    print(f"\n{'='*100}\nE. DIVERGENCE -- 'they changed position BEFORE price turned'")
    print("   price at a 20-session closing low while the ratio is already OFF its own 20-session low")
    print("   (bullish divergence), tested against the SIBLING: price at the low and the ratio at its low too.\n")
    lo20 = c <= c.rolling(20).min() * 1.002
    hi20 = c >= c.rolling(20).max() * 0.998
    chance_b = 100 * np.mean([((bot_i - i >= -3) & (bot_i - i <= 10)).any() for i in range(len(idx))])
    chance_t = 100 * np.mean([((top_i - i >= -3) & (top_i - i <= 10)).any() for i in range(len(idx))])
    print(f"   any-day chance of a 5% bottom within -3..+10 sessions: {chance_b:.0f}%   top: {chance_t:.0f}%")
    print(f"   {'ratio':<20}{'setup':<34}{'n':>4}{'fwd10':>8}{'up%':>6}{'turn<=10s':>10}   vs sibling (Welch t)")
    for col, sign in (("fii_fut_ls", 1), ("fii_all_bull", 1), ("fii_opt_dir", 1),
                      ("client_fut_ls", -1), ("pro_fut_ls", 1), ("fii_minus_client", 1)):
        r = F[col] * sign
        rng = (r.rolling(20).max() - r.rolling(20).min()).replace(0, np.nan)
        posn = (r - r.rolling(20).min()) / rng                      # 0 = at own 20d low
        for label, pm, div, sib, turn in (
            ("price LOW, ratio already UP", lo20, posn >= 0.4, posn <= 0.15, "bot"),
            ("price HIGH, ratio already DOWN", hi20, posn <= 0.6, posn >= 0.85, "top"),
        ):
            a, b = stats((pm & div).fillna(False)), stats((pm & sib).fillna(False))
            if not a or not b:
                continue
            se = np.sqrt(a["x"].var(ddof=1) / a["n"] + b["x"].var(ddof=1) / b["n"])
            print(f"   {col:<20}{label:<34}{a['n']:>4}{a['mean']:>+8.2f}{a['up']:>6.0f}{a[turn]:>9.0f}%"
                  f"   sibling n={b['n']} fwd10 {b['mean']:+.2f} turn {b[turn]:.0f}%  t={(a['mean']-b['mean'])/se:+.2f}")

    print(f"\n{'='*100}\nF. SCENARIO MATRIX -- FII index-futures long share: level zone x 5-day change x price trend")
    P = causal_pct(F["fii_fut_ls"])
    d5 = F["fii_fut_ls"].diff(5)
    above = c > c.rolling(20).mean()
    zone = pd.cut(P, [-1, 10, 30, 70, 90, 101], labels=["EXTREME SHORT <=10", "SHORT 10-30",
                                                        "MIDDLE 30-70", "LONG 70-90", "EXTREME LONG >=90"])
    print(f"   base fwd10 {base[10]:+.2f}%  fwd20 {base[20]:+.2f}%   chance bottom {chance_b:.0f}%  top {chance_t:.0f}%")
    print(f"   {'FII zone':<20}{'FII 5d':<10}{'price':<12}{'days':>5}{'epis':>5}{'fwd10':>8}{'ex':>7}{'t':>6}"
          f"{'up%':>5}{'fwd20':>8}{'bot%':>6}{'top%':>6}")
    best = 0.0
    for z in zone.cat.categories:
        for dl, dm in (("adding L", d5 > 0), ("adding S", d5 <= 0)):
            for pl_, pm in (("above 20DMA", above), ("below 20DMA", ~above)):
                m = ((zone == z) & dm & pm & P.notna()).fillna(False)
                s, s20 = stats(m, 10), stats(m, 20)
                if not s:
                    print(f"   {z:<20}{dl:<10}{pl_:<12}{int(m.sum()):>5}    -")
                    continue
                best = max(best, abs(s["t"]))
                print(f"   {z:<20}{dl:<10}{pl_:<12}{s['days']:>5}{s['n']:>5}{s['mean']:>+8.2f}{s['ex']:>+7.2f}"
                      f"{s['t']:>+6.2f}{s['up']:>5.0f}{(s20['mean'] if s20 else np.nan):>+8.2f}"
                      f"{s['bot']:>5.0f}%{s['top']:>5.0f}%")
    print(f"   best |t| in the matrix {best:.2f} across 20 cells (Bonferroni needs |t| > 3.0)")

    print(f"\n{'='*100}\nG. EDGE CASES")
    ex = query_dataframe("""SELECT DISTINCT expiry_date FROM fno_bhavcopy
                            WHERE symbol='NIFTY' AND instrument IN ('FUTIDX','IDF') ORDER BY 1""")
    if not ex.empty:
        exd = pd.to_datetime(ex["expiry_date"])
        d1 = F["fii_fut_ls"].diff().abs() * 100
        on = d1[d1.index.isin(exd)]
        off = d1[~d1.index.isin(exd)]
        print(f"   monthly futures expiry days in sample: {len(on)}")
        print(f"   |1-day change in FII long share|, pp:  expiry day median {on.median():.2f} / mean {on.mean():.2f}"
              f"   other days median {off.median():.2f} / mean {off.mean():.2f}")
    gaps = cl.loc[idx[0]:].index.difference(idx)
    print(f"   Nifty sessions with NO participant row: {len(gaps)}   latest: {[str(g.date()) for g in gaps[-6:]]}")
    lev = (F["fii_fut_ls"] * 100).resample("YE").agg(["min", "median", "max"]).round(1)
    print("   FII index-futures long share by year (why fixed thresholds like '30%' cannot work):")
    print(lev.to_string())
    big = (F["fii_fut_ls"].diff(5) * 100)
    print(f"   5-day change in FII long share: p1 {big.quantile(.01):+.1f}  p99 {big.quantile(.99):+.1f}  "
          f"latest {big.iloc[-1]:+.1f} pp  (rank {100*(big <= big.iloc[-1]).mean():.1f}%)")
    print("   biggest 5-day JUMPS in FII long share, and what Nifty did next (tradable):")
    j = big.nlargest(60)
    e = episodes(big.index.isin(j.index), 20)
    for i in np.flatnonzero(e):
        print(f"     {idx[i].date()}  +{big.iloc[i]:.1f}pp  -> fwd10 {fwd[10].iloc[i]:+.2f}%  fwd20 {fwd[20].iloc[i]:+.2f}%")


if __name__ == "__main__":
    main()
    extra()
