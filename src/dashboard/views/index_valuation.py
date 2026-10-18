"""Index P/E & P/B panel (Sector Rotation page).

Question it answers, in the user's words: "when Nifty or another index had an
inflated P/E / P/B, did it crash? When it was at the bottom, was that the best
time to invest?"

Everything shown is measured by scripts/index_valuation_study.py on 1999-2026
niftyindices history (cached in data/index_valuation_study/summary.json). The
page describes zones and what historically followed them. It does not time the
market: the study found valuation says a lot about the next 1-3 YEARS and almost
nothing about the next month, and "expensive" stayed expensive for years.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_METRICS = {"P/E": "pe", "P/B": "pb", "Dividend yield": "dy"}
_ZONE_ICON = {"Extreme cheap": "🟩🟩", "Cheap": "🟩", "Fair": "⬜",
              "Expensive": "🟧", "Extreme expensive": "🟥", "—": ""}
_SIG_COL = {"Composite (P/E+P/B+DY)": "val_score", "P/E percentile": "pe_pct",
            "P/B percentile": "pb_pct", "Div yield percentile": "dy_pct",
            "Earnings yield − G-sec percentile": "ey_gap_pct"}


def _pct(v, d=1, sign=True):
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def _rank(v):
    return "—" if v is None or pd.isna(v) else f"{v * 100:.0f}th pct"


def _zone(p):
    from src.analytics.index_valuation import valuation_zone
    z = valuation_zone(p)
    return f"{_ZONE_ICON.get(z, '')} {z}".strip()


def _verdict(S: dict) -> None:
    """Every sentence is conditional on the cached numbers, so a re-run of the study
    can never leave a stale claim on the page."""
    n = S["nifty"]
    fam = {(t["signal"], t["outcome"]): t for t in n["family"]["tests"]}
    pe12 = fam[("P/E percentile", "fwd return 12m")]
    dy24 = fam[("Div yield percentile", "fwd return 24m")]
    w = n["warning"]["val_score"]
    z = {r["zone"]: r for r in n["zones"]["Composite (P/E+P/B+DY)"]}
    rob = n["robustness"]
    pe_r = rob["P/E percentile (spliced)"]
    cr = [c for c in n["crashes"] if c["val_at_peak"] is not None]
    flagged = sum(1 for c in cr if (c["val_at_peak"] or 0) >= 0.8)
    early = [f"another {abs(c['first_cheap_signal']['further_fall']):.0%} in "
             f"{c['trough'][:4]}" for c in cr
             if c.get("first_cheap_signal")
             and (c["first_cheap_signal"]["further_fall"] or 0) <= -0.10]
    # only cheap months whose 12-month outcome is already known
    horizon_end = str(pd.Timestamp(S["audit"]["Nifty 50"]["last"]) - pd.DateOffset(years=1))[:10]
    scored = [d for d in n.get("cheap_months", []) if d <= horizon_end]
    cheap_years = sorted({d[:4] for d in scored})
    cheap_indep = (z["Cheap"]["12m"]["indep"] or 0) + (z["Extreme cheap"]["12m"]["indep"] or 0)
    cheap_dd = max(z["Cheap"]["p_dd20"] or 0, z["Extreme cheap"]["p_dd20"] or 0)

    def _surv(t):
        p = t.get("p_family")
        return "survives" if p is not None and p <= 0.05 else "does NOT survive"

    lines = [
        f"**Measured, 1999–2026 (Nifty 50, {n['months']} month-ends, point-in-time "
        f"percentiles, total-return index): valuation is a 1–3 YEAR compass, not a crash "
        f"timer.**",
        f"* **Expensive → lower returns over the next 1–2 years.** P/E percentile vs next "
        f"12 months: rank-IC {pe12['ic']:+.2f} (t {pe12['disjoint_t']:+.1f} on "
        f"{pe12['disjoint_n']} non-overlapping years; {_surv(pe12)} the "
        f"{n['family']['n_tests']}-test family, p {pe12['p_family']:.2f}). Dividend yield vs "
        f"next 24 months: IC {dy24['ic']:+.2f}, {_surv(dy24)} (p {dy24['p_family']:.3f}). "
        f"Next 1 month: nothing usable.",
        f"* **Expensive is not 'crash next'.** {flagged} of the {len(cr)} ≥20% crashes since "
        f"2003 started with the composite above its 80th percentile. But from months at "
        f"≥90th, a ≥20% fall followed within 12 months only {w['p_crash_if_ge90']:.0%} of "
        f"the time (any month: {w['base_p_crash']:.0%}), and the median next-12-month "
        f"return was {_pct(z['Extreme expensive']['12m']['median'])} (all months: "
        f"{_pct(n['zones']['Composite (P/E+P/B+DY)'][-1]['12m']['median'])}). Expensive "
        f"markets stayed expensive for years.",
        f"* **Cheap has been the best time to invest — on a tiny sample.** Composite ≤30th: "
        f"median next 12 months {_pct(z['Cheap']['12m']['median'])}, ≥20% drawdown "
        f"afterwards {cheap_dd:.0%} — but that is {len(scored)} month-ends, worth "
        f"~{cheap_indep:.0f} non-overlapping years, clustered in a handful of episodes"
        + (f" ({', '.join(cheap_years)})" if cheap_years else "") + ". "
        + ("'Cheap' was not the bottom: after the first cheap reading the index fell "
           + " and ".join(early) + "." if early else ""),
        f"* **The edge lives mostly in 1999–2012.** P/E percentile IC since 2013: "
        f"{pe_r['2013+|12m']['ic']:+.2f} (t {pe_r['2013+|12m']['disjoint_t']:+.1f}); since "
        f"2018: {pe_r['2018+|12m']['ic']:+.2f} (t {pe_r['2018+|12m']['disjoint_t']:+.1f}). "
        f"P/E drifted up for two decades (trailing earnings depressed as ROE fell), so it "
        f"read ≥70th percentile in {pe_r['share_months_ge70_since_2013']:.0%} of months "
        f"since 2013. P/B and dividend yield did not drift — trust them more today.",
    ]
    st.info(lines[0] + "\n\n" + "\n".join(lines[1:]))


def _band_chart(f: pd.DataFrame, metric: str, label: str, crashes: list[dict],
                spliced: bool) -> go.Figure:
    col = metric + ("_adj" if spliced else "")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.04)
    x = f["trade_date"]
    m, sd = f[metric + "_mean5y"], f[metric + "_sd5y"]
    if not spliced:          # the band is built on the spliced series; rescale it
        ratio = f[metric] / f[metric + "_adj"]
        m, sd = m * ratio, sd * ratio
    for k, dash, name in ((2, "dot", "±2σ"), (1, "dash", "±1σ")):
        fig.add_trace(go.Scatter(x=x, y=m + k * sd, line=dict(color="rgba(150,150,150,.6)",
                                 dash=dash, width=1), name=f"5y mean {name}",
                                 legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=m - k * sd, line=dict(color="rgba(150,150,150,.6)",
                                 dash=dash, width=1), showlegend=False,
                                 legendgroup=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=m, line=dict(color="rgba(150,150,150,.9)", width=1),
                             name="5y mean"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=f[col], line=dict(color="#4c9be8", width=1.6),
                             name=label + (" (2021-adjusted)" if spliced else " (as published)")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=f["val_score"] * 100, line=dict(color="#e8a24c", width=1.4),
                             name="Composite percentile"), row=2, col=1)
    for y, c in ((90, "rgba(220,60,60,.5)"), (70, "rgba(230,150,60,.4)"),
                 (30, "rgba(60,170,90,.4)"), (10, "rgba(40,140,70,.5)")):
        fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1), row=2, col=1)
    for c in crashes:
        fig.add_vrect(x0=c["peak"], x1=c["trough"], fillcolor="rgba(220,60,60,.12)",
                      line_width=0, row="all", col=1)
    fig.add_vline(x="2021-03-31", line=dict(color="rgba(120,120,120,.7)", dash="dash", width=1))
    fig.update_yaxes(title_text=label, row=1, col=1)
    fig.update_yaxes(title_text="pctile", range=[0, 100], row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06, x=0), hovermode="x unified")
    return fig


def _zone_frame(rows: list[dict], horizons: tuple[str, ...]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = {"Zone": f"{_ZONE_ICON.get(r['zone'], '')} {r['zone']}".strip(),
             "Months": r["months"]}
        for k in horizons:
            h = r.get(k) or {}
            d[f"{k} median"] = _pct(h.get("median"))
            d[f"{k} up %"] = _pct(h.get("hit"), 0, False)
            d[f"{k} worst"] = _pct(h.get("worst"))
            d[f"{k} indep. yrs"] = h.get("indep")
        d["P(≥20% fall in 12m)"] = _pct(r.get("p_dd20"), 0, False)
        d["P(≥10% fall in 12m)"] = _pct(r.get("p_dd10"), 0, False)
        out.append(d)
    return pd.DataFrame(out)


def _crash_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for c in rows:
        fc = c.get("first_cheap_signal") or {}
        out.append({
            "Peak": c["peak"], "Trough": c["trough"], "Fall": _pct(c["depth"]),
            "Recovered": c.get("recovered") or "not yet",
            "P/E at peak": c["pe_at_peak"],
            "Zone at peak": _zone(c["val_at_peak"]),
            "P/E pct": _rank(c["pe_pct_at_peak"]), "P/B pct": _rank(c["pb_pct_at_peak"]),
            "EPS peak→trough": _pct(c["eps_change"]),
            "P/E peak→trough": _pct(c["pe_change"]),
            "Zone at trough": _zone(c["val_at_trough"]),
            "First 'cheap' day": fc.get("date") or "never",
            "Fell further after": _pct(fc.get("further_fall")) if fc else "—",
            "12m from trough": _pct((c.get("fwd_from_trough") or {}).get("12m")),
            "36m from trough": _pct((c.get("fwd_from_trough") or {}).get("36m")),
        })
    return pd.DataFrame(out)


# Green = cheap ... red = expensive, the colours the eye already reads that way.
_VAL_SCALE = [[0.0, "#1e6f3a"], [0.15, "#2e8b4a"], [0.35, "#7fbf7b"], [0.5, "#d9d9c3"],
              [0.65, "#e8b4a0"], [0.85, "#e8826a"], [1.0, "#d9534f"]]
_RET_SCALE = [[0.0, "#b2182b"], [0.35, "#e8826a"], [0.5, "#d9d9c3"],
              [0.65, "#7fbf7b"], [1.0, "#1e6f3a"]]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_PLAIN = {"Extreme cheap": ("🟢", "Deeply undervalued"), "Cheap": ("🟢", "Undervalued"),
          "Fair": ("🟡", "Fairly valued"), "Expensive": ("🟠", "Overvalued"),
          "Extreme expensive": ("🔴", "Deeply overvalued"), "—": ("⚪", "Not enough history")}


def _zone_word(p) -> str:
    from src.analytics.index_valuation import valuation_zone
    return valuation_zone(p)


def _grid(f: pd.DataFrame, col: str, how: str = "median") -> pd.DataFrame:
    """Year x month table of `col`: the month's median, or its LAST value."""
    g = f[["trade_date", col]]
    g = g.assign(y=g["trade_date"].dt.year, m=g["trade_date"].dt.month)
    if how == "median":
        agg = g.dropna().groupby(["y", "m"])[col].median()
    else:
        # the month's LAST SESSION, blank or not: groupby().last() skips NaN and
        # would silently start Sep-2025's "next 12 months" on 22 Sep instead
        agg = g.groupby(["y", "m"]).tail(1).set_index(["y", "m"])[col]
    return agg.unstack("m").reindex(columns=range(1, 13)).sort_index()


def _forward_grid(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From each month's LAST session: total return over the next 252 sessions, and
    the worst close inside them. Blank where those 12 months have not finished --
    never a partial year shown as if it were a full one."""
    from src.analytics.index_valuation import return_level
    lvl = return_level(f).to_numpy(float)
    c = f["close_val"].to_numpy(float)
    n, h = len(f), 252
    fwd, dd = np.full(n, np.nan), np.full(n, np.nan)
    if n > h:
        fwd[: n - h] = lvl[h:] / lvl[: n - h] - 1
        # mins[i] = min(c[i..i+251]); the window AFTER day i is mins[i+1]
        mins = pd.Series(c[::-1]).rolling(h, min_periods=h).min().to_numpy()[::-1]
        dd[: n - h] = mins[1: n - h + 1] / c[: n - h] - 1
    x = f[["trade_date"]].assign(fwd=fwd, dd=dd)
    return _grid(x, "fwd", "last"), _grid(x, "dd", "last")


def _whole_history_rank(g: pd.DataFrame) -> pd.DataFrame:
    """Each cell's percentile among ALL non-blank cells. Blank cells (future months,
    pre-listing years, dropped pre-2021 history) must not sit in the denominator:
    with this year's Oct-Dec blank they did, and every colour came out too cheap."""
    flat = g.to_numpy(dtype=float).ravel()
    flat = np.sort(flat[~np.isnan(flat)])
    if not len(flat):
        return g * np.nan

    def r(v):
        if pd.isna(v):
            return np.nan
        lo = np.searchsorted(flat, v, "left")
        hi = np.searchsorted(flat, v, "right")
        return (lo + 0.5 * (hi - lo)) / len(flat)
    return g.apply(lambda col: col.map(r))


def _heatmap(z: pd.DataFrame, text: pd.DataFrame, scale, zmin, zmax, title: str,
             hover: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(dtype=float), x=_MONTHS, y=[str(y) for y in z.index],
        text=text.to_numpy(), texttemplate="%{text}", textfont=dict(size=12),
        colorscale=scale, zmin=zmin, zmax=zmax, showscale=False, xgap=2, ygap=2,
        hovertemplate=hover + "<extra></extra>", hoverongaps=False))
    fig.update_yaxes(autorange="reversed", type="category", tickfont=dict(size=11))
    fig.update_xaxes(side="top", type="category")
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)),
                      height=90 + 27 * len(z), margin=dict(l=10, r=10, t=70, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _cards(last: pd.Series, f: pd.DataFrame) -> None:
    icon, words = _PLAIN.get(_zone_word(last["val_score"]), ("⚪", "—"))
    cols = st.columns(4)

    def _since(col):
        s = f.loc[f[col + "_adj"].notna(), "trade_date"]
        return s.min().year if len(s) else None

    def _card(col, label, value, key, extra=""):
        pct, yr = last[key + "_pct"], _since(key)
        with col.container(border=True):
            st.metric(label, value)
            if pd.notna(pct):
                st.caption(f"higher than {pct:.0%} of its history since {yr}{extra}"
                           + (" — short history, read loosely" if yr and yr >= 2018 else ""))
            elif pd.isna(last[key]):
                st.caption("not published (index earnings negative)" if key == "pe"
                           else "not published")
            else:
                st.caption("not enough comparable history yet")

    _card(cols[0], "P/E ratio (trailing)",
          f"{last['pe']:.2f}" if pd.notna(last["pe"]) else "—", "pe")
    _card(cols[1], "P/B ratio (price / book)",
          f"{last['pb']:.2f}" if pd.notna(last["pb"]) else "—", "pb")
    _card(cols[2], "Dividend yield",
          f"{last['dy']:.2f}%" if pd.notna(last["dy"]) else "—", "dy",
          " (high yield = cheap)")
    with cols[3].container(border=True):
        st.metric("Zone", f"{icon} {words}")
        parts = f.attrs.get("composite_parts") or ["pe", "pb", "dy"]
        lab = " + ".join({"pe": "P/E", "pb": "P/B", "dy": "yield"}[k] for k in parts)
        st.caption((f"{lab} combined: {last['val_score']:.0%} percentile"
                    if len(parts) > 1 else
                    f"based on {lab} only: {last['val_score']:.0%} percentile")
                   + ("" if len(parts) == 3 else
                      " (the other metrics changed method in 2021 and can't be "
                      "compared across it)")
                   if pd.notna(last["val_score"]) else
                   "not enough comparable history to combine yet")


def _today_says(S: dict | None, name: str, last: pd.Series) -> None:
    """One plain paragraph: what followed this zone before, with its sample-size tag."""
    zone = _zone_word(last["val_score"])
    rows = None
    if S and name == "Nifty 50":
        rows = S["nifty"]["zones"]["Composite (P/E+P/B+DY)"]
    elif S and S["indices"].get(name):
        rows = S["indices"][name]["zones_val"]
    if not rows or zone == "—":
        return
    r = next((x for x in rows if x["zone"] == zone), None)
    base = rows[-1]
    if not r or not (r.get("12m") or {}).get("n"):
        return
    icon, words = _PLAIN[zone]
    thin = (r["12m"]["indep"] or 0) < 3
    st.info(
        f"**{icon} {name} is {words.lower()} today.** In past months like this, the next "
        f"12 months returned **{_pct(r['12m']['median'])}** (median) and were positive "
        f"**{r['12m']['hit']:.0%}** of the time — against {_pct(base['12m']['median'])} and "
        f"{base['12m']['hit']:.0%} for any month. A 20%+ fall followed "
        f"{r['p_dd20']:.0%} of the time (any month: {base['p_dd20']:.0%})."
        + (f"  \n⚠️ Small sample: only ~{r['12m']['indep']:.0f} non-overlapping years of "
           "history sit in this zone. Treat it as a pointer, not a promise." if thin else "")
        + "  \nValuation speaks about the next 1–3 years, not the next few weeks.")


def _sync_pick(src: str, dst: str) -> None:
    st.session_state[dst] = st.session_state[src]


def render_index_valuation(selected_date: date) -> None:
    from src.dashboard.cache.queries import (cached_valuation_board,
                                             cached_valuation_catalog,
                                             cached_valuation_features,
                                             cached_valuation_study)
    st.markdown("#### 📐 Index P/E & P/B — cheap or expensive, and what came next")
    try:
        cat = cached_valuation_catalog()
    except Exception as exc:
        st.warning(f"Valuation history not loaded ({exc}). Run "
                   "`python scripts/backfill_index_valuation.py`.")
        return
    if cat.empty:
        st.warning("No valuation history yet. Run `python scripts/backfill_index_valuation.py`.")
        return
    S = cached_valuation_study()
    names = cat["index_name"].tolist()
    label = {r.index_name: f"{r.index_name}   ·   {r.group}, since {pd.Timestamp(r.first_day).year}"
             for r in cat.itertuples()}
    # Two pickers (page top and right above the heatmap) share one selection.
    for k in ("iv_index", "iv_index_hm"):
        if st.session_state.get(k) not in names:
            st.session_state[k] = "Nifty 50" if "Nifty 50" in names else names[0]
    st.selectbox("Index", names, key="iv_index", format_func=label.get,
                 on_change=_sync_pick, args=("iv_index", "iv_index_hm"),
                 help="Broad market, sectoral and thematic NSE indices. 'Since' = first "
                      "year niftyindices publishes P/E, P/B and yield for it.")
    name = st.session_state["iv_index"]

    f, info = cached_valuation_features(name, selected_date)
    if f.empty:
        st.warning(f"No data for {name} on or before {selected_date}.")
        return
    last = f.iloc[-1]
    as_of = last["trade_date"].date()
    st.caption(f"{name} · latest print {as_of:%d %b %Y}")
    # A failed nightly fetch must not pass for "today": say how old the print is.
    lag = (selected_date - as_of).days
    if lag > 5:
        st.warning(f"Latest valuation for {name} is from {as_of:%d %b %Y}, {lag} days before "
                   f"the date you picked — the nightly refresh may have failed "
                   "(re-run the daily job).")
    if S and (pd.Timestamp(S.get("generated", selected_date))
              - pd.Timestamp(selected_date)).days > 31:
        st.caption(f"⏪ Viewing {selected_date:%d %b %Y}: cards and maps use only data up to "
                   "that day, but the 'past months like this' statistics were measured on "
                   f"1999–{S['generated'][:4]} and include years after it.")

    _cards(last, f)
    _today_says(S, name, last)

    # ── Heatmap 1: the valuation itself ─────────────────────────────────────
    st.markdown("##### Month-by-month valuation map")
    h1, h2, h3 = st.columns([2.2, 2.3, 2])
    h1.selectbox("Index", names, key="iv_index_hm", format_func=label.get,
                 on_change=_sync_pick, args=("iv_index_hm", "iv_index"))
    metric_lab = h2.segmented_control(
        "Heatmap of", list(_METRICS) + ["Combined zone"], default="P/E", key="iv_metric",
        help="Combined zone = P/E + P/B + dividend yield together — the same score as the "
             "Zone card. Use it when the single metrics disagree.") or "P/E"
    m = _METRICS.get(metric_lab, "val")
    years = sorted(f["trade_date"].dt.year.unique())
    if m == "val":
        # Only a point-in-time version exists: the combined score is itself a
        # percentile ranked against earlier history. Cell = that percentile.
        color_by = "Only what was known then"
        shown = _grid(f, "val_score").reindex(years)
        z = shown.copy()
        txt = shown.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}")).astype(object)
        title = (f"{name} combined zone — percentile vs earlier history   "
                 "(0 = cheapest ever · 100 = dearest ever)")
        hover = "%{y} %{x}: combined percentile %{text}"
    else:
        color_by = h3.segmented_control(
            "Colour compared with", ["Whole history", "Only what was known then"],
            default="Whole history", key="iv_colour",
            help="Whole history: each month coloured against all months 1999-2026 — the "
                 "simplest picture, but it uses hindsight (in 2003 nobody knew 2021's "
                 "levels). Only what was known then: each month coloured against the "
                 "months BEFORE it — what an investor could actually see at the time."
        ) or "Whole history"
        shown = _grid(f, m).reindex(years)         # numbers as published
        adj = _grid(f, m + "_adj")                 # 2021-comparable, for colour only
        if color_by == "Whole history":
            z = _whole_history_rank(adj)
        else:
            z = _grid(f, m + "_pct")
        if m == "dy":
            z = 1 - z                              # high yield = cheap = green
        txt = shown.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")).astype(object)
        title = f"{name} {metric_lab} — monthly median   (green = cheap · red = expensive)"
        hover = "%{y} %{x}: " + metric_lab + " %{text}" + ("%" if m == "dy" else "")
    if shown.isna().all().all():
        st.info(f"No {metric_lab} map for {name} yet: "
                + ("the combined zone needs ~6 years of comparable history "
                   "(3 to rank each metric, then 3 to rank the blend)."
                   if m == "val" else "nothing published for this period."))
    else:
        st.plotly_chart(_heatmap(z.reindex(index=shown.index), txt, _VAL_SCALE, 0, 1, title,
                                 hover), key="iv_hm_val")

    note = ""
    brk = (info or {}).get(m) if m != "val" else None
    if brk and not brk.get("dropped_pre_break"):
        note = (" Before Apr-2021 NSE used standalone earnings; colours use the "
                "consolidated-equivalent so old and new years compare fairly, numbers are "
                "shown as published.")
    elif brk:
        note = (" NSE's Apr-2021 method change moved this index's figure too much to "
                "compare across it, so pre-2021 cells are uncoloured (numbers still shown).")
    if color_by == "Only what was known then":
        note += " Cells are blank for the first ~3 years: a comparison needs history first."
    st.caption(("Each cell = that month's median." if m != "val" else
                "Each cell = that month's median combined percentile.") + note)

    # When this map and the Zone card point opposite ways, say why in plain words.
    if m in ("pe", "pb", "dy") and pd.notna(last["val_score"]):
        # the CURRENT month's cell -- not the grid's last cell, which is a blank
        # future month (Oct-Dec) for most of the year
        cur_rank = z.reindex(index=shown.index).get(as_of.month, pd.Series(dtype=float)).get(as_of.year, np.nan)
        zone_cheap = last["val_score"] <= 0.30
        zone_dear = last["val_score"] >= 0.70
        # 0.60 / 0.40: where a cell is already visibly pink / green on the scale
        if pd.notna(cur_rank) and ((cur_rank >= 0.60 and zone_cheap) or
                                   (cur_rank <= 0.40 and zone_dear)):
            others = {"pe": "P/E", "pb": "P/B", "dy": "dividend yield"}
            parts = [f"{others[k]} {'dear' if (last[k + '_pct'] if k != 'dy' else 1 - last[k + '_pct']) >= 0.5 else 'cheap'}"
                     f" ({last[k + '_pct']:.0%} pct)"
                     for k in ("pe", "pb", "dy") if pd.notna(last[k + "_pct"])]
            st.warning(
                f"**This map and the Zone card disagree — both are right.** {metric_lab} alone "
                f"looks {'expensive' if cur_rank >= 0.60 else 'cheap'} against history, but the "
                f"Zone combines all three: {', '.join(parts)}."
                + (" P/E has drifted up for two decades as earnings were depressed (return on "
                   "equity fell), so P/E alone over-states how expensive recent years are; "
                   "P/B and yield did not drift." if m == "pe" else "")
                + " Switch to **Combined zone** to see the map the card uses.")

    # ── Heatmap 2: what happened next ──────────────────────────────────────
    fwd, dd = _forward_grid(f)
    fwd, dd = fwd.reindex(years), dd.reindex(years)
    ret_txt = fwd.apply(lambda col: col.map(lambda v: "" if pd.isna(v) else f"{v * 100:+.0f}%"))
    ret_txt = ret_txt.astype(object).fillna("")
    ret_txt = ret_txt.where(~(dd <= -0.20), ret_txt.astype(str) + " ▼")
    st.plotly_chart(_heatmap(
        fwd.clip(-0.6, 0.6), ret_txt, _RET_SCALE, -0.6, 0.6,
        f"What happened next — {name} return over the following 12 months"
        "   (▼ = fell 20%+ at some point in that year)",
        "Bought end of %{x} %{y} → next 12 months %{text}"), key="iv_hm_fwd")
    st.caption("Read the two maps together: red months above should sit on weak or ▼ "
               "months here, green months on strong ones. Returns include dividends. "
               "Blank = those 12 months have not finished yet.")

    # ── Everything else, one click away ────────────────────────────────────
    if S:
        with st.expander("📊 27-year backtest verdict — does valuation predict returns and crashes?"):
            _verdict(S)
    with st.expander("📈 Trend chart with 5-year average bands"):
        spliced = st.toggle("Adjust 2021 break", value=True, key="iv_splice",
                            help="On 2021-03-31 NSE moved P/E and dividend yield from "
                                 "standalone to consolidated figures (Nifty P/E 40.4 → 33.2 "
                                 "on a −1% day). On = pre-2021 values rescaled.")
        crashes = (S or {}).get("indices", {}).get(name, {}).get("crashes", [])
        lab = metric_lab + (" (%)" if metric_lab == "Dividend yield" else "")
        bm = m if m != "val" else "pe"
        st.plotly_chart(_band_chart(f, bm, lab if m != "val" else "P/E", crashes, spliced),
                        key="iv_band")
        st.caption("Top: the metric with its 5-year mean ±1σ/±2σ. Bottom: the combined "
                   "percentile, ranked only against earlier history. Red shading = ≥20% "
                   "falls, peak to trough.")
        k = st.columns(2)
        k[0].metric("Earnings yield − 10y G-sec",
                    f"{last['ey_gap']:+.2f} pp" if pd.notna(last["ey_gap"]) else "—",
                    help="100/P/E minus the 10-year government bond yield. Below zero = "
                         "bonds pay more than equity earnings.")
        k[1].metric("P/E vs its 5-year average",
                    f"{last['pe_z5y']:+.1f} σ" if pd.notna(last["pe_z5y"]) else "—")
    if not S:
        st.warning("Study not run yet: `python scripts/index_valuation_study.py`.")
        return
    idx = S["indices"].get(name)
    with st.expander("🧮 What came next, by zone (full table)"):
        if name == "Nifty 50":
            sig = st.segmented_control("Zone by", list(_SIG_COL),
                                       default="Composite (P/E+P/B+DY)",
                                       key="iv_sig") or "Composite (P/E+P/B+DY)"
            if sig in ("Div yield percentile", "Earnings yield − G-sec percentile"):
                st.caption("Yield signals are flipped so every table reads the same way: "
                           "'Extreme cheap' = the HIGHEST-yield 10% of months.")
            st.dataframe(_zone_frame(S["nifty"]["zones"][sig], ("3m", "12m", "24m", "36m")),
                         hide_index=True)
        elif idx:
            st.dataframe(_zone_frame(idx["zones_val"], ("12m", "36m")), hide_index=True)
        st.caption("'Indep. yrs' = months ÷ horizon: how many genuinely separate outcomes the "
                   "row rests on — below ~3, treat the row as anecdote, not statistics.")
    crashes = (idx or {}).get("crashes", [])
    if crashes:
        with st.expander("💥 Every 20%+ crash: was valuation warning, and was 'cheap' the bottom?"):
            st.dataframe(_crash_frame(crashes), hide_index=True)
            st.caption("Crashes are de-ratings: trailing EPS was flat or up from peak to "
                       "trough in every one — the whole fall is the multiple.")
            if name == "Nifty 50" and S["nifty"]["corrections"]:
                st.markdown("**10–20% corrections**")
                st.dataframe(_crash_frame(S["nifty"]["corrections"]), hide_index=True)
    with st.expander("🗂️ All indices today"):
        # Expander bodies run even when collapsed; ~70 indices cost ~10s, so the
        # table is built only when asked for.
        if st.toggle("Load the all-indices table", key="iv_board_on"):
            _board(S, selected_date, cached_valuation_board)
    _methodology(S)


def _board(S: dict, selected_date: date, loader) -> None:
    try:
        bd = loader(selected_date)
    except Exception as exc:
        st.caption(f"Board unavailable: {exc}")
        return
    if bd.empty:
        return

    def _ev(n, key):
        return (S["indices"].get(n) or {}).get(key) or {}

    view = pd.DataFrame({
        "Index": bd["index_name"],
        "Zone": [" ".join(_PLAIN.get(_zone_word(v), ("⚪", "—"))) for v in bd["val_score"]],
        "Combined pct": (bd["val_score"] * 100).round(0),
        "P/E": bd["pe"].round(2), "P/E pct": (bd["pe_pct"] * 100).round(0),
        "P/B": bd["pb"].round(2), "P/B pct": (bd["pb_pct"] * 100).round(0),
        "DY %": bd["dy"].round(2), "DY pct": (bd["dy_pct"] * 100).round(0),
        "IC 12m": [_ev(n, "val_score|12m").get("ic") for n in bd["index_name"]],
        "t 12m": [_ev(n, "val_score|12m").get("disjoint_t") for n in bd["index_name"]],
        "Ranked since": [
            ", ".join(f"{mm.upper()} {str(d)[:4]}" for mm, d in
                      (("pe", r.pe_since), ("pb", r.pb_since), ("dy", r.dy_since))
                      if d is not None and str(d) >= "2021-03-31") or "full history"
            for r in bd.itertuples()],
    }).sort_values("Combined pct", ascending=False)
    st.dataframe(view, hide_index=True, column_config={
        "Combined pct": st.column_config.ProgressColumn(
            "Combined pct", min_value=0, max_value=100, format="%.0f",
            help="P/E, P/B and yield percentiles combined, each ranked only against "
                 "earlier history. High = expensive."),
        "IC 12m": st.column_config.NumberColumn(
            help="How well this index's valuation ranked its own next-12-month return. "
                 "Negative = expensive was followed by lower returns (what you want)."),
        "t 12m": st.column_config.NumberColumn(
            help="t-stat on non-overlapping years. |t| < 2 = could be chance."),
        "Ranked since": st.column_config.TextColumn(
            help="Metrics whose pre-2021 history was dropped (NSE's method change moved "
                 "them > ±43% in one day); their percentile compares with 2021+ only."),
    })
    xs = S.get("sector_cross_section", {}).get("pb_pct")
    if xs:
        st.caption(
            f"Picking sectors by 'cheapest vs its own history' did NOT work: cheap-minus-rich "
            f"{_pct(xs['cheap_minus_rich_12m'])} over 12 months vs Nifty, t {xs['nw_t']:+.2f} "
            f"over {xs['months']} months. Use the zone for the market, not to rank sectors.")


def _methodology(S: dict) -> None:
    n = S["nifty"]
    with st.expander("Method, data checks, and every test run"):
        st.markdown(f"""
**Data.** niftyindices.com P/E, P/B, dividend-yield, closing and total-return history,
1999-01-01 → {S['audit']['Nifty 50']['last']}, {len(S['indices'])} indices. India VIX from
NSE (2010+). 10y G-sec yield monthly: IMF IFS to 2011-11, OECD from 2011-12 (overlap
{S['gsec_splice']['overlap_months']} months, mean gap {S['gsec_splice']['mean_abs_diff']} pp);
a month's average is used only from the following month on.

**Traps handled.**
- *2021-03-31 methodology break* — NSE switched to consolidated earnings; Nifty P/E fell
  18% on a −1% day. Pre-break P/E scaled ×{S['breaks']['Nifty 50']['pe']['factor']}, DY
  ×{S['breaks']['Nifty 50']['dy']['factor']}; P/B had no break. Raw vs adjusted results are
  in the robustness table below.
- *Look-ahead* — every percentile ranks a day only against earlier days. A full-sample
  percentile would have called 2003 cheap using 2021's distribution.
- *Drift* — a fixed 'expensive' line learned on 1999–2012 (P/E > {n['drift']['pe_adj']['train_p80']})
  was exceeded {n['drift']['pe_adj']['share_after_above_p80']:.0%} of days after 2013. P/B's
  equivalent line: {n['drift']['pb_adj']['share_after_above_p80']:.0%}. P/E's level moved,
  P/B's and DY's did not.
- *Overlap* — daily 12-month returns overlap 251/252; inference uses month-ends, t-stats on
  non-overlapping windows, and a circular-shift null ({S['n_shift']:,} rotations).
- *Multiple testing* — {n['family']['n_tests']} tests on Nifty 50 (8 signals × 6 horizons +
  drawdown); 'family p' is the chance the BEST of all of them looks this good on noise.
- *'-' P/E* — not computed (negative index earnings), kept NULL, never 0.
""")
        fam = pd.DataFrame(n["family"]["tests"])[
            ["signal", "outcome", "ic", "disjoint_t", "disjoint_n", "nw_t", "p_shift", "p_family"]]
        st.markdown("**All pre-registered tests (Nifty 50)** — sign: P/E & P/B negative is "
                    "good, yields positive is good.")
        st.dataframe(fam, hide_index=True, column_config={
            "disjoint_n": st.column_config.NumberColumn("indep. windows")})
        rob = n["robustness"]
        rows = []
        for lab, r in rob.items():
            if not isinstance(r, dict) or "all|12m" not in r:
                continue
            rows.append({"Signal": lab,
                         **{f"{t} {h}": f"{r[f'{t}|{h}']['ic']} (t {r[f'{t}|{h}']['disjoint_t']})"
                            for t in ("all", "ex_2003_2009", "2013+", "2018+") for h in ("12m",)},
                         "Months ≥70th since 2013": _pct(r["share_months_ge70_since_2013"], 0, False)})
        st.markdown("**Robustness (added after the first look — not in the family p)**: IC with "
                    "next-12-month return (t on non-overlapping years).")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        audit = S["audit"]["Nifty 50"]
        st.caption(f"Nifty 50 audit: {audit['sessions']:,} sessions, gaps > 6 days: "
                   f"{len(audit['gaps_over_6_days'])}, P/E nulls {audit['pe_null_sessions']}, "
                   f"TRI nulls {audit['tri_null_sessions']} (close used there). "
                   f"Study generated {S['generated']}.")
