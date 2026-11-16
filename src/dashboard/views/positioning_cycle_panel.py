"""
Positioning Cycle panel for the Big Players (F&O) page.

Answers the question users bring to this page: "which ratio tells me FIIs are at a
peak or a trough, so I know the market is about to turn?" -- with the measured answer
rather than the hoped-for one. See src/analytics/positioning_cycle.py for the method
and scripts/study_fii_turning_points.py for the full study.

Every number on this panel is computed from data dated on or before the selected
date. The only frozen numbers are in the "what was tested and failed" expander, and
they are labelled with the date they were measured.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import cached_positioning_cycle

_ZONE_COLOR = {
    "xs": "#ef5350", "s": "#ff9800", "m": "#9e9e9e", "l": "#66bb6a", "xl": "#26a69a",
}
_ZONE_LABEL = {
    "xs": "EXTREME SHORT", "s": "SHORT", "m": "MIDDLE", "l": "LONG", "xl": "EXTREME LONG",
}
_PLAYER_COLOR = {"fii": "#2196f3", "pro": "#9c27b0", "client": "#ff9800", "dii": "#4caf50"}


def _fmt(v, nd: int = 1, sign: bool = False, suf: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:+.{nd}f}{suf}" if sign else f"{v:.{nd}f}{suf}"


def _ord(v) -> str:
    """Percentile as an ordinal: 1 -> '1st', 62 -> '62nd', None -> '—'."""
    if v is None:
        return "—"
    n = int(round(v))
    if n >= 100 and v < 100:
        n = 99            # 99.6 is not "the 100th" -- nothing is above 100
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _head(txt: str) -> None:
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.45);"
        f"letter-spacing:1.5px;margin:14px 0 8px'>{txt}</div>",
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────── plain answer
def _answer(v: dict) -> None:
    fii, cl = v["fii"], v["client"]
    z = v.get("today_zone") or {}
    cb, ct = v.get("chance_bottom"), v.get("chance_top")
    base10, up10 = v["base"][10], v["base_up"][10]
    zc = _ZONE_COLOR.get(fii["zone"], "#9e9e9e")
    zl = _ZONE_LABEL.get(fii["zone"], "NOT RANKED")

    if not v["have_pct"]:
        st.info("Not enough history before this date to rank positioning against its own past year.")
        return

    pct_txt = (f"{_ord(fii['pct'])} percentile of its own last {v['params']['pct_window']} sessions"
               if fii["pct"] is not None else "not ranked")
    lines = [
        f"<b>FII index-futures long share is {fii['long_share']:.1f}%</b> "
        f"({fii['long']:,.0f} long vs {fii['short']:,.0f} short contracts, long/short ratio "
        f"{_fmt(fii['ls_ratio'], 2)}) — {pct_txt}. "
        f"Last year's range was {fii['min250']:.1f}% to {fii['max250']:.1f}%, median {fii['med250']:.1f}%.",
    ]
    if cl and cl["pct"] is not None:
        lines.append(
            f"Client (retail) long share is {cl['long_share']:.1f}%, at the {_ord(cl['pct'])} percentile. "
            "Clients sit on the other side of FIIs almost by arithmetic — the four categories sum to zero."
        )

    if z.get("judged"):
        bn, tn = z.get("bottom_near"), z.get("top_near")
        verdict = []
        if tn is not None and ct and tn >= ct * 1.5:
            verdict.append(f"a swing <b>TOP</b> followed within 10 sessions in <b>{tn:.0f}%</b> of past episodes "
                           f"(any random day: {ct:.0f}%)")
        if bn is not None and cb and bn >= cb * 1.5:
            verdict.append(f"a swing <b>BOTTOM</b> followed within 10 sessions in <b>{bn:.0f}%</b> of past episodes "
                           f"(any random day: {cb:.0f}%)")
        if not verdict:
            verdict.append(f"turns did not follow more often than after a random day (top {_fmt(tn,0)}% vs {_fmt(ct,0)}%, "
                           f"bottom {_fmt(bn,0)}% vs {_fmt(cb,0)}%)")
        lines.append(
            f"In the <b style='color:{zc}'>{zl}</b> zone ({z['episodes']} separate episodes since "
            f"{v['history_from']:%b %Y}): " + "; ".join(verdict) + ". "
            f"Nifty over the next 10 sessions averaged <b>{_fmt(z.get('fwd10'), 2, True, '%')}</b> "
            f"(up {_fmt(z.get('up10'), 0)}% of the time) against {base10:+.2f}% / {up10:.0f}% on all days"
            f" — t {_fmt(z.get('t10'), 2, True)}, "
            + ("not distinguishable from normal." if abs(z.get("t10") or 0) < 2 else "a measurable difference.")
        )

    s = v.get("swing")
    if s:
        if s["last_kind"] == "BOTTOM":
            trig = s["extreme_close"] * (1 - s["reversal_needed_pct"] / 100)
            lines.append(
                f"Swing context: Nifty has risen {s['move_pct']:+.1f}% in {s['sessions']} sessions since the "
                f"{s['last_date']:%d %b} bottom ({s['last_close']:,.0f}), when FII long share was "
                f"{s['fii_ls_then']:.1f}% ({_ord(s['fii_pct_then'])} pct). High so far {s['extreme_close']:,.0f} "
                f"on {s['extreme_date']:%d %b}; a close below <b>{trig:,.0f}</b> would confirm that high as a "
                f"{s['reversal_needed_pct']:.0f}% top."
            )
        else:
            trig = s["extreme_close"] * (1 + s["reversal_needed_pct"] / 100)
            lines.append(
                f"Swing context: Nifty is {s['move_pct']:+.1f}% in {s['sessions']} sessions since the "
                f"{s['last_date']:%d %b} top ({s['last_close']:,.0f}), when FII long share was "
                f"{s['fii_ls_then']:.1f}% ({_ord(s['fii_pct_then'])} pct). Low so far {s['extreme_close']:,.0f} "
                f"on {s['extreme_date']:%d %b}; a close above <b>{trig:,.0f}</b> would confirm that low as a "
                f"{s['reversal_needed_pct']:.0f}% bottom."
            )

    body = "".join(f"<div style='margin:0 0 7px'>{x}</div>" for x in lines)
    st.markdown(
        f"<div style='border:1px solid {zc}66;border-left:4px solid {zc};border-radius:8px;"
        "padding:12px 16px;background:rgba(255,255,255,0.02);font-size:13.5px;line-height:1.55'>"
        f"<div style='font-size:11px;font-weight:700;letter-spacing:1px;color:{zc};margin-bottom:8px'>"
        f"POSITIONING STATE · {v['data_date']:%d %b %Y} · FII {zl}</div>{body}"
        "<div style='font-size:11.5px;color:rgba(255,255,255,0.5);margin-top:6px'>"
        "This is a <b>zone, not a trigger</b>. Positioning moves <i>with</i> price, not ahead of it: "
        "FIIs reach their most-short reading on the day of a bottom, not before it. Use it to know when a "
        "turn has become more likely — size and stops — never as a buy or sell signal on its own.</div></div>",
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────── player cards
def _player_cards(v: dict) -> None:
    cols = st.columns(len(v["players"]))
    for col, p in zip(cols, v["players"]):
        c = _PLAYER_COLOR.get(p["key"], "#9e9e9e")
        zc = _ZONE_COLOR.get(p["zone"], "#555")
        pct = p["pct"]
        bar = (f"<div style='height:6px;background:rgba(255,255,255,0.08);border-radius:3px;"
               f"position:relative;margin:6px 0 2px'><div style='position:absolute;left:0;top:0;"
               f"height:6px;width:{pct:.0f}%;background:{zc};border-radius:3px'></div></div>"
               if pct is not None else "")
        opt = ("" if p["opt_bull_share"] is None else
               f"<div style='font-size:11px;color:rgba(255,255,255,0.55);margin-top:4px'>"
               f"Index options bullish legs {p['opt_bull_share']:.1f}% "
               f"({_ord(p['opt_pct'])} pct)</div>")
        col.markdown(
            f"<div style='border:1px solid rgba(255,255,255,0.1);border-top:3px solid {c};"
            "border-radius:8px;padding:10px 12px;background:rgba(255,255,255,0.02);min-height:190px'>"
            f"<div style='font-weight:700;color:{c}'>{p['name']}</div>"
            f"<div style='font-size:10.5px;color:rgba(255,255,255,0.45);min-height:28px'>{p['role']}</div>"
            f"<div style='font-size:24px;font-weight:700;margin-top:4px'>{_fmt(p['long_share'])}%"
            "<span style='font-size:11px;font-weight:400;color:rgba(255,255,255,0.5)'> long share</span></div>"
            f"{bar}<div style='font-size:11px;color:{zc};font-weight:700'>"
            f"{_ZONE_LABEL.get(p['zone'], 'NOT RANKED')} · {_ord(pct)} pct</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.6);margin-top:4px'>"
            f"Change 1d {_fmt(p['chg1'],1,True)} · 5d {_fmt(p['chg5'],1,True)} · 20d {_fmt(p['chg20'],1,True)} pts</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.45)'>1-yr range {_fmt(p['min250'])}–"
            f"{_fmt(p['max250'])}% · L/S (long/short) {_fmt(p['ls_ratio'],2)}</div>{opt}</div>",
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────── chart
def _chart(v: dict) -> go.Figure:
    s: pd.DataFrame = v["series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["date"], y=s["close"], name="Nifty 50", yaxis="y1",
                             line=dict(color="rgba(255,255,255,0.85)", width=1.6)))
    fig.add_trace(go.Scatter(x=s["date"], y=s["fii_pct"], name="FII long-share percentile", yaxis="y2",
                             line=dict(color=_PLAYER_COLOR["fii"], width=1.4)))
    fig.add_trace(go.Scatter(x=s["date"], y=s["client_pct"], name="Client long-share percentile",
                             yaxis="y2", line=dict(color=_PLAYER_COLOR["client"], width=1, dash="dot")))
    for m in v["marks"]:
        top = m["kind"] == "TOP"
        fig.add_trace(go.Scatter(
            x=[m["date"]], y=[m["close"]], mode="markers+text", yaxis="y1", showlegend=False,
            marker=dict(symbol="triangle-down" if top else "triangle-up", size=11,
                        color="#ef5350" if top else "#26a69a"),
            text=[f"{m['kind'].title()} {m['date']:%d %b}"], textposition="top center" if top else "bottom center",
            textfont=dict(size=10), hoverinfo="skip"))
    for y, c in ((10, "rgba(239,83,80,0.12)"), (90, "rgba(38,166,154,0.12)")):
        fig.add_hrect(y0=0 if y == 10 else 90, y1=10 if y == 10 else 100, yref="y2",
                      fillcolor=c, line_width=0)
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        yaxis=dict(title="Nifty", gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="percentile (0–100)", overlaying="y", side="right", range=[0, 100],
                    showgrid=False),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    return fig


# ───────────────────────────────────────────────────────────── tables
def _zone_table(v: dict) -> pd.DataFrame:
    rows = []
    for z in v["zone_rows"]:
        rows.append({
            "Today": "◀" if z["today"] else "",
            "FII zone (percentile)": f"{z['label']} ({z['lo']:.0f}–{z['hi']:.0f})",
            "Days": z["days"], "Episodes": z["episodes"],
            "Top close by (%)": z["top_near"], "Bottom close by (%)": z["bottom_near"],
            "Nifty next 10d (%)": z["fwd10"], "Up 10d (%)": z["up10"], "t 10d": z["t10"],
            "Nifty next 20d (%)": z["fwd20"], "t 20d": z["t20"],
        })
    return pd.DataFrame(rows)


def _matrix_table(v: dict) -> pd.DataFrame:
    rows = []
    for m in v["matrix"]:
        rows.append({
            "Today": "◀" if m["today"] else "",
            "FII zone": m["zone"], "FII last 5 sessions": m["flow"], "Nifty vs 20-DMA (20-day moving average)": m["price"],
            "Episodes": m["episodes"], "Top close by (%)": m["top_near"], "Bottom close by (%)": m["bottom_near"],
            "Next 10d (%)": m["fwd10"], "t 10d": m["t10"], "Next 20d (%)": m["fwd20"],
            "Judged": "yes" if m["judged"] else f"no (<{v['params']['min_episodes']} episodes)",
        })
    return pd.DataFrame(rows)


def _ledger_table(v: dict) -> pd.DataFrame:
    rows = []
    for r in reversed(v["ledger"]):
        rows.append({
            "Turn date": r["date"], "Type": r["kind"], "Nifty": r["close"],
            "FII long share (%)": r["fii_ls"], "FII pct": r["fii_pct"],
            "Client pct": r["client_pct"], "Pro pct": r["pro_pct"], "FII options pct": r["fii_opt_pct"],
            "Move to next turn (%)": r["next_move_pct"], "Status": "running" if r["open"] else "",
        })
    return pd.DataFrame(rows)


_NUM = {"format": "%.1f"}


def _num_cfg(df: pd.DataFrame) -> dict:
    cfg = {}
    for c in df.columns:
        if df[c].dtype.kind == "f":
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f" if c.startswith("t ") else "%.1f")
    return cfg


# ───────────────────────────────────────────────────────────── render
def render_positioning_cycle(selected_date: date) -> None:
    _head("🔄 POSITIONING CYCLE — where each player stands against its own last year, "
          "and what that has meant at past turns")
    with st.spinner("Measuring positioning against past swing turns…"):
        v = cached_positioning_cycle(selected_date)
    if v.get("error"):
        st.info(v["error"])
        return

    for w in v["warnings"]:
        if "holiday" in w:
            continue          # shown in the method expander instead; it is housekeeping
        st.warning(w)

    _answer(v)
    st.markdown("")
    _player_cards(v)
    if not v["have_pct"] or v.get("chance_top") is None:
        return            # no percentile yet -> no zones, no odds, nothing honest to chart
    st.plotly_chart(_chart(v), width="stretch", key=f"pos_cycle_chart_{v['data_date']}")
    st.caption(
        f"Swing turns = a closing high or low followed by a {v['params']['turn_pct']:.0f}% reversal "
        "(only marked once confirmed). Shaded bands = the lowest and highest 10% of each player's own past year. "
        "Watch how the blue line reaches its extreme on the turn, not before it."
    )

    with st.expander("📊 FII zone history — what each zone has been followed by", expanded=True):
        cb, ct = v["chance_bottom"], v["chance_top"]
        st.caption(
            f"{v['sessions']:,} sessions since {v['history_from']:%b %Y}. 'Top/Bottom close by' = share of "
            f"episodes followed by a confirmed swing turn {v['params']['near_from']}–{v['params']['near_after']} "
            f"sessions later (a turn that had already happened does not count). After a random day: top "
            f"{ct:.0f}%, bottom {cb:.0f}%. Percentiles restart at NSE's Nov-2024 lot-size reset. "
            f"Next-10/20-session returns start at the next day's open (the file publishes after the close). "
            f"All days: {v['base'][10]:+.2f}% / {v['base'][20]:+.2f}%. "
            f"An episode = first day in the zone, then {v['params']['cooldown']} sessions ignored."
        )
        zt = _zone_table(v)
        st.dataframe(zt, hide_index=True, width="stretch", column_config=_num_cfg(zt))
        ts = v["turn_summary"]
        st.caption(
            f"At the {ts['bottoms']} measured bottoms, FII was in its lowest fifth on {ts['bottoms_fii_low20']} "
            f"(average percentile {_fmt(ts['bottoms_mean_pct'],0)}). At the {ts['tops']} tops it was in its "
            f"highest fifth on only {ts['tops_fii_high80']} (average {_fmt(ts['tops_mean_pct'],0)}). "
            "Bottoms show up in FII positioning far more clearly than tops do."
        )

    with st.expander("🧮 All scenarios — FII zone × last-5-session flow × price trend (20 cells)"):
        mt = _matrix_table(v)
        st.dataframe(mt, hide_index=True, width="stretch", column_config=_num_cfg(mt))
        tc = v.get("today_cell")
        if tc:
            st.caption(
                f"Today: {tc['zone']} · {tc['flow']} · Nifty {tc['price']} — {tc['episodes']} past episodes, "
                f"next 10 sessions {_fmt(tc['fwd10'],2,True,'%')} (t {_fmt(tc['t10'],2,True)}), "
                f"top close by {_fmt(tc['top_near'],0)}%, bottom {_fmt(tc['bottom_near'],0)}%."
            )
        a = v["audit"]["matrix"]
        st.caption(
            f"With {a['cells']} cells, a cell needs |t| above about {a['needed_t']:.1f} before it counts; "
            f"the best cell in the {v['audit']['measured_on']} study reached {a['best_t']:.2f}. "
            "Read the matrix as a map of where turns happened, not as a forecast."
        )

    with st.expander("📅 Every swing turn and where each player stood that day"):
        lt = _ledger_table(v)
        st.dataframe(lt, hide_index=True, width="stretch", column_config=_num_cfg(lt))

    with st.expander("🔬 Which ratios were tested, and why none of them is a turn signal"):
        a, ll = v["audit"], v["lead_lag"]
        st.markdown(
            f"""
**Question tested:** is there a long/short ratio, put/call ratio, option-posture ratio or
participant gap that reaches a known level *before* Nifty tops or bottoms?

**Ratios tested** (each read as a percentile of its own trailing year, for FII, Pro, Client and DII):
index-futures long share, index-options bullish-leg share, FII calls-vs-puts bought, FII put-writing
share, FII stock-futures long share, FII all-legs bullish share, FII-minus-Client long-share gap.

**1. Positioning follows price.** Correlation of the 5-session change in FII long share with the
*previous* 5 sessions of Nifty: **{_fmt(ll['fii_vs_past5'],2,True)}**; with the *next* 5 sessions:
**{_fmt(ll['fii_vs_next5'],2,True)}** (live, up to this date). FIIs add shorts while the market falls
and cover while it rises. Their daily change correlates **{_fmt(ll['fii_client_corr'],2,True)}** with
Client's — the file is zero-sum, so 'FII vs Client' is one measurement seen from two sides.

**2. The extreme arrives on the turn, not before it.** Around 26 bottoms the FII percentile averages
about {a['event_profile_bottoms']['m20']} twenty sessions before, about {a['event_profile_bottoms']['m5']} five
sessions before, and about {a['event_profile_bottoms']['day']} on the bottom day itself.
There is no level that is hit first and then followed by the turn.

**3. 'They turned before price' (divergence) does not work.** Price at a 20-session low while the FII
ratio is already off its own low: next 10 sessions {a['divergence']['fii_bull_fwd10']:+.2f}% (n={a['divergence']['fii_bull_n']})
vs {a['divergence']['sibling_fwd10']:+.2f}% when the ratio is still at its low (t {a['divergence']['t']:+.2f}).
No participant's divergence beat its sibling (best |t| {a['divergence']['best_abs_t_any_player']:.2f}).

**4. Turn rules, all at once.** {a['rule_family']['cells']} rule × horizon cells (low ≤10, high ≥90,
'low and turning up', 'high and turning down', for every ratio). Best |t| {a['rule_family']['best_t']:.2f};
shuffled data produces a best of {a['rule_family']['null_median']:.2f} on a typical run and
{a['rule_family']['null_95']:.2f} one time in twenty → family p = **{a['rule_family']['p']:.2f}**.

**What survived:** being near a turn. After 'FII long share in its lowest 10%', a 5% bottom followed
within 10 sessions more often than after a random day (see the zone table for the live numbers). It
narrows *when* to pay attention; it does not say *which day*.

**Data handling:** until 16 Nov 2026 the participant table held {a['copied_days_repaired']} days
whose file actually belonged to another date with the same day-of-month (for example, 09 Jan 2026's
file stored again under 09 Feb and 09 Apr). They were re-downloaded from NSE and replaced, and the
fetchers now refuse any file whose title names a different date. Raw contract counts are never compared
across years because lot sizes
changed five times; FII long share itself moved from a ~45% median (2018-24) to ~11% (2026), which is
why no fixed threshold such as "below 30%" can mean the same thing twice. Monthly expiry days move the
ratio about four times more than a normal day and are flagged. Full study:
`scripts/study_fii_turning_points.py` (measured {a['measured_on']}).
"""
        )
