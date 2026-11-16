"""
Big Players (F&O) — redesigned page body (2026-11-14).

Reading order, top to bottom:
  1. TODAY AT A GLANCE   one sentence, four player tiles, who-bought-from-whom chart
  2. a section switch    (only the chosen section renders — keeps the page fast)
       🧭 Scenario Playbook   today's scenario + what every scenario was followed by
       🔄 Positioning Cycle   each player vs its own last year, odds of a nearby turn
       👥 Player Deep-Dive    one participant's year: share, net futures, options
       (Money Flow, Raw Data and How-to-read are rendered by fao_tracker.py)

Design rule for every number here: it is measured from data dated on or before the
selected date, and it is shown next to what ALL days did, so a reader can see when a
scenario is ordinary.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.analytics.big_players import FLOW_STATES, PLAYER_ORDER

# ── palette / labels ────────────────────────────────────────────────────────
PCOL = {"FII": "#2196f3", "Pro": "#9c27b0", "Client": "#ff9800", "DII": "#4caf50"}
ROLE = {
    "FII": "Foreign institutions — biggest directional book",
    "Pro": "Broker prop desks — mostly hedging option books",
    "Client": "Retail + HNI — opposite side of FII by arithmetic",
    "DII": "Mutual funds, insurers — hedge their cash holdings",
}
FLOWCOL = {"LONG BUILD": "#26a69a", "SHORT COVER": "#66bb6a", "SHORT BUILD": "#ef5350",
           "LONG UNWIND": "#ff7043", "NO CHANGE": "#9e9e9e"}
SCENCOL = {"FII-backed rally": "#26a69a", "Rally against FII": "#ffb74d",
           "FII-backed fall": "#ef5350", "FII buying the dip": "#42a5f5",
           "Rally, FII on sidelines": "#b0bec5", "Fall, FII on sidelines": "#78909c"}
LEG_MEANING = {
    "FII-backed rally": "Nifty rose and FII's book turned more long during the rise.",
    "Rally against FII": "Nifty rose while FII's book turned more short — FII fought the rise.",
    "Rally, FII on sidelines": "Nifty rose but FII's long share barely moved (under 1 point).",
    "FII-backed fall": "Nifty fell and FII's book turned more short during the fall.",
    "FII buying the dip": "Nifty fell while FII's book turned more long — FII bought the fall.",
    "Fall, FII on sidelines": "Nifty fell but FII's long share barely moved (under 1 point).",
}
ZONE_LABEL = {"xs": "EXTREME SHORT", "s": "SHORT", "m": "MIDDLE", "l": "LONG", "xl": "EXTREME LONG"}
ZONE_COL = {"xs": "#ef5350", "s": "#ff9800", "m": "#9e9e9e", "l": "#66bb6a", "xl": "#26a69a"}
MUTED = "rgba(255,255,255,0.55)"


def _n(v, sign=True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+,.0f}" if sign else f"{v:,.0f}"


def _p(v, nd=2, sign=True, suf="%") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{nd}f}{suf}" if sign else f"{v:.{nd}f}{suf}"


def _ord(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    n = int(round(v))
    if n >= 100 and v < 100:
        n = 99
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def section_title(num: str, title: str, sub: str = "") -> None:
    st.markdown(
        f"<div style='margin:18px 0 8px'><span style='font-size:12px;font-weight:800;"
        f"color:#90caf9;letter-spacing:1.2px'>{num}</span> "
        f"<span style='font-size:17px;font-weight:700'>{title}</span>"
        + (f"<div style='font-size:12px;color:{MUTED};margin-top:2px'>{sub}</div>" if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _badge(text: str, color: str) -> str:
    return (f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;"
            f"font-weight:700;background:{color}22;color:{color};border:1px solid {color}66'>{text}</span>")


def _verdict_words(t, n_ep, min_ep) -> tuple[str, str]:
    if n_ep is None or n_ep < min_ep or t is None:
        return "TOO FEW CASES", "#9e9e9e"
    if abs(t) < 2:
        return "SAME AS A NORMAL DAY", "#9e9e9e"
    return ("BETTER THAN NORMAL" if t > 0 else "WORSE THAN NORMAL"), ("#26a69a" if t > 0 else "#ef5350")


# ══════════════════════════════════════════════════════════════ 1. TODAY
def _flow_sentence(p: dict, five: bool = False) -> str:
    dl = p["d_long5"] if five else p["d_long"]
    ds = p["d_short5"] if five else p["d_short"]
    parts = []
    if dl:
        parts.append(f"{'added' if dl > 0 else 'cut'} {abs(dl):,.0f} longs")
    if ds:
        parts.append(f"{'added' if ds > 0 else 'covered'} {abs(ds):,.0f} shorts")
    return " and ".join(parts) if parts else "left futures unchanged"


def render_today(a: dict, pb: dict | None) -> None:
    section_title("1", "Today at a glance",
                  f"Participant-wise open interest (NSE file) for {a['data_date']:%A %d %b %Y}, "
                  "index futures and index options, all four categories.")
    for w in a["warnings"]:
        st.warning(w)

    ok = {p["name"]: p for p in a["players"] if not p["missing"]}
    fii = ok.get("FII")
    if not fii:
        st.info("No FII row in the file for this date.")
        return

    st1 = FLOW_STATES[fii["state1"]][0]
    st5 = FLOW_STATES[fii["state5"]][0]
    absorb = a.get("absorb") or {}
    ab_txt = (" and ".join(f"{k} {v:.0f}%" for k, v in sorted(absorb.items(), key=lambda x: -x[1]))
              if absorb else "—")
    scen = None
    if pb and not pb.get("error"):
        scen = next((s for s in pb["scenarios"] if s["today"]), None)
    scen_html = ""
    if scen:
        c = SCENCOL.get(scen["name"], "#9e9e9e")
        scen_html = (f"<div style='margin-top:6px'>5-session picture: FII <b style='color:{FLOWCOL[st5]}'>{st5}</b> "
                     f"({_flow_sentence(fii, True)}), Nifty {_p(a['nifty_chg5'])} → scenario "
                     f"<b style='color:{c}'>{scen['name']}</b>. See section 2 for what it has meant before.</div>")
    nc = a["nifty_chg"] or 0
    st.markdown(
        "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:12px 16px;"
        "background:rgba(33,150,243,0.06);font-size:14px;line-height:1.6'>"
        f"<b>Nifty {a['nifty_close']:,.0f}</b> "
        f"<span style='color:{'#26a69a' if nc >= 0 else '#ef5350'}'>({_p(nc)})</span>. "
        f"FII <b style='color:{FLOWCOL[st1]}'>{st1}</b> today: {_flow_sentence(fii)} "
        f"(net {_n(fii['net_change'])} contracts). "
        + (f"The other side was taken by {ab_txt}." if absorb else "")
        + scen_html + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    cols = st.columns([1.25, 1, 1, 1])
    for col, name in zip(cols, PLAYER_ORDER):
        p = ok.get(name)
        if not p:
            col.info(f"{name}: no row")
            continue
        c = PCOL[name]
        s1, s5 = FLOW_STATES[p["state1"]][0], FLOW_STATES[p["state5"]][0]
        zc = ZONE_COL.get(p["zone"], "#777")
        pct = p["pct"]
        bar = (f"<div style='height:6px;background:rgba(255,255,255,0.08);border-radius:3px;margin:5px 0 2px'>"
               f"<div style='height:6px;width:{pct:.0f}%;background:{zc};border-radius:3px'></div></div>"
               if pct is not None else "")
        opt = p["opt"]
        col.markdown(
            f"<div style='border:1px solid rgba(255,255,255,0.1);border-top:3px solid {c};border-radius:10px;"
            "padding:10px 12px;background:rgba(255,255,255,0.02);min-height:270px'>"
            f"<div style='font-weight:800;font-size:16px;color:{c}'>{name}"
            + (" <span style='font-size:10px;color:#90caf9'>★ market's biggest book</span>" if name == "FII" else "")
            + f"</div><div style='font-size:10.5px;color:{MUTED};min-height:28px'>{ROLE[name]}</div>"
            f"<div style='margin:6px 0 4px'>{_badge('TODAY · ' + s1, FLOWCOL[s1])} "
            f"{_badge('5 DAYS · ' + s5, FLOWCOL[s5])}</div>"
            f"<div style='font-size:12px'>Longs <b>{_n(p['d_long'])}</b> · Shorts <b>{_n(p['d_short'])}</b> "
            f"· Net <b>{_n(p['net_change'])}</b></div>"
            f"<div style='font-size:11px;color:{MUTED}'>5 days: net {_n(p['net_change5'])}</div>"
            f"<div style='font-size:22px;font-weight:700;margin-top:6px'>{_p(p['long_share'],1,False)}"
            f"<span style='font-size:11px;font-weight:400;color:{MUTED}'> of futures book is long</span></div>"
            f"{bar}<div style='font-size:11px;color:{zc};font-weight:700'>"
            f"{ZONE_LABEL.get(p['zone'],'NOT RANKED')} · {_ord(pct)} percentile of last year</div>"
            f"<div style='font-size:11px;color:{MUTED};margin-top:6px'>Options today: calls "
            f"{_n(opt['opt_idx_call_long'])} bought / {_n(opt['opt_idx_call_short'])} sold · puts "
            f"{_n(opt['opt_idx_put_long'])} bought / {_n(opt['opt_idx_put_short'])} sold</div>"
            f"<div style='font-size:11px;font-weight:700;color:"
            f"{'#26a69a' if p['opt_sign'] > 0 else '#ef5350' if p['opt_sign'] < 0 else MUTED}'>"
            f"→ {p['opt_read']}</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(_zero_sum_chart(a), width="stretch", key=f"bp_zero_{a['data_date']}")
    with c2:
        st.plotly_chart(_options_chart(a), width="stretch", key=f"bp_opt_{a['data_date']}")
    st.caption(
        "Left: every futures contract bought is sold by someone, so the four bars always add to zero. "
        "Watch who sits opposite FII. Right: change in each option leg today. 'Sold' includes both "
        "writing new options and closing old long ones. One leg alone does not make a direction."
    )


def _zero_sum_chart(a: dict) -> go.Figure:
    ok = [p for p in a["players"] if not p["missing"]]
    names = [p["name"] for p in ok]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[p["net_change"] for p in ok], orientation="h", name="Today",
                         marker_color=[PCOL[n] for n in names],
                         text=[_n(p["net_change"]) for p in ok], textposition="auto"))
    fig.add_trace(go.Bar(y=names, x=[p["net_change5"] for p in ok], orientation="h", name="Last 5 sessions",
                         marker_color=[PCOL[n] for n in names], opacity=0.35))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)")
    fig.update_layout(title=dict(text="Who bought and who sold index futures (net contracts)", font=dict(size=13)),
                      height=260, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.15), yaxis=dict(autorange="reversed"),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False))
    return fig


def _options_chart(a: dict) -> go.Figure:
    ok = [p for p in a["players"] if not p["missing"] and p["name"] != "DII"]
    legs = [("opt_idx_call_long", "Calls bought", "#26a69a"), ("opt_idx_call_short", "Calls sold", "#80cbc4"),
            ("opt_idx_put_long", "Puts bought", "#ef5350"), ("opt_idx_put_short", "Puts sold", "#ef9a9a")]
    fig = go.Figure()
    for k, lab, c in legs:
        fig.add_trace(go.Bar(x=[p["name"] for p in ok], y=[p["opt"][k] for p in ok], name=lab, marker_color=c))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)")
    fig.update_layout(title=dict(text="Index options — change in each leg today (contracts)", font=dict(size=13)),
                      height=260, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    return fig


# ══════════════════════════════════════════════════════════════ 2. PLAYBOOK
def render_playbook(pb: dict) -> None:
    section_title("2", "Scenario Playbook — who drove each swing, and what it means now",
                  f"Part A maps every rally and fall of the last year to what FII did during it. "
                  f"Part B asks whether today's 5-session scenario says anything about what comes next.")
    _render_swings(pb)
    st.divider()
    _render_forward(pb)


def _chips(names: list[str]) -> str:
    return "".join(
        f"<div style='display:flex;gap:8px;align-items:flex-start;margin:3px 0'>"
        f"<span style='flex:0 0 14px;height:14px;border-radius:3px;background:{SCENCOL[n]};margin-top:2px'></span>"
        f"<span style='font-size:12px'><b style='color:{SCENCOL[n]}'>{n}</b> — "
        f"<span style='color:{MUTED}'>{LEG_MEANING[n]}</span></span></div>" for n in names)


def _book_view(pb: dict, key: str) -> dict:
    """pb re-pointed at one FII book, so every swing widget reads the same keys."""
    bk = (pb.get("books") or {}).get(key)
    if not bk:
        return pb
    h = pb["history"].copy()
    if key == "opt":
        h["fii_share"], h["fii_pct"] = h["opt_share"], h["opt_pct"]
        h["p10"], h["p90"] = h["opt_p10"], h["opt_p90"]
    return {**pb, "history": h, "legs": bk["legs"], "leg_stats": bk["leg_stats"],
            "current_leg": bk["current_leg"], "peaks": bk["peaks"], "legs_total": bk["legs_total"],
            "book": bk}


def _book_compare(pb: dict) -> None:
    """Both FII books side by side for today — the first thing to check is whether they agree."""
    books = pb.get("books") or {}
    if len(books) < 2:
        return
    cols = st.columns(2)
    for col, key in zip(cols, ("fut", "opt")):
        bk = books[key]
        cur = bk.get("current_leg") or {}
        pc_ = bk.get("today_pct")
        zone = ("HEAVY LONG" if pc_ is not None and pc_ >= 90 else "HEAVY SHORT" if pc_ is not None and pc_ <= 10
                else "middle")
        zc = "#26a69a" if zone == "HEAVY LONG" else "#ef5350" if zone == "HEAVY SHORT" else MUTED
        sc = SCENCOL.get(cur.get("scenario", ""), MUTED)
        swing = (f"Current swing: <b style='color:{sc}'>{cur['scenario']}</b> "
                 f"({_p(cur.get('fii_share_chg'),1,True,' pts')} since {cur['start']:%d %b})"
                 if cur else "No swing in progress.")
        col.markdown(
            "<div style='border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:9px 13px;"
            "background:rgba(255,255,255,0.02);font-size:12.5px;line-height:1.55'>"
            f"<div style='font-size:11px;font-weight:800;letter-spacing:1px;color:{MUTED}'>FII {bk['name'].upper()}</div>"
            f"<b style='font-size:18px'>{_p(bk.get('today_value'),1,False)}%</b> "
            f"{bk['metric'].replace('FII ', '').replace(' %', '')} · "
            f"<b style='color:{zc}'>{_ord(pc_)} percentile of its year — {zone}</b><br>{swing}</div>",
            unsafe_allow_html=True)
    fz, oz = books["fut"].get("today_pct"), books["opt"].get("today_pct")
    if fz is not None and oz is not None and ((fz >= 90) != (oz >= 90) or (fz <= 10) != (oz <= 10)):
        st.caption(
            f"The two books disagree on how extreme FII is today ({_ord(fz)} vs {_ord(oz)} percentile). "
            "Switch between them below to see each book's own swing record. Measured over 2018–2026: for "
            "warning of BOTTOMS the options book has the stronger record; for TOPS neither book is reliable.")


def _render_swings(pb: dict) -> None:
    st.markdown("#### A · Every swing of the last year — and whether FII was behind it")
    _book_compare(pb)
    if not pb.get("legs"):
        st.info("Not enough history to draw swings.")
        return
    c1, c2 = st.columns(2)
    c1.markdown(_chips(["FII-backed rally", "Rally against FII", "Rally, FII on sidelines"]),
                unsafe_allow_html=True)
    c2.markdown(_chips(["FII-backed fall", "FII buying the dip", "Fall, FII on sidelines"]),
                unsafe_allow_html=True)
    keys = ("fut", "opt") if pb.get("books") else ("fut",)
    for i, key in enumerate(keys):
        _render_book_graph(_book_view(pb, key), number=f"A{i + 1}")
    if pb.get("books"):
        _render_turn_levels(pb)
        tabs = st.tabs([f"📊 {pb['books'][k]['name']} — swing statistics" for k in keys])
        for tab, key in zip(tabs, keys):
            with tab:
                v = _book_view(pb, key)
                _render_leg_stats(v)
                _render_peaks(v)
                _render_swing_table(v)


def _render_positions_now(pb: dict) -> None:
    """Where each side of the book stands against its last position peak / bottom."""
    bk = pb.get("book") or {}
    now = bk.get("side_now") or {}
    if not now:
        return
    is_fut = bk.get("key") != "opt"
    names = {"L": "FII longs" if is_fut else "FII bullish side",
             "S": "FII shorts" if is_fut else "FII bearish side"}
    cols = st.columns(2)
    for col, side, colour in zip(cols, ("L", "S"), ("#26a69a", "#ef5350")):
        n = now.get(side)
        if not n:
            continue
        rising = n["direction"] == "rising"
        if is_fut:
            word = ("building" if rising else "unwinding") if side == "L" else ("building" if rising else "covering")
        else:
            word = "building" if rising else ("unwinding" if side == "L" else "covering")
        col.markdown(
            f"<div style='border-left:3px solid {colour};padding:6px 12px;font-size:12.5px;line-height:1.55'>"
            f"<b style='color:{colour}'>{names[side].upper()}: {n['now']:,.0f}</b> "
            f"<span style='color:{MUTED}'>(5-session median)</span> — {word} since the "
            f"{'bottom' if n['last_kind'] == 'TROUGH' else 'peak'} of {n['last_value']:,.0f} on "
            f"{n['last_date']:%d %b} ({_p(n['chg_pct'],0)} in {n['sessions']} sessions). "
            f"{'High' if rising else 'Low'} since then: {n['extreme_value']:,.0f} on {n['extreme_date']:%d %b}"
            f" — now {_p(n['off_extreme_pct'],0)} from it"
            + (f"; a 30% turn the other way would mark a new {'peak' if rising else 'bottom'}."
               if abs(n['off_extreme_pct'] or 0) < 30 else ".")
            + "</div>", unsafe_allow_html=True)
    st.caption(
        "How to use this: position peaks and bottoms line up with Nifty's swing turns at a median of 0 sessions "
        "(about a third before, a quarter the same day, the rest after) — about 4 in 10 futures turns and 1 in 3 "
        "options turns fall within 5 sessions of a Nifty turn, against about 1 in 4 by chance. So a fresh "
        "position peak CONFIRMS a turn as it happens; it does not warn ahead. A simple rule such as 'longs fell 10% "
        "off their 20-day high' did not beat chance in 2018–2026 (scripts/study_fii_position_peaks.py).")


def _render_book_graph(pb: dict, number: str) -> None:
    bk = pb.get("book") or {"metric": "FII long %", "name": "Index futures", "key": "fut",
                            "definition": "FII longs ÷ (longs + shorts) in index futures",
                            "thresholds": {"flat": 1.0, "fragile": 3.0}}
    legs = pb.get("legs") or []
    st.markdown(f"##### {number} · FII in **{bk['name'].upper()}** — {bk['metric']}", unsafe_allow_html=True)
    st.plotly_chart(_swing_chart(pb), width="stretch", key=f"bp_swing_{pb['data_date']}_{bk['name']}")
    st.caption(
        f"A swing ends when Nifty reverses {pb['leg_pct']:.0f}% from its high or low (▲ ▼ marks). Each band is one "
        f"swing. The label shows Nifty's move and how much {bk['metric']} changed across it (under "
        f"{bk['thresholds']['flat']:g} pt = FII on the sidelines). Second panel: FII's actual positions — "
        + ("longs and shorts in contracts (before 30 Dec 2025 converted to today's lot size); "
           if bk.get("key") != "opt" else
           "the bullish side and the bearish side in contracts. In options both sides swell and shrink together "
           "with every monthly expiry, so which side grows FASTER (the bottom panel) matters more than the levels; ")
        + "▲▼ mark position peaks and bottoms (a 30% swing in the 5-session median). Third panel: what FII did "
        "over the last 5 sessions — bars above zero are bullish actions, below zero bearish. Expiring contracts "
        "leave the book on every " + ("monthly" if bk.get("key") != "opt" else "weekly and monthly")
        + " expiry, so part of the unwinding/covering around those dates is settlement, not a decision; dotted lines mark "
        f"the Nifty turns. Bottom panel: {bk['metric']} = {bk['definition']} "
        "— watch it climb through green bands and sink through red ones. Swing ends are only known afterwards, "
        "so this is a map of what happened, not a signal."
        + (" Weekly option expiries roughly double the day's change in this ratio; each swing's ends are "
           "smoothed over up to 3 sessions so one expiry day cannot flip its tag." if bk.get("key") == "opt" else "")
    )

    _render_positions_now(pb)
    cur = pb.get("current_leg")
    if cur:
        col = SCENCOL[cur["scenario"]]
        acts = bk.get("actions") or {}
        st_now = (pb.get("book") or {}).get("today_state")
        stats = {r["scenario"]: r for r in pb["leg_stats"]}
        peer = stats.get(cur["scenario"], {})
        # the swing only ends on a reversal from its EXTREME, not from today's close
        rev = (cur["extreme_close"] * (1 - pb["leg_pct"] / 100) if cur["dir"] > 0
               else cur["extreme_close"] * (1 + pb["leg_pct"] / 100))
        st.markdown(
            f"<div style='border:2px solid {col};border-radius:10px;padding:12px 16px;background:{col}14;"
            "font-size:13.5px;line-height:1.6;margin:6px 0 12px'>"
            f"<div style='font-size:11px;font-weight:800;letter-spacing:1px;color:{col}'>SWING IN PROGRESS · "
            f"{cur['scenario'].upper()}</div>"
            f"Since the {cur['start']:%d %b} {'bottom' if cur['dir'] > 0 else 'top'} "
            f"({cur['nifty_from']:,.0f}): Nifty <b>{_p(cur['move_pct'],1)}</b> in {cur['sessions']} sessions. "
            f"{bk['metric']} {_p(cur['fii_share_from'],1,False)} → <b>{_p(cur['fii_share_to'],1,False)}</b> "
            f"({_p(cur['fii_share_chg'],1,True,' pts')}), FII net {_n(cur['fii_net_chg'])} contracts; "
            f"Client's same ratio {_p(cur['client_share_chg'],1,True,' pts')}."
            + (f" Last 5 sessions: <b>{acts.get(st_now, st_now)}</b>." if st_now and st_now != "FLAT" else "")
            + (f"<br>Compared with the {cur.get('peers', 0)} finished {cur['scenario'].lower()} swings since "
               f"{pb['history_from']:%Y}: this one is already bigger than <b>{cur.get('bigger_than_pct', 0):.0f}%</b> "
               f"and longer than {cur.get('longer_than_pct', 0):.0f}% of them (typical: "
               f"{_p(peer.get('median_move'),1,False)} over {peer.get('median_sessions', 0):.0f} sessions)."
               if cur.get("peers") else "")
            + f"<br><span style='color:{MUTED}'>The swing ends only when Nifty closes {pb['leg_pct']:.0f}% "
            f"{'below its high' if cur['dir'] > 0 else 'above its low'} of {cur['extreme_close']:,.0f} "
            f"({cur['extreme_date']:%d %b}) — a close {'below' if cur['dir'] > 0 else 'above'} about "
            f"<b>{rev:,.0f}</b>. A new {'high' if cur['dir'] > 0 else 'low'} moves that level."
            + (f" Of the {cur['reached_n']} finished {'rallies' if cur['dir'] > 0 else 'falls'} that got this far, "
               f"<b>{cur['extended_pct']:.0f}%</b> went at least 3% further before ending — size alone does not "
               "say the swing is over." if cur.get("reached_n") else
               " No finished swing in this direction has been this large, so there is no comparison.")
            + "</span></div>",
            unsafe_allow_html=True,
        )



def _turn_strip(pb: dict) -> go.Figure:
    """Percentile of each book's ratio at every past swing top / bottom, plus today."""
    fig = go.Figure()
    rng = np.random.default_rng(1)
    for key, y in (("fut", 1.0), ("opt", 0.0)):
        bk = pb["books"][key]
        for kind, dy, sym, col in (("TOP", 0.18, "triangle-down", "#ef5350"),
                                   ("BOTTOM", -0.18, "triangle-up", "#26a69a")):
            pts = [r for r in bk["turn_pcts"] if r["kind"] == kind and r["pct"] is not None]
            fig.add_trace(go.Scatter(
                x=[r["pct"] for r in pts], y=y + dy + rng.uniform(-0.07, 0.07, len(pts)), mode="markers",
                marker=dict(symbol=sym, size=9, color=col, opacity=0.8, line=dict(width=0.5, color="#111")),
                text=[f"{kind.title()} {r['date']:%d %b %Y} · {bk['metric']} {r['value']:.1f}% "
                      f"({_ord(r['pct'])} pct)" for r in pts],
                hovertemplate="%{text}<extra></extra>", showlegend=False))
        if bk.get("today_pct") is not None:
            fig.add_trace(go.Scatter(x=[bk["today_pct"]] * 2, y=[y - 0.38, y + 0.38], mode="lines+text",
                                     line=dict(color="white", width=3), text=["", f"TODAY {_ord(bk['today_pct'])}"],
                                     textposition=("top left" if bk["today_pct"] > 85 else
                                                   "top right" if bk["today_pct"] < 15 else "top center"),
                                     textfont=dict(size=11, color="white"),
                                     hoverinfo="skip", showlegend=False))
    for x0, x1, c in ((0, 10, "rgba(239,83,80,0.13)"), (90, 100, "rgba(38,166,154,0.13)")):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0, layer="above")
    fig.update_layout(
        height=330, margin=dict(l=120, r=20, t=40, b=60),
        title=dict(text="Where FII's ratio stood (percentile of its own year) at every swing ▼ top and "
                        "▲ bottom since 2018 — and today", font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1, 101], ticksuffix="th", title_standoff=12,
                   title="percentile of the trailing year  (red band = FII heavy short · green band = FII heavy long)",
                   gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(tickvals=[1.0, 0.0], ticktext=["Index FUTURES<br>long %", "Index OPTIONS<br>bullish %"],
                   range=[-0.6, 1.6], showgrid=False))
    return fig


def _render_turn_levels(pb: dict) -> None:
    st.markdown("#### A3 · At what level — and doing what — did the market turn?")
    st.plotly_chart(_turn_strip(pb), width="stretch", key=f"bp_turnstrip_{pb['data_date']}")
    lines = []
    for key in ("fut", "opt"):
        bk = pb["books"][key]
        t, b = bk["peaks"]["tops"], bk["peaks"]["bottoms"]
        today = bk.get("today_pct")
        lo_ = bk.get("level_odds") or {}
        odds = (f"From days with the ratio at {lo_['band'][0]:.0f}–{lo_['band'][1]:.0f} percentile (today's "
                f"{_ord(today)} ± 10; {lo_['episodes']} episodes), a 3% top followed within 10 sessions "
                f"<b>{_p(lo_.get('top_after'),0,False)}</b> of the time and a bottom "
                f"<b>{_p(lo_.get('bottom_after'),0,False)}</b> — against {_p(lo_.get('top_base'),0,False)} / "
                f"{_p(lo_.get('bottom_base'),0,False)} after any day."
                if lo_.get("episodes") else f"Today's {_ord(today)} percentile has too few past episodes to rate.")
        lines.append(
            f"<b>{bk['name']}</b>: bottoms came with {bk['metric']} in its bottom fifth "
            f"<b>{b.get('share_lo', 0):.0f}%</b> of the time (typical {_ord(b.get('median_pct'))} pct); tops with it "
            f"in its top fifth only <b>{t.get('share_hi', 0):.0f}%</b> (typical {_ord(t.get('median_pct'))}). "
            + odds)
    st.markdown("<div style='font-size:13px;line-height:1.65'>" + "<br>".join(lines) + "</div>",
                unsafe_allow_html=True)

    st.markdown("##### What FII was DOING just before and just after the turns")
    acol = {"LB": "#26a69a", "SC": "#42a5f5", "SB": "#ef5350", "LU": "#ff9800"}
    order = ("LB", "SC", "SB", "LU")
    for key in ("fut", "opt"):
        bk = pb["books"][key]
        ta = bk["turn_actions"]
        acts = bk["actions"]
        base = ta.get("base", {})
        head = "".join(f"<th style='padding:4px 8px;color:{acol[k]}'>{acts[k]}</th>" for k in order)
        body = ""
        for kind in ("TOP", "BOTTOM"):
            for when, lab in (("before", "5 sessions BEFORE (ending on it)"), ("after", "5 sessions AFTER")):
                cells = ""
                for k in order:
                    v = (ta.get(kind, {}).get(when) or {}).get(k)
                    bse = base.get(k) or 0
                    hot = v is not None and bse and v >= 1.35 * bse
                    cold = v is not None and bse and v <= 0.6 * bse
                    sty = (f"color:{acol[k]};font-weight:800" if hot else
                           "color:rgba(255,255,255,0.35)" if cold else "")
                    cells += f"<td style='padding:4px 8px;text-align:center;{sty}'>{_p(v,0,False)}</td>"
                tag, tc = ("▼ TOP", "#ef5350") if kind == "TOP" else ("▲ BOTTOM", "#26a69a")
                body += f"<tr><td style='padding:4px 8px'><b style='color:{tc}'>{tag}</b> · {lab}</td>{cells}</tr>"
        body += (f"<tr><td style='padding:4px 8px;color:{MUTED}'>all days (normal)</td>"
                 + "".join(f"<td style='padding:4px 8px;text-align:center;color:{MUTED}'>{_p(base.get(k),0,False)}</td>"
                           for k in order) + "</tr>")
        st.markdown(
            f"<div style='font-size:12px;font-weight:800;letter-spacing:1px;color:{MUTED};margin-top:8px'>"
            f"FII {bk['name'].upper()} — {ta.get('TOP', {}).get('n', 0)} tops, "
            f"{ta.get('BOTTOM', {}).get('n', 0)} bottoms (3% swings since 2018)</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:6px'>"
            f"<tr><th style='padding:4px 8px;text-align:left'>Window</th>{head}</tr>{body}</table>",
            unsafe_allow_html=True)
    st.info(
        "**How to read this.** Bold = at least 1.35× its normal share; faded = well below normal. "
        "**FII keeps pressing the old direction INTO the turn and flips AFTER it.** Before a top they are "
        "still building longs (futures) and almost never adding bearish options; in the week after a top the "
        "bearish buildup jumps (options: about half of all tops, p ≈ 0.001). Bottoms mirror it. So the action "
        "panel cannot warn of a peak in advance — but the **first week of flipped action is the best available "
        "confirmation** that a turn has happened, and the options book flips most clearly. "
        "(Tests: scripts/study_fii_actions_at_turns.py, 32 cells. Passing a strict 32-test correction: options "
        "bearish buildup before tops (11%) and after tops (53%), options bullish buildup after tops (11%), and options "
        "bearish buildup after bottoms (4%). Suggestive only (p 0.002–0.04): every futures cell and options bullish "
        "buildup after bottoms (53%, p 0.009).)")


def _render_swing_table(pb: dict) -> None:
    bk = pb.get("book") or {"metric": "FII long %", "thresholds": {"fragile": 3.0}}
    legs = pb.get("legs") or []
    with st.expander("📅 Swing-by-swing table (last year)", expanded=False):
        rows = []
        for g in reversed(legs):
            rows.append({
                "From": g["start"], "To": g["end"], "Sessions": g["sessions"],
                "Swing type": g["scenario"] + (" (running)" if g["open"] else "")
                + (" · partial" if g.get("partial") else ""),
                "Tag firm?": (f"fragile (<{bk['thresholds']['fragile']:g} pts)" if g.get("fragile") else "firm"),
                "Nifty from": g["nifty_from"], "Nifty to": g["nifty_to"], "Nifty move %": g["move_pct"],
                f"{bk['metric']} from": g["fii_share_from"], "to %": g["fii_share_to"],
                "FII change (pts)": g["fii_share_chg"], "FII net contracts": g["fii_net_chg"],
                "Client change (pts)": g["client_share_chg"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Nifty from": st.column_config.NumberColumn(format="%.0f"),
            "Nifty to": st.column_config.NumberColumn(format="%.0f"),
            "Nifty move %": st.column_config.NumberColumn(format="%+.1f"),
            f"{bk['metric']} from": st.column_config.NumberColumn(format="%.1f"),
            "to %": st.column_config.NumberColumn(format="%.1f"),
            "FII change (pts)": st.column_config.NumberColumn(format="%+.1f"),
            "FII net contracts": st.column_config.NumberColumn(format="%+,.0f"),
            "Client change (pts)": st.column_config.NumberColumn(format="%+.1f"),
        })


def _render_leg_stats(pb: dict) -> None:
    st.markdown(f"##### What {pb['legs_total']} swings since {pb['history_from']:%Y} show")
    rows = {(r["dir"], r["side"]): r for r in pb["leg_stats"]}
    c1, c2 = st.columns(2)
    for col, d, word in ((c1, 1, "RALLIES"), (c2, -1, "FALLS")):
        lines = []
        for side in ((1, -1, 0) if d > 0 else (-1, 1, 0)):
            r = rows.get((d, side), {})
            if not r.get("count"):
                continue
            n = r["scenario"]
            lines.append(
                f"<div style='margin:6px 0'><div style='display:flex;justify-content:space-between'>"
                f"<b style='color:{SCENCOL[n]}'>{n}</b><b>{r['share_of_dir']:.0f}% of {word.lower()}</b></div>"
                f"<div style='height:8px;background:rgba(255,255,255,0.07);border-radius:4px;margin:3px 0'>"
                f"<div style='height:8px;width:{r['share_of_dir']:.0f}%;background:{SCENCOL[n]};border-radius:4px'>"
                f"</div></div><div style='font-size:11.5px;color:{MUTED}'>{r['count']} swings · typical size "
                f"{r['median_move']:.1f}% over {r['median_sessions']:.0f} sessions · "
                f"{r['big_share']:.0f}% grew past 8% · Client on the other side in {r['client_opposite']:.0f}%</div></div>")
        col.markdown(
            "<div style='border:1px solid rgba(255,255,255,0.1);border-radius:10px;padding:10px 14px;"
            f"background:rgba(255,255,255,0.02)'><div style='font-size:11px;font-weight:800;letter-spacing:1px;"
            f"color:{MUTED}'>{word}</div>{''.join(lines)}</div>",
            unsafe_allow_html=True,
        )
    up = rows.get((1, 1), {})
    vs = rows.get((1, -1), {})
    dn = rows.get((-1, -1), {})
    st.info(
        f"**How to read this.** FII ends up on the side of most swings — {up.get('share_of_dir', 0):.0f}% "
        f"of rallies and {dn.get('share_of_dir', 0):.0f}% of falls (if FII's book moved at random it would be "
        "about 4–5 in 10). That is what 'FII rules the market' looks like in the data. Swings tagged "
        f"'fragile' in the table rest on a change of under {(pb.get('book') or {}).get('thresholds', {}).get('fragile', 3):g} "
        "points and could flip with one day's data. "
        "Rallies with FII behind them have also run further (typical "
        f"{up.get('median_move', 0):.1f}%) than rallies FII fought ({vs.get('median_move', 0):.1f}%). "
        "But FII's book builds **during** the move, not before it (section 3 shows the turn-day test). So "
        "you see whose swing it is while it runs. That helps you stay with a move and judge how mature it "
        "is; it does not call the turn."
    )


def _render_peaks(pb: dict) -> None:
    pk = pb.get("peaks")
    if not pk or not pk["tops"].get("n"):
        return
    bk = pb.get("book") or {"metric": "FII long %", "key": "fut",
                            "sig": {"short_bottom": "p ≈ 0.02 for 3% bottoms, 0.003 for 5%", "short_ok": True,
                                    "long_top": "p ≈ 0.06 — not statistically reliable", "long_ok": False}}
    m_ = bk["metric"]
    sig = bk["sig"]
    st.markdown(f"##### Is there a {m_} level where the market turns?")
    t, b = pk["tops"], pk["bottoms"]
    ct, cb = pk["chance_top"], pk["chance_bottom"]
    z = {r["label"][:13]: r for r in pk["zone_test"]}
    hl, hs = z.get("FII heavy LON"), z.get("FII heavy SHO")
    c1, c2 = st.columns(2)
    c1.markdown(
        "<div style='border:1px solid rgba(239,83,80,0.4);border-radius:10px;padding:10px 14px;"
        "background:rgba(239,83,80,0.05);font-size:12.5px;line-height:1.6'>"
        f"<b style='color:#ef5350'>AT {t['n']} TOPS</b><br>"
        f"{m_} was in its <b>heavy-long zone (top fifth of its year) at {t['share_hi']:.0f}%</b> of tops, "
        f"in the middle at {t['share_mid']:.0f}%, and even in its heavy-short zone at {t['share_lo']:.0f}%. "
        f"Typical rank at a top: {_ord(t['median_pct'])} percentile.<br>"
        + (f"Reverse check: after FII entered heavy-long (top 10%), a top followed in the next 1–10 sessions "
           f"<b>{_p(hl['top_near'],0,False)}</b> of the time — after a random day: {ct:.0f}%. "
           f"<span style='color:{MUTED}'>{sig['long_top']} (16 Nov 2026 test).</span>"
           if hl else "") + "</div>", unsafe_allow_html=True)
    c2.markdown(
        "<div style='border:1px solid rgba(38,166,154,0.4);border-radius:10px;padding:10px 14px;"
        "background:rgba(38,166,154,0.05);font-size:12.5px;line-height:1.6'>"
        f"<b style='color:#26a69a'>AT {b['n']} BOTTOMS</b><br>"
        f"{m_} was in its <b>heavy-short zone (bottom fifth of its year) at {b['share_lo']:.0f}%</b> of "
        f"bottoms, in the middle at {b['share_mid']:.0f}%, heavy-long at {b['share_hi']:.0f}%. "
        f"Typical rank at a bottom: {_ord(b['median_pct'])} percentile.<br>"
        + (f"Reverse check: after FII entered heavy-short (bottom 10%), a bottom followed in the next 1–10 "
           f"sessions <b>{_p(hs['bottom_near'],0,False)}</b> of the time — after a random day: {cb:.0f}%. "
           f"<span style='color:{MUTED}'>Survives a significance test: {sig['short_bottom']}.</span>"
           if hs else "") + "</div>", unsafe_allow_html=True)

    lt = pd.DataFrame(pk["level_test"])
    show_levels = bk.get("key", "fut") == "fut"     # the level grid (8%..30%) is a futures-share grid
    if show_levels:
      with st.expander("🔢 Fixed levels (e.g. 'above 20% → market falls') — tested since Jan 2025"):
        st.caption(
            "Fixed levels only make sense inside one regime: before NSE's Nov-2024 lot-size reset FII long % "
            "sat around 45%; since then around 11%. So this table uses Jan 2025 onward only. 'Top/bottom close "
            f"by' = a 3% swing turn in the next 1–10 sessions; any random day: top "
            f"{_p(pk['level_chance_top'],0,False)}, bottom {_p(pk['level_chance_bottom'],0,False)}.")
        st.dataframe(lt.rename(columns={"level": "FII long % level", "days": "Days", "episodes": "Episodes",
                                        "top_near": "Top close by %", "bottom_near": "Bottom close by %",
                                        "fwd10": "Nifty next 10d %"}),
                     hide_index=True, width="stretch", column_config={
                         "Top close by %": st.column_config.NumberColumn(format="%.0f"),
                         "Bottom close by %": st.column_config.NumberColumn(format="%.0f"),
                         "Nifty next 10d %": st.column_config.NumberColumn(format="%+.2f")})
    with st.expander(f"📍 {m_} at every swing top and bottom since {pb['history_from']:%Y}"):
        at = pd.DataFrame(pk["at_turns"]).iloc[::-1]
        st.dataframe(at.rename(columns={"date": "Date", "kind": "Turn", "nifty": "Nifty",
                                        "fii_long": m_, "pct": "Rank in its year (pct)",
                                        "next_move": "Next swing %"}),
                     hide_index=True, width="stretch", column_config={
                         "Nifty": st.column_config.NumberColumn(format="%.0f"),
                         m_: st.column_config.NumberColumn(format="%.1f"),
                         "Rank in its year (pct)": st.column_config.NumberColumn(format="%.0f"),
                         "Next swing %": st.column_config.NumberColumn(format="%+.1f")})


def _render_forward(pb: dict) -> None:
    st.markdown("#### B · Does today's 5-session scenario tell what comes next?")
    st.caption("Each session is labelled by FII's last-5-session futures action and Nifty's last-5-session "
               "direction, then compared with what Nifty did over the next 10 and 20 sessions (from the next "
               "open). Unlike Part A, this uses only what was known on the day.")
    t = pb["today"]
    base, bu = pb["base"], pb["base_up"]
    mins = pb["params"]["min_episodes"]
    cb, ct = pb["chance_bottom"], pb["chance_top"]
    if t["contaminated"]:
        st.warning("The last 5 sessions include a monthly futures expiry, so FII's 5-session flow partly "
                   "reflects contracts that expired. Today's scenario is shown but treat it as approximate.")

    order = [("rally_vs_fii", "rally_fii"), ("fall_fii", "fall_vs_fii")]
    by_key = {s["key"]: s for s in pb["scenarios"]}
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;font-size:11px;color:{MUTED};margin:0 4px 4px'>"
        "<span>◀ FII SELLING (short build / long unwind)</span><span>FII BUYING (long build / short cover) ▶</span></div>",
        unsafe_allow_html=True,
    )
    for row, lab in zip(order, ("NIFTY UP over 5 sessions", "NIFTY DOWN over 5 sessions")):
        st.markdown(f"<div style='font-size:11px;font-weight:700;color:{MUTED};margin:6px 4px 2px'>{lab}</div>",
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for col, key in zip(cols, row):
            s = by_key[key]
            c = SCENCOL[s["name"]]
            vw, vc = _verdict_words(s["t10"], s["episodes"], mins)
            today = s["today"]
            col.markdown(
                f"<div style='border:{'2px' if today else '1px'} solid {c if today else 'rgba(255,255,255,0.1)'};"
                f"border-radius:10px;padding:12px 14px;background:{c}{'1f' if today else '0a'};min-height:190px'>"
                f"<div style='display:flex;justify-content:space-between'><b style='color:{c};font-size:15px'>"
                f"{s['name']}</b>{_badge('◀ TODAY', c) if today else ''}</div>"
                f"<div style='font-size:12px;color:{MUTED};margin:2px 0 8px'>{s['meaning']}</div>"
                f"<div style='font-size:12.5px'>Next 10 sessions: <b>{_p(s['fwd10'])}</b> "
                f"(up {_p(s['up10'],0,False)}) · all days {_p(base[10])} (up {_p(bu[10],0,False)})</div>"
                f"<div style='font-size:12.5px'>Next 20 sessions: <b>{_p(s['fwd20'])}</b> · all days {_p(base[20])}</div>"
                f"<div style='font-size:12.5px'>Turn close by: top <b>{_p(s['top_near'],0,False)}</b> · bottom "
                f"<b>{_p(s['bottom_near'],0,False)}</b> <span style='color:{MUTED}'>(random day {ct:.0f}% / "
                f"{cb:.0f}%)</span></div>"
                f"<div style='font-size:11.5px;color:{MUTED}'>{s['episodes']} episodes · {s['share_days']:.0f}% of sessions</div>"
                f"<div style='margin-top:6px'>{_badge(vw, vc)} "
                f"<span style='font-size:11px;color:{MUTED}'>t = {_p(s['t10'],2,True,'')}</span></div></div>",
                unsafe_allow_html=True,
            )
    st.info(
        "**Answer: no.** None of the four scenarios changes Nifty's next 10–20 sessions by more than chance. "
        "A fall FII is pushing has been followed by a small *bounce* slightly more often than by more falling. "
        "What changes is how close a turn is: bottoms come more often during FII-backed falls, and tops during "
        "FII-backed rallies. Use that for position size and stop distance."
    )

    with st.expander("🔍 All 8 detailed cases — which of FII's four actions, with Nifty up or down"):
        df = pd.DataFrame(pb["detail"])
        _table(df, ["flow", "price"], {"flow": "FII 5-session action", "price": "Nifty 5 sessions"})
    with st.expander("🎯 Scenario × FII positioning zone — the combination that matters most"):
        st.caption("FII zone = where FII's long share sits within its own last 250 sessions. "
                   "Bottoms cluster where FII is already extreme short AND still pushing the fall.")
        df = pd.DataFrame(pb["by_zone"])
        _table(df, ["scenario", "zone"], {"scenario": "Scenario", "zone": "FII zone"})
    with st.expander("🤝 Who took the other side of FII — Client, Pro or DII"):
        st.caption("Over each 5-session window, the category that moved most against FII. "
                   "Client is the usual counterparty (that is the zero-sum file at work); Pro or DII taking "
                   "the other side is the less common case.")
        df = pd.DataFrame(pb["by_absorber"])
        _table(df, ["absorber", "fii"], {"absorber": "Other side taken by", "fii": "FII 5-session flow"})
    with st.expander("📐 How this was measured"):
        st.markdown(_method_md(pb))


def _method_md(pb: dict) -> str:
    pr = pb["params"]
    return f"""
- **Sessions:** {pb['sessions']:,} since {pb['history_from']:%d %b %Y}; {pb['usable']:,} used.
  {pb['excluded_expiry']:,} were left out because their 5-session window contains a monthly futures
  expiry (expiring contracts disappear from both sides and look like an 'unwind').
- **FII action** = the bigger of the 5-session change in FII index-futures longs and shorts:
  long build (+longs), long unwind (−longs), short build (+shorts), short cover (−shorts).
- **Forward return** starts at the *next* day's open, because NSE publishes the file after the close.
- **Episode** = first day of a scenario, then {pr['cooldown']} sessions ignored, so one long run counts once.
  Cases with fewer than {pr['min_episodes']} episodes are labelled *too few cases*.
- **Swing turn** = a closing high/low followed by a {pr['turn_pct']:.0f}% reversal; 'close by' means
  in the {pr['near_from']}–{pr['near_after']} sessions AFTER the signal (never counting a turn that
  had already happened).
- **t** compares the scenario with all days. With {pb['n_tests']} comparisons on this tab, one or two
  will reach |t| ≈ 2 by luck alone; about |t| > 3 is needed before a single cell means something.
"""


def _table(df: pd.DataFrame, labels: list[str], rename: dict) -> None:
    out = df[labels].rename(columns=rename).copy()
    out.insert(0, "Today", np.where(df["today"], "◀", ""))
    out["Episodes"] = df["episodes"]
    out["Next 5d %"] = df["fwd5"]
    out["Next 10d %"] = df["fwd10"]
    out["Up 10d %"] = df["up10"]
    out["t 10d"] = df["t10"]
    out["Next 20d %"] = df["fwd20"]
    out["Top close by %"] = df["top_near"]
    out["Bottom close by %"] = df["bottom_near"]
    out["Read"] = [(_verdict_words(t, n, 15)[0]) for t, n in zip(df["t10"], df["episodes"])]
    cfg = {c: st.column_config.NumberColumn(c, format="%.2f" if c.startswith(("t ", "Next")) else "%.0f")
           for c in out.columns if out[c].dtype.kind == "f"}
    st.dataframe(out, hide_index=True, width="stretch", column_config=cfg)


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _swing_chart(pb: dict) -> go.Figure:
    """Top: Nifty over coloured swing bands. Bottom: FII long % with its own
    heavy-long / heavy-short zones and the reading at every turn.

    Bands are drawn as filled polygons (traces), not layout shapes: Streamlit's
    plotly theme paints an opaque plot background that hides below-layer shapes.
    """
    h = pb["history"]
    legs = pb["legs"]
    x0, x1 = h["date"].iloc[0], h["date"].iloc[-1]
    ymax, ymin = float(np.nanmax(h["close"])), float(np.nanmin(h["close"]))
    pad = (ymax - ymin) * 0.10
    y_lo, y_hi = ymin - pad, ymax + pad * 1.25
    sh = h["fii_share"].astype(float)
    # fit the lower axis to the data: the options ratio lives in ~30-50%, so a 0 baseline
    # flattened it into a strip; the futures share still starts near 0 by itself
    s_top = float(np.nanmax([np.nanmax(sh), np.nanmax(h["p90"])]))
    s_bot = float(np.nanmin([np.nanmin(sh), np.nanmin(h["p10"])]))
    span = max(s_top - s_bot, 1.0)
    s_hi = s_top + 0.15 * span
    s_lo = max(0.0, s_bot - 0.15 * span)
    bk = pb.get("book") or {}
    is_opt = bk.get("key") == "opt"
    key = bk.get("key", "fut")
    acts = bk.get("actions") or {"LB": "Long buildup", "SC": "Short covering",
                                 "SB": "Short buildup", "LU": "Long unwinding"}
    dl5 = h.get(f"{key}_dl5", pd.Series(np.nan, index=h.index)).astype(float)
    ds5 = h.get(f"{key}_ds5", pd.Series(np.nan, index=h.index)).astype(float)
    bars = {  # what FII did over the last 5 sessions, % of each side's own book
        "LB": dl5.clip(lower=0), "SC": (-ds5).clip(lower=0),        # bullish: plotted UP
        "SB": (-ds5).clip(upper=0), "LU": dl5.clip(upper=0),        # bearish: plotted DOWN
    }
    up_sum = (bars["LB"] + bars["SC"]).values
    dn_sum = (bars["SB"] + bars["LU"]).values
    net = pd.Series(up_sum + dn_sum, index=h.index)            # bullish minus bearish action
    if dl5.notna().any():
        # scale to the typical bar, not the one expiry spike: 97th percentile each way;
        # anything beyond is cut at the edge and still readable on hover
        a_hi = max(float(np.nanpercentile(up_sum, 97)) * 1.25, 5.0)
        a_lo = min(float(np.nanpercentile(dn_sum, 3)) * 1.25, -5.0)
    else:
        a_lo, a_hi = -10.0, 10.0
    ACOL = {"LB": "#26a69a", "SC": "#42a5f5", "SB": "#ef5350", "LU": "#ff9800"}
    act_title = ("FII actions, last 5 sessions (% of each side's own book) — "
                 + " · ".join(f"<span style='color:{ACOL[k]}'>■ {acts[k]}</span>" for k in ("LB", "SC"))
                 + " (up)  /  "
                 + " · ".join(f"<span style='color:{ACOL[k]}'>■ {acts[k]}</span>" for k in ("SB", "LU"))
                 + " (down)")
    sides = bk.get("sides") or ("longs", "shorts")
    is_fut = not is_opt
    l_name = "FII LONGS" if is_fut else "FII BULLISH side (calls bought + puts written)"
    s_name = "FII SHORTS" if is_fut else "FII BEARISH side (calls written + puts bought)"
    pos_title = (f"FII positions, contracts{' in today’s lot size' if is_fut else ''} — "
                 f"<span style='color:#26a69a'>■ {l_name}</span>"
                 f"{' (left axis)' if is_fut else ''} · "
                 f"<span style='color:#ef5350'>■ {s_name}</span>{' (right axis)' if is_fut else ''}"
                 " · ▲ = position bottom, ▼ = position peak")
    act_title = act_title + " · <b>— white line = NET</b> (above 0 bullish, below 0 bearish)"
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.29, 0.27, 0.22, 0.22],
                        vertical_spacing=0.05,
                        specs=[[{}], [{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("Nifty 50 — each band is one swing, coloured by what FII did during it",
                                        pos_title, act_title,
                                        f"{(pb.get('book') or {}).get('metric', 'FII long %')}  =  "
                                        f"{(pb.get('book') or {}).get('definition', 'FII longs ÷ (longs + shorts) in index futures')}"
                                        f"   ({(pb.get('book') or {}).get('aside', 'short % = 100 − this')})"))
    for g in legs:
        s0 = max(pd.Timestamp(g["start"]), x0)
        e0 = pd.Timestamp(g["end"])
        c = SCENCOL[g["scenario"]]
        for r, (lo, hi), a in ((1, (y_lo, y_hi), 0.22), (3, (a_lo, a_hi), 0.07), (4, (s_lo, s_hi), 0.07)):
            fig.add_trace(go.Scatter(x=[s0, s0, e0, e0, s0], y=[lo, hi, hi, lo, lo], fill="toself",
                                     fillcolor=_rgba(c, a), line=dict(width=0), mode="lines",
                                     hoverinfo="skip", showlegend=False), row=r, col=1)

    # ---- top panel
    fig.add_trace(go.Scatter(x=h["date"], y=h["close"], mode="lines", name="Nifty 50", showlegend=False,
                             line=dict(color="rgba(255,255,255,0.95)", width=2),
                             hovertemplate="Nifty %{y:,.0f}<extra></extra>"), row=1, col=1)
    for g in legs:
        s0 = max(pd.Timestamp(g["start"]), x0)
        e0 = pd.Timestamp(g["end"])
        if (e0 - s0).days < 14:
            continue
        fii = g["fii_share_chg"]
        c = SCENCOL[g["scenario"]]
        fig.add_annotation(x=s0 + (e0 - s0) / 2, y=ymax + pad * 0.75, row=1, col=1, showarrow=False,
                           text=f"<b>{g['move_pct']:+.1f}%</b><br>FII {fii:+.0f} pts" if fii is not None
                           else f"<b>{g['move_pct']:+.1f}%</b>",
                           font=dict(size=11, color=c), bgcolor="rgba(14,17,23,0.75)", borderpad=2)
    closed = [g for g in legs if not g["open"] and pd.Timestamp(g["end"]) >= x0]
    # turns closer than 7 days to the previous one keep the marker but drop the text
    crowded, last = set(), None
    for g in closed:
        e0 = pd.Timestamp(g["end"])
        if last is not None and (e0 - last).days < 7:
            crowded.add(g["end"])
        last = e0
    for g in closed:
        e0 = pd.Timestamp(g["end"])
        top = g["dir"] > 0
        col = "#ef5350" if top else "#26a69a"
        fig.add_trace(go.Scatter(
            x=[e0], y=[g["nifty_to"]], mode="markers+text", showlegend=False,
            marker=dict(symbol="triangle-down" if top else "triangle-up", size=12, color=col,
                        line=dict(width=1, color="#111")),
            text=[f"{'TOP' if top else 'BOTTOM'} {g['nifty_to']:,.0f}<br>{e0:%d %b}"
                  if g["end"] not in crowded else ""],
            textposition="top center" if top else "bottom center",
            textfont=dict(size=9.5, color="rgba(255,255,255,0.8)"), hoverinfo="skip"), row=1, col=1)

    # ---- positions panel: what FII HOLDS, each side on its own axis
    Lr = h.get(f"{key}_L", pd.Series(np.nan, index=h.index)).astype(float) / 1000
    Sr = h.get(f"{key}_S", pd.Series(np.nan, index=h.index)).astype(float) / 1000
    Lz = h.get(f"{key}_Lz", pd.Series(np.nan, index=h.index)).astype(float) / 1000
    Sz = h.get(f"{key}_Sz", pd.Series(np.nan, index=h.index)).astype(float) / 1000
    def _cfmt(v):           # 1,240,000 -> 1.24M ; 62,546 -> 63k
        return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"

    for ser, smo, col, nm, sec in ((Lr, Lz, "#26a69a", l_name, False), (Sr, Sz, "#ef5350", s_name, is_fut)):
        if is_fut:          # options' daily line is a weekly-expiry sawtooth: leave it out
            fig.add_trace(go.Scatter(x=h["date"], y=ser, mode="lines", line=dict(color=col, width=1), opacity=0.25,
                                     hovertemplate=nm + " daily %{y:,.0f}k<extra></extra>", showlegend=False),
                          row=2, col=1, secondary_y=sec)
        fig.add_trace(go.Scatter(x=h["date"], y=smo, mode="lines", line=dict(color=col, width=2.6),
                                 fill=None, hovertemplate=nm + " (5-session median) %{y:,.0f}k<extra></extra>",
                                 showlegend=False), row=2, col=1, secondary_y=sec)
    pts = [t for t in bk.get("pos_turns", []) if t["value"] is not None and pd.Timestamp(t["date"]) >= x0]
    # where each point sits on SCREEN (0..1 of its own axis), so the collision test works on the
    # futures panel's two axes as well as the options panel's one
    if is_fut:
        rng_ = {"L": (np.nanmin(Lz), np.nanmax(Lz)), "S": (np.nanmin(Sz), np.nanmax(Sz))}
    else:
        both = (np.nanmin([np.nanmin(Lz), np.nanmin(Sz)]), np.nanmax([np.nanmax(Lz), np.nanmax(Sz)]))
        rng_ = {"L": both, "S": both}

    def _frac(t):
        lo_, hi_ = rng_[t["side"]]
        return (t["value"] / 1000 - lo_) / max(hi_ - lo_, 1e-9)
    for t in pts:
        td = pd.Timestamp(t["date"])
        col = "#26a69a" if t["side"] == "L" else "#ef5350"
        peak = t["kind"] == "PEAK"
        # if the other side turned within a week, labels would collide: keep the marker, drop the text
        clash = any(o is not t and o["side"] != t["side"] and abs((pd.Timestamp(o["date"]) - td).days) <= 7
                    and abs(_frac(o) - _frac(t)) < 0.12 for o in pts)
        fig.add_trace(go.Scatter(
            x=[td], y=[t["value"] / 1000], mode="markers+text", showlegend=False,
            marker=dict(symbol="triangle-down" if peak else "triangle-up", size=12, color=col,
                        line=dict(width=1, color="#111")),
            text=["" if clash and t["side"] == "S" else _cfmt(t["value"])],
            textposition="top center" if peak else "bottom center",
            textfont=dict(size=10.5, color=col), hovertemplate=(
                f"{'Longs' if t['side'] == 'L' and is_fut else 'Shorts' if is_fut else 'Bullish side' if t['side'] == 'L' else 'Bearish side'} "
                f"{'PEAK' if peak else 'BOTTOM'} {td:%d %b %Y}: {t['value']:,.0f}<extra></extra>")),
            row=2, col=1, secondary_y=(t["side"] == "S" and is_fut))
    for d_, f_ in ((pd.Timestamp("2025-12-30"), 75 / 65),):
        if x0 <= d_ <= x1:
            fig.add_vline(x=d_, line_color="rgba(255,255,255,0.45)", line_dash="dash", line_width=1, row=2, col=1)
            fig.add_annotation(x=d_, y=1, yref="y2 domain", xanchor="left", yanchor="top", showarrow=False,
                               text=" NSE lot-size cut (Nifty 75→65): earlier counts shown in today’s lots",
                               font=dict(size=9.5, color="rgba(255,255,255,0.6)"), row=2, col=1)

    # ---- middle panel: what FII DID (buildup / unwinding / covering)
    for k in ("LB", "SC", "SB", "LU"):
        fig.add_trace(go.Bar(x=h["date"], y=bars[k], name=acts[k], marker_color=ACOL[k], marker_line_width=0,
                             opacity=0.7, hovertemplate=acts[k] + " %{y:+.1f}%<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=net.rolling(3, min_periods=1).mean().clip(a_lo, a_hi),
                             mode="lines", name="NET",
                             line=dict(color="white", width=2), showlegend=False,
                             customdata=net, hovertemplate="NET action %{customdata:+.1f}% (line = 3-session average)<extra></extra>"),
                  row=3, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1, row=3, col=1)
    for g in closed:
        fig.add_vline(x=pd.Timestamp(g["end"]), line_color=("#ef5350" if g["dir"] > 0 else "#26a69a"),
                      line_width=1, line_dash="dot", opacity=0.6, row=3, col=1)

    # ---- bottom panel: FII's own heavy-long / heavy-short zones (trailing year)
    fig.add_trace(go.Scatter(x=h["date"], y=h["p90"], mode="lines", showlegend=False,
                             line=dict(color="rgba(38,166,154,0.9)", width=1, dash="dash"),
                             hovertemplate="heavy-long line %{y:.1f}%<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=[s_hi] * len(h), mode="lines", showlegend=False,
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(38,166,154,0.10)",
                             hoverinfo="skip"), row=4, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=[s_lo] * len(h), mode="lines", showlegend=False,
                             line=dict(width=0), hoverinfo="skip"), row=4, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["p10"], mode="lines", showlegend=False,
                             line=dict(color="rgba(239,83,80,0.9)", width=1, dash="dash"),
                             fill="tonexty", fillcolor="rgba(239,83,80,0.10)",
                             hovertemplate="heavy-short line %{y:.1f}%<extra></extra>"), row=4, col=1)
    if is_opt:
        # weekly option expiries put a sawtooth in the daily ratio; show it faint and draw
        # the 5-session median as the line to read
        fig.add_trace(go.Scatter(x=h["date"], y=sh, mode="lines", name="daily", showlegend=False,
                                 line=dict(color="rgba(100,181,246,0.35)", width=1),
                                 hovertemplate="daily %{y:.1f}%<extra></extra>"), row=4, col=1)
        fig.add_trace(go.Scatter(x=h["date"], y=sh.rolling(5, min_periods=1).median(), mode="lines",
                                 name="FII", showlegend=False, line=dict(color="#64b5f6", width=2.2),
                                 hovertemplate="5-session median %{y:.1f}%<extra></extra>"), row=4, col=1)
    else:
        fig.add_trace(go.Scatter(x=h["date"], y=sh, mode="lines", name="FII", showlegend=False,
                                 line=dict(color="#64b5f6", width=2),
                                 hovertemplate="FII %{y:.1f}%<extra></extra>"), row=4, col=1)
    for g in closed:
        e0 = pd.Timestamp(g["end"])
        top = g["dir"] > 0
        v = g["fii_share_to"]
        if v is None:
            continue
        fig.add_trace(go.Scatter(
            x=[e0], y=[v], mode="markers+text", showlegend=False,
            marker=dict(symbol="triangle-down" if top else "triangle-up", size=10,
                        color="#ef5350" if top else "#26a69a", line=dict(width=1, color="#111")),
            text=[f"{v:.1f}%" if g["end"] not in crowded else ""], textposition="top center" if top else "bottom center",
            textfont=dict(size=9.5, color="rgba(255,255,255,0.85)"), hoverinfo="skip"), row=4, col=1)
    lx = h["date"].iloc[int(len(h) * 0.02)]
    for y, txt, c in ((h["p90"].dropna().iloc[0] if h["p90"].notna().any() else None,
                       "FII HEAVY-LONG zone (top 10% of its year)", "#26a69a"),
                      (h["p10"].dropna().iloc[0] if h["p10"].notna().any() else None,
                       "FII HEAVY-SHORT zone (bottom 10% of its year)", "#ef5350")):
        if y is not None:
            fig.add_annotation(x=h.loc[h["p90"].notna(), "date"].iloc[0], y=y, row=4, col=1, xanchor="left", yanchor="bottom" if c == "#26a69a" else "top",
                               showarrow=False, text=txt, font=dict(size=10, color=c))

    fig.update_layout(height=1380, margin=dict(l=70, r=20, t=40, b=10), hovermode="x unified",
                      showlegend=False, barmode="relative", bargap=0.05,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    for ann in fig.layout.annotations[:4]:
        ann.font = dict(size=12.5, color="rgba(255,255,255,0.85)")
        ann.x, ann.xanchor = 0, "left"
    fig.update_yaxes(title_text="Nifty 50", gridcolor="rgba(255,255,255,0.06)", range=[y_lo, y_hi], row=1, col=1)
    fig.update_yaxes(title_text="FII action", gridcolor="rgba(255,255,255,0.06)", range=[a_lo, a_hi],
                     ticksuffix="%", zeroline=False, row=3, col=1)
    fig.update_yaxes(title_text=("Longs (k)" if is_fut else "Contracts (k)"),
                     title_font=dict(color="#26a69a" if is_fut else "rgba(255,255,255,0.8)"),
                     tickfont=dict(color="#26a69a" if is_fut else "rgba(255,255,255,0.8)"),
                     gridcolor="rgba(255,255,255,0.06)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text=("Shorts (k)" if is_fut else ""), title_font=dict(color="#ef5350"),
                     tickfont=dict(color="#ef5350"), showgrid=False, visible=is_fut, row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text=(pb.get("book") or {}).get("metric", "FII long %"), gridcolor="rgba(255,255,255,0.06)", range=[s_lo, s_hi],
                     ticksuffix="%", row=4, col=1)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", range=[x0, x1])
    return fig


# ══════════════════════════════════════════════════════════════ 4. DEEP DIVE
def render_deep_dive(selected_date: date) -> None:
    from src.dashboard.cache.queries import cached_player_history
    section_title("4", "Player Deep-Dive", "One participant's last 250 sessions.")
    try:
        who = st.segmented_control("Participant", PLAYER_ORDER, default="FII", key="bp_dd_player")
    except AttributeError:
        who = st.radio("Participant", PLAYER_ORDER, horizontal=True, key="bp_dd_player")
    who = who or "FII"
    h = cached_player_history(selected_date, who)
    if h.get("error"):
        st.info(h["error"])
        return
    df: pd.DataFrame = h["df"]
    c = PCOL[who]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["nifty"], name="Nifty 50", yaxis="y1",
                             line=dict(color="rgba(255,255,255,0.8)", width=1.4)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["long_share"], name=f"{who} long share %", yaxis="y2",
                             line=dict(color=c, width=1.6)))
    fig.update_layout(title=dict(text=f"{who}: share of index-futures book that is long, against Nifty",
                                 font=dict(size=13)),
                      height=330, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.1),
                      yaxis=dict(title="Nifty", gridcolor="rgba(255,255,255,0.06)"),
                      yaxis2=dict(title="long share %", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, width="stretch", key=f"bp_dd1_{who}")

    c1, c2 = st.columns(2)
    f2 = go.Figure()
    f2.add_trace(go.Bar(x=df["date"], y=df["net_change"], name="Daily net change",
                        marker_color=np.where(df["net_change"] >= 0, "#26a69a", "#ef5350")))
    f2.add_trace(go.Scatter(x=df["date"], y=df["net"], name="Net position (long − short)", yaxis="y2",
                            line=dict(color=c, width=1.5)))
    f2.update_layout(title=dict(text="Index futures — net position and daily change", font=dict(size=13)),
                     height=300, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     legend=dict(orientation="h", y=-0.2),
                     yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                     yaxis2=dict(overlaying="y", side="right", showgrid=False))
    c1.plotly_chart(f2, width="stretch", key=f"bp_dd2_{who}")
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=df["date"], y=df["call_net"], name="Calls: bought − sold",
                            line=dict(color="#26a69a", width=1.4)))
    f3.add_trace(go.Scatter(x=df["date"], y=df["put_net"], name="Puts: bought − sold",
                            line=dict(color="#ef5350", width=1.4)))
    f3.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    f3.update_layout(title=dict(text="Index options — net calls and net puts held", font=dict(size=13)),
                     height=300, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     legend=dict(orientation="h", y=-0.2), yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
    c2.plotly_chart(f3, width="stretch", key=f"bp_dd3_{who}")
    st.caption("Contract counts cover the last year only and are shown in today's lot size: NSE cut the "
               "Nifty lot from 75 to 65 on 30 Dec 2025, so earlier counts are scaled by 75/65. Older years "
               "had other lot sizes, which is why the long SHARE, not the counts, is used to compare across years.")

    st.markdown(f"**Last 20 sessions — what {who} did each day**")
    t = df.tail(20).iloc[::-1].copy()
    t["Date"] = pd.to_datetime(t["date"]).dt.strftime("%d %b %Y") + np.where(t["expiry"], " (expiry)", "")
    out = pd.DataFrame({
        "Date": t["Date"], "Nifty %": t["nifty_chg"], "Action": t["action"],
        "Longs Δ": t["d_long"], "Shorts Δ": t["d_short"], "Net Δ": t["net_change"],
        "Long share %": t["long_share"], "Percentile": t["pct"],
    })
    st.dataframe(out, hide_index=True, width="stretch", column_config={
        "Nifty %": st.column_config.NumberColumn(format="%+.2f"),
        "Longs Δ": st.column_config.NumberColumn(format="%+,.0f"),
        "Shorts Δ": st.column_config.NumberColumn(format="%+,.0f"),
        "Net Δ": st.column_config.NumberColumn(format="%+,.0f"),
        "Long share %": st.column_config.NumberColumn(format="%.1f"),
        "Percentile": st.column_config.NumberColumn(format="%.0f"),
    })


# ══════════════════════════════════════════════════════════════ HOW TO READ
HOW_TO_READ = """
### What this page measures
NSE publishes, every evening, how many index-futures and index-option contracts each of four groups
holds — **FII** (foreign institutions), **DII** (domestic institutions), **Pro** (broker prop desks)
and **Client** (retail and HNI). It is published *after* the close, so anything on this page can be
acted on at the next morning's open at the earliest.

### The four groups
| Group | What they are | How to read them |
|---|---|---|
| **FII** | Foreign funds, the largest directional book | Their position moves *with* Nifty: they add shorts while it falls and cover while it rises. |
| **Client** | Retail and HNI traders | Every contract has two sides, and the four groups sum to zero. Client is almost always the side opposite FII (daily correlation −0.67). Clients are most long near bottoms. |
| **Pro** | Prop desks | Usually hedging option books. Pro taking the other side of FII is less common. |
| **DII** | Mutual funds, insurers | Index futures mostly hedge their stock holdings. They are rarely active. |

### The four futures actions
| Action | Longs | Shorts | Plain meaning |
|---|---|---|---|
| **Long build** | ↑ | — | new bullish bets |
| **Short cover** | — | ↓ | bearish bets closed (buying) |
| **Short build** | — | ↑ | new bearish bets |
| **Long unwind** | ↓ | — | bullish bets closed (selling) |

### What 8½ years of data say (2018–2026, tested 2026-11-14)
- **FII positioning follows price; it does not lead it.** The 5-day change in FII's long share
  matches the *past* 5 days of Nifty (+0.48) and not the *next* 5 (+0.02).
- **No ratio peaks before a turn.** FII is most short *on* the day of a bottom, not before it. Long
  share, long/short ratio, put/call ratio, option posture and FII-minus-Client gap were all tested.
- **What does help:** after FII enters the bottom 10% of its own year, a 5% bottom followed within 10
  sessions about **twice as often** as after a random day (≈22% vs 10%; survives a significance test,
  p ≈ 0.003). It is a zone that says *pay attention*, not a buy signal — about 4 times in 5 no 5%
  bottom follows within 10 sessions.
- **Tops are much harder to see than bottoms.** FII heavy-long was followed by a top somewhat more
  often than chance, but not reliably (p ≈ 0.06–0.08).
- **Never compare contract counts across years.** NSE changed the Nifty lot size several times. FII's long
  share itself has drifted from ~45% (2018–24) to ~11% (2026). That is why every level on this page is a
  percentile of the last 250 sessions, not a fixed number — and the percentile restarts at NSE's
  Nov-2024 lot-size reset, so the new era is never ranked against the old one.
- **Expiry days** move the numbers 4× more than normal days, purely from contracts expiring. They are flagged.
- **Swings:** FII was on the side of about 3 in 4 rallies and 3 in 4 falls since 2018 (shuffled-time
  test: ~4–5 in 10 by chance), and rallies FII fought were smaller. That is the same-time link —
  positions build *with* the move. Fixed levels such as "above 20%" do not hold up.
- **Index options book** (switch on the swing map): FII options bullish % = (calls bought + puts
  written) ÷ all four index-option legs. It sides with about 8 in 10 swings (more than futures),
  also follows price rather than leading it, and its heavy-short zone gives the **strongest bottom
  warning of any FII ratio tested** (a 3% bottom in the next 10 sessions 37% vs 24%, p ≈ 0.001).
  It warns of nothing at tops. The put/call ratio of options FII *bought* carries no information.
- **What FII does around turns** (graphs A1/A2, table A3): FII keeps pressing the OLD direction into
  the turn — still adding longs before tops (futures 44% vs 31% normal), still adding shorts before
  bottoms — and flips in the week AFTER. In options the flip is sharpest: bearish buildup in about half
  of the weeks after a top (vs 30% normally) and almost never in the week after a bottom (4%). So the
  flip confirms a turn; it does not predict it.
- **Data repair (16 Nov 2026):** 298 days in the NSE participant table held another date's file. All were
  re-downloaded and replaced; every number on this page is measured on the repaired data.

### How to use it
1. Check section 1 for who did what today and who took the other side.
2. Check section 2 for which scenario you are in and how often a turn has been close in it.
3. Check section 3 for whether FII is at an extreme of its own year. Extreme short means a bottom zone.
4. Size positions and set stops from that. Take direction from price itself.
"""
