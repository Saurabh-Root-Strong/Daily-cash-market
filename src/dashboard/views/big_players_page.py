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


def _render_swings(pb: dict) -> None:
    st.markdown("#### A · Every swing of the last year — and whether FII was behind it")
    legs = pb.get("legs") or []
    if not legs:
        st.info("Not enough history to draw swings.")
        return
    c1, c2 = st.columns(2)
    c1.markdown(_chips(["FII-backed rally", "Rally against FII", "Rally, FII on sidelines"]),
                unsafe_allow_html=True)
    c2.markdown(_chips(["FII-backed fall", "FII buying the dip", "Fall, FII on sidelines"]),
                unsafe_allow_html=True)
    st.plotly_chart(_swing_chart(pb), width="stretch", key=f"bp_swing_{pb['data_date']}")
    st.caption(
        f"A swing ends when Nifty reverses {pb['leg_pct']:.0f}% from its high or low (▲ ▼ marks). Each band is one "
        "swing. The label shows Nifty's move and how much FII's long share changed across it. Bottom panel: FII's "
        "long share — watch it climb through green bands and sink through red ones. Swing ends are only known "
        "afterwards, so this is a map of what happened, not a signal."
    )

    cur = pb.get("current_leg")
    if cur:
        col = SCENCOL[cur["scenario"]]
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
            f"FII long share {_p(cur['fii_share_from'],1,False)} → <b>{_p(cur['fii_share_to'],1,False)}</b> "
            f"({_p(cur['fii_share_chg'],1,True,' pts')}), FII net {_n(cur['fii_net_chg'])} contracts; "
            f"Client long share {_p(cur['client_share_chg'],1,True,' pts')}."
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

    _render_leg_stats(pb)
    _render_peaks(pb)

    with st.expander("📅 Swing-by-swing table (last year)", expanded=False):
        rows = []
        for g in reversed(legs):
            rows.append({
                "From": g["start"], "To": g["end"], "Sessions": g["sessions"],
                "Swing type": g["scenario"] + (" (running)" if g["open"] else "")
                + (" · partial" if g.get("partial") else ""),
                "Tag firm?": "fragile (<3 pts)" if g.get("fragile") else "firm",
                "Nifty from": g["nifty_from"], "Nifty to": g["nifty_to"], "Nifty move %": g["move_pct"],
                "FII long share from %": g["fii_share_from"], "to %": g["fii_share_to"],
                "FII change (pts)": g["fii_share_chg"], "FII net contracts": g["fii_net_chg"],
                "Client change (pts)": g["client_share_chg"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Nifty from": st.column_config.NumberColumn(format="%.0f"),
            "Nifty to": st.column_config.NumberColumn(format="%.0f"),
            "Nifty move %": st.column_config.NumberColumn(format="%+.1f"),
            "FII long share from %": st.column_config.NumberColumn(format="%.1f"),
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
        "'fragile' in the table rest on a change of under 3 points and could flip with one day's data. "
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
    st.markdown("##### Is there a FII long-% level where the market turns?")
    t, b = pk["tops"], pk["bottoms"]
    ct, cb = pk["chance_top"], pk["chance_bottom"]
    z = {r["label"][:13]: r for r in pk["zone_test"]}
    hl, hs = z.get("FII heavy LON"), z.get("FII heavy SHO")
    c1, c2 = st.columns(2)
    c1.markdown(
        "<div style='border:1px solid rgba(239,83,80,0.4);border-radius:10px;padding:10px 14px;"
        "background:rgba(239,83,80,0.05);font-size:12.5px;line-height:1.6'>"
        f"<b style='color:#ef5350'>AT {t['n']} TOPS</b><br>"
        f"FII long % was in its <b>heavy-long zone (top fifth of its year) at {t['share_hi']:.0f}%</b> of tops, "
        f"in the middle at {t['share_mid']:.0f}%, and even in its heavy-short zone at {t['share_lo']:.0f}%. "
        f"Typical rank at a top: {_ord(t['median_pct'])} percentile.<br>"
        + (f"Reverse check: after FII entered heavy-long (top 10%), a top followed in the next 1–10 sessions "
           f"<b>{_p(hl['top_near'],0,False)}</b> of the time — after a random day: {ct:.0f}%. "
           f"<span style='color:{MUTED}'>Not statistically reliable (p ≈ 0.06, 16 Nov 2026 test).</span>"
           if hl else "") + "</div>", unsafe_allow_html=True)
    c2.markdown(
        "<div style='border:1px solid rgba(38,166,154,0.4);border-radius:10px;padding:10px 14px;"
        "background:rgba(38,166,154,0.05);font-size:12.5px;line-height:1.6'>"
        f"<b style='color:#26a69a'>AT {b['n']} BOTTOMS</b><br>"
        f"FII long % was in its <b>heavy-short zone (bottom fifth of its year) at {b['share_lo']:.0f}%</b> of "
        f"bottoms, in the middle at {b['share_mid']:.0f}%, heavy-long at {b['share_hi']:.0f}%. "
        f"Typical rank at a bottom: {_ord(b['median_pct'])} percentile.<br>"
        + (f"Reverse check: after FII entered heavy-short (bottom 10%), a bottom followed in the next 1–10 "
           f"sessions <b>{_p(hs['bottom_near'],0,False)}</b> of the time — after a random day: {cb:.0f}%. "
           f"<span style='color:{MUTED}'>The one zone result that survives a significance test "
           f"(p ≈ 0.02 for 3% bottoms, 0.003 for 5%).</span>"
           if hs else "") + "</div>", unsafe_allow_html=True)

    lt = pd.DataFrame(pk["level_test"])
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
    with st.expander("📍 FII long % at every swing top and bottom since " f"{pb['history_from']:%Y}"):
        at = pd.DataFrame(pk["at_turns"]).iloc[::-1]
        st.dataframe(at.rename(columns={"date": "Date", "kind": "Turn", "nifty": "Nifty",
                                        "fii_long": "FII long %", "pct": "Rank in its year (pct)",
                                        "next_move": "Next swing %"}),
                     hide_index=True, width="stretch", column_config={
                         "Nifty": st.column_config.NumberColumn(format="%.0f"),
                         "FII long %": st.column_config.NumberColumn(format="%.1f"),
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
    s_hi = float(np.nanmax([np.nanmax(sh), np.nanmax(h["p90"])])) * 1.12
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05,
                        subplot_titles=("Nifty 50 — each band is one swing, coloured by what FII did during it",
                                        "FII long %  =  FII longs ÷ (longs + shorts) in index futures   "
                                        "(short % = 100 − this)"))
    for g in legs:
        s0 = max(pd.Timestamp(g["start"]), x0)
        e0 = pd.Timestamp(g["end"])
        c = SCENCOL[g["scenario"]]
        for r, (lo, hi), a in ((1, (y_lo, y_hi), 0.22), (2, (0, s_hi), 0.07)):
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

    # ---- bottom panel: FII's own heavy-long / heavy-short zones (trailing year)
    fig.add_trace(go.Scatter(x=h["date"], y=h["p90"], mode="lines", showlegend=False,
                             line=dict(color="rgba(38,166,154,0.9)", width=1, dash="dash"),
                             hovertemplate="heavy-long line %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=[s_hi] * len(h), mode="lines", showlegend=False,
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(38,166,154,0.10)",
                             hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=[0] * len(h), mode="lines", showlegend=False,
                             line=dict(width=0), hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["p10"], mode="lines", showlegend=False,
                             line=dict(color="rgba(239,83,80,0.9)", width=1, dash="dash"),
                             fill="tonexty", fillcolor="rgba(239,83,80,0.10)",
                             hovertemplate="heavy-short line %{y:.1f}%<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=sh, mode="lines", name="FII long %", showlegend=False,
                             line=dict(color="#64b5f6", width=2),
                             hovertemplate="FII long %{y:.1f}%<extra></extra>"), row=2, col=1)
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
            textfont=dict(size=9.5, color="rgba(255,255,255,0.85)"), hoverinfo="skip"), row=2, col=1)
    lx = h["date"].iloc[int(len(h) * 0.02)]
    for y, txt, c in ((h["p90"].dropna().iloc[0] if h["p90"].notna().any() else None,
                       "FII HEAVY-LONG zone (top 10% of its year)", "#26a69a"),
                      (h["p10"].dropna().iloc[0] if h["p10"].notna().any() else None,
                       "FII HEAVY-SHORT zone (bottom 10% of its year)", "#ef5350")):
        if y is not None:
            fig.add_annotation(x=h.loc[h["p90"].notna(), "date"].iloc[0], y=y, row=2, col=1, xanchor="left", yanchor="bottom" if c == "#26a69a" else "top",
                               showarrow=False, text=txt, font=dict(size=10, color=c))

    fig.update_layout(height=680, margin=dict(l=70, r=20, t=40, b=10), hovermode="x unified",
                      showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=12.5, color="rgba(255,255,255,0.85)")
        ann.x, ann.xanchor = 0, "left"
    fig.update_yaxes(title_text="Nifty 50", gridcolor="rgba(255,255,255,0.06)", range=[y_lo, y_hi], row=1, col=1)
    fig.update_yaxes(title_text="FII long %", gridcolor="rgba(255,255,255,0.06)", range=[0, s_hi],
                     ticksuffix="%", row=2, col=1)
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
    st.caption("Contract counts are shown for the last year only: NSE changed the Nifty lot size several "
               "times, so counts from different years are not comparable. Long share is.")

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
- **Data repair (16 Nov 2026):** 298 days in the NSE participant table held another date's file. All were
  re-downloaded and replaced; every number on this page is measured on the repaired data.

### How to use it
1. Check section 1 for who did what today and who took the other side.
2. Check section 2 for which scenario you are in and how often a turn has been close in it.
3. Check section 3 for whether FII is at an extreme of its own year. Extreme short means a bottom zone.
4. Size positions and set stops from that. Take direction from price itself.
"""
