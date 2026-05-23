"""
Big Players F&O Tracker — institutional positioning via participant-wise OI & Volume.

Morgan Stanley-style analysis framework:
  • Tomorrow's Market Verdict  — composite of 9 institutional signals
  • Positioning Scorecard      — FII / DII / Pro / Retail with role interpretation
  • Full Derivative Matrix     — Index Futures | Index Options | Stock Futures | Stock Opts
  • Signal Intelligence Engine — 9 signals sorted by impact magnitude

Participant hierarchy:
  FII    = Market Driver    — foreign funds set direction; #1 signal
  DII    = Floor Support    — LIC/MFs buy dips; provides downside cushion
  Pro    = Delta Hedger     — prop desks often short futures to hedge options books
  Client = Contrarian Ind.  — retail extreme positioning is a fade signal
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_fao_cumulative,
    cached_fao_daily,
    cached_fao_latest,
    cached_fii_stats_history,
    cached_fii_stats_latest,
    cached_market_intelligence,
    cached_signal_backtest,
)
from src.dashboard.constants import GRID_COLOR, PAPER_BG, PLOT_BG, POSITIVE_COLOR, NEGATIVE_COLOR

# ── Participant colour palette ────────────────────────────────────────────────
_COLORS = {
    "FII":    "#2196f3",   # blue   — dominant institutional player
    "DII":    "#4caf50",   # green  — domestic institutions (LIC, MFs)
    "Client": "#ff9800",   # orange — retail
    "Pro":    "#9c27b0",   # purple — proprietary desks
}
_ORDER = ["FII", "DII", "Client", "Pro"]

# ── Role descriptions for positioning scorecard ───────────────────────────────
_ROLES: dict[str, tuple[str, str]] = {
    "FII":    ("MARKET DRIVER",       "Foreign funds set index direction — the #1 signal to follow"),
    "DII":    ("FLOOR SUPPORT",       "LIC, MFs, Insurance — buy on dips; SIP flows create natural floor"),
    "Pro":    ("DELTA HEDGER",        "Prop desks often neutral — short futures = hedging long options books"),
    "Client": ("CONTRARIAN SIGNAL",   "Retail — extreme longs = sell signal; extreme shorts = buy signal"),
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _net_color(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "#888888"
    return POSITIVE_COLOR if v > 0 else (NEGATIVE_COLOR if v < 0 else "#888888")


def _fmt_contracts(v) -> str:
    if v is None or (hasattr(v, "__float__") and pd.isna(float(v))):
        return "—"
    v = int(v)
    if abs(v) >= 100_000:
        return f"{v/100_000:+.2f}L"
    return f"{v:+,}"


def _score_bar(score: float, max_score: float = 14.0) -> str:
    pct = max(0, min(100, int((score / max_score + 1) * 50)))
    if score >= 5:   bar_c = "#00c853"
    elif score >= 2: bar_c = "#69f0ae"
    elif score >= -1: bar_c = "#ffca28"
    elif score >= -4: bar_c = "#ffab40"
    else:            bar_c = "#ff5252"
    return (
        f"<div style='background:#1e1e1e;border-radius:4px;height:10px;"
        f"overflow:hidden;margin:6px 0'>"
        f"<div style='width:{pct}%;height:100%;background:{bar_c};border-radius:4px'></div>"
        f"</div>"
    )


def _signal_badge(sig) -> str:
    if sig.direction > 0:    bg, tc = "#00c85318", "#00c853"
    elif sig.direction < 0:  bg, tc = "#ff525218", "#ff5252"
    else:                    bg, tc = "#88888818", "#888888"
    score_str = f"+{sig.score}" if sig.score > 0 else str(sig.score)
    return (
        f"<div style='border:1px solid {tc}44;border-radius:8px;padding:8px 12px;"
        f"background:{bg};margin:3px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-size:11px;color:{tc};font-weight:700'>"
        f"{sig.emoji} {sig.headline}</span>"
        f"<span style='font-size:11px;color:{tc};font-weight:800;"
        f"background:{tc}22;padding:1px 8px;border-radius:10px'>{score_str}</span>"
        f"</div>"
        f"<div style='font-size:10px;color:rgba(255,255,255,0.5);margin-top:3px;line-height:1.45'>"
        f"{sig.description[:180]}{'…' if len(sig.description) > 180 else ''}</div>"
        f"</div>"
    )


# ── SECTION 1: Tomorrow's Verdict Hero Panel ─────────────────────────────────

def _tomorrow_verdict_hero(mi) -> None:
    """
    Full-width hero panel: Tomorrow's direction verdict on the left,
    composite score + weekly expiry on the right.
    """
    if not mi.tomorrow_verdict:
        return

    v = mi.tomorrow_verdict
    dir_icon = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "↔"}.get(v.direction, "?")
    conf_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}.get(v.confidence, "")

    c_verdict, c_score = st.columns([3, 2])

    with c_verdict:
        st.markdown(
            f"<div style='border:2px solid {v.direction_color}88;border-radius:14px;"
            f"padding:20px 24px;background:{v.direction_color}0e'>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.3);"
            f"letter-spacing:2.5px;margin-bottom:10px'>TOMORROW'S EXPECTED MARKET MOVE</div>"
            f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:12px'>"
            f"<div style='font-size:56px;font-weight:900;color:{v.direction_color};"
            f"line-height:1;letter-spacing:-3px'>{dir_icon}</div>"
            f"<div>"
            f"<div style='font-size:36px;font-weight:900;color:{v.direction_color};"
            f"line-height:1;letter-spacing:-1px'>{v.direction}</div>"
            f"<div style='font-size:11px;font-weight:700;color:{v.direction_color};"
            f"opacity:0.85;margin-top:2px'>{conf_icon} {v.confidence} CONFIDENCE</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:13px;font-weight:600;color:rgba(255,255,255,0.9);"
            f"margin-bottom:10px;line-height:1.4'>{v.headline}</div>"
            f"<div style='display:flex;flex-direction:column;gap:4px'>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.5)'>"
            f"<b style='color:rgba(255,255,255,0.75)'>Key Driver:</b> {v.key_driver}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.4)'>"
            f"<b style='color:rgba(255,255,255,0.55)'>Key Risk:</b> {v.key_risk}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_score:
        view_c = mi.view_color

        expiry_html = ""
        if mi.weekly_expiry:
            we = mi.weekly_expiry
            exp_c = "#ffca28" if we.days_to_expiry > 2 else "#ff9800"
            expiry_html = (
                f"<div style='margin-top:12px;border-top:1px solid rgba(255,255,255,0.08);"
                f"padding-top:10px'>"
                f"<div style='font-size:9px;font-weight:700;color:{exp_c};"
                f"letter-spacing:1px;margin-bottom:3px'>"
                f"📅 WEEKLY EXPIRY — {we.expiry_date.strftime('%d %b')} "
                f"({we.days_to_expiry}D away)</div>"
                f"<div style='font-size:12px;font-weight:700;color:{exp_c};margin-bottom:4px'>"
                f"{we.bias}</div>"
                f"<div style='font-size:10px;color:rgba(255,255,255,0.45);line-height:1.4'>"
                f"{we.reasoning}</div>"
                f"</div>"
            )

        # Institutional bias note when squeeze risk changes the narrative
        bias_note = ""
        if v.short_covering_active:
            bias_note = (
                f"<div style='margin-top:10px;padding:8px 10px;border-radius:6px;"
                f"background:#69f0ae18;border:1px solid #69f0ae44'>"
                f"<div style='font-size:9px;font-weight:700;color:#69f0ae;margin-bottom:2px'>"
                f"⚡ SHORT COVERING IN PROGRESS</div>"
                f"<div style='font-size:9px;color:rgba(255,255,255,0.5);line-height:1.4'>"
                f"FII INSTITUTIONAL BIAS = BEARISH (net short).<br>"
                f"TRADING SIGNAL = UP — FII is actively buying to close shorts.<br>"
                f"These two can diverge. Trust the OI change direction, not the level."
                f"</div></div>"
            )
        elif v.squeeze_risk:
            bias_note = (
                f"<div style='margin-top:10px;padding:8px 10px;border-radius:6px;"
                f"background:#FFD60018;border:1px solid #FFD60044'>"
                f"<div style='font-size:9px;font-weight:700;color:#FFD600;margin-bottom:2px'>"
                f"⚡ SQUEEZE RISK — INSTITUTIONAL BIAS ≠ TRADING DIRECTION</div>"
                f"<div style='font-size:9px;color:rgba(255,255,255,0.5);line-height:1.4'>"
                f"FII holds a massive short position (BEARISH BIAS).<br>"
                f"But: massive short = squeeze fuel if positive catalyst triggers covering.<br>"
                f"Market can rally sharply even from a bearish-positioned FII."
                f"</div></div>"
            )

        st.markdown(
            f"<div style='border:1px solid {view_c}55;border-radius:14px;"
            f"padding:20px 24px;background:{view_c}0c'>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.3);"
            f"letter-spacing:2px;margin-bottom:8px'>QUANT SIGNAL COMPOSITE SCORE</div>"
            f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:4px'>"
            f"<div style='font-size:50px;font-weight:900;color:{view_c};line-height:1'>"
            f"{mi.composite_score:+.0f}</div>"
            f"<div>"
            f"<div style='font-size:15px;font-weight:800;color:{view_c}'>{mi.market_view}</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.3);margin-top:2px'>"
            f"range: −18 (full bear) to +18 (full bull)</div>"
            f"</div>"
            f"</div>"
            f"{_score_bar(mi.composite_score)}"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.6);line-height:1.5;margin-top:8px'>"
            f"{mi.bias_reasoning}</div>"
            f"{bias_note}"
            f"{expiry_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── SECTION 2: Institutional Positioning Scorecard ───────────────────────────

def _positioning_scorecard(latest: pd.DataFrame) -> None:
    """
    Visual scorecard: horizontal net-position bars for all 4 participants.
    Shows role interpretation + contrarian annotations.
    """
    nets: dict[str, int] = {}
    row_data: dict[str, pd.Series] = {}
    for ptype in _ORDER:
        row = latest[latest["client_type"] == ptype]
        if row.empty:
            continue
        r = row.iloc[0]
        nets[ptype] = int(r.get("fut_idx_net", 0) or 0)
        row_data[ptype] = r

    if not nets:
        return

    max_abs = max(abs(v) for v in nets.values()) or 1

    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 12px'>"
        "📊 INDEX FUTURES — INSTITUTIONAL POSITIONING SCORECARD</div>",
        unsafe_allow_html=True,
    )

    for ptype in _ORDER:
        if ptype not in nets:
            continue
        net  = nets[ptype]
        r    = row_data[ptype]
        color = _COLORS[ptype]
        role_title, role_desc = _ROLES[ptype]

        long_  = int(r.get("fut_idx_long",  0) or 0)
        short_ = int(r.get("fut_idx_short", 0) or 0)
        ls_pct = float(r.get("fut_idx_ls_pct") or 50.0)

        bar_pct   = int(abs(net) / max_abs * 78)
        bar_color = "#00c853" if net > 0 else ("#ff5252" if net < 0 else "#888")
        net_c     = bar_color

        # Special annotations
        extra_badge = ""
        if ptype == "Client":
            if net < -15_000:
                extra_badge = (
                    "<span style='background:#00c85322;color:#00c853;padding:2px 8px;"
                    "border-radius:10px;font-size:9px;font-weight:700;margin-left:8px'>"
                    "→ CONTRARIAN BULLISH</span>"
                )
            elif net > 15_000:
                extra_badge = (
                    "<span style='background:#ff525222;color:#ff5252;padding:2px 8px;"
                    "border-radius:10px;font-size:9px;font-weight:700;margin-left:8px'>"
                    "→ CONTRARIAN BEARISH</span>"
                )
        elif ptype == "FII":
            if abs(net) > 100_000:
                word = "LONG" if net > 0 else "SHORT"
                extra_badge = (
                    f"<span style='background:{color}22;color:{color};padding:2px 8px;"
                    f"border-radius:10px;font-size:9px;font-weight:700;margin-left:8px'>"
                    f"⚡ MASSIVE {word}</span>"
                )
            elif abs(net) > 50_000:
                word = "LONG" if net > 0 else "SHORT"
                extra_badge = (
                    f"<span style='background:{color}18;color:{color};padding:2px 8px;"
                    f"border-radius:10px;font-size:9px;font-weight:600;margin-left:8px'>"
                    f"↑ SIGNIFICANT {word}</span>"
                )

        ca, cb = st.columns([4, 1])
        with ca:
            st.markdown(
                f"<div style='margin-bottom:8px;padding:8px 12px;border-radius:8px;"
                f"background:{color}08;border-left:3px solid {color}66'>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
                f"<span style='font-size:13px;font-weight:800;color:{color};min-width:58px'>"
                f"{ptype}</span>"
                f"<span style='font-size:9px;font-weight:700;color:{color};opacity:0.75;"
                f"letter-spacing:0.5px'>{role_title}</span>"
                f"<span style='font-size:10px;color:rgba(255,255,255,0.3);flex:1'>"
                f"{role_desc}</span>"
                f"{extra_badge}"
                f"</div>"
                f"<div style='background:#1a1a1a;border-radius:3px;height:10px;"
                f"overflow:hidden'>"
                f"<div style='width:{bar_pct}%;height:100%;background:{bar_color};"
                f"border-radius:3px;opacity:0.85'></div>"
                f"</div>"
                f"<div style='font-size:9px;color:rgba(255,255,255,0.28);margin-top:3px'>"
                f"Long: {long_:,}  ·  Short: {short_:,}  ·  L/S ratio: {ls_pct:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown(
                f"<div style='text-align:right;padding-top:6px'>"
                f"<div style='font-size:22px;font-weight:900;color:{net_c}'>"
                f"{_fmt_contracts(net)}</div>"
                f"<div style='font-size:9px;color:rgba(255,255,255,0.28)'>net contracts</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── SECTION 3: Full Derivative Breakdown Matrix ───────────────────────────────

def _derivative_breakdown(latest: pd.DataFrame) -> None:
    """
    4-column matrix: Index Futures | Index Options | Stock Futures | Stock Options.
    Each column shows net positions for all 4 participants + interpretation.
    """
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 12px'>"
        "🎯 FULL DERIVATIVE POSITION MATRIX — All Instrument Categories</div>",
        unsafe_allow_html=True,
    )

    c_if, c_io, c_sf, c_so = st.columns(4)

    def _net_rows(col_key: str) -> str:
        """Build mini rows for a single net column."""
        html = ""
        for ptype in _ORDER:
            row = latest[latest["client_type"] == ptype]
            if row.empty:
                continue
            r     = row.iloc[0]
            net   = int(r.get(col_key, 0) or 0)
            color = _COLORS[ptype]
            net_c = "#00c853" if net > 0 else ("#ff5252" if net < 0 else "#888")
            html += (
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;padding:4px 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<span style='font-size:10px;color:{color};font-weight:700'>{ptype}</span>"
                f"<span style='font-size:11px;font-weight:800;color:{net_c}'>"
                f"{_fmt_contracts(net)}</span>"
                f"</div>"
            )
        return html

    def _card(col, title: str, body: str, footer: str) -> None:
        col.markdown(
            f"<div style='border:1px solid rgba(255,255,255,0.08);border-radius:12px;"
            f"padding:14px 16px;background:rgba(255,255,255,0.025);height:100%'>"
            f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.55);"
            f"letter-spacing:0.5px;margin-bottom:10px'>{title}</div>"
            f"{body}"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.22);margin-top:8px;"
            f"border-top:1px solid rgba(255,255,255,0.06);padding-top:6px;line-height:1.4'>"
            f"{footer}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Index Futures ─────────────────────────────────────────────────────────
    _card(c_if, "📊 INDEX FUTURES",
          _net_rows("fut_idx_net"),
          "Nifty / BankNifty / FinNifty<br>Pure directional market bets.<br>"
          "+ve = net long (bullish), −ve = net short (bearish)")

    # ── Index Options (with PCR) ──────────────────────────────────────────────
    opts_html = ""
    for ptype in _ORDER:
        row = latest[latest["client_type"] == ptype]
        if row.empty:
            continue
        r        = row.iloc[0]
        call_net = int(r.get("opt_idx_call_net", 0) or 0)
        put_net  = int(r.get("opt_idx_put_net",  0) or 0)
        delta    = int(r.get("opt_idx_net",      0) or 0)
        color    = _COLORS[ptype]
        call_c   = "#00c853" if call_net > 0 else "#ff5252"
        put_c    = "#ff5252" if put_net  > 0 else "#00c853"
        delta_c  = "#00c853" if delta > 0 else ("#ff5252" if delta < 0 else "#888")
        opts_html += (
            f"<div style='padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-size:10px;color:{color};font-weight:700'>{ptype}</span>"
            f"<span style='font-size:11px;font-weight:800;color:{delta_c}'>Δ {_fmt_contracts(delta)}</span>"
            f"</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.38)'>"
            f"C:<span style='color:{call_c};font-weight:600'>{_fmt_contracts(call_net)}</span>"
            f"&nbsp;&nbsp;P:<span style='color:{put_c};font-weight:600'>{_fmt_contracts(put_net)}</span>"
            f"</div>"
            f"</div>"
        )

    # PCR
    pcr_html = ""
    if "opt_idx_put_long" in latest.columns and "opt_idx_call_long" in latest.columns:
        tot_put  = float(latest["opt_idx_put_long"].sum())
        tot_call = float(latest["opt_idx_call_long"].sum())
        if tot_call > 0:
            pcr = tot_put / tot_call
            pcr_c = "#00c853" if pcr > 1.25 else ("#ff5252" if pcr < 0.72 else "#ffca28")
            pcr_lbl = "CONTRARIAN BUY" if pcr > 1.25 else ("CONTRARIAN SELL" if pcr < 0.72 else "NEUTRAL")
            pcr_html = (
                f"<div style='margin-top:8px;padding:6px 8px;border-radius:6px;"
                f"background:{pcr_c}18;border:1px solid {pcr_c}44;text-align:center'>"
                f"<div style='font-size:9px;color:rgba(255,255,255,0.4)'>PUT-CALL RATIO</div>"
                f"<div style='font-size:22px;font-weight:900;color:{pcr_c}'>{pcr:.2f}</div>"
                f"<div style='font-size:8px;font-weight:700;color:{pcr_c}'>{pcr_lbl}</div>"
                f"</div>"
            )

    c_io.markdown(
        f"<div style='border:1px solid rgba(255,255,255,0.08);border-radius:12px;"
        f"padding:14px 16px;background:rgba(255,255,255,0.025);height:100%'>"
        f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.55);"
        f"letter-spacing:0.5px;margin-bottom:10px'>📞 INDEX OPTIONS</div>"
        f"{opts_html}"
        f"{pcr_html}"
        f"<div style='font-size:9px;color:rgba(255,255,255,0.22);margin-top:8px;"
        f"border-top:1px solid rgba(255,255,255,0.06);padding-top:6px;line-height:1.4'>"
        f"Δ = Call Net − Put Net<br>C/P = Call / Put net OI</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Stock Futures ─────────────────────────────────────────────────────────
    _card(c_sf, "🏭 STOCK FUTURES",
          _net_rows("fut_stk_net"),
          "Single-stock F&amp;O<br>Sector rotation + conviction plays.<br>"
          "FII +ve = accumulating specific names")

    # ── Stock Options ─────────────────────────────────────────────────────────
    stk_opt_html = ""
    for ptype in _ORDER:
        row = latest[latest["client_type"] == ptype]
        if row.empty:
            continue
        r        = row.iloc[0]
        call_net = int(r.get("opt_stk_call_net", 0) or 0)
        put_net  = int(r.get("opt_stk_put_net",  0) or 0)
        color    = _COLORS[ptype]
        call_c   = "#00c853" if call_net > 0 else "#ff5252"
        put_c    = "#ff5252" if put_net  > 0 else "#00c853"
        stk_opt_html += (
            f"<div style='padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.05)'>"
            f"<div style='font-size:10px;color:{color};font-weight:700;margin-bottom:1px'>{ptype}</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.38)'>"
            f"C:<span style='color:{call_c};font-weight:600'>{_fmt_contracts(call_net)}</span>"
            f"&nbsp;&nbsp;P:<span style='color:{put_c};font-weight:600'>{_fmt_contracts(put_net)}</span>"
            f"</div>"
            f"</div>"
        )

    _card(c_so, "🎯 STOCK OPTIONS",
          stk_opt_html,
          "Stock-level options positioning.<br>"
          "C/P = Call / Put net OI<br>+ve C = bullish directional bets")


# ── PCR expanded view ─────────────────────────────────────────────────────────

def _pcr_metric(latest: pd.DataFrame) -> None:
    """PCR display with interpretation guide."""
    if latest.empty or "opt_idx_put_long" not in latest.columns:
        return
    total_put  = float(latest["opt_idx_put_long"].sum())
    total_call = float(latest["opt_idx_call_long"].sum())
    if total_call == 0:
        return
    pcr = total_put / total_call

    if pcr > 1.3:
        label = "Contrarian BULLISH"
        color = "#00c853"
        note  = "Excessive put-buying → fear/hedging already in place; expect upward reversal"
    elif pcr < 0.7:
        label = "Contrarian BEARISH"
        color = "#ff5252"
        note  = "Excessive call-buying → complacency/greed; risk of sudden fall"
    else:
        label = "NEUTRAL"
        color = "#ffca28"
        note  = "Balanced put/call positioning — no extreme contrarian signal"

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f"<div style='border:1px solid {color}55;border-radius:8px;padding:12px 16px;"
            f"background:{color}11;text-align:center'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.5);font-weight:600;"
            f"letter-spacing:.5px'>PUT-CALL RATIO (PCR)</div>"
            f"<div style='font-size:34px;font-weight:800;color:{color};line-height:1.1'>"
            f"{pcr:.2f}</div>"
            f"<div style='font-size:11px;color:{color};font-weight:600'>{label}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-top:3px'>{note}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div style='font-size:12px;color:rgba(255,255,255,0.55);padding:8px 4px'>"
            "<b style='color:rgba(255,255,255,0.8)'>PCR = Total Put OI ÷ Total Call OI</b><br><br>"
            "<b style='color:#00c853'>&gt; 1.3 — Contrarian Bullish:</b> "
            "Everyone already hedged downside. Markets rarely fall when participants are already protected. "
            "Historically marks short-term bottoms.<br><br>"
            "<b style='color:#ffca28'>0.7 – 1.3 — Neutral:</b> "
            "Normal put/call balance. No extreme contrarian signal.<br><br>"
            "<b style='color:#ff5252'>&lt; 0.7 — Contrarian Bearish:</b> "
            "Everyone buying calls — complacency at peaks. "
            "Smart money shorts into this euphoria."
            "</div>",
            unsafe_allow_html=True,
        )


# ── Existing chart functions (used in tabs) ───────────────────────────────────

def _cumulative_chart(cum: pd.DataFrame) -> go.Figure:
    """Cumulative Index Futures net per participant over time."""
    if cum.empty:
        return go.Figure()
    fig = go.Figure()
    for ptype in _ORDER:
        grp = cum[cum["client_type"] == ptype].sort_values("trade_date")
        if grp.empty:
            continue
        color = _COLORS[ptype]
        fig.add_trace(go.Scatter(
            x=grp["trade_date"], y=grp["cum_fut_idx_net"],
            name=ptype, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{ptype}</b><br>%{{x|%d %b %Y}}<br>"
                "Cumulative Net: <b>%{y:+,}</b> contracts<extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_dash="dash", line_width=1.5,
                  line_color="rgba(255,255,255,0.30)",
                  annotation_text="Flat (zero net)", annotation_position="top right",
                  annotation_font=dict(size=10, color="rgba(255,255,255,0.4)"))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Cumulative Net Contracts", showgrid=True,
                   gridcolor=GRID_COLOR, tickformat=","),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=12)),
        height=380, margin=dict(t=30, b=50, l=80, r=40),
        hovermode="x unified",
    )
    return fig


def _cumulative_options_chart(cum: pd.DataFrame) -> go.Figure:
    """FII + DII cumulative options: Call Net (solid) vs Put Net (dashed)."""
    if cum.empty or "cum_opt_idx_call_net" not in cum.columns:
        return go.Figure()
    fig = go.Figure()
    for ptype in ("FII", "DII"):
        grp = cum[cum["client_type"] == ptype].sort_values("trade_date")
        if grp.empty:
            continue
        color = _COLORS[ptype]
        fig.add_trace(go.Scatter(
            x=grp["trade_date"], y=grp["cum_opt_idx_call_net"],
            name=f"{ptype} Call Net", mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(f"<b>{ptype} Call Net</b><br>%{{x|%d %b}}: <b>%{{y:+,}}</b><extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=grp["trade_date"], y=grp["cum_opt_idx_put_net"],
            name=f"{ptype} Put Net", mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            hovertemplate=(f"<b>{ptype} Put Net</b><br>%{{x|%d %b}}: <b>%{{y:+,}}</b><extra></extra>"),
        ))
    fig.add_hline(y=0, line_dash="dash", line_width=1.5,
                  line_color="rgba(255,255,255,0.30)")
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Cumulative Net Contracts", showgrid=True,
                   gridcolor=GRID_COLOR, tickformat=","),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        height=320, margin=dict(t=40, b=40, l=80, r=40),
        hovermode="x unified",
    )
    return fig


def _pcr_chart(cum: pd.DataFrame) -> go.Figure:
    """Daily Put-Call Ratio trend."""
    if cum.empty or "opt_idx_put_long" not in cum.columns:
        return go.Figure()
    daily_pcr = (
        cum.groupby("trade_date")[["opt_idx_put_long", "opt_idx_call_long"]]
        .sum().reset_index().sort_values("trade_date")
    )
    daily_pcr["pcr"] = (
        daily_pcr["opt_idx_put_long"]
        / daily_pcr["opt_idx_call_long"].replace(0, float("nan"))
    )
    daily_pcr = daily_pcr.sort_values("trade_date")
    line_colors = [
        "#00c853" if v > 1.3 else ("#ff5252" if v < 0.7 else "#ffca28")
        for v in daily_pcr["pcr"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_pcr["trade_date"], y=daily_pcr["pcr"],
        mode="lines+markers",
        marker=dict(color=line_colors, size=4),
        line=dict(color="#90caf9", width=2),
        hovertemplate="%{x|%d %b %Y}<br>PCR: <b>%{y:.3f}</b><extra></extra>",
        name="PCR",
    ))
    fig.add_hline(y=1.3, line_dash="dash", line_color="#00c853", line_width=1.2,
                  annotation_text="1.3 — contrarian bullish",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="#00c853"))
    fig.add_hline(y=0.7, line_dash="dash", line_color="#ff5252", line_width=1.2,
                  annotation_text="0.7 — contrarian bearish",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#ff5252"))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.20)", line_width=1)
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Put-Call Ratio", showgrid=True, gridcolor=GRID_COLOR),
        height=260, margin=dict(t=30, b=40, l=60, r=140),
        hovermode="x unified", showlegend=False,
    )
    return fig


def _daily_net_chart(cum: pd.DataFrame, participant: str) -> go.Figure:
    """Bar chart: daily Index Futures net for one participant (last 60 days)."""
    grp = cum[cum["client_type"] == participant].sort_values("trade_date").tail(60)
    if grp.empty:
        return go.Figure()
    colors = ["#00c853" if v >= 0 else "#ff5252" for v in grp["daily_fut_idx_net"]]
    color  = _COLORS.get(participant, "#4c78a8")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["trade_date"], y=grp["daily_fut_idx_net"],
        marker_color=colors, opacity=0.85,
        hovertemplate=(
            f"<b>{participant}</b><br>%{{x|%d %b %Y}}<br>"
            "Daily Net: <b>%{y:+,}</b> contracts<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=grp["trade_date"], y=grp["cum_fut_idx_net"],
        name="Cumulative Net", mode="lines",
        line=dict(color=color, width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Cumulative: <b>%{y:+,}</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b", tickfont=dict(size=10)),
        yaxis=dict(title="Daily Net", showgrid=True, gridcolor=GRID_COLOR, tickformat=","),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right",
                    showgrid=False, tickformat=","),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        height=280, margin=dict(t=30, b=40, l=70, r=70),
        hovermode="x unified", barmode="relative",
    )
    return fig


def _ls_trend_chart(cum: pd.DataFrame) -> go.Figure:
    """FII + DII Long-to-Short % trend over time."""
    fig = go.Figure()
    for ptype in ("FII", "DII"):
        grp = cum[cum["client_type"] == ptype].sort_values("trade_date")
        if grp.empty or "fut_idx_ls_pct" not in grp.columns:
            continue
        color = _COLORS[ptype]
        fig.add_trace(go.Scatter(
            x=grp["trade_date"], y=grp["fut_idx_ls_pct"],
            name=f"{ptype} L/S%", mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{ptype}</b> %{{x|%d %b}}: L/S = <b>%{{y:.1f}}%</b><extra></extra>"
            ),
        ))
    fig.add_hline(y=50, line_dash="dash", line_width=1.2,
                  line_color="rgba(255,255,255,0.25)",
                  annotation_text="50% (equal long/short)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="rgba(255,255,255,0.4)"))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Long as % of Total (Long+Short)",
                   showgrid=True, gridcolor=GRID_COLOR,
                   ticksuffix="%", range=[0, 100]),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=12)),
        height=280, margin=dict(t=30, b=40, l=70, r=40),
        hovermode="x unified",
    )
    return fig


# ── Position tables (used in tab_pos) ────────────────────────────────────────

def _daily_table(daily: pd.DataFrame) -> None:
    if daily.empty:
        st.caption("No daily data available.")
        return
    pivot_rows = []
    for td, grp in daily.groupby("trade_date"):
        row: dict = {"Date": td}
        for pt in _ORDER:
            r = grp[grp["client_type"] == pt]
            if r.empty:
                row[f"{pt}_Long"] = row[f"{pt}_Short"] = row[f"{pt}_Net"] = row[f"{pt}_LS%"] = None
            else:
                r = r.iloc[0]
                row[f"{pt}_Long"]  = int(r["fut_idx_long"]  or 0)
                row[f"{pt}_Short"] = int(r["fut_idx_short"] or 0)
                row[f"{pt}_Net"]   = int(r["fut_idx_net"]   or 0)
                ls = r.get("fut_idx_ls_pct")
                row[f"{pt}_LS%"] = round(float(ls), 1) if ls is not None and not pd.isna(ls) else None
        pivot_rows.append(row)
    df = pd.DataFrame(pivot_rows).sort_values("Date", ascending=False)
    col_cfg: dict = {"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}
    for pt in _ORDER:
        col_cfg[f"{pt}_Long"]  = st.column_config.NumberColumn(f"{pt} Long",  format="%,d")
        col_cfg[f"{pt}_Short"] = st.column_config.NumberColumn(f"{pt} Short", format="%,d")
        col_cfg[f"{pt}_Net"]   = st.column_config.NumberColumn(f"{pt} Net",   format="%+,d")
        col_cfg[f"{pt}_LS%"]   = st.column_config.NumberColumn(f"{pt} L/S%",  format="%.1f%%")
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _options_daily_table(daily: pd.DataFrame) -> None:
    if daily.empty or "opt_idx_call_net" not in daily.columns:
        st.caption("No options data available.")
        return
    pivot_rows = []
    for td, grp in daily.groupby("trade_date"):
        row: dict = {"Date": td}
        for pt in _ORDER:
            r = grp[grp["client_type"] == pt]
            if r.empty:
                row[f"{pt}_Call"] = row[f"{pt}_Put"] = row[f"{pt}_Delta"] = None
            else:
                r = r.iloc[0]
                row[f"{pt}_Call"]  = int(r.get("opt_idx_call_net", 0) or 0)
                row[f"{pt}_Put"]   = int(r.get("opt_idx_put_net",  0) or 0)
                row[f"{pt}_Delta"] = int(r.get("opt_idx_net",      0) or 0)
        pivot_rows.append(row)
    df = pd.DataFrame(pivot_rows).sort_values("Date", ascending=False)
    col_cfg: dict = {"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}
    for pt in _ORDER:
        col_cfg[f"{pt}_Call"]  = st.column_config.NumberColumn(f"{pt} Call Net",  format="%+,d")
        col_cfg[f"{pt}_Put"]   = st.column_config.NumberColumn(f"{pt} Put Net",   format="%+,d")
        col_cfg[f"{pt}_Delta"] = st.column_config.NumberColumn(f"{pt} Opt Delta", format="%+,d")
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _cumulative_table(cum: pd.DataFrame) -> None:
    if cum.empty:
        st.caption("No cumulative data available.")
        return
    pivot_rows = []
    for td, grp in cum.groupby("trade_date"):
        row: dict = {"Date": td}
        for pt in _ORDER:
            r = grp[grp["client_type"] == pt]
            if r.empty:
                row[f"{pt}_CumNet"] = row[f"{pt}_LS%"] = None
            else:
                r = r.iloc[0]
                row[f"{pt}_CumNet"] = int(r["cum_fut_idx_net"] or 0)
                ls = r.get("fut_idx_ls_pct")
                row[f"{pt}_LS%"] = round(float(ls), 1) if ls is not None and not pd.isna(ls) else None
        pivot_rows.append(row)
    df = pd.DataFrame(pivot_rows).sort_values("Date", ascending=False)
    col_cfg: dict = {"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}
    for pt in _ORDER:
        col_cfg[f"{pt}_CumNet"] = st.column_config.NumberColumn(f"{pt} Cum Net", format="%+,d")
        col_cfg[f"{pt}_LS%"]    = st.column_config.NumberColumn(f"{pt} L/S%",    format="%.1f%%")
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _options_cumulative_table(cum: pd.DataFrame) -> None:
    if cum.empty or "cum_opt_idx_call_net" not in cum.columns:
        st.caption("No cumulative options data available.")
        return
    pivot_rows = []
    for td, grp in cum.groupby("trade_date"):
        row: dict = {"Date": td}
        for pt in _ORDER:
            r = grp[grp["client_type"] == pt]
            if r.empty:
                row[f"{pt}_CumCall"] = row[f"{pt}_CumPut"] = row[f"{pt}_CumDelta"] = None
            else:
                r = r.iloc[0]
                row[f"{pt}_CumCall"]  = int(r.get("cum_opt_idx_call_net", 0) or 0)
                row[f"{pt}_CumPut"]   = int(r.get("cum_opt_idx_put_net",  0) or 0)
                row[f"{pt}_CumDelta"] = int(r.get("cum_opt_idx_net",      0) or 0)
        pivot_rows.append(row)
    df = pd.DataFrame(pivot_rows).sort_values("Date", ascending=False)
    col_cfg: dict = {"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY")}
    for pt in _ORDER:
        col_cfg[f"{pt}_CumCall"]  = st.column_config.NumberColumn(f"{pt} Cum Call",  format="%+,d")
        col_cfg[f"{pt}_CumPut"]   = st.column_config.NumberColumn(f"{pt} Cum Put",   format="%+,d")
        col_cfg[f"{pt}_CumDelta"] = st.column_config.NumberColumn(f"{pt} Cum Delta", format="%+,d")
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


# ── FII Index Focus Analysis ──────────────────────────────────────────────────

_INDEX_DEFS: list[tuple[str, str, str, str, str]] = [
    # (display_name, futures_cat,         options_cat,          color,     icon)
    ("NIFTY",      "NIFTY FUTURES",      "NIFTY OPTIONS",      "#2196f3", "🔵"),
    ("BANKNIFTY",  "BANKNIFTY FUTURES",  "BANKNIFTY OPTIONS",  "#ff9800", "🟠"),
    ("FINNIFTY",   "FINNIFTY FUTURES",   "FINNIFTY OPTIONS",   "#4caf50", "🟢"),
    ("MIDCPNIFTY", "MIDCPNIFTY FUTURES", "MIDCPNIFTY OPTIONS", "#9c27b0", "🟣"),
    ("NIFTYNXT50", "NIFTYNXT50 FUTURES", "NIFTYNXT50 OPTIONS", "#00bcd4", "🔷"),
]


def _index_strategy(
    fut_net: int,
    opt_net: int,
    total_act: int,
    fii_call_net: int | None = None,
    fii_put_net: int | None = None,
) -> tuple[str, str, str]:
    """
    Returns (verdict_label, verdict_color, one_line_reason).

    Uses call/put split from participant OI when available (gives exact direction).
    Falls back to futures-first logic when only combined options net is available.
    """
    if total_act < 500:
        return "INACTIVE", "#444444", "Negligible FII activity in this index"

    fut_bull = fut_net > 300
    fut_bear = fut_net < -300
    fut_flat = not fut_bull and not fut_bear

    # ── Path A: call/put split available → exact options direction ────────────
    if fii_call_net is not None and fii_put_net is not None:
        opt_delta  = fii_call_net - fii_put_net      # +ve = bullish, -ve = bearish
        opt_bull   = opt_delta >  10_000
        opt_bear   = opt_delta < -10_000

        if fut_bull and opt_bull:
            return "🟢 BULLISH", "#00c853", f"Long futures ({fut_net:+,}) + call-heavy options → dual confirmation"
        elif fut_bull and opt_bear:
            return "🟡 CAUTIOUSLY BEARISH", "#ff9800", f"Long futures ({fut_net:+,}) but put-heavy options — options overrule futures here"
        elif fut_bull:
            return "🟢 MILDLY BULLISH", "#69f0ae", f"Long futures ({fut_net:+,}), options balanced"
        elif fut_bear and opt_bear:
            return "🔴 BEARISH", "#ff5252", f"Short futures ({fut_net:+,}) + put-heavy options → strong bearish"
        elif fut_bear and opt_bull:
            return "🟡 CONFLICTED", "#ffca28", f"Short futures ({fut_net:+,}) but call-heavy options — wait for clarity"
        elif fut_bear:
            return "🔴 BEARISH", "#ff5252", f"Net short futures ({fut_net:+,})"
        elif opt_bear:
            return "🟠 OPTIONS BEARISH", "#ff9800", "Futures flat; FII holding more puts than calls"
        elif opt_bull:
            return "🟢 OPTIONS BULLISH", "#69f0ae", "Futures flat; FII holding more calls than puts"
        else:
            return "⚪ NEUTRAL", "#888888", "Futures flat; options balanced — rangebound stance"

    # ── Path B: combined options net only → futures direction is primary ──────
    if fut_bull:
        return "🟢 BULLISH", "#69f0ae", f"Long futures ({fut_net:+,}) — see Index Options section for call/put split"
    elif fut_bear:
        return "🔴 BEARISH", "#ff5252", f"Short futures ({fut_net:+,})"
    elif opt_net < -1_000:
        return "⚪ OPTION WRITER", "#888888", "Flat futures; selling options — collecting premium (rangebound)"
    elif opt_net > 1_000:
        return "🔵 OPTION BUYER", "#90caf9", "Flat futures; buying options — directional bet (check call/put split)"
    else:
        return "⚪ NEUTRAL", "#888888", "No significant directional positioning"


def _fii_index_focus(
    fii_stats: pd.DataFrame,
    fii_stats_h: pd.DataFrame,
    participant_latest: pd.DataFrame | None = None,
) -> None:
    """
    Per-index FII positioning analysis — clear BULLISH/BEARISH/NEUTRAL verdict per index.

    Uses participant OI call/put split (for Nifty = 86% of index options) to give
    exact options direction, not just net contracts.
    """
    if fii_stats.empty:
        return

    df = fii_stats.copy()
    if "net_value_cr" not in df.columns:
        df["net_value_cr"]  = df["buy_value_cr"]  - df["sell_value_cr"]
    if "net_contracts" not in df.columns:
        df["net_contracts"] = df["buy_contracts"] - df["sell_contracts"]

    # ── Extract FII call/put net from participant OI (index-level aggregate) ──
    # Since NIFTY is ~86% of index options volume, aggregate call/put ≈ NIFTY stance
    fii_call_net: int | None = None
    fii_put_net:  int | None = None
    if participant_latest is not None and not participant_latest.empty:
        fii_row = participant_latest[participant_latest["client_type"] == "FII"]
        if not fii_row.empty:
            r = fii_row.iloc[0]
            fii_call_net = int(r.get("opt_idx_call_net", 0) or 0)
            fii_put_net  = int(r.get("opt_idx_put_net",  0) or 0)

    # ── Build per-index stats ─────────────────────────────────────────────────
    idx_stats: list[dict] = []
    for name, fut_cat, opt_cat, color, icon in _INDEX_DEFS:
        fr  = df[df["category"] == fut_cat]
        or_ = df[df["category"] == opt_cat]
        if fr.empty and or_.empty:
            continue

        fut_net_c  = int(fr["net_contracts"].iloc[0])  if not fr.empty  else 0
        fut_net_cr = float(fr["net_value_cr"].iloc[0]) if not fr.empty  else 0.0
        opt_net_c  = int(or_["net_contracts"].iloc[0]) if not or_.empty else 0
        opt_net_cr = float(or_["net_value_cr"].iloc[0]) if not or_.empty else 0.0

        buy_act  = (int(fr["buy_contracts"].iloc[0])  if not fr.empty  else 0) + \
                   (int(or_["buy_contracts"].iloc[0]) if not or_.empty else 0)
        sell_act = (int(fr["sell_contracts"].iloc[0]) if not fr.empty  else 0) + \
                   (int(or_["sell_contracts"].iloc[0]) if not or_.empty else 0)
        total_act = buy_act + sell_act

        # ── Week-over-week: find the most recent session that is 5+ trading
        #    days before the latest date (not positional index, avoids holiday gaps)
        trend_str = ""
        if not fii_stats_h.empty:
            latest_date = fii_stats_h["trade_date"].max()
            all_dates   = sorted(fii_stats_h["trade_date"].unique())
            # 5 trading days ago: last date >= 7 calendar days back
            cutoff = latest_date - pd.Timedelta(days=7)
            prior_dates = [d for d in all_dates if d <= cutoff]
            if prior_dates:
                prev_date = prior_dates[-1]
                prev_df = fii_stats_h[fii_stats_h["trade_date"] == prev_date].copy()
                if "net_contracts" not in prev_df.columns:
                    prev_df["net_contracts"] = prev_df["buy_contracts"] - prev_df["sell_contracts"]
                prev_fr  = prev_df[prev_df["category"] == fut_cat]
                prev_or_ = prev_df[prev_df["category"] == opt_cat]
                prev_fut = int(prev_fr["net_contracts"].iloc[0])  if not prev_fr.empty  else 0
                prev_opt = int(prev_or_["net_contracts"].iloc[0]) if not prev_or_.empty else 0
                chg_fut  = fut_net_c - prev_fut
                chg_opt  = opt_net_c - prev_opt
                if abs(chg_fut) > 200 or abs(chg_opt) > 500:
                    fa = "+" if chg_fut >= 0 else ""
                    oa = "+" if chg_opt >= 0 else ""
                    trend_str = (
                        f"vs {prev_date.strftime('%d %b')} — "
                        f"Fut: {fa}{chg_fut:,}  Opt: {oa}{chg_opt:,}"
                    )

        # Apply call/put split only to primary index (NIFTY = index 0 in _INDEX_DEFS)
        use_call_put = (name == "NIFTY" and fii_call_net is not None)

        idx_stats.append({
            "name": name, "color": color, "icon": icon,
            "fut_net_c": fut_net_c, "opt_net_c": opt_net_c,
            "fut_net_cr": fut_net_cr, "opt_net_cr": opt_net_cr,
            "total_act": total_act,
            "total_net_cr": fut_net_cr + opt_net_cr,
            "trend": trend_str,
            "call_net": fii_call_net if use_call_put else None,
            "put_net":  fii_put_net  if use_call_put else None,
        })

    if not idx_stats:
        return

    # Sort by activity, drop indices with < 0.5% of total activity
    idx_stats.sort(key=lambda x: x["total_act"], reverse=True)
    total_all = sum(s["total_act"] for s in idx_stats) or 1
    visible   = [s for s in idx_stats if s["total_act"] / total_all >= 0.005]
    if not visible:
        visible = idx_stats[:3]
    max_act = visible[0]["total_act"] or 1

    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 12px'>"
        "🎯 FII INDEX FOCUS — Where Are They Actively Positioning?</div>",
        unsafe_allow_html=True,
    )

    rank_labels = ["PRIMARY", "2ND", "3RD", "4TH", "5TH"]
    cols = st.columns(max(1, len(visible)))

    for i, s in enumerate(visible):
        color   = s["color"]
        act_pct = s["total_act"] / total_all * 100
        bar_pct = int(s["total_act"] / max_act * 82)

        verdict, v_color, reason = _index_strategy(
            s["fut_net_c"], s["opt_net_c"], s["total_act"],
            s["call_net"], s["put_net"],
        )

        fut_c  = "#00c853" if s["fut_net_c"] > 0 else ("#ff5252" if s["fut_net_c"] < 0 else "#888")
        opt_c  = "#00c853" if s["opt_net_c"] > 0 else ("#ff5252" if s["opt_net_c"] < 0 else "#888")
        flow_c = "#00c853" if s["total_net_cr"] > 0 else "#ff5252"
        flow_str = (
            f"+₹{s['total_net_cr']:,.0f} Cr" if s["total_net_cr"] > 0
            else f"−₹{abs(s['total_net_cr']):,.0f} Cr"
        )
        cp_note = ""
        if s["call_net"] is not None:
            opt_delta = s["call_net"] - s["put_net"]
            d_c = "#00c853" if opt_delta > 0 else "#ff5252"
            cp_note = (
                f"<div style='font-size:9px;color:rgba(255,255,255,0.35);margin-top:3px'>"
                f"Call Net: <span style='color:{d_c}'>{s['call_net']:+,}</span> "
                f"· Put Net: <span style='color:{d_c}'>{s['put_net']:+,}</span> "
                f"· Delta: <span style='color:{d_c};font-weight:700'>{opt_delta:+,}</span>"
                f"</div>"
            )

        trend_html = ""
        if s["trend"]:
            trend_html = (
                f"<div style='font-size:9px;color:rgba(255,255,255,0.3);"
                f"margin-top:5px;font-style:italic'>{s['trend']}</div>"
            )

        cols[i].markdown(
            f"<div style='border:1px solid {color}55;border-radius:12px;"
            f"padding:14px 16px;background:{color}0b;height:100%'>"

            # Header
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:3px'>"
            f"<span style='font-size:13px;font-weight:900;color:{color}'>"
            f"{s['icon']} {s['name']}</span>"
            f"<span style='font-size:9px;font-weight:700;background:{color}22;"
            f"color:{color};padding:1px 7px;border-radius:8px'>"
            f"{rank_labels[i] if i < len(rank_labels) else ''}</span>"
            f"</div>"

            # Activity bar
            f"<div style='font-size:9px;color:rgba(255,255,255,0.35);margin-bottom:5px'>"
            f"{act_pct:.0f}% of total FII F&amp;O volume</div>"
            f"<div style='background:#1a1a1a;border-radius:3px;height:7px;"
            f"overflow:hidden;margin-bottom:10px'>"
            f"<div style='width:{bar_pct}%;height:100%;background:{color};"
            f"opacity:0.75;border-radius:3px'></div>"
            f"</div>"

            # Futures + Options net numbers
            f"<div style='display:grid;grid-template-columns:1fr 1fr;"
            f"gap:5px;margin-bottom:6px'>"
            f"<div style='background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px'>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.4)'>Futures Net</div>"
            f"<div style='font-size:15px;font-weight:800;color:{fut_c}'>{s['fut_net_c']:+,}</div>"
            f"</div>"
            f"<div style='background:rgba(255,255,255,0.03);border-radius:6px;padding:6px 8px'>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.4)'>Options Net</div>"
            f"<div style='font-size:15px;font-weight:800;color:{opt_c}'>{s['opt_net_c']:+,}</div>"
            f"</div>"
            f"</div>"

            f"{cp_note}"

            # Net flow
            f"<div style='font-size:11px;font-weight:700;color:{flow_c};"
            f"margin:6px 0'>Net Flow: {flow_str}</div>"

            # Verdict badge — the key clear signal
            f"<div style='padding:6px 10px;background:{v_color}1a;"
            f"border:1px solid {v_color}44;border-radius:8px;margin-top:4px'>"
            f"<div style='font-size:13px;font-weight:900;color:{v_color};line-height:1.2'>"
            f"{verdict}</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.5);margin-top:2px;"
            f"line-height:1.4'>{reason}</div>"
            f"</div>"

            f"{trend_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Plain-English Summary ─────────────────────────────────────────────────
    if visible:
        parts = []
        for s in visible[:2]:
            act_pct = int(s["total_act"] / total_all * 100)
            v, _, reason = _index_strategy(s["fut_net_c"], s["opt_net_c"], s["total_act"],
                                           s["call_net"], s["put_net"])
            parts.append(f"**{s['icon']} {s['name']}** ({act_pct}%): {v} — {reason.lower()}")
        hidden = [s["name"] for s in idx_stats if s not in visible]
        if hidden:
            parts.append(f"*{', '.join(hidden)}: < 0.5% activity — excluded*")
        st.caption("  ·  ".join(parts))


# ── FII Money Flow (tab) ──────────────────────────────────────────────────────

def _fii_stats_table(fii_stats: pd.DataFrame) -> None:
    if fii_stats.empty:
        st.info("No FII Derivatives Statistics data. Run: `python -m src.cli backfill-fii-stats 365`")
        return
    data_date = fii_stats["trade_date"].iloc[0]
    st.caption(f"FII per-index buy/sell — {data_date.strftime('%d %b %Y')}")
    display = fii_stats.copy()
    display["net_value_cr"]  = display["buy_value_cr"] - display["sell_value_cr"]
    display["net_contracts"] = display["buy_contracts"] - display["sell_contracts"]
    col_cfg = {
        "category":       st.column_config.TextColumn("Index / Category"),
        "buy_contracts":  st.column_config.NumberColumn("Buy Contracts",  format="%,d"),
        "sell_contracts": st.column_config.NumberColumn("Sell Contracts", format="%,d"),
        "net_contracts":  st.column_config.NumberColumn("Net Contracts",  format="%+,d",
            help="Buy − Sell contracts. +ve = FII net buyer."),
        "buy_value_cr":   st.column_config.NumberColumn("Buy (Cr)",  format="%.2f"),
        "sell_value_cr":  st.column_config.NumberColumn("Sell (Cr)", format="%.2f"),
        "net_value_cr":   st.column_config.NumberColumn("Net Flow (Cr)", format="%+.2f",
            help="Buy − Sell value in ₹ Crore. +ve = net buyer (money in). −ve = net seller (outflow)."),
        "oi_contracts":   st.column_config.NumberColumn("OI Contracts", format="%,d"),
        "trade_date": None, "oi_value_cr": None,
    }
    st.dataframe(
        display.sort_values("category"),
        hide_index=True, use_container_width=True, column_config=col_cfg,
        column_order=["category","buy_contracts","sell_contracts","net_contracts",
                      "buy_value_cr","sell_value_cr","net_value_cr","oi_contracts"],
    )


def _fii_flow_chart(fii_stats_hist: pd.DataFrame) -> go.Figure:
    if fii_stats_hist.empty:
        return go.Figure()
    # Use per-index futures only (not the aggregate "INDEX FUTURES" row which
    # double-counts Nifty + BankNifty + FinNifty + MidcapNifty)
    cats = ["NIFTY FUTURES", "BANKNIFTY FUTURES", "FINNIFTY FUTURES", "MIDCPNIFTY FUTURES"]
    cat_colors = {
        "NIFTY FUTURES":      "#2196f3",
        "BANKNIFTY FUTURES":  "#ff9800",
        "FINNIFTY FUTURES":   "#4caf50",
        "MIDCPNIFTY FUTURES": "#9c27b0",
    }
    fig = go.Figure()
    for cat in cats:
        grp = fii_stats_hist[fii_stats_hist["category"] == cat].sort_values("trade_date")
        if grp.empty:
            continue
        bar_colors = ["#00c853" if v >= 0 else "#ff5252" for v in grp["net_value_cr"]]
        fig.add_trace(go.Bar(
            x=grp["trade_date"], y=grp["net_value_cr"],
            name=cat.replace(" FUTURES", ""),
            marker_color=cat_colors[cat],
            opacity=0.8,
            hovertemplate=(
                f"<b>{cat}</b><br>%{{x|%d %b}}<br>"
                "Net Flow: <b>Rs. %{y:+.2f} Cr</b><extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1.2)
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b", tickfont=dict(size=10)),
        yaxis=dict(title="Net Buy/(Sell) Rs. Crore", showgrid=True,
                   gridcolor=GRID_COLOR, tickformat="+.0f"),
        barmode="group",
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        height=300, margin=dict(t=30, b=40, l=80, r=40),
        hovermode="x unified",
    )
    return fig


# ── FII Today Activity Panel ──────────────────────────────────────────────────

def _fii_today_panel(mi) -> None:
    """
    Three independent data sources → three independent signals → one plain-English market reading.
    Futures OI (direction bet) | Options OI+PCR (sentiment/hedging) | Rupee Flow (money conviction)
    """
    import re as _re

    snap = mi.fii_today
    if snap is None:
        return

    # ─── 1. DERIVE FUTURES SIGNAL ─────────────────────────────────────────────
    oi_delta = snap.oi_net_change  # positive = net new longs, negative = net new shorts
    cum_net  = snap.cumulative_net

    if oi_delta > 5_000:
        fut_label, fut_color = "ADDING LONGS", "#00c853"
        fut_what  = f"Added {snap.long_change:+,} long contracts today (net {oi_delta:+,})"
        fut_means = "FII placing new bullish bets. Fresh buying into the market."
    elif oi_delta < -5_000:
        fut_label, fut_color = "ADDING SHORTS", "#ff5252"
        fut_what  = f"Added {snap.short_change:+,} short contracts today (net {oi_delta:+,})"
        fut_means = "FII placing new bearish bets. Selling pressure likely to continue."
    elif snap.action_label == "SHORT COVERING" or (oi_delta > 0 and cum_net < -50_000):
        fut_label, fut_color = "COVERING SHORTS", "#ff9800"
        fut_what  = f"Reducing short position (net {oi_delta:+,} today)"
        fut_means = "FII buying back their shorts. This IS buying pressure, even if they are still net short overall."
    elif snap.action_label == "LONG UNWINDING" or (oi_delta < 0 and cum_net > 50_000):
        fut_label, fut_color = "REDUCING LONGS", "#ff9800"
        fut_what  = f"Reducing long position (net {oi_delta:+,} today)"
        fut_means = "FII selling their longs. Profit-taking or risk reduction."
    else:
        fut_label, fut_color = "HOLDING POSITION", "#888888"
        fut_what  = f"Minimal change today (net {oi_delta:+,})"
        fut_means = "FII not making new directional bets. Market awaiting a catalyst."

    if abs(cum_net) > 150_000:
        pos_note = f"Total open: {cum_net:+,} ({'MASSIVE SHORT' if cum_net < 0 else 'MASSIVE LONG'} — squeeze potential)"
        pos_note_c = "#ff9800"
    elif abs(cum_net) > 75_000:
        pos_note = f"Total open: {cum_net:+,} ({'heavily short' if cum_net < 0 else 'heavily long'})"
        pos_note_c = "#ffca28"
    else:
        pos_note = f"Total open: {cum_net:+,}"
        pos_note_c = "rgba(255,255,255,0.35)"

    fut_dir = 1 if oi_delta > 2_000 else (-1 if oi_delta < -2_000 else 0)

    # ─── 2. DERIVE OPTIONS SIGNAL ─────────────────────────────────────────────
    pcr_sig = next((s for s in mi.signals if s.category == "PCR"), None)
    opt_sig = next((s for s in mi.signals if s.category == "Options OI"), None)

    pcr_match = _re.search(r'PCR\s+([\d.]+)', pcr_sig.headline) if pcr_sig else None
    pcr_val   = float(pcr_match.group(1)) if pcr_match else None

    if pcr_val and pcr_val > 1.25:
        pcr_label, pcr_color = "EXTREME FEAR", "#ff9800"
        pcr_means = (f"PCR {pcr_val:.2f} — Everyone is buying put options (protection). "
                     "This is actually a contrarian BULLISH sign — market has already priced in the fall.")
    elif pcr_val and pcr_val > 1.10:
        pcr_label, pcr_color = "DEFENSIVE", "#ffca28"
        pcr_means = (f"PCR {pcr_val:.2f} — More puts than calls. People protecting their portfolio. "
                     "Mild caution but not extreme.")
    elif pcr_val and pcr_val < 0.72:
        pcr_label, pcr_color = "COMPLACENT", "#ff5252"
        pcr_means = (f"PCR {pcr_val:.2f} — Too many call buyers, not enough protection. "
                     "When everyone is buying calls at highs, it usually means a correction is near.")
    elif pcr_val and pcr_val < 0.82:
        pcr_label, pcr_color = "SLIGHTLY COMPLACENT", "#ffca28"
        pcr_means = f"PCR {pcr_val:.2f} — Slight excess of calls. Mild caution signal."
    else:
        pcr_label, pcr_color = "NEUTRAL", "#888888"
        pcr_means = (f"PCR {pcr_val:.2f} — Balanced market." if pcr_val else
                     "PCR data not available.")

    # FII options stance (are they holding more puts or calls?)
    if opt_sig:
        if opt_sig.direction > 0:
            fii_opt_label, fii_opt_color = "MORE CALLS", "#69f0ae"
            fii_opt_means = "FII options position is bullish (more calls than puts held net)."
        else:
            fii_opt_label, fii_opt_color = "MORE PUTS", "#ff9800"
            fii_opt_means = "FII options position is defensive (more puts than calls held net). Could be hedging or directional bet down."
    else:
        fii_opt_label, fii_opt_color = "N/A", "#555555"
        fii_opt_means = "Options stance data unavailable."

    opt_dir = (pcr_sig.direction if pcr_sig else 0)

    # ─── 3. DERIVE MONEY FLOW SIGNAL ──────────────────────────────────────────
    has_flow = snap.total_net_cr != 0.0

    if not has_flow:
        flow_label, flow_color = "NO DATA", "#555555"
        flow_what  = "FII stats not loaded"
        flow_means = "Run: python -m src.cli backfill-fii-stats 90 to enable this signal."
    elif snap.total_net_cr > 500:
        flow_label, flow_color = "NET BUYER", "#00c853"
        flow_what  = f"Bought ₹{snap.total_net_cr:,.0f} Cr  (Fut ₹{snap.fut_net_cr:+,.0f} + Opt ₹{snap.opt_net_cr:+,.0f})"
        flow_means = "Real money confirms bullish stance. FII deploying capital on long side."
    elif snap.total_net_cr < -500:
        flow_label, flow_color = "NET SELLER", "#ff5252"
        flow_what  = f"Sold ₹{abs(snap.total_net_cr):,.0f} Cr  (Fut ₹{snap.fut_net_cr:+,.0f} + Opt ₹{snap.opt_net_cr:+,.0f})"
        flow_means = "Real money confirms bearish stance. Capital moving out of F&O positions."
    else:
        flow_label, flow_color = "NEUTRAL FLOW", "#888888"
        flow_what  = f"Net ₹{snap.total_net_cr:+,.0f} Cr  (Fut ₹{snap.fut_net_cr:+,.0f} + Opt ₹{snap.opt_net_cr:+,.0f})"
        flow_means = "No strong conviction in money deployment. FII not taking a big directional bet today."

    flow_dir = 1 if snap.total_net_cr > 200 else (-1 if snap.total_net_cr < -200 else 0)

    # ─── 4. COMBINED VERDICT ──────────────────────────────────────────────────
    score = fut_dir + opt_dir + flow_dir
    n_agree = sum(1 for d in [fut_dir, opt_dir, flow_dir]
                  if d != 0 and ((d > 0) == (score > 0)))

    squeeze_active = mi.tomorrow_verdict and mi.tomorrow_verdict.squeeze_risk
    covering_active = mi.tomorrow_verdict and mi.tomorrow_verdict.short_covering_active

    if covering_active:
        combined_label, combined_color = "SHORT COVERING — LEAN LONG", "#ff9800"
        combined_action = (
            "FII is actively buying back their short positions. Each covered short = 1 BUY order. "
            "Even if their overall stance is bearish, the TRANSACTION today is bullish. "
            "Market likely to see upward pressure."
        )
    elif score >= 2:
        combined_label, combined_color = "BULLISH — EXPECT BUYING", "#00c853"
        combined_action = (
            f"{n_agree} out of 3 data sources agree: bullish. "
            "FII positioning, options market, and money flow all point UP. "
            "High conviction setup — market likely to see buying interest."
        )
    elif score <= -2:
        combined_label, combined_color = "BEARISH — EXPECT SELLING", "#ff5252"
        combined_action = (
            f"{n_agree} out of 3 data sources agree: bearish. "
            "FII adding shorts, options defensive, money flowing out. "
            + ("BUT ⚡ SQUEEZE RISK ACTIVE: massive short + DII buying = sharp reversal possible on any positive news. "
               "Do NOT add fresh shorts." if squeeze_active else
               "Selling pressure likely to continue unless DII steps in.")
        )
    elif score > 0:
        combined_label, combined_color = "MILD BULLISH TILT", "#69f0ae"
        combined_action = (
            "More bullish than bearish signals today, but not all 3 agree. "
            "Smaller position size recommended. Wait for clearer confirmation."
        )
    elif score < 0:
        combined_label, combined_color = "MILD BEARISH TILT", "#ff9800"
        combined_action = (
            "More bearish than bullish signals today, but mixed picture. "
            "Avoid aggressive longs. "
            + ("⚡ Squeeze risk active — do not short." if squeeze_active else
               "Small short bias but high risk if global news turns positive.")
        )
    else:
        combined_label, combined_color = "NO CLEAR DIRECTION", "#888888"
        combined_action = (
            "All 3 data sources are neutral or conflicting. "
            "Best action: Stay flat. Wait for 2+ sources to agree before taking a trade."
        )

    # ─── RENDER ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:14px 0 10px'>"
        "🔍 HOW FIIs ARE POSITIONED TODAY — 3 Independent Data Sources</div>",
        unsafe_allow_html=True,
    )

    def _src_card(col, icon, src_name, src_note, label, color, what, means):
        col.markdown(
            f"<div style='border:1px solid {color}44;border-radius:10px;"
            f"padding:12px 14px;background:{color}09;height:100%'>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.3);letter-spacing:1px;margin-bottom:4px'>"
            f"{icon} {src_name}</div>"
            f"<div style='font-size:8px;color:rgba(255,255,255,0.2);margin-bottom:8px'>{src_note}</div>"
            f"<div style='font-size:18px;font-weight:900;color:{color};line-height:1.1'>{label}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:7px;"
            f"border-top:1px solid rgba(255,255,255,0.06);padding-top:7px'>{what}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.35);margin-top:5px;"
            f"font-style:italic'>{means}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    _src_card(
        c1, "📈", "INDEX FUTURES", "Most important — FII directional bets",
        fut_label, fut_color, fut_what,
        f"{fut_means}  {pos_note}",
    )

    _src_card(
        c2, "🎯", "INDEX OPTIONS", "PCR = market fear/greed  |  FII Options position",
        pcr_label, pcr_color,
        f"PCR: {pcr_val:.2f}" if pcr_val else "PCR N/A",
        f"{pcr_means}  FII position: {fii_opt_label} — {fii_opt_means}",
    )

    _src_card(
        c3, "💰", "RUPEE FLOW (₹ Cr)", "Actual money deployed — confirms or denies OI",
        flow_label, flow_color, flow_what, flow_means,
    )

    # Combined verdict card
    n_sources = f"{n_agree}/3 sources agree" if score != 0 else "Sources conflicting"
    c4.markdown(
        f"<div style='border:2px solid {combined_color}77;border-radius:10px;"
        f"padding:12px 14px;background:{combined_color}14;height:100%'>"
        f"<div style='font-size:9px;color:rgba(255,255,255,0.3);letter-spacing:1px;margin-bottom:4px'>"
        f"⚡ TODAY'S READING</div>"
        f"<div style='font-size:8px;color:rgba(255,255,255,0.2);margin-bottom:8px'>{n_sources}</div>"
        f"<div style='font-size:16px;font-weight:900;color:{combined_color};line-height:1.2'>"
        f"{combined_label}</div>"
        f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:8px;"
        f"border-top:1px solid rgba(255,255,255,0.06);padding-top:8px;line-height:1.5'>"
        f"{combined_action}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ─── PLAIN ENGLISH MARKET READING ─────────────────────────────────────────
    dii_row   = None
    dii_net_v = 0
    for s in mi.signals:
        if "dii" in s.name.lower() or "squeeze" in s.name.lower():
            if "47" in s.description or "dii" in s.description.lower():
                dii_net_v = snap.cumulative_net  # proxy
                break

    # Compose the narrative from actual data
    fut_sentence = (
        f"FII {'added' if oi_delta < 0 else 'added'} "
        f"{abs(snap.short_change if oi_delta < 0 else snap.long_change):,} "
        f"{'short' if oi_delta < 0 else 'long'} contracts today"
        f"{' (₹' + f'{abs(snap.total_net_cr):,.0f}' + ' Cr ' + ('selling' if snap.total_net_cr < 0 else 'buying') + ')' if has_flow else ''}."
    )

    squeeze_sentence = (
        "⚡ However, FII already holds a MASSIVE short position ("
        f"{cum_net:+,} net) with DII actively buying. "
        "This creates a SQUEEZE SETUP — any positive global news can force FII to cover, causing a sudden sharp rally. "
        "Do NOT add fresh short positions in this environment."
        if (squeeze_active and oi_delta < 0) else
        "FII is covering shorts — even if their overall position is still bearish, the DAILY ACTION is buying. This supports the upside."
        if covering_active else ""
    )

    pcr_sentence = (
        f"Options market: PCR {pcr_val:.2f} — {pcr_means}"
        if pcr_val else ""
    )

    action_sentence = (
        "Trading view: " + (
            "Lean long (buy dips). Squeeze + covering = upside momentum."
            if (squeeze_active or covering_active) and score >= 0 else
            "Avoid new shorts. Squeeze risk too high — losses can be sharp and sudden."
            if squeeze_active and score < 0 else
            "Bullish bias. Consider buying on dips with tight stop."
            if score >= 2 else
            "Bearish bias. Sell strength, not weakness. Tight stop on any short."
            if score <= -2 else
            "Mixed signals. Best to stay flat and watch for tomorrow's opening trend."
        )
    )

    narrative = "  ".join(filter(None, [fut_sentence, pcr_sentence, squeeze_sentence, action_sentence]))

    st.markdown(
        f"<div style='border:1px solid rgba(255,255,255,0.07);border-radius:8px;"
        f"padding:12px 16px;background:rgba(255,255,255,0.02);margin-top:10px'>"
        f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.3);"
        f"letter-spacing:1.5px;margin-bottom:6px'>📝 PLAIN ENGLISH MARKET READING</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.7);line-height:1.7'>{narrative}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    pass  # rendering complete above


# ── Backtest rendering helpers ────────────────────────────────────────────────

def _pct_color(pct: float) -> str:
    if pct >= 0.65:  return "#00c853"
    if pct >= 0.55:  return "#69f0ae"
    if pct >= 0.45:  return "#ffca28"
    return "#ff5252"


def _backtest_summary_cards(bt) -> None:
    """Row of 5 key accuracy metric cards."""
    c1, c2, c3, c4, c5 = st.columns(5)

    def _card(col, label, val, sub, color, tooltip):
        col.markdown(
            f"<div title='{tooltip}' style='border:1px solid {color}44;border-radius:12px;"
            f"padding:14px 16px;background:{color}0c;text-align:center;cursor:help'>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.35);"
            f"letter-spacing:1.5px;margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:32px;font-weight:900;color:{color};line-height:1'>{val}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    total = bt.total_signals
    n_correct = sum(1 for r in bt.records if r.outcome == "CORRECT")

    _card(c1, "OVERALL HIT RATE",
          f"{bt.hit_rate_overall:.0%}",
          f"{n_correct}/{total} signals correct",
          _pct_color(bt.hit_rate_overall),
          "OVERALL HIT RATE: % of ALL signals (UP + DOWN + SIDEWAYS) predicted correctly. "
          "Includes SIDEWAYS calls which are hard to get right. "
          "Do NOT use this alone for trading — look at Directional Accuracy instead. "
          "Green >= 65%, Yellow >= 45%, Red < 45%.")

    _card(c2, "DIRECTIONAL ACCURACY",
          f"{bt.hit_rate_directional:.0%}",
          f"UP+DOWN: {bt.correct_directional}/{bt.directional_signals}",
          _pct_color(bt.hit_rate_directional),
          "DIRECTIONAL ACCURACY: The most important trading metric. "
          "Of all UP and DOWN calls only (ignoring SIDEWAYS), how many were correct? "
          "TRADING RULE: Only take trades when system gives UP or DOWN signal. "
          "Above 55% = edge over random. Above 60% = strong edge. "
          f"({bt.correct_directional} correct out of {bt.directional_signals} directional calls.)")

    _card(c3, "SIMULATED P&L",
          f"{bt.cumulative_pnl:+.1f}%",
          f"avg {bt.avg_pnl_per_signal:+.2f}% / signal | Sharpe {bt.sharpe_sim:.2f}",
          "#00c853" if bt.cumulative_pnl > 0 else "#ff5252",
          "SIMULATED P&L: If you followed every signal with 1 unit unleveraged — "
          "long Nifty on UP signal, short Nifty on DOWN signal, flat on SIDEWAYS — "
          "this is your total return over the backtest period. "
          "Sharpe Ratio: annualised (mean return / std). "
          "Sharpe > 1.0 = good risk-adjusted. > 2.0 = excellent. Negative = losing strategy. "
          "Remember: real trading has brokerage costs, slippage, and you would use leverage.")

    _card(c4, "HIGH CONF. ACCURACY",
          f"{bt.hit_rate_high:.0%}",
          f"{bt.high_conf_correct}/{bt.high_conf_signals} HIGH confidence",
          _pct_color(bt.hit_rate_high),
          "HIGH CONFIDENCE ACCURACY: Accuracy when system says HIGH confidence specifically. "
          "Ideally this should be HIGHER than overall accuracy — meaning the system knows when it is sure. "
          "If HIGH conf accuracy is lower than overall, the confidence calibration is broken. "
          "TRADING RULE: High confidence signals should get larger position size.")

    _card(c5, "MAX DRAWDOWN",
          f"{bt.max_drawdown:.1f}%",
          "worst underwater stretch",
          "#ff5252" if bt.max_drawdown < -3 else "#ffca28",
          "MAX DRAWDOWN: The worst cumulative loss you would have suffered following signals before recovering. "
          "Example: -6% means at some point you were down 6% before the system recovered. "
          "Green if better than -3%. Yellow -3% to -6%. Red worse than -6%. "
          "RISK MANAGEMENT: Never risk more than 2x Max Drawdown on this strategy in your total capital.")


def _backtest_breakdown_row(bt) -> None:
    """Signal-type and confidence breakdown."""
    c_up, c_dn, c_sw, c_sq, c_cov = st.columns(5)

    # Pre-compute squeeze/covering counts correctly (accuracy is a float 0-1, need int counts)
    _sq_total  = sum(1 for r in bt.records if r.squeeze_risk and r.verdict in ("UP","DOWN"))
    _sq_correct = round(bt.squeeze_accuracy * _sq_total) if _sq_total > 0 else 0
    _cov_total  = sum(1 for r in bt.records if r.short_covering and r.verdict in ("UP","DOWN"))
    _cov_correct = round(bt.covering_accuracy * _cov_total) if _cov_total > 0 else 0

    def _mini(col, emoji, label, correct, total, color, tooltip):
        rate = correct / total if total > 0 else 0
        col.markdown(
            f"<div title='{tooltip}' style='border:1px solid rgba(255,255,255,0.07);border-radius:8px;"
            f"padding:10px 12px;text-align:center;background:rgba(255,255,255,0.02);cursor:help'>"
            f"<div style='font-size:18px'>{emoji}</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.4);letter-spacing:1px'>{label}</div>"
            f"<div style='font-size:22px;font-weight:900;color:{_pct_color(rate)}'>{rate:.0%}</div>"
            f"<div style='font-size:9px;color:rgba(255,255,255,0.3)'>{correct}/{total}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    _mini(c_up, "▲", "UP SIGNALS", bt.correct_up, bt.up_signals, "#00c853",
          "UP SIGNALS: Times the system predicted market will go UP next day. "
          f"{bt.correct_up} correct out of {bt.up_signals} UP calls. "
          "TRADING ACTION: Buy Nifty futures or buy call options. "
          "Best UP signals have covering=True AND squeeze=True — those are highest conviction.")

    _mini(c_dn, "▼", "DOWN SIGNALS", bt.correct_down, bt.down_signals, "#ff5252",
          "DOWN SIGNALS: Times system predicted market will fall next day. "
          f"{bt.down_signals} DOWN signals in this period. "
          "DOWN signal now requires FII to be ACTIVELY BUILDING fresh shorts (delta_5d < -10K) "
          "AND score <= -7 AND no squeeze floor. Very rare — backtest showed pure score-based "
          "DOWN calls were 0% accurate because bearish FII positioning = squeeze fuel, not sell signal. "
          "If 0/0: no DOWN trade this period = correct (stay flat is better than wrong short).")

    _mini(c_sw, "↔", "SIDEWAYS", bt.correct_sideways, bt.sideways_signals, "#ffca28",
          "SIDEWAYS: Times system said no clear direction — stay flat, do not trade. "
          f"{bt.correct_sideways} correct out of {bt.sideways_signals}. "
          "Accuracy is low because market moves directionally most days (+/-0.25%). "
          "TRADING ACTION: Do NOT trade on SIDEWAYS signals. "
          "If you have open positions, use SIDEWAYS as a warning to trail stop-losses tighter.")

    _mini(c_sq, "⚡", "SQUEEZE RISK", _sq_correct, _sq_total, "#ff9800",
          "SQUEEZE RISK: Active on days where FII holds a massive short position "
          "AND DII is buying (or retail is extreme short) — creating a structural floor. "
          "When squeeze is active, any positive catalyst forces FII to cover (BUY) = sudden rally. "
          "TRADING RULE: NEVER build fresh short positions when squeeze risk is active. "
          "Squeeze risk + any UP signal = highest conviction trade.")

    _mini(c_cov, "🔄", "SHORT COVERING", _cov_correct, _cov_total, "#69f0ae",
          "SHORT COVERING: Active when FII is ACTIVELY REDUCING a massive short position. "
          "Each covered contract = 1 BUY transaction. FII covering = institutional buying pressure. "
          f"{_cov_correct} correct out of {_cov_total} covering signals. "
          "This is the system's #1 edge signal. "
          "TRADING ACTION: When covering is active, go long immediately. "
          "When covering stops (delta_5d drops below 10K), tighten stop-loss.")


def _backtest_timeline_chart(records) -> go.Figure:
    """
    Combined chart: actual Nifty returns (bars) + signal verdict overlay (markers).
    Green marker = CORRECT signal, red = WRONG. Arrow = verdict direction.
    """
    if not records:
        return go.Figure()

    dates     = [r.next_date  for r in records]
    actuals   = [r.next_day_pct for r in records]
    bar_cols  = ["rgba(0,200,83,0.55)" if v > 0 else "rgba(255,82,82,0.55)" for v in actuals]

    # Signal marker position (slightly above/below bar)
    marker_y  = [a + (0.15 if a >= 0 else -0.15) for a in actuals]
    marker_sym = []
    marker_col = []
    marker_txt = []
    for r in records:
        sym = {"UP": "triangle-up", "DOWN": "triangle-down", "SIDEWAYS": "diamond"}.get(r.verdict, "circle")
        col = "#00c853" if r.outcome == "CORRECT" else "#ff5252"
        txt = (f"<b>{r.signal_date.strftime('%d %b')}</b><br>"
               f"Signal: {r.verdict} ({r.confidence})<br>"
               f"Score: {r.composite_score:+.0f} | {r.market_view}<br>"
               f"Actual: {r.next_day_pct:+.2f}%<br>"
               f"<b>{r.outcome}</b>"
               + (f"<br>⚡ Squeeze Risk" if r.squeeze_risk else "")
               + (f"<br>🔄 Short Covering" if r.short_covering else ""))
        marker_sym.append(sym)
        marker_col.append(col)
        marker_txt.append(txt)

    fig = go.Figure()

    # Actual Nifty returns bars
    fig.add_trace(go.Bar(
        x=dates, y=actuals,
        name="Actual Nifty Return",
        marker_color=bar_cols,
        hovertemplate="%{x|%d %b %Y}<br>Actual: <b>%{y:+.2f}%</b><extra></extra>",
    ))

    # Signal verdict markers (CORRECT = filled, WRONG = outline)
    fig.add_trace(go.Scatter(
        x=dates, y=marker_y,
        mode="markers",
        name="Signal Verdict",
        marker=dict(
            symbol=marker_sym,
            color=marker_col,
            size=12,
            line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
        ),
        text=marker_txt,
        hovertemplate="%{text}<extra></extra>",
    ))

    # ±0.25% threshold lines
    fig.add_hline(y=0.25,  line_dash="dot", line_color="rgba(0,200,83,0.4)", line_width=1)
    fig.add_hline(y=-0.25, line_dash="dot", line_color="rgba(255,82,82,0.4)", line_width=1)
    fig.add_hline(y=0,     line_dash="solid", line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b", tickfont=dict(size=10)),
        yaxis=dict(title="Nifty 50 Next-Day % Change", showgrid=True,
                   gridcolor=GRID_COLOR, ticksuffix="%"),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        height=320, margin=dict(t=30, b=40, l=70, r=40),
        hovermode="closest",
        annotations=[
            dict(x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
                 text="▲ = UP signal  ▼ = DOWN  ◆ = SIDEWAYS  🟢 = CORRECT  🔴 = WRONG",
                 font=dict(size=9, color="rgba(255,255,255,0.4)"), align="left"),
        ],
    )
    return fig


def _backtest_pnl_chart(records) -> go.Figure:
    """Cumulative simulated P&L vs Buy-and-Hold."""
    if not records:
        return go.Figure()

    dates    = [r.next_date  for r in records]
    sim_pnls = [r.pnl_sim    for r in records]
    bnh_pnls = [r.next_day_pct for r in records]

    cum_sim = pd.Series(sim_pnls).cumsum().tolist()
    cum_bnh = pd.Series(bnh_pnls).cumsum().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum_sim,
        name="Signal Strategy P&L",
        mode="lines",
        line=dict(color="#2196f3", width=2.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
        hovertemplate="%{x|%d %b}<br>Cumulative: <b>%{y:+.2f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=cum_bnh,
        name="Buy & Hold Nifty",
        mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dash"),
        hovertemplate="%{x|%d %b}<br>Buy&Hold: <b>%{y:+.2f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b", tickfont=dict(size=10)),
        yaxis=dict(title="Cumulative % Return", showgrid=True,
                   gridcolor=GRID_COLOR, ticksuffix="%"),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        height=280, margin=dict(t=30, b=40, l=70, r=40),
        hovermode="x unified",
    )
    return fig


def _backtest_detail_drill(records, threshold_pct: float) -> None:
    """Date-picker drill-down: show full signal detail for a selected date."""
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 10px'>🔍 DATE DRILL-DOWN — Signal Detail for Any Date</div>",
        unsafe_allow_html=True,
    )
    date_options = sorted([r.signal_date for r in records], reverse=True)
    if not date_options:
        return

    sel = st.selectbox(
        "Select signal date to inspect",
        date_options,
        format_func=lambda d: d.strftime("%d %b %Y"),
        key="bt_drilldown_date",
    )
    rec = next((r for r in records if r.signal_date == sel), None)
    if not rec:
        return

    outcome_c = "#00c853" if rec.outcome == "CORRECT" else "#ff5252"
    dir_icon  = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "↔"}.get(rec.verdict, "?")
    act_icon  = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "↔"}.get(rec.actual_direction, "?")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown(
            f"<div style='border:2px solid {outcome_c}66;border-radius:12px;"
            f"padding:16px 20px;background:{outcome_c}0a'>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.3);"
            f"letter-spacing:2px;margin-bottom:8px'>SIGNAL on {sel.strftime('%d %b %Y')} (evening)</div>"
            f"<div style='display:flex;gap:20px;align-items:center;margin-bottom:10px'>"
            f"<div style='font-size:48px;color:{outcome_c};font-weight:900'>{dir_icon}</div>"
            f"<div>"
            f"<div style='font-size:28px;font-weight:900;color:{outcome_c}'>{rec.verdict}</div>"
            f"<div style='font-size:11px;color:{outcome_c};opacity:0.75'>"
            f"{rec.confidence} CONFIDENCE</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:6px'>"
            f"Composite Score: <b style='color:{outcome_c}'>{rec.composite_score:+.0f}</b> "
            f"({rec.market_view})</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);line-height:1.5'>"
            f"<b>Key Driver:</b> {rec.key_driver}<br>"
            f"<b>Key Risk:</b> {rec.key_risk}</div>"
            + (f"<div style='margin-top:8px;font-size:10px;color:#FFD600'>⚡ SQUEEZE RISK flagged</div>"
               if rec.squeeze_risk else "")
            + (f"<div style='margin-top:4px;font-size:10px;color:#69f0ae'>🔄 SHORT COVERING flagged</div>"
               if rec.short_covering else "")
            + f"</div>",
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            f"<div style='border:2px solid {outcome_c}66;border-radius:12px;"
            f"padding:16px 20px;background:{outcome_c}0a'>"
            f"<div style='font-size:9px;font-weight:700;color:rgba(255,255,255,0.3);"
            f"letter-spacing:2px;margin-bottom:8px'>WHAT ACTUALLY HAPPENED on {rec.next_date.strftime('%d %b %Y')}</div>"
            f"<div style='display:flex;gap:20px;align-items:center;margin-bottom:10px'>"
            f"<div style='font-size:48px;"
            f"color:{'#00c853' if rec.next_day_pct>0 else '#ff5252'};font-weight:900'>{act_icon}</div>"
            f"<div>"
            f"<div style='font-size:36px;font-weight:900;"
            f"color:{'#00c853' if rec.next_day_pct>0 else '#ff5252'}'>"
            f"{rec.next_day_pct:+.2f}%</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.5)'>Nifty 50 actual return</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:13px;font-weight:700;color:{outcome_c};margin-bottom:6px'>"
            f"{rec.outcome}: Signal said {rec.verdict}, market went {rec.actual_direction}</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.5)'>"
            f"Threshold: ±{threshold_pct:.2f}%  |  "
            f"Sim P&L: <span style='color:{'#00c853' if rec.pnl_sim>0 else '#ff5252'}'>"
            f"{rec.pnl_sim:+.2f}%</span></div>"
            + (
                f"<div style='margin-top:10px;padding:8px 10px;border-radius:6px;"
                f"background:rgba(0,200,83,0.1);border:1px solid rgba(0,200,83,0.3)'>"
                f"<div style='font-size:10px;color:#00c853'>💡 Why it was correct:</div>"
                f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:2px'>"
                f"{'Short covering drove the rally despite bearish OI level.' if rec.short_covering else 'Institutional positioning aligned with market move.'}"
                f"</div></div>"
                if rec.outcome == "CORRECT" else
                f"<div style='margin-top:10px;padding:8px 10px;border-radius:6px;"
                f"background:rgba(255,82,82,0.1);border:1px solid rgba(255,82,82,0.3)'>"
                f"<div style='font-size:10px;color:#ff5252'>⚠️ Why it was wrong:</div>"
                f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:2px'>"
                f"{'Squeeze/covering trigger likely overrode the positioning signal.' if rec.squeeze_risk else 'External catalyst overrode institutional positioning.'}"
                f"</div></div>"
            )
            + f"</div>",
            unsafe_allow_html=True,
        )


def _backtest_results_table(records) -> None:
    """Full sortable table of all backtest records."""
    if not records:
        return
    rows = []
    for r in records:
        rows.append({
            "Signal Date": r.signal_date,
            "Verdict":     r.verdict,
            "Conf.":       r.confidence,
            "Score":       r.composite_score,
            "View":        r.market_view,
            "Squeeze":     "⚡" if r.squeeze_risk else "",
            "Covering":    "🔄" if r.short_covering else "",
            "Actual %":    r.next_day_pct,
            "Actual Dir":  r.actual_direction,
            "Outcome":     r.outcome,
            "Sim P&L %":   r.pnl_sim,
        })
    df = pd.DataFrame(rows).sort_values("Signal Date", ascending=False)
    col_cfg = {
        "Signal Date": st.column_config.DateColumn("Date",    format="DD MMM YY"),
        "Score":       st.column_config.NumberColumn("Score",  format="%+.0f"),
        "Actual %":    st.column_config.NumberColumn("Nifty%", format="%+.2f"),
        "Sim P&L %":   st.column_config.NumberColumn("SimP&L", format="%+.2f"),
    }
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _render_backtest(selected_date: date) -> None:
    """Full backtest mode render."""
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:2px;margin:4px 0 14px'>"
        "📊 SIGNAL ACCURACY BACKTEST — Walk-Forward Historical Validation</div>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c_days, c_thresh, c_run = st.columns([2, 2, 1])
    with c_days:
        bt_days = st.selectbox(
            "Backtest Window",
            [30, 45, 60, 90],
            index=2,
            format_func=lambda d: f"Last {d} trading sessions",
            key="bt_days",
        )
    with c_thresh:
        bt_thresh = st.selectbox(
            "Directional Threshold",
            [0.15, 0.25, 0.35, 0.50],
            index=1,
            format_func=lambda t: f"±{t:.2f}% Nifty (current: ±{t*24:.0f}pts)",
            key="bt_thresh",
        )
    with c_run:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Methodology explainer
    with st.expander("📖 Backtest Methodology — How is accuracy measured?", expanded=False):
        st.markdown(f"""
**Walk-Forward (No Look-Ahead Bias):**
For each historical date D, the signal is computed using ONLY data available at close of day D.
No future information is used. This is exactly what you'd have seen on that evening.

**Scoring Rule (threshold = ±{bt_thresh:.2f}%):**
| Signal Said | Nifty Next Day | Outcome |
|-------------|----------------|---------|
| UP | > +{bt_thresh:.2f}% | ✅ CORRECT |
| UP | < +{bt_thresh:.2f}% | ❌ WRONG |
| DOWN | < −{bt_thresh:.2f}% | ✅ CORRECT |
| DOWN | > −{bt_thresh:.2f}% | ❌ WRONG |
| SIDEWAYS | Between ±{bt_thresh:.2f}% | ✅ CORRECT |
| SIDEWAYS | Outside ±{bt_thresh:.2f}% | ❌ WRONG |

**Simulated P&L:**
- UP signal → go long 1 unit Nifty → P&L = actual next-day Nifty %
- DOWN signal → go short 1 unit → P&L = -(actual Nifty %)
- SIDEWAYS → stay flat → P&L = 0%
- This is unleveraged — real trading with leverage would multiply all figures.

**Important context:**
- FII signals are medium-term indicators, not day-trading signals
- Short covering setups may show "WRONG" but were contextually correct (market squeezed up)
- ±{bt_thresh:.2f}% threshold filters out noise — small moves don't count either way
        """)

    if not run_btn:
        st.info(
            "Configure settings above and click **▶ Run Backtest** to evaluate signal accuracy "
            f"over the last {bt_days} trading sessions.\n\n"
            "The engine will run `get_market_intelligence()` for each historical date — "
            "first run takes ~5-15 seconds, then results are cached for 1 hour."
        )
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.spinner(f"Running walk-forward backtest over {bt_days} sessions… (computing signals for each date, no look-ahead)"):
        bt = cached_signal_backtest(selected_date, backtest_days=bt_days, threshold_pct=bt_thresh)

    if bt.total_signals == 0:
        st.warning(
            "No backtest data. Ensure F&O participant data and Nifty 50 index data are loaded.\n\n"
            "Run: `python -m src.cli backfill 90` and `python -m src.cli backfill-fao 90`"
        )
        return

    st.caption(
        f"Evaluated **{bt.total_signals}** signals ending {selected_date.strftime('%d %b %Y')} "
        f"| Threshold: ±{bt_thresh:.2f}% | "
        f"Longest correct streak: {bt.longest_correct_streak} | Longest wrong streak: {bt.longest_wrong_streak}"
    )

    # ── Summary cards ─────────────────────────────────────────────────────────
    _backtest_summary_cards(bt)
    st.markdown("")

    # ── Trading decision panel ────────────────────────────────────────────────
    dir_acc = bt.hit_rate_directional
    sharpe  = bt.sharpe_sim
    cov_acc = bt.covering_accuracy
    _cov_n  = sum(1 for r in bt.records if r.short_covering and r.verdict in ("UP","DOWN"))

    if dir_acc >= 0.60 and sharpe >= 1.0:
        _decision_color = "#00c853"; _decision_icon = "✅"; _decision_label = "TRADEABLE SIGNAL"
        _decision_text = (
            f"Directional accuracy {dir_acc:.0%} + Sharpe {sharpe:.2f} — "
            "system has a statistically meaningful edge. You can trade UP signals with covering+squeeze flags."
        )
    elif dir_acc >= 0.50 and sharpe >= 0:
        _decision_color = "#ffca28"; _decision_icon = "⚠️"; _decision_label = "TRADE WITH CAUTION"
        _decision_text = (
            f"Directional accuracy {dir_acc:.0%} + Sharpe {sharpe:.2f} — "
            "marginal edge. Only take HIGH CONFIDENCE signals. Use small position size (50% of normal)."
        )
    else:
        _decision_color = "#ff5252"; _decision_icon = "🚫"; _decision_label = "DO NOT TRADE"
        _decision_text = (
            f"Directional accuracy {dir_acc:.0%} or Sharpe {sharpe:.2f} — "
            "no edge in this window. The signal engine is in a losing phase. Stay flat."
        )

    st.markdown(
        f"<div style='border:1px solid {_decision_color}66;border-radius:10px;"
        f"padding:12px 16px;background:{_decision_color}11;margin:8px 0'>"
        f"<span style='font-size:10px;font-weight:700;color:{_decision_color};"
        f"letter-spacing:1.5px'>{_decision_icon} TRADING DECISION: {_decision_label}</span>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.7);margin-top:6px'>{_decision_text}</div>"
        f"<div style='font-size:11px;color:rgba(255,255,255,0.45);margin-top:8px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px'>"
        f"📌 <b>Rule 1 — Only trade UP+covering+squeeze:</b> "
        f"Short covering accuracy = {cov_acc:.0%} on {_cov_n} signals. These are highest conviction. "
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"📌 <b>Rule 2 — Never short on score alone:</b> "
        f"Bearish score = FII massive short = squeeze fuel. Trading against squeeze = dangerous. "
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"📌 <b>Rule 3 — Ignore SIDEWAYS:</b> "
        f"SIDEWAYS = no trade. Do not force a trade when system is unsure."
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Signal type breakdown ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:12px 0 8px'>"
        "📊 ACCURACY BY SIGNAL TYPE</div>",
        unsafe_allow_html=True,
    )
    _backtest_breakdown_row(bt)

    # Notable calls
    if bt.best_call or bt.worst_call:
        c_best, c_worst = st.columns(2)
        if bt.best_call:
            b = bt.best_call
            c_best.success(
                f"🏆 **Best call:** {b.signal_date.strftime('%d %b')} → "
                f"Said {b.verdict}, Nifty moved {b.next_day_pct:+.2f}% → "
                f"P&L: +{b.pnl_sim:.2f}%"
            )
        if bt.worst_call:
            w = bt.worst_call
            c_worst.error(
                f"💀 **Worst miss:** {w.signal_date.strftime('%d %b')} → "
                f"Said {w.verdict}, Nifty moved {w.next_day_pct:+.2f}% → "
                f"P&L: {w.pnl_sim:.2f}%"
            )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 8px'>"
        "📈 SIGNAL VERDICTS vs ACTUAL NIFTY RETURNS</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Bars = actual Nifty return day after signal. "
        "Markers = signal verdict (▲UP ▼DOWN ◆SIDEWAYS). "
        "Green marker = CORRECT, Red = WRONG. Dotted lines = ±threshold."
    )
    st.plotly_chart(
        _backtest_timeline_chart(bt.records),
        use_container_width=True,
        key="bt_timeline",
    )

    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 8px'>"
        "💰 SIMULATED CUMULATIVE P&L — Strategy vs Buy &amp; Hold</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Blue = following signals (long on UP, short on DOWN, flat on SIDEWAYS). "
        "Grey dash = buying and holding Nifty throughout. 1-unit unleveraged."
    )
    st.plotly_chart(
        _backtest_pnl_chart(bt.records),
        use_container_width=True,
        key="bt_pnl",
    )

    # ── Date drill-down ───────────────────────────────────────────────────────
    st.divider()
    _backtest_detail_drill(bt.records, bt_thresh)

    # ── Full table ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:4px 0 8px'>"
        "📋 COMPLETE SIGNAL HISTORY TABLE</div>",
        unsafe_allow_html=True,
    )
    _backtest_results_table(bt.records)


# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    st.subheader("🎯 Big Players — F&O Institutional Intelligence")

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "View Mode",
        ["🧠 Live Intelligence", "📊 Backtest History"],
        horizontal=True,
        key="fao_mode",
        help="Live: today's signal engine. Backtest: walk-forward accuracy on historical signals.",
    )
    st.markdown("---")

    if mode == "📊 Backtest History":
        _render_backtest(selected_date)
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, _ = st.columns([2, 2, 4])
    with c1:
        data_type = st.radio(
            "Data Type", ["OI", "Vol"], horizontal=True,
            help="OI = Open Interest (positions held). Vol = Volume (daily traded).",
        )
    with c2:
        lookback_opt = st.selectbox(
            "Lookback", ["1 Month", "3 Months", "6 Months", "1 Year"], index=3,
        )
    lookback_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    lookback_days = lookback_map[lookback_opt]
    start_date = selected_date - timedelta(days=lookback_days)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading F&O participant data…"):
        latest      = cached_fao_latest(selected_date, data_type)
        cum         = cached_fao_cumulative(selected_date, start_date, data_type)
        daily       = cached_fao_daily(selected_date, lookback_days + 10, data_type)
        fii_stats   = cached_fii_stats_latest(selected_date)
        fii_stats_h = cached_fii_stats_history(selected_date, lookback_days)

    if latest.empty:
        st.warning(
            "No F&O participant data found. "
            "Run the backfill to load history:\n\n"
            "```\npython -m src.cli backfill-fao 365\n```"
        )
        return

    data_date = latest["trade_date"].iloc[0]
    st.caption(
        f"**{data_type}** data — as of **{data_date.strftime('%d %b %Y')}** "
        f"| Lookback: {lookback_opt} (from {start_date.strftime('%d %b %Y')})"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: TOMORROW'S VERDICT (Market Intelligence Engine)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
        "letter-spacing:1.5px;margin:8px 0 10px'>"
        "🧠 MARKET INTELLIGENCE ENGINE — 9-Signal Composite Analysis</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running quant signal engine…"):
        mi = cached_market_intelligence(selected_date)

    if mi.market_view == "No Data":
        st.info("Run F&O backfill first to enable market intelligence signals.")
    else:
        # Hero verdict panel
        _tomorrow_verdict_hero(mi)
        st.markdown("")

        # Alerts
        for alert in mi.alerts:
            st.warning(f"🚨 {alert}")

        # ── FII Today Activity Panel (3-layer: OI + Volume + Rupee Flow) ──────
        _fii_today_panel(mi)

        # Signal Intelligence Engine
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
            "letter-spacing:1.5px;margin:14px 0 8px'>"
            "⚡ SIGNAL BREAKDOWN — sorted by impact magnitude</div>",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        for i, sig in enumerate(mi.signals):
            col = col_a if i % 2 == 0 else col_b
            col.markdown(_signal_badge(sig), unsafe_allow_html=True)

        if mi.fii_flow_available:
            st.caption("✓ FII Money Flow signal included (fii_derivatives_stats data available).")
        else:
            st.caption(
                "FII Money Flow signal not available — "
                "run `python -m src.cli backfill-fii-stats 365` to enable ₹Cr signals."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: INSTITUTIONAL POSITIONING SCORECARD
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    _positioning_scorecard(latest)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: FULL DERIVATIVE MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("")
    _derivative_breakdown(latest)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: FII INDEX FOCUS + DERIVATIVES STATISTICS
    # ══════════════════════════════════════════════════════════════════════════
    if not fii_stats.empty:
        st.divider()
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.35);"
            "letter-spacing:1.5px;margin:4px 0 10px'>"
            "💰 FII DERIVATIVES STATISTICS — Rupee Flow (₹ Crore)</div>",
            unsafe_allow_html=True,
        )
        # Index focus cards — uses participant OI call/put split for Nifty verdict
        _fii_index_focus(fii_stats, fii_stats_h, participant_latest=latest)

        st.markdown("")
        # Raw table + trend chart side-by-side (in expander to save space)
        with st.expander("📋 Full Per-Index Breakdown Table", expanded=False):
            c_tbl, c_chart = st.columns([1, 1])
            with c_tbl:
                _fii_stats_table(fii_stats)
            with c_chart:
                if not fii_stats_h.empty:
                    st.caption("Net FII buy/sell flow by category")
                    st.plotly_chart(
                        _fii_flow_chart(fii_stats_h),
                        use_container_width=True,
                        key="fii_flow_summary",
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # TABS: Charts | Position Analysis | FII Money Flow History
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    tab_charts, tab_pos, tab_fii = st.tabs(
        ["📈 Charts & PCR", "📋 Position Tables", "💰 FII Flow History"]
    )

    with tab_charts:
        st.markdown("#### Put-Call Ratio (Contrarian Indicator)")
        st.caption(
            "PCR = Total Put OI ÷ Total Call OI. "
            "Above 1.3 = fear peak → contrarian buy. Below 0.7 = complacency → contrarian sell."
        )
        _pcr_metric(latest)
        st.plotly_chart(_pcr_chart(cum), use_container_width=True)

        st.divider()
        st.markdown("#### Cumulative Index Futures Net Positions")
        st.caption(
            "Running sum of (Long − Short) since start date. "
            "Rising above zero = sustained accumulation. Falling below zero = distribution phase."
        )
        st.plotly_chart(_cumulative_chart(cum), use_container_width=True)

        st.markdown("#### FII & DII Long-to-Short % Trend")
        st.caption("Above 50% = net long; below 50% = net short. FII drives direction; DII provides floor.")
        st.plotly_chart(_ls_trend_chart(cum), use_container_width=True)

        st.markdown("#### Daily Net — Index Futures (last 60 days)")
        st.caption("Bar = daily net; dotted line = cumulative. Green bars = net long days. Red = net short.")
        c_fii, c_dii = st.columns(2)
        c_client, c_pro = st.columns(2)
        with c_fii:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{_COLORS['FII']}'>"
                f"FII — Market Driver</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(_daily_net_chart(cum, "FII"), use_container_width=True)
        with c_dii:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{_COLORS['DII']}'>"
                f"DII — Floor Support</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(_daily_net_chart(cum, "DII"), use_container_width=True)
        with c_client:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{_COLORS['Client']}'>"
                f"Client (Retail) — Contrarian Signal (read in reverse)</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(_daily_net_chart(cum, "Client"), use_container_width=True)
        with c_pro:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:{_COLORS['Pro']}'>"
                f"Pro (Prop Desks) — Delta Hedger</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(_daily_net_chart(cum, "Pro"), use_container_width=True)

        st.divider()
        st.markdown("#### Cumulative Index Options — FII & DII Call vs Put Net")
        st.caption(
            "Solid = Call Net (bullish bets). Dashed = Put Net (bearish/hedge bets). "
            "FII Call rising + Put falling → double bullish confirmation."
        )
        st.plotly_chart(_cumulative_options_chart(cum), use_container_width=True)

    with tab_pos:
        st.markdown("#### Daily Net Positions — Index Futures")
        st.caption("Net = Long − Short contracts per day. Positive = net long that day.")
        _daily_table(daily)

        st.markdown("#### Daily Net Positions — Index Options")
        st.caption(
            "Call Net = Call Long − Call Short. Put Net = Put Long − Put Short. "
            "Options Delta = Call Net − Put Net."
        )
        _options_daily_table(daily)

        st.markdown("#### Cumulative Net Positions — Index Futures")
        st.caption(
            f"Running total from {start_date.strftime('%d %b %Y')}. "
            "L/S% = Long as % of (Long + Short). Below 50% = net short."
        )
        _cumulative_table(cum)

        st.markdown("#### Cumulative Net Positions — Index Options")
        _options_cumulative_table(cum)

    with tab_fii:
        st.markdown("#### FII Derivatives Statistics — Daily Buy/Sell Activity")
        st.caption(
            "FII buy/sell contracts and rupee value (₹ Crore) by contract type. "
            "Net Flow (Cr) = Buy − Sell. Positive = net buyer (bullish), Negative = net seller."
        )
        _fii_stats_table(fii_stats)

        if not fii_stats_h.empty:
            st.markdown("#### Net FII Flow Trend — Index F&O (₹ Crore)")
            st.caption(
                "Daily net rupee flow: Nifty Futures, BankNifty Futures, Index Futures aggregate. "
                "Large positive bars = institutional money entering. Confirms or contradicts OI signal."
            )
            st.plotly_chart(_fii_flow_chart(fii_stats_h), use_container_width=True, key="fii_flow_tab")

            with st.expander("Full FII Stats History Table"):
                hist_display = fii_stats_h.copy()
                hist_display["net_value_cr"]  = hist_display["buy_value_cr"] - hist_display["sell_value_cr"]
                hist_display["net_contracts"] = hist_display["buy_contracts"] - hist_display["sell_contracts"]
                col_cfg = {
                    "trade_date":    st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                    "category":      st.column_config.TextColumn("Category"),
                    "net_contracts": st.column_config.NumberColumn("Net Contracts", format="%+,d"),
                    "net_value_cr":  st.column_config.NumberColumn("Net Flow (Cr)",  format="%+.2f"),
                    "buy_value_cr":  st.column_config.NumberColumn("Buy (Cr)",        format="%.2f"),
                    "sell_value_cr": st.column_config.NumberColumn("Sell (Cr)",       format="%.2f"),
                    "oi_contracts":  st.column_config.NumberColumn("OI Contracts",    format="%,d"),
                    "buy_contracts": None, "sell_contracts": None,
                }
                st.dataframe(
                    hist_display.sort_values(["trade_date", "category"], ascending=[False, True]),
                    hide_index=True, use_container_width=True, column_config=col_cfg,
                    column_order=["trade_date","category","net_contracts",
                                  "buy_value_cr","sell_value_cr","net_value_cr","oi_contracts"],
                )
        else:
            st.info("Run `python -m src.cli backfill-fii-stats 365` to load FII statistics history.")

    # ── How to Read (bottom) ──────────────────────────────────────────────────
    with st.expander("📖 How to Read This Page", expanded=False):
        st.markdown("""
**⚡ CRITICAL: Institutional Bias ≠ Tomorrow's Trading Direction**

This is the most important concept for real trading:

| Situation | What it means | What market does |
|-----------|--------------|-----------------|
| FII net SHORT -200K (static, not changing) | Bearish BIAS (they are positioned bearish) | **Range-bound or squeeze** — DII floor prevents fall; any catalyst triggers covering rally |
| FII net SHORT -200K **and adding more** (delta_5d negative, z < -1) | **Fresh Short Build** → high conviction BEARISH | Market likely falls — they are actively pressing the short |
| FII net SHORT -200K **and reducing** (delta_5d positive, z > 1) | **Short Covering in Progress** → net BUYING | Market rallies — each covered short = 1 buy transaction. The massive short IS the fuel. |
| FII net SHORT + DII net LONG (divergence) | **Squeeze Setup** — floor created | Consolidation or rally; short thesis cannot play out while DII buys |

**Rule: Watch the DIRECTION of OI change, not just the level. A -200K FII short that is decreasing is BULLISH for tomorrow. A -200K FII short that is increasing is BEARISH.**

---

**The 4 Participants — Who Does What**
| Participant | Who they are | Role in signals |
|-------------|-------------|-----------------|
| **🔵 FII** | Foreign Institutional Investors (FPIs, foreign hedge funds, GDR) | **#1 Signal** — sets market direction. Net long = market goes up. Net short = market falls. |
| **🟢 DII** | Domestic Institutions (LIC, Mutual Funds, Insurance, pension funds) | **Floor support** — buy on dips using SIP inflows. DII net short in futures usually = hedging equity portfolio (not bearish). |
| **🟣 Pro** | Proprietary desks (brokers trading own capital) | **Delta hedger** — typically short Index Futures to hedge long options exposure. Don't panic if Pro is net short. |
| **🟠 Client** | Retail traders & HNI | **Contrarian indicator** — retail is wrong at extremes. Extreme Client short = buy signal. Extreme Client long = sell signal. |

---

**Futures vs Options**
| Instrument | What it tells you |
|---|---|
| **Index Futures** | Pure directional bet. FII net long = bullish; net short = bearish. No ambiguity. |
| **Index Options — Call Net** | +ve = buying calls → bullish directional bet. −ve = selling calls → expecting no rally. |
| **Index Options — Put Net** | +ve = buying puts → hedging or bearish bet. −ve = selling puts → bullish (expectation market won't fall). |
| **Options Delta** | Call Net − Put Net. Positive = overall bullish options stance. |
| **Stock Futures** | Single-stock conviction plays. FII accumulating specific names → sector rotation signal. |

---

**PCR (Put-Call Ratio) — Contrarian Indicator**
- **PCR > 1.3**: Everyone has already bought puts (hedged downside). Markets rarely fall when participants are already protected → **contrarian BUY signal**.
- **PCR 0.7–1.3**: Normal balanced positioning → no extreme signal.
- **PCR < 0.7**: Excessive call buying → complacency at peaks → **contrarian SELL signal**.

---

**OI vs Volume**
- **OI (Open Interest)**: Contracts *outstanding* at day's end — actual positions being held. **Better signal for conviction.**
- **Volume**: Contracts traded during the day — activity, but positions may have reversed. Good for detecting sudden moves.

---

**Key signals to watch**
- 🟢 **FII Net Futures +ve & rising** → strong bullish trend; institutions accumulating
- 🔴 **FII Net Futures −ve & falling** → institutional distribution; bearish
- 🟡 **FII buying puts aggressively** → smart-money hedge; possible reversal ahead
- 📊 **PCR > 1.3** → contrarian buy (put crowd near capitulation → bounce likely)
- 📊 **PCR < 0.5** → contrarian sell (complacent bulls → correction risk)
- ⚡ **Client Net extreme short** while **FII Net long** → high-conviction bullish setup
- ⚖️ **FII short + DII long** → institutional standoff; range-bound, no trend

---

**The 9 Signal Engine**
The Composite Score (range −18 to +18) combines:
1. OI-Price Context (Fresh Long / Short Cover / Long Unwind / Fresh Short)
2. FII Futures 5D Trend (z-score of position change)
3. Consecutive Day Pattern (3+ consecutive days in one direction)
4. FII-DII Alignment (both long = strongest; both short = most bearish)
5. FII Volume Spike (unusual activity + OI direction = accumulation/distribution)
6. PCR Trend (contrarian at extremes)
7. FII Options Stance Shift (delta change over 5 days)
8. Retail Contrarian (extreme Client positioning)
9. FII Money Flow (₹Cr net buy/sell from fii_derivatives_stats)

Score ≥ +7 → HIGH confidence UP | Score ≤ −7 → HIGH confidence DOWN | −3 to +3 → SIDEWAYS
        """)
