"""
Index Prediction Dashboard — tomorrow's directional forecast for all major F&O indices.

Layout per index:
  • Prediction card — direction badge, score, PCR, carry, FII net, VIX
  • Tab 1 Verdict & Signals  — verdict panel + signal breakdown by tier
  • Tab 2 Key Levels         — put wall / max pain / call wall + carry detail
  • Tab 3 Today vs Yesterday — comparison table
  • Tab 4 OI Structure       — call/put OI chart (options chain heatmap)
  • Tab 5 Market Context     — VIX trend, sector breadth, FII positioning table
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from src.analytics.index_prediction import IndexPrediction, IndexSignal, MarketContext
from src.dashboard.cache.queries import cached_index_predictions
from src.dashboard.constants import GRID_COLOR, PAPER_BG, PLOT_BG

# ── Colour palette ────────────────────────────────────────────────────────────
_DIR_EMOJI  = {"UP": "↑", "DOWN": "↓", "SIDEWAYS": "↔"}
_CONF_COLOR = {"HIGH": "#00C853", "MEDIUM": "#FFD600", "LOW": "#78909C"}
_CAT_COLOR  = {
    "Price Action":   "#9c27b0",
    "Futures OI":     "#2196f3",
    "Options OI":     "#ff9800",
    "Carry":          "#4caf50",
    "Institutional":  "#f44336",
    "Market Context": "#00bcd4",
    "Sector":         "#8bc34a",
}
_POS  = "#69f0ae"
_NEG  = "#ff5252"
_NEU  = "#78909C"


_TOOLTIP_CSS = """<style>
.nse-tip{position:relative;display:inline-flex;align-items:center;gap:5px}
.nse-tip .ti{display:inline-flex;align-items:center;justify-content:center;
  width:15px;height:15px;border-radius:50%;background:#2a2a2a;border:1px solid #555;
  color:#777;font-size:9px;cursor:help;flex-shrink:0;font-weight:700}
.nse-tip .tb{visibility:hidden;opacity:0;position:absolute;z-index:9999;
  left:0;top:115%;background:#1a1a2e;color:#ccc;border:1px solid #444;
  border-radius:6px;padding:10px 14px;font-size:12px;width:320px;
  line-height:1.6;transition:opacity 0.15s;pointer-events:none;font-weight:400}
.nse-tip:hover .tb{visibility:visible;opacity:1}
</style>"""


def _h(text: str, tip: str, level: int = 4) -> None:
    """Render a heading with a hover ? tooltip explaining the section."""
    safe = tip.replace('"', "&quot;").replace("'", "&#39;")
    tag  = f"h{level}"
    st.markdown(
        f'{_TOOLTIP_CSS}<{tag} style="margin-bottom:2px" class="nse-tip">'
        f'{text}&nbsp;<span class="ti">?</span>'
        f'<span class="tb">{safe}</span>'
        f'</{tag}>',
        unsafe_allow_html=True,
    )


def _pct_color(v: Optional[float]) -> str:
    return _POS if (v or 0) >= 0 else _NEG


def _fmt(v: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    return "—" if v is None else f"{v:,.{decimals}f}{suffix}"


def _delta_html(today: Optional[float], yesterday: Optional[float],
                suffix: str = "", fmt: str = ".2f") -> str:
    if today is None:
        return "<span style='color:#888'>—</span>"
    val = f"{today:{fmt}}{suffix}"
    if yesterday is None:
        return f"<span style='color:#ccc'>{val}</span>"
    delta = today - yesterday
    color = _POS if delta > 0 else (_NEG if delta < 0 else _NEU)
    sign  = "+" if delta >= 0 else ""
    return (
        f"<span style='color:#ccc'>{val}</span> "
        f"<span style='color:{color};font-size:0.75em'>({sign}{delta:{fmt}}{suffix})</span>"
    )


# ── Prediction card ───────────────────────────────────────────────────────────

_MEMORY_COLOR = {"Trending": "#4CAF50", "Random Walk": "#9E9E9E", "Mean-Reverting": "#FF9800"}
_HMM_COLOR    = {"Bull": "#4CAF50", "Sideways": "#FFD600", "Bear": "#EF5350", "Unknown": "#9E9E9E"}
_ENT_COLOR    = {"Ordered": "#4CAF50", "Moderate": "#9E9E9E", "Chaotic": "#EF5350", "Unknown": "#9E9E9E"}


def _render_card(pred: IndexPrediction) -> None:
    dir_emoji  = _DIR_EMOJI.get(pred.direction, "↔")
    conf_color = _CONF_COLOR.get(pred.confidence, "#78909C")
    dir_color  = pred.direction_color
    ctx        = pred.market_context

    # FII net display
    fii_net   = ctx.fii_fut_idx_net if ctx else 0
    fii_str   = f"{fii_net:+,}" if ctx and ctx.fao_date else "—"
    fii_emoji = "🐂" if fii_net > 80_000 else ("🐻" if fii_net < -80_000 else "⚪")

    # VIX display
    vix_str   = f"{ctx.vix_close:.1f}" if ctx and ctx.vix_close else "—"
    vix_trend = ""
    if ctx and ctx.vix_pct_chg is not None:
        vix_trend = f" ({'↓' if ctx.vix_pct_chg < 0 else '↑'}{abs(ctx.vix_pct_chg):.1f}%)"
    vix_color = (_POS if (ctx.vix_pct_chg or 0) < 0 else _NEG) if ctx and ctx.vix_pct_chg else _NEU

    # FII position change display (covering vs adding)
    fii_chg     = ctx.fii_net_change_1d if ctx else 0
    fii_chg_str = ""
    if ctx and ctx.fii_prev_fao_date and fii_chg != 0:
        fii_chg_color = _POS if fii_chg > 0 else _NEG
        fii_chg_label = "COV" if fii_chg > 0 else "ADD"
        fii_chg_str   = (
            f"<span style='color:{fii_chg_color};font-size:0.70em'>"
            f" ({fii_chg_label} {abs(fii_chg):,})</span>"
        )

    # ── Statistical Regime badges ─────────────────────────────────────────────
    regime_html = ""
    r = pred.regime
    if r and not r.error:
        h_col  = _MEMORY_COLOR.get(r.memory_label, "#9E9E9E")
        h_abbr = {"Trending": "TREND", "Random Walk": "RND", "Mean-Reverting": "M-REV"}.get(r.memory_label, "?")

        hmm_col   = _HMM_COLOR.get(r.hmm_state, "#9E9E9E")
        hmm_label = r.hmm_state if r.hmm_state != "Unknown" else "—"

        ent_col   = _ENT_COLOR.get(r.entropy_label, "#9E9E9E")
        ent_abbr  = {"Ordered": "ORD", "Moderate": "MOD", "Chaotic": "CHAOS"}.get(r.entropy_label, "?")

        def _badge(label: str, value: str, color: str) -> str:
            return (
                f"<span style='display:inline-flex;flex-direction:column;align-items:center;"
                f"background:rgba(255,255,255,0.04);border:1px solid {color}44;"
                f"border-radius:5px;padding:2px 7px;margin-right:4px'>"
                f"<span style='font-size:0.60em;color:#888;letter-spacing:0.5px'>{label}</span>"
                f"<span style='font-size:0.78em;font-weight:700;color:{color}'>{value}</span>"
                f"</span>"
            )

        regime_html = (
            f"<div style='margin-top:7px;border-top:1px solid #2a2a2a;padding-top:6px'>"
            f"<div style='font-size:0.65em;color:#555;margin-bottom:3px;letter-spacing:0.5px'>STATISTICAL REGIME</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:2px'>"
            f"{_badge('HURST', f'{r.memory_avg:.3f} {h_abbr}', h_col)}"
            f"{_badge('HMM', hmm_label, hmm_col)}"
            f"{_badge('ENTROPY', ent_abbr, ent_col)}"
            f"</div>"
            f"</div>"
        )

    st.markdown(
        f"""
        <div style="
            background:#1e1e1e; border:1px solid {dir_color};
            border-radius:10px; padding:16px 18px; height:100%;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:1.05em;font-weight:700;color:#fff">{pred.display_name}</span>
                <span style="
                    background:{dir_color}22;color:{dir_color};
                    border:1px solid {dir_color};border-radius:6px;
                    padding:2px 10px;font-size:0.9em;font-weight:700;
                ">{dir_emoji} {pred.direction}</span>
            </div>
            <div style="font-size:1.4em;font-weight:700;color:#fff;margin-bottom:2px">
                {_fmt(pred.spot_close, 0) if pred.spot_close else '—'}
            </div>
            <div style="font-size:0.9em;color:{'#69f0ae' if (pred.day_change_pct or 0)>=0 else '#ff5252'};margin-bottom:10px">
                {'▲' if (pred.day_change_pct or 0)>=0 else '▼'} {abs(pred.day_change_pct or 0):.2f}%
                &nbsp;|&nbsp;
                <span style="color:{conf_color};font-size:0.85em">{pred.confidence} CONF</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.8em;color:#aaa">
                <div>Score <b style="color:#fff">{pred.composite_score:+.1f}</b></div>
                <div>PCR <b style="color:#fff">{_fmt(pred.pcr) if pred.pcr else '—'}</b></div>
                <div>Carry <b style="color:#fff">{_fmt(pred.carry_pct_ann, 1, '% ann') if pred.carry_pct_ann is not None else '—'}</b></div>
                <div>DTE <b style="color:#fff">{pred.days_to_expiry}d</b></div>
                <div>FII <b style="color:#fff">{fii_emoji} {fii_str}</b>{fii_chg_str}</div>
                <div>VIX <b style="color:{vix_color}">{vix_str}{vix_trend}</b></div>
            </div>
            <div style="margin-top:8px;font-size:0.75em;color:#888;border-top:1px solid #333;padding-top:6px">
                S: <b style="color:#69f0ae">{_fmt(pred.levels.top_put_strike, 0) if pred.levels.top_put_strike else '—'}</b>
                &nbsp;|&nbsp;
                MP: <b style="color:#FFD600">{_fmt(pred.levels.max_pain, 0) if pred.levels.max_pain else '—'}</b>
                &nbsp;|&nbsp;
                R: <b style="color:#ff5252">{_fmt(pred.levels.top_call_strike, 0) if pred.levels.top_call_strike else '—'}</b>
            </div>
            {regime_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Today vs Yesterday ────────────────────────────────────────────────────────

def _render_comparison(pred: IndexPrediction) -> None:
    t = pred.today
    y = pred.yesterday
    _h("Today vs Yesterday", "Side-by-side comparison of today's key metrics vs yesterday. Use this to spot momentum shifts — e.g. PCR rising, FII net getting more short, or carry turning negative are early warning signs.")
    if t is None:
        st.caption("No snapshot data available.")
        return

    rows = [
        ("Spot Close",    t.spot_close,   y.spot_close if y else None,   ".0f",  ""),
        ("Day Change %",  t.pct_chg,      y.pct_chg if y else None,      ".2f",  "%"),
        ("Futures Settle",t.fut_settle,   y.fut_settle if y else None,   ".0f",  ""),
        ("Carry % ann",   t.carry_pct_ann,y.carry_pct_ann if y else None,".2f",  "%"),
        ("Carry Pts",     t.carry_pts,    y.carry_pts if y else None,    ".0f",  " pts"),
        ("Futures OI",    t.fut_oi,       y.fut_oi if y else None,       ",.0f", ""),
        ("Call OI (Near)",t.call_oi,      y.call_oi if y else None,      ",.0f", ""),
        ("Put OI (Near)", t.put_oi,       y.put_oi if y else None,       ",.0f", ""),
        ("PCR (Near)",    t.pcr,          y.pcr if y else None,          ".2f",  ""),
        ("Total Volume",  t.total_volume, y.total_volume if y else None, ",.0f", ""),
    ]

    td = t.trade_date.strftime("%d %b")
    yd = y.trade_date.strftime("%d %b") if y else "Yesterday"
    html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.85em">
        <thead>
            <tr style="border-bottom:1px solid #333;color:#888">
                <th style="text-align:left;padding:4px 8px">Metric</th>
                <th style="text-align:right;padding:4px 8px">{td}</th>
                <th style="text-align:right;padding:4px 8px">{yd}</th>
                <th style="text-align:right;padding:4px 8px">Change</th>
            </tr>
        </thead><tbody>
    """
    for label, tv, yv, fmt, suf in rows:
        ts = f"{tv:{fmt}}{suf}" if tv is not None else "—"
        ys = f"{yv:{fmt}}{suf}" if yv is not None else "—"
        if tv is not None and yv is not None:
            delta = tv - yv
            sign  = "+" if delta >= 0 else ""
            dc    = _POS if delta > 0 else (_NEG if delta < 0 else _NEU)
            ds    = f"<span style='color:{dc}'>{sign}{delta:{fmt}}{suf}</span>"
        else:
            ds = "—"
        html += f"""
        <tr style="border-bottom:1px solid #222">
            <td style="padding:5px 8px;color:#aaa">{label}</td>
            <td style="padding:5px 8px;text-align:right;color:#fff">{ts}</td>
            <td style="padding:5px 8px;text-align:right;color:#666">{ys}</td>
            <td style="padding:5px 8px;text-align:right">{ds}</td>
        </tr>"""
    html += "</tbody></table>"
    st.html(html.strip())


# ── OI Structure chart ────────────────────────────────────────────────────────

def _render_oi_chart(pred: IndexPrediction) -> None:
    expiry_label = "Weekly" if (pred.fno_symbol == "NIFTY" and pred.monthly_expiry) else "Near-Month"
    _h(f"OI Structure — {expiry_label} Options Chain", "Call OI (red, right) = resistance levels where sellers have written calls. Put OI (green, left) = support levels where sellers have written puts. The strike with highest combined OI is Max Pain — markets often gravitate here into expiry. Wide call wall above spot = ceiling; wide put wall below = floor.")
    if pred.near_expiry is None or pred.spot_close is None:
        st.caption("Options chain data not available.")
        return

    from src.dashboard.cache.queries import cached_options_chain, cached_index_options_chain
    chain_df = cached_options_chain(pred.as_of_date, pred.fno_symbol, pred.near_expiry, "OPTIDX")
    if chain_df is None or chain_df.empty:
        chain_df = cached_index_options_chain(pred.as_of_date, pred.fno_symbol, pred.near_expiry, n_strikes=20)
    if chain_df is None or chain_df.empty:
        st.caption("Options chain not available for this expiry.")
        return

    spot = pred.spot_close

    # Both chain functions return pivoted DataFrames: one row per strike,
    # columns ce_oi / pe_oi (not raw option_type rows).
    # Use dict lookup to avoid float-precision issues with .loc[]
    ce_oi_map = dict(zip(chain_df["strike_price"], chain_df["ce_oi"]))
    pe_oi_map = dict(zip(chain_df["strike_price"], chain_df["pe_oi"]))
    all_s     = sorted(float(s) for s in ce_oi_map)
    band      = spot * 0.08
    strikes   = [s for s in all_s if abs(s - spot) <= band] or all_s

    call_oi = [int(ce_oi_map.get(s, 0)) for s in strikes]
    put_oi  = [int(pe_oi_map.get(s, 0)) for s in strikes]

    # Use NUMERIC y-axis (not categorical strings) so add_hline works in Plotly 6.x.
    # Tick labels are set explicitly via tickvals/ticktext.
    fig = go.Figure()
    fig.add_trace(go.Bar(y=strikes, x=call_oi,
                         name="Call OI", orientation="h", marker_color="#ef5350", opacity=0.85))
    fig.add_trace(go.Bar(y=strikes, x=[-v for v in put_oi],
                         name="Put OI",  orientation="h", marker_color="#26a69a", opacity=0.85))

    if strikes:
        cls = min(strikes, key=lambda s: abs(s - spot))
        fig.add_hline(y=float(cls), line_color="#FFD600", line_width=2, line_dash="dash",
                      annotation=dict(text=f"Spot {spot:.0f}", font_color="#FFD600"))
    if pred.levels.max_pain and strikes:
        mp = min(strikes, key=lambda s: abs(s - pred.levels.max_pain))
        fig.add_hline(y=float(mp), line_color="#FF9800", line_width=2, line_dash="dot",
                      annotation=dict(text=f"MaxPain {pred.levels.max_pain:.0f}", font_color="#FF9800"))

    fig.update_layout(
        barmode="overlay", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#ccc", size=11),
        height=max(300, len(strikes) * 22),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Put OI  <--  |  -->  Call OI", gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="#555", zerolinewidth=2, tickformat=","),
        yaxis=dict(title="Strike", gridcolor=GRID_COLOR,
                   tickvals=strikes, ticktext=[str(int(s)) for s in strikes]),
        legend=dict(orientation="h", x=0, y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Nifty: monthly expiry OI chain ────────────────────────────────────────
    if pred.fno_symbol == "NIFTY" and pred.monthly_expiry is not None and pred.spot_close is not None:
        spot_close = pred.spot_close
        m_dte = (pred.monthly_expiry - pred.as_of_date).days
        st.markdown(
            f"<div style='margin-top:16px;font-size:0.75em;color:#888;font-weight:600;"
            f"letter-spacing:0.05em'>── MONTHLY EXPIRY OI  "
            f"({pred.monthly_expiry.strftime('%d %b')} · {m_dte}d) ──</div>",
            unsafe_allow_html=True,
        )
        from src.dashboard.cache.queries import cached_options_chain, cached_index_options_chain
        m_chain = cached_options_chain(pred.as_of_date, pred.fno_symbol, pred.monthly_expiry, "OPTIDX")
        if m_chain is None or m_chain.empty:
            m_chain = cached_index_options_chain(pred.as_of_date, pred.fno_symbol, pred.monthly_expiry, n_strikes=20)
        if m_chain is not None and not m_chain.empty:
            ce_m = dict(zip(m_chain["strike_price"], m_chain["ce_oi"]))
            pe_m = dict(zip(m_chain["strike_price"], m_chain["pe_oi"]))
            all_ms   = sorted(float(s) for s in ce_m)
            band_m   = spot_close * 0.10
            strikes_m = [s for s in all_ms if abs(s - spot_close) <= band_m] or all_ms
            call_oi_m = [int(ce_m.get(s, 0)) for s in strikes_m]
            put_oi_m  = [int(pe_m.get(s, 0)) for s in strikes_m]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(y=strikes_m, x=call_oi_m,
                                   name="Call OI (Monthly)", orientation="h",
                                   marker_color="#ef9a9a", opacity=0.80))
            fig_m.add_trace(go.Bar(y=strikes_m, x=[-v for v in put_oi_m],
                                   name="Put OI (Monthly)",  orientation="h",
                                   marker_color="#80cbc4", opacity=0.80))
            if strikes_m:
                cls_m = min(strikes_m, key=lambda s: abs(s - spot_close))
                fig_m.add_hline(y=float(cls_m), line_color="#FFD600", line_width=2,
                                line_dash="dash",
                                annotation=dict(text=f"Spot {spot_close:.0f}", font_color="#FFD600"))
            if pred.monthly_max_pain and strikes_m:
                mmp_s = min(strikes_m, key=lambda s: abs(s - pred.monthly_max_pain))
                fig_m.add_hline(y=float(mmp_s), line_color="#FF9800", line_width=2,
                                line_dash="dot",
                                annotation=dict(text=f"M-MaxPain {pred.monthly_max_pain:.0f}",
                                                font_color="#FF9800"))
            fig_m.update_layout(
                barmode="overlay", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(color="#ccc", size=11),
                height=max(280, len(strikes_m) * 22),
                margin=dict(l=10, r=30, t=10, b=10),
                xaxis=dict(title="Put OI  <--  |  -->  Call OI", gridcolor=GRID_COLOR,
                           zeroline=True, zerolinecolor="#555", zerolinewidth=2, tickformat=","),
                yaxis=dict(title="Strike", gridcolor=GRID_COLOR,
                           tickvals=strikes_m, ticktext=[str(int(s)) for s in strikes_m]),
                legend=dict(orientation="h", x=0, y=1.02),
            )
            st.plotly_chart(fig_m, use_container_width=True)
            if pred.monthly_pcr is not None:
                mpcr_clr = "#69f0ae" if pred.monthly_pcr > 1.0 else ("#ff5252" if pred.monthly_pcr < 0.75 else "#FFD600")
                st.caption(
                    f"Monthly PCR: **{pred.monthly_pcr:.2f}**  "
                    f"· Monthly Max Pain: **{pred.monthly_max_pain:.0f}**  "
                    f"· Call OI: {pred.monthly_call_oi:,}  · Put OI: {pred.monthly_put_oi:,}"
                )
        else:
            st.caption("Monthly options chain not available for this date.")


# ── Signal breakdown ──────────────────────────────────────────────────────────

def _render_signals(pred: IndexPrediction) -> None:
    n_core = 17
    n_extra = 3 if pred.fno_symbol == "NIFTY" and pred.monthly_expiry is not None else 0
    n_desc = f"{n_core + n_extra}" if n_extra else str(n_core)
    multi_note = " + 3 Nifty multi-expiry signals (weekly vs monthly PCR, dual max pain, gamma wall)" if n_extra else ""
    _h("Signal Breakdown", f"All {n_desc} signals grouped by tier{multi_note}. Green = bullish, Red = bearish. Weighted composite score drives the final UP/DOWN/SIDEWAYS verdict. High-confidence calls require 3+ signals aligned in the same direction.")
    if not pred.signals:
        st.caption("No signals computed.")
        return

    # Group by tier for readability
    tier_order = ["Institutional", "Options OI", "Futures OI", "Carry",
                  "Price Action", "Market Context", "Sector"]
    grouped: dict[str, list[IndexSignal]] = {}
    for s in pred.signals:
        grouped.setdefault(s.category, []).append(s)

    for tier in tier_order:
        if tier not in grouped:
            continue
        cat_color = _CAT_COLOR.get(tier, "#888")
        st.markdown(
            f"<div style='font-size:0.75em;color:{cat_color};font-weight:700;"
            f"letter-spacing:0.05em;margin:8px 0 4px'>── {tier.upper()} ──</div>",
            unsafe_allow_html=True,
        )
        for sig in grouped[tier]:
            arrow       = "↑" if sig.direction > 0 else ("↓" if sig.direction < 0 else "↔")
            score_color = _POS if sig.score > 0 else (_NEG if sig.score < 0 else _NEU)
            st.markdown(
                f"""
                <div style="
                    background:#1a1a1a;border-left:3px solid {cat_color};
                    border-radius:6px;padding:8px 12px;margin-bottom:6px;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#fff;font-weight:600">{sig.emoji} {sig.headline}</span>
                        <span style="
                            background:{score_color}22;color:{score_color};
                            border:1px solid {score_color};border-radius:4px;
                            padding:1px 8px;font-weight:700;font-size:0.85em
                        ">{arrow} {sig.score:+.1f}</span>
                    </div>
                    <div style="color:#888;font-size:0.78em;margin-top:3px">
                        {sig.description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Key levels ────────────────────────────────────────────────────────────────

def _render_key_levels(pred: IndexPrediction) -> None:
    lv = pred.levels
    if all(v is None for v in [lv.top_put_strike, lv.max_pain, lv.top_call_strike]):
        return
    _h("Key Levels", "Put Wall = strong support (where put sellers will defend the market). Call Wall = strong resistance (where call sellers will suppress rallies). Max Pain = level where option writers lose the least — markets magnetically move toward this as expiry nears, especially in the last 5 trading days.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div style="background:#1a3d2e;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#69f0ae;font-size:0.8em;font-weight:600">SUPPORT (Put Wall)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.top_put_strike, 0) if lv.top_put_strike else '—'}</div>
                <div style="color:#888;font-size:0.75em">OI: {lv.put_oi_at_top:,}</div>
                {f'<div style="color:#555;font-size:0.7em">2nd: {lv.second_put_strike:.0f}</div>' if lv.second_put_strike else ''}
            </div>""", unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div style="background:#3d3320;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#FFD600;font-size:0.8em;font-weight:600">MAX PAIN (Expiry Pin)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.max_pain, 0) if lv.max_pain else '—'}</div>
                <div style="color:#888;font-size:0.75em">DTE: {pred.days_to_expiry}d</div>
                {f'<div style="color:#888;font-size:0.72em">Expiry: {pred.near_expiry.strftime("%d %b") if pred.near_expiry else ""}</div>' if pred.near_expiry else ''}
            </div>""", unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div style="background:#3d1a1a;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#ff5252;font-size:0.8em;font-weight:600">RESISTANCE (Call Wall)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.top_call_strike, 0) if lv.top_call_strike else '—'}</div>
                <div style="color:#888;font-size:0.75em">OI: {lv.call_oi_at_top:,}</div>
                {f'<div style="color:#555;font-size:0.7em">2nd: {lv.second_call_strike:.0f}</div>' if lv.second_call_strike else ''}
            </div>""", unsafe_allow_html=True,
        )

    # ── Nifty 50 multi-expiry panel ───────────────────────────────────────────
    if pred.fno_symbol == "NIFTY" and pred.monthly_expiry is not None:
        st.markdown("---")
        _h(
            "Multi-Expiry Structure (Nifty 50 — Weekly + Monthly)",
            "Nifty 50 is the only index with both weekly and monthly expiry. "
            "This creates a two-layer OI structure: weekly gamma pins the market this week; "
            "monthly max pain dictates where spot drifts after weekly expiry resolves.",
        )
        me1, me2, me3, me4 = st.columns(4)

        w_dte = (pred.weekly_expiry  - pred.as_of_date).days if pred.weekly_expiry else "—"
        m_dte = (pred.monthly_expiry - pred.as_of_date).days

        with me1:
            pcr_color   = "#69f0ae" if (pred.weekly_pcr or 0) > 1.0 else ("#ff5252" if (pred.weekly_pcr or 0) < 0.75 else "#FFD600")
            wpcr_str    = f"{pred.weekly_pcr:.2f}" if pred.weekly_pcr is not None else "—"
            st.markdown(
                f"""<div style="background:#1e2a1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">WEEKLY PCR</div>
                    <div style="color:{pcr_color};font-size:1.5em;font-weight:700">{wpcr_str}</div>
                    <div style="color:#555;font-size:0.7em">{pred.weekly_expiry.strftime('%d %b') if pred.weekly_expiry else '—'}  ({w_dte}d)</div>
                </div>""", unsafe_allow_html=True,
            )
        with me2:
            mpcr_color  = "#69f0ae" if (pred.monthly_pcr or 0) > 1.0 else ("#ff5252" if (pred.monthly_pcr or 0) < 0.75 else "#FFD600")
            mpcr_str    = f"{pred.monthly_pcr:.2f}" if pred.monthly_pcr is not None else "—"
            st.markdown(
                f"""<div style="background:#1e1e2a;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">MONTHLY PCR</div>
                    <div style="color:{mpcr_color};font-size:1.5em;font-weight:700">{mpcr_str}</div>
                    <div style="color:#555;font-size:0.7em">{pred.monthly_expiry.strftime('%d %b')}  ({m_dte}d)</div>
                </div>""", unsafe_allow_html=True,
            )
        with me3:
            mmp = pred.monthly_max_pain
            if mmp and pred.spot_close:
                mmp_gap = (mmp - pred.spot_close) / pred.spot_close * 100
                mmp_color = "#69f0ae" if mmp_gap > 0 else "#ff5252"
                mmp_str = f"{mmp:.0f}  ({mmp_gap:+.1f}%)"
            else:
                mmp_color = "#888"; mmp_str = "—"
            st.markdown(
                f"""<div style="background:#2a2a1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">MONTHLY MAX PAIN</div>
                    <div style="color:{mmp_color};font-size:1.2em;font-weight:700">{mmp_str}</div>
                    <div style="color:#555;font-size:0.7em">Post-weekly gravitational pull</div>
                </div>""", unsafe_allow_html=True,
            )
        with me4:
            gr = pred.gamma_ratio
            if gr is not None:
                gr_pct = gr * 100
                gr_color = "#ff9800" if gr_pct > 58 else ("#40c4ff" if gr_pct < 28 else "#888")
                gr_label = "⚠ Weekly Pinning" if gr_pct > 58 else ("↔ Balanced" if gr_pct > 28 else "↗ Monthly Dominates")
            else:
                gr_pct = 0; gr_color = "#888"; gr_label = "—"
            st.markdown(
                f"""<div style="background:#2a1e1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">GAMMA RATIO</div>
                    <div style="color:{gr_color};font-size:1.5em;font-weight:700">{gr_pct:.0f}%</div>
                    <div style="color:{gr_color};font-size:0.68em">{gr_label}</div>
                    <div style="color:#555;font-size:0.65em">Weekly OI / Total OI</div>
                </div>""", unsafe_allow_html=True,
            )


# ── Verdict panel ─────────────────────────────────────────────────────────────

def _render_verdict(pred: IndexPrediction) -> None:
    dir_color  = pred.direction_color
    conf_color = _CONF_COLOR.get(pred.confidence, "#78909C")
    dir_emoji  = _DIR_EMOJI.get(pred.direction, "↔")
    st.markdown(
        f"""
        <div style="
            background:#1e1e1e;border:1px solid {dir_color};
            border-radius:10px;padding:16px 20px;margin-bottom:12px;
        ">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                <div style="
                    background:{dir_color};color:#000;font-weight:700;
                    font-size:1.1em;padding:6px 16px;border-radius:8px
                ">{dir_emoji} {pred.direction}</div>
                <div style="color:{conf_color};font-weight:600">{pred.confidence} CONFIDENCE</div>
                <div style="color:#888;margin-left:auto;font-size:0.85em">
                    Score: <b style="color:#fff">{pred.composite_score:+.1f}</b>
                    &nbsp;|&nbsp; Signals: <b style="color:#fff">{len(pred.signals)}</b>
                </div>
            </div>
            <div style="color:#ddd;font-size:0.9em;margin-bottom:10px">{pred.headline}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82em">
                <div>
                    <div style="color:#69f0ae;font-weight:600;margin-bottom:2px">Key Driver</div>
                    <div style="color:#aaa">{pred.key_driver}</div>
                </div>
                <div>
                    <div style="color:#ff5252;font-weight:600;margin-bottom:2px">Key Risk</div>
                    <div style="color:#aaa">{pred.key_risk}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Market Context tab ────────────────────────────────────────────────────────

_IDX_LABELS = {
    "NIFTY": "Nifty 50", "BANKNIFTY": "Bank Nifty",
    "FINNIFTY": "Fin Nifty", "MIDCPNIFTY": "Midcap Nifty",
}


def _render_regime_panel(pred: IndexPrediction) -> None:
    """
    Statistical Regime tab — Hurst R/S, DFA, HMM, Permutation Entropy.

    Shows the three advanced mathematical signals computed in
    src/analytics/regime_detection.py with full interpretation.
    """
    r = pred.regime
    if r is None:
        st.info("Statistical regime signals not computed for this prediction.")
        return
    if r.error:
        st.warning(f"Regime detection error: {r.error}")
        return

    st.markdown(
        "<div style='font-size:12px;color:rgba(255,255,255,0.45);margin-bottom:16px'>"
        "Advanced statistical tools that measure market MEMORY, hidden REGIME STATE, "
        "and return COMPLEXITY — dimensions that OI/price signals cannot capture. "
        f"Computed on {r.data_points} trading days of historical returns."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 1. Hurst Exponent + DFA ───────────────────────────────────────────────
    _h("🌊 Market Memory — Hurst Exponent & DFA",
       "Hurst R/S (Hurst 1951) and DFA (Peng 1994) measure the autocorrelation "
       "decay law of daily returns. H > 0.58: returns are PERSISTENT — past direction "
       "predicts future direction (trend-following works). H < 0.42: returns are "
       "ANTI-PERSISTENT — mean-reversion works. H ≈ 0.50: random walk — no memory edge.")

    _HURST_COLORS = {
        "Trending":       "#4CAF50",
        "Random Walk":    "#9E9E9E",
        "Mean-Reverting": "#FF9800",
    }
    h_color = _HURST_COLORS.get(r.memory_label, "#9E9E9E")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hurst R/S",   f"{r.hurst_rs:.4f}",
              help="Rescaled Range estimator. Range: [0,1]. Above 0.5 = trending.")
    c2.metric("DFA Alpha",   f"{r.dfa_alpha:.4f}",
              help="Detrended Fluctuation Analysis exponent. More robust than R/S on non-stationary series.")
    c3.metric("Average H",   f"{r.memory_avg:.4f}",
              help="Mean of R/S and DFA — reduces estimation noise.")
    c4.metric("Signal Score", f"{r.memory_score:+.1f}",
              help="Contributes to the 23-signal composite. +2=strong bullish persistence, -2=strong bearish persistence.")

    st.markdown(
        f"<div style='padding:12px 16px;border-left:4px solid {h_color};"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;margin:8px 0'>"
        f"<div style='font-size:13px;font-weight:700;color:{h_color}'>"
        f"{r.memory_label}  (H = {r.memory_avg:.4f})</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.6);margin-top:4px'>"
        f"{r.memory_note}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Visual gauge: Hurst on 0→1 axis with zones
    fig_h = go.Figure()
    # Zones
    fig_h.add_shape(type="rect", x0=0, x1=0.42, y0=0, y1=1,
                    fillcolor="rgba(255,152,0,0.15)", line_width=0)
    fig_h.add_shape(type="rect", x0=0.42, x1=0.58, y0=0, y1=1,
                    fillcolor="rgba(158,158,158,0.10)", line_width=0)
    fig_h.add_shape(type="rect", x0=0.58, x1=1.0, y0=0, y1=1,
                    fillcolor="rgba(76,175,80,0.15)", line_width=0)
    # Labels
    for xpos, label, col in [(0.21, "Mean-Reverting", "#FF9800"),
                               (0.50, "Random Walk",    "#9E9E9E"),
                               (0.79, "Trending",       "#4CAF50")]:
        fig_h.add_annotation(x=xpos, y=0.5, text=label, showarrow=False,
                              font=dict(size=11, color=col))
    # Markers
    for val, name, color, sym in [
        (r.hurst_rs,  "R/S",  "#2196F3", "diamond"),
        (r.dfa_alpha, "DFA",  "#FF5722", "circle"),
        (r.memory_avg,"Avg",  "#FFD700", "star"),
    ]:
        fig_h.add_trace(go.Scatter(
            x=[val], y=[0.5], mode="markers+text",
            marker=dict(size=16, color=color, symbol=sym,
                        line=dict(width=2, color="white")),
            text=[f"<b>{name}</b><br>{val:.4f}"],
            textposition="top center",
            name=name,
            showlegend=True,
        ))
    fig_h.update_layout(
        height=160, template="plotly_dark",
        xaxis=dict(range=[0, 1], title="Hurst / DFA exponent", tickformat=".2f",
                   showgrid=False),
        yaxis=dict(visible=False),
        margin=dict(t=20, b=40, l=20, r=20),
        legend=dict(orientation="h", y=-0.6, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_h, use_container_width=True,
                    key=f"hurst_{pred.fno_symbol}")

    st.divider()

    # ── 2. Hidden Markov Model Regime ─────────────────────────────────────────
    _h("🔮 Hidden Markov Model — Market Regime",
       "3-state Gaussian HMM fitted on 90 days of [daily_return, rolling_5D_vol]. "
       "Detects the latent regime the market is currently in. "
       "Bull: positive return + normal vol. Bear: negative return + HIGH vol. "
       "Sideways: near-zero return + LOW vol. Score = p_Bull×3 − p_Bear×3.")

    if r.hmm_state == "Unknown" or not r.hmm_probs:
        st.info("HMM not computed (insufficient data).")
    else:
        p_bull, p_side, p_bear = r.hmm_probs

        _STATE_COLOR = {"Bull": "#4CAF50", "Sideways": "#FFD600", "Bear": "#EF5350"}
        state_color  = _STATE_COLOR.get(r.hmm_state, "#9E9E9E")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Current State",  r.hmm_state,
                  help="Most probable hidden state at today's close.")
        h2.metric("State Prob",     f"{r.hmm_prob:.1%}",
                  help="Probability of the detected state. Higher = more confident regime ID.")
        h3.metric("HMM Score",      f"{r.hmm_score:+.2f}",
                  help="p_Bull×3 − p_Bear×3. Range ≈ [−3, +3]. Contributes to 23-signal composite.")
        h4.metric("Data Window",    f"{min(r.data_points, 90)}D",
                  help="Number of trading days used to fit the HMM.")

        # Probability bar — Bull / Sideways / Bear
        fig_hmm = go.Figure()
        for label, prob, color in [
            ("Bull",     p_bull, "#4CAF50"),
            ("Sideways", p_side, "#FFD600"),
            ("Bear",     p_bear, "#EF5350"),
        ]:
            fig_hmm.add_trace(go.Bar(
                name=label, x=[prob], y=["State"],
                orientation="h",
                marker_color=color,
                text=f"<b>{label}</b> {prob:.1%}",
                textposition="inside" if prob > 0.08 else "outside",
                insidetextanchor="middle",
                hovertemplate=f"{label}: {prob:.3%}<extra></extra>",
            ))
        fig_hmm.update_layout(
            barmode="stack", height=80, template="plotly_dark",
            xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
            yaxis=dict(visible=False),
            margin=dict(t=10, b=30, l=10, r=10),
            legend=dict(orientation="h", y=-1.0, x=0.5, xanchor="center"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_hmm, use_container_width=True,
                        key=f"hmm_{pred.fno_symbol}")

        st.markdown(
            f"<div style='padding:10px 14px;border-left:4px solid {state_color};"
            f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;font-size:12px;"
            f"color:rgba(255,255,255,0.6)'>{r.hmm_note}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 3. Permutation Entropy + Sample Entropy ───────────────────────────────
    _h("🌪️ Entropy — Market Complexity & Predictability",
       "Permutation Entropy (Bandt & Pompe 2002) measures the complexity of ordinal "
       "return patterns. PE → 0: highly ordered, repeating patterns — high predictability. "
       "PE → 1: maximum disorder — signals are less reliable, widen stops. "
       "Sample Entropy measures irregularity without self-matching bias. "
       "When PE > 0.72 (Chaotic), HIGH confidence verdicts are automatically capped at MEDIUM.")

    _ENT_COLORS = {"Ordered": "#4CAF50", "Moderate": "#FFD600", "Chaotic": "#EF5350", "Unknown": "#9E9E9E"}
    ent_color = _ENT_COLORS.get(r.entropy_label, "#9E9E9E")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Perm Entropy",    f"{r.perm_entropy:.4f}",
              help="Permutation Entropy on 60D returns (order=4). Range [0,1]. Low = ordered/predictable.")
    e2.metric("Sample Entropy",
              f"{r.samp_entropy:.4f}" if (r.samp_entropy is not None and
              r.samp_entropy == r.samp_entropy) else "—",
              help="SampEn on 60D returns (m=2). Low = regular/predictable. High = irregular/complex.")
    e3.metric("Complexity",      r.entropy_label,
              help="Ordered (<0.50), Moderate (0.50–0.72), Chaotic (>0.72).")
    e4.metric("Conf Multiplier", f"{r.entropy_conf:.2f}×",
              help="When Chaotic: HIGH verdict capped to MEDIUM. When Ordered: normal HIGH allowed.")

    # Entropy gauge
    fig_e = go.Figure()
    fig_e.add_shape(type="rect", x0=0, x1=0.50, y0=0, y1=1,
                    fillcolor="rgba(76,175,80,0.15)", line_width=0)
    fig_e.add_shape(type="rect", x0=0.50, x1=0.72, y0=0, y1=1,
                    fillcolor="rgba(255,213,0,0.10)", line_width=0)
    fig_e.add_shape(type="rect", x0=0.72, x1=1.0, y0=0, y1=1,
                    fillcolor="rgba(239,83,80,0.15)", line_width=0)
    for xpos, label, col in [(0.25, "Ordered", "#4CAF50"),
                               (0.61, "Moderate", "#FFD600"),
                               (0.86, "Chaotic",  "#EF5350")]:
        fig_e.add_annotation(x=xpos, y=0.5, text=label, showarrow=False,
                              font=dict(size=11, color=col))
    # Threshold lines
    for xv, xt in [(0.50, "0.50"), (0.72, "0.72")]:
        fig_e.add_vline(x=xv, line_dash="dot",
                        line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_e.add_annotation(x=xv, y=0.05, text=xt, showarrow=False,
                              font=dict(size=9, color="rgba(255,255,255,0.4)"))
    fig_e.add_trace(go.Scatter(
        x=[r.perm_entropy], y=[0.5], mode="markers+text",
        marker=dict(size=18, color=ent_color, symbol="diamond",
                    line=dict(width=2, color="white")),
        text=[f"<b>PE = {r.perm_entropy:.4f}</b>"],
        textposition="top center",
        showlegend=False,
    ))
    fig_e.update_layout(
        height=160, template="plotly_dark",
        xaxis=dict(range=[0, 1], title="Permutation Entropy", tickformat=".2f",
                   showgrid=False),
        yaxis=dict(visible=False),
        margin=dict(t=20, b=40, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_e, use_container_width=True,
                    key=f"entropy_{pred.fno_symbol}")

    st.markdown(
        f"<div style='padding:10px 14px;border-left:4px solid {ent_color};"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;font-size:12px;"
        f"color:rgba(255,255,255,0.6)'>{r.entropy_note}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 4. Combined interpretation ────────────────────────────────────────────
    st.markdown("#### Combined Statistical Reading")

    h_label  = r.memory_label
    hmm_state = r.hmm_state
    ent_label = r.entropy_label

    bullets = []
    if h_label == "Trending" and r.memory_score > 0:
        bullets.append(("🌊", "#4CAF50",
                         f"Persistent uptrend (H={r.memory_avg:.3f}) — momentum signals have statistical backing."))
    elif h_label == "Trending" and r.memory_score < 0:
        bullets.append(("🌊", "#EF5350",
                         f"Persistent downtrend (H={r.memory_avg:.3f}) — bearish momentum has statistical backing."))
    elif h_label == "Mean-Reverting":
        bullets.append(("🔄", "#FF9800",
                         f"Anti-persistent returns (H={r.memory_avg:.3f}) — strong directional moves are likely to reverse."))
    else:
        bullets.append(("🎲", "#9E9E9E",
                         f"Near-random walk (H={r.memory_avg:.3f}) — no autocorrelation edge."))

    if hmm_state == "Sideways":
        bullets.append(("↔️", "#FFD600",
                         "HMM detects a sustained sideways/consolidation regime — "
                         "directional breakouts lack statistical regime support."))
    elif hmm_state == "Bull":
        bullets.append(("📈", "#4CAF50",
                         f"HMM confirms Bull regime (p={r.hmm_prob:.0%}) — "
                         "regime supports long positioning."))
    elif hmm_state == "Bear":
        bullets.append(("📉", "#EF5350",
                         f"HMM confirms Bear regime (p={r.hmm_prob:.0%}) — "
                         "regime supports caution / short positioning."))

    if ent_label == "Chaotic":
        bullets.append(("🌪️", "#EF5350",
                         f"High entropy (PE={r.perm_entropy:.4f}) — return patterns are disordered. "
                         "Reduce position size, widen stops, cap HIGH confidence verdicts at MEDIUM."))
    elif ent_label == "Ordered":
        bullets.append(("🎯", "#4CAF50",
                         f"Low entropy (PE={r.perm_entropy:.4f}) — patterns are predictable and repeating. "
                         "Directional signals have elevated reliability."))

    for emoji, color, text in bullets:
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;"
            f"background:rgba(255,255,255,0.03);border-left:3px solid {color};"
            f"border-radius:0 6px 6px 0;font-size:13px'>"
            f"{emoji} <span style='color:{color};font-weight:600'></span> "
            f"<span style='color:rgba(255,255,255,0.75)'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_market_context(pred: IndexPrediction) -> None:
    ctx = pred.market_context
    if ctx is None:
        st.caption("Market context not available.")
        return

    # ── India VIX ─────────────────────────────────────────────────────────────
    _h("India VIX", "India VIX measures expected 30-day market volatility (fear gauge). VIX above 20 = high fear = options expensive = markets unstable. VIX below 13 = complacency = cheap options = potential for sharp moves. Contrarian signal: extreme high VIX often marks a bottom; extreme low VIX often marks a top.")
    v1, v2, v3, v4 = st.columns(4)
    vix_chg_str = f"{ctx.vix_pct_chg:+.2f}%" if ctx.vix_pct_chg is not None else "—"
    vix_5d_str  = f"{ctx.vix_5d_chg_pct:+.1f}%" if ctx.vix_5d_chg_pct is not None else "—"
    v1.metric("India VIX",    f"{ctx.vix_close:.2f}" if ctx.vix_close else "—")
    v2.metric("Today Change", vix_chg_str)
    v3.metric("5-Day Change", vix_5d_str)
    v4.metric("Regime",       ctx.vix_regime)

    if ctx.vix_close is not None:
        vix = ctx.vix_close
        chg = ctx.vix_pct_chg or 0
        if vix < 12:
            interp = "Complacency zone — be cautious of unexpected spikes"
            ic = _NEG
        elif vix < 18:
            interp = f"Normal range — {'fear clearing (bullish)' if chg < -3 else 'anxiety building (bearish)' if chg > 3 else 'stable'}"
            ic = _POS if chg < -3 else (_NEG if chg > 3 else _NEU)
        elif vix < 22:
            interp = f"Elevated — {'fear receding (relief rally likely)' if chg < -5 else 'uncertainty high, wide swings expected'}"
            ic = _POS if chg < -5 else _NEG
        else:
            interp = "Fear zone — contrarian buy signal; capitulation near"
            ic = _POS
        st.markdown(
            f"<div style='color:{ic};font-size:0.85em;margin-top:4px'>📊 {interp}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Sector Breadth ────────────────────────────────────────────────────────
    _h("Sector Breadth", "Fraction of NSE sectors showing above-average delivery volume today. High breadth (70%+) = broad market participation = rally has conviction. Low breadth (below 30%) = narrow market = only 1-2 sectors driving the index = fragile rally. Use this to confirm or doubt the index-level signal.")
    if ctx.breadth_pct_advancing is not None:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Advancing",     str(ctx.advancing_sectors))
        b2.metric("Declining",     str(ctx.declining_sectors))
        b3.metric("Total Sectors", str(ctx.total_sectors))
        b4.metric("% Advancing",   f"{ctx.breadth_pct_advancing:.0f}%")

        adv_pct   = ctx.breadth_pct_advancing
        bar_color = _POS if adv_pct >= 65 else (_NEG if adv_pct <= 35 else "#FFD600")
        st.markdown(
            f"""<div style="background:#333;border-radius:4px;height:12px;width:100%;margin:6px 0">
                <div style="background:{bar_color};border-radius:4px;height:12px;width:{adv_pct:.0f}%"></div>
            </div>
            <div style="font-size:0.78em;color:#888">
                {ctx.advancing_sectors} advancing &nbsp;·&nbsp; {ctx.declining_sectors} declining
                &nbsp;·&nbsp; {ctx.total_sectors - ctx.advancing_sectors - ctx.declining_sectors} flat
            </div>""",
            unsafe_allow_html=True,
        )

        if ctx.cyclical_avg_chg is not None and ctx.defensive_avg_chg is not None:
            st.markdown("**Sector Rotation**")
            r1, r2, r3 = st.columns(3)
            mode_color = _POS if ctx.risk_mode == "RISK-ON" else (_NEG if ctx.risk_mode == "RISK-OFF" else _NEU)
            r1.metric("Cyclicals (Bank/Auto/Metal)", f"{ctx.cyclical_avg_chg:+.2f}%")
            r2.metric("Defensives (FMCG/Pharma/IT)", f"{ctx.defensive_avg_chg:+.2f}%")
            r3.metric("Risk Mode", ctx.risk_mode)
            st.markdown(
                f"<div style='color:{mode_color};font-size:0.85em'>"
                f"{'⚡ Institutions rotating into growth sectors — RISK-ON' if ctx.risk_mode == 'RISK-ON' else '🛡️ Flight to defensives — RISK-OFF' if ctx.risk_mode == 'RISK-OFF' else '⚖️ Mixed rotation — no dominant theme'}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Sector breadth data not available for this date.")

    st.divider()

    # ── FII/DII/Client/Pro positioning ────────────────────────────────────────
    _h("Institutional Positioning (Index Futures OI)", "Net futures contracts held by each participant type. FII (Foreign Institutions) = smart money — follow their direction. DII (Domestic Institutions) = often contrarian to FII. Client = retail — contrarian indicator (when retail is very long, be cautious). Pro = proprietary desks. Key rule: FII net short but REDUCING position = covering = bullish price pressure.")
    if ctx.fao_date:
        lag = (ctx.trade_date - ctx.fao_date).days
        st.caption(
            f"Source: FAO Participant OI as of {ctx.fao_date.strftime('%d %b %Y')}"
            + (f" ({lag} day lag)" if lag > 0 else " (current)")
        )

        parts = [
            ("FII",    ctx.fii_fut_idx_net,    "Foreign Institutional Investors"),
            ("DII",    ctx.dii_fut_idx_net,    "Domestic Institutional Investors"),
            ("Client", ctx.client_fut_idx_net, "Retail / Client"),
            ("Pro",    ctx.pro_fut_idx_net,    "Proprietary Desks"),
        ]
        html = """
        <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
            <thead>
                <tr style="border-bottom:1px solid #333;color:#888">
                    <th style="text-align:left;padding:5px 8px">Participant</th>
                    <th style="text-align:right;padding:5px 8px">Net Contracts</th>
                    <th style="text-align:left;padding:5px 8px;color:#666">Interpretation</th>
                </tr>
            </thead><tbody>
        """
        for name, net, desc in parts:
            color   = _POS if net > 0 else (_NEG if net < 0 else _NEU)
            bias    = "LONG 🐂" if net > 50_000 else ("SHORT 🐻" if net < -50_000 else "NEUTRAL")
            bar_w   = min(abs(net) / 5_000, 100)
            bar_col = _POS if net > 0 else (_NEG if net < 0 else _NEU)
            html += f"""
            <tr style="border-bottom:1px solid #1a1a1a">
                <td style="padding:6px 8px;color:#ccc;font-weight:600">{name}</td>
                <td style="padding:6px 8px;text-align:right">
                    <span style="color:{color};font-weight:700">{net:+,}</span>
                </td>
                <td style="padding:6px 8px">
                    <div style="background:#333;border-radius:3px;height:6px;width:100px;display:inline-block;vertical-align:middle;margin-right:6px">
                        <div style="background:{bar_col};border-radius:3px;height:6px;width:{bar_w:.0f}px"></div>
                    </div>
                    <span style="color:#888;font-size:0.85em">{bias}</span>
                </td>
            </tr>"""
        html += "</tbody></table>"
        st.html(html.strip())

        # ── FII Position Change (Day-over-Day) ────────────────────────────────
        if ctx.fii_prev_fao_date:
            st.markdown("**FII Position Change (Day-over-Day)**")
            chg       = ctx.fii_net_change_1d
            chg_color = _POS if chg > 0 else (_NEG if chg < 0 else _NEU)
            action    = "COVERING SHORTS 🔄" if chg > 0 else ("ADDING SHORTS 🐻" if chg < 0 else "FLAT")
            interp    = (
                "FII buying back shorts — creates upward price pressure; watch for reversal trigger."
                if chg > 0 else
                "FII building more shorts — increasing bearish conviction. Doubling down on existing position."
                if chg < 0 else
                "FII position unchanged day-over-day."
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(f"Net {ctx.fii_prev_fao_date.strftime('%d %b')}", f"{ctx.fii_fut_idx_net_prev:+,}")
            pc2.metric(f"Net {ctx.fao_date.strftime('%d %b')}",           f"{ctx.fii_fut_idx_net:+,}")
            pc3.metric("1-Day Change",                                     f"{chg:+,}")
            st.markdown(
                f"<div style='color:{chg_color};font-size:0.85em;padding:4px 0'>"
                f"<b>{action}</b> — {interp}</div>",
                unsafe_allow_html=True,
            )

        # ── FII Stock Futures (market-neutral context) ─────────────────────────
        if ctx.fii_stock_fut_net_cr is not None:
            sfc       = ctx.fii_stock_fut_net_cr
            sfc_color = _POS if sfc > 0 else _NEG
            sfc_txt   = f"Rs{abs(sfc):,.0f}Cr {'Net BUY' if sfc > 0 else 'Net SELL'}"
            note = ""
            if sfc > 500 and ctx.fii_fut_idx_net < -50_000:
                note = " — Market-Neutral Pairs Trade: index short is a hedge, not pure directional bearish."
            elif sfc < -500 and ctx.fii_fut_idx_net < -50_000:
                note = " — Double bearish: selling both index futures AND stock futures."
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;background:#1a1a1a;"
                f"border-left:3px solid {sfc_color};border-radius:6px;font-size:0.85em'>"
                f"<b style='color:#ccc'>FII Stock Futures (Today):</b> "
                f"<span style='color:{sfc_color}'>{sfc_txt}</span>"
                f"<span style='color:#888'>{note}</span></div>",
                unsafe_allow_html=True,
            )

        # ── FII Options Book ──────────────────────────────────────────────────
        if ctx.fii_opt_delta != 0:
            st.markdown("**FII Options Book**")
            o1, o2, o3 = st.columns(3)
            delta_color = _POS if ctx.fii_opt_delta > 0 else _NEG
            o1.metric("FII Net Call OI", f"{ctx.fii_opt_call_net:+,}")
            o2.metric("FII Net Put OI",  f"{ctx.fii_opt_put_net:+,}")
            o3.metric("Options Delta",   f"{ctx.fii_opt_delta:+,}")
            bias_txt = (
                "FII positioned for upside via options (net long calls)"
                if ctx.fii_opt_delta > 0
                else "FII heavily hedged — net long puts (expect downside or volatility)"
            )
            st.markdown(
                f"<div style='color:{delta_color};font-size:0.82em;margin-top:4px'>{bias_txt}</div>",
                unsafe_allow_html=True,
            )
            opt_flow_cr = ctx.fii_opt_net_flows.get(pred.fno_symbol)
            if opt_flow_cr is not None:
                ofl_color = _POS if opt_flow_cr > 0 else _NEG
                ofl_txt   = f"Rs{abs(opt_flow_cr):,.0f}Cr {'Net BUY' if opt_flow_cr > 0 else 'Net SELL'}"
                st.markdown(
                    f"<div style='margin-top:6px;font-size:0.82em;color:#aaa'>"
                    f"FII Options Flow Today ({pred.fno_symbol}): "
                    f"<span style='color:{ofl_color};font-weight:600'>{ofl_txt}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── FII Today's Futures Flow ──────────────────────────────────────────
        flow_cr = ctx.fii_symbol_flows.get(pred.fno_symbol)
        if flow_cr is not None:
            flow_color = _POS if flow_cr > 0 else _NEG
            flow_txt   = f"Rs{abs(flow_cr):,.0f}Cr {'Net BUY' if flow_cr > 0 else 'Net SELL'}"
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;background:#1a1a1a;"
                f"border-left:3px solid {flow_color};border-radius:6px;"
                f"color:{flow_color};font-weight:600;font-size:0.9em'>"
                f"FII Net Futures Flow Today — {pred.fno_symbol}: {flow_txt}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── FII 5-Day Cumulative Flow (All Indices) ───────────────────────────
        if ctx.fii_cumul_flows_5d:
            _h("FII 5-Day Cumulative Futures Flow (Rs Crore)", "Sum of FII net futures buying/selling over the last 5 trading days in Rupee Crore terms. Single-day flow can be noise (expiry, rebalancing). 5-day cumulative flow reveals the TRUE institutional trend. Positive = net accumulation; Negative = net distribution. Strong Accumulation above Rs 5,000 Cr is significant.")
            st.caption("5D rolling net — sustained institutional pressure; single-day flows can be noise")
            html5 = """
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
                <thead>
                    <tr style="border-bottom:1px solid #333;color:#888">
                        <th style="text-align:left;padding:5px 8px">Index</th>
                        <th style="text-align:right;padding:5px 8px">5D Net (Rs Cr)</th>
                        <th style="text-align:left;padding:5px 8px">Signal</th>
                    </tr>
                </thead><tbody>
            """
            for sym, label in _IDX_LABELS.items():
                cumul = ctx.fii_cumul_flows_5d.get(sym)
                if cumul is None:
                    continue
                c_col = _POS if cumul > 0 else (_NEG if cumul < 0 else _NEU)
                if cumul >= 5_000:    sig = "Strong Accumulation"
                elif cumul >= 2_000:  sig = "Accumulation"
                elif cumul > -2_000:  sig = "Neutral"
                elif cumul > -5_000:  sig = "Distribution"
                else:                 sig = "Heavy Distribution"
                hl = " background:#252015;" if sym == pred.fno_symbol else ""
                name_color = "#FFD600" if sym == pred.fno_symbol else "#ccc"
                name_weight = "700" if sym == pred.fno_symbol else "400"
                dir_arrow = "▲" if cumul >= 0 else "▼"
                html5 += (
                    f"<tr style='border-bottom:1px solid #1a1a1a;{hl}'>"
                    f"<td style='padding:6px 8px;color:{name_color};font-weight:{name_weight}'>{label}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:{c_col};font-weight:700'>"
                    f"{dir_arrow} Rs{abs(cumul):,.0f}</td>"
                    f"<td style='padding:6px 8px;color:#aaa'>{sig}</td>"
                    f"</tr>"
                )
            html5 += "</tbody></table>"
            st.html(html5.strip())

        # ── FII OI Buildup Trend (All Indices) ────────────────────────────────
        if ctx.fii_oi_cr_latest and ctx.fii_oi_cr_5d_ago:
            _h("FII OI Rs-Cr Buildup Trend (5-Day)", "Whether FII open interest is growing or shrinking over 5 days. Growing OI + FII net short = FIIs adding new short bets = increasing bearish conviction. Shrinking OI + FII net short = FIIs covering shorts = potential reversal signal. Growing OI + FII net long = building bullish conviction.")
            st.caption("Growing OI + net short = adding conviction | Shrinking OI + net short = covering (reversal risk)")
            html_oi = """
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
                <thead>
                    <tr style="border-bottom:1px solid #333;color:#888">
                        <th style="text-align:left;padding:5px 8px">Index</th>
                        <th style="text-align:right;padding:5px 8px">OI Now (Rs Cr)</th>
                        <th style="text-align:right;padding:5px 8px">OI 5D Ago</th>
                        <th style="text-align:right;padding:5px 8px">5D Chg</th>
                        <th style="text-align:left;padding:5px 8px">Signal</th>
                    </tr>
                </thead><tbody>
            """
            fii_net = ctx.fii_fut_idx_net
            for sym, label in _IDX_LABELS.items():
                now = ctx.fii_oi_cr_latest.get(sym)
                ago = ctx.fii_oi_cr_5d_ago.get(sym)
                if now is None or ago is None or ago == 0:
                    continue
                chg_pct = (now - ago) / ago * 100
                chg_col = _POS if chg_pct > 0 else _NEG
                if chg_pct >= 10 and fii_net < -50_000:    sig = "Building Shorts"
                elif chg_pct >= 10 and fii_net > 50_000:   sig = "Building Longs"
                elif chg_pct <= -10 and fii_net < -50_000: sig = "Covering Shorts"
                elif chg_pct <= -10 and fii_net > 50_000:  sig = "Reducing Longs"
                else:                                        sig = "Stable"
                hl = " background:#252015;" if sym == pred.fno_symbol else ""
                name_color  = "#FFD600" if sym == pred.fno_symbol else "#ccc"
                name_weight = "700" if sym == pred.fno_symbol else "400"
                html_oi += (
                    f"<tr style='border-bottom:1px solid #1a1a1a;{hl}'>"
                    f"<td style='padding:6px 8px;color:{name_color};font-weight:{name_weight}'>{label}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:#fff'>Rs{now:,.0f}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:#666'>Rs{ago:,.0f}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:{chg_col};font-weight:700'>{chg_pct:+.1f}%</td>"
                    f"<td style='padding:6px 8px;color:#aaa'>{sig}</td>"
                    f"</tr>"
                )
            html_oi += "</tbody></table>"
            st.html(html_oi.strip())

    else:
        st.caption("FAO participant data not available for this date.")

    # ── PE Valuation ─────────────────────────────────────────────────────────
    if ctx.nifty_pe is not None:
        st.divider()
        _h("Nifty 50 Valuation", "Price-to-Earnings ratio of Nifty 50. Historical mean is ~22x. Above 24x = expensive = lower margin of safety for bulls. Below 18x = cheap = strong long-term entry. PE alone does not predict short-term direction but sets the risk context — at PE 26x a 5% dip is healthy correction; at PE 18x the same dip is a buying opportunity.")
        pe = ctx.nifty_pe
        if pe < 16:
            pe_color = _POS; pe_label = "Cheap"
            pe_note  = "Below historical mean (~22x) — attractive entry; cushion on downside"
        elif pe < 22:
            pe_color = _POS; pe_label = "Fair Value (Low)"
            pe_note  = "Below mean — reasonable valuation, no major earnings headwind"
        elif pe < 28:
            pe_color = _NEU; pe_label = "Fair Value (High)"
            pe_note  = "Near/above mean — fully priced; requires earnings support to sustain"
        else:
            pe_color = _NEG; pe_label = "Expensive"
            pe_note  = "Above historical norms (>28x) — elevated correction risk on negative catalysts"
        pe1, pe2, pe3 = st.columns(3)
        pe1.metric("Nifty PE Ratio",   f"{pe:.1f}x")
        pe2.metric("Valuation Zone",   pe_label)
        pe3.metric("Historical Mean",  "~22x")
        st.markdown(
            f"<div style='color:{pe_color};font-size:0.85em;margin-top:4px'>📊 {pe_note}</div>",
            unsafe_allow_html=True,
        )


# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    st.subheader("Index Prediction — Tomorrow's Directional Forecast")
    st.caption(
        "17-signal quant engine: OI-Price Matrix · Carry · Max Pain · PCR · Wyckoff Range · "
        "Price Mean-Reversion · FII Institutional · FII Options Delta · FII Flow · "
        "FII 5D Cumulative · FII OI Buildup · FII Position Change · Short Squeeze Setup · "
        "India VIX · Sector Breadth · Cyclical/Defensive Rotation · PE Valuation"
    )

    with st.expander("📖 How to read this page", expanded=False):
        st.markdown("""
**Prediction Cards (top row)** — one card per index (Nifty 50, Bank Nifty, Fin Nifty, Midcap Nifty).
- **Direction badge** (🟢 UP / 🔴 DOWN / 🟡 SIDEWAYS) — tomorrow's expected move from the 17-signal engine
- **Score** — composite signal score (−20 to +20). Above +3 = bullish bias, below −3 = bearish bias, −3 to +3 = sideways/uncertain
- **PCR** — Put-Call Ratio. Above 1.3 = peak fear, options writers are heavily on the put side (contrarian bullish). Below 0.70 = complacency, call-heavy (contrarian bearish)
- **Carry** — annualised futures premium vs spot. Above 8% = market expects a rise. Below 3% or negative = futures trading at discount = smart money is cautious or bearish
- **FII net** — Foreign Institutional net futures contracts today. Positive = net long, negative = net short
- **S / MP / R** — Support / Max Pain / Resistance for the nearest expiry (max pain = strike with least OI writer pain)

**Summary Bar (below cards)** — market-wide context shared across all 4 indices.
- **India VIX** — fear gauge. Below 13 = calm market. 13–17 = normal. 17–20 = elevated caution. Above 20 = high fear / sharp moves likely
- **Breadth** — how many of the tracked sectors are advancing today. Above 60% = broad rally. Below 40% = broad selloff. Mixed = rotation
- **FII Fut Net** — total FII net futures position (index futures, all indices combined). Persistent negative = institutions are hedging or shorting the market
- **FII 1D Chg** — change in FII futures position vs yesterday. COVERING = reducing shorts (bullish signal). ADDING = adding to shorts (bearish signal)
- **Risk Mode** — derived from VIX + breadth + FII positioning: RISK-ON (buy-the-dip) / RISK-OFF (reduce exposure) / NEUTRAL
- **VIX Regime** — CALM / NORMAL / ELEVATED / HIGH based on VIX level; determines how much signal weight to give to volatility signals

**Detail Tabs** — select an index from the dropdown, then explore:
- **Verdict & Signals** — full 17-signal breakdown with individual scores and explanations
- **Key Levels** — put wall, call wall, max pain chart with cost-of-carry context
- **Today vs Yesterday** — spot changes in OI, carry, PCR, breadth to see what shifted overnight
- **OI Structure** — call/put OI chart showing where options writers have concentrated positions (these act as support/resistance)
- **Market Context** — India VIX trend, FII 5-day cumulative flow, sector breadth table, Nifty PE valuation

**Tip:** Use the Trading Date dropdown (sidebar) to select a past date and see what the engine predicted — then compare to what actually happened next day.
        """)

    preds = cached_index_predictions(selected_date)
    if not preds:
        st.warning("No prediction data available. Fetch F&O bhavcopy first.")
        return

    # ── Prediction cards ──────────────────────────────────────────────────────
    cols = st.columns(len(preds))
    for col, pred in zip(cols, preds):
        with col:
            _render_card(pred)

    st.divider()

    # ── Market-wide summary bar (breadth + VIX + FII) ─────────────────────────
    ctx = preds[0].market_context if preds else None
    if ctx:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("India VIX",
                  f"{ctx.vix_close:.1f}" if ctx.vix_close else "—",
                  f"{ctx.vix_pct_chg:+.2f}%" if ctx.vix_pct_chg else None,
                  help="NSE India VIX (volatility index). Below 13 = calm. 13–17 = normal. 17–20 = cautious. Above 20 = high fear, sharp moves likely. Delta shown is today's % change.")
        m2.metric("Breadth",
                  f"{ctx.breadth_pct_advancing:.0f}%" if ctx.breadth_pct_advancing else "—",
                  f"{ctx.advancing_sectors}/{ctx.total_sectors} up",
                  help="Sector breadth: % of tracked sectors with positive price change today. Above 60% = broad rally. Below 40% = broad selloff. Mixed = rotation between sectors.")
        m3.metric("FII Fut Net", f"{ctx.fii_fut_idx_net:+,}" if ctx.fao_date else "—",
                  help="FII net index futures position today (contracts). Positive = net long (bullish), negative = net short (hedging or bearish). Sustained large negative = institutional caution.")
        if ctx.fii_prev_fao_date:
            _chg = ctx.fii_net_change_1d
            m4.metric("FII 1D Chg",
                      f"{_chg:+,}",
                      "COVERING" if _chg > 0 else ("ADDING" if _chg < 0 else "FLAT"),
                      help="Change in FII futures position vs previous F&O day. COVERING = reducing short positions (bullish signal). ADDING = increasing shorts (bearish signal). FLAT = no meaningful change.")
        else:
            m4.metric("FII 1D Chg", "—",
                      help="Change in FII futures position vs previous F&O day. Not available when prior day data is missing.")
        m5.metric("Risk Mode",   ctx.risk_mode,
                  help="Composite market mode derived from VIX level + sector breadth + FII positioning. RISK-ON = favourable for longs. RISK-OFF = reduce exposure, prefer cash/hedges. NEUTRAL = mixed signals.")
        m6.metric("VIX Regime",  ctx.vix_regime,
                  help="VIX classification: CALM (VIX < 13), NORMAL (13–17), ELEVATED (17–20), HIGH (> 20). Regime affects signal weights — in HIGH regime, mean-reversion signals get more weight than trend signals.")
        st.divider()

    # ── Index selector ────────────────────────────────────────────────────────
    sym_options  = [p.fno_symbol for p in preds]
    selected_sym = st.selectbox(
        "Select Index for Detail",
        options=sym_options,
        format_func=lambda s: next(p.display_name for p in preds if p.fno_symbol == s),
    )
    detail = next((p for p in preds if p.fno_symbol == selected_sym), None)
    if detail is None:
        return
    if not detail.data_available:
        st.info(detail.note or "No F&O data available for this index on the selected date.")
        return

    # ── Detail tabs ───────────────────────────────────────────────────────────
    tab_verdict, tab_levels, tab_compare, tab_oi, tab_ctx, tab_regime = st.tabs([
        "Verdict & Signals", "Key Levels", "Today vs Yesterday",
        "OI Structure", "Market Context", "📐 Statistical Regime",
    ])

    with tab_verdict:
        _render_verdict(detail)
        _render_signals(detail)

    with tab_levels:
        _render_key_levels(detail)
        _h("Cost of Carry", "Annualised implied carry = (Futures price - Spot) / Spot / days_to_expiry * 365 * 100. At India repo rate ~6.5%, fair carry is 6-7%. Above 8% = market expects rise (bullish bias in futures). Below 3% = market expects fall or uncertainty. Negative carry = futures trading at discount = strong bearish signal from smart money.")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Futures Price", _fmt(detail.futures_price, 0) if detail.futures_price else "—")
        cb.metric("Spot Close",    _fmt(detail.spot_close, 0)    if detail.spot_close    else "—")
        cc.metric("Carry (pts)",   _fmt(detail.carry_pts, 0, " pts") if detail.carry_pts is not None else "—")
        cd.metric("Carry % ann",   _fmt(detail.carry_pct_ann, 1, "%") if detail.carry_pct_ann is not None else "—")
        if detail.carry_label and detail.carry_label != "No Data":
            carry_color = (
                "#00C853" if "Bullish"  in detail.carry_label else
                "#FF5252" if "Bearish"  in detail.carry_label or "Backwardation" in detail.carry_label else
                "#FFD600"
            )
            st.markdown(
                f"<div style='color:{carry_color};font-weight:600;margin-top:4px'>"
                f"Carry Signal: {detail.carry_label}</div>",
                unsafe_allow_html=True,
            )

    with tab_compare:
        _render_comparison(detail)

    with tab_oi:
        _render_oi_chart(detail)

    with tab_ctx:
        _render_market_context(detail)

    with tab_regime:
        _render_regime_panel(detail)
