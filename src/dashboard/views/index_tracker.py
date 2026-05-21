"""
Index Tracker — momentum dashboard for all major NSE indices.

Shows:
  1. KPI cards — Nifty 50, Bank Nifty, VIX
  2. Snapshot table — all tracked indices with 1D/1W/1M/3M returns + vs Nifty50
  3. Trend chart — OHLC line for any selected index
  4. Comparison chart — any index vs Nifty 50 (normalised to 100)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.constants import POSITIVE_COLOR, NEGATIVE_COLOR, PLOT_BG, PAPER_BG, GRID_COLOR
from src.dashboard.cache.queries import (
    cached_index_snapshot,
    cached_index_history,
    cached_index_heatmap,
)
from src.analytics.index_momentum import TRACKED_INDICES


# ── Colour helpers ────────────────────────────────────────────────────────────

def _pct_color(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "#888"
    return POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR


def _fmt_pct(v, plus: bool = True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    prefix = "+" if plus and v > 0 else ""
    return f"{prefix}{v:.2f}%"


# ── KPI card ──────────────────────────────────────────────────────────────────

def _kpi(label: str, close: float | None, chg_1d: float | None,
         chg_1w: float | None, chg_1m: float | None) -> None:
    c1d = _pct_color(chg_1d)
    c1w = _pct_color(chg_1w)
    c1m = _pct_color(chg_1m)
    close_s = f"{close:,.2f}" if close else "—"
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04);border-radius:8px;"
        f"padding:14px 18px;text-align:center'>"
        f"<div style='font-size:11px;color:#888;text-transform:uppercase;"
        f"letter-spacing:1px;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700'>{close_s}</div>"
        f"<div style='margin-top:6px;display:flex;justify-content:center;gap:14px;font-size:12px'>"
        f"<span>1D <b style='color:{c1d}'>{_fmt_pct(chg_1d)}</b></span>"
        f"<span>1W <b style='color:{c1w}'>{_fmt_pct(chg_1w)}</b></span>"
        f"<span>1M <b style='color:{c1m}'>{_fmt_pct(chg_1m)}</b></span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ── Snapshot table ────────────────────────────────────────────────────────────

def _snapshot_table(snap: pd.DataFrame) -> None:
    if snap.empty:
        st.info("No index data for this date. Run: `python -m src.cli backfill-indices 120`")
        return

    rows_html = ""
    for _, r in snap.iterrows():
        def _cell(v, plus=True):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "<td style='padding:5px 10px;text-align:right;color:#555'>—</td>"
            color = _pct_color(v)
            return (f"<td style='padding:5px 10px;text-align:right;"
                    f"color:{color};font-weight:600'>{_fmt_pct(v, plus)}</td>")

        def _plain(v, fmt="{:.2f}"):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "<td style='padding:5px 10px;text-align:right;color:#555'>—</td>"
            return f"<td style='padding:5px 10px;text-align:right'>{fmt.format(v)}</td>"

        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
            f"<td style='padding:5px 10px;font-size:13px;font-weight:500'>{r['index_name']}</td>"
            f"{_plain(r.get('close_val'), '{:,.2f}')}"
            + _cell(r.get("pct_chg_1d"))
            + _cell(r.get("ret_1w"))
            + _cell(r.get("ret_1m"))
            + _cell(r.get("ret_3m"))
            + _cell(r.get("vs_nifty50"))
            + _plain(r.get("pe_ratio"), "{:.1f}x")
            + "</tr>"
        )

    headers = ["Index", "Close", "1D %", "1W %", "1M %", "3M %", "vs Nifty50 (1M)", "P/E"]
    header_html = "".join(
        f"<th style='padding:6px 10px;font-size:11px;color:rgba(255,255,255,0.45);"
        f"font-weight:600;text-transform:uppercase;letter-spacing:0.5px;"
        f"text-align:{'left' if i==0 else 'right'}'>{h}</th>"
        for i, h in enumerate(headers)
    )

    st.markdown(
        f"<div style='overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='border-bottom:2px solid rgba(255,255,255,0.12)'>"
        f"{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# ── Line chart ────────────────────────────────────────────────────────────────

def _trend_chart(hist: pd.DataFrame, index_name: str) -> go.Figure:
    if hist.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["trade_date"], y=hist["close_val"],
        mode="lines", name=index_name,
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.06)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: %{y:,.2f}<extra></extra>",
    ))

    # Add daily % change as bar on secondary axis
    if "pct_chg" in hist.columns:
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
                  for v in hist["pct_chg"].fillna(0)]
        fig.add_trace(go.Bar(
            x=hist["trade_date"], y=hist["pct_chg"],
            name="Daily %", marker_color=colors,
            opacity=0.55, yaxis="y2",
            hovertemplate="<b>%{x|%d %b}</b><br>Change: %{y:+.2f}%<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="Index Value", showgrid=True, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Daily %", overlaying="y", side="right",
                    showgrid=False, zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.2)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.08, x=0),
        height=360, margin=dict(t=20, b=40, l=60, r=60),
        hovermode="x unified",
    )
    return fig


# ── Comparison chart ──────────────────────────────────────────────────────────

def _comparison_chart(
    hist_a: pd.DataFrame, name_a: str,
    hist_b: pd.DataFrame, name_b: str,
) -> go.Figure:
    """Normalise both series to 100 at the start and overlay them."""
    fig = go.Figure()
    for hist, name, color in [
        (hist_a, name_a, "#4fc3f7"),
        (hist_b, name_b, "#ff8f00"),
    ]:
        if hist.empty:
            continue
        base = hist["close_val"].iloc[0]
        if not base:
            continue
        normed = hist["close_val"] / base * 100
        fig.add_trace(go.Scatter(
            x=hist["trade_date"], y=normed,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d %b %Y}}<br>Normalised: %{{y:.2f}}<extra></extra>",
        ))

    fig.add_hline(y=100, line_color="rgba(255,255,255,0.25)", line_dash="dot")
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="Indexed to 100", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", y=1.08, x=0),
        height=340, margin=dict(t=20, b=40, l=60, r=20),
        hovermode="x unified",
    )
    return fig


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _heatmap(heat: pd.DataFrame) -> go.Figure:
    if heat.empty:
        return go.Figure()

    heat = heat.dropna(subset=["pct_chg"]).copy()
    heat["pct_chg"] = pd.to_numeric(heat["pct_chg"], errors="coerce")
    heat = heat.dropna(subset=["pct_chg"]).sort_values("pct_chg", ascending=False)

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in heat["pct_chg"]]

    fig = go.Figure(go.Bar(
        x=heat["pct_chg"],
        y=heat["index_name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in heat["pct_chg"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.4)",
                   ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9)),
        height=max(400, len(heat) * 18 + 80),
        margin=dict(t=10, b=40, l=220, r=80),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    st.title("Index Tracker")
    st.caption(
        "Live momentum tracker for all major NSE indices. "
        "Returns computed from NiftyIndices daily snapshots."
    )

    snap = cached_index_snapshot(selected_date)

    if snap.empty:
        st.warning(
            f"No index data found for {selected_date.strftime('%d %b %Y')}. "
            "Run the following command to backfill historical index data:\n\n"
            "```\npython -m src.cli backfill-indices 120\n```"
        )
        return

    def _get(name: str, col: str):
        row = snap[snap["index_name"] == name]
        if row.empty:
            return None
        v = row.iloc[0].get(col)
        return None if pd.isna(v) else float(v)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_list = [
        ("Nifty 50",        k1),
        ("Nifty Bank",      k2),
        ("Nifty IT",        k3),
        ("Nifty Midcap 50", k4),
        ("India VIX",       k5),
    ]
    for idx_name, col in kpi_list:
        with col:
            _kpi(
                idx_name,
                _get(idx_name, "close_val"),
                _get(idx_name, "pct_chg_1d"),
                _get(idx_name, "ret_1w"),
                _get(idx_name, "ret_1m"),
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_snap, tab_chart, tab_compare, tab_heat = st.tabs([
        "📊 Snapshot Table",
        "📈 Index Chart",
        "🔀 Compare Indices",
        "🌡️ Daily Heatmap",
    ])

    with tab_snap:
        st.caption(
            "All major indices — 1D/1W/1M/3M returns. "
            "vs Nifty50 = index 1M return minus Nifty 50 1M return (outperformance)."
        )
        _snapshot_table(snap)

    with tab_chart:
        col_idx, col_lb = st.columns([3, 1])
        with col_idx:
            idx_choice = st.selectbox(
                "Select Index",
                options=[r for r in TRACKED_INDICES if r in snap["index_name"].values],
                key="it_idx",
            )
        with col_lb:
            lb = st.selectbox("Lookback", options=[30, 60, 90, 120], index=1,
                              key="it_lb")

        hist = cached_index_history(idx_choice, selected_date, lb)
        if hist.empty:
            st.info("No history data yet.")
        else:
            # Summary row
            s1, s2, s3, s4 = st.columns(4)
            row = snap[snap["index_name"] == idx_choice]
            if not row.empty:
                r = row.iloc[0]
                s1.metric("Close",    f"{r['close_val']:,.2f}"  if pd.notna(r.get('close_val')) else "—")
                s2.metric("1D",       _fmt_pct(r.get("pct_chg_1d")))
                s3.metric("1M",       _fmt_pct(r.get("ret_1m")))
                s4.metric("vs Nifty", _fmt_pct(r.get("vs_nifty50")),
                          help="Outperformance vs Nifty 50 over 1 month")
            st.plotly_chart(_trend_chart(hist, idx_choice), use_container_width=True)

    with tab_compare:
        st.caption(
            "Normalised comparison — both indices set to 100 at the start of the period. "
            "Shows relative outperformance visually."
        )
        all_idx = [r for r in TRACKED_INDICES if r in snap["index_name"].values]
        cc1, cc2, cc3 = st.columns([3, 3, 1])
        with cc1:
            idx_a = st.selectbox("Index A", options=all_idx,
                                 index=0, key="cmp_a")
        with cc2:
            b_options = [i for i in all_idx if i != idx_a]
            idx_b = st.selectbox("Index B", options=b_options,
                                 index=b_options.index("Nifty 50") if "Nifty 50" in b_options else 0,
                                 key="cmp_b")
        with cc3:
            lb2 = st.selectbox("Lookback", [30, 60, 90, 120], index=1, key="cmp_lb")

        hist_a = cached_index_history(idx_a, selected_date, lb2)
        hist_b = cached_index_history(idx_b, selected_date, lb2)
        fig = _comparison_chart(hist_a, idx_a, hist_b, idx_b)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)

            # Outperformance stat
            if not hist_a.empty and not hist_b.empty and len(hist_a) > 1 and len(hist_b) > 1:
                ret_a = (hist_a["close_val"].iloc[-1] - hist_a["close_val"].iloc[0]) / hist_a["close_val"].iloc[0] * 100
                ret_b = (hist_b["close_val"].iloc[-1] - hist_b["close_val"].iloc[0]) / hist_b["close_val"].iloc[0] * 100
                diff  = ret_a - ret_b
                winner = idx_a if diff >= 0 else idx_b
                st.caption(
                    f"**{idx_a}**: {ret_a:+.2f}%  |  "
                    f"**{idx_b}**: {ret_b:+.2f}%  |  "
                    f"Outperformer: **{winner}** by {abs(diff):.2f}%"
                )

    with tab_heat:
        heat = cached_index_heatmap(selected_date)
        if heat.empty:
            st.info("No heatmap data.")
        else:
            st.caption(f"All {len(heat)} indices ranked by today's % change.")
            fig_heat = _heatmap(heat)
            if fig_heat.data:
                st.plotly_chart(fig_heat, use_container_width=True)
