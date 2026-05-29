"""
FPI Capital Flow Radar — NSDL FPI investment tracking dashboard page.

Shows where Foreign Portfolio Investors are deploying capital:
  Equity vs Debt vs Hybrid allocation, risk appetite score,
  and a 15-20 day forward market outlook signal.

Data source: NSDL FPI archive (manual drop-folder — data/fpi_imports/).
"""
from __future__ import annotations

from datetime import date

import streamlit as st

__all__ = ["render"]


# ── Signal styling helpers ────────────────────────────────────────────────────

_SIGNAL_COLORS = {
    "STRONGLY BULLISH": ("#0a4f0a", "#d4edda"),
    "BULLISH":          ("#155724", "#d4edda"),
    "MILDLY BULLISH":   ("#1a5c1a", "#e8f5e9"),
    "NEUTRAL":          ("#383d41", "#e2e3e5"),
    "MILDLY BEARISH":   ("#7d2c00", "#ffecd2"),
    "BEARISH":          ("#721c24", "#f8d7da"),
    "STRONGLY BEARISH": ("#4a0000", "#f8d7da"),
    "N/A":              ("#383d41", "#e2e3e5"),
}

_SIGNAL_EMOJI = {
    "STRONGLY BULLISH": "🚀",
    "BULLISH":          "📈",
    "MILDLY BULLISH":   "↗️",
    "NEUTRAL":          "➡️",
    "MILDLY BEARISH":   "↘️",
    "BEARISH":          "📉",
    "STRONGLY BEARISH": "⚠️",
    "N/A":              "❓",
}


def _fmt_cr(val: float) -> str:
    """Format value in Crores with sign and scale."""
    sign = "+" if val >= 0 else "-"
    v = abs(val)
    if v >= 1_000:
        return f"{sign}₹{v/1000:.1f}K Cr"
    return f"{sign}₹{v:,.0f} Cr"


# ── Import guide ──────────────────────────────────────────────────────────────

_IMPORT_GUIDE = """
<div style='background:#f0f7ff;border:1px solid #b8d4f0;border-radius:8px;padding:14px 18px;font-size:0.85em;line-height:1.7'>
<b>How to import NSDL FPI data</b><br>
<ol style='margin:6px 0 0 0;padding-left:20px'>
  <li>Go to <b>https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx</b></li>
  <li>Select report type: <b>FPI Purchase / Sales / Net Investment</b></li>
  <li>Choose month range → click <b>Download</b> → save the Excel file</li>
  <li>Drop the <code>.xls</code> or <code>.xlsx</code> file into <code>data/fpi_imports/</code></li>
  <li>Run: <code>python -m src.cli import-fpi</code> (or click the button below)</li>
</ol>
<br>
<b>Tips:</b> Download one file per month (or multi-month range). Re-importing the same file is safe — rows are overwritten by date.
</div>
"""

# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_fpi_date_range,
        cached_fpi_summary,
        cached_fpi_category_breakdown,
        cached_fpi_risk_appetite,
        cached_fpi_15d_outlook,
        cached_fpi_available_dates,
    )

    st.markdown(
        "<h2 title='Foreign Portfolio Investor capital allocation tracker — equity vs debt vs hybrid flows from NSDL monthly archive data' "
        "style='cursor:help'>🌍 FPI Capital Flow Radar</h2>",
        unsafe_allow_html=True,
    )

    # ── Data status bar ───────────────────────────────────────────────────────
    min_d, max_d = cached_fpi_date_range()
    available_dates = cached_fpi_available_dates()
    n_days = len(available_dates)

    if n_days == 0:
        st.warning(
            "No FPI data in database. Import NSDL Excel files using the guide below."
        )
        st.markdown(_IMPORT_GUIDE, unsafe_allow_html=True)
        _render_import_button()
        return

    min_label = min_d.strftime('%d %b %Y') if min_d else "—"
    max_label = max_d.strftime('%d %b %Y') if max_d else "—"
    st.markdown(
        f"<div style='color:#555;font-size:0.82em;margin-bottom:8px'>"
        f"Data: <b>{min_label}</b> → <b>{max_label}</b> "
        f"&nbsp;|&nbsp; <b>{n_days}</b> trading days loaded"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Use latest available date if selected_date has no FPI data
    effective_date = selected_date if selected_date in available_dates else (max_d or selected_date)
    if effective_date != selected_date:
        st.caption(
            f"No FPI data for {selected_date.strftime('%d %b %Y')} — showing latest: "
            f"{effective_date.strftime('%d %b %Y')}"
        )

    # ── 15-day outlook ────────────────────────────────────────────────────────
    outlook = cached_fpi_15d_outlook(effective_date)
    signal  = outlook.get("signal", "N/A")
    fg, bg  = _SIGNAL_COLORS.get(signal, ("#383d41", "#e2e3e5"))
    emoji   = _SIGNAL_EMOJI.get(signal, "")

    st.markdown(
        f"<div style='background:{bg};border-left:5px solid {fg};border-radius:6px;"
        f"padding:10px 16px;margin-bottom:12px'>"
        f"<span style='font-size:1.1em;font-weight:700;color:{fg}'>"
        f"{emoji} 15-20 Day Market Outlook: {signal}</span><br>"
        f"<span style='color:{fg};font-size:0.88em'>{outlook.get('rationale','')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    eq15   = outlook.get("equity_15d", 0.0)
    dbt15  = outlook.get("debt_15d", 0.0)
    ra_pct = outlook.get("risk_score", 0.0)
    cf     = outlook.get("capital_flight", False)
    n_pts  = outlook.get("days_of_data", 0)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        eq_color = "#155724" if eq15 >= 0 else "#721c24"
        st.markdown(
            f"<div title='Cumulative FPI equity net investment over last {n_pts} sessions. Positive = buying, Negative = selling.' "
            f"style='cursor:help;background:#f8f9fa;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:0.75em;color:#666'>Equity Flow ({n_pts}d)</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:{eq_color}'>{_fmt_cr(eq15)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with k2:
        dbt_color = "#155724" if dbt15 >= 0 else "#721c24"
        st.markdown(
            f"<div title='Cumulative FPI debt (Debt + Debt-VRR) net investment over last {n_pts} sessions. Positive = inflow into debt instruments.' "
            f"style='cursor:help;background:#f8f9fa;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:0.75em;color:#666'>Debt Flow ({n_pts}d)</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:{dbt_color}'>{_fmt_cr(dbt15)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with k3:
        ra_color = "#155724" if ra_pct > 40 else "#721c24" if ra_pct < 10 else "#383d41"
        ra_label = "Risk-On" if ra_pct > 40 else "Risk-Off" if ra_pct < 10 else "Mixed"
        st.markdown(
            f"<div title='Risk Appetite Score = Equity Net / (|Equity| + |Debt| + |Hybrid|) × 100. "
            f"High % = FPIs prefer equity (risk assets). Low / negative = defensive shift to debt.' "
            f"style='cursor:help;background:#f8f9fa;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:0.75em;color:#666'>Risk Appetite Score</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:{ra_color}'>{ra_pct:+.1f}%</div>"
            f"<div style='font-size:0.7em;color:{ra_color}'>{ra_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with k4:
        cf_color = "#721c24" if cf else "#155724"
        cf_text  = "YES ⚠️" if cf else "NO ✓"
        st.markdown(
            f"<div title='Capital Flight = FPIs selling BOTH equity AND debt simultaneously. "
            f"Signals a broad risk-off exit from Indian markets — historically precedes sharp corrections.' "
            f"style='cursor:help;background:#f8f9fa;border-radius:8px;padding:12px;text-align:center'>"
            f"<div style='font-size:0.75em;color:#666'>Capital Flight Risk</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:{cf_color}'>{cf_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_trend, tab_alloc, tab_ra, tab_data, tab_import = st.tabs([
        "📈 Equity vs Debt Trend",
        "🥧 Allocation Breakdown",
        "🎯 Risk Appetite Timeline",
        "📋 Raw Data",
        "📥 Import Guide",
    ])

    with tab_trend:
        _render_trend(effective_date, cached_fpi_summary)

    with tab_alloc:
        _render_allocation(effective_date, cached_fpi_category_breakdown)

    with tab_ra:
        _render_risk_appetite(effective_date, cached_fpi_risk_appetite)

    with tab_data:
        _render_raw_data(effective_date, cached_fpi_summary)

    with tab_import:
        st.markdown(_IMPORT_GUIDE, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _render_import_button()


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _render_trend(as_of_date: date, cached_fpi_summary) -> None:
    import plotly.graph_objects as go

    st.markdown(
        "<h4 title='Daily equity vs debt net investment flow. Bars above 0 = FPI buying; below 0 = FPI selling. "
        "Sustained equity buying (green bars) is bullish for Nifty; sustained selling (red) is bearish.' "
        "style='cursor:help'>Equity vs Debt Net Investment (Daily)</h4>",
        unsafe_allow_html=True,
    )

    lookback = st.radio(
        "Window", [60, 90, 180], index=1,
        format_func=lambda d: f"{d} cal days",
        horizontal=True, key="fpi_trend_window",
    )

    df = cached_fpi_summary(as_of_date, lookback)
    if df.empty:
        st.info("No data for selected window.")
        return

    eq_df   = df[df["category"] == "Equity"].copy()
    debt_df = df[df["category"].isin(["Debt", "Debt-VRR"])].groupby("trade_date")["net_investment_cr"].sum().reset_index()
    debt_df.columns = ["trade_date", "net_investment_cr"]

    fig = go.Figure()

    # Equity bars — green/red
    if not eq_df.empty:
        colors = ["#2d8a4e" if v >= 0 else "#c0392b" for v in eq_df["net_investment_cr"]]
        fig.add_trace(go.Bar(
            x=eq_df["trade_date"], y=eq_df["net_investment_cr"],
            name="Equity Net", marker_color=colors,
            hovertemplate="%{x|%d %b %Y}<br>Equity Net: ₹%{y:+,.0f} Cr<extra></extra>",
        ))

    # Debt line
    if not debt_df.empty:
        fig.add_trace(go.Scatter(
            x=debt_df["trade_date"], y=debt_df["net_investment_cr"],
            name="Debt Net (Debt + VRR)", mode="lines",
            line=dict(color="#2980b9", width=2, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>Debt Net: ₹%{y:+,.0f} Cr<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="solid", line_color="#888", line_width=1)
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=None, yaxis_title="Net Investment (₹ Cr)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_allocation(as_of_date: date, cached_fpi_category_breakdown) -> None:
    import plotly.graph_objects as go

    st.markdown(
        "<h4 title='Cumulative FPI net investment breakdown by asset class over the selected window. "
        "A high equity share confirms risk-on appetite; a shift toward debt/hybrid signals defensive positioning.' "
        "style='cursor:help'>Category-Wise Cumulative Flow</h4>",
        unsafe_allow_html=True,
    )

    window = st.radio(
        "Window", [15, 30, 60, 90], index=1,
        format_func=lambda d: f"{d} cal days",
        horizontal=True, key="fpi_alloc_window",
    )

    df = cached_fpi_category_breakdown(as_of_date, window)
    if df.empty:
        st.info("No data.")
        return

    col_bar, col_pie = st.columns([3, 2])

    _CAT_COLORS = {
        "Equity":   "#2d8a4e",
        "Debt":     "#2980b9",
        "Debt-VRR": "#1a5276",
        "Hybrid":   "#8e44ad",
        "Others":   "#7f8c8d",
    }

    with col_bar:
        df_sorted = df.sort_values("net_investment_cr", ascending=True)
        colors = [_CAT_COLORS.get(c, "#aaa") for c in df_sorted["category"]]
        fig_bar = go.Figure(go.Bar(
            x=df_sorted["net_investment_cr"],
            y=df_sorted["category"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Net: ₹%{x:+,.0f} Cr<extra></extra>",
        ))
        fig_bar.add_vline(x=0, line_color="#888", line_width=1)
        fig_bar.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Net Investment (₹ Cr)",
            plot_bgcolor="#fafafa", paper_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        # Pie of absolute flows (gross purchase) to show where FPIs are most active
        df_abs = df[df["gross_purchase_cr"] > 0].copy()
        if not df_abs.empty:
            fig_pie = go.Figure(go.Pie(
                labels=df_abs["category"],
                values=df_abs["gross_purchase_cr"],
                marker_colors=[_CAT_COLORS.get(c, "#aaa") for c in df_abs["category"]],
                hole=0.4,
                hovertemplate="%{label}<br>Gross Buy: ₹%{value:,.0f} Cr<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Gross Purchase Share", font=dict(size=12)),
                showlegend=True,
                legend=dict(orientation="v", font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Summary table
    display = df[["category", "gross_purchase_cr", "gross_sales_cr", "net_investment_cr", "net_pct"]].copy()
    display.columns = ["Category", "Gross Buy (Cr)", "Gross Sell (Cr)", "Net (Cr)", "Net Share %"]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category":        st.column_config.TextColumn("Category"),
            "Gross Buy (Cr)":  st.column_config.NumberColumn("Gross Buy (₹Cr)", format="%.0f"),
            "Gross Sell (Cr)": st.column_config.NumberColumn("Gross Sell (₹Cr)", format="%.0f"),
            "Net (Cr)":        st.column_config.NumberColumn("Net (₹Cr)", format="%+.0f",
                help="Positive = net inflow; Negative = net outflow."),
            "Net Share %":     st.column_config.NumberColumn("Net Share %", format="%+.1f",
                help="This category's share of total FPI net allocation. High equity % = risk-on."),
        },
    )


def _render_risk_appetite(as_of_date: date, cached_fpi_risk_appetite) -> None:
    import plotly.graph_objects as go

    st.markdown(
        "<h4 title='Risk Appetite Score = Equity Net / (|Equity| + |Debt| + |Hybrid|) × 100. "
        "Score > 50% = FPIs strongly preferring equity (risk-on). Score < 10% = risk-off shift to debt. "
        "Sustained score above 50% historically precedes market rallies.' "
        "style='cursor:help'>FPI Risk Appetite Score Over Time</h4>",
        unsafe_allow_html=True,
    )

    lookback = st.radio(
        "Window", [60, 90, 180], index=1,
        format_func=lambda d: f"{d} cal days",
        horizontal=True, key="fpi_ra_window",
    )

    df = cached_fpi_risk_appetite(as_of_date, lookback)
    if df.empty:
        st.info("No risk appetite data.")
        return

    fig = go.Figure()

    # Filled area for risk score
    fig.add_trace(go.Scatter(
        x=df["trade_date"], y=df["risk_score"],
        name="Risk Appetite Score (%)",
        fill="tozeroy",
        fillcolor="rgba(45,138,78,0.15)",
        line=dict(color="#2d8a4e", width=2),
        hovertemplate="%{x|%d %b %Y}<br>Risk Score: %{y:+.1f}%<extra></extra>",
    ))

    # Reference bands
    fig.add_hrect(y0=50, y1=100, fillcolor="rgba(45,138,78,0.06)", line_width=0,
                  annotation_text="Risk-On Zone (>50%)", annotation_position="top right",
                  annotation=dict(font_size=10, font_color="#2d8a4e"))
    fig.add_hrect(y0=-100, y1=10, fillcolor="rgba(192,57,43,0.06)", line_width=0,
                  annotation_text="Risk-Off Zone (<10%)", annotation_position="bottom right",
                  annotation=dict(font_size=10, font_color="#c0392b"))
    fig.add_hline(y=0, line_color="#888", line_width=1, line_dash="dash")

    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Risk Appetite Score (%)",
        xaxis_title=None,
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 7-day rolling avg equity vs debt side-by-side
    c1, c2 = st.columns(2)
    with c1:
        if "equity_net" in df.columns:
            fig2 = go.Figure()
            df["eq_7d"] = df["equity_net"].rolling(7, min_periods=1).mean()
            colors = ["#2d8a4e" if v >= 0 else "#c0392b" for v in df["equity_net"]]
            fig2.add_trace(go.Bar(
                x=df["trade_date"], y=df["equity_net"],
                name="Equity Daily", marker_color=colors, opacity=0.5,
                hovertemplate="%{x|%d %b %Y}<br>₹%{y:+,.0f} Cr<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=df["trade_date"], y=df["eq_7d"],
                name="7d Avg", line=dict(color="#155724", width=2),
                hovertemplate="%{x|%d %b %Y}<br>7d Avg: ₹%{y:+,.0f} Cr<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color="#888", line_width=1)
            fig2.update_layout(
                height=240, margin=dict(l=0, r=0, t=20, b=0),
                title="Equity Net Flow", title_font_size=12,
                plot_bgcolor="#fafafa", paper_bgcolor="white",
                yaxis=dict(gridcolor="#eee"), showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        if "debt_net" in df.columns:
            fig3 = go.Figure()
            df["dbt_7d"] = df["debt_net"].rolling(7, min_periods=1).mean()
            dbt_colors = ["#2980b9" if v >= 0 else "#c0392b" for v in df["debt_net"]]
            fig3.add_trace(go.Bar(
                x=df["trade_date"], y=df["debt_net"],
                name="Debt Daily", marker_color=dbt_colors, opacity=0.5,
                hovertemplate="%{x|%d %b %Y}<br>₹%{y:+,.0f} Cr<extra></extra>",
            ))
            fig3.add_trace(go.Scatter(
                x=df["trade_date"], y=df["dbt_7d"],
                name="7d Avg", line=dict(color="#1a5276", width=2),
                hovertemplate="%{x|%d %b %Y}<br>7d Avg: ₹%{y:+,.0f} Cr<extra></extra>",
            ))
            fig3.add_hline(y=0, line_color="#888", line_width=1)
            fig3.update_layout(
                height=240, margin=dict(l=0, r=0, t=20, b=0),
                title="Debt Net Flow (Debt + VRR)", title_font_size=12,
                plot_bgcolor="#fafafa", paper_bgcolor="white",
                yaxis=dict(gridcolor="#eee"), showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)


def _render_raw_data(as_of_date: date, cached_fpi_summary) -> None:
    st.markdown(
        "<h4 title='Complete raw FPI flow data from NSDL. Each row = one trading day and category. '  "
        "style='cursor:help'>Complete FPI Flow Data</h4>",
        unsafe_allow_html=True,
    )

    lookback = st.radio(
        "Show last", [30, 60, 90, 180], index=1,
        format_func=lambda d: f"{d} cal days",
        horizontal=True, key="fpi_raw_window",
    )

    df = cached_fpi_summary(as_of_date, lookback)
    if df.empty:
        st.info("No data for selected window.")
        return

    import pandas as pd
    display = df.copy()
    display["trade_date"] = pd.to_datetime(display["trade_date"])

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "trade_date":        st.column_config.DateColumn("Date", format="DD MMM YY",
                help="Trading date"),
            "category":          st.column_config.TextColumn("Category",
                help="Asset class: Equity (stocks), Debt (bonds), Debt-VRR (variable rate repo), Hybrid, Others"),
            "gross_purchase_cr": st.column_config.NumberColumn("Gross Buy (₹Cr)", format="%.0f",
                help="Total FPI purchases in ₹ Crores"),
            "gross_sales_cr":    st.column_config.NumberColumn("Gross Sell (₹Cr)", format="%.0f",
                help="Total FPI sales in ₹ Crores"),
            "net_investment_cr": st.column_config.NumberColumn("Net (₹Cr)", format="%+.0f",
                help="Net investment = Gross Buy - Gross Sell. Positive = net inflow into India."),
        },
    )
    st.caption(f"{len(df):,} rows | {df['trade_date'].nunique()} dates | {df['category'].nunique()} categories")


def _render_import_button() -> None:
    """Button to trigger import-fpi via CLI subprocess (keeps ingestion out of dashboard)."""
    import subprocess
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    st.markdown("**Quick Import** (runs `python -m src.cli import-fpi`)")
    if st.button("📥 Import FPI Files Now", type="primary"):
        with st.spinner("Scanning data/fpi_imports/ and importing..."):
            proc = subprocess.run(
                [sys.executable, "-m", "src.cli", "import-fpi"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            st.success(output or "Import complete.")
            st.cache_data.clear()
        else:
            st.error(f"Import failed (exit {proc.returncode}):\n{output}")
