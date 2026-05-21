"""
Big Players F&O Tracker — institutional positioning via participant-wise OI & Volume.

Answers:
  • Are FII buying or shorting Index Futures? (direction of market)
  • Are DII counterbalancing? (long-term support / distribution)
  • Is retail (Client) trapped on the wrong side? (contrarian signal)
  • Who is buying Calls vs Puts? (options market positioning)
  • What does PCR say? (put-call ratio as contrarian indicator)

Data source: NSE F&O participant-wise OI and Volume CSVs (fao_participant_oi_*.csv)
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_fao_available_dates,
    cached_fao_cumulative,
    cached_fao_daily,
    cached_fao_latest,
    cached_fii_stats_history,
    cached_fii_stats_latest,
    cached_market_intelligence,
)
from src.dashboard.constants import GRID_COLOR, NEGATIVE_COLOR, PAPER_BG, PLOT_BG, POSITIVE_COLOR

# ── Participant colours consistent throughout the page ────────────────────────
_COLORS = {
    "FII":    "#2196f3",   # blue   — dominant institutional player
    "DII":    "#4caf50",   # green  — domestic institutions (LIC, MFs)
    "Client": "#ff9800",   # orange — retail
    "Pro":    "#9c27b0",   # purple — proprietary desks
}
_ORDER = ["FII", "DII", "Client", "Pro"]


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


def _signal_box(fii_row: pd.Series | None, dii_row: pd.Series | None) -> None:
    """Plain-English market signal based on FII + DII Index Futures net."""
    if fii_row is None:
        st.info("No F&O data loaded yet. Run backfill to load history.")
        return

    fii_net = float(fii_row.get("fut_idx_net", 0) or 0)
    dii_net = float(dii_row.get("fut_idx_net", 0) or 0) if dii_row is not None else 0

    fii_bull = fii_net > 0
    dii_bull = dii_net > 0

    if fii_bull and dii_bull:
        signal = "🟢 STRONG BULLISH"
        color  = "#00c853"
        msg = (
            f"Both FII ({_fmt_contracts(fii_net)} contracts) and "
            f"DII ({_fmt_contracts(dii_net)} contracts) are net LONG in Index Futures. "
            "All institutional money aligned upward — highest conviction bullish setup."
        )
    elif fii_bull and not dii_bull:
        signal = "🟩 BULLISH (FII-led)"
        color  = "#69f0ae"
        msg = (
            f"FII net long {_fmt_contracts(fii_net)} contracts. "
            f"DII net short {_fmt_contracts(dii_net)} — possibly hedging long equity portfolios. "
            "FII driving market direction; watch DII for reversal signal."
        )
    elif not fii_bull and dii_bull:
        signal = "🟨 MIXED — FII Selling, DII Supporting"
        color  = "#ffca28"
        msg = (
            f"FII net short {_fmt_contracts(fii_net)} contracts — bearish directional bet. "
            f"DII net long {_fmt_contracts(dii_net)} — providing floor support. "
            "Market range-bound; FII exit = selling pressure, DII buy = downside limited."
        )
    else:
        signal = "🔴 BEARISH"
        color  = "#ff5252"
        msg = (
            f"FII net short {_fmt_contracts(fii_net)} contracts AND "
            f"DII net short {_fmt_contracts(dii_net)} contracts. "
            "Both institutional players positioned for downside — avoid leveraged longs."
        )

    st.markdown(
        f"<div style='background:rgba(255,255,255,0.04);border-left:4px solid {color};"
        f"padding:10px 16px;border-radius:0 8px 8px 0;margin-bottom:8px'>"
        f"<span style='font-size:16px;font-weight:700;color:{color}'>{signal}</span>"
        f"<div style='font-size:13px;color:rgba(255,255,255,0.75);margin-top:4px'>{msg}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _kpi_cards(latest: pd.DataFrame) -> None:
    """4 participant KPI cards: Index Futures Long / Short / Net / L/S%."""
    cols = st.columns(4)
    for i, ptype in enumerate(_ORDER):
        row = latest[latest["client_type"] == ptype]
        if row.empty:
            cols[i].metric(ptype, "—")
            continue
        r    = row.iloc[0]
        net  = int(r.get("fut_idx_net", 0) or 0)
        lspct = r.get("fut_idx_ls_pct")
        ls_str = f"{lspct:.1f}% L/S" if lspct is not None and not pd.isna(lspct) else "—"
        color = _COLORS[ptype]
        net_c = "#00c853" if net > 0 else ("#ff5252" if net < 0 else "#888")

        cols[i].markdown(
            f"<div style='border:1px solid {color}33;border-radius:8px;padding:10px 14px;"
            f"background:{color}0d'>"
            f"<div style='font-size:11px;color:{color};font-weight:600;letter-spacing:.5px'>"
            f"{ptype} — Index Futures</div>"
            f"<div style='margin-top:6px;display:flex;gap:12px;font-size:12px;"
            f"color:rgba(255,255,255,0.6)'>"
            f"<span>Long: <b style='color:rgba(255,255,255,0.9)'>"
            f"{int(r.get('fut_idx_long', 0) or 0):,}</b></span>"
            f"<span>Short: <b style='color:rgba(255,255,255,0.9)'>"
            f"{int(r.get('fut_idx_short', 0) or 0):,}</b></span></div>"
            f"<div style='margin-top:4px;font-size:20px;font-weight:700;color:{net_c}'>"
            f"{net:+,}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.45)'>{ls_str}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _options_kpi_cards(latest: pd.DataFrame) -> None:
    """4 participant KPI cards: Index Options Call Net / Put Net / Delta proxy."""
    cols = st.columns(4)
    for i, ptype in enumerate(_ORDER):
        row = latest[latest["client_type"] == ptype]
        if row.empty:
            cols[i].metric(ptype, "—")
            continue
        r        = row.iloc[0]
        call_net = int(r.get("opt_idx_call_net", 0) or 0)
        put_net  = int(r.get("opt_idx_put_net",  0) or 0)
        delta    = int(r.get("opt_idx_net",       0) or 0)
        color    = _COLORS[ptype]
        call_c   = "#00c853" if call_net > 0 else ("#ff5252" if call_net < 0 else "#888")
        put_c    = "#ff5252" if put_net  > 0 else ("#00c853" if put_net  < 0 else "#888")
        delta_c  = "#00c853" if delta    > 0 else ("#ff5252" if delta    < 0 else "#888")

        cols[i].markdown(
            f"<div style='border:1px solid {color}33;border-radius:8px;padding:10px 14px;"
            f"background:{color}0d'>"
            f"<div style='font-size:11px;color:{color};font-weight:600;letter-spacing:.5px'>"
            f"{ptype} — Index Options</div>"
            f"<div style='margin-top:6px;display:flex;gap:12px;font-size:11px;"
            f"color:rgba(255,255,255,0.6)'>"
            f"<span>Call: <b style='color:{call_c}'>{call_net:+,}</b></span>"
            f"<span>Put: <b style='color:{put_c}'>{put_net:+,}</b></span></div>"
            f"<div style='margin-top:2px;font-size:10px;color:rgba(255,255,255,0.35)'>"
            f"Options Delta (Call Net − Put Net)</div>"
            f"<div style='font-size:20px;font-weight:700;color:{delta_c}'>{delta:+,}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _pcr_metric(latest: pd.DataFrame) -> None:
    """Put-Call Ratio computed from latest snapshot across all participants."""
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
            "<b style='color:rgba(255,255,255,0.8)'>How to read PCR (Put-Call Ratio = Total Put OI ÷ Total Call OI):</b><br><br>"
            "<b style='color:#00c853'>&gt; 1.3 — Contrarian Bullish:</b> "
            "Excessive put-buying means everyone has already hedged downside. Markets rarely fall when participants are already protected. "
            "This is a buy signal — the bearish consensus is usually wrong at extremes.<br><br>"
            "<b style='color:#ffca28'>0.7 – 1.3 — Neutral:</b> "
            "Normal put/call balance. No strong contrarian directional signal from options market.<br><br>"
            "<b style='color:#ff5252'>&lt; 0.7 — Contrarian Bearish:</b> "
            "Everyone is buying calls, expecting the market to rise. Complacency at peaks. "
            "Smart money shorts into this euphoria — a fall often follows extreme call-buying."
            "</div>",
            unsafe_allow_html=True,
        )


def _cumulative_chart(cum: pd.DataFrame) -> go.Figure:
    """Line chart: cumulative Index Futures net per participant over time."""
    if cum.empty:
        return go.Figure()

    fig = go.Figure()

    for ptype in _ORDER:
        grp = cum[cum["client_type"] == ptype].sort_values("trade_date")
        if grp.empty:
            continue
        color = _COLORS[ptype]
        fig.add_trace(go.Scatter(
            x=grp["trade_date"],
            y=grp["cum_fut_idx_net"],
            name=ptype,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{ptype}</b><br>"
                "%{x|%d %b %Y}<br>"
                "Cumulative Net: <b>%{y:+,}</b> contracts"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=0, line_dash="dash", line_width=1.5,
        line_color="rgba(255,255,255,0.30)",
        annotation_text="Flat (zero net)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.4)"),
    )
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Cumulative Net Contracts", showgrid=True,
                   gridcolor=GRID_COLOR, tickformat=","),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=12)),
        height=380,
        margin=dict(t=30, b=50, l=80, r=40),
        hovermode="x unified",
    )
    return fig


def _cumulative_options_chart(cum: pd.DataFrame) -> go.Figure:
    """FII + DII cumulative options net: Call Net (solid) vs Put Net (dashed)."""
    if cum.empty or "cum_opt_idx_call_net" not in cum.columns:
        return go.Figure()

    fig = go.Figure()

    for ptype in ("FII", "DII"):
        grp = cum[cum["client_type"] == ptype].sort_values("trade_date")
        if grp.empty:
            continue
        color = _COLORS[ptype]
        fig.add_trace(go.Scatter(
            x=grp["trade_date"],
            y=grp["cum_opt_idx_call_net"],
            name=f"{ptype} Call Net",
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{ptype} Call Net</b><br>"
                "%{x|%d %b}: <b>%{y:+,}</b> contracts<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=grp["trade_date"],
            y=grp["cum_opt_idx_put_net"],
            name=f"{ptype} Put Net",
            mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            hovertemplate=(
                f"<b>{ptype} Put Net</b><br>"
                "%{x|%d %b}: <b>%{y:+,}</b> contracts<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash", line_width=1.5,
                  line_color="rgba(255,255,255,0.30)")
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%b '%y", tickfont=dict(size=11)),
        yaxis=dict(title="Cumulative Net Contracts", showgrid=True,
                   gridcolor=GRID_COLOR, tickformat=","),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        height=320,
        margin=dict(t=40, b=40, l=80, r=40),
        hovermode="x unified",
    )
    return fig


def _pcr_chart(cum: pd.DataFrame) -> go.Figure:
    """Daily Put-Call Ratio trend (total put OI / total call OI across all participants)."""
    if cum.empty or "opt_idx_put_long" not in cum.columns:
        return go.Figure()

    daily_pcr = (
        cum.groupby("trade_date")[["opt_idx_put_long", "opt_idx_call_long"]]
        .sum()
        .reset_index()
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
        x=daily_pcr["trade_date"],
        y=daily_pcr["pcr"],
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
        height=260,
        margin=dict(t=30, b=40, l=60, r=140),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def _daily_net_chart(cum: pd.DataFrame, participant: str) -> go.Figure:
    """Bar chart: daily Index Futures net for one participant."""
    grp = cum[cum["client_type"] == participant].sort_values("trade_date").tail(60)
    if grp.empty:
        return go.Figure()

    colors = ["#00c853" if v >= 0 else "#ff5252" for v in grp["daily_fut_idx_net"]]
    color  = _COLORS.get(participant, "#4c78a8")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["trade_date"],
        y=grp["daily_fut_idx_net"],
        marker_color=colors,
        opacity=0.85,
        hovertemplate=(
            f"<b>{participant}</b><br>"
            "%{x|%d %b %Y}<br>"
            "Daily Net: <b>%{y:+,}</b> contracts"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=grp["trade_date"],
        y=grp["cum_fut_idx_net"],
        name="Cumulative Net",
        mode="lines",
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
            x=grp["trade_date"],
            y=grp["fut_idx_ls_pct"],
            name=f"{ptype} L/S%",
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{ptype}</b> %{{x|%d %b}}: L/S = <b>%{{y:.1f}}%</b>"
                "<extra></extra>"
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


def _daily_table(daily: pd.DataFrame) -> None:
    """Pivot table: rows = date, columns = participant × Futures metrics."""
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
        col_cfg[f"{pt}_Long"]  = st.column_config.NumberColumn(
            f"{pt} Long",  format="%,d",
            help=f"Total open long (buy) futures contracts held by {pt}. Contracts, not lots.",
        )
        col_cfg[f"{pt}_Short"] = st.column_config.NumberColumn(
            f"{pt} Short", format="%,d",
            help=f"Total open short (sell) futures contracts held by {pt}.",
        )
        col_cfg[f"{pt}_Net"]   = st.column_config.NumberColumn(
            f"{pt} Net",   format="%+,d",
            help=f"{pt} Net = Long − Short. Positive = net long (bullish bias). Negative = net short (bearish).",
        )
        col_cfg[f"{pt}_LS%"]   = st.column_config.NumberColumn(
            f"{pt} L/S%",  format="%.1f%%",
            help=f"Long ÷ (Long + Short) × 100 for {pt}. Above 50% = more longs than shorts.",
        )
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _options_daily_table(daily: pd.DataFrame) -> None:
    """Pivot table: rows = date, columns = participant × Options metrics."""
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
        col_cfg[f"{pt}_Call"]  = st.column_config.NumberColumn(
            f"{pt} Call Net", format="%+,d",
            help=f"{pt} Call Long − Call Short. Positive = net call buyer → bullish directional bet.",
        )
        col_cfg[f"{pt}_Put"]   = st.column_config.NumberColumn(
            f"{pt} Put Net",  format="%+,d",
            help=f"{pt} Put Long − Put Short. Positive = net put buyer → hedging or bearish bet.",
        )
        col_cfg[f"{pt}_Delta"] = st.column_config.NumberColumn(
            f"{pt} Opt Delta", format="%+,d",
            help=f"{pt} Call Net − Put Net. Positive = overall bullish options stance. Negative = bearish/hedged.",
        )
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _cumulative_table(cum: pd.DataFrame) -> None:
    """Cumulative Futures table: running totals per participant."""
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
        col_cfg[f"{pt}_CumNet"] = st.column_config.NumberColumn(
            f"{pt} Cum Net", format="%+,d",
            help=f"Cumulative running total of {pt} Net futures contracts since period start. Rising = accumulating longs. Falling = adding shorts.",
        )
        col_cfg[f"{pt}_LS%"]    = st.column_config.NumberColumn(
            f"{pt} L/S%",    format="%.1f%%",
            help=f"Long ÷ (Long + Short) × 100 for {pt} on this date. Above 50% = net long; below 50% = net short.",
        )
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _options_cumulative_table(cum: pd.DataFrame) -> None:
    """Cumulative Options table: running totals for Call/Put/Delta per participant."""
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
        col_cfg[f"{pt}_CumCall"]  = st.column_config.NumberColumn(
            f"{pt} Cum Call",  format="%+,d",
            help=f"Cumulative {pt} Call Net since period start. Rising = sustained call buying (bullish bets growing).",
        )
        col_cfg[f"{pt}_CumPut"]   = st.column_config.NumberColumn(
            f"{pt} Cum Put",   format="%+,d",
            help=f"Cumulative {pt} Put Net since period start. Rising = sustained put buying (hedging / bearish bets growing).",
        )
        col_cfg[f"{pt}_CumDelta"] = st.column_config.NumberColumn(
            f"{pt} Cum Delta", format="%+,d",
            help=f"Cumulative {pt} Options Delta (Cum Call − Cum Put). Positive = bullish net options stance over the period.",
        )
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg)


def _score_bar(score: float, max_score: float = 14.0) -> str:
    """Visual bar showing composite score -max to +max."""
    pct = max(0, min(100, int((score / max_score + 1) * 50)))
    if score >= 5:   bar_c = "#00c853"
    elif score >= 2: bar_c = "#69f0ae"
    elif score >= -1: bar_c = "#ffca28"
    elif score >= -4: bar_c = "#ffab40"
    else:            bar_c = "#ff5252"
    return (
        f"<div style='background:#1e1e1e;border-radius:4px;height:10px;overflow:hidden;margin:4px 0'>"
        f"<div style='width:{pct}%;height:100%;background:{bar_c};border-radius:4px'></div>"
        f"</div>"
    )


def _signal_badge(sig) -> str:
    """Compact coloured badge HTML for one signal."""
    if sig.direction > 0:    bg, tc = "#00c85322", "#00c853"
    elif sig.direction < 0:  bg, tc = "#ff525222", "#ff5252"
    else:                    bg, tc = "#88888822", "#888888"
    score_str = f"+{sig.score}" if sig.score > 0 else str(sig.score)
    return (
        f"<div style='border:1px solid {tc}55;border-radius:6px;padding:6px 10px;"
        f"background:{bg};margin:3px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-size:11px;color:{tc};font-weight:600'>"
        f"{sig.emoji} {sig.headline}</span>"
        f"<span style='font-size:11px;color:{tc};font-weight:700;"
        f"background:{tc}22;padding:1px 6px;border-radius:10px'>{score_str}</span>"
        f"</div>"
        f"<div style='font-size:10px;color:rgba(255,255,255,0.5);margin-top:3px;line-height:1.4'>"
        f"{sig.description[:160]}{'…' if len(sig.description) > 160 else ''}</div>"
        f"</div>"
    )


def _render_market_intelligence(selected_date: date) -> None:
    """Full Market Intelligence section — composite score, signals, expiry view."""
    from src.analytics.market_intelligence import MarketIntelligence

    with st.spinner("Computing market intelligence…"):
        mi = cached_market_intelligence(selected_date)

    if mi.market_view == "No Data":
        st.info("Run F&O backfill first to enable market intelligence signals.")
        return

    # ── TOMORROW'S VERDICT — shown first, most prominent ─────────────────────
    if mi.tomorrow_verdict:
        v = mi.tomorrow_verdict
        dir_icon = {"UP": "▲", "DOWN": "▼", "SIDEWAYS": "↔"}.get(v.direction, "?")
        conf_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "⚪"}.get(v.confidence, "")
        st.markdown(
            f"<div style='border:2px solid {v.direction_color}88;border-radius:12px;"
            f"padding:16px 20px;background:{v.direction_color}14;margin-bottom:12px'>"
            f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.45);"
            f"letter-spacing:1.5px;margin-bottom:8px'>TOMORROW'S EXPECTED MOVE</div>"
            f"<div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap'>"
            f"<div style='font-size:38px;font-weight:900;color:{v.direction_color};"
            f"letter-spacing:-1px;line-height:1'>{dir_icon} {v.direction}</div>"
            f"<div style='flex:1;min-width:200px'>"
            f"<div style='font-size:13px;font-weight:600;color:rgba(255,255,255,0.9);"
            f"margin-bottom:6px'>{v.headline}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.55)'>"
            f"<b style='color:rgba(255,255,255,0.7)'>Key Driver:</b> {v.key_driver}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.45);margin-top:3px'>"
            f"<b style='color:rgba(255,255,255,0.55)'>Key Risk:</b> {v.key_risk}</div>"
            f"</div>"
            f"<div style='text-align:center;min-width:80px'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4)'>CONFIDENCE</div>"
            f"<div style='font-size:15px;font-weight:700;color:{v.direction_color}'>"
            f"{conf_icon} {v.confidence}</div>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Composite Score Banner ────────────────────────────────────────────────
    view_c = mi.view_color
    c_score, c_view, c_expiry = st.columns([1, 2, 2])

    with c_score:
        st.markdown(
            f"<div style='border:1px solid {view_c}55;border-radius:10px;padding:14px 12px;"
            f"background:{view_c}11;text-align:center'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.5);letter-spacing:.5px'>"
            f"COMPOSITE SCORE</div>"
            f"<div style='font-size:36px;font-weight:800;color:{view_c};line-height:1.1'>"
            f"{mi.composite_score:+.0f}</div>"
            f"{_score_bar(mi.composite_score)}"
            f"<div style='font-size:12px;font-weight:700;color:{view_c};margin-top:4px'>"
            f"{mi.market_view}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_view:
        st.markdown(
            f"<div style='border:1px solid rgba(255,255,255,0.08);border-radius:10px;"
            f"padding:14px 12px;background:rgba(255,255,255,0.03);height:100%'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:.5px;margin-bottom:6px'>"
            f"BIAS REASONING</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.75);line-height:1.5'>"
            f"{mi.bias_reasoning}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with c_expiry:
        if mi.weekly_expiry:
            we = mi.weekly_expiry
            exp_c = "#ffca28" if we.days_to_expiry > 2 else "#ff9800"
            st.markdown(
                f"<div style='border:1px solid {exp_c}44;border-radius:10px;"
                f"padding:14px 12px;background:{exp_c}0a;height:100%'>"
                f"<div style='font-size:10px;color:{exp_c};letter-spacing:.5px;font-weight:600'>"
                f"WEEKLY EXPIRY ({we.expiry_date.strftime('%d %b')})</div>"
                f"<div style='font-size:13px;font-weight:700;color:{exp_c};margin:4px 0'>"
                f"{we.bias} &nbsp;·&nbsp; {we.days_to_expiry}D to go</div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.55);line-height:1.4'>"
                f"{we.reasoning}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Alerts ────────────────────────────────────────────────────────────────
    for alert in mi.alerts:
        st.warning(f"🚨 {alert}")

    # ── Signal Cards (sorted by priority = highest |score| first) ─────────────
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:rgba(255,255,255,0.4);"
        "letter-spacing:1px;margin:10px 0 6px'>SIGNAL BREAKDOWN — sorted by impact</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    for i, sig in enumerate(mi.signals):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(_signal_badge(sig), unsafe_allow_html=True)

    if mi.fii_flow_available:
        st.caption("FII Money Flow signal included (fii_derivatives_stats data available).")
    else:
        st.caption(
            "FII Money Flow signal not available — run `python -m src.cli backfill-fii-stats 365` "
            "to enable rupee-value signals."
        )


_FII_STATS_SECTION_ORDER = [
    ("INDEX FUTURES",      ["NIFTY FUTURES", "BANKNIFTY FUTURES", "FINNIFTY FUTURES",
                            "MIDCPNIFTY FUTURES", "NIFTYNXT50 FUTURES"]),
    ("INDEX OPTIONS",      ["NIFTY OPTIONS", "BANKNIFTY OPTIONS", "FINNIFTY OPTIONS",
                            "MIDCPNIFTY OPTIONS", "NIFTYNXT50 OPTIONS"]),
    ("STOCK FUTURES",      []),
    ("STOCK OPTIONS",      []),
]


def _fii_stats_table(fii_stats: pd.DataFrame) -> None:
    """Display FII Derivatives Statistics per-index breakdown."""
    if fii_stats.empty:
        st.info(
            "No FII Derivatives Statistics data available. "
            "Run: `python -m src.cli backfill-fii-stats 365`"
        )
        return

    data_date = fii_stats["trade_date"].iloc[0]
    st.caption(f"FII per-index buy/sell activity — {data_date.strftime('%d %b %Y')}")

    display = fii_stats.copy()
    display["net_value_cr"]  = display["buy_value_cr"] - display["sell_value_cr"]
    display["net_contracts"] = display["buy_contracts"] - display["sell_contracts"]

    col_cfg = {
        "category":       st.column_config.TextColumn("Index / Category",
            help="Contract category: Index Futures, Index Call/Put Options, Stock Futures, Stock Options, Total."),
        "buy_contracts":  st.column_config.NumberColumn("Buy Contracts",  format="%,d",
            help="Number of contracts bought by FIIs today."),
        "sell_contracts": st.column_config.NumberColumn("Sell Contracts", format="%,d",
            help="Number of contracts sold by FIIs today."),
        "net_contracts":  st.column_config.NumberColumn("Net Contracts",  format="%+,d",
            help="Buy − Sell contracts. Positive = FII net buyer. Large positive = strong institutional accumulation."),
        "buy_value_cr":   st.column_config.NumberColumn("Buy (Cr)",  format="%.2f",
            help="Rupee value of FII buys (₹ Crore). 1 Cr = ₹10 million."),
        "sell_value_cr":  st.column_config.NumberColumn("Sell (Cr)", format="%.2f",
            help="Rupee value of FII sells (₹ Crore)."),
        "net_value_cr":   st.column_config.NumberColumn("Net Flow (Cr)", format="%+.2f",
            help="Buy − Sell value in ₹ Crore. Positive = FII net buyer (money flowing in). Negative = net seller (outflow)."),
        "oi_contracts":   st.column_config.NumberColumn("OI Contracts", format="%,d",
            help="Total open interest held by FIIs in this category at end of day."),
        "trade_date":     None,
        "oi_value_cr":    None,
    }
    st.dataframe(
        display.sort_values("category"),
        hide_index=True,
        use_container_width=True,
        column_config=col_cfg,
        column_order=[
            "category", "buy_contracts", "sell_contracts", "net_contracts",
            "buy_value_cr", "sell_value_cr", "net_value_cr", "oi_contracts",
        ],
    )


def _fii_flow_chart(fii_stats_hist: pd.DataFrame) -> go.Figure:
    """Net FII flow by key index over time (bar chart)."""
    if fii_stats_hist.empty:
        return go.Figure()

    cats = ["NIFTY FUTURES", "BANKNIFTY FUTURES", "INDEX FUTURES"]
    cat_colors = {
        "NIFTY FUTURES":    "#2196f3",
        "BANKNIFTY FUTURES":"#ff9800",
        "INDEX FUTURES":    "#9c27b0",
    }

    fig = go.Figure()
    for cat in cats:
        grp = (
            fii_stats_hist[fii_stats_hist["category"] == cat]
            .sort_values("trade_date")
        )
        if grp.empty:
            continue
        bar_colors = ["#00c853" if v >= 0 else "#ff5252" for v in grp["net_value_cr"]]
        fig.add_trace(go.Bar(
            x=grp["trade_date"],
            y=grp["net_value_cr"],
            name=cat,
            marker_color=bar_colors if cat == "Index Futures" else cat_colors[cat],
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


def render(selected_date: date) -> None:
    st.subheader("🎯 Big Players — F&O Participant Activity")

    # ── How to Read ───────────────────────────────────────────────────────────
    with st.expander("📖 How to Read This Page", expanded=False):
        st.markdown("""
**Who are the participants?**
| Participant | Who they are | Why they matter |
|-------------|-------------|-----------------|
| **FII** | Foreign Institutional Investors (FPIs, foreign funds) | Largest movers — their net position direction often drives the index |
| **DII** | Domestic Institutional Investors (mutual funds, insurance, LIC) | Counter-balance to FII; accumulate on dips |
| **Client** | Retail & HNI traders | Often contrarian indicator — usually wrong at extremes |
| **Pro** | Proprietary desks (brokers trading own capital) | Informed short-term traders; follow their direction |

---

**Futures columns**
| Column | Meaning | Signal |
|--------|---------|--------|
| **Long** | Open long contracts held | — |
| **Short** | Open short contracts held | — |
| **Net** | Long − Short | **+ve = net long (bullish)** / −ve = net short (bearish) |
| **L/S%** | Long ÷ (Long + Short) × 100 | Above 50% = more longs than shorts |
| **Cum Net** | Running total of Net since period start | Trend in positioning — rising = accumulating longs |

---

**Options columns**
| Column | Meaning | Signal |
|--------|---------|--------|
| **Call Net** | Call Long − Call Short | +ve = net call buyer → bullish |
| **Put Net** | Put Long − Put Short | +ve = net put buyer → hedging / bearish bias |
| **Opt Delta** | Call Net − Put Net | +ve = overall bullish options stance |

---

**FII Money Flow (Cr)**
| Column | Meaning | Signal |
|--------|---------|--------|
| **Buy / Sell (Cr)** | Rupee value traded | — |
| **Net Flow (Cr)** | Buy − Sell | **+ve = FII net buyer** (money flowing in) |
| **OI Contracts** | Total open interest held | Size of position |

---

**Key signals to watch**
- 🟢 **FII Net Futures +ve & rising** → strong bullish trend; institutions accumulating
- 🔴 **FII Net Futures −ve & falling** → institutional distribution; bearish
- 🟡 **FII buying puts aggressively** → smart-money hedge; possible reversal ahead
- 📊 **PCR > 1.3** → contrarian bullish (put crowd near capitulation → bounce likely)
- 📊 **PCR < 0.5** → contrarian bearish (complacent bulls → correction risk)
- ⚡ **Client Net short** while **FII Net long** → high-conviction bullish setup
        """)

    # ── Controls row ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        data_type = st.radio("Data Type", ["OI", "Vol"], horizontal=True,
                             help="OI = Open Interest (positions held). Vol = Volume (daily traded).")
    with c2:
        lookback_opt = st.selectbox(
            "Lookback Period",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            index=3,
        )
    lookback_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    lookback_days = lookback_map[lookback_opt]
    start_date = selected_date - timedelta(days=lookback_days)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading F&O participant data…"):
        latest       = cached_fao_latest(selected_date, data_type)
        cum          = cached_fao_cumulative(selected_date, start_date, data_type)
        daily        = cached_fao_daily(selected_date, lookback_days + 10, data_type)
        fii_stats    = cached_fii_stats_latest(selected_date)
        fii_stats_h  = cached_fii_stats_history(selected_date, lookback_days)

    if latest.empty:
        st.warning(
            "No F&O participant data found. "
            "Run the backfill to load history:\n\n"
            "```\npython -m src.cli backfill-fao 365\n```"
        )
        return

    data_date = latest["trade_date"].iloc[0]
    st.caption(
        f"**{data_type}** data — latest: **{data_date.strftime('%d %b %Y')}** "
        f"| Lookback: {lookback_opt} (from {start_date.strftime('%d %b %Y')})"
    )

    # ── Market Intelligence Section ───────────────────────────────────────────
    st.markdown(
        "<div style='font-size:13px;font-weight:700;color:rgba(255,255,255,0.4);"
        "letter-spacing:1.5px;margin:4px 0 8px'>MARKET INTELLIGENCE — QUANT SIGNAL ENGINE</div>",
        unsafe_allow_html=True,
    )
    _render_market_intelligence(selected_date)
    st.divider()

    # ── FII Derivatives Statistics ────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:13px;font-weight:700;color:rgba(255,255,255,0.4);"
        "letter-spacing:1.5px;margin:4px 0 8px'>FII DERIVATIVES STATISTICS — MONEY FLOW (Rs. Cr)</div>",
        unsafe_allow_html=True,
    )
    c_tbl, c_chart = st.columns([1, 1])
    with c_tbl:
        _fii_stats_table(fii_stats)
    with c_chart:
        if not fii_stats_h.empty:
            st.caption("Net FII buy/sell flow by category (last period)")
            st.plotly_chart(_fii_flow_chart(fii_stats_h), use_container_width=True, key="fii_flow_summary")
    st.divider()

    # ── Signal interpretation ─────────────────────────────────────────────────
    fii_row = latest[latest["client_type"] == "FII"].iloc[0] if "FII" in latest["client_type"].values else None
    dii_row = latest[latest["client_type"] == "DII"].iloc[0] if "DII" in latest["client_type"].values else None
    _signal_box(fii_row, dii_row)

    # ── Index Futures KPI cards ───────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:rgba(255,255,255,0.4);"
        "letter-spacing:1px;margin:8px 0 4px'>INDEX FUTURES — NET POSITIONS</div>",
        unsafe_allow_html=True,
    )
    _kpi_cards(latest)
    st.markdown("")

    # ── Index Options KPI cards ───────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:rgba(255,255,255,0.4);"
        "letter-spacing:1px;margin:8px 0 4px'>INDEX OPTIONS — CALL / PUT NET + DELTA</div>",
        unsafe_allow_html=True,
    )
    _options_kpi_cards(latest)
    st.markdown("")

    # ── PCR metric ────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:12px;font-weight:600;color:rgba(255,255,255,0.4);"
        "letter-spacing:1px;margin:8px 0 4px'>PUT-CALL RATIO (CONTRARIAN SIGNAL)</div>",
        unsafe_allow_html=True,
    )
    _pcr_metric(latest)
    st.markdown("")

    # ── Tabs: Charts | Tables ─────────────────────────────────────────────────
    tab_charts, tab_daily, tab_cumul, tab_fii = st.tabs(
        ["📈 Charts", "📋 Daily Net Table", "📊 Cumulative Table", "💰 FII Money Flow"]
    )

    with tab_charts:
        # ── Futures section ──
        st.markdown("#### Cumulative Index Futures Net Position")
        st.caption(
            "Running sum of (Long − Short) since start date. "
            "Above zero = net long (bullish). Below zero = net short (bearish). "
            "Crossover at zero = structural shift in positioning."
        )
        st.plotly_chart(_cumulative_chart(cum), use_container_width=True)

        st.markdown("#### FII & DII Long-to-Short % (Index Futures)")
        st.caption(
            "Long / (Long + Short) × 100. "
            "Above 50% = more long than short. Below 50% = net short side dominant."
        )
        st.plotly_chart(_ls_trend_chart(cum), use_container_width=True)

        st.markdown("#### Daily Net — Index Futures (last 60 days)")
        c_fii, c_dii = st.columns(2)
        c_client, c_pro = st.columns(2)
        with c_fii:
            st.caption("**FII**")
            st.plotly_chart(_daily_net_chart(cum, "FII"), use_container_width=True)
        with c_dii:
            st.caption("**DII**")
            st.plotly_chart(_daily_net_chart(cum, "DII"), use_container_width=True)
        with c_client:
            st.caption("**Client (Retail)**")
            st.plotly_chart(_daily_net_chart(cum, "Client"), use_container_width=True)
        with c_pro:
            st.caption("**Pro (Proprietary)**")
            st.plotly_chart(_daily_net_chart(cum, "Pro"), use_container_width=True)

        # ── Options section ──
        st.divider()
        st.markdown("#### Cumulative Index Options — FII & DII Call vs Put Net")
        st.caption(
            "Solid line = Call Net (Long Calls − Short Calls). Dashed = Put Net (Long Puts − Short Puts). "
            "Positive Call Net = buying calls (bullish). "
            "Positive Put Net = buying puts (bearish hedge / directional short). "
            "When FII Call Net rises and Put Net falls → strong bullish options stance."
        )
        st.plotly_chart(_cumulative_options_chart(cum), use_container_width=True)

        st.markdown("#### Put-Call Ratio Daily Trend")
        st.caption(
            "PCR = Total Put OI ÷ Total Call OI (summed across all participants). "
            "Green zone (>1.3) = contrarian buy. Red zone (<0.7) = contrarian sell."
        )
        st.plotly_chart(_pcr_chart(cum), use_container_width=True)

    with tab_daily:
        st.markdown("#### Daily Net Positions — Index Futures")
        st.caption("Net = Long − Short contracts per day. Positive = net long that day.")
        _daily_table(daily)

        st.markdown("#### Daily Net Positions — Index Options")
        st.caption(
            "Call Net = Call Long − Call Short. Put Net = Put Long − Put Short. "
            "Options Delta = Call Net − Put Net (positive = overall bullish options stance)."
        )
        _options_daily_table(daily)

    with tab_cumul:
        st.markdown("#### Cumulative Net Positions — Index Futures")
        st.caption(
            f"Running total from {start_date.strftime('%d %b %Y')}. "
            "L/S% = Long as % of total (Long + Short). Below 50% = net short."
        )
        _cumulative_table(cum)

        st.markdown("#### Cumulative Net Positions — Index Options")
        st.caption(
            "Cumulative = running sum of (Long − Short) per type since start date. "
            "Cum Delta = Cum Call Net − Cum Put Net (options market directional bias)."
        )
        _options_cumulative_table(cum)

    with tab_fii:
        st.markdown("#### FII Derivatives Statistics — Daily Buy/Sell Activity")
        st.caption(
            "FII buy/sell contracts and rupee value (Crore) by contract type. "
            "Net Flow (Cr) = Buy Value − Sell Value. Positive = net buyer (bullish), Negative = net seller (bearish)."
        )
        _fii_stats_table(fii_stats)

        if not fii_stats_h.empty:
            st.markdown("#### Net FII Flow Trend — Index F&O (Crore)")
            st.caption("Daily net rupee flow: Index Futures, Index Call Options, Index Put Options. "
                       "Large positive = institutional money entering index. Confirms or contradicts OI signal.")
            st.plotly_chart(_fii_flow_chart(fii_stats_h), use_container_width=True, key="fii_flow_tab")

            # Full history table
            with st.expander("Full FII Stats History Table"):
                hist_display = fii_stats_h.copy()
                hist_display["net_value_cr"] = hist_display["buy_value_cr"] - hist_display["sell_value_cr"]
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
                    column_order=["trade_date","category","net_contracts","buy_value_cr","sell_value_cr","net_value_cr","oi_contracts"],
                )
        else:
            st.info("Run `python -m src.cli backfill-fii-stats 365` to load FII statistics history.")

    # ── How to read ───────────────────────────────────────────────────────────
    with st.expander("📖 How to read this page", expanded=False):
        st.markdown("""
**Why Index Futures matter:**
Index Futures (Nifty/Bank Nifty) cannot be used for delivery — they are pure directional bets.
When FII *buys* Index Futures, they are making a *leveraged bet that the market goes UP*.
When they *short* Index Futures, they expect the market to *fall* or are *hedging* equity longs.

**Why Index Options matter:**
Options give directional information with asymmetric risk.
- **Call Net > 0** (Long Calls > Short Calls): participant is *buying* calls = bullish directional bet
- **Put Net > 0** (Long Puts > Short Puts): participant is *buying* puts = expecting fall or hedging longs
- **Options Delta** = Call Net − Put Net: positive = overall bullish stance, negative = bearish/hedged

**The 4 participants:**
| Participant | Who they are | Typical behaviour |
|---|---|---|
| **FII** | Foreign Institutional Investors | Trend-setting. Net long = bullish; net short = bearish. Most watched. |
| **DII** | Domestic Institutions (LIC, MFs, etc.) | Contrarian buyers. Buy when FII sells. Net short futures usually = hedging equity. |
| **Client** | Retail / small traders | Often wrong at extremes — extreme long = sell signal; extreme short = buy. |
| **Pro** | Proprietary desks | Market-makers and arbitrageurs. Usually short futures (delta-hedging long options). |

**OI vs Volume:**
- **OI (Open Interest)**: Contracts outstanding at end of day — actual *position* being held.
- **Volume**: Contracts traded during the day — *activity* but positions may have been reversed.
- OI is the better signal for conviction. Volume is good for detecting sudden activity.

**Key signals:**
- FII cumulative futures OI crosses **above zero** → structural long build-up (strong buy for index)
- FII cumulative futures OI falling sharply → unwinding longs or building shorts (caution)
- FII **buying calls + put net negative** → double bullish confirmation via options
- **PCR > 1.3** → excessive put-buying, contrarian bullish (everyone already hedged)
- **PCR < 0.7** → excessive call-buying, contrarian bearish (complacency at peaks)
- Retail (Client) net short at extremes → contrarian buy signal (they're usually wrong at turns)
        """)
