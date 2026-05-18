from datetime import date
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.analytics.sector_aggregator import get_sector_master_performance

_TABLE_COLS = [
    "sector",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "1W_deliv_pct",     "2W_deliv_pct",     "1M_deliv_pct",     "3M_deliv_pct",
]

_COL_CONFIG = {
    "sector":            st.column_config.TextColumn("Sector", width="medium"),
    "1W_price_chg_pct":  st.column_config.NumberColumn("1W Price Chg %",  format="%.2f%%",
                            help="Turnover-weighted avg price change — last 7 calendar days"),
    "2W_price_chg_pct":  st.column_config.NumberColumn("2W Price Chg %",  format="%.2f%%",
                            help="Turnover-weighted avg price change — last 14 calendar days"),
    "1M_price_chg_pct":  st.column_config.NumberColumn("1M Price Chg %",  format="%.2f%%",
                            help="Turnover-weighted avg price change — last 30 calendar days"),
    "3M_price_chg_pct":  st.column_config.NumberColumn("3M Price Chg %",  format="%.2f%%",
                            help="Turnover-weighted avg price change — last 90 calendar days"),
    "1W_deliv_pct":      st.column_config.NumberColumn("1W Delivery %",   format="%.1f%%",
                            help="Turnover-weighted avg delivery % — last 7 days. Higher = more conviction buying"),
    "2W_deliv_pct":      st.column_config.NumberColumn("2W Delivery %",   format="%.1f%%",
                            help="Turnover-weighted avg delivery % — last 14 days"),
    "1M_deliv_pct":      st.column_config.NumberColumn("1M Delivery %",   format="%.1f%%",
                            help="Turnover-weighted avg delivery % — last 30 days"),
    "3M_deliv_pct":      st.column_config.NumberColumn("3M Delivery %",   format="%.1f%%",
                            help="Turnover-weighted avg delivery % — last 90 days (long-term baseline)"),
}


def _normalize(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def _compute_outlook(df: pd.DataFrame) -> pd.DataFrame:
    """Score sectors for expected 1-2 week performance."""
    out = df.copy()

    # Delivery momentum: recent vs long-term baseline
    out["_deliv_momentum"] = out["1W_deliv_pct"] - out["3M_deliv_pct"]

    # Price momentum: 1W and 2W
    out["_price_momentum"] = out.get("2W_price_chg_pct", out["1W_price_chg_pct"])

    # Base conviction: sustained high delivery over 3M
    out["_base_conviction"] = out["3M_deliv_pct"]

    # Acceleration: delivery rising across all windows (1W > 2W > 1M)
    out["_accel"] = (
        (out["1W_deliv_pct"] > out.get("2W_deliv_pct", out["1M_deliv_pct"])).astype(float) +
        (out.get("2W_deliv_pct", out["1M_deliv_pct"]) > out["1M_deliv_pct"]).astype(float)
    )

    # Composite score (0–100)
    out["Score"] = (
        _normalize(out["_deliv_momentum"]) * 35 +
        _normalize(out["_base_conviction"]) * 25 +
        _normalize(out["_price_momentum"]) * 25 +
        _normalize(out["_accel"])           * 15
    ).round(1)

    # Signal label
    def _signal(row):
        d_building = row["_deliv_momentum"] > 0
        p_positive = row["_price_momentum"] > 0
        if d_building and p_positive:
            return "🟢 Accumulating"
        elif d_building and not p_positive:
            return "🟡 Buying Dips"
        elif not d_building and p_positive:
            return "🟠 Weak Rally"
        else:
            return "🔴 Distributing"

    out["Signal"] = out.apply(_signal, axis=1)

    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols).sort_values("Score", ascending=False).reset_index(drop=True)


def _kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Top Deliv (1W)",  df.nlargest(1,"1W_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1W_deliv_pct').iloc[0]['1W_deliv_pct']:.1f}%",
              help="Highest conviction buying this week")
    c2.metric("Top Deliv (1M)",  df.nlargest(1,"1M_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1M_deliv_pct').iloc[0]['1M_deliv_pct']:.1f}%",
              help="Highest conviction buying this month")
    c3.metric("Top Deliv (3M)",  df.nlargest(1,"3M_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'3M_deliv_pct').iloc[0]['3M_deliv_pct']:.1f}%",
              help="Highest sustained delivery over 3 months")
    c4.metric("Best Price (1W)", df.nlargest(1,"1W_price_chg_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1W_price_chg_pct').iloc[0]['1W_price_chg_pct']:+.2f}%",
              help="Best price performer this week")
    c5.metric("Best Price (2W)", df.nlargest(1,"2W_price_chg_pct").iloc[0]["sector"]
              if "2W_price_chg_pct" in df.columns else "—",
              f"{df.nlargest(1,'2W_price_chg_pct').iloc[0]['2W_price_chg_pct']:+.2f}%"
              if "2W_price_chg_pct" in df.columns else "—",
              help="Best price performer last 2 weeks")


def _comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    col_map = {
        "Delivery %":     ("1W_deliv_pct",     "2W_deliv_pct",     "1M_deliv_pct",     "3M_deliv_pct"),
        "Price Change %": ("1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct"),
    }
    cols   = [c for c in col_map[metric] if c in df.columns]
    labels = {"1W_deliv_pct":"1W","2W_deliv_pct":"2W","1M_deliv_pct":"1M","3M_deliv_pct":"3M",
              "1W_price_chg_pct":"1W","2W_price_chg_pct":"2W","1M_price_chg_pct":"1M","3M_price_chg_pct":"3M"}
    colors = ["#4c78a8","#72b7b2","#f58518","#54a24b"]

    sdf = df.sort_values(cols[-1], ascending=True).fillna(0)
    fig = go.Figure()
    for col, color in zip(cols, colors):
        fig.add_trace(go.Bar(
            y=sdf["sector"], x=sdf[col], name=labels[col],
            orientation="h", marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{y}}</b><br>{labels[col]}: %{{x:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title=metric,
        xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, r=40, t=40, b=50),
        height=max(440, len(df) * 28 + 120),
        bargap=0.18, bargroupgap=0.06,
    )
    return fig


def _outlook_chart(scored: pd.DataFrame) -> go.Figure:
    top = scored.head(10)
    colors = top["Signal"].map({
        "🟢 Accumulating": "#2ca02c",
        "🟡 Buying Dips":  "#f0b429",
        "🟠 Weak Rally":   "#f58518",
        "🔴 Distributing": "#d62728",
    }).tolist()

    fig = go.Figure(go.Bar(
        x=top["Score"],
        y=top["sector"],
        orientation="h",
        marker_color=colors,
        text=top["Signal"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Composite Score (0–100)", range=[0, 105]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, r=200, t=20, b=40),
        height=360,
    )
    return fig


def render(selected_date: date, min_turnover: float) -> None:
    st.subheader("Sector Performance — Multi-Period Summary")
    st.caption(
        f"Turnover-weighted metrics as of **{selected_date.strftime('%d %b %Y')}**  "
        "· Sorted by 3M Delivery % ↓"
    )

    df = get_sector_master_performance(selected_date, min_turnover_lacs=min_turnover)
    if df.empty:
        st.warning("No data available. Run a backfill first.")
        return

    _kpi_row(df)
    st.markdown("---")

    # ── 1-2 Week Outlook ─────────────────────────────────────────────────────
    st.markdown("### 📈 1–2 Week Sector Outlook")
    st.caption(
        "Composite score: **35%** delivery momentum (1W vs 3M baseline) + "
        "**25%** base conviction (3M delivery) + "
        "**25%** recent price momentum (2W) + "
        "**15%** acceleration (delivery trending up across windows)"
    )

    scored = _compute_outlook(df)

    # Top picks callout
    top3 = scored.head(3)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
        col.metric(
            label=row["Signal"] + " — " + row["sector"],
            value=f"Score: {row['Score']:.0f}/100",
            delta=f"Deliv trending: {row['1W_deliv_pct']:.1f}% vs {row['3M_deliv_pct']:.1f}% (3M avg)",
        )

    st.plotly_chart(_outlook_chart(scored), use_container_width=True)

    with st.expander("📊 Full Scored Table", expanded=False):
        st.dataframe(
            scored[["sector", "Signal", "Score",
                    "1W_deliv_pct", "2W_deliv_pct", "3M_deliv_pct",
                    "1W_price_chg_pct", "2W_price_chg_pct"]].rename(columns={
                "1W_deliv_pct": "1W Deliv %", "2W_deliv_pct": "2W Deliv %",
                "3M_deliv_pct": "3M Deliv %",
                "1W_price_chg_pct": "1W Price %", "2W_price_chg_pct": "2W Price %",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Master table ──────────────────────────────────────────────────────────
    with st.expander("📋 Master Performance Table", expanded=True):
        show_cols = [c for c in _TABLE_COLS if c in df.columns]
        col_cfg   = {k: v for k, v in _COL_CONFIG.items() if k in show_cols}
        st.dataframe(
            df[show_cols],
            column_config=col_cfg,
            use_container_width=True,
            hide_index=True,
            height=min(700, (len(df) + 1) * 36 + 4),
        )

    # ── Visual comparison ─────────────────────────────────────────────────────
    st.markdown("### Visual Comparison")
    metric = st.radio("Compare sectors by", ["Delivery %", "Price Change %"], horizontal=True)
    st.plotly_chart(_comparison_chart(df, metric), use_container_width=True)

    with st.expander("ℹ️ Signal Legend", expanded=False):
        st.markdown("""
        | Signal | Meaning | Next 1-2 Week Implication |
        |--------|---------|--------------------------|
        | 🟢 **Accumulating** | Delivery rising + price positive | Likely to continue upward — institutions buying on strength |
        | 🟡 **Buying Dips** | Delivery rising + price negative | Smart money absorbing weakness — potential bounce |
        | 🟠 **Weak Rally** | Delivery falling + price positive | Price up but no conviction — rally may not sustain |
        | 🔴 **Distributing** | Delivery falling + price negative | Institutions selling — avoid or reduce exposure |
        """)
