"""
FII / DII daily CASH-market flows view.

The single most-watched institutional-flow gauge: are foreign investors (FII/FPI)
buying or selling the cash market, and are domestic institutions (DII) absorbing
it? net = buy − sell (₹ Cr). Going-forward data is the NSE provisional figure;
older days are backfilled from Groww / CSV.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.repository import query_dataframe


@st.cache_data(ttl=300)
def _load(limit: int = 250) -> pd.DataFrame:
    df = query_dataframe(
        "SELECT trade_date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net, source "
        "FROM fii_dii_cash ORDER BY trade_date", []
    )
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _fmtcr(v) -> str:
    return f"₹{v:+,.0f} Cr" if v is not None else "—"


def _kpi(col, label, value, *, good_positive=True, suffix=" Cr"):
    if value is None or pd.isna(value):
        col.metric(label, "—"); return
    color = "normal"
    col.metric(label, f"₹{value:,.0f}{suffix}",
               delta=("inflow" if value >= 0 else "outflow") if good_positive else None,
               delta_color=("normal" if value >= 0 else "inverse"))


def render(selected_date: date) -> None:
    st.markdown("## 💰 FII / DII — Daily Cash Market Flows")
    st.caption(
        "Provisional cash-segment buy/sell/net (₹ Cr). **FII/FPI** = foreign · "
        "**DII** = domestic mutual funds/insurers. Net negative = selling. "
        "Source: NSE (live) + Groww/CSV (history)."
    )

    df = _load()
    if df.empty:
        st.info("No FII/DII cash data yet. Use the controls below to fetch from NSE "
                "or import a Groww CSV.")
    else:
        win = st.radio("Window", [30, 60, 90, 250], index=1, horizontal=True,
                       format_func=lambda d: ("All" if d == 250 else f"Last {d}d"), key="fiidii_win")
        d = df.tail(win).copy()
        latest = d.iloc[-1]

        # ── KPIs ──────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, f"FII net · {latest['trade_date']:%d %b}", latest["fii_net"])
        _kpi(c2, f"DII net · {latest['trade_date']:%d %b}", latest["dii_net"])
        _kpi(c3, "FII net · last 5d", d["fii_net"].tail(5).sum())
        _kpi(c4, "DII net · last 5d", d["dii_net"].tail(5).sum())

        # ── Daily FII vs DII net (bars) + cumulative (lines) ──────────────────
        d["fii_cum"] = d["fii_net"].cumsum()
        d["dii_cum"] = d["dii_net"].cumsum()
        fig = go.Figure()
        fig.add_bar(x=d["trade_date"], y=d["fii_net"], name="FII net",
                    marker_color=["#ff5252" if v < 0 else "#69f0ae" for v in d["fii_net"]])
        fig.add_bar(x=d["trade_date"], y=d["dii_net"], name="DII net",
                    marker_color=["#ff9100" if v < 0 else "#40c4ff" for v in d["dii_net"]],
                    opacity=0.55)
        fig.add_scatter(x=d["trade_date"], y=d["fii_cum"], name="FII cumulative",
                        yaxis="y2", line=dict(color="#ff5252", width=2))
        fig.add_scatter(x=d["trade_date"], y=d["dii_cum"], name="DII cumulative",
                        yaxis="y2", line=dict(color="#40c4ff", width=2))
        fig.update_layout(
            barmode="group", height=420, template="plotly_dark",
            margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="Daily net ₹ Cr", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis2=dict(title="Cumulative ₹ Cr", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── 🧠 Flow Intelligence engine ───────────────────────────────────────
        try:
            from src.analytics.fii_dii_intelligence import get_flow_intelligence
            intel = get_flow_intelligence(selected_date)
        except Exception:
            intel = {}
        if intel:
            _tagcol = {"ABSORBED SELLING": "#40c4ff", "BROAD RISK-OFF": "#ff5252",
                       "ALIGNED BUYING": "#00c853", "FII-LED BUYING": "#69f0ae",
                       "BALANCED": "#9e9e9e"}
            tc = _tagcol.get(intel["regime_tag"], "#9e9e9e")
            st.markdown("#### 🧠 Flow Intelligence — what institutions are doing & what it means")
            r1, r2, r3 = st.columns(3)
            r1.markdown(f"<div style='font-size:11px;color:#888'>FII REGIME (the pressure)</div>"
                        f"<div style='font-size:15px;font-weight:600'>{intel['fii_regime']}</div>"
                        f"<div style='font-size:10.5px;color:#888'>5d {_fmtcr(intel['fii_5d'])} · "
                        f"streak {intel['fii_streak']:+d}d · today {intel['fii_z']:+.1f}σ</div>",
                        unsafe_allow_html=True)
            r2.markdown(f"<div style='font-size:11px;color:#888'>DII STANCE (the floor)</div>"
                        f"<div style='font-size:15px;font-weight:600'>{intel['dii_stance']}</div>"
                        f"<div style='font-size:10.5px;color:#888'>5d {_fmtcr(intel['dii_5d'])}"
                        + (f" · absorbing {intel['absorption_pct']:.0f}%" if intel.get('absorption_pct') else "")
                        + "</div>", unsafe_allow_html=True)
            r3.markdown(f"<div style='font-size:11px;color:#888'>SETUP</div>"
                        f"<div style='font-size:15px;font-weight:700;color:{tc}'>{intel['regime_tag']}</div>"
                        f"<div style='font-size:10.5px;color:#888'>next-day lean: "
                        f"<b>{intel['forward_lean']}</b></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='border-left:4px solid {tc};padding:7px 12px;margin:8px 0;"
                f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>"
                f"<div style='font-size:13px'>{intel['narrative']}</div>"
                f"<div style='font-size:12.5px;color:rgba(255,255,255,0.82);margin-top:5px'>"
                f"📊 <b>What it means:</b> {intel['structural_read']}</div>"
                f"<div style='font-size:12px;color:#7fdbff;margin-top:5px'>➡️ <b>Next-day lean: "
                f"{intel['forward_lean']}</b> — {intel['forward_note']}</div></div>",
                unsafe_allow_html=True,
            )
            st.caption("⚖️ Evidence: " + intel["validation"])

        # ── Data table ────────────────────────────────────────────────────────
        with st.expander("📋 Daily data", expanded=False):
            show = d.sort_values("trade_date", ascending=False).copy()
            show["trade_date"] = show["trade_date"].dt.strftime("%d %b %Y")
            st.dataframe(
                show[["trade_date", "fii_buy", "fii_sell", "fii_net",
                      "dii_buy", "dii_sell", "dii_net", "source"]],
                use_container_width=True, hide_index=True,
            )

    # Data auto-updates daily from NSE via the nightly job (cmd_daily), so no
    # manual fetch/backfill/upload controls are shown here. Backfill + CSV import
    # remain available programmatically in fii_dii_cash_fetcher for one-off/cloud use.
