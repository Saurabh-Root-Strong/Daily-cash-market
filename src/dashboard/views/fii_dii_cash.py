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

    # ── 📅 1–2 Week Outlook (consolidated, from >1yr of data) ─────────────────
    try:
        from src.analytics.fii_dii_intelligence import get_two_week_outlook
        ow = get_two_week_outlook(selected_date)
    except Exception:
        ow = {}
    if ow:
        c = ow["bias_color"]
        _sup = "".join(f"<li>{s}</li>" for s in ow["supports"][:3])
        _rsk = "".join(f"<li>{s}</li>" for s in ow["risks"][:3]) or "<li>None pressing right now.</li>"
        st.markdown(
            f"<div style='border:1px solid {c}55;border-left:5px solid {c};border-radius:8px;"
            f"padding:10px 16px;margin-bottom:6px;background:rgba(255,255,255,0.02)'>"
            f"<div style='font-size:11px;color:#888;letter-spacing:.5px'>📅 1–2 WEEK OUTLOOK · "
            f"synthesised from &gt;1yr of FII/DII + index data</div>"
            f"<div style='font-size:19px;font-weight:800;color:{c};margin:2px 0'>{ow['bias']}</div>"
            f"<div style='font-size:12.5px;color:rgba(255,255,255,0.85)'>{ow['base_case']}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px;font-size:11.5px'>"
            f"<div><b style='color:#69f0ae'>Supports</b><ul style='margin:2px 0 0 16px;padding:0'>{_sup}</ul></div>"
            f"<div><b style='color:#ff9100'>Risks</b><ul style='margin:2px 0 0 16px;padding:0'>{_rsk}</ul></div>"
            f"</div>"
            f"<div style='font-size:12px;color:#ffd600;margin-top:7px'>🎯 <b>Trip-wire:</b> {ow['tripwire']}</div>"
            f"<div style='font-size:10.5px;color:#777;margin-top:4px'>Confidence: {ow['confidence']}</div>"
            f"</div>", unsafe_allow_html=True,
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

        # ── 📊 Historical pattern — what this regime led to + flow-vs-index ────
        try:
            from src.analytics.fii_dii_intelligence import get_flow_history_pattern
            pat = get_flow_history_pattern(selected_date)
        except Exception:
            pat = {}
        if pat and pat.get("current_dist"):
            cd = pat["current_dist"]
            st.markdown("##### 📊 Historical pattern — *when flows looked like now, what did the index do?*")
            st.markdown(
                f"<div style='font-size:12.5px'>Over the last <b>{pat['n_days']}</b> sessions, whenever the regime "
                f"was <b>{pat['current_regime']}</b> (today's state), the Nifty <b>{pat['horizon']} trading days later</b> "
                f"averaged <b>{cd['mean']:+.2f}%</b> (median {cd['median']:+.2f}%, up {cd['pos_pct']:.0f}% of the time, "
                f"n={cd['n']}).</div>", unsafe_allow_html=True)
            st.info("🔎 " + pat["interpretation"])
            s = pd.DataFrame(pat["series"]); s["trade_date"] = pd.to_datetime(s["trade_date"])
            ofig = go.Figure()
            ofig.add_scatter(x=s["trade_date"], y=s["fii_cum"], name="FII cumulative",
                             line=dict(color="#ff5252", width=2))
            ofig.add_scatter(x=s["trade_date"], y=s["dii_cum"], name="DII cumulative",
                             line=dict(color="#40c4ff", width=2))
            ofig.add_scatter(x=s["trade_date"], y=s["close_val"], name="Nifty 50",
                             yaxis="y2", line=dict(color="#ffd600", width=2.5))
            ofig.update_layout(height=340, template="plotly_dark",
                               margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.15),
                               yaxis=dict(title="Cumulative flow ₹ Cr"),
                               yaxis2=dict(title="Nifty", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(ofig, use_container_width=True)
            st.caption(
                f"FII flow ↔ index (concurrent 5d) correlation **{pat['contemporaneous_corr']:+.2f}** — FII drives the "
                "tape in real time. Watch how FII cumulative (red) falls all year while Nifty (gold) holds — DII "
                "cumulative (blue) absorbing it. A descriptive ~1-year pattern on flow-skewed data, not a forecast."
            )

        # ── 🧠 Flow Event Memory — significant moves & what followed ───────────
        try:
            from src.analytics.fii_dii_intelligence import get_flow_events
            ev = get_flow_events(selected_date)
        except Exception:
            ev = {}
        if ev:
            st.markdown("##### 🧠 Flow Event Memory — significant moves & what followed")
            te = ev.get("today_event")
            if te:
                ts = ev["type_stats"].get(te["type"])
                msg = (f"🚨 **TODAY is a significant flow event: {te['type']}** — "
                       f"FII ₹{te['fii_net']:+,.0f} Cr (z{te['z']:+.1f}).")
                if ts:
                    msg += (f" Historically the **{ts['n']}** times this happened, Nifty was "
                            f"**{ts['mean10']:+.1f}%** two weeks later ({ts['pos_pct']:.0f}% up).")
                st.warning(msg)
            if ev["type_stats"]:
                st.caption("What each event type has historically led to (2 weeks later, realised windows only):")
                rows = [{"Event": k, "2-wk avg %": v["mean10"], "% up": v["pos_pct"], "times": v["n"]}
                        for k, v in sorted(ev["type_stats"].items(), key=lambda x: -x[1]["n"])]
                st.dataframe(
                    pd.DataFrame(rows), hide_index=True, use_container_width=True,
                    column_config={
                        "Event": st.column_config.TextColumn(
                            "Event", help="The significant flow-event type — a sudden huge FII buy/sell, "
                            "or FIIs reversing direction. Why: these stand-out moments often mark turning points."),
                        "2-wk avg %": st.column_config.NumberColumn(
                            "2-wk avg %", format="%+.2f%%",
                            help="Average Nifty 50 return over the next ~2 weeks (10 trading days) AFTER this event "
                            "happened, historically. Why: tells you which way the market TENDED to go next."),
                        "% up": st.column_config.NumberColumn(
                            "% up", format="%.0f%%",
                            help="Of all the times this event occurred, the share where Nifty was HIGHER 2 weeks "
                            "later. Why: >50% = bullish tendency, <50% = bearish tendency."),
                        "times": st.column_config.NumberColumn(
                            "times", help="How many times this event has occurred in the history (sample size). "
                            "Why: small counts (~10) mean treat it as suggestive, not reliable."),
                    })
            with st.expander(f"📜 Event log — {ev['total_events']} significant flow events on record", expanded=False):
                el = pd.DataFrame(ev["events"])
                if not el.empty:
                    st.dataframe(
                        el[["date", "type", "fii_net", "dii_net", "z", "fwd5", "fwd10"]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            "date": st.column_config.TextColumn(
                                "Date", help="The day this significant flow event occurred."),
                            "type": st.column_config.TextColumn(
                                "Event", help="Huge FII Buy/Sell = FII net beyond ±2σ of its 60-day norm "
                                "(a genuinely outsized day); reversal = FIIs flipping direction after a 3+ day streak."),
                            "fii_net": st.column_config.NumberColumn(
                                "FII net ₹Cr", format="%+,.0f",
                                help="Foreign-institution net CASH flow that day. Negative = net selling, positive = "
                                "buying. Why: FIIs are the marginal price-setter — this is the main pressure."),
                            "dii_net": st.column_config.NumberColumn(
                                "DII net ₹Cr", format="%+,.0f",
                                help="Domestic-institution net cash flow (mutual funds/insurers). Why: DIIs are the "
                                "'floor' that absorbs FII selling — when positive, they cushion the market."),
                            "z": st.column_config.NumberColumn(
                                "z-score", format="%+.1f",
                                help="How EXTREME that day's FII flow was vs its own trailing 60-day average, in "
                                "standard deviations. Why: |z| ≥ 2 flags a genuinely unusual day, not just a big number."),
                            "fwd5": st.column_config.NumberColumn(
                                "Nifty +1wk %", format="%+.2f",
                                help="Nifty 50 return over the 5 trading days (~1 week) AFTER the event. "
                                "Blank = that window hasn't completed yet."),
                            "fwd10": st.column_config.NumberColumn(
                                "Nifty +2wk %", format="%+.2f",
                                help="Nifty 50 return over the 10 trading days (~2 weeks) AFTER the event. "
                                "Blank = that window hasn't completed yet. Why: shows what the move 'led to'."),
                        })
            st.caption("⚖️ Small samples (~10/type) — a descriptive memory, not a forecast. Pattern: extreme FII "
                       "BUYS mildly precede weakness (piling in marks tops); huge SELLS tend to exhaust (flat/bounce).")

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
