"""
Backtest view — replay sector rotation signals on any past date and measure
actual outcomes.  Shows both BUY and SELL side recommendations with P&L.
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.constants import POSITIVE_COLOR, NEGATIVE_COLOR, PLOT_BG, PAPER_BG, GRID_COLOR

# ── Signal metadata ───────────────────────────────────────────────────────────
_INVEST_SIGNALS = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
_EXIT_SIGNALS   = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}

_SECTOR_COLOR = {
    "BUY":   "#00c853",
    "SELL":  "#d50000",
    "WATCH": "#ffab40",
    "HOLD":  "#888888",
}

# conviction labels → side
_BUY_CONVICTIONS  = {"Strong", "Buying", "Watch"}
_SELL_CONVICTIONS = {"Exit", "Reducing", "Fading"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _assign_conviction(stocks: pd.DataFrame, invest_signal: bool) -> pd.DataFrame:
    stocks = stocks.copy()
    valid_d = stocks["wtd_deliv_per"].dropna()
    hi = float(valid_d.quantile(0.67)) if len(valid_d) >= 3 else float(valid_d.max() if len(valid_d) else 0)
    lo = float(valid_d.quantile(0.33)) if len(valid_d) >= 3 else float(valid_d.min() if len(valid_d) else 0)

    def _conv(r):
        d = float(r["wtd_deliv_per"]) if pd.notna(r["wtd_deliv_per"]) else 0.0
        p = float(r["price_chg_pct"])  if pd.notna(r["price_chg_pct"])  else 0.0
        if invest_signal:
            if   d >= hi and p < 0: return "Strong"
            elif d >= hi:           return "Buying"
            elif d >= lo:           return "Watch"
            else:                   return "Weak"
        else:
            if   d <= lo and p > 0: return "Exit"
            elif d <= lo:           return "Reducing"
            elif p > 0:             return "Fading"
            else:                   return "Neutral"

    stocks["conviction"] = stocks.apply(_conv, axis=1)
    return stocks


def _get_prices(symbols: list, trade_date: date) -> dict:
    from src.data.repository import query_dataframe
    if not symbols:
        return {}
    ph = ", ".join("?" * len(symbols))
    df = query_dataframe(
        f"SELECT symbol, close_price FROM daily_data WHERE trade_date = ? AND symbol IN ({ph})",
        [trade_date] + list(symbols),
    )
    return dict(zip(df["symbol"], df["close_price"]))


def _sector_side(action: str) -> str:
    a = (action or "").upper()
    if a.startswith("BUY"):   return "BUY"
    if a.startswith("EXIT") or a.startswith("REDUCE"): return "SELL"
    if a.startswith("WATCH"): return "WATCH"
    return "HOLD"


def _pnl_chart(picks: list[dict], side: str) -> go.Figure:
    """Horizontal bar chart.  For SELL picks, DOWN = correct = green."""
    valid = sorted([r for r in picks if not math.isnan(r["pnl_pct"])],
                   key=lambda r: r["pnl_pct"])
    if not valid:
        return go.Figure()

    labels = [r["symbol"] for r in valid]
    values = [r["pnl_pct"] for r in valid]

    if side == "BUY":
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
    else:  # SELL — going down was the correct call
        colors = [POSITIVE_COLOR if v <= 0 else NEGATIVE_COLOR for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition="outside", textfont=dict(size=9),
        customdata=[[r["company"][:30], r["sector"], r["conviction"],
                     r["entry"], r["exit"]] for r in valid],
        hovertemplate=(
            "<b>%{y}</b>  %{customdata[0]}<br>"
            "Sector: %{customdata[1]}<br>"
            "Signal: %{customdata[2]}<br>"
            "Entry: %{customdata[3]:,.2f}  |  Exit: %{customdata[4]:,.2f}<br>"
            "Return: %{x:+.2f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(title="Return %", showgrid=True, gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.5)", ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=9)),
        height=max(320, len(valid) * 20 + 60),
        margin=dict(t=10, b=40, l=120, r=90),
    )
    return fig


def _stock_table(picks: list[dict], side: str,
                 signal_date: date, check_date: date) -> None:
    """Render coloured dataframe.  For BUY: green=up.  For SELL: green=down."""
    entry_col = f"Price {signal_date.strftime('%d %b')}"
    exit_col  = f"Price {check_date.strftime('%d %b')}"
    pnl_col   = "Return %"

    rows = []
    for r in sorted(picks, key=lambda x: -x["pnl_pct"] if not math.isnan(x["pnl_pct"]) else -9999):
        rows.append({
            "Symbol":    r["symbol"],
            "Company":   r["company"],
            "Sector":    r["sector"],
            "Signal":    r["conviction"],
            "Deliv %":   round(r["deliv_pct"], 1),
            entry_col:   round(r["entry"], 2) if r["entry"] else None,
            exit_col:    round(r["exit"],  2) if r["exit"]  else None,
            pnl_col:     round(r["pnl_pct"], 2) if not math.isnan(r["pnl_pct"]) else None,
        })

    df = pd.DataFrame(rows)

    def _color(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "color:#888"
        if side == "BUY":
            return f"color:{POSITIVE_COLOR}" if val >= 0 else f"color:{NEGATIVE_COLOR}"
        else:  # SELL: red = went up (bad), green = went down (good)
            return f"color:{POSITIVE_COLOR}" if val <= 0 else f"color:{NEGATIVE_COLOR}"

    st.dataframe(
        df.style.applymap(_color, subset=[pnl_col]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Symbol":  st.column_config.TextColumn("Symbol",  width="small"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Sector":  st.column_config.TextColumn("Sector",  width="medium"),
            "Signal":  st.column_config.TextColumn("Signal",  width="small"),
            "Deliv %": st.column_config.NumberColumn("Deliv %", format="%.1f%%"),
            entry_col: st.column_config.NumberColumn(entry_col, format="₹%.2f"),
            exit_col:  st.column_config.NumberColumn(exit_col,  format="₹%.2f"),
            pnl_col:   st.column_config.NumberColumn(pnl_col,   format="%+.2f%%"),
        },
    )


def _kpis(picks: list[dict], side: str) -> None:
    valid   = [r for r in picks if not math.isnan(r["pnl_pct"])]
    no_data = len(picks) - len(valid)
    if not valid:
        st.caption("No price data for this period.")
        return

    avg = sum(r["pnl_pct"] for r in valid) / len(valid)
    if side == "BUY":
        correct = [r for r in valid if r["pnl_pct"] >= 0]
        label   = "Went UP (correct)"
    else:
        correct = [r for r in valid if r["pnl_pct"] <= 0]
        label   = "Went DOWN (correct)"

    accuracy = 100 * len(correct) / len(valid)
    best  = max(r["pnl_pct"] for r in valid)
    worst = min(r["pnl_pct"] for r in valid)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Picks", f"{len(picks)}", delta=f"{no_data} missing data" if no_data else None,
              delta_color="off")
    k2.metric("Signal Accuracy", f"{accuracy:.0f}%", help=label)
    k3.metric("Avg Return", f"{avg:+.2f}%")
    k4.metric("Best", f"{best:+.2f}%")
    k5.metric("Worst", f"{worst:+.2f}%")


# ── Main render ───────────────────────────────────────────────────────────────

def render(available_dates: list) -> None:
    st.title("Backtest — Signal Validation")
    st.caption(
        "Replays sector rotation logic on a past **Signal Date** and measures "
        "what actually happened by the **Check Date**. "
        "Green on BUY = stock went up (correct). Green on SELL = stock went down (correct)."
    )

    if len(available_dates) < 2:
        st.warning("Need at least 2 dates in the database.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        default_sig_idx = min(7, len(available_dates) - 2)
        signal_date: date = st.selectbox(
            "Signal Date  (signals generated on this day)",
            options=available_dates,
            index=default_sig_idx,
            format_func=lambda d: d.strftime("%d %b %Y (%a)"),
            key="bt_signal",
        )

    with c2:
        valid_check = [d for d in available_dates if d > signal_date]
        if not valid_check:
            st.warning("No data available after signal date.")
            return
        check_date: date = st.selectbox(
            "Check Date  (measure outcome on this day)",
            options=valid_check,
            index=0,
            format_func=lambda d: d.strftime("%d %b %Y (%a)"),
            key="bt_check",
        )

    with c3:
        min_turnover = st.number_input(
            "Min Turnover (Lacs)",
            min_value=0.0, max_value=500.0, value=5.0, step=5.0,
            key="bt_min_to",
        )

    trading_days = sum(1 for d in available_dates if signal_date < d <= check_date)
    st.caption(
        f"Holding period: **{trading_days} trading day{'s' if trading_days != 1 else ''}**  "
        f"({signal_date.strftime('%d %b %Y')}  to  {check_date.strftime('%d %b %Y')})"
    )

    if not st.button("Run Backtest", type="primary"):
        st.info("Set dates above then click **Run Backtest**.")
        return

    _execute(signal_date, check_date, min_turnover)


# ── Execution ─────────────────────────────────────────────────────────────────

def _execute(signal_date: date, check_date: date, min_turnover: float) -> None:
    from src.analytics.sector_rotation import get_sector_rotation, get_sector_stocks_rotation

    with st.spinner("Loading sector signals..."):
        rotation = get_sector_rotation(signal_date, min_turnover_lacs=min_turnover)

    # classify sectors into BUY / SELL / WATCH / HOLD
    rotation["_side"] = rotation["action"].apply(_sector_side)
    buy_sectors  = rotation[rotation["_side"] == "BUY" ].sort_values("accum_score", ascending=False)
    sell_sectors = rotation[rotation["_side"] == "SELL"].sort_values("accum_score", ascending=True)
    watch_sectors= rotation[rotation["_side"].isin(["WATCH","HOLD"])].sort_values("accum_score", ascending=False)

    # ── Sector overview ────────────────────────────────────────────────────────
    st.subheader(f"Sector Signals  —  {signal_date.strftime('%d %b %Y')}")
    _sector_overview(buy_sectors, sell_sectors, watch_sectors)

    # ── Collect all stock picks ────────────────────────────────────────────────
    all_sectors_for_stocks = pd.concat([buy_sectors, sell_sectors], ignore_index=True)
    n = len(all_sectors_for_stocks)
    if n == 0:
        st.info("No actionable sectors on this date.")
        return

    buy_picks  = []
    sell_picks = []
    progress   = st.progress(0, text="Fetching stocks...")

    for i, (_, sec_row) in enumerate(all_sectors_for_stocks.iterrows()):
        sector = sec_row["sector"]
        side   = sec_row["_side"]
        progress.progress((i + 1) / n, text=f"{sector}...")

        stocks = get_sector_stocks_rotation(sector, signal_date, min_turnover_lacs=min_turnover)
        if stocks.empty:
            continue

        invest = sec_row["signal"] in _INVEST_SIGNALS
        stocks = _assign_conviction(stocks, invest_signal=invest)

        if side == "BUY":
            wanted = _BUY_CONVICTIONS - {"Watch"}   # Strong + Buying
        else:
            wanted = _SELL_CONVICTIONS - {"Fading"}  # Exit + Reducing

        picks_df = stocks[stocks["conviction"].isin(wanted)].copy()
        if picks_df.empty:
            continue

        syms  = picks_df["symbol"].tolist()
        e_map = _get_prices(syms, signal_date)
        x_map = _get_prices(syms, check_date)

        for _, row in picks_df.iterrows():
            sym   = row["symbol"]
            entry = e_map.get(sym, 0.0)
            exit_ = x_map.get(sym, 0.0)
            pnl   = (exit_ - entry) / entry * 100 if (entry > 0 and exit_ > 0) else float("nan")
            rec = {
                "side":       side,
                "sector":     sector,
                "symbol":     sym,
                "company":    str(row.get("company_name", "")),
                "conviction": row["conviction"],
                "deliv_pct":  float(row.get("wtd_deliv_per", 0.0) or 0.0),
                "entry":      entry,
                "exit":       exit_,
                "pnl_pct":    pnl,
            }
            if side == "BUY":
                buy_picks.append(rec)
            else:
                sell_picks.append(rec)

    progress.empty()

    # ── Tabs: BUY | SELL | Full Table ─────────────────────────────────────────
    tab_buy, tab_sell, tab_all = st.tabs([
        f"📈  BUY Picks  ({len(buy_picks)})",
        f"📉  SELL / AVOID Picks  ({len(sell_picks)})",
        f"📋  Full Table  ({len(buy_picks)+len(sell_picks)})",
    ])

    with tab_buy:
        if not buy_picks:
            st.info("No Strong/Buying picks for BUY sectors on this date.")
        else:
            st.caption(
                f"Sectors flagged as **BUY** on {signal_date.strftime('%d %b')} → "
                f"stocks with highest delivery conviction (Strong / Buying). "
                f"Green = went UP ✅  Red = went DOWN ❌"
            )
            _kpis(buy_picks, "BUY")
            st.plotly_chart(_pnl_chart(buy_picks, "BUY"), use_container_width=True)
            _stock_table(buy_picks, "BUY", signal_date, check_date)

    with tab_sell:
        if not sell_picks:
            st.info("No Exit/Reducing picks for SELL sectors on this date.")
        else:
            st.caption(
                f"Sectors flagged as **SELL/AVOID** on {signal_date.strftime('%d %b')} → "
                f"stocks with lowest delivery (institutions exiting). "
                f"Green = went DOWN ✅ (signal correct)  Red = went UP ❌ (signal wrong)"
            )
            _kpis(sell_picks, "SELL")
            st.plotly_chart(_pnl_chart(sell_picks, "SELL"), use_container_width=True)
            _stock_table(sell_picks, "SELL", signal_date, check_date)

    with tab_all:
        st.caption("All BUY and SELL picks in one table. Sorted by return %.")
        all_picks = buy_picks + sell_picks
        if not all_picks:
            st.info("No picks found.")
            return

        entry_col = f"Price {signal_date.strftime('%d %b')}"
        exit_col  = f"Price {check_date.strftime('%d %b')}"
        rows = []
        for r in sorted(all_picks, key=lambda x: -x["pnl_pct"] if not math.isnan(x["pnl_pct"]) else -9999):
            rows.append({
                "Action":    r["side"],
                "Symbol":    r["symbol"],
                "Company":   r["company"],
                "Sector":    r["sector"],
                "Signal":    r["conviction"],
                "Deliv %":   round(r["deliv_pct"], 1),
                entry_col:   round(r["entry"], 2) if r["entry"] else None,
                exit_col:    round(r["exit"],  2) if r["exit"]  else None,
                "Return %":  round(r["pnl_pct"], 2) if not math.isnan(r["pnl_pct"]) else None,
            })

        df_all = pd.DataFrame(rows)

        def _color_all(val, col_name, row_data):
            # applied per-cell in the Return % column
            return ""

        def _style_return(val):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "color:#888"
            return f"color:{POSITIVE_COLOR}" if val >= 0 else f"color:{NEGATIVE_COLOR}"

        st.dataframe(
            df_all.style.applymap(_style_return, subset=["Return %"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Action":   st.column_config.TextColumn("Action",  width="small"),
                "Symbol":   st.column_config.TextColumn("Symbol",  width="small"),
                "Company":  st.column_config.TextColumn("Company", width="medium"),
                "Sector":   st.column_config.TextColumn("Sector",  width="medium"),
                "Signal":   st.column_config.TextColumn("Signal",  width="small"),
                "Deliv %":  st.column_config.NumberColumn("Deliv %",  format="%.1f%%"),
                entry_col:  st.column_config.NumberColumn(entry_col,  format="₹%.2f"),
                exit_col:   st.column_config.NumberColumn(exit_col,   format="₹%.2f"),
                "Return %": st.column_config.NumberColumn("Return %", format="%+.2f%%"),
            },
        )

    st.divider()
    st.caption(
        f"Signal: {signal_date.strftime('%d %b %Y')}  |  "
        f"Check: {check_date.strftime('%d %b %Y')}  |  "
        f"Min turnover: {min_turnover:.0f} Lacs"
    )


# ── Sector overview cards ─────────────────────────────────────────────────────

def _sector_overview(buy_df, sell_df, watch_df) -> None:
    col_b, col_s, col_w = st.columns(3)

    with col_b:
        st.markdown(f"**BUY  ({len(buy_df)} sectors)**")
        for _, row in buy_df.iterrows():
            dm  = row.get("deliv_momentum"); p = row.get("price_1w")
            dm_s = f"{dm:+.1f}%" if pd.notna(dm) else "—"
            p_s  = f"{p:+.2f}%" if pd.notna(p)  else "—"
            dm_c = POSITIVE_COLOR if (pd.notna(dm) and dm > 0) else NEGATIVE_COLOR
            p_c  = POSITIVE_COLOR if (pd.notna(p)  and p  > 0) else NEGATIVE_COLOR
            st.markdown(
                f"<div style='padding:5px 8px;margin:2px 0;"
                f"background:rgba(0,200,83,0.08);border-left:3px solid #00c853;"
                f"border-radius:0 4px 4px 0'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<b style='font-size:13px'>{row['sector']}</b>"
                f"<span style='font-size:11px;color:#00c853'>{row['accum_score']:.0f}/100</span>"
                f"</div>"
                f"<div style='font-size:11px;margin-top:2px'>"
                f"Deliv chg <b style='color:{dm_c}'>{dm_s}</b>"
                f"&nbsp;·&nbsp;1W price <b style='color:{p_c}'>{p_s}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with col_s:
        st.markdown(f"**SELL / AVOID  ({len(sell_df)} sectors)**")
        for _, row in sell_df.iterrows():
            dm  = row.get("deliv_momentum"); p = row.get("price_1w")
            dm_s = f"{dm:+.1f}%" if pd.notna(dm) else "—"
            p_s  = f"{p:+.2f}%" if pd.notna(p)  else "—"
            dm_c = POSITIVE_COLOR if (pd.notna(dm) and dm > 0) else NEGATIVE_COLOR
            p_c  = POSITIVE_COLOR if (pd.notna(p)  and p  > 0) else NEGATIVE_COLOR
            st.markdown(
                f"<div style='padding:5px 8px;margin:2px 0;"
                f"background:rgba(213,0,0,0.08);border-left:3px solid #d50000;"
                f"border-radius:0 4px 4px 0'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<b style='font-size:13px'>{row['sector']}</b>"
                f"<span style='font-size:11px;color:#d50000'>{row['signal'][:22]}</span>"
                f"</div>"
                f"<div style='font-size:11px;margin-top:2px'>"
                f"Deliv chg <b style='color:{dm_c}'>{dm_s}</b>"
                f"&nbsp;·&nbsp;1W price <b style='color:{p_c}'>{p_s}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with col_w:
        st.markdown(f"**WATCH / HOLD  ({len(watch_df)} sectors)**")
        for _, row in watch_df.iterrows():
            st.markdown(
                f"<div style='padding:4px 8px;margin:2px 0;"
                f"background:rgba(255,255,255,0.02);border-left:3px solid #555;"
                f"border-radius:0 4px 4px 0'>"
                f"<span style='font-size:12px;color:rgba(255,255,255,0.55)'>"
                f"{row['sector']}</span>"
                f"<span style='font-size:11px;color:#666;float:right'>"
                f"{row['signal'][:20]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
