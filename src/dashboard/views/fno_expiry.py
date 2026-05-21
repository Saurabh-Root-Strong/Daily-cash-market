"""
F&O Expiry Structure page.

Three tabs:
  Index Structure  — all expiries for a selected index with OI chart + chain drilldown
  Stock Matrix     — near/next/far side-by-side for all F&O stocks with roll signal
  Options Chain    — full CE/PE OI butterfly chart for any symbol + expiry
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]

_ROLL_HELP = {
    "Rolling Fwd": "Near OI declining while next OI rising — expiry rollover in progress",
    "Building":    "Near OI rising > 50K — fresh positions being added",
    "Unwinding":   "Both near and next OI declining — positions being closed",
    "Neutral":     "OI changes within normal range",
}
_ROLL_COLORS = {
    "Rolling Fwd": "#ff9800",
    "Building":    "#4caf50",
    "Unwinding":   "#f44336",
    "Neutral":     "#9e9e9e",
}


def _fmt_oi(v) -> str:
    if v is None:
        return "—"
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ── Index Structure tab ────────────────────────────────────────────────────────

def _render_index_structure(trade_date: date) -> None:
    from src.dashboard.cache.queries import cached_index_full_structure, cached_options_chain

    col_sym, _ = st.columns([1, 3])
    with col_sym:
        symbol = st.selectbox("Index", _INDEX_SYMBOLS, key="idx_sym")

    df = cached_index_full_structure(trade_date, symbol)
    if df.empty:
        st.info(f"No F&O data for **{symbol}** on {trade_date}. Fetch data for this date first.")
        return

    # ── KPI row ────────────────────────────────────────────────────────────────
    near_rows = df[df["expiry_tier"] == "Near Month"]
    near      = near_rows.iloc[0].to_dict() if not near_rows.empty else {}

    spot     = _safe_float(near.get("spot_price"))
    pcr      = _safe_float(near.get("pcr"))
    mp       = _safe_float(near.get("max_pain"))
    mp_dist  = _safe_float(near.get("mp_dist_pct"))
    dte      = near.get("days_to_expiry")

    call_total = df["call_oi"].fillna(0).sum()
    put_total  = df["put_oi"].fillna(0).sum()
    overall_pcr = round(put_total / call_total, 2) if call_total > 0 else None

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Spot Price",      f"₹{spot:,.0f}" if spot else "N/A",
              help="Index spot — sourced from daily_data; may show N/A for index symbols not stored there")
    k2.metric("Near Expiry DTE", f"{dte}d" if dte is not None else "—")
    k3.metric("Near Month PCR",  f"{pcr:.2f}" if pcr else "—",
              help="Put/Call OI ratio for near month expiry. >1 = more puts (hedging), <0.7 = more calls (bullish bets)")
    k4.metric("Max Pain (Near)", f"₹{mp:,.0f}" if mp else "—",
              delta=f"{mp_dist:+.1f}% from spot" if mp_dist is not None else None,
              delta_color="off",
              help="Strike where option writers' aggregate loss is minimised. Price tends to gravitate here at expiry")
    k5.metric("Overall PCR",     f"{overall_pcr:.2f}" if overall_pcr else "—",
              help="Aggregate Put/Call OI ratio across all expiries for this index")

    st.divider()

    # ── OI distribution chart ──────────────────────────────────────────────────
    chart_df = df[(df["call_oi"].fillna(0) + df["put_oi"].fillna(0)) > 0].copy()
    if not chart_df.empty:
        chart_df["label"] = chart_df.apply(
            lambda r: f"{r['expiry_label']}  ({r['expiry_tier']})", axis=1
        )
        fig = go.Figure()
        fig.add_bar(
            name="Call OI",
            y=chart_df["label"],
            x=chart_df["call_oi"].fillna(0),
            orientation="h",
            marker_color="#ef5350",
            hovertemplate="<b>%{y}</b><br>Call OI: %{x:,.0f}<extra></extra>",
        )
        fig.add_bar(
            name="Put OI",
            y=chart_df["label"],
            x=chart_df["put_oi"].fillna(0),
            orientation="h",
            marker_color="#26a69a",
            hovertemplate="<b>%{y}</b><br>Put OI: %{x:,.0f}<extra></extra>",
        )
        fig.update_layout(
            title=f"{symbol} — Call vs Put OI by Expiry",
            barmode="group",
            height=max(300, len(chart_df) * 38 + 120),
            xaxis_title="Open Interest (contracts)",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"idx_oi_bar_{symbol}")

    # ── Expiry structure table ─────────────────────────────────────────────────
    st.markdown("#### Expiry Structure")
    display_cols = [
        "expiry_label", "expiry_tier", "days_to_expiry",
        "fut_oi", "fut_chg_oi", "fut_basis_pct",
        "call_oi", "put_oi", "pcr", "max_pain", "mp_dist_pct",
        "top_ce_strike", "top_pe_strike",
    ]
    display = df[[c for c in display_cols if c in df.columns]].copy()
    display.columns = [
        c.replace("expiry_label", "Expiry")
         .replace("expiry_tier", "Tier")
         .replace("days_to_expiry", "DTE")
         .replace("fut_oi", "Fut OI")
         .replace("fut_chg_oi", "Chg OI")
         .replace("fut_basis_pct", "Basis%")
         .replace("call_oi", "Call OI")
         .replace("put_oi", "Put OI")
         .replace("pcr", "PCR")
         .replace("max_pain", "Max Pain")
         .replace("mp_dist_pct", "MP Dist%")
         .replace("top_ce_strike", "Top CE")
         .replace("top_pe_strike", "Top PE")
        for c in display.columns
    ]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "DTE":      st.column_config.NumberColumn("DTE", format="%d"),
            "Fut OI":   st.column_config.NumberColumn("Fut OI",  format="%d"),
            "Chg OI":   st.column_config.NumberColumn("Chg OI",  format="%d"),
            "Basis%":   st.column_config.NumberColumn("Basis%",  format="%.2f%%"),
            "Call OI":  st.column_config.NumberColumn("Call OI", format="%d"),
            "Put OI":   st.column_config.NumberColumn("Put OI",  format="%d"),
            "PCR":      st.column_config.NumberColumn("PCR",     format="%.2f"),
            "Max Pain": st.column_config.NumberColumn("Max Pain", format="₹%.0f"),
            "MP Dist%": st.column_config.NumberColumn("MP Dist%", format="%.1f%%"),
            "Top CE":   st.column_config.NumberColumn("Top CE Strike"),
            "Top PE":   st.column_config.NumberColumn("Top PE Strike"),
        },
    )

    # ── Options chain drilldown ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Options Chain Drilldown")

    expiry_list   = df["expiry_date"].tolist()
    expiry_labels = df.apply(
        lambda r: f"{r['expiry_label']}  —  {r['expiry_tier']}", axis=1
    ).tolist()

    sel_i = st.selectbox(
        "Select expiry to view chain",
        range(len(expiry_list)),
        format_func=lambda i: expiry_labels[i],
        key="idx_chain_exp",
    )
    chain = cached_options_chain(trade_date, symbol, expiry_list[sel_i], "OPTIDX")
    if not chain.empty:
        _render_chain_chart(chain, symbol, expiry_list[sel_i], spot, key="idx_chain")
        with st.expander("Full chain table"):
            _render_chain_table(chain)
    else:
        st.info("No options data for this expiry.")


# ── Stock Matrix tab ───────────────────────────────────────────────────────────

def _render_stock_matrix(trade_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_stock_expiry_matrix,
        cached_stock_monthly_expiries,
        cached_options_chain,
    )

    expiries = cached_stock_monthly_expiries(trade_date)
    df = cached_stock_expiry_matrix(trade_date)

    if df.empty:
        st.info("No F&O stock data available for this date.")
        return

    # ── Filters ────────────────────────────────────────────────────────────────
    sectors      = sorted(df["sector"].dropna().unique().tolist())
    roll_signals = ["Rolling Fwd", "Building", "Unwinding", "Neutral"]

    fc1, fc2 = st.columns(2)
    with fc1:
        sel_sectors = st.multiselect("Sector filter", sectors, key="sm_sectors")
    with fc2:
        sel_rolls = st.multiselect("Roll signal filter", roll_signals, key="sm_rolls")

    filtered = df.copy()
    if sel_sectors:
        filtered = filtered[filtered["sector"].isin(sel_sectors)]
    if sel_rolls:
        filtered = filtered[filtered["roll_signal"].isin(sel_rolls)]

    if filtered.empty:
        st.info("No stocks match current filters.")
        return

    # ── Roll signal KPIs ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    rc = filtered["roll_signal"].value_counts()
    k1.metric("🔄 Rolling Fwd", rc.get("Rolling Fwd", 0),
              help=_ROLL_HELP["Rolling Fwd"])
    k2.metric("📈 Building",    rc.get("Building",    0),
              help=_ROLL_HELP["Building"])
    k3.metric("📉 Unwinding",   rc.get("Unwinding",   0),
              help=_ROLL_HELP["Unwinding"])
    k4.metric("⚖️ Neutral",     rc.get("Neutral",     0))

    st.divider()

    # ── Per-expiry inner tabs ──────────────────────────────────────────────────
    exp_tab_labels = []
    for i, label in enumerate(["Near Month", "Next Month", "Far Month"]):
        suffix = f" ({expiries[i].strftime('%d %b')})" if i < len(expiries) else ""
        exp_tab_labels.append(f"{label}{suffix}")

    tab_near, tab_next, tab_far = st.tabs(exp_tab_labels)
    for tab, prefix in zip([tab_near, tab_next, tab_far], ["near", "next", "far"]):
        with tab:
            _render_expiry_tab(filtered, prefix)

    # ── Options chain drilldown ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Options Chain Drilldown")

    dc1, dc2 = st.columns(2)
    with dc1:
        sel_sym = st.selectbox(
            "Stock",
            filtered["symbol"].tolist(),
            key="sm_sym",
        )
    with dc2:
        _exp_labels = ["Near", "Next", "Far"]
        exp_map = {
            f"{_exp_labels[i] if i < len(_exp_labels) else f'Exp {i+1}'} ({e.strftime('%d %b')})" : e
            for i, e in enumerate(expiries)
        }
        sel_exp_key = st.selectbox("Expiry", list(exp_map.keys()), key="sm_exp")
        sel_expiry  = exp_map[sel_exp_key]

    if sel_sym and sel_expiry:
        chain = cached_options_chain(trade_date, sel_sym, sel_expiry, "OPTSTK")
        if not chain.empty:
            spot_row = filtered[filtered["symbol"] == sel_sym]
            spot     = _safe_float(spot_row["spot_price"].iloc[0]) if not spot_row.empty else None
            _render_chain_chart(chain, sel_sym, sel_expiry, spot, key="sm_chain")
            with st.expander("Full chain table"):
                _render_chain_table(chain)
        else:
            st.info(f"No options chain for **{sel_sym}** / {sel_expiry}.")


def _render_expiry_tab(df: pd.DataFrame, prefix: str) -> None:
    col_map = {
        f"{prefix}_fut_oi":    ("Fut OI",   "%d",      "Futures open interest (contracts)"),
        f"{prefix}_chg_oi":    ("Chg OI",   "%d",      "Change in futures OI vs previous session"),
        f"{prefix}_settle":    ("Settle",   "₹%.2f",   "Futures settlement price"),
        f"{prefix}_basis_pct": ("Basis%",   "%.2f%%",  "(Futures − Spot) ÷ Spot × 100. +ve = premium"),
        f"{prefix}_call_oi":   ("Call OI",  "%d",      "Total call open interest for this expiry"),
        f"{prefix}_put_oi":    ("Put OI",   "%d",      "Total put open interest for this expiry"),
        f"{prefix}_pcr":       ("PCR",      "%.2f",    "Put/Call OI ratio. >1 = more puts (hedging), <0.7 = bullish"),
        f"{prefix}_max_pain":  ("Max Pain", "₹%.0f",   "Strike minimising option writers' payout at expiry"),
        f"{prefix}_mp_dist_pct": ("MP Dist%", "%.1f%%","(Spot − Max Pain) ÷ Max Pain × 100"),
    }
    present_cols  = [c for c in col_map if c in df.columns]
    display_names = {c: col_map[c][0] for c in present_cols}
    fmt_map       = {col_map[c][0]: col_map[c][1] for c in present_cols}
    help_map      = {col_map[c][0]: col_map[c][2] for c in present_cols}

    display = df[["symbol", "sector", "spot_price", "roll_signal"] + present_cols].copy()
    display = display.rename(columns={**display_names, "spot_price": "Spot",
                                      "symbol": "Symbol", "sector": "Sector",
                                      "roll_signal": "Roll Signal"})

    col_config: dict = {
        "Spot": st.column_config.NumberColumn("Spot", format="₹%.2f"),
    }
    for nice, fmt in fmt_map.items():
        col_config[nice] = st.column_config.NumberColumn(nice, format=fmt,
                                                          help=help_map.get(nice))

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


# ── Standalone Options Chain tab ───────────────────────────────────────────────

def _render_standalone_chain(trade_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_stock_monthly_expiries,
        cached_index_full_structure,
        cached_options_chain,
    )

    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown(
            "Enter any F&O symbol (stock or index). Select an expiry. The butterfly chart shows "
            "**Call OI on the left** (red) and **Put OI on the right** (green). "
            "The strike with maximum total OI is the key support/resistance. "
            "**Max Pain** (orange dashed line) is where option writers' losses are minimised — "
            "price tends to converge here heading into expiry."
        )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        raw_sym  = st.text_input("Symbol (e.g. RELIANCE, NIFTY)", value="NIFTY", key="sc_sym")
        symbol   = raw_sym.upper().strip()
    with c3:
        instr_type = st.radio("Type", ["Index", "Stock"], key="sc_type", horizontal=True)
    instrument = "OPTIDX" if instr_type == "Index" else "OPTSTK"

    if instrument == "OPTSTK":
        expiries = cached_stock_monthly_expiries(trade_date)
    else:
        idx_df   = cached_index_full_structure(trade_date, symbol)
        expiries = sorted(set(idx_df["expiry_date"].tolist())) if not idx_df.empty else []

    if not expiries:
        st.info(f"No expiry data found for **{symbol}** on {trade_date}.")
        return

    with c2:
        sel_expiry = st.selectbox(
            "Expiry",
            expiries,
            format_func=lambda d: d.strftime("%d %b '%y"),
            key="sc_exp",
        )

    chain = cached_options_chain(trade_date, symbol, sel_expiry, instrument)
    if chain.empty:
        st.info(f"No options chain for **{symbol}** / {sel_expiry.strftime('%d %b %y')}.")
        return

    _render_chain_chart(chain, symbol, sel_expiry, spot=None, key="sc_chain")
    st.divider()
    st.markdown("#### Full Chain Table")
    _render_chain_table(chain)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _render_chain_chart(
    chain: pd.DataFrame,
    symbol: str,
    expiry: date,
    spot: Optional[float],
    key: str = "chain",
) -> None:
    """Butterfly OI chart: CE bars on the left (negative x), PE bars on the right."""
    if chain.empty:
        return

    # Limit to strikes with at least 10% of max OI to avoid a very sparse chart
    threshold = chain["total_oi"].max() * 0.05
    vis = chain[chain["total_oi"] >= threshold].copy()
    if vis.empty:
        vis = chain.copy()

    strikes = vis["strike_price"].tolist()
    max_val = max(vis["ce_oi"].max(), vis["pe_oi"].max(), 1) * 1.15

    mp_strikes = chain.loc[chain["is_max_pain"] == True, "strike_price"]
    mp = float(mp_strikes.iloc[0]) if not mp_strikes.empty else None

    fig = go.Figure()
    fig.add_bar(
        name="Call OI (CE)",
        y=strikes,
        x=[-float(v) for v in vis["ce_oi"]],
        orientation="h",
        marker_color="rgba(239,83,80,0.85)",
        hovertemplate="Strike %{y}<br>CE OI: %{customdata:,.0f}<extra></extra>",
        customdata=vis["ce_oi"].tolist(),
    )
    fig.add_bar(
        name="Put OI (PE)",
        y=strikes,
        x=vis["pe_oi"].tolist(),
        orientation="h",
        marker_color="rgba(38,166,154,0.85)",
        hovertemplate="Strike %{y}<br>PE OI: %{customdata:,.0f}<extra></extra>",
        customdata=vis["pe_oi"].tolist(),
    )

    shapes: list[dict] = []
    annotations: list[dict] = []

    if spot:
        shapes.append(dict(
            type="line", x0=-max_val, x1=max_val,
            y0=spot, y1=spot,
            line=dict(color="#ffd600", width=2, dash="dot"),
        ))
        annotations.append(dict(
            x=max_val * 0.02, y=spot,
            text=f"  Spot ₹{spot:,.0f}",
            xanchor="left", font=dict(color="#ffd600", size=11),
            showarrow=False,
        ))

    if mp:
        shapes.append(dict(
            type="line", x0=-max_val, x1=max_val,
            y0=mp, y1=mp,
            line=dict(color="#ff9800", width=2, dash="dash"),
        ))
        annotations.append(dict(
            x=-max_val * 0.02, y=mp,
            text=f"Max Pain ₹{mp:,.0f}  ",
            xanchor="right", font=dict(color="#ff9800", size=11),
            showarrow=False,
        ))

    # Symmetric x-axis ticks
    tick_step = max_val / 2
    tickvals  = [-max_val, -tick_step, 0, tick_step, max_val]
    ticktext  = [_fmt_oi(abs(v)) for v in tickvals]

    fig.update_layout(
        title=f"{symbol} — Options Chain  |  {expiry.strftime('%d %b %y')}",
        barmode="overlay",
        height=max(400, len(vis) * 20 + 140),
        xaxis=dict(
            range=[-max_val, max_val],
            zeroline=True, zerolinecolor="#555",
            title="← CE OI   |   PE OI →",
            tickvals=tickvals, ticktext=ticktext,
        ),
        yaxis=dict(title="Strike Price"),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def _render_chain_table(chain: pd.DataFrame) -> None:
    st.dataframe(
        chain,
        use_container_width=True,
        hide_index=True,
        column_config={
            "strike_price":  st.column_config.NumberColumn("Strike",     format="%.0f"),
            "ce_oi":         st.column_config.NumberColumn("CE OI",      format="%d"),
            "ce_chg_oi":     st.column_config.NumberColumn("CE Chg OI",  format="%d"),
            "ce_close":      st.column_config.NumberColumn("CE Price",   format="₹%.2f"),
            "pe_oi":         st.column_config.NumberColumn("PE OI",      format="%d"),
            "pe_chg_oi":     st.column_config.NumberColumn("PE Chg OI",  format="%d"),
            "pe_close":      st.column_config.NumberColumn("PE Price",   format="₹%.2f"),
            "total_oi":      st.column_config.NumberColumn("Total OI",   format="%d"),
            "pcr_at_strike": st.column_config.NumberColumn("PCR",        format="%.2f"),
            "is_max_pain":   st.column_config.CheckboxColumn("Max Pain?"),
        },
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def render(trade_date: date) -> None:
    st.title("🗓️ F&O Expiry Structure")
    st.caption(
        f"As of **{trade_date.strftime('%d %b %Y (%a)')}**  |  "
        "Data from NSE F&O bhavcopy  |  All OI figures in contracts"
    )

    with st.expander("ℹ️ How to Read This Page", expanded=False):
        st.markdown("""
**Index Structure** — Full expiry ladder for NIFTY / BANKNIFTY / etc.
NIFTY has weekly options (every Tuesday) + near/next/far monthly + quarterly + long-dated expiries.
BANKNIFTY has monthly + quarterly. Others have monthly only.

**Stock Matrix** — All F&O stocks shown across Near / Next / Far month in parallel tabs.
| Column | What it means |
|--------|--------------|
| Fut OI | Total open futures contracts (near expiry most active) |
| Chg OI | Change vs previous session. +ve = new positions added, −ve = being closed |
| Basis% | (Futures − Spot) ÷ Spot × 100. Positive = futures at premium (bullish); negative = discount (bearish) |
| PCR | Put/Call OI ratio. >1 = more puts (hedges/bearish bets), <0.7 = call heavy (bullish) |
| Max Pain | Strike where option writers lose least — price gravitates here near expiry |
| MP Dist% | How far spot is from Max Pain. Positive = spot above max pain (may pull down) |

**Roll Signal** (near month):
- 🔄 **Rolling Fwd** — Near OI falling, next OI rising: participants moving positions to next expiry (expiry week behaviour)
- 📈 **Building** — Near OI rising > 50K: fresh positions being taken
- 📉 **Unwinding** — Both near and next OI falling: positions being closed, possible exit
- ⚖️ **Neutral** — Normal range

**Options Chain** — Butterfly chart: CE (calls) on the left, PE (puts) on the right.
High CE OI at a strike = call writing resistance. High PE OI at a strike = put writing support.
Max Pain (orange dashed line) = expiry settlement magnet.
        """)

    tab_idx, tab_stocks, tab_chain = st.tabs([
        "📊 Index Structure",
        "📋 Stock Matrix",
        "🔍 Options Chain",
    ])

    with tab_idx:
        _render_index_structure(trade_date)

    with tab_stocks:
        _render_stock_matrix(trade_date)

    with tab_chain:
        _render_standalone_chain(trade_date)
