"""
F&O Stock Signals — expiry-aware view with cross-expiry OI comparison.

Expiry structure (stocks): Near Month | Next Month | Far Month
All data tables filter to the selected expiry. A cross-expiry roll-signal
column shows whether smart money is building, rolling, or unwinding positions
across the near→next month boundary.

Composite 5-factor signal (score −2 to +2):
  OI Matrix (35%) | Cost of Carry (20%) | PCR (20%) | Rollover (15%) | Max Pain (10%)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_fno_composite_signals,
    cached_fno_dates_available,
    cached_stock_monthly_expiries,
    cached_stock_expiry_matrix,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_SIG_COLOR = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#69F0AE",
    "HOLD":        "#78909C",
    "SELL":        "#FF6D00",
    "STRONG SELL": "#D50000",
}
_SIG_EMOJI = {
    "STRONG BUY":  "🟢",
    "BUY":         "🟩",
    "HOLD":        "⚪",
    "SELL":        "🟠",
    "STRONG SELL": "🔴",
}
_ALL_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

_ROLL_EMOJI = {
    "Rolling Fwd": "🔄",
    "Building":    "📈",
    "Unwinding":   "📉",
    "Neutral":     "⚖️",
}
_ROLL_COLOR = {
    "Rolling Fwd": "#ff9800",
    "Building":    "#00C853",
    "Unwinding":   "#f44336",
    "Neutral":     "#78909C",
}
_EXP_NAMES    = ["Near Month", "Next Month", "Far Month"]
_EXP_PREFIXES = ["near", "next", "far"]


def _fmt_oi(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v  # already formatted — don't process again
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(fv):
        return "—"
    iv = int(fv)
    if abs(iv) >= 1_000_000:
        return f"{iv/1_000_000:.2f}M"
    return f"{iv/1_000:.0f}K"


# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    st.subheader("📋 F&O Stock Signals")

    with st.expander("📖 How to Read This Page", expanded=False):
        st.markdown("""
**Expiry Selector** — Choose Near / Next / Far month. All OI, Basis%, PCR and Max Pain numbers
come from that specific expiry. The **Roll Signal** column is *always* based on Near vs Next OI
comparison — it tells you where smart money is positioned across the near→next boundary.

| Roll Signal | Meaning | Trading Implication |
|-------------|---------|---------------------|
| 🔄 Rolling Fwd | Near OI falling, Next OI rising | Participants moving to next contract — normal pre-expiry behaviour |
| 📈 Building | Near OI rising > 50K | Fresh longs/shorts added in near month — strong conviction |
| 📉 Unwinding | Both near & next OI falling | Positions being closed — possible exit / hedges unwound |
| ⚖️ Neutral | Changes within normal range | No clear directional bias |

**Cross-Expiry Read (example):** RELIANCE near month OI ↓ + next month OI ↑ + price ↑
→ Participants rolling long positions forward = bullish continuation expected.

**OI-Price Matrix (the core signal — 35%):**

The most important futures signal comes from combining price direction with open interest change:

| Price | OI | Signal | Meaning |
|-------|----|--------|---------|
| ↑ | ↑ | **Long Buildup** 🟢 | Fresh longs entering — STRONG conviction |
| ↓ | ↑ | **Short Buildup** 🔴 | Fresh shorts entering — STRONG conviction |
| ↑ | ↓ | **Short Cover** 🟩 | Shorts exiting — weaker bullish |
| ↓ | ↓ | **Long Unwind** 🟠 | Longs exiting — weaker bearish |

Score magnitude scales with how large the price move and OI change are.

**5-Factor Composite Score:**
| Factor | Weight | Bullish when |
|--------|--------|-------------|
| 🔄 OI Matrix | 35% | Long Buildup (OI↑ + Price↑) |
| 🏦 Cost of Carry | 20% | Futures annualised basis > 7% fair value |
| 📊 PCR (Contrarian) | 20% | PCR > 1.5 (panic puts = floor signal) |
| 📊 Rollover | 15% | Near OI building OR forward roll with rising price |
| 🎯 Max Pain | 10% | Spot below max pain (only relevant ≤7 days to expiry) |

**Cost of Carry:** Scored vs India fair value (~7% p.a. = repo rate minus dividend yield).
Futures basis > 12% ann = bullish demand premium; backwardation = bearish / stress signal.

**Max Pain is expiry-proximity weighted** — nearly irrelevant when >14 days out,
full weight only in the final 3 days. Don't over-read it mid-month.
        """)

    fno_dates = cached_fno_dates_available()
    if not fno_dates:
        st.warning(
            "No F&O Bhavcopy data loaded yet.  Run:\n"
            "```\npython -m src.cli backfill-fno\n```"
        )
        return

    # ── Controls row ──────────────────────────────────────────────────────────
    col_date, col_oi, col_sig, col_sec = st.columns([2, 1, 2, 2])

    with col_date:
        fno_date = st.selectbox(
            "F&O Date",
            options=fno_dates,
            index=0,
            format_func=lambda d: d.strftime("%d %b %Y (%a)"),
            key="fnos_date",
        )
    with col_oi:
        min_oi_k = st.number_input(
            "Min Futures OI (K)",
            min_value=0, max_value=5000, value=50, step=50,
            key="fnos_min_oi",
        )
    min_oi = int(min_oi_k * 1000)

    # ── Expiry selector ───────────────────────────────────────────────────────
    expiries = cached_stock_monthly_expiries(fno_date)
    if expiries:
        exp_labels = [
            f"{_EXP_NAMES[i]} ({e.strftime('%d %b')})"
            for i, e in enumerate(expiries[:3])
        ]
    else:
        exp_labels = _EXP_NAMES[:]

    sel_exp = st.radio(
        "Expiry",
        options=exp_labels,
        horizontal=True,
        key="fnos_exp",
        help="Filter all OI, Basis%, PCR and Max Pain data to this specific expiry. "
             "Roll Signal always compares Near vs Next month.",
    )
    exp_idx    = exp_labels.index(sel_exp)
    exp_prefix = _EXP_PREFIXES[exp_idx]

    # ── Load data ─────────────────────────────────────────────────────────────
    df_sig = cached_fno_composite_signals(fno_date, min_fut_oi=min_oi)
    if df_sig.empty:
        st.info("No F&O stock data for this date / OI filter.")
        return

    df_mat = cached_stock_expiry_matrix(fno_date, min_fut_oi=min_oi)

    # Merge expiry matrix into composite signals
    if not df_mat.empty:
        mat_keep_cols = (
            ["symbol", "spot_price", "roll_signal"] +
            [f"{p}_{c}"
             for p in _EXP_PREFIXES
             for c in ["fut_oi", "chg_oi", "basis_pct", "pcr", "max_pain", "mp_dist_pct"]]
        )
        mat_keep_cols = [c for c in mat_keep_cols if c in df_mat.columns]
        df = df_sig.merge(df_mat[mat_keep_cols], on="symbol", how="left")
    else:
        df = df_sig.copy()

    # ── Signal / Sector filters ───────────────────────────────────────────────
    all_sectors = sorted(df["sector"].dropna().unique().tolist())

    with col_sig:
        sig_filter = st.multiselect(
            "Signal Filter",
            options=_ALL_SIGNALS,
            default=_ALL_SIGNALS,
            key="fnos_sig",
        )
    with col_sec:
        sec_filter = st.multiselect(
            "Sector Filter",
            options=all_sectors,
            default=all_sectors,
            key="fnos_sec",
        )

    active_sigs = sig_filter if sig_filter else _ALL_SIGNALS
    active_secs = sec_filter if sec_filter else all_sectors

    if not sig_filter:
        with col_sig:
            st.caption("ℹ️ No signal selected → showing all")
    if not sec_filter:
        with col_sec:
            st.caption("ℹ️ No sector selected → showing all")

    df_filtered = df[
        df["signal_label"].isin(active_sigs) &
        df["sector"].isin(active_secs)
    ].copy()

    # ── KPI bar ───────────────────────────────────────────────────────────────
    _render_kpi_bar(df, exp_prefix)
    st.divider()

    tab_sector, tab_all, tab_scanner = st.tabs([
        "🏗️ By Sector",
        "📋 All Stocks",
        "🔥 Scanner",
    ])

    with tab_sector:
        _render_sector_view(df_filtered, exp_prefix)

    with tab_all:
        _render_all_stocks(df_filtered, exp_prefix)

    with tab_scanner:
        _render_scanner(df)


# ── KPI bar ───────────────────────────────────────────────────────────────────

def _render_kpi_bar(df: pd.DataFrame, prefix: str = "near") -> None:
    sig_counts = df["signal_label"].value_counts()
    net_score  = df["composite_score"].sum()
    bias_label = "Bullish" if net_score > 15 else "Bearish" if net_score < -15 else "Neutral"

    # Row 1 — composite signals (always near-month based, fixed)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("F&O Stocks",    f"{len(df)}")
    c2.metric("🟢 STRONG BUY",  f"{sig_counts.get('STRONG BUY', 0)}")
    c3.metric("🟩 BUY",         f"{sig_counts.get('BUY', 0)}")
    c4.metric("⚪ HOLD",        f"{sig_counts.get('HOLD', 0)}")
    c5.metric("🟠 SELL",        f"{sig_counts.get('SELL', 0)}")
    c6.metric("🔴 STRONG SELL", f"{sig_counts.get('STRONG SELL', 0)}")
    c7.metric("Market Bias", bias_label, f"Net: {net_score:+.1f}")

    # Row 2 — expiry-specific metrics (CHANGE when expiry changes)
    oi_col  = f"{prefix}_fut_oi"
    chg_col = f"{prefix}_chg_oi"
    pcr_col = f"{prefix}_pcr"
    bas_col = f"{prefix}_basis_pct"

    if oi_col in df.columns:
        exp_name     = _EXP_NAMES[_EXP_PREFIXES.index(prefix)] if prefix in _EXP_PREFIXES else prefix.title()
        total_oi     = df[oi_col].fillna(0).sum()
        oi_up        = int((df[chg_col].fillna(0) > 0).sum()) if chg_col in df.columns else 0
        oi_down      = int((df[chg_col].fillna(0) < 0).sum()) if chg_col in df.columns else 0
        avg_pcr      = df[pcr_col].dropna().mean() if pcr_col in df.columns else None
        avg_basis    = df[bas_col].dropna().mean() if bas_col in df.columns else None

        st.caption(f"**{exp_name} expiry metrics** (updates per expiry selection)")
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric(
            f"{exp_name} Total OI",
            f"{total_oi/1_000_000:.1f}M" if total_oi >= 1_000_000 else f"{total_oi/1_000:.0f}K",
            help=f"Sum of all F&O stocks' futures OI for {exp_name}",
        )
        e2.metric(
            "OI Increasing",
            f"{oi_up} stocks",
            help=f"Stocks where {exp_name} futures OI rose vs previous session — fresh positions added",
        )
        e3.metric(
            "OI Decreasing",
            f"{oi_down} stocks",
            help=f"Stocks where {exp_name} futures OI fell — positions being closed",
        )
        e4.metric(
            "Avg PCR",
            f"{avg_pcr:.2f}" if avg_pcr is not None and not pd.isna(avg_pcr) else "—",
            help=f"Average Put/Call ratio across all stocks for {exp_name}. >1 = more puts (bearish hedge), <0.7 = more calls (bullish bets)",
        )
        e5.metric(
            "Avg Basis%",
            f"{avg_basis:+.2f}%" if avg_basis is not None and not pd.isna(avg_basis) else "—",
            help=f"Average futures basis (Futures − Spot) / Spot × 100 for {exp_name}. Positive = premium (bullish)",
        )

    # Row 3 — roll signal counts (change when thresholds trigger)
    if "roll_signal" in df.columns:
        rc = df["roll_signal"].value_counts()
        st.caption("**Roll Signal** — Near vs Next month OI comparison")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🔄 Rolling Fwd", rc.get("Rolling Fwd", 0),
                  help="Near OI falling 3%+ AND Next OI rising — rollover in progress")
        r2.metric("📈 Building",    rc.get("Building",    0),
                  help="Near OI rising 3%+ — fresh long/short positions added")
        r3.metric("📉 Unwinding",   rc.get("Unwinding",   0),
                  help="Both near & next OI falling — positions being closed")
        r4.metric("⚖️ Neutral",     rc.get("Neutral",     0))


# ── Sector View tab ───────────────────────────────────────────────────────────

def _render_sector_view(df: pd.DataFrame, prefix: str) -> None:
    if df.empty:
        st.info("No stocks match current filters.")
        return

    _cscore = {"STRONG BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG SELL": -2}
    records = []
    for sector, grp in df.groupby("sector"):
        sc  = grp["signal_label"].value_counts().to_dict()
        rc  = grp["roll_signal"].value_counts().to_dict() if "roll_signal" in grp.columns else {}
        records.append({
            "sector":       sector,
            "strong_buy":   sc.get("STRONG BUY",  0),
            "buy":          sc.get("BUY",          0),
            "hold":         sc.get("HOLD",         0),
            "sell":         sc.get("SELL",         0),
            "strong_sell":  sc.get("STRONG SELL",  0),
            "stock_count":  len(grp),
            "dominant":     grp["signal_label"].mode().iloc[0],
            "net_score":    float(grp["signal_label"].map(_cscore).fillna(0).sum()),
            "rolling_fwd":  rc.get("Rolling Fwd", 0),
            "building":     rc.get("Building",    0),
            "unwinding":    rc.get("Unwinding",   0),
        })
    sec_df = pd.DataFrame(records).sort_values("net_score", ascending=False)

    _sector_signal_chart(sec_df)
    _sector_oi_chart(df, prefix)
    st.markdown("---")

    for _, row in sec_df.iterrows():
        sector   = row["sector"]
        s_stocks = df[df["sector"] == sector]
        if s_stocks.empty:
            continue

        dom   = row["dominant"]
        emoji = _SIG_EMOJI.get(dom, "⚪")
        bull  = int(row["strong_buy"] + row["buy"])
        bear  = int(row["strong_sell"] + row["sell"])
        ns    = row["net_score"]
        roll_str = ""
        if row.get("rolling_fwd", 0) or row.get("building", 0) or row.get("unwinding", 0):
            roll_str = (
                f"  |  🔄{int(row['rolling_fwd'])} "
                f"📈{int(row['building'])} "
                f"📉{int(row['unwinding'])}"
            )

        header = (
            f"{emoji} **{sector}**  ({int(row['stock_count'])} stocks)  "
            f"🟢{bull}  🔴{bear}  Net: **{ns:+.0f}**{roll_str}"
        )
        with st.expander(header, expanded=False):
            for industry in sorted(s_stocks["industry"].unique()):
                ind_stocks = s_stocks[s_stocks["industry"] == industry]
                if ind_stocks.empty:
                    continue
                st.markdown(f"##### 🔹 {industry}  ({len(ind_stocks)} stocks)")
                _render_stock_table(ind_stocks, prefix)
                st.markdown("")


def _sector_signal_chart(sec_df: pd.DataFrame) -> None:
    fig = go.Figure()
    for sig, col in [
        ("STRONG BUY",  "#00C853"),
        ("BUY",         "#69F0AE"),
        ("HOLD",        "#607D8B"),
        ("SELL",        "#FF6D00"),
        ("STRONG SELL", "#D50000"),
    ]:
        key = sig.lower().replace(" ", "_")
        if key not in sec_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=sig,
            x=sec_df["sector"],
            y=sec_df[key],
            marker_color=col,
            # Each trace shows its own value; hovermode="x unified" combines them
            hovertemplate=f"<b>%{{x}}</b><br>{sig}: %{{y}} stocks<extra></extra>",
        ))
    fig.update_layout(
        title="Signal Distribution by Sector  (hover over any bar to see all signals for that sector)",
        barmode="stack",
        height=380,
        template="plotly_dark",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Stock Count",
        margin=dict(t=60, b=80),
        # Unified hover: shows all 5 signal values for the hovered sector at once
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(20,20,20,0.92)", bordercolor="#555", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key="sector_signal_chart")


def _sector_oi_chart(df: pd.DataFrame, prefix: str) -> None:
    """OI by sector for the selected expiry — updates when expiry changes."""
    oi_col = f"{prefix}_fut_oi"
    if oi_col not in df.columns:
        return

    exp_name = _EXP_NAMES[_EXP_PREFIXES.index(prefix)] if prefix in _EXP_PREFIXES else prefix.title()

    sector_oi = (
        df.groupby("sector")[oi_col]
        .sum()
        .dropna()
        .sort_values(ascending=False)
        .reset_index()
    )
    sector_oi.columns = ["sector", "total_oi"]
    if sector_oi.empty:
        return

    # Colour each bar by dominant roll signal for that sector
    roll_dominant = {}
    if "roll_signal" in df.columns:
        for sec, grp in df.groupby("sector"):
            roll_dominant[sec] = grp["roll_signal"].mode().iloc[0] if not grp.empty else "Neutral"

    colors = [
        _ROLL_COLOR.get(roll_dominant.get(s, "Neutral"), "#1976d2")
        for s in sector_oi["sector"]
    ]

    fig = go.Figure(go.Bar(
        x=sector_oi["sector"],
        y=sector_oi["total_oi"] / 1_000_000,
        marker_color=colors,
        text=(sector_oi["total_oi"] / 1_000_000).apply(lambda v: f"{v:.1f}M"),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{exp_name} Fut OI: %{{y:.2f}}M contracts<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=(
            f"{exp_name} — Total Futures OI by Sector  "
            f"(bar colour = dominant roll signal: "
            f"<span style='color:#00C853'>■</span> Building  "
            f"<span style='color:#ff9800'>■</span> Rolling Fwd  "
            f"<span style='color:#f44336'>■</span> Unwinding  "
            f"<span style='color:#78909C'>■</span> Neutral)"
        ),
        height=360,
        template="plotly_dark",
        xaxis_tickangle=-35,
        yaxis_title="Open Interest (Million contracts)",
        margin=dict(t=70, b=80),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(20,20,20,0.92)", bordercolor="#555", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"sector_oi_chart_{prefix}")


# ── All Stocks tab ────────────────────────────────────────────────────────────

def _render_all_stocks(df: pd.DataFrame, prefix: str) -> None:
    st.markdown(f"**{len(df)} stocks** after filters")

    # Decide comparison columns: always show Near + Next OI side-by-side
    oi_col   = f"{prefix}_fut_oi"
    chg_col  = f"{prefix}_chg_oi"
    bas_col  = f"{prefix}_basis_pct"
    pcr_col  = f"{prefix}_pcr"
    mp_col   = f"{prefix}_max_pain"
    mpd_col  = f"{prefix}_mp_dist_pct"

    has_matrix = oi_col in df.columns

    sort_base = {
        "Composite Score ↓": ("composite_score", False),
        "Composite Score ↑": ("composite_score", True),
        "Futures OI ↓":      (oi_col if has_matrix else "fut_oi", False),
        "Price Chg % ↓":     ("price_chg_pct", False),
        "Price Chg % ↑":     ("price_chg_pct", True),
        "PCR ↓":             (pcr_col if has_matrix else "stock_pcr", False),
        "PCR ↑":             (pcr_col if has_matrix else "stock_pcr", True),
        "CoC % ↓":           ("coc_pct", False),
        "Near OI ↓":         ("near_fut_oi", False) if "near_fut_oi" in df.columns else ("composite_score", False),
        "Next OI ↓":         ("next_fut_oi", False) if "next_fut_oi" in df.columns else ("composite_score", False),
    }
    # Drop options whose sort col doesn't exist
    sort_opts = {k: v for k, v in sort_base.items() if v[0] in df.columns}

    col_sort, col_roll = st.columns([2, 2])
    with col_sort:
        sort_col = st.selectbox("Sort by", list(sort_opts.keys()), key="fnos_sort")
    with col_roll:
        if "roll_signal" in df.columns:
            roll_filter = st.multiselect(
                "Roll Signal filter",
                options=["Rolling Fwd", "Building", "Unwinding", "Neutral"],
                key="fnos_roll",
            )
        else:
            roll_filter = []

    scol, asc = sort_opts[sort_col]
    df_view = df.copy()
    if roll_filter:
        df_view = df_view[df_view["roll_signal"].isin(roll_filter)]
    df_view = df_view.sort_values(scol, ascending=asc, na_position="last")

    if df_view.empty:
        st.info("No stocks match current roll signal filter.")
        return

    # ── Build display table ───────────────────────────────────────────────────
    base_cols = ["symbol", "company_name", "sector"]
    signal_cols = ["signal_label", "composite_score"]

    if has_matrix:
        # Expiry-specific columns
        exp_cols = [c for c in [oi_col, chg_col, bas_col, pcr_col, mp_col, mpd_col]
                    if c in df_view.columns]
        # Cross-expiry OI comparison (always Near + Next)
        cross_cols = [c for c in ["near_fut_oi", "next_fut_oi", "far_fut_oi"]
                      if c in df_view.columns and c != oi_col]
        roll_cols = ["roll_signal"] if "roll_signal" in df_view.columns else []
        spot_cols = ["spot_price"] if "spot_price" in df_view.columns else []

        all_cols = base_cols + spot_cols + roll_cols + exp_cols + cross_cols + signal_cols
    else:
        # Fallback: original composite-signal columns
        all_cols = base_cols + ["signal_label", "composite_score",
                                 "price_chg_pct", "coc_pct", "stock_pcr",
                                 "mp_distance_pct", "fut_oi"]
        all_cols = [c for c in all_cols if c in df_view.columns]

    disp = df_view[all_cols].copy()

    # Format expiry-specific OI / numeric columns
    def _fmt_v(v, fmt):
        if pd.isna(v) or v is None:
            return "—"
        return fmt.format(v)

    # dict.fromkeys deduplicates while preserving order — avoids double-formatting
    # when oi_col == "near_fut_oi" (prefix="near"), which causes int("1.43M") crash
    for c in dict.fromkeys([oi_col, "near_fut_oi", "next_fut_oi", "far_fut_oi"]):
        if c in disp.columns:
            disp[c] = disp[c].apply(_fmt_oi)
    for c in [chg_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: _fmt_oi(v) if pd.notna(v) else "—")
    for c in [bas_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    for c in [pcr_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    for c in [mp_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
    for c in [mpd_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
    if "spot_price" in disp.columns:
        disp["spot_price"] = disp["spot_price"].apply(
            lambda v: f"₹{v:,.2f}" if pd.notna(v) else "—"
        )
    if "roll_signal" in disp.columns:
        disp["roll_signal"] = disp["roll_signal"].apply(
            lambda s: f"{_ROLL_EMOJI.get(s, '')} {s}" if pd.notna(s) else "—"
        )
    if "signal_label" in disp.columns:
        disp["signal_label"] = disp["signal_label"].map(
            lambda s: f"{_SIG_EMOJI.get(s, '')} {s}"
        )
    if "composite_score" in disp.columns:
        disp["composite_score"] = disp["composite_score"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        )

    # Human-readable column names
    exp_name = _EXP_NAMES[_EXP_PREFIXES.index(prefix)] if prefix in _EXP_PREFIXES else prefix.title()
    col_rename = {
        "symbol":        "Symbol",
        "company_name":  "Company",
        "sector":        "Sector",
        "spot_price":    "Spot",
        "roll_signal":   "Roll Signal",
        oi_col:          f"{exp_name} Fut OI",
        chg_col:         f"{exp_name} Chg OI",
        bas_col:         f"{exp_name} Basis%",
        pcr_col:         f"{exp_name} PCR",
        mp_col:          f"{exp_name} Max Pain",
        mpd_col:         f"{exp_name} MP Dist%",
        "near_fut_oi":   "Near OI",
        "next_fut_oi":   "Next OI",
        "far_fut_oi":    "Far OI",
        "signal_label":  "Signal",
        "composite_score": "Score",
        # fallback names
        "price_chg_pct": "Fut Chg%",
        "coc_pct":       "CoC%",
        "stock_pcr":     "PCR",
        "mp_distance_pct": "MP Dist%",
        "fut_oi":        "Fut OI",
    }
    disp = disp.rename(columns={k: v for k, v in col_rename.items() if k in disp.columns})

    st.dataframe(disp, use_container_width=True, hide_index=True, height=580,
                 column_config={
                     "Roll Signal": st.column_config.TextColumn("Roll Signal",
                         help="Near vs Next month OI comparison.\n\n"
                              "🔄 Rolling Fwd = Near OI↓ + Next OI↑ (rolling positions forward)\n"
                              "📈 Building = Near OI rising >50K (fresh positions)\n"
                              "📉 Unwinding = Both OI falling (exiting)\n"
                              "⚖️ Neutral = within normal range"),
                     "Near OI": st.column_config.TextColumn("Near OI",
                         help="Near month futures open interest — always shown for cross-expiry comparison"),
                     "Next OI": st.column_config.TextColumn("Next OI",
                         help="Next month futures open interest — compare with Near OI to read rollover activity"),
                     "Far OI":  st.column_config.TextColumn("Far OI",
                         help="Far month futures open interest"),
                     f"{exp_name} Basis%": st.column_config.TextColumn(f"{exp_name} Basis%",
                         help="(Futures settlement − Spot) ÷ Spot × 100\n\n"
                              "Positive = futures premium (bulls paying carry, bullish)\n"
                              "Negative = futures discount (hedging / roll pressure, bearish)"),
                     f"{exp_name} PCR": st.column_config.TextColumn(f"{exp_name} PCR",
                         help="Put/Call OI ratio for this expiry.\n\n"
                              "PCR > 1.5 = panic puts → contrarian bullish\n"
                              "PCR < 0.7 = call heavy → contrarian bearish"),
                     f"{exp_name} Max Pain": st.column_config.TextColumn(f"{exp_name} Max Pain",
                         help="Strike where option writers' total loss is minimised.\n"
                              "Price gravitates here heading into expiry."),
                 })


# ── Stock drilldown table (used inside sector expanders) ──────────────────────

def _render_stock_table(df: pd.DataFrame, prefix: str) -> None:
    oi_col  = f"{prefix}_fut_oi"
    chg_col = f"{prefix}_chg_oi"
    pcr_col = f"{prefix}_pcr"
    mp_col  = f"{prefix}_max_pain"

    has_matrix = oi_col in df.columns

    if has_matrix:
        base = ["symbol", "company_name", "signal_label", "composite_score",
                "roll_signal", "oi_matrix_signal", oi_col, chg_col, pcr_col, mp_col,
                "price_chg_pct", "coc_ann",
                "score_oi", "score_coc", "score_pcr", "score_roll", "score_mp"]
        cols = [c for c in base if c in df.columns]
    else:
        cols = ["symbol", "company_name", "signal_label", "composite_score",
                "price_chg_pct", "coc_ann", "stock_pcr", "mp_distance_pct",
                "score_oi", "score_coc", "score_pcr", "score_roll", "score_mp", "fut_oi"]
        cols = [c for c in cols if c in df.columns]

    disp = df[cols].copy()

    if "signal_label" in disp.columns:
        disp["signal_label"] = disp["signal_label"].map(lambda s: f"{_SIG_EMOJI.get(s, '')} {s}")
    if "composite_score" in disp.columns:
        disp["composite_score"] = disp["composite_score"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        )
    if "roll_signal" in disp.columns:
        disp["roll_signal"] = disp["roll_signal"].apply(
            lambda s: f"{_ROLL_EMOJI.get(s, '')} {s}" if pd.notna(s) else "—"
        )
    for c in [oi_col, chg_col, "fut_oi"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(_fmt_oi)
    for c in [pcr_col, "stock_pcr"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    for c in [mp_col]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
    for c in ["price_chg_pct", "coc_pct", "coc_ann"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    if "mp_distance_pct" in disp.columns:
        disp["mp_distance_pct"] = disp["mp_distance_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )

    exp_name = _EXP_NAMES[_EXP_PREFIXES.index(prefix)] if prefix in _EXP_PREFIXES else prefix.title()
    rename = {
        "symbol": "Symbol", "company_name": "Company",
        "signal_label": "Signal", "composite_score": "Score",
        "roll_signal": "Roll", "oi_matrix_signal": "OI Matrix",
        oi_col: "Fut OI", chg_col: "Chg OI",
        pcr_col: "PCR", mp_col: "Max Pain",
        "price_chg_pct": "Fut Chg%", "coc_ann": "CoC Ann%",
        "coc_pct": "CoC%",
        "stock_pcr": "PCR", "mp_distance_pct": "MP Dist%",
        "fut_oi": "Fut OI",
        "score_oi": "🔄 OI", "score_coc": "🏦",
        "score_pcr": "📊", "score_roll": "📊 Roll", "score_mp": "🎯",
    }
    disp = disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns})
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Scanner tab ───────────────────────────────────────────────────────────────

def _render_scanner(df: pd.DataFrame) -> None:
    col_bull, col_bear = st.columns(2)

    top_buy  = df[df["composite_score"] > 0].nlargest(15, "composite_score")
    top_sell = df[df["composite_score"] < 0].nsmallest(15, "composite_score")

    with col_bull:
        st.markdown("#### 🟢 Top Bullish")
        if not top_buy.empty:
            st.plotly_chart(
                _score_bar_chart(top_buy, title="Top Bullish", color="#00C853"),
                use_container_width=True, key="scanner_bull_chart",
            )
            _scanner_table(top_buy)
        else:
            st.info("No bullish stocks.")

    with col_bear:
        st.markdown("#### 🔴 Top Bearish")
        if not top_sell.empty:
            st.plotly_chart(
                _score_bar_chart(top_sell, ascending=True, title="Top Bearish", color="#D50000"),
                use_container_width=True, key="scanner_bear_chart",
            )
            _scanner_table(top_sell)
        else:
            st.info("No bearish stocks.")

    # ── Factor heatmap ────────────────────────────────────────────────────────
    st.markdown("#### 🧬 Factor Breakdown — Top 30 by |Score|")
    top30 = df.iloc[df["composite_score"].abs().argsort()[::-1].iloc[:30]]

    score_cols = [c for c in ["score_oi", "score_coc", "score_pcr", "score_roll", "score_mp"]
                  if c in top30.columns]
    if score_cols:
        labels = {"score_oi": "🔄 OI Matrix", "score_coc": "🏦 CoC (Ann%)",
                  "score_pcr": "📊 PCR", "score_roll": "📊 Rollover", "score_mp": "🎯 MaxPain"}
        fig_heat = go.Figure(data=go.Heatmap(
            z=[top30[c].tolist() for c in score_cols],
            x=top30["symbol"].tolist(),
            y=[labels.get(c, c) for c in score_cols],
            colorscale=[
                [0.0, "#D50000"], [0.25, "#FF6D00"],
                [0.5,  "#607D8B"], [0.75, "#69F0AE"], [1.0, "#00C853"],
            ],
            zmin=-2, zmax=2,
            text=[top30[c].apply(lambda v: f"{int(v):+d}").tolist() for c in score_cols],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat.update_layout(
            title="Factor Scores (−2 bearish → +2 bullish)",
            height=320, template="plotly_dark",
            xaxis_tickangle=-40, margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="scanner_heatmap")

    # ── Top OI + roll signal ──────────────────────────────────────────────────
    st.markdown("#### 💎 Highest Futures OI + Roll Signal")
    oi_src = "near_fut_oi" if "near_fut_oi" in df.columns else "fut_oi"
    top_oi = df.nlargest(20, oi_src) if oi_src in df.columns else pd.DataFrame()

    if not top_oi.empty:
        colors = [_SIG_COLOR.get(s, "#9E9E9E") for s in top_oi["signal_label"]]
        oi_vals = top_oi[oi_src].fillna(0) / 1_000_000

        hover_roll = (top_oi["roll_signal"].map(
            lambda s: f"{_ROLL_EMOJI.get(s, '')} {s}"
        ).tolist() if "roll_signal" in top_oi.columns
        else [""] * len(top_oi))

        fig_oi = go.Figure(go.Bar(
            x=top_oi["symbol"],
            y=oi_vals,
            marker_color=colors,
            text=top_oi["signal_label"].map(lambda s: _SIG_EMOJI.get(s, "")),
            textposition="outside",
            customdata=hover_roll,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "OI: %{y:.2f}M<br>"
                "Roll: %{customdata}<extra></extra>"
            ),
        ))
        fig_oi.update_layout(
            title="Top 20 by Near Futures OI (colour = signal, hover = roll signal)",
            height=340, template="plotly_dark",
            yaxis_title="OI (Million)", xaxis_tickangle=-40,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_oi, use_container_width=True, key="scanner_oi_chart")

    # ── Roll signal breakdown table ───────────────────────────────────────────
    if "roll_signal" in df.columns:
        st.markdown("#### 🔄 Roll Signal Breakdown")
        r1, r2 = st.columns(2)

        rolling = df[df["roll_signal"] == "Rolling Fwd"].sort_values(
            "composite_score", ascending=False
        )
        building = df[df["roll_signal"] == "Building"].sort_values(
            "composite_score", ascending=False
        )

        with r1:
            st.markdown(f"**🔄 Rolling Fwd** ({len(rolling)} stocks)")
            if not rolling.empty:
                _scanner_table(rolling.head(15), show_roll=True)
            else:
                st.info("None")
        with r2:
            st.markdown(f"**📈 Building** ({len(building)} stocks)")
            if not building.empty:
                _scanner_table(building.head(15), show_roll=False)
            else:
                st.info("None")


def _score_bar_chart(
    df: pd.DataFrame,
    title: str = "",
    color: str = "#00C853",
    ascending: bool = False,
) -> go.Figure:
    df_s = df.sort_values("composite_score", ascending=ascending)
    fig = go.Figure(go.Bar(
        x=df_s["symbol"],
        y=df_s["composite_score"],
        marker_color=color,
        text=df_s["composite_score"].apply(lambda v: f"{v:+.2f}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=title, height=300, template="plotly_dark",
        yaxis_title="Composite Score", xaxis_tickangle=-35,
        margin=dict(t=50, b=60),
    )
    return fig


def _scanner_table(df: pd.DataFrame, show_roll: bool = True) -> None:
    base = ["symbol", "sector", "signal_label", "composite_score",
            "oi_matrix_signal", "price_chg_pct", "coc_ann", "stock_pcr"]

    if show_roll and "roll_signal" in df.columns:
        cols_wanted = base + ["roll_signal", "near_fut_oi", "next_fut_oi"]
    else:
        cols_wanted = base + ["near_fut_oi"] if "near_fut_oi" in df.columns else base + ["fut_oi"]

    cols = [c for c in cols_wanted if c in df.columns]
    disp = df[cols].copy()

    if "signal_label" in disp.columns:
        disp["signal_label"] = disp["signal_label"].map(lambda s: f"{_SIG_EMOJI.get(s, '')} {s}")
    if "composite_score" in disp.columns:
        disp["composite_score"] = disp["composite_score"].apply(lambda v: f"{v:+.2f}")
    if "roll_signal" in disp.columns:
        disp["roll_signal"] = disp["roll_signal"].apply(
            lambda s: f"{_ROLL_EMOJI.get(s, '')} {s}" if pd.notna(s) else "—"
        )
    for c in ["price_chg_pct", "coc_ann"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    if "stock_pcr" in disp.columns:
        disp["stock_pcr"] = disp["stock_pcr"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    for c in ["near_fut_oi", "next_fut_oi", "fut_oi"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(_fmt_oi)

    rename = {
        "symbol": "Symbol", "sector": "Sector",
        "signal_label": "Signal", "composite_score": "Score",
        "oi_matrix_signal": "OI Matrix",
        "price_chg_pct": "Fut Chg%", "coc_ann": "CoC Ann%", "stock_pcr": "PCR",
        "roll_signal": "Roll Signal",
        "near_fut_oi": "Near OI", "next_fut_oi": "Next OI", "fut_oi": "Fut OI",
    }
    disp = disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns})
    st.dataframe(disp, use_container_width=True, hide_index=True)
