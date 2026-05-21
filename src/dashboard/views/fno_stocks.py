"""
F&O Stock Signals — institutional-grade 5-factor composite signal per stock.

Factors (weighted composite score -2.0 to +2.0):
  1. Price Direction  (25%) — futures move vs yesterday's settlement
  2. Cost of Carry    (25%) — futures premium/discount to spot
  3. PCR Contrarian   (20%) — put/call OI ratio (crowd positioning)
  4. Max Pain Gravity (20%) — spot distance from max-pain strike
  5. Volume Activity  (10%) — turnover × price direction
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_fno_composite_signals,
    cached_fno_dates_available,
)

# ── Colour / label maps ───────────────────────────────────────────────────────
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
_SCORE_COL = {  # for factor score columns (+2/-2 integer)
    2:  "#00C853",
    1:  "#69F0AE",
    0:  "#78909C",
    -1: "#FF6D00",
    -2: "#D50000",
}
_ALL_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]


def render(selected_date: date) -> None:
    st.subheader("📋 F&O Stock Signals")

    with st.expander("📖 How to Read This Page", expanded=False):
        st.markdown("""
**What is this page?**
Every F&O stock is scored daily on 5 factors. Each factor produces a sub-score from **-2 (bearish) to +2 (bullish)**. The weighted composite score drives the final signal.

| Factor | Weight | What it measures | Bullish condition |
|--------|--------|-----------------|-------------------|
| **📈 Price Direction** | 25% | Futures close vs yesterday's settlement | Fut close > yesterday settlement |
| **🏦 Cost of Carry (CoC)** | 25% | Futures premium/discount vs same-day spot | Futures at premium → longs paying to hold |
| **📊 PCR (Contrarian)** | 20% | Put/Call OI ratio — crowd positioning | PCR > 1.5 = too many puts → contrarian bullish |
| **🎯 Max Pain** | 20% | Spot price distance from max-pain strike | Spot BELOW max pain → bullish magnetic pull |
| **📦 Volume Activity** | 10% | Turnover magnitude × price direction | High volume + rising price |

**Signal thresholds (composite score):**
| Signal | Score range | Meaning |
|--------|-------------|---------|
| 🟢 STRONG BUY | ≥ 1.0 | All/most factors bullish |
| 🟩 BUY | ≥ 0.4 | More bullish than bearish |
| ⚪ HOLD | -0.4 to +0.4 | Mixed or flat signals |
| 🟠 SELL | -1.0 to -0.4 | More bearish factors |
| 🔴 STRONG SELL | ≤ -1.0 | Strong multi-factor bearish alignment |

**Key terms:**
- **Fut Chg%** — futures close vs *yesterday's settlement* (not today's open). Captures overnight conviction.
- **CoC%** — positive means futures trade at a premium to spot (bulls paying carry). Negative = discount (roll pressure, bearish).
- **PCR** — contrarian: high puts mean crowd is fearful → often a bottom signal. Low puts = complacency → top signal.
- **MP Dist%** — how far spot is from the max-pain strike. Negative = spot below max pain (bullish gravity). Positive = spot above (gravity pulls it down).
- **Factor scores** (📈🏦📊🎯📦) show the raw -2 to +2 sub-score for each factor — useful to see *which* factor is driving the signal.
        """)

    fno_dates = cached_fno_dates_available()
    if not fno_dates:
        st.warning(
            "No F&O Bhavcopy data loaded yet.  Run:\n"
            "```\npython -m src.cli backfill-fno\n```"
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
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

    df = cached_fno_composite_signals(fno_date, min_fut_oi=min_oi)
    if df.empty:
        st.info("No F&O stock data for this date / OI filter.")
        return

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

    # Empty multiselect = no filter applied (show all), not "hide everything"
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
    _render_kpi_bar(df)
    st.divider()

    tab_sector, tab_all, tab_scanner = st.tabs([
        "🏗️ By Sector",
        "📋 All Stocks",
        "🔥 Scanner",
    ])

    with tab_sector:
        _render_sector_view(df_filtered)

    with tab_all:
        _render_all_stocks(df_filtered)

    with tab_scanner:
        _render_scanner(df)


# ── KPI bar ───────────────────────────────────────────────────────────────────

def _render_kpi_bar(df: pd.DataFrame) -> None:
    sig_counts = df["signal_label"].value_counts()
    net_score  = df["composite_score"].sum()

    bias_label = (
        "Bullish" if net_score > 15 else
        "Bearish" if net_score < -15 else
        "Neutral"
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("F&O Stocks",    f"{len(df)}")
    c2.metric("🟢 STRONG BUY",  f"{sig_counts.get('STRONG BUY', 0)}")
    c3.metric("🟩 BUY",         f"{sig_counts.get('BUY', 0)}")
    c4.metric("⚪ HOLD",        f"{sig_counts.get('HOLD', 0)}")
    c5.metric("🟠 SELL",        f"{sig_counts.get('SELL', 0)}")
    c6.metric("🔴 STRONG SELL", f"{sig_counts.get('STRONG SELL', 0)}")
    c7.metric(
        "Market Bias",
        bias_label,
        f"Net: {net_score:+.1f}",
    )


# ── Sector View tab ───────────────────────────────────────────────────────────

def _render_sector_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No stocks match current filters.")
        return

    # Build sector summary from filtered df so counts == displayed stocks
    _cscore = {"STRONG BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG SELL": -2}
    records = []
    for sector, grp in df.groupby("sector"):
        sc = grp["signal_label"].value_counts().to_dict()
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
        })
    sec_df = pd.DataFrame(records).sort_values("net_score", ascending=False)

    _sector_signal_chart(sec_df)
    st.markdown("---")

    for _, row in sec_df.iterrows():
        sector  = row["sector"]
        s_stocks = df[df["sector"] == sector]
        if s_stocks.empty:
            continue

        dom   = row["dominant"]
        emoji = _SIG_EMOJI.get(dom, "⚪")
        bull  = int(row["strong_buy"] + row["buy"])
        bear  = int(row["strong_sell"] + row["sell"])
        ns    = row["net_score"]

        header = (
            f"{emoji} **{sector}**  ({int(row['stock_count'])} stocks)  "
            f"🟢{bull}  🔴{bear}  Net: **{ns:+.0f}**"
        )
        with st.expander(header, expanded=False):
            for industry in sorted(s_stocks["industry"].unique()):
                ind_stocks = s_stocks[s_stocks["industry"] == industry]
                if ind_stocks.empty:
                    continue
                st.markdown(f"##### 🔹 {industry}  ({len(ind_stocks)} stocks)")
                _render_stock_table(ind_stocks)
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
        ))

    fig.update_layout(
        title="Signal Distribution by Sector",
        barmode="stack",
        height=380,
        template="plotly_dark",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Stock Count",
        margin=dict(t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_stock_table(df: pd.DataFrame) -> None:
    disp = df[[
        "symbol", "company_name", "signal_label",
        "composite_score",
        "price_chg_pct", "coc_pct", "stock_pcr",
        "mp_distance_pct",
        "score_price", "score_coc", "score_pcr", "score_mp", "score_vol",
        "fut_oi",
    ]].copy()

    disp["signal_label"] = disp["signal_label"].map(
        lambda s: f"{_SIG_EMOJI.get(s, '')} {s}"
    )
    disp["composite_score"] = disp["composite_score"].apply(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
    )
    disp["price_chg_pct"] = disp["price_chg_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["coc_pct"] = disp["coc_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["mp_distance_pct"] = disp["mp_distance_pct"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
    )
    disp["stock_pcr"] = disp["stock_pcr"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )
    disp["fut_oi"] = disp["fut_oi"].apply(
        lambda v: f"{v/1_000_000:.2f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K"
    )

    disp.columns = [
        "Symbol", "Company", "Signal",
        "Score",
        "Fut Chg%", "CoC%", "PCR",
        "MP Dist%",
        "📈Price", "🏦CoC", "📊PCR", "🎯MaxPain", "📦Vol",
        "Fut OI",
    ]
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config={
        "Symbol":    st.column_config.TextColumn("Symbol",   help="NSE ticker symbol"),
        "Company":   st.column_config.TextColumn("Company",  help="Company full name from NSE sector master"),
        "Signal":    st.column_config.TextColumn("Signal",
            help="5-factor composite signal\n\n"
                 "STRONG BUY ≥ 1.0  |  BUY ≥ 0.4  |  HOLD  |  SELL > -1.0  |  STRONG SELL ≤ -1.0"),
        "Score":     st.column_config.TextColumn("Score",
            help="Composite Score (−2.0 to +2.0)\n\n"
                 "= 25%×Price + 25%×CoC + 20%×PCR + 20%×MaxPain + 10%×Volume"),
        "Fut Chg%":  st.column_config.TextColumn("Fut Chg%",
            help="Futures Price Change %\n\n"
                 "Formula: (fut_close − yesterday_settlement) / yesterday_settlement × 100\n\n"
                 "Scoring: ≥+2% → +2  |  ≥+0.5% → +1  |  0 → 0  |  ≤−0.5% → −1  |  ≤−2% → −2"),
        "CoC%":      st.column_config.TextColumn("CoC%",
            help="Cost of Carry %\n\n"
                 "Formula: (fut_close − spot_close) / spot_close × 100\n\n"
                 "Positive = futures premium (bulls paying carry) → bullish\n"
                 "Negative = futures discount (roll pressure / hedging) → bearish"),
        "PCR":       st.column_config.TextColumn("PCR",
            help="Put/Call Ratio (near-month options)\n\n"
                 "Formula: Put OI / Call OI  — contrarian indicator\n\n"
                 "PCR > 1.5 = panic puts → contrarian bullish (+2)\n"
                 "PCR < 0.5 = call frenzy → contrarian bearish (−2)"),
        "MP Dist%":  st.column_config.TextColumn("MP Dist%",
            help="Max Pain Distance %\n\n"
                 "Formula: (spot_price − max_pain_strike) / max_pain_strike × 100\n\n"
                 "Negative = spot BELOW max pain → bullish gravity (price pulled UP toward max pain)\n"
                 "Positive = spot ABOVE max pain → bearish gravity (price pulled DOWN)\n\n"
                 "Max Pain = strike where total option-writer payout is minimised"),
        "📈Price":   st.column_config.TextColumn("📈 Price Score",
            help="Price Direction factor score (−2 to +2)\n\n"
                 "Based on Futures Chg% vs yesterday's settlement\n"
                 "+2 = strong up  |  +1 = up  |  0 = flat  |  −1 = down  |  −2 = strong down\n\n"
                 "Weight in composite: 25%"),
        "🏦CoC":     st.column_config.TextColumn("🏦 CoC Score",
            help="Cost of Carry factor score (−2 to +2)\n\n"
                 "Based on CoC% (futures premium vs spot)\n"
                 "+2 = strong premium  |  0 = neutral  |  −2 = strong discount\n\n"
                 "Weight in composite: 25%"),
        "📊PCR":     st.column_config.TextColumn("📊 PCR Score",
            help="PCR Contrarian factor score (−2 to +2)\n\n"
                 "Contrarian: high put bias = bullish signal\n"
                 "+2 = PCR > 1.5  |  +1 = PCR 1.0–1.5  |  0 = neutral  |  −2 = PCR < 0.5\n\n"
                 "Weight in composite: 20%"),
        "🎯MaxPain": st.column_config.TextColumn("🎯 Max Pain Score",
            help="Max Pain Gravity factor score (−2 to +2)\n\n"
                 "Spot BELOW max pain → bullish (+score)  |  Spot ABOVE → bearish\n"
                 "+2 = spot ≥3% below max pain  |  0 = at max pain  |  −2 = ≥3% above\n\n"
                 "Weight in composite: 20%"),
        "📦Vol":     st.column_config.TextColumn("📦 Volume Score",
            help="Volume Activity factor score (−2 to +2)\n\n"
                 "High volume (>1.5× median) amplifies price direction:\n"
                 "+2 = high vol + price up >1%  |  +1 = high vol + price up\n"
                 "−1 = high vol + price down  |  −2 = high vol + price down >1%  |  0 = normal vol\n\n"
                 "Weight in composite: 10%"),
        "Fut OI":    st.column_config.TextColumn("Fut OI",
            help="Futures Open Interest (near-month)\n\n"
                 "Total outstanding contracts. High OI = large institutional positioning"),
    })


# ── All Stocks tab ────────────────────────────────────────────────────────────

def _render_all_stocks(df: pd.DataFrame) -> None:
    st.markdown(f"**{len(df)} stocks** after filters")

    sort_opts = {
        "Composite Score (↓)": ("composite_score", False),
        "Composite Score (↑)": ("composite_score", True),
        "Futures OI (↓)":      ("fut_oi",           False),
        "Price Chg % (↓)":     ("price_chg_pct",    False),
        "Price Chg % (↑)":     ("price_chg_pct",    True),
        "PCR (↓)":             ("stock_pcr",        False),
        "PCR (↑)":             ("stock_pcr",        True),
        "CoC % (↓)":           ("coc_pct",          False),
    }
    sort_col = st.selectbox(
        "Sort by",
        options=list(sort_opts.keys()),
        key="fnos_sort",
    )
    col, asc = sort_opts[sort_col]
    df_sorted = df.sort_values(col, ascending=asc, na_position="last")

    disp = df_sorted[[
        "symbol", "company_name", "sector",
        "signal_label", "composite_score",
        "price_chg_pct", "coc_pct", "stock_pcr", "mp_distance_pct",
        "score_price", "score_coc", "score_pcr", "score_mp", "score_vol",
        "fut_oi", "value_lacs",
    ]].copy()

    disp["signal_label"] = disp["signal_label"].map(
        lambda s: f"{_SIG_EMOJI.get(s, '')} {s}"
    )
    disp["composite_score"] = disp["composite_score"].apply(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
    )
    disp["price_chg_pct"] = disp["price_chg_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["coc_pct"] = disp["coc_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["mp_distance_pct"] = disp["mp_distance_pct"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
    )
    disp["stock_pcr"] = disp["stock_pcr"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )
    disp["fut_oi"] = disp["fut_oi"].apply(
        lambda v: f"{v/1_000_000:.2f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K"
    )
    disp["value_lacs"] = disp["value_lacs"].apply(
        lambda v: f"{v/100:.1f}Cr" if pd.notna(v) else "—"
    )

    disp.columns = [
        "Symbol", "Company", "Sector",
        "Signal", "Score",
        "Fut Chg%", "CoC%", "PCR", "MP Dist%",
        "📈Price", "🏦CoC", "📊PCR", "🎯MaxPain", "📦Vol",
        "Fut OI", "F&O Turnover",
    ]
    st.dataframe(disp, use_container_width=True, hide_index=True, height=560)


# ── Scanner tab ───────────────────────────────────────────────────────────────

def _render_scanner(df: pd.DataFrame) -> None:
    # ── Top movers by composite score ─────────────────────────────────────
    col_bull, col_bear = st.columns(2)

    top_buy  = df[df["composite_score"] > 0].nlargest(15, "composite_score")
    top_sell = df[df["composite_score"] < 0].nsmallest(15, "composite_score")

    with col_bull:
        st.markdown("#### 🟢 Top Bullish (by Score)")
        if not top_buy.empty:
            st.plotly_chart(
                _score_bar_chart(top_buy, title="Top Bullish", color="#00C853"),
                use_container_width=True,
            )
            _scanner_table(top_buy)
        else:
            st.info("No bullish stocks with current data.")

    with col_bear:
        st.markdown("#### 🔴 Top Bearish (by Score)")
        if not top_sell.empty:
            st.plotly_chart(
                _score_bar_chart(top_sell, ascending=True, title="Top Bearish", color="#D50000"),
                use_container_width=True,
            )
            _scanner_table(top_sell)
        else:
            st.info("No bearish stocks with current data.")

    # ── Factor heatmap ────────────────────────────────────────────────────
    st.markdown("#### 🧬 Factor Breakdown — Top 30 Stocks by |Score|")
    top30 = df.iloc[df["composite_score"].abs().argsort()[::-1].iloc[:30]]

    fig_heat = go.Figure(data=go.Heatmap(
        z=[
            top30["score_price"].tolist(),
            top30["score_coc"].tolist(),
            top30["score_pcr"].tolist(),
            top30["score_mp"].tolist(),
            top30["score_vol"].tolist(),
        ],
        x=top30["symbol"].tolist(),
        y=["📈 Price Dir", "🏦 CoC", "📊 PCR", "🎯 MaxPain", "📦 Volume"],
        colorscale=[
            [0.0, "#D50000"],
            [0.25, "#FF6D00"],
            [0.5,  "#607D8B"],
            [0.75, "#69F0AE"],
            [1.0,  "#00C853"],
        ],
        zmin=-2, zmax=2,
        text=[
            top30["score_price"].apply(lambda v: f"{int(v):+d}").tolist(),
            top30["score_coc"].apply(lambda v: f"{int(v):+d}").tolist(),
            top30["score_pcr"].apply(lambda v: f"{int(v):+d}").tolist(),
            top30["score_mp"].apply(lambda v: f"{int(v):+d}").tolist(),
            top30["score_vol"].apply(lambda v: f"{int(v):+d}").tolist(),
        ],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_heat.update_layout(
        title="Factor Scores per Stock (−2 bearish → +2 bullish)",
        height=320,
        template="plotly_dark",
        xaxis_tickangle=-40,
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Highest Futures OI (institutional focus) ──────────────────────────
    st.markdown("#### 💎 Highest Futures OI (Institutional Focus)")
    top_oi = df.nlargest(20, "fut_oi")
    if not top_oi.empty:
        colors = [_SIG_COLOR.get(s, "#9E9E9E") for s in top_oi["signal_label"]]
        fig_oi = go.Figure(go.Bar(
            x=top_oi["symbol"],
            y=top_oi["fut_oi"] / 1_000_000,
            marker_color=colors,
            text=top_oi["signal_label"].map(lambda s: _SIG_EMOJI.get(s, "")),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Fut OI: %{y:.2f}M<br>"
                "Signal: %{text}<extra></extra>"
            ),
        ))
        fig_oi.update_layout(
            title="Top 20 Stocks by Futures OI (colour = signal)",
            height=340,
            template="plotly_dark",
            yaxis_title="OI (Million Contracts)",
            xaxis_tickangle=-40,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_oi, use_container_width=True)


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
        title=title,
        height=300,
        template="plotly_dark",
        yaxis_title="Composite Score",
        xaxis_tickangle=-35,
        margin=dict(t=50, b=60),
    )
    return fig


def _scanner_table(df: pd.DataFrame) -> None:
    disp = df[[
        "symbol", "sector", "signal_label", "composite_score",
        "price_chg_pct", "coc_pct", "stock_pcr", "fut_oi",
    ]].copy()
    disp["signal_label"] = disp["signal_label"].map(
        lambda s: f"{_SIG_EMOJI.get(s, '')} {s}"
    )
    disp["composite_score"] = disp["composite_score"].apply(lambda v: f"{v:+.2f}")
    disp["price_chg_pct"]   = disp["price_chg_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["coc_pct"] = disp["coc_pct"].apply(
        lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
    )
    disp["stock_pcr"] = disp["stock_pcr"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )
    disp["fut_oi"] = disp["fut_oi"].apply(
        lambda v: f"{v/1_000_000:.2f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K"
    )
    disp.columns = ["Symbol", "Sector", "Signal", "Score", "Fut Chg%", "CoC%", "PCR", "Fut OI"]
    st.dataframe(disp, use_container_width=True, hide_index=True, column_config={
        "Symbol":   st.column_config.TextColumn("Symbol",
            help="NSE ticker symbol"),
        "Sector":   st.column_config.TextColumn("Sector",
            help="Sector from NSE sector master"),
        "Signal":   st.column_config.TextColumn("Signal",
            help="5-factor composite signal\n\n"
                 "STRONG BUY ≥ 1.0  |  BUY ≥ 0.4  |  HOLD  |  SELL > -1.0  |  STRONG SELL ≤ -1.0\n\n"
                 "Driven by: Price Direction (25%) + Cost of Carry (25%) + PCR Contrarian (20%) "
                 "+ Max Pain Gravity (20%) + Volume Activity (10%)"),
        "Score":    st.column_config.TextColumn("Score",
            help="Composite Score (−2.0 to +2.0)\n\n"
                 "Weighted sum of 5 factor sub-scores, each −2 to +2.\n"
                 "Positive = net bullish alignment  |  Negative = net bearish"),
        "Fut Chg%": st.column_config.TextColumn("Fut Chg%",
            help="Futures Price Change %\n\n"
                 "Formula: (fut_close − yesterday_settlement) / yesterday_settlement × 100\n\n"
                 "Scoring: ≥+2% → +2  |  ≥+0.5% → +1  |  −0.5% to +0.5% → 0  |  ≤−0.5% → −1  |  ≤−2% → −2\n\n"
                 "Uses yesterday's settlement (not today's open) to capture overnight conviction"),
        "CoC%":     st.column_config.TextColumn("CoC%",
            help="Cost of Carry %\n\n"
                 "Formula: (fut_close − spot_close) / spot_close × 100\n\n"
                 "Positive = futures at PREMIUM to cash → bulls are paying to hold positions (bullish)\n"
                 "Negative = futures at DISCOUNT → roll pressure / hedging (bearish)\n\n"
                 "Scoring: ≥+3% → +2  |  1–3% → +1  |  −0.5% to +1% → 0  |  ≤−0.5% → −1  |  ≤−2% → −2"),
        "PCR":      st.column_config.TextColumn("PCR",
            help="Put/Call Ratio — near-month options open interest\n\n"
                 "Formula: Total Put OI / Total Call OI\n\n"
                 "Contrarian indicator (crowd is usually wrong at extremes):\n"
                 "PCR > 1.5 = too many puts → panic / over-hedging → score +2 (contrarian bullish)\n"
                 "PCR > 1.0 → +1  |  0.7–1.0 → 0  |  < 0.7 → −1  |  < 0.5 → −2\n\n"
                 "When everyone is buying puts (protecting downside), the market often rallies"),
        "Fut OI":   st.column_config.TextColumn("Fut OI",
            help="Futures Open Interest — near-month contract\n\n"
                 "Total outstanding futures contracts (not traded volume).\n"
                 "High OI = large institutional positioning  |  Rising OI + rising price = strong trend\n"
                 "Minimum OI filter applied — only stocks above threshold are shown"),
    })
