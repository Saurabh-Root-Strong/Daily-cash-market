"""\nSLB — Securities Lending & Borrowing view.\n\nAnswers three questions for the chosen session:\n  1. Who is being borrowed (= shorted overnight) and how hard?   -> borrow board\n  2. What would I earn lending shares I already hold?             -> lend board\n  3. Does heavy borrowing tell me anything about the next weeks?  -> measured evidence\n\nThe measured answer to (3) is a real but narrow one and the page says exactly how\nnarrow: the top decile of borrowing is followed by underperformance, deciles 1-8\nsay nothing, and the gap is too small to pay for an actual short after the borrow\nfee and STT. So the flag is an AVOID / TRIM hint for longs, never a short call.\n"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics import slb_signals as sl
from src.dashboard.column_help import nc as _hnc, tc as _htc, dc as _hdc

_SECTIONS = ["Borrow board", "Lend your shares", "Market trend", "Stock history",
             "Does it predict?", "How SLB works"]


@st.cache_data(ttl=600)
def _latest(d: date) -> date | None:
    return sl.latest_slb_date(d)


@st.cache_data(ttl=600)
def _board(d: date) -> pd.DataFrame:
    df = sl.borrow_board(d)
    return df


@st.cache_data(ttl=600)
def _lend(d: date) -> pd.DataFrame:
    return sl.lend_board(d)


@st.cache_data(ttl=3600)
def _history(start: date, end: date) -> pd.DataFrame:
    return sl.market_history(start, end)


@st.cache_data(ttl=3600)
def _mix(start: date, end: date) -> pd.DataFrame:
    return sl.series_mix(start, end)


@st.cache_data(ttl=600)
def _symbol(symbol: str, start: date, end: date) -> pd.DataFrame:
    return sl.symbol_history(symbol, start, end)


@st.cache_data(ttl=600)
def _evidence() -> dict:
    return sl.load_evidence()


def _neg_years(L: dict) -> str:
    y = L.get("d10_vs_mid_by_year_bps", {})
    return f"{sum(1 for v in y.values() if v < 0)} of {len(y)} years"


def _pos_years(L: dict) -> str:
    y = [k for k, v in L.get("d10_vs_mid_by_year_bps", {}).items() if v >= 0]
    return ", ".join(y) if y else "no year"


def _both_line(ev: dict) -> str:
    """Measured grade of the red 'both' group vs the single flags (not assumed stronger)."""
    fg = ev.get("diagnostics", {}).get("flag_groups", {})
    b, f, h = fg.get("both"), fg.get("fresh_only"), fg.get("heavy_only")
    if not (b and f and h):
        return ""
    return (f"Historically the worst on average: **{abs(b['vs_unflagged_bps']) / 100:.1f}% worse** "
            f"than unflagged stocks over the next month (fresh only {abs(f['vs_unflagged_bps']) / 100:.1f}%, "
            f"heavy only {abs(h['vs_unflagged_bps']) / 100:.1f}%) — but it held in only "
            f"{b['years_negative']} of {b['n_years']} years, so it is bigger, not more certain.")


def _find(ev: dict, sig: str, h: int, uni: str = "slb_active", var: str = "sector_neutral") -> dict:
    for r in ev.get("results", []):
        if (r["signal"], r["h"], r["universe"], r.get("variant")) == (sig, h, uni, var):
            return r
    return {}


def render(selected_date: date) -> None:
    st.header("🏦 SLB — Securities Lending & Borrowing")
    st.caption(
        "Lenders park shares with NSE Clearing for a fee; borrowers take them and must "
        "return them by a fixed date. In Indian cash equity this is the **only** way to "
        "hold a short sale overnight, so the shares out on loan are the closest thing NSE "
        "publishes to per-stock short interest."
    )

    d = _latest(selected_date)
    if d is None:
        st.info("No SLB data in the database yet. Run `python scripts/backfill_slb.py`.")
        return
    if d != selected_date:
        st.caption(f"SLB files for {selected_date:%d %b %Y} are not in yet — showing {d:%d %b %Y}.")

    section = st.segmented_control("Section", _SECTIONS, default=_SECTIONS[0],
                                   key="slb_section", label_visibility="collapsed")
    section = section or _SECTIONS[0]
    ev = _evidence()

    if section == "Borrow board":
        _render_borrow(d, ev)
    elif section == "Lend your shares":
        _render_lend(d, ev)
    elif section == "Market trend":
        _render_trend(d)
    elif section == "Stock history":
        _render_symbol(d)
    elif section == "Does it predict?":
        _render_evidence(ev)
    else:
        _render_rules()


# ── Borrow board ─────────────────────────────────────────────────────────────

def _flag_label(r: pd.Series) -> str:
    if r["flag_fresh"] and r["flag_heavy"]:
        return "🔴 Fresh + heavy"
    if r["flag_fresh"]:
        return "🟠 Fresh borrowing"
    if r["flag_heavy"]:
        return "🟠 Heavily borrowed"
    return "" if r["in_study"] else "· not scored"


def _render_borrow(d: date, ev: dict) -> None:
    df = _board(d)
    if df.empty:
        st.info("No open SLB positions on this session.")
        return
    hist = _history(d - timedelta(days=45), d)
    total = float(df["oi_cr"].sum())
    prev = float(hist["oi_cr"].iloc[-21]) if len(hist) > 21 else None
    today_fee = float(hist["fee_cr"].iloc[-1]) * 100 if len(hist) else 0.0

    with st.container(horizontal=True):
        st.metric("Out on loan", f"₹{total:,.0f} Cr",
                  delta=None if prev is None else f"{(total / prev - 1) * 100:+.1f}% vs 20 sessions ago",
                  border=True,
                  help="Total value of shares that short sellers have borrowed and not yet "
                       "returned, across all stocks. Falling = short sellers are returning "
                       "shares (less bearish overall); rising = more bets against the market.")
        st.metric("Stocks borrowed", f"{len(df):,}", border=True,
                  help="How many stocks have at least some shares out on loan today.")
        st.metric("Flagged (top decile)", f"{int((df['flag_fresh'] | df['flag_heavy']).sum())}",
                  border=True,
                  help=f"Stocks where short sellers are betting unusually hard: the top 10% "
                       f"for fresh borrowing and/or for size of the borrowed pile, out of the "
                       f"{df.attrs.get('n_study', 0)} stocks that can be scored today. A stock "
                       f"with both flags is counted once.")
        st.metric("Fee paid today", f"₹{today_fee:,.1f} lakh", border=True,
                  help="Total fee that borrowers paid to lenders on today's new loans and "
                       "loan extensions, across all stocks.")

    lad = ev.get("diagnostics", {}).get("ladders", {})
    if lad.get("doi5_21") and lad.get("dtc_21"):
        f21, h21 = lad["doi5_21"], lad["dtc_21"]
        st.warning(
            f"**What a flag means** (tested on 2018–{ev.get('built_through', '')[:4]} data)  \n"
            "A flag = short sellers are betting unusually hard that this stock will fall.  \n"
            f"• 🟠 **Fresh borrowing:** over the next month these stocks did about "
            f"**{abs(f21['d10_vs_mid_bps']) / 100:.1f}% worse** than similar stocks in their "
            f"sector. This held in **{_neg_years(f21)}**, so it is the more reliable flag.  \n"
            f"• 🟠 **Heavily borrowed:** also about **{abs(h21['d10_vs_mid_bps']) / 100:.1f}% worse**, "
            f"but it held in only **{_neg_years(h21)}** (it failed in {_pos_years(h21)}), so "
            "trust it less.  \n"
            "Stocks without a flag: no signal either way.  \n"
            "**What to do:** if you plan to BUY a flagged stock, think twice — skip it or buy "
            "less. **Do not short it:** the borrowing fee (~4–5% a year) and taxes cost more "
            "than the small gain. More detail under *Does it predict?*",
            icon=":material/flag:",
        )

    c1, c2 = st.columns([1, 2])
    only = c1.pills("Show", ["Flagged only", "Scored names only"], selection_mode="multi",
                    key="slb_show", label_visibility="collapsed")
    q = c2.text_input("Search symbol", key="slb_q", placeholder="Search symbol…",
                      label_visibility="collapsed")
    v = df.copy()
    v["Flag"] = v.apply(_flag_label, axis=1)
    if "Flagged only" in (only or []):
        v = v.loc[v["flag_fresh"] | v["flag_heavy"]]
    if "Scored names only" in (only or []):
        v = v.loc[v["in_study"]]
    if q:
        v = v.loc[v["symbol"].str.contains(q.strip().upper(), regex=False)]

    # A 20-session % change on a near-zero base is noise (+9,843% on a few shares).
    small_base = v["oi_20"] < 0.05 * v["oi"]
    v["chg20"] = v["chg20_pct"].where(~small_base)
    v["New?"] = small_base.map({True: "new", False: ""})
    out = pd.DataFrame({
        "Symbol": v["symbol"], "Sector": v["sector"], "Flag": v["Flag"],
        "Out on loan (₹ Cr)": v["oi_cr"],
        "Days to cover": v["dtc"],
        "5-session change": v["doi5"].round(2) + 0.0,
        "20-session change %": v["chg20"],
        "New?": v["New?"],
        "Borrow fee % / yr": v["fee_ann"] * 100,
        "Price 21d %": v["r21"] * 100,
        "Liquidity rank": v["liq_rank"],
    }).sort_values("Out on loan (₹ Cr)", ascending=False)

    # Today's actual cut-offs, so a flag can be checked against its own number.
    sc = df.loc[df["in_study"]]
    cut_heavy = sc["dtc"].quantile(0.9) if len(sc) else float("nan")
    cut_fresh = sc["doi5"].quantile(0.9) if len(sc) else float("nan")
    lad = ev.get("diagnostics", {}).get("ladders", {})
    yrs_f = _neg_years(lad.get("doi5_21", {}))
    yrs_h = _neg_years(lad.get("dtc_21", {}))
    gap = abs(lad.get("doi5_21", {}).get("d10_vs_mid_bps", 0)) / 100

    # Header tooltips open ABOVE the table and get clipped when tall, so each one is
    # kept to a line or two; the full reading guide lives in this expander instead.
    with st.expander("How to read this table", icon=":material/help:"):
        st.markdown(
            "**A flag means short sellers are betting unusually hard against the stock.** "
            "Flags go to the top 10% of scored stocks each day, so the cut-off moves daily.\n\n"
            f"- 🟠 **Fresh borrowing** — lots of NEW shares borrowed in the last 5 days. "
            f"Today's cut-off: 5-session change above **{cut_fresh:+.2f}**. The more reliable flag "
            f"(worked in {yrs_f}).\n"
            f"- 🟠 **Heavily borrowed** — the borrowed pile is already very big. Today's cut-off: "
            f"days to cover above **{cut_heavy:.2f}**. It measures SIZE, not direction, so a stock "
            f"can be flagged while its pile shrinks. Less reliable (worked in {yrs_h}).\n"
            f"- 🔴 **Fresh + heavy** — the stock is above BOTH cut-offs at once (5-session change "
            f"above **{cut_fresh:+.2f}** AND days to cover above **{cut_heavy:.2f}**): a big pile "
            f"that is still growing. {_both_line(ev)}\n"
            "- **Blank** — normal, no information.\n"
            "- **· not scored** — outside the top 300 by turnover, or a split / bonus / buyback in "
            "the last 25 sessions distorts the numbers.\n\n"
            f"**What to do:** think twice before BUYING. Flagged stocks did ~{gap:.1f}% worse than "
            "their sector over the next month. Not a reason to short, not a crash prediction.\n\n"
            "**Columns, by example**\n"
            "- *Days to cover 12.7* — short sellers would need ~12–13 days of the stock's normal "
            "delivery buying to return every borrowed share. Under 1 = small pile.\n"
            "- *5-session change +3.68* — in 5 days they borrowed as many shares as ~3.7 days of "
            "normal delivery. Plus = piling in, minus = returning, ~0 = no change.\n"
            "- *20-session change −61%* — the pile is 61% smaller than a month ago. **None** + "
            "**new** = almost nothing was borrowed a month ago, so a % would be meaningless.\n"
            "- *Borrow fee 11.4%/yr* — borrowing ₹1 lakh of the stock costs ~₹11,400 a year. "
            "Normal is ~1–3%; a high fee means shares are scarce and demand to short is strong.\n"
            "- *Price 21d* — shorts often chase falling stocks; the study removes that effect.\n"
            "- *Liquidity rank* — 1 = most traded NSE stock; only ranks 1–300 can be flagged."
        )

    st.dataframe(
        out, hide_index=True, width="stretch", height=560,
        column_config={
            "Symbol": _htc("Symbol", pinned=True),
            "Sector": _htc("Sector"),
            "Flag": _htc("Flag", help=(
                "Short sellers betting unusually hard against the stock (top 10% today). "
                "🔴 = fresh + heavy. See 'How to read this table' above.")),
            "Out on loan (₹ Cr)": _hnc("Out on loan (₹ Cr)", format="%.1f", help=(
                "Value of the stock's shares borrowed by short sellers and not yet returned.")),
            "Days to cover": _hnc("Days to cover", format="%.2f", help=(
                f"Borrowed shares ÷ normal daily delivery. Flagged 'heavy' above {cut_heavy:.2f} today.")),
            "5-session change": _hnc("5-session change", format="%+.2f", help=(
                f"New borrowing in 5 days, in days of delivery. Flagged 'fresh' above {cut_fresh:+.2f} today.")),
            "20-session change %": _hnc("20-session change %", format="%+.0f%%", help=(
                "Change in borrowed shares over ~1 month. None = too small a month ago to compare.")),
            "New?": _htc("New?", help="Short sellers only just started on this stock."),
            "Borrow fee % / yr": _hnc("Borrow fee % / yr", format="%.2f%%", help=(
                "Yearly cost to borrow, % of price. High = scarce shares, strong short demand.")),
            "Price 21d %": _hnc("Price 21d %", format="%+.1f%%", help="Share price move over ~1 month."),
            "Liquidity rank": _hnc("Liquidity rank", format="%d", help=(
                "1 = most traded. Only ranks 1–300 can be flagged.")),
        },
    )

    fc = sl.foreclosures(d)
    if not fc.empty:
        with st.expander(f"Corporate-action foreclosures ({len(fc)}) — loans here close mechanically"):
            st.caption("Every SLB position in these names is force-closed on the record date, so a "
                       "collapse in 'out on loan' there is NOT short covering.")
            st.dataframe(fc[["symbol", "description", "record_date", "shut_start", "shut_end",
                             "series_scope"]], hide_index=True, width="stretch",
                         column_config={"record_date": _hdc("Record date"),
                                        "shut_start": _hdc("No new SLB from"),
                                        "shut_end": _hdc("Shut until")})


# ── Lend board ───────────────────────────────────────────────────────────────

def _render_lend(d: date, ev: dict) -> None:
    df = _lend(d)
    q = ev.get("diagnostics", {}).get("fee_ann_quantiles_pct", {})
    st.markdown(
        "**Lending idle long-term holdings** earns the fee below on top of the shares' own "
        "return. NSE Clearing guarantees the return of the shares, not the borrower."
    )
    if q:
        with st.container(horizontal=True):
            st.metric("Typical fee (median)", f"{q.get('0.5', 0):.1f} % / yr", border=True,
                      help="Median annualised fee across traded fresh contracts (≥₹1 lakh, ≥10 days), "
                           "2018 onward.")
            st.metric("Upper quartile", f"{q.get('0.75', 0):.1f} % / yr", border=True)
            st.metric("Top 10 %", f"{q.get('0.9', 0):.1f} % / yr", border=True,
                      help="High fees sit on hard-to-borrow names, the same names the study flags "
                           "as likely to underperform: a high lending yield can be a warning.")
    if df.empty:
        st.info("No SLB contract traded on this session.")
        return

    fresh = df.loc[df["series_kind"] != "ROLLOVER"]
    roll = df.loc[df["series_kind"] == "ROLLOVER"]
    st.subheader("Fresh lending today", divider="gray")
    st.caption("Each row is one contract. The fee is ₹ per share for the WHOLE remaining tenure; "
               "the yield column annualises it. Contracts under ~10 days show inflated yields: a "
               "₹0.50 fee on 3 days is 60 %/yr on paper but only ₹0.50 in hand.")
    st.dataframe(
        fresh.assign(kind=fresh["series_kind"].map({"NO_AGM": "X (AGM-proof)", "REGULAR": "Regular"}))[
            ["symbol", "series", "kind", "expiry_date", "days", "fee_vwap", "close_price",
             "yield_ann_pct", "quantity", "notional_cr", "trades"]],
        hide_index=True, width="stretch",
        column_config={
            "symbol": _htc("Symbol"), "series": _htc("Series"), "kind": _htc("Contract type"),
            "expiry_date": _hdc("Return date", help="Reverse-leg date: the day the borrower must "
                                                   "return the shares (first Tuesday of the month "
                                                   "since 2026, first Thursday before)."),
            "days": _hnc("Days", format="%d"),
            "fee_vwap": _hnc("Fee ₹ / share", format="%.2f",
                             help="Volume-weighted fee for the whole tenure."),
            "close_price": _hnc("Share price", format="%.2f"),
            "yield_ann_pct": _hnc("Yield % / yr", format="%.2f%%"),
            "quantity": _hnc("Shares lent", format="%d"),
            "notional_cr": _hnc("Value lent (₹ Cr)", format="%.2f"),
            "trades": _hnc("Trades", format="%d"),
        },
    )
    if not roll.empty:
        with st.expander(f"Rollovers today ({len(roll)}) — extension fees, not annualised"):
            st.caption("A rollover contract prices EXTENDING an existing loan into a later month. Its "
                       "fee covers the extension, not the days to the near expiry, so dividing by "
                       "those days would overstate the yield many times over.")
            st.dataframe(roll[["symbol", "series", "expiry_date", "fee_vwap", "quantity", "value"]],
                         hide_index=True, width="stretch",
                         column_config={"expiry_date": _hdc("Near expiry"),
                                        "fee_vwap": _hnc("Fee ₹ / share", format="%.2f"),
                                        "quantity": _hnc("Shares", format="%d"),
                                        "value": _hnc("Fee paid ₹", format="%.0f")})

    with st.expander("Before you lend — what actually changes for you", expanded=True):
        st.markdown(
            "- **You cannot sell the shares while lent.** Getting them back early needs a *recall*, "
            "which is best-effort and priced by the market (you give up part of the fee), and is "
            "allowed only until 3 working days before the return date.\n"
            "- **Dividends still reach you**: NSE Clearing debits the borrower and credits you on the "
            "record date (+1 working day).\n"
            "- **Voting rights do not.** X-series loans run through AGMs/EGMs; regular 01–12 series "
            "are force-closed for them.\n"
            "- **Bonus, merger, buyback, demerger** → the loan is foreclosed on the record date and "
            "the unused fee is refunded pro rata by you.\n"
            "- **Splits** → positions are adjusted; you get the post-split quantity back.\n"
            "- **Tax**: lending is not a 'transfer' (Income Tax circular 2/2008), so no capital-gains "
            "event; the fee itself is taxable income. No STT on SLB trades.\n"
            "- **Broker**: your broker must be an NSE Clearing SLB participant, and brokers typically "
            "keep a share of the fee. "
            "Not every broker offers SLB — confirm before planning around it.\n"
            "- **Risk**: NSE Clearing guarantees settlement; if a borrower fails to return, a buy-in "
            "auction runs, and failing that you are paid a close-out price (the higher of the max "
            "trade price over the last two days or 25 % above the close)."
        )


# ── Market trend ─────────────────────────────────────────────────────────────

def _render_trend(d: date) -> None:
    h = _history(date(2018, 1, 1), d)
    if h.empty:
        st.info("No history.")
        return
    h["trade_date"] = pd.to_datetime(h["trade_date"])
    st.subheader("Shares out on loan (₹ Cr)", divider="gray")
    st.caption("Outstanding borrow × cash close. The market was ₹2–3.6k Cr from 2018 to 2024 and "
               "has grown several-fold since 2025: a signal level from 2019 is not comparable with "
               "today's, which is why every flag here is a same-day RANK, not a fixed threshold.")
    st.line_chart(h.set_index("trade_date")[["oi_cr"]].rename(columns={"oi_cr": "Out on loan ₹ Cr"}),
                  height=280)
    st.subheader("Stocks with anything on loan", divider="gray")
    st.line_chart(h.set_index("trade_date")[["names"]].rename(columns={"names": "Stocks"}), height=220)

    m = h.set_index("trade_date")[["fee_cr"]].resample("ME").sum()
    m.index = m.index.strftime("%Y-%m")
    st.subheader("Lending fees paid per month (₹ Cr)", divider="gray")
    st.bar_chart(m.rename(columns={"fee_cr": "Fees ₹ Cr"}), height=240)

    mix = _mix(date(2017, 1, 1), d)
    if not mix.empty:
        p = mix.pivot(index="year", columns="series_kind", values="fee_cr").fillna(0)
        p = p.rename(columns={"NO_AGM": "X series (AGM-proof)", "REGULAR": "Regular 01–12",
                              "ROLLOVER": "Rollovers"})
        p.index = p.index.astype(str)
        st.subheader("Fee rupees by contract type", divider="gray")
        st.caption("Regular series are practically extinct since 2019: nearly all fresh lending is "
                   "in X series, which survive AGMs.")
        st.bar_chart(p, height=260)


# ── Stock history ────────────────────────────────────────────────────────────

def _render_symbol(d: date) -> None:
    b = _board(d)
    opts = b.sort_values("oi_cr", ascending=False)["symbol"].tolist() if not b.empty else []
    c1, c2 = st.columns([2, 1])
    sym = c1.selectbox("Stock", opts, key="slb_sym") if opts else c1.text_input("Stock", key="slb_sym_t")
    span = c2.segmented_control("Window", ["6M", "1Y", "3Y", "All"], default="1Y", key="slb_span")
    if not sym:
        return
    days = {"6M": 183, "1Y": 365, "3Y": 1095}.get(span or "1Y", 3650)
    h = _symbol(sym, d - timedelta(days=days), d)
    if h.empty:
        st.info("No history for this stock.")
        return
    h["trade_date"] = pd.to_datetime(h["trade_date"])
    h["dq20"] = h["deliv_qty"].rolling(20, min_periods=5).mean()
    h["days_to_cover"] = h["oi"] / h["dq20"]
    x = h.set_index("trade_date")
    st.subheader(f"{sym} — price", divider="gray")
    st.line_chart(x[["close_price"]].rename(columns={"close_price": "Close ₹"}), height=220)
    st.subheader("Shares out on loan", divider="gray")
    st.line_chart(x[["oi"]].rename(columns={"oi": "Borrowed shares"}), height=220)
    st.subheader("Days to cover (borrowed ÷ 20-session avg delivery)", divider="gray")
    st.line_chart(x[["days_to_cover"]].rename(columns={"days_to_cover": "Days to cover"}), height=200)
    fees = h.loc[h["fee_vwap"].notna(), ["trade_date", "fee_vwap", "lent_qty", "min_days", "close_price"]]
    if not fees.empty:
        with st.expander(f"Fee prints ({len(fees)} sessions)"):
            st.dataframe(fees.sort_values("trade_date", ascending=False), hide_index=True,
                         width="stretch",
                         column_config={"trade_date": _hdc("Date"),
                                        "fee_vwap": _hnc("Fee ₹ / share", format="%.2f"),
                                        "lent_qty": _hnc("Shares lent", format="%d"),
                                        "min_days": _hnc("Shortest tenure (days)", format="%d"),
                                        "close_price": _hnc("Close ₹", format="%.2f")})


# ── Evidence ─────────────────────────────────────────────────────────────────

def _render_evidence(ev: dict) -> None:
    if not ev:
        st.info("Study not built yet. Run `python scripts/build_slb_study.py`.")
        return
    dg = ev.get("diagnostics", {})
    st.markdown(
        f"Measured over **{ev['results'][0]['n_dates']:,} sessions** (Mar-2018 → "
        f"{ev.get('built_through')}), top-300 liquidity names. The three signals, two horizons "
        "and two universes were fixed **before** any result was seen: 12 candidates, so a "
        "test needs |t| > 2.87 to survive (Bonferroni)."
    )
    st.success(
        "**Verdict — a real but narrow AVOID flag.** Fresh SLB borrowing is followed by "
        "underperformance after controls for momentum, volatility, liquidity, sector, dividends "
        "and futures basis, in "
        f"{_neg_years(dg.get('ladders', {}).get('doi5_21', {}))}. The heavily-borrowed level is "
        "the weaker cousin: similar average, less steady "
        f"({_neg_years(dg.get('ladders', {}).get('dtc_21', {}))}). Both live entirely in the top "
        "decile, and neither is large enough to pay for a short.", icon=":material/verified:")

    rows = []
    names = {"dtc": "Heavily borrowed (days to cover)", "doi5": "Fresh borrowing (5-session change)",
             "fee_ann": "Borrow fee (annualised)"}
    for r in ev["results"]:
        if r.get("variant") != "sector_neutral":
            continue
        rows.append({"Signal": names[r["signal"]], "Universe": {"top300": "Top-300 liquid",
                     "slb_active": "Borrowed names only"}[r["universe"]],
                     "Horizon": f"{r['h']} sessions", "IC": r["ic"],
                     "Regression bps": r["fm_beta_bps"], "Regression t": r["fm_t"],
                     "Top−bottom quintile bps": r["q5_q1_bps"], "Quintile t": r["q5_q1_t"],
                     "Years same sign": f"{r['years_same_sign']}/{r['n_years']}",
                     "Survives": "✅" if abs(r["fm_t"]) > 2.87 else "—"})
    st.subheader("All 12 pre-registered tests (sector-neutral)", divider="gray")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "IC": _hnc("IC", format="%+.4f", help="Information Coefficient: average daily rank "
                                              "correlation between signal and forward return."),
        "Regression bps": _hnc("Regression bps", format="%+.1f",
                               help="Fama-MacBeth: forward excess return per 1 standard deviation of "
                                    "the signal's rank, holding past 5d and 21d return, volatility "
                                    "and liquidity fixed."),
        "Regression t": _hnc("Regression t", format="%+.2f",
                             help="Newey-West t-stat (overlapping windows corrected)."),
        "Top−bottom quintile bps": _hnc("Top−bottom quintile bps", format="%+.1f"),
        "Quintile t": _hnc("Quintile t", format="%+.2f"),
    })
    st.caption("The borrow-fee signal is the weakest: its rank correlation is negative every "
               "year, but its quintile spread is flat and it fades once returns are measured "
               "against the stock's own sector — expensive-to-borrow is largely a sector trait. "
               "It is shown on the board for cost, not used for the flag.")

    lad = dg.get("ladders", {})
    if lad:
        st.subheader("Where the effect lives — decile ladder, 21 sessions", divider="gray")
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "doi5_21", "Fresh borrowing"), (c2, "dtc_21", "Heavily borrowed")):
            L = lad.get(key, {})
            if not L:
                continue
            s = pd.DataFrame({"Decile": [f"D{i}" for i in range(1, 11)],
                              "Sector-neutral bps": L["decile_bps"]}).set_index("Decile")
            col.markdown(f"**{title}** — D10 vs D4–D7: {L['d10_vs_mid_bps']:+.0f} bps "
                         f"(t {L['d10_vs_mid_t']:+.1f})")
            col.bar_chart(s, height=220)
            yrs = L.get("d10_vs_mid_by_year_bps", {})
            col.caption("By year (bps): " + " · ".join(f"{k}: {v:+.0f}" for k, v in yrs.items()))

    hc = dg.get("haircut", {})
    if hc:
        st.subheader("Is it just DHFL, YES Bank and Jet Airways?", divider="gray")
        st.caption("The worst k names removed from EVERY decile (removing them only from the top "
                   "decile would flatter any signal). Average sector-neutral 21-session return, bps.")
        t = pd.DataFrame({f"worst {k} removed": v for k, v in hc.get("doi5", {}).items()}).T
        st.dataframe(t, width="stretch")
        pn = dg.get("per_name", {}).get("doi5", {})
        if pn:
            st.caption(f"Per stock: in its own top-decile weeks, the median name did "
                       f"{pn['median_bps']:+.0f} bps worse than in its own middle-decile weeks; "
                       f"{pn['share_negative'] * 100:.0f}% of {pn['n']} names did worse.")

    bc = dg.get("basis_control", {})
    if bc:
        st.subheader("Is it just cash-futures arbitrage?", divider="gray")
        st.caption(f"Part of SLB borrowing funds *reverse arbitrage* (borrow + sell the share, buy "
                   f"the cheaper future). Borrowing does lean into cheap futures (rank correlation "
                   f"{bc.get('corr_doi5_basis', 0):+.2f}), but adding the futures basis as a "
                   f"control keeps most of the signal (from {bc.get('from')}):")
        br = []
        for k, lab in (("doi5_5", "Fresh, 5 sessions"), ("doi5_21", "Fresh, 21 sessions"),
                       ("dtc_5", "Heavy, 5 sessions"), ("dtc_21", "Heavy, 21 sessions")):
            if k in bc:
                br.append({"Test": lab, "Without basis bps": bc[k]["without"]["bps"],
                           "t": bc[k]["without"]["t"], "With basis bps": bc[k]["with"]["bps"],
                           "t ": bc[k]["with"]["t"]})
        st.dataframe(pd.DataFrame(br), hide_index=True, width="stretch")

    fee = dg.get("d10_fee_ann_median_pct", {})
    st.subheader("Why not short it?", divider="gray")
    st.markdown(
        f"- Top-decile names cost **{fee.get('doi5', 0):.1f}–{fee.get('dtc', 0):.1f} % a year** to "
        "borrow (median) ≈ 35–40 bps for a 21-session hold, and the fee is charged for the whole "
        "contract even if you return early.\n"
        "- STT 0.1 % on the short sale + 0.1 % on the buy-back ≈ **20 bps**, plus brokerage and "
        "impact.\n"
        "- The gap is **relative** to other stocks. A naked short also carries the market: in "
        "an up-trending market it loses even when the stock lags.\n"
        "- ~50 bps of relative edge against ~60+ bps of cost → **no net short edge**. As a reason to "
        "skip or trim a long, it costs nothing."
    )


# ── Rules ────────────────────────────────────────────────────────────────────

def _render_rules() -> None:
    st.markdown(
        "**Who can take part** — every investor category, retail included (SEBI); institutions "
        "need a Custodial Participant code. Your broker must be an NSE Clearing SLB participant.\n\n"
        "**Eligible stocks** — every F&O stock, liquid index ETFs, and Group-I stocks with "
        "market-wide position limit ≥ ₹100 Cr and ≥ ₹100 Cr monthly cash turnover; list revised "
        "monthly (daily file `SLB_ELG_SEC_DDMMYYYY.csv`).\n\n"
        "**Contracts** — 12 monthly tenures, up to 12 months including rollovers. Each month has two "
        "series: numeric `01`–`12` (force-closed for every corporate action including AGM/EGM) and "
        "`X1`–`XD` (`XO`=Oct, `XN`=Nov, `XD`=Dec; NOT closed for AGM/EGM). Two-letter series "
        "(`OD`, `ON`, `O1`, `O2`…) are rollovers into a later month.\n\n"
        "**Return date** — first Tuesday of the month in 2026 files (the NSE Clearing FAQ still "
        "says Thursday; the files are what settle). Session 09:15–17:00.\n\n"
        "**Fee** — market-determined, ₹ per share for the whole tenure; no STT and transaction "
        "charges waived.\n\n"
        "**Borrower margins** — Value-at-Risk + extreme-loss margin at the cash-market rates, "
        "mark-to-market daily, plus 100 % of the lending price (previous close) and the fee upfront.\n\n"
        "**Position limits** — market-wide 10 % of free float; participant / institution 10 % of "
        "that; a single client 1 %.\n\n"
        "**Corporate actions** — dividend: passed from borrower to lender on the record date. "
        "Split: quantities adjusted. Bonus / merger / buyback / demerger / open offer: foreclosed "
        "on the record date, fee refunded pro rata. No new SLB trades during the shut period.\n\n"
        "**Failures** — lender fails first leg: closed out at the higher of 25 % of the close or "
        "(max price T..T+1 − close). Borrower fails to return: buy-in auction, then close-out at the "
        "higher of the max price over two days or 25 % above the close.\n\n"
        "**Tax** — lending/borrowing is not a transfer (Income Tax circular 2/2008).\n\n"
        "Sources: [NSE Clearing SLBS FAQ (2025-07)]"
        "(https://www.nseclearing.in/sites/default/files/2025-07/FAQ%20for%20SLB%20Scheme.pdf) · "
        "[NSE Clearing SLBS](https://www.nseclearing.in/clearing-settlement/slbs) · "
        "[NSE SLB market data](https://www.nseindia.com/market-data/securities-lending-and-borrowing)"
    )
    st.subheader("Data traps this page handles", divider="gray")
    st.markdown(
        "- The fee bhavcopy (`SLBM_BC_*.DAT`) has **no header**; columns were identified by "
        "reconciliation (value ÷ quantity sits inside [low, high] on 100 % of rows).\n"
        "- **Fees are per whole tenure** → annualised by days left; rollovers are never annualised.\n"
        "- A **split** rescales borrowed quantity while the 20-session delivery base is still "
        "pre-split, and a **bonus / buyback** forecloses every loan: names within 25 sessions of a "
        "corporate action are not scored.\n"
        "- NSE's `prev_close` is **not adjusted for dividends**, so the study adds the payout back "
        "on the ex-date (12,897 dividends parsed from NSE's corporate-action feed).\n"
        "- The market grew several-fold in 2025–26, so flags are same-day ranks, not fixed levels."
    )
