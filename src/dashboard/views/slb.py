"""
SLB — Securities Lending & Borrowing view.

Answers three questions for the chosen session:
  1. Who is being borrowed (= shorted overnight) and how hard?   -> borrow board
  2. What would I earn lending shares I already hold?             -> lend board
  3. Does heavy borrowing tell me anything about the next weeks?  -> measured evidence

The measured answer to (3) is a real but narrow one and the page says exactly how
narrow: the top decile of borrowing is followed by underperformance, deciles 1-8
say nothing, and the gap is too small to pay for an actual short after the borrow
fee and STT. So the flag is an AVOID / TRIM hint for longs, never a short call.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics import slb_episodes as se
from src.analytics import slb_signals as sl
from src.dashboard.column_help import nc as _hnc, tc as _htc, dc as _hdc

_SECTIONS = ["Borrow board", "Build-up timeline", "Stock history", "Lend your shares",
             "Market trend", "Does it predict?", "How SLB works"]


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


@st.cache_data(ttl=600)
def _timeline_ev() -> dict:
    return se.load_timeline()


@st.cache_data(ttl=600, show_spinner="Tracing every build-up (roll-adjusted)…")
def _episodes(d: date) -> pd.DataFrame:
    return se.episode_table(d)


@st.cache_data(ttl=600)
def _streaks(d: date) -> pd.DataFrame:
    return se.flag_streaks(d)


@st.cache_data(ttl=600)
def _ep_detail(symbol: str, d: date, days: int):
    return se.episode_detail(symbol, d, lookback_days=days)


def _with_timeline(board: pd.DataFrame, d: date) -> pd.DataFrame:
    """Board rows + when the build-up began, its phase, and when the flag began."""
    ep = _episodes(d)
    fs = _streaks(d)
    v = board.copy()
    if not ep.empty:
        v = v.merge(ep[["symbol", "start", "start_known", "age", "phase", "move_since_pct",
                        "avg_borrow_px", "shorts_pnl_pct"]], on="symbol", how="left")
    if not fs.empty:
        v = v.merge(fs, on="symbol", how="left")
        since = []
        for fr, hv, fs_, hs_ in zip(v["flag_fresh"], v["flag_heavy"],
                                    v.get("fresh_since"), v.get("heavy_since")):
            ds = [x for x, on in ((fs_, fr), (hs_, hv)) if on and x is not None and pd.notna(x)]
            since.append(min(ds) if ds else None)
        v["flagged_since"] = since
    for c in ("start", "flagged_since"):
        if c in v:
            v[c] = pd.to_datetime(v[c])
    return v


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
    elif section == "Build-up timeline":
        _render_timeline(d)
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
    v = _with_timeline(df, d)
    v["Flag"] = v.apply(_flag_label, axis=1)
    if "Flagged only" in (only or []):
        v = v.loc[v["flag_fresh"] | v["flag_heavy"]]
    if "Scored names only" in (only or []):
        v = v.loc[v["in_study"]]
    if q:
        v = v.loc[v["symbol"].str.contains(q.strip().upper(), regex=False)]

    out = pd.DataFrame({
        "Symbol": v["symbol"], "Sector": v["sector"], "Flag": v["Flag"],
        "Flagged since": v.get("flagged_since"),
        "Building since": v.get("start"),
        "Phase": v.get("phase"),
        "Out on loan (₹ Cr)": v["oi_cr"],
        "Days to cover": v["dtc"],
        "5-session change": v["doi5"].round(2) + 0.0,
        "Price since start %": v.get("move_since_pct"),
        "Borrow fee % / yr": v["fee_ann"] * 100,
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
            "- *Flagged since* — the first day of the flag's current unbroken run.\n"
            "- *Building since* — when this pile of borrowing began: the day after it was last "
            "under a quarter of today's size (monthly loan rolls are not counted as new "
            "borrowing). *Price since start* = how the share price moved since that day.\n"
            "- *Phase* — **Building** (pile grew >5% in 5 sessions), **Holding**, or "
            "**Covering** (pile ≥30% below its peak: shares being returned).\n"
            "- *Borrow fee 11.4%/yr* — borrowing ₹1 lakh of the stock costs ~₹11,400 a year. "
            "Normal is ~1–3%; a high fee means shares are scarce and demand to short is strong.\n"
            "- The *Build-up timeline* section shows every pile in detail; *Stock history* shows "
            "one stock's build-up week by week."
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
            "Flagged since": _hdc("Flagged since", format="DD MMM YYYY",
                                  help="First day of the flag's current unbroken run."),
            "Building since": _hdc("Building since", format="DD MMM YYYY",
                                   help="Day this pile of borrowing began (rolls not counted)."),
            "Phase": _htc("Phase", help="Building / Holding / Covering (≥30% off peak)."),
            "Price since start %": _hnc("Price since start %", format="%+.1f%%",
                                        help="Share price move since 'Building since'."),
            "Borrow fee % / yr": _hnc("Borrow fee % / yr", format="%.2f%%", help=(
                "Yearly cost to borrow, % of price. High = scarce shares, strong short demand.")),

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


# ── Build-up timeline ───────────────────────────────────────────────────────

_PHASE_ICON = {"Building": "🔺 Building", "Holding": "⏸ Holding", "Covering": "🔻 Covering"}


def _timing_line(tl: dict, key: str) -> str:
    t = tl.get("tests", {}).get(key, {})
    if not t:
        return ""
    verdict = "clears" if t.get("passes") else "does NOT clear"
    return (f"{t['a_bps'] / 100:+.2f}% vs {t['b_bps'] / 100:+.2f}% "
            f"(gap {t['diff_bps'] / 100:+.2f}%, t {t['t']:+.1f}, {t['years_same_sign']} of "
            f"{t['n_years']} years) — {verdict} the bar of |t| > {tl.get('bar', 2.64)}")


def _render_timeline(d: date) -> None:
    board = _board(d)
    ep = _episodes(d)
    if ep.empty or board.empty:
        st.info("No build-ups to show for this session.")
        return
    v = _with_timeline(board, d)
    v = v.loc[v["start"].notna()].copy()
    ep_i = ep.set_index("symbol")
    for c in ("start_qty", "now_qty", "peak_qty", "peak_date", "off_peak_pct",
              "price_start", "price_now"):
        v[c] = v["symbol"].map(ep_i[c])
    v["Flag"] = v.apply(_flag_label, axis=1)

    st.markdown(
        "**When did the borrowing start, and how was it added?** A *build-up* starts on the "
        "day after the borrowed pile was last **under a quarter of today's size**. Monthly "
        "loan **rolls are not counted as new borrowing**: before each return date many "
        "borrowers take next month's loan while the old one is still open, which briefly "
        "double-counts the same short."
    )
    with st.container(horizontal=True):
        for ph in ("Building", "Holding", "Covering"):
            n = int((v["phase"] == ph).sum())
            st.metric(_PHASE_ICON[ph], f"{n}", border=True,
                      help={"Building": "Pile grew more than 5% in the last 5 sessions.",
                            "Holding": "Pile roughly steady.",
                            "Covering": "Pile at least 30% below its peak: shares being returned."}[ph])

    tl = _timeline_ev()
    if tl.get("tests"):
        with st.expander("Does the timing change what a flag means? (tested)", icon=":material/science:"):
            dsc = tl.get("describe", {})
            st.markdown(
                "Six pre-registered tests on 2018–" + tl.get("built_through", "")[:4] + " data, "
                "sector-neutral, next month (21 sessions). **None is strong enough to act on** — "
                "treat the columns below as a description of what is happening, not a forecast.\n\n"
                f"- **New flag vs old flag** (fresh flag in its first 5 sessions vs 11+): "
                f"{_timing_line(tl, 'T1_flag_age_21')}. Old, persistent flags leaned WORSE.\n"
                f"- **Shorts losing vs winning** (price >5% above vs >5% below the average borrow "
                f"price): {_timing_line(tl, 'T2_shorts_pnl_21')}. No short squeeze: stocks where "
                "shorts were losing did not bounce more.\n"
                f"- **Covering vs holding** (pile ≥30% off its peak vs near it, after a heavy "
                f"build-up): {_timing_line(tl, 'T3_covering_21')}. Covering leaned BETTER — the "
                "closest of the three, still not proven.\n\n"
                f"A fresh flag first appears a median **{dsc.get('first_fresh_flag_episode_age_median', 0):.0f} "
                f"sessions** into a build-up, and a fresh flag lasts a median "
                f"**{dsc.get('fresh_streak_median', 0):.0f} sessions**."
            )

    c1, c2 = st.columns([2, 1])
    phases = c1.pills("Phase", list(_PHASE_ICON.values()), selection_mode="multi",
                      key="slb_tl_phase", label_visibility="collapsed")
    flagged = c2.toggle("Flagged only", value=True, key="slb_tl_flag")
    if phases:
        keep = [k for k, lab in _PHASE_ICON.items() if lab in phases]
        v = v.loc[v["phase"].isin(keep)]
    if flagged:
        v = v.loc[v["flag_fresh"] | v["flag_heavy"]]

    out = pd.DataFrame({
        "Symbol": v["symbol"], "Flag": v["Flag"], "Phase": v["phase"].map(_PHASE_ICON),
        "Building since": v["start"], "Sessions": v["age"],
        "Pile then (shares)": v["start_qty"], "Pile now (shares)": v["now_qty"],
        "Grown ×": v["now_qty"] / v["start_qty"].where(v["start_qty"] > 0),
        "Peak on": pd.to_datetime(v["peak_date"]), "Off peak %": v["off_peak_pct"],
        "Price then": v["price_start"], "Price now": v["price_now"],
        "Price since start %": v["move_since_pct"],
        "Avg borrow price": v["avg_borrow_px"],
        "Shorts' P&L %": v["shorts_pnl_pct"],
    }).sort_values("Building since", ascending=False)
    st.dataframe(out, hide_index=True, width="stretch", height=520, column_config={
        "Symbol": _htc("Symbol", pinned=True),
        "Building since": _hdc("Building since", format="DD MMM YYYY",
                               help="Day after the pile was last under 25% of today's size."),
        "Sessions": _hnc("Sessions", format="%d", help="Trading sessions since the build-up began."),
        "Pile then (shares)": _hnc("Pile then (shares)", format="compact",
                                   help="Borrowed shares on the first day of the build-up."),
        "Pile now (shares)": _hnc("Pile now (shares)", format="compact",
                                  help="Borrowed shares today, roll-adjusted."),
        "Grown ×": _hnc("Grown ×", format="%.1f×"),
        "Peak on": _hdc("Peak on", format="DD MMM YYYY", help="Day the pile was largest."),
        "Off peak %": _hnc("Off peak %", format="%+.0f%%",
                           help="How far the pile is below its peak. ≤ −30% = Covering."),
        "Price then": _hnc("Price then", format="%.2f", help="Close on the first day (split-adjusted)."),
        "Price now": _hnc("Price now", format="%.2f"),
        "Price since start %": _hnc("Price since start %", format="%+.1f%%"),
        "Avg borrow price": _hnc("Avg borrow price", format="%.2f", help=(
            "Estimated average price of the borrowed shares still open: the closes on the days "
            "the pile grew, oldest shares returned first. Not anyone's reported fill.")),
        "Shorts' P&L %": _hnc("Shorts' P&L %", format="%+.1f%%", help=(
            "+ = price is below the average borrow price (short sellers winning); "
            "− = above it (short sellers losing). Ignores fees; arbitrage borrowers have no "
            "directional P&L.")),
    })
    st.caption("Borrowing is not always a bet on a fall: part of it funds cash-futures "
               "arbitrage (borrow and sell the share, buy the cheaper future), which has no "
               "direction. The average borrow price is an estimate from daily closes.")


# ── Stock history ────────────────────────────────────────────────────────────

def _render_symbol(d: date) -> None:
    b = _board(d)
    opts = []
    if not b.empty:
        bb = b.assign(f=(b["flag_fresh"] | b["flag_heavy"]).astype(int))
        opts = bb.sort_values(["f", "oi_cr"], ascending=False)["symbol"].tolist()
    c1, c2 = st.columns([2, 1])
    sym = c1.selectbox("Stock", opts, key="slb_sym") if opts else c1.text_input("Stock", key="slb_sym_t")
    span = c2.segmented_control("Window", ["6M", "1Y", "3Y"], default="1Y", key="slb_span")
    if not sym:
        return
    days = {"6M": 183, "1Y": 400, "3Y": 1100}.get(span or "1Y", 400)
    eff, epi = _ep_detail(sym, d, days)
    if eff.empty:
        st.info("No SLB history for this stock in this window.")
        return
    eff = eff.sort_values("trade_date").copy()
    eff["avg_borrow_px"] = se.fifo_cost(eff["eff"].values.astype(float),
                                        eff["close_adj"].values.astype(float))

    if epi:
        with st.container(horizontal=True):
            st.metric("Building since", f"{epi['start']:%d %b %Y}", border=True,
                      help="Day after the pile was last under 25% of today's size"
                           + ("" if epi["start_known"] else " — older than this window"))
            st.metric("Sessions", f"{epi['age']}", border=True)
            st.metric("Phase", _PHASE_ICON.get(epi["phase"], epi["phase"]), border=True)
            st.metric("Pile then → now", f"{epi['start_qty'] / 1e5:,.1f} → {epi['now_qty'] / 1e5:,.1f} lakh",
                      border=True, help="Borrowed shares, roll- and split-adjusted.")
            st.metric("Price since start", f"{epi['move_since_pct']:+.1f}%", border=True)
            st.metric("Avg borrow price", f"₹{epi['avg_borrow_px']:,.2f}",
                      delta=f"shorts {'winning' if epi['shorts_pnl_pct'] > 0 else 'losing'} "
                            f"{abs(epi['shorts_pnl_pct']):.1f}%",
                      delta_color="off", border=True)
    else:
        st.info("Nothing meaningful is out on loan in this stock today.")

    x = eff.set_index("trade_date")
    st.subheader(f"{sym} — price vs the short sellers' average borrow price", divider="gray")
    st.line_chart(x[["close_adj", "avg_borrow_px"]].rename(
        columns={"close_adj": "Close ₹ (split-adjusted)", "avg_borrow_px": "Avg borrow price ₹"}),
        height=240)
    st.caption("When the close is BELOW the borrow-price line, short sellers are sitting on a profit.")
    st.subheader("Shares out on loan — raw vs roll-adjusted", divider="gray")
    st.line_chart(x[["raw", "eff"]].rename(columns={"raw": "Raw (file total)",
                                                     "eff": "Roll-adjusted (used here)"}), height=240)
    st.caption("The raw line jumps up in the week before each monthly return date because "
               "rolled loans are counted twice; the roll-adjusted line removes that.")

    if epi:
        w = se.weekly_adds(eff, epi["start"])
        if not w.empty:
            st.subheader(f"How the pile was built, week by week (since {epi['start']:%d %b %Y})",
                         divider="gray")
            st.bar_chart(w.set_index("week")[["added"]].rename(columns={"added": "Shares added (−returned)"}),
                         height=200)
            st.dataframe(w.sort_values("week", ascending=False), hide_index=True, width="stretch",
                         column_config={
                             "week": _hdc("Week of", format="DD MMM YYYY"),
                             "added": _hnc("Shares added", format="compact",
                                           help="Net roll-adjusted change that week; − = returned."),
                             "avg_close": _hnc("Avg close ₹", format="%.2f"),
                             "pile_end": _hnc("Pile at week end", format="compact"),
                             "sessions": _hnc("Sessions", format="%d"),
                         })

    fee = _symbol(sym, d - timedelta(days=days), d)
    fees = fee.loc[fee["fee_vwap"].notna(), ["trade_date", "fee_vwap", "lent_qty", "min_days",
                                             "close_price"]] if not fee.empty else fee
    if len(fees):
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

    tl = _timeline_ev()
    if tl.get("tests"):
        st.subheader("Does the timing of a build-up matter?", divider="gray")
        st.caption("Pre-registered: 3 questions × 2 horizons = 6 tests, bar |t| > "
                   f"{tl.get('bar', 2.64)}. Sector-neutral, entry at the next day's close.")
        names = {"T1_flag_age": "New fresh flag (≤5 sessions) minus old (≥11)",
                 "T2_shorts_pnl": "Shorts losing (price >5% above avg borrow) minus winning",
                 "T3_covering": "Covering (pile ≥30% off peak) minus holding, after heavy flag"}
        rows = []
        for k, lab in names.items():
            for h in (5, 21):
                t = tl["tests"].get(f"{k}_{h}", {})
                if t:
                    rows.append({"Question": lab, "Horizon": f"{h} sessions",
                                 "Group A bps": t["a_bps"], "Group B bps": t["b_bps"],
                                 "A − B bps": t["diff_bps"], "t": t["t"],
                                 "Years same sign": f"{t['years_same_sign']}/{t['n_years']}",
                                 "Passes": "✅" if t["passes"] else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("None passes: when and how a pile was built does not reliably change what the "
                   "flag means. The timeline columns are description, not forecast.")

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
