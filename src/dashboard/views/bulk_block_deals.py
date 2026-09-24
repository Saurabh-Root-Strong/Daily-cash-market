"""
Bulk & Block Deals view.

Reads the NSE bulk-deal / block-deal archive (2010-present in the DB, scored from
2018 where daily_data starts) and shows, for a chosen session, who traded size in
which name — plus the measured answer to the question the page exists for: what
happens to those stocks over the following one to two months.

The measured answer is uncomfortable and the page says so out loud: a bulk deal is
followed by UNDERPERFORMANCE, worst when the deal nets to a BUY, and a block deal
is followed by nothing at all. Nothing here is dressed up as a buy signal.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics import deal_signals as ds
from src.dashboard.column_help import nc as _hnc, tc as _htc, dc as _hdc


@st.cache_data(ttl=60)
def _stamp() -> str:
    """Build stamp of the deal tables, used as a cache KEY by every loader below.

    A plain TTL is the wrong tool here. The nightly job rebuilds these tables at
    ~10:30; with ttl=1800 the page kept serving the previous snapshot for up to half
    an hour after that, and the symptom was invisible: the date picker's max_value is
    derived from the cached frame, so 17 and 18 September were simply greyed out as
    if NSE had never published them, while the sidebar's own trading date said 18 Sep.
    Keying on (last session, built_at) means a rebuild invalidates on the next rerun,
    and a table that has NOT changed is never re-read.
    """
    try:
        df = ds.query_dataframe(
            "SELECT max(trade_date) d, max(built_at) b FROM deal_client_events", [])
        raw = ds.query_dataframe("SELECT max(trade_date) d FROM bulk_block_deals", [])
        return f"{df.d.iat[0]}|{df.b.iat[0]}|{raw.d.iat[0]}"
    except Exception:
        return "unknown"


def _feed_ahead_of_scored() -> tuple[object, object]:
    """(newest session in the raw feed, newest session that has been scored)."""
    try:
        df = ds.query_dataframe(
            "SELECT (SELECT max(trade_date) FROM bulk_block_deals) raw, "
            "       (SELECT max(trade_date) FROM deal_client_events) scored", [])
        return df.raw.iat[0], df.scored.iat[0]
    except Exception:
        return None, None


@st.cache_data(ttl=600)
def _deals(d: date, stamp: str = "") -> pd.DataFrame:
    return ds.load_deals(d)


@st.cache_data(ttl=600)
def _flow(d: date, stamp: str = "") -> pd.DataFrame:
    return ds.load_deal_flow(d)


@st.cache_data(ttl=1800)
def _outcomes(stamp: str = "") -> pd.DataFrame:
    return ds.load_outcomes()


@st.cache_data(ttl=1800)
def _client_events(stamp: str = "") -> pd.DataFrame:
    return ds.load_client_events("BUY")


@st.cache_data(ttl=1800)
def _client_sells(stamp: str = "") -> pd.DataFrame:
    """The sell side, needed to tell crowded BUYING from distribution."""
    return ds.load_client_events("SELL")


@st.cache_data(ttl=1800)
def _leaderboard(d_from: date, d_to: date, min_events: int,
                 stamp: str = "") -> pd.DataFrame:
    """Cached on the primitives, not the DataFrame.

    entity_leaderboard walks ~7,000 client groups and costs ~0.9s. Uncached it ran
    again on every slider drag, every type filter and every keystroke in the search
    box, because Streamlit re-executes the whole script each time.
    """
    ev = _client_events(stamp)
    evw = ev[(ev["trade_date"] >= pd.Timestamp(d_from))
             & (ev["trade_date"] <= pd.Timestamp(d_to))]
    return ds.entity_leaderboard(evw, min_events=min_events)


@st.cache_data(ttl=1800)
def _base_rates(stamp: str = "") -> dict:
    """Measured 1-month base rate per event kind, read from the outcomes table.

    Computed, never hardcoded: these numbers move whenever the table is rebuilt, and
    a stale literal in the UI is how a dashboard starts quietly lying.
    """
    out = ds.load_outcomes()
    if out.empty:
        return {}
    bulk = out[out.is_bulk == 1]
    rates = {}
    for key, sub in (("NET BUY", bulk[bulk.side == "BUY"]),
                     ("NET SELL", bulk[bulk.side == "SELL"]),
                     ("ROUND TRIP", bulk[bulk.side == "FLAT"]),
                     ("BLOCK", out[(out.is_block == 1) & (out.is_bulk == 0)])):
        if len(sub) >= 50:
            m, t, _ = ds._nw_t(sub.groupby("trade_date")["m21"].mean(), 21)
            rates[key] = (m * 100, t)
    return rates


def _flag(side: str, deal_type: str) -> str:
    """The measured base rate for this kind of event, as a one-glance tag."""
    rates = _base_rates(_stamp())
    key = "BLOCK" if deal_type == "BLOCK" else side
    if key not in rates:
        return ""
    pct, t = rates[key]
    if abs(t) < 2:
        return f"⚪ no measured edge ({pct:+.1f}% / 1m)"
    return f"{'🔻' if pct < 0 else '🟢'} {pct:+.1f}% / 1m (t {t:+.1f})"


def _horizon_cfg(df: pd.DataFrame) -> dict:
    """Shared config for the grouped horizon tables. Every header gets a tooltip —
    a bare "t 2 weeks" column is unreadable to anyone who did not build it."""
    cfg = {
        "Group": _htc("Group", width="large",
                      help="The set of buys being measured. Groups are not exclusive "
                           "across tables — the same buy can appear in a type row and "
                           "in a track-record row."),
        "Buys": _hnc("Buys", format="%d",
                     help="Number of client-days in this group, after netting each "
                          "client's own same-day buys and sells. A same-day round trip "
                          "is not a purchase and is excluded."),
    }
    for name in ds.HORIZON_COLS.values():
        cfg[name] = _hnc(
            name, format="%+.2f%%",
            help=f"Average excess return over {name}, measured from the NEXT session's "
                 "open (deals are disclosed after the close, so entering at the deal-day "
                 "close would be a lookahead) against matched non-event stocks — same "
                 "session, same liquidity quintile, same volume-spike decile, same "
                 "same-day move. 0% means 'behaved like comparable stocks that day'.",
        )
        cfg[f"t {name}"] = _hnc(
            f"t {name}", format="%+.2f",
            help="Newey-West t-statistic (lag = the horizon), which corrects for the "
                 "fact that overlapping windows share market moves. |t| below about 2 "
                 "is indistinguishable from noise no matter how large the % looks. "
                 "A big % next to a small t means a few outliers, not an edge.",
        )
    return cfg


def render(selected_date: date) -> None:
    st.header("🧾 Bulk & Block Deals")
    st.caption(
        "NSE names the counterparty on every bulk deal (a single client trading "
        "> 0.5% of a company's shares in a day) and on both sides of every block "
        "deal (negotiated trades ≥ ₹10 Cr / 5 lakh shares)."
    )

    stamp = _stamp()
    out = _outcomes(stamp)
    if out.empty:
        st.warning(
            "Forward-outcome table not built yet — run "
            "`python scripts/build_deal_outcomes.py`. The daily tables below still work."
        )

    # "Big Players Money" leads. It is the page's durable content -- who took size,
    # how their names have moved, and what the record says about following them --
    # while "Today's deals" is one session that is empty on a quiet day. Streamlit
    # executes every tab body regardless of which is selected, so the order costs
    # nothing; the `with` blocks below stay in their original sequence.
    tab_who, tab_day, tab_track, tab_ev = st.tabs(
        ["Big Players Money", "Today's deals", "Recent deals & how they aged",
         "What it predicts"]
    )

    # ── Tab 2: the session ────────────────────────────────────────────────────
    with tab_day:
        flow = _flow(selected_date, stamp)
        raw = _deals(selected_date, stamp)
        if flow.empty:
            st.info(f"No bulk or block deals reported for {selected_date:%d %b %Y}.")
        else:
            bulk = flow[flow.deal_type == "BULK"]
            block = flow[flow.deal_type == "BLOCK"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bulk-deal stocks", f"{len(bulk)}")
            c2.metric("Block-deal stocks", f"{len(block)}")
            c3.metric("Total deal value", f"₹{flow.value_cr.sum():,.0f} Cr")
            c4.metric("Net-BUY stocks", f"{int((bulk.side == 'NET BUY').sum())}",
                      help="Bulk deals where buys exceeded sells after netting the "
                           "same client's own round trips.")

            st.markdown("#### Netted by stock")
            st.caption(
                "Netted, not gross: one client buying and selling the same quantity "
                "the same day is an intraday round trip (a quarter of all bulk "
                "symbol-days, mostly HFT market makers), and gross quantity would "
                "score that as heavy buying."
            )
            show = flow.assign(
                Flag=[_flag(s, t) for s, t in zip(flow.side, flow.deal_type)],
                net_lakh=flow.qty_net / 1e5,
            )[["symbol", "security_name", "deal_type", "side", "Flag", "net_lakh",
               "value_cr", "net_pct_vol", "turnover_cr", "deliv_per", "n_clients",
               "block_window_rows", "clients", "buckets"]]
            st.dataframe(
                show, hide_index=True, width="stretch",
                column_config={
                    "symbol": _htc("Symbol"),
                    "security_name": _htc("Company"),
                    "deal_type": _htc("Type", help="BULK = >0.5% of shares by one client. "
                                                   "BLOCK = negotiated ≥₹10 Cr trade, both sides named."),
                    "side": _htc("Net side", help="After netting each client's own same-day "
                                                  "buys and sells. Block deals are two-sided by "
                                                  "construction, so they net to ROUND TRIP."),
                    "Flag": _htc("Historical base rate",
                                 help="What happened after this kind of event over 2018-2026, "
                                      "vs matched non-event days. See the 'What it predicts' tab."),
                    "net_lakh": _hnc("Net qty (lakh)", format="%+.2f"),
                    "value_cr": _hnc("Deal value (₹Cr)", format="%.2f"),
                    "net_pct_vol": _hnc("Net % of day vol", format="%+.1f%%",
                                        help="Net deal quantity as a share of the stock's regular-market "
                                             "traded quantity that session. Blank when the quantity went "
                                             "through the block window, which settles outside the regular "
                                             "market and is not part of that volume figure."),
                    "turnover_cr": _hnc("Stock turnover (₹Cr)", format="%.1f"),
                    "deliv_per": _hnc("Delivery %", format="%.1f%%"),
                    "n_clients": _hnc("Clients", format="%d"),
                    "block_window_rows": _hnc("Block-window lines", format="%d",
                                              help="Deal lines executed in the negotiated block window. "
                                                   "A block trade that also crosses the 0.5% bulk "
                                                   "threshold is filed in BOTH NSE feeds; the value "
                                                   "column counts it once."),
                    "clients": _htc("Client names", width="large"),
                    "buckets": _htc("Client type"),
                },
            )

            with st.expander("Every individual deal line for this session"):
                st.dataframe(
                    raw[["deal_type", "symbol", "client_name", "buy_sell", "quantity",
                         "trade_price", "value_cr", "pct_of_day_vol", "remarks"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "deal_type": _htc("Type"), "symbol": _htc("Symbol"),
                        "client_name": _htc("Client", width="large"),
                        "buy_sell": _htc("Side"),
                        "quantity": _hnc("Quantity", format="%d"),
                        "trade_price": _hnc("Price (₹)", format="%.2f"),
                        "value_cr": _hnc("Value (₹Cr)", format="%.2f"),
                        "pct_of_day_vol": _hnc("% of day vol", format="%.1f%%"),
                        "remarks": _htc("Remarks"),
                    },
                )

    # ── Tab 3: recent events and their realised excess ────────────────────────
    with tab_track:
        if out.empty:
            st.info("Build the outcomes table to populate this tab.")
        else:
            n_days = st.slider("Look back (calendar days)", 30, 365, 120, step=30)
            rec = ds.recent_events(out, days=n_days)
            kind = st.segmented_control("Event kind", ["Bulk", "Block", "All"],
                                        default="Bulk", key="deal_kind")
            if kind == "Bulk":
                rec = rec[rec.is_bulk == 1]
            elif kind == "Block":
                rec = rec[(rec.is_block == 1) & (rec.is_bulk == 0)]
            st.caption(
                "1m / 2m columns are the stock's return from the NEXT session's open, "
                "minus what matched non-event stocks did over the same window. Blank "
                "means the window has not closed yet."
            )
            show = rec[["trade_date", "symbol", "side", "value_cr", "n_clients",
                        "m21", "m42"]].copy()
            show["m21"] = show["m21"] * 100
            show["m42"] = show["m42"] * 100
            st.dataframe(
                show, hide_index=True, width="stretch", height=460,
                column_config={
                    "trade_date": _hdc("Deal date", format="DD MMM YYYY"),
                    "symbol": _htc("Symbol"),
                    "side": _htc("Net side"),
                    "value_cr": _hnc("Deal value (₹Cr)", format="%.2f"),
                    "n_clients": _hnc("Clients", format="%d"),
                    "m21": _hnc("1m vs peers %", format="%+.2f%%"),
                    "m42": _hnc("2m vs peers %", format="%+.2f%%"),
                },
            )

    # ── Tab 1: the names ──────────────────────────────────────────────────────
    with tab_who:
        ev = _client_events(stamp)
        if ev.empty:
            st.info("Run `python scripts/build_client_events.py` to populate this tab.")
        else:
            st.markdown("#### Does following a specific buyer work?")
            st.error(
                "No. Ranking entities on their past buys and then following the best of "
                "them loses money at every horizon from 1 week to 3 months. The table "
                "below is the out-of-sample test, and the best-track-record quartile is "
                "the worst performer at 1 week. Measured, not assumed.",
                icon="🚫",
            )
            wf = ds.walk_forward_table(ev, min_prior=5)
            st.dataframe(wf, hide_index=True, width="stretch",
                         column_config=_horizon_cfg(wf))
            st.caption(
                "Each entity is ranked by the average outcome of its OWN earlier buys — "
                "counting only buys whose 1-month window had already closed, so nothing "
                "unresolved leaks into the ranking — then its next buys are measured. "
                "Requiring the prior record to be above +3% makes the forward result "
                "worse, not better (−2.1% at 1 month): the ranking mean-reverts, which "
                "is the signature of luck, not skill."
            )

            st.markdown("#### By type of buyer")
            ett = ds.entity_type_table(ev)
            st.dataframe(ett, hide_index=True, width="stretch",
                         column_config=_horizon_cfg(ett))
            st.caption(
                "Broker prop desks look positive at 1-2 weeks (+0.5%) — that is not an "
                "edge: t = 0.69, the MEDIAN buy is −1.45%, only 41% beat their peers, "
                "and the same group is the worst performer by 3 months (−4.9%, t −2.52, "
                "median −9.1%). Its positive years are 2018/2021/2023 (all t < 1.7) "
                "while 2025 and 2026 are negative. A positive average with a negative "
                "median is a right tail, not a repeatable trade."
            )
            st.caption(
                "Mutual funds and insurers are the only buyer type that prints positive, "
                "peaking at 1 month (+1.2%) and gone by 2. Treat it as unproven: across "
                "the 35 type × horizon cells searched, a permutation that reshuffles WHO "
                "bought within each day produces a bigger best |t| than the real data "
                "does (null median 4.40 vs observed 3.54, p = 0.99). The median MF buy "
                "is −1.1% and only 37% beat their peers — the positive average is a "
                "right tail, not a typical trade. FPI / foreign buying is NOT positive."
            )

            st.markdown("#### Repeat buying — the 'accumulation' story")
            rb = ds.repeat_buy_table(ev)
            st.dataframe(rb, hide_index=True, width="stretch",
                         column_config=_horizon_cfg(rb))
            st.caption(
                "An entity coming back to buy the same name again within 3 months is the "
                "single most negative pattern in this dataset — −7.5% over 3 months. "
                "Persistent bulk-deal buying is distribution, not accumulation."
            )

            st.markdown("#### Every client, every horizon")
            st.caption(
                "The full per-client table: what the stock did after each named client "
                "bought, at 1 week, 2 weeks, 1 month, 2 months and 3 months. Click any "
                "header to sort, hover any header for what the column means, and click "
                "a row to see the individual stocks that client bought. Read MEDIAN "
                "next to the average \u2014 a name can average +27% on a median of "
                "\u22120.5%, meaning one lottery ticket carried it while the typical trade lost."
            )

            dmin = ev["trade_date"].min().date()
            dmax = ev["trade_date"].max().date()

            # TWO single-date inputs, not one range input. st.date_input in range mode
            # returns a 1-TUPLE on the rerun that fires between the two clicks, so the
            # period is momentarily invalid and whatever the code does with it (fall
            # back to the full range, or blank the table) flashes on screen mid-pick.
            # Two independent dates are always valid, so there is no in-between state.
            # Session dates, not calendar dates. "Last 10 days" in market speak means 10
            # TRADING sessions: 10 calendar days back from 15 Sep 2026 lands on 06 Sep
            # and catches only 4 sessions, because of the weekend plus the 14 Sep gap.
            # A session count gives the same amount of data whatever the holidays do.
            sessions = ev["trade_date"].dt.date.drop_duplicates().sort_values().tolist()

            def _nth_last_session(n: int) -> date:
                return sessions[-n] if len(sessions) >= n else sessions[0]

            if "deal_from" not in st.session_state:
                st.session_state.deal_from = _nth_last_session(10)
                st.session_state.deal_to = dmax

            def _preset(days: int | None) -> None:
                st.session_state.deal_to = dmax
                st.session_state.deal_from = dmin if days is None else max(dmin, dmax - timedelta(days=days))

            def _preset_sessions(n: int) -> None:
                st.session_state.deal_to = dmax
                st.session_state.deal_from = _nth_last_session(n)

            # The picker can only offer sessions that survived SCORING (the join to
            # daily_data plus >= 5 matched controls). If the nightly deal fetch ran but
            # the rebuild did not, the newest sessions exist in the raw feed and are
            # still absent here -- previously that looked exactly like "NSE published
            # nothing", with no way to tell the two apart from the screen.
            raw_max, scored_max = _feed_ahead_of_scored()
            if raw_max is not None and scored_max is not None and raw_max > scored_max:
                st.warning(
                    f"Deals are loaded up to **{raw_max:%d %b %Y}**, but scored only to "
                    f"**{scored_max:%d %b %Y}**, so the picker stops at the earlier date. "
                    "The analytics rebuild has not run since the last fetch — run "
                    "`python scripts/build_client_events.py`.",
                    icon="⚠️",
                )

            pc = st.container(horizontal=True)
            pc.caption(
                f"Quick range:  ·  deals available {dmin:%d %b %Y} → {dmax:%d %b %Y}"
            )
            pc.button("5D", on_click=_preset_sessions, args=(5,), key="dp5d",
                      help="Last 5 TRADING sessions. Counted in sessions, not calendar "
                           "days: 5 calendar days back from a Monday is a Wednesday, "
                           "which would be 3 sessions, not 5.")
            pc.button("10D", on_click=_preset_sessions, args=(10,), key="dp10d",
                      help="Last 10 TRADING sessions (the default). Counted in sessions, "
                           "not calendar days, so holidays cannot shrink the window.")
            pc.button("1M", on_click=_preset, args=(30,), key="dp1",
                      help="Last 30 calendar days of deals.")
            pc.button("3M", on_click=_preset, args=(91,), key="dp3")
            pc.button("6M", on_click=_preset, args=(182,), key="dp6")
            pc.button("1Y", on_click=_preset, args=(365,), key="dp12",
                      help="A year is the shortest window where every horizon, "
                           "including 3 months, has resolved for most buys.")
            pc.button("All", on_click=_preset, args=(None,), key="dpall",
                      help="Everything from 2018.")

            fc1, fc2, fc3 = st.columns([1, 1, 2.4])
            d_from = fc1.date_input(
                "From", key="deal_from", min_value=dmin, max_value=dmax, format="DD/MM/YYYY",
                help=f"Deals struck ON or AFTER this date. Selectable range is "
                     f"{dmin:%d %b %Y} to {dmax:%d %b %Y} — anything outside it is "
                     "greyed out because there is no deal data there.",
            )
            d_to = fc2.date_input(
                "To", key="deal_to", min_value=dmin, max_value=dmax, format="DD/MM/YYYY",
                help=f"Deals struck ON or BEFORE this date. The calendar stops at "
                     f"{dmax:%d %b %Y}, the most recent session in the deal feed. "
                     "Dates after it are greyed out because those sessions have not "
                     "happened or have not been fetched yet, not because they are "
                     "excluded.",
            )
            if d_from > d_to:
                st.error("'From' is after 'To' — showing nothing. Swap the two dates.")
                d_from, d_to = d_to, d_from

            evw = ev[(ev["trade_date"] >= pd.Timestamp(d_from))
                     & (ev["trade_date"] <= pd.Timestamp(d_to))]

            span_days = (d_to - d_from).days + 1
            n_sessions = evw["trade_date"].dt.date.nunique()
            fc3.caption(
                f"**{len(evw):,} buys** by **{evw['client_key'].nunique():,} clients** "
                f"across **{n_sessions} trading sessions** "
                f"({d_from:%d %b %Y} \u2192 {d_to:%d %b %Y}, {span_days} calendar days)."
            )
            cov = ds.horizon_coverage(evw)
            if any(v < 0.5 for v in cov.values()):
                fc3.warning(
                    "Forward window not closed yet for: "
                    + ", ".join(f"{n} ({v*100:.0f}% resolved)"
                                for n, v in cov.items() if v < 0.5)
                    + ". Those columns describe only the oldest buys in the window, not "
                      "the window \u2014 read them as blank. Use 1Y or All to judge "
                      "performance; short windows are for seeing WHO bought, not how it went.",
                    icon="\u23f3",
                )

            gc1, gc2, gc3 = st.columns([1, 1.4, 1.2])
            max_possible = int(evw.groupby("client_key").size().max()) if not evw.empty else 1
            slider_max = max(5, min(100, max_possible))
            # Scale the default to the window, counted in SESSIONS. Over 8 years a
            # 15-buy floor keeps the table readable; over 10 sessions it would hide 90%
            # of the clients (161 -> 15), including every institution that bought once
            # or twice -- which is exactly what a short window is meant to show.
            if n_sessions <= 25:
                span_default = 1
            elif n_sessions <= 120:
                span_default = 5
            elif n_sessions <= 250:
                span_default = 8
            else:
                span_default = 15
            default_min = min(span_default, slider_max)
            min_ev = gc1.slider(
                "Minimum buys by that client", 1, slider_max, default_min, step=1,
                help="Hide clients with fewer buys than this. Small samples are where "
                     "extreme averages come from: with 15 buys, a single +80% trade "
                     "moves the average by more than 5 percentage points. The ceiling "
                     "adapts to the busiest client in the selected period.",
            )
            all_types = [ds.bucket_label(b) for b in ds.CLIENT_BUCKETS]
            picked = gc2.multiselect(
                "Client type", options=all_types, default=[],
                help="Filter by who the client is. Classification is keyword-based on "
                     "the name NSE publishes, so it is a label, not a verified fact: "
                     "'Individual / HNI' means the name looks like a person, and anyone "
                     "here traded >0.5% of a company in a day. Leave empty for all types.",
            )
            search = gc3.text_input(
                "Search client name", value="",
                help="Substring match on the client name, case-insensitive. Useful for "
                     "checking one desk across every stock it bought.",
            )

            lb = _leaderboard(d_from, d_to, min_ev, stamp)
            if not lb.empty and picked:
                lb = lb[lb["Type"].isin(picked)]
            if not lb.empty and search.strip():
                lb = lb[lb["Client"].str.contains(search.strip(), case=False, na=False)]
            lb = lb.reset_index(drop=True)

            if lb.empty:
                st.info(
                    "No client matches those filters. A short date range plus a high "
                    "minimum-buys setting is the usual cause."
                )
            else:
                st.caption(f"{len(lb)} clients shown \u2014 click a row for that client's stocks.")
                cols = ["Client", "Type", "Buys", "Stocks", "Days",
                        "First buy", "Last buy",
                        "Since buy median", "vs Nifty median", "In profit now %",
                        "Median % of delivery",
                        "Sold back %"]
                for nm in ds.HORIZON_COLS.values():
                    cols += [nm, f"{nm} median", f"{nm} n"]
                cols += ["Hit rate 1 month"]
                picked_rows = st.dataframe(
                    lb[cols], hide_index=True, width="stretch", height=430,
                    column_config=_leaderboard_cfg(),
                    on_select="rerun", selection_mode="single-row", key="deal_client_pick",
                )
                sel = list(picked_rows.selection.rows) if picked_rows.selection else []
                if sel:
                    row = lb.iloc[sel[0]]
                    st.markdown(f"##### {row['Client']} \u2014 every stock bought in this period")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Buys", int(row["Buys"]))
                    m2.metric("Distinct stocks", int(row["Stocks"]))
                    def _pct(x: float) -> str:
                        return "\u2014" if pd.isna(x) else f"{x:+.2f}%"
                    m3.metric("Median move since buy", _pct(row["Since buy median"]),
                              help="Typical move from this client's own deal price to "
                                   "the latest close, split-adjusted. Raw, not excess "
                                   "- compare with the Nifty column in the table.")
                    m4.metric("Median 1 month vs peers", _pct(row["1 month median"]),
                              help="The typical trade's 1-month excess over matched "
                                   "peers. Blank until a buy in the window is a month old.")
                    trades = ds.client_trades(evw, row["client_key"])
                    st.dataframe(
                        trades, hide_index=True, width="stretch", height=330,
                        column_config=_trades_cfg(),
                    )
                    st.caption(
                        "One row per stock-day this client ended net long. Returns are "
                        "measured from the NEXT session's open and are excess over "
                        "matched non-event stocks; blank means that horizon has not "
                        "closed yet."
                    )

            st.markdown("##### Same stock, several buyers")
            st.caption(
                "Which stocks more than one named client bought in this window, and "
                "who else was SELLING them at the same time. Click a row for every "
                "client on both sides."
            )
            st.info(
                "**Agreement does not improve the odds \u2014 it worsens them.** Counting "
                "the distinct clients that bought the same stock within 5 sessions: "
                "1 buyer returns \u22121.43% over 1 month, 2 buyers \u22121.24%, **3 buyers "
                "\u22122.04% at 1 month and \u22127.17% at 3 months (t \u22125.15)**. Dropping HFT "
                "desks from the count does not rescue it (3 buyers: \u22126.83% / 3mo), nor "
                "does requiring 2+ institutions (\u22120.05% / 1mo, \u22123.35% / 3mo), nor net "
                "breadth. The best of ~20 such cuts reaches |t| 3.25 against a "
                "permutation null whose 95th percentile is 3.67 \u2014 **p = 0.21**, i.e. "
                "noise. Use this to see WHO is in a name, not as confirmation.",
                icon="\U0001f4cb",
            )

            cc1, cc2 = st.columns([1, 1])
            min_buyers = cc1.slider(
                "Minimum distinct buyers", 1, 8, 2, step=1, key="deal_min_buyers",
                help="Show only stocks bought by at least this many DIFFERENT client "
                     "entities in the selected period. 2 answers 'did anyone else buy "
                     "this too?'. Note that near-duplicate spellings of one firm are "
                     "merged first, so a desk cannot count twice.",
            )
            ex_hft = cc2.checkbox(
                "Ignore HFT market makers in the count", value=False,
                key="deal_cons_nohft",
                help="HFT desks quote both sides all day and appear in 27-45% of every "
                     "multi-buyer cluster. Tick this to count only clients that took a "
                     "view - three 'buyers' that are three market makers is inventory, "
                     "not three opinions.",
            )

            sells_all = _client_sells(stamp)
            sw = sells_all[(sells_all["trade_date"] >= pd.Timestamp(d_from))
                           & (sells_all["trade_date"] <= pd.Timestamp(d_to))] \
                if not sells_all.empty else sells_all
            evw_c = evw[evw["bucket"].map(ds.bucket_label).isin(picked)] if picked else evw
            cons = ds.stock_consensus(evw_c, sw, exclude_hft=ex_hft)
            if not cons.empty:
                cons = cons[cons["Buyers"] >= min_buyers].reset_index(drop=True)

            if cons.empty:
                st.info(
                    "No stock in this window was bought by that many different clients. "
                    "Lower the minimum, or widen the date range."
                )
            else:
                st.caption(
                    f"{len(cons)} stocks \u2014 click a row for the clients on both sides."
                )
                ccols = ["Symbol", "Buyers", "Sellers", "Net buyers", "Buys", "Days",
                         "First buy", "Last buy", "Who",
                         "Net bought (Cr)", "Value (Cr)",
                         "Median % of delivery", "Since buy median", "vs Nifty median"]
                for nm in ds.HORIZON_COLS.values():
                    ccols += [nm, f"{nm} n"]
                pick_stock = st.dataframe(
                    cons[ccols], hide_index=True, width="stretch", height=380,
                    column_config=_consensus_cfg(),
                    on_select="rerun", selection_mode="single-row", key="deal_stock_pick",
                )
                ssel = list(pick_stock.selection.rows) if pick_stock.selection else []
                if ssel:
                    srow = cons.iloc[ssel[0]]
                    st.markdown(
                        f"##### {srow['Symbol']} \u2014 {int(srow['Buyers'])} buyers, "
                        f"{int(srow['Sellers'])} sellers in this window"
                    )
                    if srow["Sellers"] >= srow["Buyers"]:
                        st.warning(
                            f"As many disclosed SELLERS as buyers "
                            f"({int(srow['Sellers'])} vs {int(srow['Buyers'])}). Crowded "
                            "buying with a crowded exit is distribution, and stocks with "
                            "net breadth of \u22122 returned \u22124.82% over 1 month.",
                            icon="\u26a0\ufe0f",
                        )
                    st.dataframe(
                        ds.stock_participants(evw_c, sw, srow["Symbol"]),
                        hide_index=True, width="stretch", height=300,
                        column_config=_participants_cfg(),
                    )
                    st.caption(
                        "Both sides as NSE published them. Remember the disclosure rule "
                        "is one-sided: a leg under 0.5% of the company is never printed, "
                        "so an absent seller is not proof nobody sold."
                    )

            with st.expander("Is a reported 'net long' actually a purchase?"):
                st.markdown(
                    """
A disclosed net long is an **upper bound** on real buying, never the true net, and
three separate mechanisms can inflate it. All three are now measured:

- **The disclosure rule is one-sided.** NSE publishes a client's leg only when that
  leg crosses 0.5% of the company in a single day. Buy 0.6% and sell 0.4% the same
  session and only the BUY is printed. The netting here can only net what was
  published.
- **It may never have settled.** A real accumulation shows up in delivery: the
  shares that were *not* squared off intraday. Measured properly, this test finds
  almost no impossible purchases \u2014 only **10 of 12,662 bulk buys (0.08%)** exceed
  the whole day's delivered quantity. What it does reveal is how *meaningful* each
  buyer's position is: HFT market makers sit at a median of **1.4% of the day's
  delivery** (inventory, not accumulation), while mutual funds sit at **42%** and
  other funds at **46%** \u2014 they take a large share of everything genuinely
  changing hands. That column is now on the table as "Size vs delivery".
- **It may be a cross.** 2,701 bulk buy legs are exactly matched in size and price
  by another account's sell the same day in the same stock — the shape of a
  pre-arranged transfer rather than open-market buying.
- **It may be sold straight back.** 14.9% of buys see the same client disclosed as a
  net seller in that stock within 21 days, 18.6% within 63.

**And here is the part that matters:** filtering to what looks like *genuine,
settled, held* accumulation does not rescue the result — it makes it worse.
Bulk buys that fit inside the day's delivery **and** show no disclosed exit within
3 months return **−1.24% / 1wk, −2.03% / 1mo, −3.89% / 3mo (t −4.25)**, versus
−0.35% / 1mo for the churned and crossed ones. So the negative edge is not an
artifact of fake buying. Real-looking accumulation underperforms too.

The one exception, and treat it as a hypothesis rather than a finding: mutual-fund
and insurance buys that both settled and were held (n=151) return **+4.61% over 3
months (t 3.06)** — the only cut where the institutional signal strengthens with
time instead of fading. It is a small sample reached after slicing the same data
several ways, which is exactly how false positives are manufactured.
"""
                )

            st.caption(
                "Formally searched for a name that works: 137 clients with \u226515 buys, a "
                "per-client p-value from shuffling WHICH client bought within each day, "
                "then Benjamini-Hochberg false-discovery control. At 1 month, 8 names "
                "survive \u2014 5 of them NEGATIVE. The only name positive at 1, 2 and 3 "
                "months (Goldman Sachs Bank Europe SE ODI) has all 110 of its buys "
                "inside Aug-2025\u2192Jun-2026, decays from +4.4% to +0.7% at 3 months "
                "across its own two halves, drops to t=1.3 once its five biggest "
                "winners are removed, and is contradicted by every other ODI issuer "
                "(Morgan Stanley \u22128.6%, Soci\u00e9t\u00e9 G\u00e9n\u00e9rale \u22123.2% at 3 months). "
                "Across all clients positive in their first half, their later buys "
                "returned \u22120.7% to \u22122.6%."
            )

    # ── Tab 4: the evidence ───────────────────────────────────────────────────
    with tab_ev:
        if out.empty:
            st.info("Build the outcomes table to populate this tab.")
        else:
            st.markdown("#### Forward excess return after a deal")
            st.dataframe(ds.evidence_table(out), hide_index=True, width="stretch",
                         column_config=_evidence_cfg())
            st.markdown("#### Bulk net BUY, split by how liquid the stock is")
            st.caption("It is not only a microcap effect — it survives in the most "
                       "traded tier.")
            st.dataframe(ds.evidence_by_size(out), hide_index=True, width="stretch",
                         column_config=_evidence_cfg())

            st.markdown("#### How this is measured")
            st.markdown(
                """
- **Entry is the next session's open.** Deals are disclosed after the close, so
  scoring from the deal-day close would be a lookahead.
- **The benchmark is matched non-event days**, not an index: same session, same
  liquidity quintile, same volume-spike decile, same same-day-move decile. A deal
  day is by construction a high-volume, big-move day in a small name, and all
  three of those revert on their own — an index benchmark would credit that
  reversion to the deal. A placebo drawn from the same control pool prints
  **−0.00% (t = −0.16)**.
- **Corporate actions** cannot fake a move: returns compound `close / prev_close`
  (NSE's prev_close is ex-adjusted), never `close_t+n / close_t`.
- **t-statistics are Newey-West** (lag = horizon). The forward windows overlap
  heavily; a plain t would be inflated several-fold.
- **Multiple testing:** the 44 candidates searched here were run against 400
  permutations that reassign the event label within the same matched cells. The
  null's best |t| has median 2.6 and 95th percentile 4.0; the observed best is
  5.4, giving a reality-check **p = 0.0025**.
- **Coverage:** series EQ only, 2018 onward, where `daily_data` begins. Audited
  against the raw feed: **28.6% of deal stock-days are never scored** — SME
  platform 6,857 (SM) + 2,871 (ST), trade-to-trade 2,477 (BE) + 184 (BZ), and 916
  with no price row at all (delisted, renamed, or suspended). Newly listed stocks
  are also excluded until they have 20 sessions of history, because the liquidity
  control needs a 20-day average — so **listing-day and anchor-investor deals never
  appear here**. Delisted names dropping out biases measured returns *upward*, so
  the negative result is, if anything, understated.
- **Freshness:** the tables are rebuilt nightly after the fetch. If the rebuild has
  not run, a banner says so at the top of the client section rather than letting the
  page quietly show a stale window.
"""
            )
            st.markdown("#### What this does NOT support")
            st.markdown(
                """
- **Not a short signal in practice.** Most bulk-deal names have no stock-futures
  or SLB leg, so the negative side is largely unharvestable. Treat it as an
  avoid / trim flag.
- **Block deals are not a signal at all** — measured excess is ~0 at both
  horizons, and the year-by-year sign flips. The apparent +1% they show against a
  *median* benchmark is a skew artifact: cross-sectional stock returns are
  right-skewed, so a median benchmark flatters every event.
- **The MF / insurance positive is thin.** It is a few hundred events, it is not
  directional (their *sells* look similar to their buys), and it fades by two
  months. Most plausibly it says the company is institution-grade, not that the
  trade was well timed.
"""
            )


def _leaderboard_cfg() -> dict:
    """Column config for the per-client table. Every header carries a tooltip:
    without one, "1 month med" is ambiguous enough to be read as the opposite of
    what it is."""
    cfg = {
        "Client": _htc("Client", width="large",
                       help="The client name exactly as NSE publishes it. NSE types "
                            "these by hand, so one desk can appear under several "
                            "spellings; near-duplicates are merged where the spelling "
                            "allows, but not always."),
        "Type": _htc("Type", help="Keyword classification of the client name — a label, "
                                  "not a verified fact."),
        "Buys": _hnc("Buys", format="%d",
                     help="Number of days this client ended NET LONG in a stock after "
                          "netting its own same-day buys and sells. Same-day round "
                          "trips are excluded, they are not purchases."),
        "Stocks": _hnc("Stocks", format="%d",
                       help="Distinct stocks bought. A high Buys with a low Stocks means "
                            "the record rests on repeat trades in a few names."),
        "Hit rate 1 month": _hnc("Hit rate 1m", format="%.0f%%",
                                 help="Share of this client's buys that BEAT matched "
                                      "peers over 1 month. 50% is a coin flip. A high "
                                      "average with a hit rate near 40% means the "
                                      "average is carried by a few outliers."),
        "Median % of delivery": _hnc(
            "Size vs delivery", format="%.0f%%",
            help="How big this client's typical open-market purchase was against ALL "
                 "shares DELIVERED in that stock that day. Delivery = the quantity "
                 "that was not squared off intraday, i.e. shares that genuinely "
                 "changed hands. 1-2% means the net position was trivial next to real "
                 "settling flow - inventory, not accumulation (every HFT desk sits "
                 "here). 40-80% means this client took most of the stock actually "
                 "delivered that day, which is what dominant buying looks like. Above "
                 "100% is arithmetically impossible for a delivered purchase and "
                 "implies an offsetting sale that was never published. BULK rows only: "
                 "block trades settle outside the regular market and never enter the "
                 "delivery figure, so they are left out rather than failed."),
        "Sold back %": _hnc("Sold back %", format="%.0f%%",
                            help="Share of buys where the SAME client reappears as a "
                                 "net seller in the SAME stock within 3 months. "
                                 "Warning on the other direction: NSE only publishes a "
                                 "leg once it crosses 0.5% of the company in one day, "
                                 "so a slower or smaller exit is INVISIBLE. A low "
                                 "number is not proof the position was held."),
        "Since buy median": _hnc(
            "Move since buy", format="%+.1f%%",
            help="MEDIAN move from the price this client paid (its average deal price "
                 "that day) to the latest close, across its buys in the period. "
                 "Split- and bonus-adjusted, and followed through spells in the BE "
                 "trade-to-trade series. Never blank, unlike the horizon columns, so "
                 "it works on a 10-day window. Caution: it is RAW, not excess - in a "
                 "rising market every buyer looks right - and it mixes ages: a buy from "
                 "yesterday and one from two years ago count equally. Short windows "
                 "keep the ages comparable."),
        "vs Nifty median": _hnc(
            "vs Nifty 50", format="%+.1f pp",
            help="MEDIAN of (stock move since buy minus Nifty 50 move over the same "
                 "dates), in percentage points. Positive = the typical stock this "
                 "client bought beat the index since the purchase. The Nifty leg runs "
                 "from the deal-day close, the stock leg from the deal price. Bulk-deal "
                 "names are mostly small caps, which can drift away from the Nifty 50 "
                 "for reasons that have nothing to do with the buyer."),
        "In profit now %": _hnc(
            "In profit now", format="%.0f%%",
            help="Share of this client's buys trading ABOVE the price it paid, at the "
                 "latest close. 50% is a coin flip. A high median move with a low "
                 "in-profit share means a few big winners are carrying it."),
        "Days": _hnc("Days", format="%d",
                     help="Distinct SESSIONS this client bought on. Compare it with "
                          "Buys: a desk showing 40 buys across 3 days is really three "
                          "decisions, not forty, and its averages are far less solid "
                          "than the Buys count suggests."),
        "First buy": _hdc("First buy", format="DD MMM YYYY",
                          help="Date of the earliest purchase in the selected period. "
                               "A record confined to a short span is one market regime, "
                               "not a track record."),
        "Last buy": _hdc("Last buy", format="DD MMM YYYY",
                         help="Date of the most recent purchase in the selected period. "
                              "An old date means the desk has stopped appearing and "
                              "these numbers are history, not current behaviour."),
    }
    for nm in ds.HORIZON_COLS.values():
        cfg[nm] = _hnc(
            nm, format="%+.2f%%",
            help=f"AVERAGE return over {nm} after this client bought, measured from the "
                 "NEXT session's open and stated as excess over matched non-event "
                 "stocks (same day, liquidity, volume spike and same-day move). "
                 "Averages are pulled hard by single outliers — read the median beside it.",
        )
        cfg[f"{nm} median"] = _hnc(
            f"{nm} med", format="%+.2f%%",
            help=f"MEDIAN excess return over {nm} — the typical trade. When the median "
                 "is negative and the average is positive, most of this client's buys "
                 "lost and one or two winners carried the average.",
        )
    return cfg

def _trades_cfg() -> dict:
    """Column config for the per-client drill-down."""
    cfg = {
        "Date": _hdc("Deal date", format="DD MMM YYYY"),
        "Symbol": _htc("Symbol"),
        "Deal": _htc("Type", help="BULK = >0.5% of the company's shares by one client. "
                                  "BLOCK = negotiated trade in the block window."),
        "Qty (lakh)": _hnc("Net qty (lakh)", format="%+.2f",
                           help="Shares bought minus sold by this client in this stock "
                                "that day, in lakhs."),
        "Value (Cr)": _hnc("Value (Rs Cr)", format="%.2f",
                           help="Traded value of this client's lines in this stock."),
        "Avg price": _hnc("Avg price (Rs)", format="%.2f",
                          help="Quantity-weighted price of this client's deal lines."),
        "Last price": _hnc("Last price (Rs)", format="%.2f",
                           help="Latest close on record, NOT adjusted. After a split or "
                                "bonus it is not comparable with Avg price - use "
                                "'Since buy', which is adjusted."),
        "Since buy %": _hnc("Since buy", format="%+.1f%%",
                            help="How far the stock has moved from THIS client's average "
                                 "deal price to the latest close. Split- and bonus-"
                                 "adjusted, and followed through BE (trade-to-trade) "
                                 "spells. Raw move, not excess: compare with the Nifty "
                                 "column. The client's own price is used, so it can differ "
                                 "from the horizon columns, which enter at the next open."),
        "Nifty same span %": _hnc("Nifty same span", format="%+.1f%%",
                                  help="Nifty 50 from the deal-day close to the same "
                                       "'Price as of' date. The yardstick for Since buy."),
        "Sessions since": _hnc("Sessions since", format="%d",
                               help="Trading sessions from the deal to the latest price. "
                                    "0 = bought in the latest session, so Since buy only "
                                    "shows where it closed against the deal price."),
        "Price as of": _hdc("Price as of", format="DD MMM YYYY",
                            help="Date of the latest close used. Older than the latest "
                                 "session means the stock stopped trading (suspended, "
                                 "delisted, or moved to a segment not tracked here) - "
                                 "the move is frozen at that date, and the real loss "
                                 "may be far worse."),
        "% of day delivery": _hnc("% of day delivery", format="%.0f%%",
                                  help="Net purchase as a share of ALL shares delivered "
                                       "in that stock that day. Over 100% means it "
                                       "cannot have been a delivered accumulation. "
                                       "Meaningless for BLOCK rows - those settle "
                                       "outside the regular market."),
        "Exit after (days)": _hnc("Exit after (days)", format="%.0f",
                                  help="Calendar days until this client next shows up "
                                       "as a net SELLER in this stock. Blank = no "
                                       "disclosed exit, which is NOT the same as held: "
                                       "an exit below 0.5% of the company never gets "
                                       "published."),
        "% of 20d ADV": _hnc("% of 20d ADV", format="%+.1f%%",
                             help="Net purchase as a share of the stock's average daily "
                                  "traded value over the prior 20 sessions - how big "
                                  "the buy was relative to normal liquidity."),
        "Stock turnover (Cr)": _hnc("Stock turnover (Rs Cr)", format="%.1f",
                                    help="The stock's total traded value that session."),
    }
    for nm in ds.HORIZON_COLS.values():
        cfg[nm] = _hnc(nm, format="%+.2f%%",
                       help=f"Excess return over {nm} from the next session's open, "
                            "against matched non-event stocks. Blank = window still open.")
    return cfg

def _consensus_cfg() -> dict:
    """Column config for the per-stock consensus table."""
    cfg = {
        "Symbol": _htc("Symbol", help="NSE symbol, series EQ."),
        "Buyers": _hnc("Buyers", format="%d",
                       help="DISTINCT client entities that ended net long in this stock "
                            "during the selected period. Spelling variants of one firm "
                            "are merged first, so a desk cannot be counted twice. "
                            "Measured: a higher count does NOT predict a better move."),
        "Sellers": _hnc("Sellers", format="%d",
                        help="Distinct clients disclosed as net SELLERS of the same "
                             "stock in the same period. The count NSE publishes is "
                             "one-sided - a leg below 0.5% of the company never "
                             "appears - so this is a floor, not the full exit."),
        "Net buyers": _hnc("Net buyers", format="%+d",
                           help="Buyers minus Sellers. The honest version of crowding: "
                                "stocks at -2 returned -4.82% over 1 month (t -2.56), "
                                "the most negative cell in the whole breadth test."),
        "Buys": _hnc("Buys", format="%d",
                     help="Client-days of net buying, so one desk buying on 3 days "
                          "counts 3. Compare with Buyers and Days."),
        "Days": _hnc("Days", format="%d",
                     help="Distinct sessions on which any client bought."),
        "Who": _htc("Who bought", width="large",
                    help="The buyers by type, e.g. '2x HFT market maker, 1x Broker prop "
                         "desk'. Read it before the count: market makers quote both "
                         "sides all day and make up 27-45% of multi-buyer clusters."),
        "Net bought (Cr)": _hnc(
            "Net bought (Rs Cr)", format="%+.1f",
            help="Net value actually bought: net quantity x deal price, summed. This "
                 "is the number to read. It can be a small fraction of Traded value "
                 "when a desk bought and sold nearly the same amount."),
        "Value (Cr)": _hnc(
            "Traded (Rs Cr)", format="%.1f",
            help="GROSS traded value of every disclosed buy line, before netting that "
                 "client's own sales. Big here and small in 'Net bought' means churn."),
        "First buy": _hdc("First buy", format="DD MMM YYYY"),
        "Last buy": _hdc("Last buy", format="DD MMM YYYY",
                         help="Latest buying session. Buying spread over several days "
                              "is accumulation; all of it on one day is usually one "
                              "block being distributed to several accounts."),
        "Median % of delivery": _hnc(
            "Size vs delivery", format="%.0f%%",
            help="Median net purchase as a share of ALL shares delivered in the stock "
                 "that day, across the buys. Low single digits means the flow was "
                 "trivial next to real settling volume. BULK rows only."),
        "Since buy median": _hnc(
            "Move since buy", format="%+.1f%%",
            help="Median move from each buyer's own deal price to the latest close, "
                 "split-adjusted. Raw, not excess."),
        "vs Nifty median": _hnc(
            "vs Nifty 50", format="%+.1f pp",
            help="Median of (move since buy minus Nifty 50 over the same dates)."),
    }
    for nm in ds.HORIZON_COLS.values():
        cfg[nm] = _hnc(nm, format="%+.2f%%",
                       help=f"Average excess return over {nm} across this stock's buy "
                            "events, vs matched non-event stocks. One stock is a tiny "
                            "sample - read it as an anecdote, not evidence.")
        cfg[f"{nm} n"] = _hnc(f"{nm} n", format="%d",
                              help=f"How many of the buys have a CLOSED {nm} window. "
                                   "0 means the column beside it is blank, not zero.")
    return cfg


def _participants_cfg() -> dict:
    """Column config for the both-sides client list of one stock."""
    return {
        "Date": _hdc("Date", format="DD MMM YYYY"),
        "Side": _htc("Side", help="BOUGHT = the client ended the day net long in this "
                                  "stock. SOLD = net short for the day. Netting is per "
                                  "client per stock per day, so an intraday round trip "
                                  "is neither."),
        "Client": _htc("Client", width="large", help="Name as NSE publishes it."),
        "Type": _htc("Type", help="Keyword classification of the name - a label, not a "
                                  "verified fact."),
        "Deal": _htc("Deal", help="BULK = >0.5% of the company in a day. BLOCK = "
                                  "negotiated trade in the block window."),
        "Qty (lakh)": _hnc("Net qty (lakh)", format="%.2f",
                           help="Absolute net quantity for that client that day."),
        "Net value (Cr)": _hnc("Net value (Rs Cr)", format="%.2f",
                               help="Net quantity x deal price - what this client "
                                    "actually added or removed."),
        "Value (Cr)": _hnc("Traded (Rs Cr)", format="%.2f",
                           help="GROSS value of that client's disclosed lines, before "
                                "netting its own opposite trades. Much larger than Net "
                                "value means the client was churning, not positioning."),
        "Avg price": _hnc("Avg price (Rs)", format="%.2f",
                          help="Quantity-weighted price of that client's lines."),
        "% of day delivery": _hnc("% of day delivery", format="%.0f%%",
                                  help="Net quantity against all shares delivered that "
                                       "day. Meaningless for BLOCK rows."),
        "Since buy %": _hnc("Since deal", format="%+.1f%%",
                            help="Move from that client's deal price to the latest "
                                 "close, split-adjusted."),
    }


def _evidence_cfg() -> dict:
    return {
        "Event": _htc("Event", width="large"),
        "Events": _hnc("Events", format="%d"),
        "1 month %": _hnc("1 month vs peers", format="%+.2f%%"),
        "t (1m)": _hnc("t (1m)", format="%+.2f",
                       help="Newey-West t on the daily mean. |t| under ~2 is noise."),
        "2 months %": _hnc("2 months vs peers", format="%+.2f%%"),
        "t (2m)": _hnc("t (2m)", format="%+.2f"),
        "Beat peers %": _hnc("Beat peers", format="%.1f%%",
                             help="Share of events that outperformed their matched peers "
                                  "over 21 sessions. 50% would be a coin flip."),
    }
