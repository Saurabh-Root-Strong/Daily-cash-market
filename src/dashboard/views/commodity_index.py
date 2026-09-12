"""Commodity vs Index panel (Sector Rotation page).

Question it answers, in the user's words: "when crude oil rises for several days,
or jumps sharply, how does Nifty do the next day and the next week?"

The page DESCRIBES. The study behind it (scripts/crude_vs_index_study.py and
friends) found no forward edge for crude, so there is no arrow here -- only every
past occurrence, what the index did after it, and what the index does on an
ordinary day, side by side.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.analytics.commodity_index import (COMMODITIES, INDEX_CHOICES,
                                           MATRIX_MIN_MEDIAN_CR, STUDY,
                                           pattern_label)

_KINDS = {
    "Up N sessions in a row": "up_streak",
    "Down N sessions in a row": "down_streak",
    "Jumps in one session": "spike_up",
    "Falls in one session": "spike_down",
    "Rises over K sessions": "rise_k",
    "Falls over K sessions": "fall_k",
}


# |rank correlation| a sample of 26 independent weeks exceeds by chance only
# about 1 time in 20. Below it, "Opposite" would be a coin reading tea leaves.
_LINK_BAND = 0.4


def _link_word(r: float | None) -> str:
    if r is None:
        return "not enough weeks"
    if r <= -_LINK_BAND:
        return "Opposite"
    if r >= _LINK_BAND:
        return "Together"
    return "Unrelated"


def _rank_word(p: float | None, unit: str) -> str | None:
    """'bigger than 97% of weeks' for a rise, 'a fall deeper than 98%' for a fall."""
    if p is None:
        return None
    if p >= 0.5:
        return f"rise bigger than {p:.0%} of {unit}"
    return f"fall deeper than {1 - p:.0%} of {unit}"


def _pct(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _rail(commodity: str, lab: str) -> None:
    g = STUDY["grid"].get(commodity)          # None for the four matrix-only ones
    cr = STUDY["crude"]
    by = cr["same_week_corr_by_year"]
    if commodity == "CRUDE OIL":
        st.warning(
            f"**Measured, not assumed: crude oil does NOT tell you where Nifty goes "
            f"next.** {STUDY['window']}, {g['tests']:,} tests (streaks of 2-5 up/down "
            f"days, sharp one-day jumps, big 5/10/20-session moves × 14 indices × "
            f"next day / next 5 / 10 / 20 sessions). {g['nominal']} looked "
            f"'significant' — about the {g['chance']} that pure chance produces — and "
            f"**none survive** a correction for testing that many things.\n\n"
            f"**What IS true: crude and Nifty often move at the same time — but which "
            f"way depends on the year** (see *Year by year* below: it runs from about "
            f"{max(by.values()):+.1f} to {min(by.values()):+.1f}, where −1 = always "
            f"opposite and +1 = always together). Over the whole period it is about "
            f"zero. **2026 is the strongest opposite year on record** (as of the "
            f"Sep 2026 study), so what you see on the chart this year is real. But it "
            f"describes the same week, and it has not predicted itself: the last 26 "
            f"weeks' direction matched the next 13 weeks' {cr['regime_sign_agree']:.0%} "
            f"of the time ({cr['regime_steps']} tries; of the strong readings only "
            f"{cr['regime_strong_held']} held). Inside an 'opposite' spell a crude-up "
            f"week was followed by a weaker Nifty week, but not measurably (t "
            f"{cr['neg_regime_next_week_t']:+.2f} on 26 weeks — too few to call). In "
            f"2026 much of the reaction came at the next open (crude's day vs Nifty's "
            f"next gap {cr['gap_corr_2026']:+.2f}), before a trade off the MCX close "
            f"was possible.\n\n"
            f"**The rupee matters more than the oil, most years.** MCX crude = "
            f"dollar crude × USD/INR. Split apart over 2018-2026, the rupee's "
            f"same-week link to Nifty is {cr['usdinr_link_full']:+.2f} (a weaker rupee "
            f"goes with a weaker Nifty in every period) while dollar crude's is "
            f"{cr['usd_crude_link_full']:+.2f}. 2026 is the exception — dollar crude "
            f"alone is {cr['usd_crude_link_2026']:+.2f} even with the rupee held "
            f"fixed. Neither forecasts next week.")
    elif g is None:
        m = STUDY["matrix"]
        st.warning(
            f"**{lab} was tested inside the commodity × sector map, not on its own "
            f"grid.** {m['window']}, {m['rc_tests']:,} tests across nine commodities "
            f"and sixteen indices. The only readings that beat the same search run "
            f"on shuffled data are base-metal crashes landing in Nifty Metal's "
            f"OPENING GAP — a reaction, not a trade. Nothing here forecasts a "
            f"sector; the map below is an exposure fact, not a signal.")
    else:
        note = STUDY["survivor_note"].get(commodity)
        head = (f"**{lab}: {g['tests']:,} tests, {g['nominal']} nominal hits vs "
                f"{g['chance']} expected by chance, {g['fdr']} survive a correction "
                f"across all {STUDY['cumulative_tests']:,} tests run on five "
                f"commodities ({g['tradable']} of them tradable).**")
        st.warning(head + ("\n\n" + note if note else
                           "\n\nNothing here forecasts the index. The tables below "
                           "describe what happened; they are not a signal."))


def render_commodity_index(selected_date: date) -> None:
    from src.dashboard.cache.queries import (cached_commodity_episodes,
                                             cached_commodity_index_link,
                                             cached_commodity_paths,
                                             cached_commodity_state,
                                             cached_commodity_yearly,
                                             cached_month_ahead,
                                             cached_sector_matrix)
    st.markdown(
        "Does a move in **crude oil** (or gold, silver, natural gas, copper) move "
        "the Indian market? This panel lines up **MCX (Multi Commodity Exchange)** "
        "futures against Nifty indices from July 2018, and shows every past time a "
        "pattern happened and what the index did after it.")

    c1, c2 = st.columns(2)
    commodity = c1.selectbox(
        "Commodity", list(COMMODITIES), format_func=COMMODITIES.get,
        key="ci_commodity",
        help="MCX front-month futures in rupees — the price India actually pays, "
             "so a crude move here includes the rupee's move. Example: crude at "
             "$70 with USD/INR at 88 is about ₹6,160 a barrel.")
    index = c2.selectbox(
        "Index", INDEX_CHOICES, key="ci_index",
        help="The Nifty index to compare against. Nifty 50 is the market; the "
             "sector indices show who is hit hardest (e.g. Oil & Gas companies "
             "often RISE with crude, paint and airline stocks usually fall).")
    lab = COMMODITIES[commodity]

    try:
        s = cached_commodity_state(selected_date, commodity, index)
    except Exception as exc:                                   # noqa: BLE001
        st.error(f"Commodity vs Index unavailable: {exc}")
        return
    if not s.data_ok:
        st.info(s.note)
        return

    _rail(commodity, lab)
    if s.liquid_note:
        st.error(s.liquid_note)

    # ── today's read ─────────────────────────────────────────────────────────
    st.markdown(f"#### {lab} right now — MCX close {s.mcx_date:%d %b %Y}")
    m = st.columns(5)
    m[0].metric("Price (₹)", f"{s.close:,.0f}", _pct(s.r1), delta_color="off",
                help=f"Last MCX close on or before {selected_date:%d %b}. The small "
                     "number is the one-session change. MCX trades until 23:30, "
                     "after NSE has closed, so this move is news Nifty has NOT seen "
                     "yet — it shows up in Nifty's NEXT open.")
    m[1].metric("Last 5 sessions", _pct(s.r5, 1), _rank_word(s.pct5, "weeks"),
                delta_color="off", delta_arrow="off",
                help="Move over the last 5 MCX sessions, and how it ranks against "
                     "every 5-session move of the previous 3 years. Example: "
                     "'rise bigger than 97% of weeks' = one of the largest weekly "
                     "rises in 3 years; 'fall deeper than 90%' = a big weekly fall.")
    m[2].metric("Last 20 sessions", _pct(s.r20, 1), _rank_word(s.pct20, "months"),
                delta_color="off", delta_arrow="off",
                help="Same idea over about a month (20 sessions).")
    streak = (f"up {s.up_streak} in a row" if s.up_streak else
              f"down {s.dn_streak} in a row" if s.dn_streak else "flat")
    m[3].metric("Streak", streak,
                help="How many MCX sessions in a row the price has closed higher "
                     "(or lower). Example: 'up 3 in a row' = three higher closes "
                     "back to back.")
    m[4].metric(f"Moving vs {index}", _link_word(s.corr_26w),
                None if s.corr_26w is None else f"{s.corr_26w:+.2f} over {s.weeks_26w} weeks",
                delta_color="off", delta_arrow="off",
                help="How the two moved in the SAME week over the last 26 weeks "
                     f"(rank correlation, −1 to +1). −0.6 = in most weeks when the "
                     f"commodity rose, the index fell. 'Opposite' / 'Together' only "
                     f"beyond ±{_LINK_BAND} — with 26 weeks, smaller readings happen "
                     f"by chance. It describes the recent past; it is NOT a "
                     "forecast — the study found this relationship flips from year "
                     "to year and does not predict itself.")
    if s.stale:
        st.error(s.lag_note)
    elif s.lag_note:
        st.caption(s.lag_note)
    if s.corr_full is not None:
        st.caption(f"Since July 2018 the same-week link is {s.corr_full:+.2f} "
                   f"({_link_word(s.corr_full).lower()}). The last 26 weeks are "
                   f"{s.corr_26w:+.2f}.")

    paths = cached_commodity_paths(selected_date, commodity, index, 250)
    if not paths.empty:
        st.markdown(f"**Last 12 months, both rebased to 100** — above 100 = higher "
                    f"than a year ago")
        st.caption("Commodity line = the quoted MCX front-month price, so it takes a "
                   "small step at each monthly contract roll.")
        st.line_chart(paths, height=260, color=["#60a5fa", "#f59e0b"])

    # ── pattern lookup ───────────────────────────────────────────────────────
    st.markdown(f"#### When {lab.lower()} did this before, what did {index} do next?")
    p1, p2, p3 = st.columns([2, 1, 1])
    kind_lbl = p1.radio("Pattern", list(_KINDS), horizontal=True, key="ci_kind",
                        help="Pick what the commodity did. Each past occurrence is "
                             "counted ONCE, from its first day — a 7-day rally is "
                             "one event, not five.")
    kind = _KINDS[kind_lbl]
    n, thr, k = 5, 3.0, 5
    if kind in ("up_streak", "down_streak"):
        n = int(p2.number_input("N sessions", 2, 9, 5, key="ci_n",
                                help="Example: 5 = five higher closes back to back."))
    elif kind in ("spike_up", "spike_down"):
        thr = float(p2.number_input("Move bigger than (%)", 1.0, 15.0, 3.0, 0.5,
                                    key="ci_thr1",
                                    help="Example: 3 = a single session up (or "
                                         "down) more than 3%."))
    else:
        k = int(p2.selectbox("K sessions", [5, 10, 20], key="ci_k",
                             help="The window. 5 ≈ a week, 20 ≈ a month."))
        thr = float(p3.number_input("Move bigger than (%)", 1.0, 40.0,
                                    {5: 5.0, 10: 8.0, 20: 10.0}[k], 0.5,
                                    key=f"ci_thrk{k}",
                                    help="Example: 10 over 20 sessions = up more "
                                         "than 10% in about a month."))

    ev, sm = cached_commodity_episodes(selected_date, commodity, index, kind, n, thr, k)
    if ev.empty:
        st.info(f"'{pattern_label(commodity, kind, n, thr, k)}' never happened "
                f"since July 2018. Try a smaller number.")
    else:
        st.markdown(f"**{sm['label']}** — happened **{sm['episodes']} times** since "
                    f"July 2018 (a new time only after at least "
                    f"{max(sm['window'], 5)} sessions without it).")
        cols = st.columns(3)
        for col, key, name in ((cols[0], "next1", "Next day"),
                               (cols[1], "next5", "Next 5 sessions"),
                               (cols[2], "next20", "Next 20 sessions")):
            x = sm[key]
            if not x["n"]:
                continue
            diff = x["avg"] - x["base_avg"]
            if x["noise"] is None:
                luck, band = "too few to judge", "not computed — fewer than 3 independent cases"
            else:
                luck = ("inside the luck band" if abs(diff) <= x["noise"]
                        else "outside the luck band")
                band = f"±{x['noise'] * 100:.2f}%"
            col.metric(f"{index} — {name}", _pct(x["avg"]),
                       f"{diff * 100:+.2f}% vs a normal day",
                       delta_color="off", delta_arrow="off",
                       help=f"Average {index} move after the pattern, close to close. "
                            f"Ordinary days: {_pct(x['base_avg'])}. Luck band: {band} "
                            f"— how far the average of {x['n_eff']} random ordinary "
                            f"days lands from normal about 1 time in 20 "
                            f"({x['n_eff']} = the cases whose windows do not overlap). "
                            f"Outside it on ONE pattern means little: try 20 patterns "
                            f"and about one lands outside by chance — the study tried "
                            f"{STUDY['cumulative_tests']:,}.")
            col.caption(f"Up {x['up']:.0%} of {x['n']} times (normal: "
                        f"{x['base_up']:.0%}) · {luck}")
        if sm.get("today_in"):
            st.info(f"**{lab} is in this pattern right now** (MCX close "
                    f"{sm['today_mcx']:%d %b %Y}). The table is what followed it before "
                    f"— a record, not a forecast.")

        tab = pd.DataFrame({
            f"{lab} date": pd.to_datetime(ev.mcx_date).dt.date,
            f"{index} day": pd.to_datetime(ev.session).dt.date,
            f"{lab} ₹": ev.cmd_close,
            f"{lab} move %": ev.cmd_move * 100,
            f"{index} during %": ev.idx_same * 100,
            "Next day %": ev.next1 * 100,
            "Next day pts": ev.pts1,
            "Next 5 %": ev.next5 * 100,
            "Next 5 pts": ev.pts5,
            "Next 20 %": ev.next20 * 100,
            "Next 20 pts": ev.pts20,
        })
        win = sm["window"]
        # Styler, not column formats: a window that has not finished as of the
        # selected date must read "—", and NumberColumn prints NaN as "None".
        fmt = {f"{lab} ₹": "{:,.0f}", f"{lab} move %": "{:+.1f}%",
               f"{index} during %": "{:+.2f}%"}
        fmt.update({"Next day %": "{:+.2f}%", "Next day pts": "{:+,.0f}"})
        # Streamlit renders a missing number as "None" whatever na_rep says, so
        # the only columns that can be unfinished are pre-formatted as text.
        for c_, f_ in (("Next 5 %", "{:+.2f}%"), ("Next 20 %", "{:+.2f}%"),
                       ("Next 5 pts", "{:+,.0f}"), ("Next 20 pts", "{:+,.0f}")):
            tab[c_] = [("—" if pd.isna(v) else f_.format(v)) for v in tab[c_]]
        st.dataframe(
            tab.style.format(fmt, na_rep="—"), hide_index=True, width="stretch",
            height=380,
            column_config={
                f"{lab} date": st.column_config.DateColumn(
                    f"{lab} date", format="DD MMM YYYY",
                    help="The MCX session on which the pattern first completed (its "
                         "close is at 23:30). Example: for 'up 5 in a row' this is "
                         "the date of the 5th higher close."),
                f"{index} day": st.column_config.DateColumn(
                    f"{index} day", format="DD MMM YYYY",
                    help="The FIRST NSE session after that MCX close — the first "
                         "day Nifty could react. Usually the next calendar day."),
                f"{lab} ₹": st.column_config.Column(
                    f"{lab} ₹",
                    help="MCX front-month closing price that day, in rupees."),
                f"{lab} move %": st.column_config.Column(
                    f"{lab} move %",
                    help=f"How much the commodity moved over the pattern "
                         f"({win} session{'s' if win > 1 else ''}). Example: +8.9% "
                         f"= up 8.9% across the streak."),
                f"{index} during %": st.column_config.Column(
                    f"{index} during %",
                    help=f"What {index} did over the same {win} NSE "
                         f"session{'s' if win > 1 else ''}, up to the close before "
                         f"its reaction day. Negative while crude rose = the "
                         f"'opposite' pattern you see on charts."),
                "Next day %": st.column_config.Column(
                    "Next day %",
                    help=f"{index} close on its reaction day vs the close before. "
                         f"Example: −0.59% = it fell 0.59% the next day. Includes "
                         f"the opening gap, which you could not have traded off "
                         f"the MCX close."),
                "Next day pts": st.column_config.Column(
                    "Next day pts",
                    help="The same move in index points. Example: −141 = Nifty "
                         "closed 141 points lower."),
                "Next 5 %": st.column_config.Column(
                    "Next 5 %",
                    help="Close 5 NSE sessions later (≈ one week) vs the close "
                         "before the reaction day. — = that week has not "
                         "finished yet as of the selected date."),
                "Next 5 pts": st.column_config.Column(
                    "Next 5 pts",
                    help="The same one-week move in points."),
                "Next 20 %": st.column_config.Column(
                    "Next 20 %",
                    help="Close 20 NSE sessions later (≈ one month). — = not "
                         "finished yet."),
                "Next 20 pts": st.column_config.Column(
                    "Next 20 pts",
                    help="The same one-month move in points."),
            })

    # ── by year ──────────────────────────────────────────────────────────────
    st.markdown(f"#### Year by year: weeks {lab.lower()} rose more than 3%")
    yr = cached_commodity_yearly(selected_date, commodity, index, 3.0)
    if not yr.empty:
        yt = pd.DataFrame({
            "Year": yr.year.astype(str),
            "Weeks": yr.weeks,
            f"{lab} up >3% weeks": yr.up_weeks,
            f"{index} that week %": yr.same_avg * 100,
            f"{index} fell that week": yr.same_down * 100,
            "Fell in any week": yr.all_down * 100,
            "Next week %": yr.next_avg * 100,
            "Same-week link": yr["corr"],
        })
        st.dataframe(
            yt, hide_index=True, width="stretch",
            column_config={
                "Year": st.column_config.TextColumn(
                    "Year", help="Calendar year. The current year is partial."),
                "Weeks": st.column_config.NumberColumn(
                    "Weeks", help="Non-overlapping 5-session weeks in the year "
                                  "(each week counted once)."),
                f"{lab} up >3% weeks": st.column_config.NumberColumn(
                    f"{lab} up >3% weeks",
                    help="How many of those weeks the commodity rose more than 3%. "
                         "Example: 18 out of 35 = a big up-week about one week in "
                         "two."),
                f"{index} that week %": st.column_config.NumberColumn(
                    f"{index} that week %", format="%+.2f%%",
                    help="Average index move in those same weeks. Example: −1.0% "
                         "= in the commodity's big up-weeks the index fell 1% on "
                         "average."),
                f"{index} fell that week": st.column_config.NumberColumn(
                    f"{index} fell that week", format="%.0f%%",
                    help="Share of those weeks in which the index fell. Compare it "
                         "with the next column to see if it is worse than usual."),
                "Fell in any week": st.column_config.NumberColumn(
                    "Fell in any week", format="%.0f%%",
                    help="Share of ALL weeks that year in which the index fell — "
                         "the normal rate. Example: 78% in the column before vs 66% "
                         "here = the commodity's up-weeks were worse than a typical "
                         "week for the index."),
                "Next week %": st.column_config.NumberColumn(
                    "Next week %", format="%+.2f%%",
                    help="Average index move in the week AFTER such an up-week — "
                         "the forecast question. Near zero in almost every year."),
                "Same-week link": st.column_config.NumberColumn(
                    "Same-week link", format="%+.2f",
                    help="Rank correlation of the weekly moves that year, −1 to +1. "
                         "−0.6 = strongly opposite; +0.4 = moved together; around 0 "
                         "= unrelated. It changes sign across years."),
            })

    # ── which index is most exposed ──────────────────────────────────────────
    st.markdown(f"#### Which index moves most with {lab.lower()}?")
    li = cached_commodity_index_link(selected_date, commodity)
    if not li.empty:
        lt = pd.DataFrame({"Index": li["index"], "Since 2018": li.corr_full,
                           "Last 52 weeks": li.corr_52w, "Last 26 weeks": li.corr_26w})
        lt = lt.sort_values("Last 26 weeks")
        _h = ("Same-week rank correlation with {lab}, −1 to +1. Negative = the "
              "index tends to fall in weeks {lab} rises. Example: −0.70 = in most "
              "weeks {lab} rose, this index fell; +0.40 = they mostly rose "
              "together.")
        st.dataframe(
            lt, hide_index=True, width="stretch",
            column_config={
                "Index": st.column_config.TextColumn(
                    "Index", help="Nifty index. Sorted most-opposite first over the "
                                  "last 26 weeks."),
                "Since 2018": st.column_config.NumberColumn(
                    "Since 2018", format="%+.2f",
                    help=_h.format(lab=lab.lower()) + " Whole period."),
                "Last 52 weeks": st.column_config.NumberColumn(
                    "Last 52 weeks", format="%+.2f",
                    help=_h.format(lab=lab.lower()) + " Last year only."),
                "Last 26 weeks": st.column_config.NumberColumn(
                    "Last 26 weeks", format="%+.2f",
                    help=_h.format(lab=lab.lower()) + " Last six months only."),
            })

    # ── this month and next ─────────────────────────────────────────────────
    st.divider()
    mo = STUDY["monthly"]
    st.markdown(f"#### This month and next — {lab.lower()}")
    ma = cached_month_ahead(selected_date, commodity)
    if not ma:
        st.info("Not enough monthly history for this commodity.")
    else:
        mv, rk = ma["month_move"], ma["month_rank"]
        word = ("one of its biggest up-months" if rk >= 0.9 else
                "one of its biggest down-months" if rk <= 0.1 else
                "an ordinary month")
        st.markdown(
            f"**{lab} is {mv * 100:+.1f}% over the last 21 sessions** "
            f"(bigger than {rk:.0%} of months since 2018 — {word}, group "
            f"{ma['quintile']} of 5).")
        st.warning(
            f"**The forecast column below is empty on purpose.** Ranking sectors by "
            f"their commodity exposure was tested properly — sensitivities fitted "
            f"only on earlier months, sectors scored from that month's commodity "
            f"moves, then compared with what they did NEXT month over "
            f"{mo['wf_months']} months. The ranking scored {mo['wf_ic']:+.3f} "
            f"(t {mo['wf_t']:+.2f}), i.e. slightly **wrong**, right only "
            f"{mo['wf_pos_share']:.0%} of months; the top-minus-bottom basket lost "
            f"{abs(mo['wf_spread']):.2f}% a month. Single commodity→sector pairs: "
            f"{mo['pair_tests']} tests, {mo['pair_nominal']} nominal vs "
            f"{mo['pair_chance']} by chance, {mo['pair_fdr']} survive, and the best "
            f"one (t {mo['pair_rc_best']:.2f}) is below the median of a shuffled "
            f"search (t {mo['pair_rc_null_median']:.2f}), p {mo['pair_rc_p']:.2f}. "
            f"With {mo['months']} months, only an IC above {mo['ic_needed']:.2f} "
            f"could have been detected at all — so this is 'no usable signal', "
            f"measured, not 'no signal exists'.\n\n"
            f"**What the table IS for:** the same-month columns are the exposure you "
            f"already carry. If you hold metal stocks while copper moves, you are "
            f"long copper whether you meant to be or not.")

        t = ma["table"].head(8).copy()
        band = ma["band"]
        mtab = pd.DataFrame({
            "Sector": t.sector,
            "Same-month link": t.same_month_ic,
            # beta is already "sector % per commodity %" -- scaling it by 100
            # printed +13.07% where the truth is +0.13% per 1% move
            "Per 1% move": t.beta,
            "Goes with this month": t.implied_same_month * 100,
            "Actually did": t.actual_same_month * 100,
            "Next month (similar months)": t.next_month_mean * 100,
            "Normal next month": t.next_month_base * 100,
            "Luck band ±": t.next_month_band * 100,
        })
        st.dataframe(
            mtab.style.format({
                "Same-month link": "{:+.2f}", "Per 1% move": "{:+.2f}%",
                "Goes with this month": "{:+.1f}%", "Actually did": "{:+.1f}%",
                "Next month (similar months)": "{:+.1f}%",
                "Normal next month": "{:+.1f}%", "Luck band ±": "{:.1f}%"},
                na_rep="—"),
            hide_index=True, width="stretch",
            column_config={
                "Sector": st.column_config.Column(
                    "Sector", help="Nifty index, measured against Nifty 50 so a "
                                   "good month for the whole market does not count."),
                "Same-month link": st.column_config.Column(
                    "Same-month link",
                    help=f"Rank correlation of monthly moves, −1 to +1, over "
                         f"{ma['months']} months. Beyond ±{band:.2f} is more than "
                         f"chance. Example: +0.51 for copper and Nifty Metal = they "
                         f"rose and fell together in most months."),
                "Per 1% move": st.column_config.Column(
                    "Per 1% move",
                    help="How much the sector moved against the market for each 1% "
                         "the commodity moved, in the SAME month. Example: +0.56 "
                         "means copper +10% went with Nifty Metal about +5.6% "
                         "better than the market."),
                "Goes with this month": st.column_config.Column(
                    "Goes with this month",
                    help="The previous column applied to this month's actual "
                         "commodity move. It is the co-move that usually "
                         "accompanies a move this size — NOT a forecast of "
                         "anything, and the month it describes has already "
                         "happened."),
                "Actually did": st.column_config.Column(
                    "Actually did",
                    help="What the sector really did over the same 21 sessions, "
                         "against the market. Compare it with the column before: a "
                         "big gap means the sector moved for reasons other than "
                         "this commodity."),
                "Next month (similar months)": st.column_config.Column(
                    "Next month (similar months)",
                    help="Average of what this sector did over the FOLLOWING month, "
                         "in past months when the commodity sat in the same group "
                         "of 5. A base rate from a handful of months, not a "
                         "forecast — compare it with the two columns after it."),
                "Normal next month": st.column_config.Column(
                    "Normal next month",
                    help="What the sector does against the market in an ordinary "
                         "month. If the previous column is not clearly away from "
                         "this one, nothing is being said."),
                "Luck band ±": st.column_config.Column(
                    "Luck band ±",
                    help="Two standard errors. A difference smaller than this is "
                         "what a handful of random months produce by chance — and "
                         "almost every row here is inside it."),
            })
        inside = int((((t.next_month_mean - t.next_month_base).abs())
                      <= t.next_month_band).sum())
        st.caption(
            f"{inside} of {len(t)} sectors sit inside their luck band for next "
            f"month · base rates use {int(t.next_month_n.max())} past months in "
            f"this group · the same-month columns use {ma['months']} months")

    # ── commodity x sector map ───────────────────────────────────────────────
    st.divider()
    m = STUDY["matrix"]
    st.markdown("#### Which sector moves with which commodity")
    st.markdown(
        f"Every commodity against every sector, {m['window']}, in "
        f"non-overlapping weeks. This is **exposure, not prediction**: it says "
        f"the two moved together in the same week, which is what you need for "
        f"position risk — not what you need to buy the sector tomorrow.")
    st.warning(
        f"**The forecast version is empty, and that is measured.** "
        f"{m['forward_tests']:,} forward tests (commodity move → sector next day / "
        f"week / month): {m['forward_nominal']} looked significant against "
        f"{m['forward_chance']} expected by chance, and when the same search is run "
        f"on shuffled commodities ({m['rc_tests']:,} tests, 500 shuffles) the ONLY "
        f"readings that beat it are **base-metal crashes hitting Nifty Metal's "
        f"opening gap** — which happens overnight, before you can act. Every "
        f"tradable candidate falls inside the noise of the search: natural gas → "
        f"Energy next day scores "
        f"{m['rc_candidates']['NATURALGAS 10d top 10% -> Nifty Energy next day']:.2f} "
        f"against a {m['rc_null_95']:.2f} bar, and lead → PSE over 20 days drops from "
        f"t {m['lead_hac_t']:.2f} to {m['lead_overlap_adjusted']:.2f} once "
        f"overlapping windows are counted honestly.")

    mode = st.radio(
        "Show", ["Sector on its own (market removed)", "Raw sector move"],
        horizontal=True, key="ci_matrix_mode",
        help="'Sector on its own' subtracts Nifty 50 from each sector first, so a "
             "week when everything rose does not make every sector look like a "
             "commodity play. Example: copper vs Nifty Metal is +0.39 raw and "
             "+0.43 once the market is removed — it is genuinely about metal "
             "stocks. Most other pairs shrink towards zero instead.")
    try:
        mx = cached_sector_matrix(selected_date, 6.0,
                                  mode.startswith("Sector on its own"))
    except Exception as exc:                                   # noqa: BLE001
        st.error(f"Matrix unavailable: {exc}")
        mx = {}
    if mx:
        rho, band = mx["rho"], mx["band"]

        def _tint(v):
            if pd.isna(v):
                return ""
            if v >= band * 2:
                return "color:#16a34a;font-weight:700"
            if v >= band:
                return "color:#16a34a"
            if v <= -band * 2:
                return "color:#dc2626;font-weight:700"
            if v <= -band:
                return "color:#dc2626"
            return "color:#9ca3af"

        show = rho.copy()
        show.index.name = "Sector"
        cfg = {c: st.column_config.NumberColumn(
            COMMODITIES.get(c, c), format="%+.2f",
            help=f"How {COMMODITIES.get(c, c).lower()} and this sector moved in the "
                 f"SAME week, −1 to +1, over {mx['weeks']} weeks. "
                 f"Beyond ±{band:.2f} is more than chance; grey is nothing. "
                 f"Example: +0.43 = in most weeks they rose and fell together.")
            for c in rho.columns}
        st.dataframe(show.reset_index().style.format(
            {c: "{:+.2f}" for c in rho.columns}, na_rep="—").map(
                _tint, subset=list(rho.columns)),
            hide_index=True, width="stretch",
            column_config={"Sector": st.column_config.TextColumn(
                "Sector", help="Nifty index. Bank, IT and Pharma are controls — "
                               "they should show nothing, and mostly do."), **cfg})
        st.caption(
            f"{mx['weeks']} independent weeks · anything between −{band:.2f} and "
            f"+{band:.2f} is indistinguishable from chance · "
            + ("sector minus Nifty 50" if mx["sector_only"] else "raw sector move"))

        b = mx["beta"]
        pairs = (rho.stack().rename("link").reset_index()
                 .rename(columns={"level_0": "Sector", "level_1": "cm"}))
        pairs["Move for a 1% commodity move"] = [
            b.loc[r.Sector, r.cm] for r in pairs.itertuples()]
        pairs["Commodity"] = pairs.cm.map(lambda c: COMMODITIES.get(c, c))
        pairs = pairs.reindex(pairs.link.abs().sort_values(ascending=False).index)
        st.markdown("**Strongest links**")
        st.dataframe(
            pairs.head(10)[["Commodity", "Sector", "link",
                            "Move for a 1% commodity move"]],
            hide_index=True, width="stretch",
            column_config={
                "Commodity": st.column_config.TextColumn(
                    "Commodity", help="MCX front-month futures, in rupees."),
                "Sector": st.column_config.TextColumn(
                    "Sector", help="The Nifty index it moved with."),
                "link": st.column_config.NumberColumn(
                    "Same-week link", format="%+.2f",
                    help="Rank correlation over the window, −1 to +1."),
                "Move for a 1% commodity move": st.column_config.NumberColumn(
                    "Sector moves", format="%+.2f%%",
                    help="Size, not just direction: how much the sector moved in "
                         "the same week for each 1% the commodity moved. Example: "
                         "+0.49 = copper up 10% went with Nifty Metal up about 5% "
                         "against the market. Use it for position risk — if you "
                         "hold metal stocks you are holding a copper position."),
            })

    with st.expander("How this was tested"):
        st.markdown(
            "- **Timing.** MCX closes at 23:30, NSE at 15:30. Every past event is "
            "matched to the FIRST NSE session after the MCX close, so nothing uses "
            "a price Nifty could not have known.\n"
            "- **Roll-safe prices.** The front-month contract is held until 5 days "
            "before expiry, and returns are measured inside one contract, so a "
            "month-end roll never shows up as a fake jump (same series as the "
            "Commodity_Forex_Market project).\n"
            "- **Events counted once.** A streak is one event from its first day; "
            "counting every day of a long rally would inflate the sample.\n"
            "- **Honest statistics.** Differences are judged against ordinary days "
            "with Newey-West errors, a shuffle test that keeps each series' own "
            "rhythm, a correction for the thousands of combinations tried, split "
            "halves (2018-22 vs 2022-26), and a cut of Mar-May 2020 when MCX crude "
            "went through zero.\n"
            "- **Data holes.** All of June 2018 is missing at the source, so the "
            "study starts 2 Jul 2018. 31 Dec 2024 is also missing for crude.\n"
            "- **Liquidity.** A commodity-year whose median session trades below "
            f"Rs {MATRIX_MIN_MEDIAN_CR:.0f} Cr is dropped from the map: MCX nickel's "
            "median day is about Rs 1 Cr from 2022 and lead's Rs 18 Cr by 2026, so "
            "their closes are quotes, not a market.\n"
            "- **Overlap.** A 20-session return sampled daily is not one "
            "observation per day. Counting it honestly turned the best 20-day "
            "'edge' in the map from t 4.2 into 1.6.\n"
            "- **The search itself is tested.** Every commodity is circular-shifted "
            "500 times and the whole grid rebuilt, so the bar is the best result "
            "the same search finds when nothing can possibly predict anything.\n"
            "- Scripts: `scripts/crude_vs_index_study.py`, "
            "`crude_vs_index_regime.py`, `crude_vs_index_survivors.py`, "
            "`commodity_sector_matrix.py`, `commodity_sector_survivors.py`, "
            "`commodity_sector_reality_check.py`.")
