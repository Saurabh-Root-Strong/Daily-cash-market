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

from src.analytics.commodity_index import (COMMODITIES, INDEX_CHOICES, STUDY,
                                           pattern_label)

_KINDS = {
    "Up N sessions in a row": "up_streak",
    "Down N sessions in a row": "down_streak",
    "Jumps in one session": "spike_up",
    "Falls in one session": "spike_down",
    "Rises over K sessions": "rise_k",
    "Falls over K sessions": "fall_k",
}


def _link_word(r: float | None) -> str:
    if r is None:
        return "not enough weeks"
    if r <= -0.3:
        return "Opposite"
    if r >= 0.3:
        return "Together"
    return "Unrelated"


def _pct(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v * 100:+.{d}f}%"


def _rail(commodity: str, lab: str) -> None:
    g = STUDY["grid"][commodity]
    cr = STUDY["crude"]
    by = cr["same_week_corr_by_year"]
    if commodity == "CRUDE OIL":
        st.warning(
            f"**Measured, not assumed: crude oil does NOT tell you where Nifty goes "
            f"next.** {STUDY['window']}, {g['tests']:,} tests (streaks of 2-5 up/down "
            f"days, sharp one-day jumps, big 5/10/20-session moves × 14 indices × "
            f"next day / next 5 / 10 / 20 sessions). {g['nominal']} looked "
            f"'significant' — about the {g['chance']} that pure chance produces — and "
            f"**none survive** a correction for testing that many things. Example: "
            f"after crude rose 5+ sessions in a row, Nifty's next day averaged "
            f"{cr['n50_up5_next_day']:+.3f}% against {cr['n50_base_next_day']:+.3f}% "
            f"on an ordinary day — a gap well inside luck.\n\n"
            f"**What IS true: crude and Nifty often move at the same time — but which "
            f"way depends on the year** (see *Year by year* below: it runs from about "
            f"{max(by.values()):+.1f} to {min(by.values()):+.1f}, where −1 = always "
            f"opposite and +1 = always together). Over the whole period it is about "
            f"zero. **2026 is the strongest opposite year on record**, so what you "
            f"see on the chart this year is real. But it describes the same week, and "
            f"it does not predict "
            f"itself: the last half-year's direction matched the next half-year's "
            f"only {cr['regime_sign_agree']:.0%} of the time ({cr['regime_halves']} "
            f"half-years). Even inside an 'opposite' spell, a crude-up week did not "
            f"lead a weak Nifty NEXT week (t {cr['neg_regime_next_week_t']:+.2f}). "
            f"By the time crude's move is known, Nifty has usually already reacted "
            f"at the open.")
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
                                             cached_commodity_yearly)
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

    # ── today's read ─────────────────────────────────────────────────────────
    st.markdown(f"#### {lab} right now — MCX close {s.mcx_date:%d %b %Y}")
    m = st.columns(5)
    m[0].metric("Price (₹)", f"{s.close:,.0f}", _pct(s.r1),
                help=f"Last MCX close on or before {selected_date:%d %b}. The small "
                     "number is the one-session change. MCX trades until 23:30, "
                     "after NSE has closed, so this move is news Nifty has NOT seen "
                     "yet — it shows up in Nifty's NEXT open.")
    m[1].metric("Last 5 sessions", _pct(s.r5, 1),
                None if s.pct5 is None else f"bigger than {s.pct5:.0%} of weeks",
                delta_color="off",
                help="Move over the last 5 MCX sessions, and how it ranks against "
                     "every 5-session move of the previous 3 years. Example: "
                     "'bigger than 97% of weeks' = one of the largest weekly rises "
                     "in 3 years.")
    m[2].metric("Last 20 sessions", _pct(s.r20, 1),
                None if s.pct20 is None else f"bigger than {s.pct20:.0%} of months",
                delta_color="off",
                help="Same idea over about a month (20 sessions).")
    streak = (f"up {s.up_streak} in a row" if s.up_streak else
              f"down {s.dn_streak} in a row" if s.dn_streak else "flat")
    m[3].metric("Streak", streak,
                help="How many MCX sessions in a row the price has closed higher "
                     "(or lower). Example: 'up 3 in a row' = three higher closes "
                     "back to back.")
    m[4].metric(f"Moving vs {index}", _link_word(s.corr_26w),
                None if s.corr_26w is None else f"{s.corr_26w:+.2f} over {s.weeks_26w} weeks",
                delta_color="off",
                help="How the two moved in the SAME week over the last 26 weeks "
                     "(rank correlation, −1 to +1). −0.6 = in most weeks when the "
                     "commodity rose, the index fell. It describes the recent past; it is NOT a "
                     "forecast — the study found this relationship flips from year "
                     "to year and does not predict itself.")
    if s.lag_note:
        st.caption(s.lag_note)
    if s.corr_full is not None:
        st.caption(f"Since July 2018 the same-week link is {s.corr_full:+.2f} "
                   f"({_link_word(s.corr_full).lower()}). The last 26 weeks are "
                   f"{s.corr_26w:+.2f}.")

    paths = cached_commodity_paths(selected_date, commodity, index, 250)
    if not paths.empty:
        st.markdown(f"**Last 12 months, both rebased to 100** — above 100 = higher "
                    f"than a year ago")
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
                    f"July 2018.")
        cols = st.columns(3)
        for col, key, name in ((cols[0], "next1", "Next day"),
                               (cols[1], "next5", "Next 5 sessions"),
                               (cols[2], "next20", "Next 20 sessions")):
            x = sm[key]
            if not x["n"]:
                continue
            diff = x["avg"] - x["base_avg"]
            luck = "inside the luck band" if abs(diff) <= x["noise"] else "OUTSIDE the luck band"
            col.metric(f"{index} — {name}", _pct(x["avg"]),
                       f"{diff * 100:+.2f}% vs a normal day",
                       delta_color="off",
                       help=f"Average {index} move after the pattern, close to close. "
                            f"Ordinary days: {_pct(x['base_avg'])}. The luck band is "
                            f"±{x['noise'] * 100:.2f}% — the gap {x['n']} random "
                            f"ordinary days would often show by chance. Inside it = "
                            f"you cannot tell this apart from any other day.")
            col.caption(f"Up {x['up']:.0%} of {x['n']} times (normal: "
                        f"{x['base_up']:.0%}) · {luck}")

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
            "- Scripts: `scripts/crude_vs_index_study.py`, "
            "`crude_vs_index_regime.py`, `crude_vs_index_survivors.py`.")
