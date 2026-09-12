"""Commodity vs Index — does a crude (or gold, silver ...) move move Nifty?

One construction, used by BOTH the study (scripts/crude_vs_index_study.py) and
the dashboard panel, so the number on screen is the number that was tested.

THE TIMING TRAP this module exists to get right
-----------------------------------------------
MCX trades until 23:30 IST; NSE closes at 15:30. An MCX close dated T contains
eight hours of news that did not exist when Nifty closed on T. So:

* a commodity state is attached to an NSE session S only if it is the last MCX
  close STRICTLY BEFORE S (`align_to_sessions`, allow_exact_matches=False) --
  i.e. it was known before S opened;
* "next day" for Nifty is session S, measured close(S-1) -> close(S). The part
  of that move that happens at the open (the gap) is not tradable off the MCX
  close -- it happened while you slept -- and the study reports it separately.

Data: `commodity_daily`, a copy of the Commodity_Forex_Market project's roll-safe
front-month MCX series (see src/ingestion/commodity_sync.py). Prices are in
RUPEES -- the import bill India pays -- so a crude move here includes the rupee.

What the study found (frozen in STUDY below, pinned by tests)
-------------------------------------------------------------
No forward edge. The same-time link is real but flips sign by regime, and the
regime does not forecast itself. This module therefore DESCRIBES: it lists every
past occurrence of a pattern and what the index did next, beside what the index
does on an ordinary day. It never prints an arrow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "COMMODITIES", "INDEX_CHOICES", "PATTERNS", "START", "STUDY",
    "load_commodity", "load_index_ohlc", "commodity_features",
    "align_to_sessions", "index_outcomes", "pattern_mask", "episode_starts",
    "liquid_years",
    "CommodityState", "get_commodity_state", "get_pattern_episodes",
    "get_yearly_link", "get_index_link", "get_rebased_paths",
    "get_sector_matrix", "get_month_ahead",
]

# name in commodity_daily -> plain label
COMMODITIES = {
    "CRUDE OIL": "Crude oil",
    "NATURALGAS": "Natural gas",
    "GOLD": "Gold",
    "SILVER": "Silver",
    "COPPER": "Copper",
    "ZINC": "Zinc",
    "ALUMINIUM": "Aluminium",
    "LEAD": "Lead",
    "NICKEL": "Nickel",
}

# Sectors for the commodity x sector map. The last three are controls: if Bank,
# IT or Pharma showed a strong metal link, the method would be measuring the
# market, not the metal.
SECTOR_MATRIX = [
    "Nifty Metal", "Nifty Commodities", "Nifty Oil & Gas", "Nifty Energy",
    "Nifty PSE", "Nifty Auto", "Nifty Consumer Durables", "Nifty FMCG",
    "Nifty Infrastructure", "Nifty Realty", "NIFTY Midcap 100",
    "NIFTY Smallcap 100", "Nifty 50", "Nifty Bank", "Nifty IT", "Nifty Pharma",
]

# A commodity-year whose MEDIAN front-month session is below this is not a
# market: MCX NICKEL's median day is Rs 1 Cr from 2022, LEAD's Rs 18 Cr by 2026.
MATRIX_MIN_MEDIAN_CR = 50.0

INDEX_CHOICES = [
    "Nifty 50", "Nifty Bank", "NIFTY Midcap 100", "NIFTY Smallcap 100",
    "Nifty Auto", "Nifty Oil & Gas", "Nifty Energy", "Nifty FMCG", "Nifty IT",
    "Nifty Metal", "Nifty Pharma", "Nifty Realty", "Nifty Infrastructure",
    "Nifty PSE",
]

# All of June 2018 is missing exchange-wide in the CFM source (MCX served May's
# file under June's URL), so every rolling window is started after the hole.
START = pd.Timestamp("2018-07-02")

# A crude state older than this many calendar days is not carried onto an NSE
# session (a long MCX gap must never masquerade as "yesterday's crude").
_STALE_DAYS = 5

# Calendar days between two MCX rows beyond which the later row's return is not
# a one-session move (longest real MCX closure is ~4 days: a long weekend).
_BREAK_DAYS = 7

# pattern key -> (label template, needs)
PATTERNS = {
    "up_streak":   "{c} up {n}+ sessions in a row",
    "down_streak": "{c} down {n}+ sessions in a row",
    "spike_up":    "{c} jumps more than {thr:g}% in one session",
    "spike_down":  "{c} falls more than {thr:g}% in one session",
    "rise_k":      "{c} up more than {thr:g}% over {k} sessions",
    "fall_k":      "{c} down more than {thr:g}% over {k} sessions",
}

# ── Frozen study results (scripts/crude_vs_index_study.py + _regime.py) ──────
# Run 2026-09-11 on 2018-07-02..2026-09-10. Pinned by tests so the page cannot
# quietly claim something the study did not measure.
STUDY = {
    "window": "Jul 2018 – Sep 2026",
    # every grid, all five commodities together -- the honest multiple-testing count
    "cumulative_tests": 8222,
    # per commodity: tests, nominal |t|>1.96, expected by chance, survivors of
    # BH-FDR q<0.10 across ALL 8,222 tests, and how many of those are tradable
    # (i.e. not the overnight gap, which happens before you can act)
    "grid": {
        "CRUDE OIL":  dict(tests=1680, nominal=101, chance=84, fdr=0, tradable=0),
        "GOLD":       dict(tests=1507, nominal=42,  chance=75, fdr=0, tradable=0),
        "SILVER":     dict(tests=1680, nominal=95,  chance=84, fdr=1, tradable=0),
        "NATURALGAS": dict(tests=1680, nominal=133, chance=84, fdr=6, tradable=6),
        "COPPER":     dict(tests=1675, nominal=191, chance=84, fdr=4, tradable=1),
    },
    # the survivors, in words, after scripts/crude_vs_index_survivors.py tried to
    # kill them (permutation, own-momentum control, market-neutral, placebo shifts)
    "survivor_note": {
        "NATURALGAS": (
            "After natural gas's 10-session move ranks in its top 10%, Nifty Energy, "
            "PSE, Oil & Gas and Metal have risen about +0.3% to +0.4% more than usual "
            "from the next open to the next close (Energy: 204 days, 61 episodes, "
            "t +4.0, positive in 7 of 8 years). Shifting the signal 5 or 21 sessions "
            "either way kills it, so it is specific to the next day, and it is not "
            "the opening gap reversing (holding the gap fixed: t +4.2). It is found "
            "IN-SAMPLE, and Nifty Energy has no futures, so it would have to be "
            "traded through its stocks. Treat it as a candidate, not a signal."),
        "COPPER": (
            "When copper rises two or more sessions in a row, Nifty Metal is up about "
            "+0.4% more than usual the next day. Most of that arrives in the opening "
            "gap — metal stocks price London copper overnight — so it happens before "
            "you can act on the MCX close."),
        "SILVER": (
            "Silver's one survivor is Nifty Metal's opening gap after a +3% silver day: "
            "a reaction that happens overnight, not something you can trade."),
    },
    # ── commodity x sector matrix (scripts/commodity_sector_matrix.py,
    # _survivors.py, _reality_check.py), 6 years to 11 Sep 2026 ──────────────
    "matrix": {
        "window": "Sep 2020 – Sep 2026", "weeks": 297,
        "same_week_pairs": 144, "same_week_fdr": 24, "sector_only_fdr": 19,
        "forward_tests": 5024, "forward_nominal": 374, "forward_chance": 251,
        "forward_fdr": 9,
        # Reality Check: 10,385 tests, best real statistic vs the best the SAME
        # search finds on circular-shifted commodities (500 shifts)
        "rc_tests": 10385, "rc_best": 8.01, "rc_null_95": 5.92, "rc_p": 0.000,
        "rc_beats": ["COPPER 1-day crash -> Nifty Metal opening gap",
                     "ZINC 1-day crash -> Nifty Metal opening gap",
                     "ALUMINIUM 1-day crash -> Nifty Metal opening gap"],
        # every tradable candidate, measured against that same search-aware bar
        "rc_candidates": {
            "NATURALGAS 10d top 10% -> Nifty Energy next day": 2.83,
            "LEAD up 3+ -> Nifty PSE next 20 days": 1.57,
            "CRUDE OIL 10d bottom 10% -> Nifty FMCG next day": 2.12,
            "COPPER up 3+ -> Nifty Metal next day": 1.71,
        },
        # strongest same-week exposures, sector minus Nifty 50: (rho, beta)
        "top_exposures": {
            "COPPER/Nifty Metal": (0.43, 0.49), "ALUMINIUM/Nifty Metal": (0.40, 0.45),
            "ZINC/Nifty Metal": (0.37, 0.34), "SILVER/Nifty Metal": (0.30, 0.21),
            "GOLD/Nifty Metal": (0.23, 0.31), "CRUDE OIL/Nifty Metal": (0.18, 0.12),
            "COPPER/Nifty Commodities": (0.28, 0.18),
            "CRUDE OIL/Nifty Oil & Gas": (0.035, 0.035),
            "ZINC/Nifty FMCG": (-0.20, -0.14), "COPPER/Nifty FMCG": (-0.16, -0.13),
        },
        # the LEAD lesson: a 20-day return sampled daily is not 65 observations
        "lead_hac_t": 4.22, "lead_overlap_adjusted": 1.57,
        # commodity weekly returns are not independent bets
        "gold_silver_corr": 0.80, "copper_zinc_corr": 0.65,
    },
    # ── monthly (scripts/commodity_sector_monthly.py), 94 months Aug 2018-Jul 2026
    "monthly": {
        "months": 94, "ic_needed": 0.20,
        # walk-forward: betas fitted on earlier months only, sectors ranked by
        # this month's commodity moves, scored against next month's excess
        "wf_months": 58, "wf_ic": -0.105, "wf_t": -2.41, "wf_pos_share": 0.34,
        "wf_spread": -0.72, "wf_spread_t": -1.34,
        "wf_ic_h1": -0.155, "wf_ic_h2": -0.055,
        "wf_own_momentum_ic": 0.046,
        # single pairs, non-overlapping months
        "pair_tests": 252, "pair_nominal": 15, "pair_chance": 13, "pair_fdr": 0,
        "pair_rc_best": 3.17, "pair_rc_null_median": 3.40, "pair_rc_p": 0.70,
        # same-MONTH co-move, which is what the scenario table uses
        "same_month_fdr": 13, "same_month_tests": 117,
        "same_month_top": {"COPPER/Nifty Metal": (0.54, 0.58),
                           "ALUMINIUM/Nifty Metal": (0.53, 0.66),
                           "ZINC/Nifty Metal": (0.45, 0.46),
                           "COPPER/Nifty Commodities": (0.40, 0.24),
                           "SILVER/Nifty Metal": (0.32, 0.21),
                           "GOLD/Nifty Realty": (-0.26, -0.40)},
    },
    "crude": {
        "tests": 1680,
        "nominal": 101,
        "expected_by_chance": 84,
        "fdr_survivors": 0,
        # Nifty 50 after crude up 5+ sessions in a row (n = 58, 32 episodes)
        "n50_up5_next_day": -0.016, "n50_base_next_day": 0.045,
        "n50_up5_t": -0.68,
        # same-week link, Nifty 50 vs crude, non-overlapping weeks, by year
        "same_week_corr_by_year": {2018: -0.22, 2019: -0.01, 2020: 0.26,
                                   2021: 0.21, 2022: -0.19, 2023: 0.10,
                                   2024: -0.13, 2025: -0.09, 2026: -0.68},
        "same_week_corr_full": 0.011,
        # does the recent link predict the next one? Trailing 26 weeks vs the
        # NEXT 13 weeks, 29 disjoint steps (scripts/crude_vs_index_audit.py B).
        # Sign carried over 45% of the time; of the 4 strong readings (|r|>0.4)
        # only 1 held -- 2026's. Too few to show persistence either way.
        "regime_sign_agree": 0.45, "regime_steps": 29,
        "regime_strong_held": "1 of 4",
        # inside a negative regime, crude's week does not lead Nifty's next week
        "neg_regime_next_week_t": -0.62,
        # crude day -> NEXT Nifty opening gap, by year (reaction, not tradable):
        # only 2026 (-0.41) and 2018 (-0.19) show the gap absorbing crude news
        "gap_corr_2026": -0.41,
        # RUPEE SPLIT (audit A): MCX crude = USD crude x USD/INR. Same-week link
        # with Nifty 50, 2018-2026: USD/INR -0.39 (t -6.8, every period), USD
        # crude +0.04. In 2026 USD crude -0.55 (t -3.7 with the rupee held
        # fixed) -- 2026 is genuinely about oil; the rupee is the steady driver.
        "usdinr_link_full": -0.39, "usd_crude_link_full": 0.04,
        "usd_crude_link_2026": -0.55, "usd_crude_t_2026": -3.7,
    },
}


# ── loading ──────────────────────────────────────────────────────────────────
def liquid_years(commodity: str, floor_cr: float) -> set[int]:
    """Calendar years whose MEDIAN front-month session traded at least floor_cr.

    MCX NICKEL's median session is Rs 1 Cr from 2022 and LEAD's Rs 18 Cr by
    2026: the closes exist, but they are quotes, not a market.
    """
    df = query_dataframe(
        "SELECT year(trade_date) y, median(turnover_cr) m FROM commodity_daily "
        "WHERE commodity = ? GROUP BY 1", [commodity])
    return {int(r.y) for r in df.itertuples() if float(r.m) >= floor_cr}


def load_commodity(commodity: str, as_of: Optional[date] = None,
                   min_median_turnover_cr: float = 0.0) -> pd.DataFrame:
    """Front-month series for one commodity: close, ret1, chained level.

    `min_median_turnover_cr` drops illiquid calendar years. It is applied HERE,
    before the break detection below, so that a window spanning a dropped year
    is treated as a gap rather than measured straight across it -- filtering
    after loading would print a year-long move as a 10-session one.
    """
    sql = ("SELECT trade_date, close, ret1 FROM commodity_daily "
           "WHERE commodity = ? AND trade_date >= ?")
    params: list = [commodity, START.date()]
    if as_of is not None:
        sql += " AND trade_date <= ?"
        params.append(as_of)
    if min_median_turnover_cr > 0:
        keep = liquid_years(commodity, min_median_turnover_cr)
        if not keep:
            return pd.DataFrame(columns=["close", "ret1", "lvl", "brk"])
        sql += " AND year(trade_date) IN (" + ", ".join(str(y) for y in sorted(keep)) + ")"
    df = query_dataframe(sql + " ORDER BY trade_date", params)
    if df.empty:
        return pd.DataFrame(columns=["close", "ret1", "lvl", "brk"])
    df["trade_date"] = pd.to_datetime(df.trade_date)
    df = df.set_index("trade_date")
    # A row more than _BREAK_DAYS after the one before it is not a one-session
    # return. CFM measures `ret1` within the contract, so across the June-2018
    # source hole the 2 Jul 2018 row carries the whole 31 May -> 2 Jul move
    # (+12.9% on crude) and would read as a one-day spike. The first row of the
    # window has no prior session inside it at all.
    gap = df.index.to_series().diff().dt.days
    df["brk"] = gap.isna() | (gap > _BREAK_DAYS)
    df.loc[df["brk"], "ret1"] = 0.0
    df["ret1"] = df["ret1"].fillna(0.0)
    df["lvl"] = (1 + df.ret1).cumprod()
    return df


def load_index_ohlc(indices: list[str], as_of: Optional[date] = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open and close per index, on Nifty 50's session calendar."""
    names = list(dict.fromkeys(["Nifty 50", *indices]))
    ph = ", ".join("?" * len(names))
    sql = (f"SELECT trade_date, index_name, open_val, close_val FROM index_data "
           f"WHERE trade_date >= ? AND index_name IN ({ph})")
    params: list = [(START - pd.Timedelta(days=60)).date(), *names]
    if as_of is not None:
        sql += " AND trade_date <= ?"
        params.append(as_of)
    df = query_dataframe(sql, params)
    if df.empty:
        e = pd.DataFrame()
        return e, e
    df["trade_date"] = pd.to_datetime(df.trade_date)
    o = df.pivot(index="trade_date", columns="index_name", values="open_val")
    c = df.pivot(index="trade_date", columns="index_name", values="close_val")
    days = c["Nifty 50"].dropna().index
    return o.reindex(days), c.reindex(days)


# ── construction ─────────────────────────────────────────────────────────────
def commodity_features(p: pd.DataFrame) -> pd.DataFrame:
    """Causal features at each MCX close. Nothing looks past the row's own date."""
    f = pd.DataFrame(index=p.index)
    r = p["ret1"]
    up = (r > 0).astype(int)
    dn = (r < 0).astype(int)
    f["up_streak"] = up.groupby((up == 0).cumsum()).cumsum()
    f["dn_streak"] = dn.groupby((dn == 0).cumsum()).cumsum()
    f["r1"] = r
    brk = (p["brk"] if "brk" in p else pd.Series(False, index=p.index)).astype(int)
    brk_iloc = brk.iloc[1:]                    # the window's own first row is not a gap
    since_brk = brk_iloc.reindex(p.index, fill_value=0)
    for k in (3, 5, 10, 20):
        f[f"r{k}"] = p["lvl"] / p["lvl"].shift(k) - 1
        # a k-session window that spans a source hole is not a k-session move
        spans = since_brk.rolling(k, min_periods=1).max().astype(bool)
        f.loc[spans, f"r{k}"] = np.nan
    # one-session move in units of its own trailing 60-session volatility
    sd = r.shift(1).rolling(60, min_periods=40).std()
    f["z1"] = r / sd
    f.loc[since_brk.astype(bool), "z1"] = np.nan
    # the k-session move ranked against its own trailing three years
    for k in (5, 10, 20):
        f[f"pct{k}"] = f[f"r{k}"].rolling(756, min_periods=250).rank(pct=True)
    f["close"] = p["close"]
    f["lvl"] = p["lvl"]
    return f


def align_to_sessions(feat: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Commodity state at the last MCX close STRICTLY BEFORE each NSE session."""
    left = pd.DataFrame({"trade_date": pd.DatetimeIndex(days)})
    f = feat.reset_index().rename(columns={feat.index.name or "index": "mcx_date"})
    if "mcx_date" not in f.columns:
        f = f.rename(columns={f.columns[0]: "mcx_date"})
    m = pd.merge_asof(left.sort_values("trade_date"), f.sort_values("mcx_date"),
                      left_on="trade_date", right_on="mcx_date",
                      allow_exact_matches=False, direction="backward")
    stale = (m.trade_date - m.mcx_date).dt.days > _STALE_DAYS
    m.loc[stale, [c for c in feat.columns]] = np.nan
    m.loc[stale, "mcx_date"] = pd.NaT
    return m.set_index("trade_date")


def index_outcomes(o: pd.DataFrame, c: pd.DataFrame,
                   horizons=(5, 10, 20)) -> dict[str, pd.DataFrame]:
    """Per NSE session S. `gap`/`oc1`/`fwdH` for the study; `ccH` for display."""
    pc = c.shift(1)
    out = {"gap": o / pc - 1, "oc1": c / o - 1, "cc1": c / pc - 1,
           "pts1": c - pc}
    for h in horizons:
        out[f"fwd{h}"] = c.shift(-(h - 1)) / o - 1          # from the open, tradable
        out[f"cc{h}"] = c.shift(-(h - 1)) / pc - 1          # close to close
        out[f"pts{h}"] = c.shift(-(h - 1)) - pc
    for k in (3, 5, 10, 20):
        out[f"same{k}"] = pc / c.shift(1 + k) - 1           # the k sessions to S-1
    return out


def pattern_mask(a: pd.DataFrame, kind: str, n: int = 3, thr: float = 3.0,
                 k: int = 5) -> pd.Series:
    """Boolean per NSE session: did the commodity state before it match?"""
    t = thr / 100.0
    if kind == "up_streak":
        m = a["up_streak"] >= n
    elif kind == "down_streak":
        m = a["dn_streak"] >= n
    elif kind == "spike_up":
        m = a["r1"] > t
    elif kind == "spike_down":
        m = a["r1"] < -t
    elif kind == "rise_k":
        m = a[f"r{k}"] > t
    elif kind == "fall_k":
        m = a[f"r{k}"] < -t
    else:
        raise ValueError(f"unknown pattern {kind!r}")
    return m.fillna(False).astype(bool)


def episode_starts(mask: pd.Series, cooldown: int = 1) -> pd.Series:
    """First session of each run of True. A 7-session crude rally is ONE event,
    not five overlapping ones -- counting every day of it would inflate n.

    `cooldown`: a new event also needs the pattern to have been OFF for that many
    sessions. Without it a rally that hovers at the line ("up >10% over 20
    sessions": on, off, on) was counted again and again -- 32 of 60 crude
    episodes restarted within four weeks of the previous one.
    """
    m = mask.astype(bool)
    recent = (m.shift(1, fill_value=False).astype(int)
              .rolling(max(cooldown, 1), min_periods=1).max().astype(bool))
    return m & ~recent


def _independent(idx_pos: np.ndarray, h: int) -> int:
    """How many of these start positions are at least `h` sessions apart -- the
    number of NON-overlapping outcome windows, i.e. the honest sample size."""
    n, last = 0, -10**9
    for i in idx_pos:
        if i - last >= h:
            n, last = n + 1, i
    return n


def _pattern_window(kind: str, n: int, k: int) -> int:
    return n if kind in ("up_streak", "down_streak") else 1 if kind.startswith("spike") else k


def pattern_label(commodity: str, kind: str, n: int = 3, thr: float = 3.0,
                  k: int = 5) -> str:
    return PATTERNS[kind].format(c=COMMODITIES.get(commodity, commodity),
                                 n=n, thr=thr, k=k)


# ── panel payloads ───────────────────────────────────────────────────────────
@dataclass
class CommodityState:
    commodity: str
    data_ok: bool = False
    note: str = ""
    mcx_date: Optional[date] = None      # MCX close that the read is from
    close: Optional[float] = None
    r1: Optional[float] = None
    r5: Optional[float] = None
    r20: Optional[float] = None
    pct5: Optional[float] = None
    pct20: Optional[float] = None
    up_streak: int = 0
    dn_streak: int = 0
    corr_26w: Optional[float] = None     # same-week link, last 26 weeks
    corr_full: Optional[float] = None    # same-week link, since Jul 2018
    weeks_26w: int = 0
    lag_note: str = ""
    stale: bool = False                  # MCX data is more than one session behind
    liquid_note: str = ""                # set when this commodity barely trades now
    extras: dict = field(default_factory=dict)


def _weekly_blocks(feat: pd.DataFrame, c: pd.Series) -> pd.DataFrame:
    """Non-overlapping 5-session blocks ENDING on the latest session, so every
    point is independent and the most recent week is always the last one."""
    same = c / c.shift(5) - 1
    # commodity 5-session move as of the MCX close on or before the block end
    fx = feat[["r5"]].reset_index().rename(columns={feat.index.name or "index": "d"})
    fx = fx.rename(columns={fx.columns[0]: "d"})
    left = pd.DataFrame({"t": c.index})
    m = pd.merge_asof(left, fx.sort_values("d"), left_on="t", right_on="d",
                      direction="backward")
    stale = (m.t - m.d).dt.days > _STALE_DAYS
    m.loc[stale, "r5"] = np.nan
    w = pd.DataFrame({"cmd5": m.r5.values, "idx5": same.values}, index=c.index)
    w = w[w.index >= START + pd.Timedelta(days=10)]
    pos = np.arange(len(w))[::-1][::5][::-1]      # every 5th, anchored at the end
    return w.iloc[pos].dropna()


def _corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    if len(x) < 8:
        return None
    v = x.rank().corr(y.rank())
    return None if pd.isna(v) else float(v)


def get_commodity_state(as_of: date, commodity: str = "CRUDE OIL",
                        index: str = "Nifty 50") -> CommodityState:
    s = CommodityState(commodity=commodity)
    if pd.Timestamp(as_of) < START:
        s.note = (f"The commodity history starts {START:%d %b %Y} (all of June 2018 "
                  f"is missing at the source). Pick a later date.")
        return s
    p = load_commodity(commodity, as_of)
    if p.empty:
        s.note = ("No MCX data. Run `python -m src.cli sync-commodity` "
                  "(copies it from the Commodity_Forex_Market project).")
        return s
    f = commodity_features(p)
    last = f.iloc[-1]
    s.data_ok = True
    s.mcx_date = f.index[-1].date()
    s.close = float(last.close)
    for a in ("r1", "r5", "r20", "pct5", "pct20"):
        v = last[a]
        setattr(s, a, None if pd.isna(v) else float(v))
    s.up_streak = int(last.up_streak)
    s.dn_streak = int(last.dn_streak)
    # is this still a market? MCX nickel and lead have all but stopped trading,
    # and a price nobody traded at cannot be compared with anything.
    med = query_dataframe(
        "SELECT median(turnover_cr) m FROM commodity_daily WHERE commodity = ? "
        "AND year(trade_date) = ?", [commodity, as_of.year])
    if not med.empty and pd.notna(med.m.iloc[0]) and float(med.m.iloc[0]) < MATRIX_MIN_MEDIAN_CR:
        yrs = sorted(liquid_years(commodity, MATRIX_MIN_MEDIAN_CR))
        s.liquid_note = (
            f"**MCX {COMMODITIES.get(commodity, commodity).lower()} has all but "
            f"stopped trading**: the median {as_of.year} session is Rs "
            f"{float(med.m.iloc[0]):,.0f} Cr. Its closes are quotes, not a market, so "
            f"recent readings mean little"
            + (f" — it last traded properly in {yrs[-1]}." if yrs else "."))
    _, c = load_index_ohlc([index], as_of)
    if s.mcx_date < as_of:
        # NSE sessions the MCX copy is behind. One is normal (MCX closes at 23:30,
        # after the 19:30 daily run); more means the sync has stopped.
        behind = int(((c.index > pd.Timestamp(s.mcx_date))
                      & (c.index <= pd.Timestamp(as_of))).sum()) if not c.empty else 0
        if behind > 1:
            s.stale = True
            s.lag_note = (f"MCX data stops at {s.mcx_date:%d %b %Y} — {behind} NSE "
                          f"sessions behind. The commodity sync has not run; use "
                          f"`python -m src.cli sync-commodity` or Refresh Data.")
        else:
            s.lag_note = (f"Latest MCX close is {s.mcx_date:%d %b} — MCX trades till "
                          f"23:30, so {as_of:%d %b}'s close lands after the evening "
                          f"sync.")
    if not c.empty and index in c:
        w = _weekly_blocks(f, c[index].dropna())
        s.corr_full = _corr(w.cmd5, w.idx5)
        tail = w.tail(26)
        s.weeks_26w = len(tail)
        s.corr_26w = _corr(tail.cmd5, tail.idx5)
    return s


def get_pattern_episodes(as_of: date, commodity: str, index: str, kind: str,
                         n: int = 3, thr: float = 3.0, k: int = 5
                         ) -> tuple[pd.DataFrame, dict]:
    """Every past time the pattern started, and what the index did next.

    Returns (episodes newest first, summary). Outcomes that fall after `as_of`
    are left blank -- the panel never reads the future of the selected date.
    """
    p = load_commodity(commodity, as_of)
    o, c = load_index_ohlc([index], as_of)
    if p.empty or c.empty or index not in c:
        return pd.DataFrame(), {}
    f = commodity_features(p)
    a = align_to_sessions(f, c.index)
    oc = index_outcomes(o[[index]], c[[index]])
    valid = a["r1"].notna() & (a.index >= START) & c[index].notna()
    mask = pattern_mask(a, kind, n=n, thr=thr, k=k) & valid
    win = _pattern_window(kind, n, k)
    starts = episode_starts(mask, cooldown=max(win, 5))
    lvl = f["lvl"]
    cmd_move = (lvl / lvl.shift(win) - 1).reindex(a["mcx_date"].values).values
    idx_same = (c[index].shift(1) / c[index].shift(1 + win) - 1)

    def col(key):
        return oc[key][index]

    rows = pd.DataFrame({
        "mcx_date": a["mcx_date"],
        "session": a.index,
        "cmd_close": a["close"],
        "cmd_move": cmd_move,
        "idx_same": idx_same,
        "idx_close_before": c[index].shift(1),
        "next1": col("cc1"), "pts1": col("pts1"),
        "next5": col("cc5"), "pts5": col("pts5"),
        "next20": col("cc20"), "pts20": col("pts20"),
        "gap": col("gap"),
    }, index=a.index)
    ev = rows[starts.reindex(rows.index).fillna(False).values].copy()

    base = rows[valid.values]
    # does the LATEST MCX close (which may be tonight's, not yet in any NSE
    # session) sit inside the pattern? evaluated on the raw MCX row.
    today = pattern_mask(f.iloc[[-1]], kind, n=n, thr=thr, k=k)
    summary = {"label": pattern_label(commodity, kind, n, thr, k),
               "episodes": len(ev), "window": win,
               "today_in": bool(today.iloc[0]), "today_mcx": f.index[-1].date()}
    pos = np.flatnonzero(starts.reindex(rows.index).fillna(False).values)
    for h, hz in (("next1", 1), ("next5", 5), ("next20", 20)):
        e = ev[h].dropna()
        b = base[h].dropna()
        # episodes whose outcome windows do not overlap: the honest n. Two
        # rallies 3 sessions apart share 17 of their 20 next-20 sessions.
        ok_pos = pos[~np.isnan(rows[h].values[pos].astype(float))]
        n_eff = _independent(ok_pos, hz)
        summary[h] = dict(
            n=len(e), n_eff=n_eff,
            avg=float(e.mean()) if len(e) else None,
            up=float((e > 0).mean()) if len(e) else None,
            base_avg=float(b.mean()) if len(b) else None,
            base_up=float((b > 0).mean()) if len(b) else None,
            # 2 standard errors of the mean of n_eff ordinary sessions: a random
            # sample that size lands this far from normal about 1 time in 20.
            noise=float(2 * b.std() / np.sqrt(n_eff)) if n_eff >= 3 else None)
    return ev.iloc[::-1].reset_index(drop=True), summary


def get_yearly_link(as_of: date, commodity: str, index: str = "Nifty 50",
                    up_thr: float = 3.0) -> pd.DataFrame:
    """Per calendar year: weeks the commodity rose > up_thr%, and the index then."""
    p = load_commodity(commodity, as_of)
    _, c = load_index_ohlc([index], as_of)
    if p.empty or c.empty or index not in c:
        return pd.DataFrame()
    f = commodity_features(p)
    w = _weekly_blocks(f, c[index].dropna())
    w["next"] = w["idx5"].shift(-1)       # the following block
    rows = []
    for y, g in w.groupby(w.index.year):
        u = g[g.cmd5 > up_thr / 100]
        rows.append(dict(
            year=int(y), weeks=len(g), up_weeks=len(u),
            same_avg=float(u.idx5.mean()) if len(u) else None,
            same_down=float((u.idx5 < 0).mean()) if len(u) else None,
            all_down=float((g.idx5 < 0).mean()),
            next_avg=float(u.next.dropna().mean()) if u.next.notna().any() else None,
            corr=_corr(g.cmd5, g.idx5)))
    return pd.DataFrame(rows)


def get_index_link(as_of: date, commodity: str) -> pd.DataFrame:
    """Same-week link of every index to the commodity: full period, last 52 and 26 weeks."""
    p = load_commodity(commodity, as_of)
    _, c = load_index_ohlc(INDEX_CHOICES, as_of)
    if p.empty or c.empty:
        return pd.DataFrame()
    f = commodity_features(p)
    rows = []
    for ix in INDEX_CHOICES:
        if ix not in c:
            continue
        w = _weekly_blocks(f, c[ix].dropna())
        if len(w) < 30:
            continue
        rows.append(dict(index=ix, corr_full=_corr(w.cmd5, w.idx5),
                         corr_52w=_corr(w.tail(52).cmd5, w.tail(52).idx5),
                         corr_26w=_corr(w.tail(26).cmd5, w.tail(26).idx5),
                         since=w.index[0].date()))
    return pd.DataFrame(rows)


def _matrix_weekly(as_of: date, years: float, commodities: list[str],
                    sectors: list[str]):
    """Non-overlapping 5-session returns for every commodity and sector."""
    _, c = load_index_ohlc(sectors, as_of)
    if c.empty:
        return None, None, None
    have = [s for s in sectors if s in c.columns and c[s].notna().sum() > 100]
    start = c.index.max() - pd.Timedelta(days=int(years * 365.25))
    days = c.index[c.index >= start]
    if len(days) < 120:
        return None, None, None
    cols = {}
    for cm in commodities:
        p = load_commodity(cm, as_of, min_median_turnover_cr=MATRIX_MIN_MEDIAN_CR)
        if len(p) < 250:
            continue
        cols[cm] = p["lvl"].reindex(days, method="ffill")
    if not cols:
        return None, None, None
    cw = pd.DataFrame(cols, index=days)
    cw = cw / cw.shift(5) - 1
    iw = c[have].reindex(days)
    iw = iw / iw.shift(5) - 1
    pos = np.arange(len(days))[::-1][::5][::-1]     # anchored on the latest week
    return cw.iloc[pos], iw.iloc[pos], have


def get_sector_matrix(as_of: date, years: float = 6.0,
                      sector_only: bool = True) -> dict:
    """Same-week link between every commodity and every sector.

    `sector_only` subtracts Nifty 50 from each sector first, which answers "is
    this about the SECTOR" rather than "was it a good week for everything".
    Returns rank correlations, betas (1% commodity move -> beta% sector move),
    and the number of independent weeks behind them.
    """
    cw, iw, sectors = _matrix_weekly(as_of, years, list(COMMODITIES), SECTOR_MATRIX)
    if cw is None:
        return {}
    mkt = iw["Nifty 50"] if "Nifty 50" in iw else None
    rho = pd.DataFrame(index=sectors, columns=cw.columns, dtype=float)
    beta = rho.copy()
    for ix in sectors:
        y = iw[ix]
        if sector_only and mkt is not None:
            if ix == "Nifty 50":
                continue                      # the market minus itself is zero
            y = y - mkt
        for cm in cw.columns:
            g = pd.DataFrame({"x": cw[cm], "y": y}).dropna()
            if len(g) < 60:
                continue
            rho.loc[ix, cm] = float(g.x.rank().corr(g.y.rank()))
            v = float(g.x.var())
            beta.loc[ix, cm] = float(g.x.cov(g.y) / v) if v else np.nan
    rho = rho.dropna(how="all").dropna(axis=1, how="all")
    weeks = int(len(cw.dropna(how="all")))
    # |rho| a sample this size clears by chance about 1 time in 20
    band = 1.96 / np.sqrt(max(weeks, 2))
    return {"rho": rho, "beta": beta.reindex_like(rho), "weeks": weeks,
            "band": float(band), "sector_only": sector_only,
            "years": years, "as_of": as_of}


_MONTH = 21          # sessions in a "month" for the month-ahead table


def get_month_ahead(as_of: date, commodity: str, years: float = 8.2) -> dict:
    """Where the commodity stands this month, and what that has meant.

    Two columns that must never be confused:

      `implied_same_month` -- the co-move that goes WITH this commodity move in
          the SAME month (beta x the move). Measured, strong for metals, and
          not a forecast: it is what tends to have happened alongside, and it
          is what tells you the risk in a position you already hold.

      `next_month_*` -- what the sector actually did over the FOLLOWING month
          when this commodity was in the same state. Base rates. The study
          found ranking sectors this way scored a rank IC of -0.105 (t -2.41)
          -- i.e. slightly WRONG -- so these are shown with their luck band and
          no arrow.
    """
    p = load_commodity(commodity, as_of, min_median_turnover_cr=MATRIX_MIN_MEDIAN_CR)
    _, c = load_index_ohlc(SECTOR_MATRIX, as_of)
    if p.empty or c.empty or "Nifty 50" not in c:
        return {}
    start = c.index.max() - pd.Timedelta(days=int(years * 365.25))
    days = c.index[c.index >= start]
    lvl = p["lvl"].reindex(days, method="ffill")
    mom = lvl / lvl.shift(_MONTH) - 1
    now = mom.iloc[-1]
    if pd.isna(now):
        return {}
    rank = float((mom.dropna() <= now).mean())
    mkt = c["Nifty 50"].reindex(days)

    # non-overlapping month grid, anchored on the latest session
    pos = np.arange(len(days))[::-1][::_MONTH][::-1]
    pos = pos[pos >= _MONTH]
    x = mom.values[pos]
    fwd_pos = pos + _MONTH
    ok_fwd = fwd_pos < len(days)
    # the state we are in now, in quintile terms
    qs = pd.qcut(pd.Series(x).dropna(), 5, labels=False, duplicates="drop")
    q_now = int(pd.Series([now]).apply(
        lambda v: (pd.Series(x).dropna() <= v).mean() * 5).clip(0, 4.999).iloc[0])

    rows = []
    for ix in SECTOR_MATRIX:
        if ix == "Nifty 50" or ix not in c:
            continue
        px = c[ix].reindex(days)
        ex_same = (px / px.shift(_MONTH) - 1) - (mkt / mkt.shift(_MONTH) - 1)
        g = pd.DataFrame({"x": mom, "y": ex_same}).dropna()
        g = g.loc[g.index.isin(days[pos])]
        if len(g) < 40:
            continue
        var = float(g.x.var())
        beta = float(g.x.cov(g.y) / var) if var else np.nan
        ic = float(g.x.rank().corr(g.y.rank()))
        # forward month, entered after this month's close
        fex = ((px.shift(-_MONTH) / px - 1) - (mkt.shift(-_MONTH) / mkt - 1))
        fv = fex.values[pos][ok_fwd]
        xv = x[ok_fwd]
        qq = pd.qcut(pd.Series(xv), 5, labels=False, duplicates="drop")
        sel = (qq == q_now).values & ~np.isnan(fv)
        base = fv[~np.isnan(fv)]
        nxt = fv[sel]
        rows.append(dict(
            sector=ix, same_month_ic=ic, beta=beta,
            implied_same_month=beta * now if pd.notna(beta) else np.nan,
            actual_same_month=float(ex_same.iloc[-1]) if pd.notna(ex_same.iloc[-1]) else np.nan,
            next_month_mean=float(np.mean(nxt)) if len(nxt) else np.nan,
            next_month_n=int(len(nxt)),
            next_month_base=float(np.mean(base)) if len(base) else np.nan,
            next_month_band=(float(2 * np.std(base, ddof=1) / np.sqrt(len(nxt)))
                             if len(nxt) > 2 and len(base) > 2 else np.nan)))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df = df.reindex(df.same_month_ic.abs().sort_values(ascending=False).index)
    months = int(len(pd.Series(x).dropna()))
    return {"commodity": commodity, "as_of": as_of, "month_move": float(now),
            "month_rank": rank, "quintile": q_now + 1, "months": months,
            "band": float(1.96 / np.sqrt(max(months, 2))), "table": df,
            "mcx_date": p.index[-1].date()}


def get_rebased_paths(as_of: date, commodity: str, index: str,
                      sessions: int = 250) -> pd.DataFrame:
    """Commodity and index rebased to 100, last `sessions` NSE sessions."""
    p = load_commodity(commodity, as_of)
    _, c = load_index_ohlc([index], as_of)
    if p.empty or c.empty or index not in c:
        return pd.DataFrame()
    ix = c[index].dropna().tail(sessions)
    # The QUOTED front-month price, which is what a trader reads as "crude went
    # up". Not the chained level: in 2026's steep backwardation the chained
    # series adds ~18 points of roll yield a year (+93% vs +75% quoted), which a
    # held futures position earned but the import bill did not. Cost: a small
    # step on each monthly roll (the calendar spread), noted in the caption.
    cm = p["close"].reindex(ix.index, method="ffill")
    first = cm.first_valid_index()
    if first is None:
        return pd.DataFrame()
    ix, cm = ix.loc[first:], cm.loc[first:]
    return pd.DataFrame({index: ix / ix.iloc[0] * 100,
                         COMMODITIES.get(commodity, commodity): cm / cm.iloc[0] * 100})
