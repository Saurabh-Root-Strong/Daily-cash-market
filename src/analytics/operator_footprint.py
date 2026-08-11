"""
Operator footprint — unusual F&O positioning in single stocks.

THE QUESTION
    Someone with size builds a position in a stock's futures or options before it
    moves. That leaves a footprint: open interest appearing where it normally does
    not, at a size that is abnormal FOR THAT STOCK, with a premium/price direction
    that says whether it was bought or written.

WHAT "UNUSUAL" HAS TO MEAN (raw OI is meaningless)
    Three normalisations, all necessary:

    1. DAYS-TO-EXPIRY ALIGNED. Open interest builds mechanically through a cycle.
       Comparing day 3 of a cycle with day 20 of another is noise. Every
       comparison here is made against the SAME days-to-expiry in prior cycles.

    2. MONEYNESS BUCKETS, NOT STRIKES. A strike that was 10% OTM last cycle may be
       ITM now, so "the 21 strike vs the 21 strike" breaks the moment the stock
       moves. Strikes are bucketed by strike/spot, and a bucket is compared with
       the same bucket in prior cycles.

    3. RELATIVE TO THE STOCK'S OWN BOOK. A crore of OI is enormous in a small name
       and invisible in RELIANCE, so buildup is also expressed as a share of that
       stock's own total OI.

    A normaliser worth less than Rs 1 Cr is REJECTED rather than used: dividing by
    almost nothing manufactures huge multiples. Measured, those cases read a median
    38.4x against 4.3x for the rest.

KNOWN SELECTION BIAS — this list skews to CALM stocks, not busy ones
    Measured 2026-08-10, the scored names had a median annualised volatility of
    27.2% against 39.0% for the whole F&O universe. That is mechanical: a quiet
    stock has a stable OI baseline, so a deviation stands out, while a volatile
    name is noisy enough that nothing looks abnormal. Useful — unusual size in a
    quiet stock IS the more interesting event — but it means "where size showed
    up" is not the same as "where the action is".

KNOWN CONFOUND IN THE BUY/WRITE LABEL — measured, not theoretical
    An option's premium moves with the underlying, so "OI up + premium up" is
    mostly just "OI up on a day the stock rose". Measured on 2026-08-07's top
    strikes: a CALL's premium moved with spot 83% of the time, and a PUT's moved
    against spot 94% of the time. The BUYING/WRITING label is therefore ~85-90%
    determined by the day's price direction rather than by independent order
    flow, and this is very likely WHY the backtest found call BUYING and call
    WRITING scoring identically (+0.337 vs +0.334) — they are largely "OI rose on
    an up day" vs "OI rose on a down day". Separating true demand would need the
    premium move net of delta x spot move, i.e. greeks this dataset does not
    carry. Read the label as a description of the day, not as order flow.

POSITION INTERPRETATION (OI direction x price direction)
    Futures / options both follow the standard reading:
        OI up   + price up    -> LONG BUILDUP      (buyers opening)
        OI up   + price down  -> SHORT BUILDUP     (writers/sellers opening)
        OI down + price up    -> SHORT COVERING    (shorts buying back)
        OI down + price down  -> LONG UNWINDING    (longs exiting)
    For options the premium is the price, so a call with rising OI and rising
    premium is CALL BUYING (bullish), while rising OI with falling premium is
    CALL WRITING (bearish — someone is selling upside).

WHY ITM MATTERS (the case that motivated this)
    Ordinary retail flow lives in OTM strikes; they are cheap. Heavy open interest
    appearing ITM is unusual, costs real money, and is much harder to explain as
    noise. ITM buildup is therefore scored separately and weighted higher.

HONESTY
    Nothing here is a validated signal until scripts/backtest_operator_footprint.py
    says so. This codebase has repeatedly found F&O open-interest reads to be
    DESCRIPTIVE at index level (see the index-prediction and CE/PE-crossover
    studies). Stock level is a different question, and the answer is whatever the
    backtest returns — the UI must show that number, not a story.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_operator_footprint", "ACTION_LABELS",
           "OPTION_ACTION_LABELS", "ACTION_LEAN"]

# OI direction x price direction -> what actually happened.
# FUTURES ONLY. Here "long"/"short" refer to the underlying, so the label carries
# its usual market meaning and the direction is unambiguous.
ACTION_LABELS = {
    ("up", "up"):     ("LONG BUILDUP",   "new buyers opening positions"),
    ("up", "down"):   ("SHORT BUILDUP",  "new sellers/writers opening"),
    ("down", "up"):   ("SHORT COVERING", "sellers buying back"),
    ("down", "down"): ("LONG UNWINDING", "buyers exiting"),
}

# OPTIONS need their own labels. Reusing the futures wording READS BACKWARDS on a
# put: "SHORT BUILDUP" on a PE means short the OPTION — i.e. put WRITING, which is
# BULLISH — but every trader reads "short buildup" as bearish. Real case that
# exposed it: TITAN PE 5000 on 2026-08-10, premium 147.50 -> 45.35 with OI
# 256k -> 781k (Rs 267 Cr) while the stock rose 3.02%. That is put writing, a
# bullish act, and it was rendering as "SHORT BUILDUP". So options get labels that
# name the instrument and the act, plus the directional lean spelled out.
OPTION_ACTION_LABELS = {
    ("CE", "up", "up"):     ("CALL BUYING",     "buying upside — bullish"),
    ("CE", "up", "down"):   ("CALL WRITING",    "selling upside — bearish / capping"),
    ("CE", "down", "up"):   ("CALL WRITERS COVERING", "call sellers buying back — bullish"),
    ("CE", "down", "down"): ("CALL LONGS EXITING",    "call buyers closing — bearish"),
    ("PE", "up", "up"):     ("PUT BUYING",      "buying downside — bearish"),
    ("PE", "up", "down"):   ("PUT WRITING",     "selling downside — bullish"),
    ("PE", "down", "up"):   ("PUT WRITERS COVERING",  "put sellers buying back — bearish"),
    ("PE", "down", "down"): ("PUT LONGS EXITING",     "put buyers closing — bullish"),
}

# What each option action implies for the STOCK. Single source of truth so the
# dashboard never has to re-derive direction from a label and get it backwards.
ACTION_LEAN = {
    "CALL BUYING": "bullish",           "PUT WRITING": "bullish",
    "CALL WRITERS COVERING": "bullish", "PUT LONGS EXITING": "bullish",
    "CALL WRITING": "bearish",          "PUT BUYING": "bearish",
    "PUT WRITERS COVERING": "bearish",  "CALL LONGS EXITING": "bearish",
}

_MIN_NOTIONAL_CR = 1.0      # ignore strikes below this notional — % moves there are noise
_PRIOR_CYCLES = 3           # "compared to the last 3 expiries", as specified
# Floor for the flow normaliser, in Rs Cr. OI flow is zero-inflated — most strikes
# build nothing on most days — so an unfloored denominator turns a Rs 2 Cr trade
# into a "20x". Below this we are dividing by noise, so we divide by the floor.
_FLOW_FLOOR_CR = 2.0
# Minimum money added for a strike to count as an event at all (Rs Cr). This is a
# FLOOR under the per-bucket gate below, not the gate itself.
_MIN_ADD_CR = 15.0
# The event bar is set per moneyness bucket, at this quantile of that bucket's own
# historical daily builds. A FLAT rupee gate is a liquidity filter wearing a size
# filter's clothes: ATM books are deep and clear it constantly while ITM books are
# thin and never do. Measured 2026-08-11 — under a flat Rs 15 Cr bar, ATM cleared
# on 10.43% of strike-days and took 38.9% of all hits, while ITM cleared 1.16% and
# deep ITM 0.66%; the scored list ran 21 ATM / 0 ITM in the top 25 and ZERO deep
# ITM overall, so the ITM-heavy case this tab was built for could never surface —
# and that was true DESPITE ITM carrying a 1.4x score weight.
# Each bucket's own p99 is: deep ITM 29.4, ITM 31.4, ATM 84.8, OTM 67.5,
# deep OTM 43.5 (Rs Cr). Note every one of those is ABOVE the old flat 15, so this
# is strictly stricter, not a loosening — it rebalances by raising ATM's bar, not
# by lowering ITM's. Hit share becomes deep ITM 2.8 / ITM 10.8 / ATM 25.0 /
# OTM 41.1 / deep OTM 20.2, and total events fall about 10x.
_BUCKET_GATE_Q = 0.99
# Share of a symbol's strikes claiming ALL their OI is new, above which the day is
# treated as a corporate-action ladder reset rather than positioning. Measured
# baseline is 0.0-0.7% on normal sessions vs 45.5% on a real adjustment, so the
# threshold sits far from both.
_CA_NEW_SHARE = 0.20
# ...but only on a chain wide enough for the share to mean anything (see the guard).
_CA_MIN_STRIKES = 20
# A contract this new has no positioning history to be abnormal against, so it is
# excluded — as a NEW LISTING, not mislabelled as a corporate action.
_NEW_LISTING_DAYS = 30


def _classify(oi_chg: float, px_chg: float, is_new: bool = False,
              option_type: str | None = None) -> tuple[str, str]:
    """
    `px_chg` is the CONTRACT'S OWN price change — the option premium for options,
    the futures price for futures. Pass `option_type` for options so the label
    names the act (writing vs buying) instead of borrowing futures wording that
    inverts on puts. See OPTION_ACTION_LABELS.
    """
    # A strike NSE only listed today has no prior premium, so the OI/price read is
    # undefined — but that is not missing data, it is the most notable case there
    # is: a brand-new strike opening with size already in it. Label it as such
    # rather than "NO DATA", which reads like a fault.
    if is_new:
        return ("NEW STRIKE", "listed today, opened with size already in it")
    if oi_chg is None or px_chg is None or oi_chg != oi_chg or px_chg != px_chg:
        return ("NO DATA", "")
    if abs(oi_chg) < 1e-9:
        return ("FLAT OI", "no position change")
    _oi = "up" if oi_chg > 0 else "down"
    _px = "up" if px_chg >= 0 else "down"
    if option_type in ("CE", "PE"):
        return OPTION_ACTION_LABELS[(option_type, _oi, _px)]
    return ACTION_LABELS[(_oi, _px)]


def _moneyness_bucket(strike: float, spot: float, opt: str) -> str:
    """Bucket by how far the strike sits from spot, from the OPTION's point of view."""
    if not spot or spot <= 0 or strike != strike:
        return "n/a"
    m = strike / spot - 1.0
    if opt == "PE":
        m = -m                       # a low-strike put is OTM, mirror it
    if m <= -0.10:
        return "deep ITM"
    if m <= -0.02:
        return "ITM"
    if m < 0.02:
        return "ATM"
    if m < 0.10:
        return "OTM"
    return "deep OTM"


def get_operator_footprint(as_of_date: date,
                           min_notional_cr: float = _MIN_NOTIONAL_CR,
                           min_add_cr: float = _MIN_ADD_CR,
                           top_n: int = 25) -> dict:
    """
    Per-stock unusual F&O positioning as of `as_of_date`. Causal — reads only
    that session and prior ones.

    Returns {"ok", "as_of", "stocks": DataFrame, "strikes": DataFrame, "meta"}.
    `stocks` is one row per symbol with the futures read and option skew;
    `strikes` is the individual strike-level footprints that drove it.
    """
    out: dict = {"ok": False, "as_of": as_of_date}

    sess = query_dataframe(
        "SELECT max(trade_date) d FROM fno_bhavcopy WHERE trade_date <= ?", [as_of_date])
    if sess.empty or pd.isna(sess["d"].iloc[0]):
        out["error"] = "no F&O data on or before that date"
        return out
    d0 = pd.Timestamp(sess["d"].iloc[0]).date()
    out["as_of"] = d0

    # spot closes, to derive moneyness
    spot = query_dataframe(
        "SELECT symbol, close_price AS spot FROM daily_data "
        "WHERE trade_date = ? AND series IN ('EQ','SM','ST')", [d0])
    spot_map = dict(zip(spot["symbol"], spot["spot"])) if not spot.empty else {}

    # ── futures: one row per symbol per expiry, plus the day's change ──────────
    fut = query_dataframe(
        """
        SELECT symbol, expiry_date, close_price, open_interest, chg_in_oi,
               value_lacs, contracts
        FROM fno_bhavcopy
        WHERE trade_date = ? AND instrument = 'FUTSTK'
        """, [d0])
    fut_prev = query_dataframe(
        """
        SELECT symbol, expiry_date, close_price AS prev_close
        FROM fno_bhavcopy
        WHERE instrument = 'FUTSTK'
          AND trade_date = (SELECT max(trade_date) FROM fno_bhavcopy
                            WHERE trade_date < ? AND instrument='FUTSTK')
        """, [d0])
    if not fut.empty:
        fut = fut.merge(fut_prev, on=["symbol", "expiry_date"], how="left")
        fut["px_chg"] = fut["close_price"] - fut["prev_close"]
        # near month = the earliest live expiry
        fut = fut.sort_values(["symbol", "expiry_date"])
        near = fut.groupby("symbol", as_index=False).first()
        near["fut_action"] = [
            _classify(o, p)[0] for o, p in zip(near["chg_in_oi"], near["px_chg"])]
        near["fut_oi_pct_chg"] = np.where(
            (near["open_interest"] - near["chg_in_oi"]) > 0,
            near["chg_in_oi"] / (near["open_interest"] - near["chg_in_oi"]) * 100.0,
            np.nan)
    else:
        near = pd.DataFrame(columns=["symbol", "fut_action", "fut_oi_pct_chg",
                                     "open_interest", "chg_in_oi"])

    # ── options: today's strike-level book ────────────────────────────────────
    opt = query_dataframe(
        """
        SELECT symbol, expiry_date, option_type, strike_price, close_price,
               open_interest, chg_in_oi, value_lacs
        FROM fno_bhavcopy
        WHERE trade_date = ? AND instrument = 'OPTSTK' AND open_interest > 0
        """, [d0])
    if opt.empty:
        out["error"] = "no stock options that day"
        return out
    opt_prev = query_dataframe(
        """
        SELECT symbol, expiry_date, option_type, strike_price,
               close_price AS prev_close
        FROM fno_bhavcopy
        WHERE instrument = 'OPTSTK'
          AND trade_date = (SELECT max(trade_date) FROM fno_bhavcopy
                            WHERE trade_date < ? AND instrument='OPTSTK')
        """, [d0])
    opt = opt.merge(opt_prev, on=["symbol", "expiry_date", "option_type", "strike_price"],
                    how="left")

    opt["spot"] = opt["symbol"].map(spot_map)
    opt["moneyness"] = [
        _moneyness_bucket(s, sp, t)
        for s, sp, t in zip(opt["strike_price"], opt["spot"], opt["option_type"])]
    opt["notional_cr"] = opt["open_interest"] * opt["spot"] / 1e7
    opt["prem_chg"] = opt["close_price"] - opt["prev_close"]
    # new strike = no row yesterday, so all of today's OI is "added"
    opt["is_new"] = opt["prev_close"].isna() & (opt["chg_in_oi"] >= opt["open_interest"] * 0.999)
    # ── CORPORATE-ACTION GUARD ────────────────────────────────────────────────
    # On an ex-dividend/bonus/split adjustment NSE RE-PRICES THE WHOLE LADDER, so
    # every strike looks brand new and every position looks like it arrived today.
    # It is the same money with a new contract identity, and `chg_in_oi` equals
    # `open_interest` only because the predecessor contract no longer exists to
    # difference against. Real case: INDUSTOWER on 2026-08-10 went ex a large
    # dividend (spot -4.66%), its 143 strikes moved from every strike price being
    # a multiple of 5 to every one being 1 mod 5, and 45.5% of them claimed all
    # their OI was new — against 0.0-0.7% on the surrounding sessions. It ranked
    # #1 on the tab as "NEW STRIKE, Rs 104 Cr". Separation is huge, so a simple
    # share test is enough. The whole symbol-day is dropped, not just the flagged
    # strikes: once identities change, "money that arrived today" is not
    # measurable for that name at all.
    # The share test needs a MINIMUM CHAIN WIDTH. On a symbol with 6 listed strikes
    # a single genuinely new one is 17% and two are 33%, so an illiquid name trips
    # the threshold without any corporate action: measured over a year, 20 thin
    # symbol-days (<20 strikes) would have been falsely suppressed. A real ladder
    # reset re-prices the WHOLE chain, so it can only be diagnosed on a wide one.
    _new_share = opt.groupby("symbol")["is_new"].transform("mean")
    _chain_n = opt.groupby("symbol")["strike_price"].transform("size")
    # A YOUNG LISTING looks identical to a ladder reset — every strike is new
    # because the contract only just started trading. Real examples: RADICO and
    # GVT&D both first appear 2026-05-27 with 100% of a 15- and 9-strike chain
    # "new", and GVT&D again on 05-29 with 24 of 33 new as its chain filled out.
    # Suppressing them is right (there is no baseline to measure flow against) but
    # calling it a corporate action would be a false explanation, so the two are
    # separated and reported apart.
    _first = query_dataframe(
        """SELECT symbol, MIN(trade_date) AS first_seen
           FROM fno_bhavcopy WHERE instrument = 'OPTSTK' GROUP BY symbol""")
    _fmap = dict(zip(_first["symbol"], pd.to_datetime(_first["first_seen"])))
    _age = opt["symbol"].map(
        lambda sy: (pd.Timestamp(d0) - _fmap[sy]).days if sy in _fmap else 9999)
    opt["new_listing"] = _age < _NEW_LISTING_DAYS
    opt["corp_action"] = ((_new_share > _CA_NEW_SHARE)
                          & (_chain_n >= _CA_MIN_STRIKES)
                          & ~opt["new_listing"])
    opt.loc[opt["corp_action"], "is_new"] = False
    _acts = [_classify(o, p, bool(n), t) for o, p, n, t
             in zip(opt["chg_in_oi"], opt["prem_chg"], opt["is_new"], opt["option_type"])]
    opt["action"] = [a[0] if isinstance(a, tuple) else a for a in _acts]
    opt["action_note"] = [a[1] if isinstance(a, tuple) else "" for a in _acts]
    opt["oi_prev"] = opt["open_interest"] - opt["chg_in_oi"]
    opt["oi_pct_chg"] = np.where(opt["oi_prev"] > 0,
                                 opt["chg_in_oi"] / opt["oi_prev"] * 100.0, np.nan)

    # ── the normaliser: same moneyness bucket, same days-to-expiry, prior cycles ─
    # DTE is BUCKETED, not exact. Matching exact days-to-expiry across cycles is
    # far too sparse — prior cycles then contribute 1-2 observations and the ratio
    # explodes (127x, 148x readings that mean nothing). Weekly buckets give a
    # populated comparison while still respecting that OI builds through a cycle.
    opt["dte"] = (pd.to_datetime(opt["expiry_date"]) - pd.Timestamp(d0)).dt.days
    # A contract settling tonight cannot carry a position. Its OI moves are
    # settlement mechanics (on a recent expiry, 9,151 of 15,693 strikes had
    # FALLING OI vs 958 rising). The flow score already ignores falling OI, which
    # is why only ~1 of 400 scored strikes was ever an expiring one, but excluding
    # them outright removes the category rather than relying on that.
    # ── ROLL DETECTION — must run BEFORE the dte>0 filter below, because the leg
    # being unwound is usually the contract expiring tonight.
    # A roll is the same position moving forward: OI falls in a nearer contract and
    # rises in a later one at the SAME stock/side/strike. It is not someone opening
    # a new view. Audited 2026-08-11 over 6 expiry weeks: 31-53% of flagged
    # next-month builds had exactly this signature. Routine roll VOLUME is already
    # absorbed by the DTE-matched baseline (expiry-week multiples 6.1x vs 5.3-6.3x
    # mid-cycle), so this is not about suppressing them — it is so a roll is not
    # read as fresh conviction. Flagged, not dropped: an operator who rolls forward
    # is choosing to STAY, which is itself worth seeing.
    # (A stock-level version of this test was tried and thrown away — it read ~100%
    # every day, because in expiry week every stock unwinds its whole front month.)
    _k = ["symbol", "option_type", "strike_price"]
    _self = opt[_k + ["dte", "chg_in_oi"]]
    _pairs = _self.merge(_self, on=_k, suffixes=("", "_o"))
    _pairs = _pairs[_pairs["dte_o"] < _pairs["dte"]]          # nearer contract only
    if not _pairs.empty:
        _pairs["_unwound"] = (-_pairs["chg_in_oi_o"]).clip(lower=0)
        _unw = (_pairs.groupby(_k + ["dte"])["_unwound"].sum().reset_index())
        opt = opt.merge(_unw, on=_k + ["dte"], how="left")
    else:
        opt["_unwound"] = 0.0
    opt["roll_cover"] = (opt["_unwound"].fillna(0.0)
                         / opt["chg_in_oi"].clip(lower=1))
    # A roll requires a BUILD to roll into. Without the chg_in_oi>0 term the clip
    # makes the denominator 1 for any strike whose OI fell, so a nearby unwind
    # produced an enormous roll_cover and stamped ROLLED on a position that was
    # being closed, not carried forward.
    opt["is_roll"] = (opt["roll_cover"] >= 0.5) & (opt["chg_in_oi"] > 0)

    opt = opt[opt["dte"] > 0].copy()
    opt["dte_b"] = (opt["dte"] // 7).clip(0, 12)
    hist = query_dataframe(
        f"""
        WITH base AS (
            SELECT o.trade_date, o.symbol, o.option_type, o.strike_price,
                   o.expiry_date, o.open_interest, o.chg_in_oi,
                   date_diff('day', o.trade_date, o.expiry_date) AS dte,
                   d.close_price AS spot
            FROM fno_bhavcopy o
            LEFT JOIN daily_data d
                   ON d.symbol = o.symbol AND d.trade_date = o.trade_date
                  AND d.series IN ('EQ','SM','ST')
            WHERE o.instrument = 'OPTSTK' AND o.open_interest > 0
              AND o.trade_date < ? AND o.trade_date >= ?::date - 400
              -- COMPLETED cycles only. Without this the window also contains rows
              -- from contracts that have not expired yet, and since `keep` below
              -- takes the LATEST expiries those live contracts won the selection:
              -- measured 2026-08-11, 42% of every baseline was built from expiries
              -- still trading, i.e. from the current cycle's own earlier days. A
              -- position an operator spent last week building then sat INSIDE the
              -- "normal" it was being compared against, suppressing the very thing
              -- this tab exists to find. "Last 3 expiries" must mean 3 that ENDED.
              AND o.expiry_date < ?
        )
        SELECT symbol, option_type, dte, expiry_date,
               strike_price, open_interest, chg_in_oi, spot
        FROM base
        """, [d0, d0, d0])
    prior = pd.DataFrame()
    bucket_gate = None          # per-moneyness event bar; None -> flat min_add_cr
    if not hist.empty:
        hist["moneyness"] = [
            _moneyness_bucket(s, sp, t)
            for s, sp, t in zip(hist["strike_price"], hist["spot"], hist["option_type"])]
        # keep the last _PRIOR_CYCLES COMPLETED expiries only (query filters
        # expiry_date < d0, so the latest ones here are genuinely finished cycles)
        keep = (hist.groupby("symbol")["expiry_date"]
                    .apply(lambda s: sorted(s.unique())[-_PRIOR_CYCLES:]).to_dict())
        hist = hist[[e in keep.get(sym, []) for sym, e in
                     zip(hist["symbol"], hist["expiry_date"])]]
        hist["dte_b"] = (hist["dte"] // 7).clip(0, 12)
        prior = (hist.groupby(["symbol", "option_type", "moneyness", "dte_b"])
                     ["open_interest"].agg(["median", "std", "count"])
                     .reset_index()
                     .rename(columns={"median": "oi_norm", "std": "oi_sd", "count": "n_obs"}))
        # A normaliser built on one or two observations is not a normaliser.
        prior = prior[prior["n_obs"] >= 5]
        # FLOW normaliser, in RUPEES: what a day's OI build is normally worth in
        # this bucket. Needed because the LEVEL ratio below cannot select events
        # (see _MIN_ADD_CR) — this is what the multiple is actually quoted from.
        hist["_add_cr"] = hist["chg_in_oi"].clip(lower=0) * hist["spot"] / 1e7
        _flow = (hist[hist["_add_cr"] > 0]
                 .groupby(["symbol", "option_type", "moneyness", "dte_b"])["_add_cr"]
                 .agg(["median", "count"]).reset_index()
                 .rename(columns={"median": "add_norm_cr", "count": "n_flow"}))
        prior = prior.merge(_flow[_flow["n_flow"] >= 5],
                            on=["symbol", "option_type", "moneyness", "dte_b"], how="left")
        # Per-bucket event bar, from PRIOR sessions only so it is causal — never
        # from today's cross-section, which would let the day set its own bar.
        _hl = hist[(hist["_add_cr"] > 0)
                   & (hist["open_interest"] * hist["spot"] / 1e7 >= min_notional_cr)]
        bucket_gate = (_hl.groupby("moneyness", observed=True)["_add_cr"]
                          .quantile(_BUCKET_GATE_Q))

    if not prior.empty:
        opt = opt.merge(prior, on=["symbol", "option_type", "moneyness", "dte_b"], how="left")
        # A ratio measured against a near-empty base is arithmetic, not a signal.
        # Measured 2026-08-10: the 5% of strikes whose normaliser was worth under
        # Rs 1 Cr had a median 38.4x vs 4.3x for everything else — a 9x reading
        # produced entirely by dividing by almost nothing. If we cannot establish
        # what "normal" is, we do not quote a multiple; those strikes fall back to
        # a neutral 1.0 and are ranked on size and flow alone.
        opt["norm_cr"] = opt["oi_norm"] * opt["spot"] / 1e7
        _norm_ok = (opt["oi_norm"] > 0) & (opt["norm_cr"] >= 1.0)
        opt["oi_vs_norm"] = np.where(_norm_ok,
                                     opt["open_interest"] / opt["oi_norm"], np.nan)
        opt["norm_thin"] = (opt["oi_norm"] > 0) & (opt["norm_cr"] < 1.0)
        # THE MULTIPLE THAT GETS QUOTED — money added today against what a day's
        # build is normally worth here, with the baseline FLOORED in rupees.
        # Audited 2026-08-11: an unfloored ratio cannot select. 69.3% of all "10x"
        # readings sat on a baseline under Rs 0.5 Cr and represented a median
        # Rs 2.3 Cr trade, because OI flow is zero-inflated (median daily build per
        # strike is Rs 0.0 Cr) so the denominator collapses and any ordinary print
        # divides into a spectacular number. Flooring at _FLOW_FLOOR_CR means a
        # multiple can only get large when the money is genuinely large.
        opt["add_cr"] = opt["chg_in_oi"].clip(lower=0) * opt["spot"] / 1e7
        if "add_norm_cr" not in opt.columns:
            opt["add_norm_cr"] = np.nan
        # Only quote a multiple where a real baseline exists. If the bucket's normal
        # daily build is itself under the floor we never established what normal is,
        # and add_cr/floor would just be the rupee figure wearing a ratio costume
        # (measured: the top strikes all came out at exactly 2.0x = money/floor).
        # Those fall back to a neutral 1.0 and rank on money alone.
        _flow_ok = opt["add_norm_cr"].notna() & (opt["add_norm_cr"] >= _FLOW_FLOOR_CR)
        opt["add_vs_norm"] = np.where(_flow_ok,
                                      opt["add_cr"] / opt["add_norm_cr"], np.nan)
        opt["flow_thin"] = ~_flow_ok
        opt["oi_z"] = np.where((opt["oi_sd"] > 0) & (opt["n_obs"] >= 2),
                               (opt["open_interest"] - opt["oi_norm"]) / opt["oi_sd"], np.nan)
    else:
        opt["oi_vs_norm"] = np.nan; opt["oi_z"] = np.nan
        opt["oi_norm"] = np.nan; opt["n_obs"] = 0
        opt["add_cr"] = opt["chg_in_oi"].clip(lower=0) * opt["spot"] / 1e7
        opt["add_vs_norm"] = np.nan; opt["norm_thin"] = False; opt["flow_thin"] = True

    # share of the stock's own option book — size relative to itself
    tot = opt.groupby("symbol")["open_interest"].transform("sum")
    opt["book_share_pct"] = np.where(tot > 0, opt["open_interest"] / tot * 100.0, np.nan)

    liq = opt[opt["notional_cr"] >= min_notional_cr].copy()

    # ── footprint score. ITM carries more weight: ordinary retail flow is OTM,
    # so real money appearing ITM is the harder-to-explain event.
    _ITM_W = {"deep ITM": 1.6, "ITM": 1.4, "ATM": 1.0, "OTM": 0.7, "deep OTM": 0.5}
    liq["itm_w"] = liq["moneyness"].map(_ITM_W).fillna(1.0)
    # Score FLOW, not standing level. Ranking by how large the book already is just
    # surfaces the biggest strikes every day (the first build of this scored
    # "FLAT OI" positions top — no activity at all). The question is what CHANGED,
    # so the driver is today's OI ADD, sized against the stock's own book, with the
    # level-vs-normal ratio as a multiplier and ITM weighted up.
    liq["oi_add_share_pct"] = np.where(
        tot.reindex(liq.index) > 0,
        liq["chg_in_oi"].clip(lower=0) / tot.reindex(liq.index) * 100.0, 0.0)
    # per-bucket event bar — needed by BOTH the ranking and the gate below
    liq["gate_cr"] = (liq["moneyness"].map(bucket_gate)
                      if bucket_gate is not None else np.nan)
    liq["gate_cr"] = liq["gate_cr"].fillna(min_add_cr).clip(lower=min_add_cr)
    #
    # RANKED ON MONEY, NOT ON A RATIO. The standing-level ratio (`oi_vs_norm`) used
    # to be a multiplier here and it was the wrong selector: audited over 1.77M
    # strike-days, it printed >=10x on 5.5% of them, fired for RELIANCE/GAIL/CANBK on
    # ~every one of 210 sessions, and the median strike reading >=10x had Rs 0.0 Cr
    # of fresh OI. It measures how skewed a bucket is, not that anything happened.
    # What did survive audit: gate on ABSOLUTE RUPEES ADDED and the positions stick —
    # 80.1% of Rs 25 Cr+/10x events still held half the added OI 5 days later vs
    # 66.3% for level-matched non-events, and that comparison is biased toward the
    # control. Size is damped with log1p because the very largest prints are the
    # LEAST sticky (median retention Rs 15-25 Cr 1.33 -> Rs 250 Cr+ 0.91) — they are
    # rolls and spreads, not accumulation, so "biggest" must not mean "top".
    # Ranked on money EXPRESSED IN ITS OWN BUCKET'S UNITS. Using raw rupees put 21
    # of the top 25 in ATM and none in ITM even after the per-bucket gate let ITM
    # through, because an ATM book is simply bigger — the gate fixed eligibility
    # but the ranking still measured depth. `gate_cr` is a large, stable, causally
    # derived per-bucket constant (Rs 20-85 Cr), so dividing by it is a change of
    # units, not the degenerate per-strike ratio that broke the original metric.
    liq["add_vs_gate"] = liq["add_cr"] / liq["gate_cr"].clip(lower=1e-9)
    # `add_vs_norm` is DELIBERATELY NOT IN THE SCORE. It is undefined for an entire
    # subpopulation — measured 2026-08-11, 100% of ITM and deep-ITM strikes are
    # flow_thin because their books are too quiet to establish a per-strike
    # baseline, so they all fell back to a neutral 1.0 while OTM carried a median
    # 11.37x. That is an ~11x ranking handicap applied for being thin, and it
    # buried the exact case this tab exists for: PAYTM CE 1540 (ITM, spot 1584)
    # with Rs 126.6 Cr added scored 2.8 against top scores near 40. Scale is
    # already handled by add_vs_gate, which is defined everywhere. The per-strike
    # multiple stays as a DISPLAYED descriptor.
    liq["footprint"] = (
        np.log1p(liq["add_vs_gate"].clip(lower=0))                # money in bucket units
        * liq["itm_w"]                                            # ITM is the harder signal
        * (1.0 + liq["oi_add_share_pct"].clip(0, 25) / 25.0))     # share of its own book
    # a strike with no fresh OI is not a footprint, whatever its standing size
    liq.loc[liq["chg_in_oi"] <= 0, "footprint"] = 0.0
    # An event has to be worth real money to be an event at all. Without this the
    # list fills with Rs 2 Cr trades wearing a big multiple. The bar is per
    # moneyness bucket (see _BUCKET_GATE_Q) so a deep, busy ATM book and a thin ITM
    # book are each judged against their own scale, with min_add_cr as a hard floor.
    liq.loc[liq["add_cr"] < liq["gate_cr"], "footprint"] = 0.0
    # corporate-action symbol-days carry no measurable flow (see the guard above)
    liq.loc[liq["corp_action"] | liq["new_listing"], "footprint"] = 0.0
    liq = liq.sort_values("footprint", ascending=False)

    # ── per-symbol roll-up: which side is being built, and how one-sided is it ──
    def _side(g: pd.DataFrame) -> pd.Series:
        ce = g[g.option_type == "CE"]; pe = g[g.option_type == "PE"]
        ce_add = float(ce.loc[ce.chg_in_oi > 0, "chg_in_oi"].sum())
        pe_add = float(pe.loc[pe.chg_in_oi > 0, "chg_in_oi"].sum())
        tot_add = ce_add + pe_add
        top = g.nlargest(1, "footprint")
        return pd.Series({
            "call_oi_added": ce_add, "put_oi_added": pe_add,
            "call_share_of_adds_pct": (100.0 * ce_add / tot_add) if tot_add > 0 else np.nan,
            "top_strike": float(top["strike_price"].iloc[0]) if len(top) else np.nan,
            "top_type": top["option_type"].iloc[0] if len(top) else "",
            "top_expiry": top["expiry_date"].iloc[0] if len(top) else None,
            "top_moneyness": top["moneyness"].iloc[0] if len(top) else "",
            "top_action": top["action"].iloc[0] if len(top) else "",
            "top_oi_vs_norm": float(top["oi_vs_norm"].iloc[0]) if len(top) else np.nan,
            "top_is_roll": bool(top["is_roll"].iloc[0]) if len(top) else False,
            "top_add_cr": float(top["add_cr"].iloc[0]) if len(top) else np.nan,
            "top_add_vs_norm": float(top["add_vs_norm"].iloc[0]) if len(top) else np.nan,
            "top_notional_cr": float(top["notional_cr"].iloc[0]) if len(top) else np.nan,
            "footprint": float(top["footprint"].iloc[0]) if len(top) else np.nan,
            # Count EVENTS (money that actually showed up), not strikes sitting
            # above a bucket median — the old `oi_vs_norm >= 2` counted 46% of the
            # universe and so was not a count of anything unusual.
            "n_unusual": int((g["footprint"] > 0).sum()),
        })

    stocks = liq.groupby("symbol").apply(_side).reset_index() if not liq.empty else pd.DataFrame()
    if not stocks.empty and not near.empty:
        stocks = stocks.merge(
            near[["symbol", "fut_action", "fut_oi_pct_chg", "open_interest", "chg_in_oi"]]
            .rename(columns={"open_interest": "fut_oi", "chg_in_oi": "fut_oi_chg"}),
            on="symbol", how="left")
    if not stocks.empty:
        stocks["spot"] = stocks["symbol"].map(spot_map)
        # Deliberately NOT truncated to top_n here. The caller needs the full
        # ranked list so a price filter can be applied BEFORE the cut — filtering
        # an already-truncated top 25 would leave three names. The scan takes ~20s,
        # so it is cached whole and sliced by the UI; making price a cache key
        # would re-run the scan on every slider nudge.
        stocks = stocks.sort_values("footprint", ascending=False)

    out.update(ok=True, stocks=stocks, strikes=liq.head(400),
               meta={"n_symbols": int(opt["symbol"].nunique()),
                     "n_strikes_liquid": int(len(liq)),
                     "prior_cycles": _PRIOR_CYCLES,
                     "min_notional_cr": min_notional_cr,
                     "min_add_cr": min_add_cr,
                     "n_events": int((liq["footprint"] > 0).sum()),
                     "corp_action_symbols": sorted(
                         liq.loc[liq["corp_action"], "symbol"].unique().tolist()),
                     "new_listing_symbols": sorted(
                         liq.loc[liq["new_listing"], "symbol"].unique().tolist()),
                     "has_norm": bool(not prior.empty)})
    return out
