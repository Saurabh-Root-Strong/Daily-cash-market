"""
Bulk / block deal analytics.

WHAT THE DATA IS
    NSE discloses a BULK deal when a single client trades > 0.5% of a company's
    listed shares in a day (both legs of that client's day are reported), and a
    BLOCK deal when a trade of >= Rs 10 Cr / 5 lakh shares is put through the
    separate block window (BOTH counterparties are named, so a block deal is
    two-sided by construction -- 99.7% of block symbol-days net to exactly zero
    quantity. "Net buying" is not a meaningful concept for block deals).

WHAT IT PREDICTS  (measured, not assumed -- see scripts/build_deal_outcomes.py)
    Over 2018-2026 the forward 1-2 month excess return after a bulk deal is
    NEGATIVE, and after a block deal is nothing. The excess is measured against
    matched non-event stock-days: same session, same liquidity quintile, same
    volume-spike decile, same same-day-move decile -- because deal days are big
    volume, big move days and those revert on their own. A placebo drawn from
    that same control pool prints -0.00% (t = -0.16).

    The naive reading -- "a big buyer stepped in, that is bullish" -- is wrong in
    the data. A net BUY is among the worst cases (-1.9% / 21d, -3.1% / 42d) and a
    same-day round trip by one client is worse still (-2.3% / -3.9%; a quarter of
    bulk symbol-days, mostly HFT market makers). Only mutual-fund / insurance
    participation prints positive, on a small sample that fades by 42 sessions.

    A 44-candidate reality check on this search returns p = 0.0025. That null is
    drawn from control days, whose 1-month dispersion is 0.79x the events' own, so
    read the p as optimistic by roughly that factor -- the headline still clears,
    the marginal cells do not.

ROBUSTNESS (audited, not assumed)
    - survives the benchmark definition: with NO same-day-move control the effect is
      LARGER (-2.9% / 21d), so it is not an artifact of over-conditioning on a
      post-treatment variable
    - survives clustering: one event per stock per quarter -1.7% (t -4.2); collapsed
      to one number per stock -2.2% (t -5.7, clustered by symbol)
    - holds in RAW terms, not just relative: bulk net-BUY stocks average -2.1% over
      the next month (median -3.5%, only 41% up) while the market averaged +1.0%
    - weakens where the matched cell is well populated (n_ctrl >= 20: -1.1%, t -2.0),
      so treat the magnitude as the softer number and the sign as the firm one

CAN YOU FOLLOW A SPECIFIC BUYER?  (scripts/build_client_events.py)
    No. Tested over 15,671 entity-buy events / 4,062 entities at 1wk, 2wk, 1mo,
    2mo and 3mo:
      - a naive leaderboard looks superb (top entity +27% per buy) -- that is what
        ranking ~94 eligible entities on noisy returns produces
      - WALK-FORWARD, ranking each entity only on its own already-resolved earlier
        buys, the best-record quartile then returns -1.1% / 1wk, -1.6% / 1mo,
        -2.4% / 2mo. Every quartile is negative; the information coefficient decays
        to +0.006 (t = 0.26) once 20 prior buys are required
      - demanding a prior record above +3% makes the next buys WORSE (-2.1% / 1mo,
        -4.3% / 3mo): the ranking mean-reverts, the signature of luck
      - MF / insurance is the only positive buyer type (+1.2% / 1mo, t = 2.94) but
        it fails a 35-candidate permutation (null median best |t| 4.40 > observed
        3.54, p = 0.99), its median buy is -1.1%, and only 37% beat their peers
      - FPI / foreign buying is NOT positive at any horizon
      - repeat buying by the same entity in the same name within 3 months is the
        most negative pattern in the whole dataset: -7.5% over 3 months

HOW TO USE IT
    As an AVOID / TRIM flag, not a short. Most bulk-deal names have no SLB or F&O
    leg, so the negative side is not directly harvestable; it holds in the most
    liquid tier too (> Rs 200 Cr turnover: -1.8% / 21d, t = -2.9).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "classify_client", "bucket_label", "CLIENT_BUCKETS",
    "load_deals", "load_deal_flow", "load_outcomes",
    "evidence_table", "evidence_by_size", "evidence_by_client",
    "client_track_record", "recent_events",
    "load_client_events", "entity_type_table", "walk_forward_table",
    "repeat_buy_table", "entity_leaderboard", "horizon_coverage",
    "client_trades", "HORIZON_COLS",
]

# -- Client classification ----------------------------------------------------
# Ordered: the first matching bucket wins. HFT names are checked before the
# generic broker keywords because "NK SECURITIES RESEARCH" is a market maker,
# not a retail broker, and its round trips are the single largest block of rows.

_HFT = ("GRAVITON", "HRTI", "TOWER RESEARCH", "XTX", "JUMP TRADING", "CITADEL",
        "ALPHAGREP", "QE SECURITIES", "NK SECURITIES", "CROSSLAND", "IRAGE",
        "MATHISYS", "AAKRAYA", "MICROCURVES", "JUNOMONETA", "SHAASTRA",
        "SURJECTIVE", "MUSIGMA", "OLGA TRADING", "SHARE INDIA SECURITIES",
        "JAINAM", "SETU SECURITIES", "MANSI SHARE", "MANSI SHARES",
        "MULTIPLIER SHARE", "ADROIT FINANCIAL", "PARTH INFIN", "SKSE",
        "L7 HITECH", "SW CAPITAL", "QUADEYE", "OPTIVER", "DRW", "HAP TRADING")
_MF = ("MUTUAL FUND", "MUTUALFUND", " MF ", "ASSET MANAGEMENT", "AMC",
       "LIFE INSURANCE", "INSURANCE COMPANY", "GENERAL INSURANCE", "PENSION",
       "PROVIDENT", "UNIT TRUST", "SBI FUNDS", "ICICI PRUDENTIAL", "HDFC TRUSTEE",
       "NIPPON LIFE", "MIRAE", "DSP ", "FRANKLIN", "ADITYA BIRLA SUN LIFE",
       "AXIS MUTUAL", "KOTAK MAHINDRA MUTUAL", "UTI ", "QUANT MUTUAL", "WHITEOAK",
       "BANDHAN MUTUAL", "INVESCO", "TRUSTEE")
_FPI = ("MAURITIUS", "SINGAPORE", "CYPRUS", "LUXEMBOURG", "CAYMAN", "MASTER FUND",
        "OFFSHORE", "GLOBAL FUND", "GOLDMAN SACHS", "MORGAN STANLEY",
        "CITIGROUP GLOBAL", "SOCIETE GENERALE", "BOFA", "MERRILL", "JPMORGAN",
        "J P MORGAN", "NOMURA", "HSBC", "BARCLAYS", "ODI", "GOVERNMENT PENSION",
        "ABU DHABI", "VANGUARD", "BLACKROCK", "FIDELITY", "EASTSPRING",
        "COPTHALL", "ELARA", "APMS", "LTS INVESTMENT", "VIKASA", "MINERVA VENTURES")
_BROKER = ("SECURITIES", "BROKING", "BROKERS", "STOCK ADVISORS", "CAPITAL MARKETS",
           "RESEARCH CAPITAL", "FINANCIAL SERVICES", "WEALTH")
_CORP = ("PRIVATE LIMITED", "PVT LTD", "LIMITED", "LLP", "ENTERPRISES", "TRADING",
         "VENTURES", "INFRA", "HOLDINGS", "VYAPAR", "CAPITAL", "TRUST", "FOUNDATION")

# Anything still carrying fund-shaped words after the named lists have had their
# turn. Without this, "SMALLCAP WORLD FUND INC" and "EMERGING MARKETS EQUITY FUND"
# fell through to INDIVIDUAL, which both understates fund activity and pollutes the
# individual bucket with foreign institutions.
_FUND = ("FUND", "FUNDS", "PORTFOLIO", "INVESTMENT TRUST", "ENDOWMENT",
         "RETIREMENT SYSTEM", "SOVEREIGN", "PLC", "PTE", "SICAV")

CLIENT_BUCKETS = ["MF_INSURANCE", "FPI_FOREIGN", "OTHER_FUND", "INDIVIDUAL",
                  "CORPORATE", "BROKER_PROP", "HFT_PROP"]

# Display labels use the trading-desk vocabulary. "Individual / HNI" is accurate
# rather than flattering: to appear here at all, a person had to trade more than
# 0.5% of a company's shares in one day, which is an HNI-sized ticket by definition.
_BUCKET_LABEL = {
    "MF_INSURANCE": "Mutual fund / Insurance",
    "OTHER_FUND":   "Other fund / PMS",
    "FPI_FOREIGN":  "FPI / Foreign",
    "INDIVIDUAL":   "Individual / HNI",
    "CORPORATE":    "Corporate / LLP",
    "BROKER_PROP":  "Broker prop desk",
    "HFT_PROP":     "HFT market maker",
}


def _compile(keywords: tuple[str, ...]) -> "re.Pattern[str]":
    """Word-boundary matcher.

    Plain substring matching is wrong here and was silently mislabelling people as
    institutions: "UTI " matches inside "SHRUTI  MOHTA", "DSP " inside a name
    ending in DSP, "AMC" inside "RAMCHANDRA". The worst single case put an
    individual's +81% trade at the top of the mutual-fund winners list.
    """
    parts = [rf"\b{re.escape(k.strip())}\b" for k in keywords]
    return re.compile("|".join(parts))


_HFT_RE = _compile(_HFT)
_MF_RE = _compile(_MF)
_FPI_RE = _compile(_FPI)
_BROKER_RE = _compile(_BROKER)
_CORP_RE = _compile(_CORP)
_FUND_RE = _compile(_FUND)


def classify_client(name: str) -> str:
    """Bucket a raw NSE client string. Keyword rules, so it is a label, not a fact."""
    u = (name or "").upper()
    if _HFT_RE.search(u):
        return "HFT_PROP"
    if _MF_RE.search(u):
        return "MF_INSURANCE"
    if _FPI_RE.search(u):
        return "FPI_FOREIGN"
    if _BROKER_RE.search(u):
        return "BROKER_PROP"
    if _FUND_RE.search(u):
        return "OTHER_FUND"
    if re.search(r"\bHUF\b", u):
        return "INDIVIDUAL"
    if _CORP_RE.search(u):
        return "CORPORATE"
    return "INDIVIDUAL"


def bucket_label(bucket: str) -> str:
    return _BUCKET_LABEL.get(bucket, bucket)


# -- Reads --------------------------------------------------------------------

def load_deals(trade_date: date, lookback_days: int = 0) -> pd.DataFrame:
    """Raw deal rows for a date (or a window ending on it), enriched with the
    event-day cash-market context needed to size the deal."""
    start = trade_date - timedelta(days=lookback_days)
    df = query_dataframe(
        """
        SELECT d.trade_date, d.deal_type, d.symbol, d.security_name, d.client_name,
               d.buy_sell, d.quantity, d.trade_price, d.remarks,
               p.close_price, p.ttl_trd_qnty, p.turnover_lacs, p.deliv_per
        FROM bulk_block_deals d
        LEFT JOIN daily_data p
               ON p.symbol = d.symbol AND p.trade_date = d.trade_date AND p.series = 'EQ'
        WHERE d.trade_date BETWEEN ? AND ?
        ORDER BY d.trade_date DESC, d.quantity * d.trade_price DESC
        """,
        [start, trade_date],
    )
    if df.empty:
        return df
    df["value_cr"] = df["quantity"] * df["trade_price"] / 1e7
    df["client_bucket"] = df["client_name"].map(classify_client)

    # A trade put through the BLOCK window that also crosses the 0.5% bulk
    # threshold is filed in BOTH feeds, identically. 1,724 such twins since 2018,
    # worth Rs 5.19 lakh Cr -- adding the two feeds up double-counts them. Mark
    # the twin so callers can total the value once.
    key = ["trade_date", "symbol", "client_name", "buy_sell", "quantity", "trade_price"]
    blocks = set(map(tuple, df.loc[df.deal_type == "BLOCK", key].itertuples(index=False, name=None)))
    twin = df[key].apply(tuple, axis=1).isin(blocks) if blocks else pd.Series(False, index=df.index)
    df["via_block_window"] = twin
    df["duplicate_of_block"] = twin & (df.deal_type == "BULK")

    # Block-window quantity settles OUTSIDE the regular market, so it is not part
    # of ttl_trd_qnty -- a block trade routinely prints more than 100% of the day's
    # "volume". Only regular-market lines get a share-of-volume figure.
    on_market = ~df["via_block_window"]
    df["pct_of_day_vol"] = pd.NA
    df.loc[on_market, "pct_of_day_vol"] = (
        100 * df.loc[on_market, "quantity"] / df.loc[on_market, "ttl_trd_qnty"].replace(0, pd.NA)
    )
    return df


def load_deal_flow(trade_date: date) -> pd.DataFrame:
    """Per symbol-day netting. Same-day BUY and SELL by the same client is an
    intraday round trip, not accumulation, so gross quantity is never the signal.

    Rows filed in both feeds are counted once (see load_deals), so the value
    totals here are real money, not the sum of two copies of the same trade.
    """
    df = load_deals(trade_date)
    if df.empty:
        return df
    df = df.copy()
    # Netting keeps EVERY bulk row, including the ones also filed as block, so the
    # NET BUY / NET SELL label here is the same event definition the base rates in
    # `deal_event_outcomes` were measured on. Only the money total is deduplicated.
    df["signed"] = df["quantity"].where(df["buy_sell"] == "BUY", -df["quantity"])
    df["value_cr_once"] = df["value_cr"].where(~df["duplicate_of_block"], 0.0)
    g = df.groupby(["trade_date", "symbol", "deal_type"], as_index=False).agg(
        security_name=("security_name", "first"),
        qty_net=("signed", "sum"),
        qty_gross=("quantity", "sum"),
        value_cr=("value_cr_once", "sum"),
        n_deals=("quantity", "size"),
        block_window_rows=("via_block_window", "sum"),
        n_clients=("client_name", "nunique"),
        vwap=("trade_price", "mean"),
        close_price=("close_price", "first"),
        ttl_trd_qnty=("ttl_trd_qnty", "first"),
        turnover_lacs=("turnover_lacs", "first"),
        deliv_per=("deliv_per", "first"),
        clients=("client_name", lambda s: " | ".join(sorted(set(s))[:4])),
        buckets=("client_bucket", lambda s: ", ".join(sorted({bucket_label(x) for x in s}))),
    )
    g["side"] = g["qty_net"].apply(
        lambda q: "NET BUY" if q > 0 else ("NET SELL" if q < 0 else "ROUND TRIP")
    )
    # Share-of-volume only where the quantity actually went through the regular
    # market; block-window quantity is not inside ttl_trd_qnty at all.
    onmkt = df[~df["via_block_window"]].groupby(
        ["trade_date", "symbol", "deal_type"], as_index=False)["signed"].sum()
    onmkt = onmkt.rename(columns={"signed": "qty_net_onmkt"})
    g = g.merge(onmkt, on=["trade_date", "symbol", "deal_type"], how="left")
    g["net_pct_vol"] = 100 * g["qty_net_onmkt"] / g["ttl_trd_qnty"].replace(0, pd.NA)
    g["turnover_cr"] = g["turnover_lacs"] / 100
    return g.sort_values("value_cr", ascending=False)


def load_outcomes() -> pd.DataFrame:
    """The measured event panel written by scripts/build_deal_outcomes.py.

    Returns an empty DataFrame if the builder has never run - every caller copes.
    """
    try:
        df = query_dataframe("SELECT * FROM deal_event_outcomes", [])
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# -- Measured evidence (computed live off the outcomes table) -----------------

def _nw_t(series, lag: int):
    """Newey-West t on a daily mean series. Forward windows overlap, so a plain
    t would be inflated several-fold."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 30:
        return float("nan"), float("nan"), n
    m = x.mean()
    e = x - m
    s = float(e @ e) / n
    for lag_i in range(1, lag + 1):
        s += 2 * (1 - lag_i / (lag + 1)) * (float(e[lag_i:] @ e[:-lag_i]) / n)
    se = (max(s, 1e-18) / n) ** 0.5
    return m, m / se, n


def _row(sub: pd.DataFrame, label: str) -> dict:
    m21, t21, _ = _nw_t(sub.groupby("trade_date")["m21"].mean(), 21)
    m42, t42, _ = _nw_t(sub.groupby("trade_date")["m42"].mean(), 42)
    return {"Event": label, "Events": len(sub),
            "1 month %": m21 * 100, "t (1m)": t21,
            "2 months %": m42 * 100, "t (2m)": t42,
            "Beat peers %": 100 * (sub["m21"] > 0).mean() if len(sub) else float("nan")}


def evidence_table(out: pd.DataFrame) -> pd.DataFrame:
    """Headline table: forward excess vs matched non-event days, by event kind."""
    if out.empty:
        return pd.DataFrame()
    bulk = out[out.is_bulk == 1]
    rows = [
        _row(bulk, "Any BULK deal"),
        _row(bulk[bulk.side == "BUY"], "BULK - net BUY"),
        _row(bulk[bulk.side == "SELL"], "BULK - net SELL"),
        _row(bulk[bulk.side == "FLAT"], "BULK - same-day round trip"),
        _row(out[(out.is_block == 1) & (out.is_bulk == 0)], "BLOCK deal (no bulk that day)"),
    ]
    return pd.DataFrame([r for r in rows if r["Events"] >= 50])


def evidence_by_size(out: pd.DataFrame) -> pd.DataFrame:
    """Does it only live in illiquid names? (It does not.)"""
    if out.empty:
        return pd.DataFrame()
    b = out[(out.is_bulk == 1) & (out.side == "BUY")].copy()
    b["cr"] = b["turnover_lacs"] / 100
    bands = [(0, 1, "< Rs 1 Cr"), (1, 10, "Rs 1-10 Cr"), (10, 50, "Rs 10-50 Cr"),
             (50, 200, "Rs 50-200 Cr"), (200, 1e12, "> Rs 200 Cr")]
    rows = [_row(b[(b.cr >= lo) & (b.cr < hi)], f"BULK net BUY - turnover {lab}")
            for lo, hi, lab in bands]
    return pd.DataFrame([r for r in rows if r["Events"] >= 50])


def evidence_by_client(out: pd.DataFrame) -> pd.DataFrame:
    """Who was on the buy side, and did it matter?"""
    if out.empty or "buy_buckets" not in out.columns:
        return pd.DataFrame()
    rows = []
    for b in CLIENT_BUCKETS:
        sub = out[out["buy_buckets"].fillna("").str.contains(b, regex=False)]
        if len(sub) >= 80:
            rows.append(_row(sub, f"{bucket_label(b)} on the BUY side"))
    return pd.DataFrame(rows)


def client_track_record(out: pd.DataFrame, min_events: int = 25,
                        side: str = "BUY") -> pd.DataFrame:
    """Per-client forward record. Descriptive only: with hundreds of names ranked,
    the tails are what a random ranking produces too. Read it as "what happened
    after this name appeared", not as evidence that the name has skill."""
    if out.empty or "client_rows" not in out.columns:
        return pd.DataFrame()
    ex = out[["trade_date", "symbol", "m21", "m42", "client_rows"]].copy()
    # regex=False matters: "||" as a regex is an empty alternation and splits
    # every character, which silently produced an empty leaderboard.
    ex = ex.assign(
        client_rows=ex["client_rows"].fillna("").str.split("||", regex=False)
    ).explode("client_rows")
    parts = ex["client_rows"].str.split("::", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.DataFrame()
    ex["deal_side"] = parts[0]
    ex["client_name"] = parts[1]
    ex = ex[(ex["deal_side"] == side) & ex["client_name"].notna()]
    g = ex.groupby("client_name").agg(
        events=("m21", "size"),
        avg_1m=("m21", lambda s: s.mean() * 100),
        avg_2m=("m42", lambda s: s.mean() * 100),
        beat_pct=("m21", lambda s: 100 * (s > 0).mean()),
    ).reset_index()
    g = g[g.events >= min_events]
    if g.empty:
        return g
    g["bucket"] = g["client_name"].map(classify_client).map(bucket_label)
    return g.sort_values("avg_1m", ascending=False)


# -- Entity-level: does any specific buyer's name predict anything? -----------

HORIZON_COLS = {"m5": "1 week", "m10": "2 weeks", "m21": "1 month",
                "m42": "2 months", "m63": "3 months"}
_HLAG = {"m5": 5, "m10": 10, "m21": 21, "m42": 42, "m63": 63}


def load_client_events(side: str = "BUY") -> pd.DataFrame:
    """Per (entity, stock, day) events from scripts/build_client_events.py."""
    try:
        df = query_dataframe(
            "SELECT * FROM deal_client_events WHERE side = ?", [side])
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["bucket"] = df["client_name"].map(classify_client)
    return df


def _hrow(sub: pd.DataFrame, label: str) -> dict:
    row = {"Group": label, "Buys": len(sub)}
    for col, name in HORIZON_COLS.items():
        if col not in sub.columns:
            continue
        m, t, _ = _nw_t(sub.groupby("trade_date")[col].mean(), _HLAG[col])
        row[name] = m * 100
        row[f"t {name}"] = t
    return row


def entity_type_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Forward excess by WHO bought, across every horizon."""
    if ev.empty:
        return pd.DataFrame()
    rows = [_hrow(ev[ev.bucket == b], bucket_label(b)) for b in CLIENT_BUCKETS
            if (ev.bucket == b).sum() >= 150]
    rows.append(_hrow(ev, "ALL buyers"))
    return pd.DataFrame(rows)


def walk_forward_table(ev: pd.DataFrame, min_prior: int = 5) -> pd.DataFrame:
    """THE test: rank entities on their record so far, then measure only their
    LATER buys. If "follow the buyer with the good history" worked, the best
    quartile would be positive here. It is not."""
    if ev.empty or "prior_mean" not in ev.columns:
        return pd.DataFrame()
    w = ev[(ev.prior_n >= min_prior) & ev.prior_mean.notna()].copy()
    if len(w) < 500:
        return pd.DataFrame()
    w["pq"] = w.groupby(w.trade_date.dt.to_period("Q"))["prior_mean"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4,
                          labels=["worst 25%", "Q2", "Q3", "best 25%"])
        if s.notna().sum() >= 8 else None)
    order = ["best 25%", "Q3", "Q2", "worst 25%"]
    return pd.DataFrame([_hrow(w[w.pq == q], f"prior record {q}")
                         for q in order if (w.pq == q).sum() >= 100])


def entity_leaderboard(ev: pd.DataFrame, min_events: int = 15) -> pd.DataFrame:
    """Every named client with enough buys, at all five horizons.

    Shows MEDIAN and hit-rate next to the mean on purpose. A name can average +26%
    on a median of -0.5%, which means one or two lottery tickets carried it and the
    typical trade lost - the single most common way a leaderboard like this is
    misread. Descriptive: see walk_forward_table() for whether any of it is usable.
    """
    if ev.empty:
        return pd.DataFrame()
    rows = []
    for key, g in ev.groupby("client_key"):
        if len(g) < min_events:
            continue
        row = {
            "client_key": key,
            "Client": g["client_name"].mode().iat[0],
            "Type": bucket_label(classify_client(g["client_name"].iat[0])),
            "Buys": len(g),
            "Stocks": g["symbol"].nunique(),
            # Distinct SESSIONS, which is not the same as Buys: a desk that bought
            # 12 stocks on one morning has 12 buys but one independent observation.
            # Without this column a 40-buy record looks 40x more solid than it is.
            "Days": g["trade_date"].nunique(),
            "First buy": g["trade_date"].min(),
            "Last buy": g["trade_date"].max(),
        }
        # Move since purchase, to the latest close. Unlike the horizon columns it is
        # never blank, so a 10-session window still says how the buys are doing. It
        # is RAW, not excess: in a rising market everyone looks right, which is why
        # the Nifty comparison sits beside it. Median, not mean -- one multibagger
        # bought years ago would otherwise swamp every recent trade.
        if "since_pct" in g.columns and g["since_pct"].notna().any():
            sp = g["since_pct"]
            row["Since buy median"] = sp.median() * 100
            row["In profit now %"] = 100 * sp.gt(0).sum() / int(sp.notna().sum())
            row["vs Nifty median"] = ((sp - g["nifty_since_pct"]).median() * 100
                                      if "nifty_since_pct" in g.columns else float("nan"))
        else:
            row["Since buy median"] = row["In profit now %"] = float("nan")
            row["vs Nifty median"] = float("nan")
        # Did the purchase plausibly SETTLE? Only judgeable for BULK: a BLOCK trade
        # is negotiated outside the regular market and never enters deliv_qty, so
        # scoring block buyers on delivery would condemn every placement.
        bulk = g[g["deal_type"] == "BULK"]
        if "pct_of_delivery" in g.columns and len(bulk):
            fits = bulk["pct_of_delivery"].le(100)
            row["Settled %"] = 100 * fits.mean()
            row["Median % of delivery"] = bulk["pct_of_delivery"].median()
        else:
            row["Settled %"] = float("nan")
            row["Median % of delivery"] = float("nan")
        if "exit_days" in g.columns:
            row["Sold back %"] = 100 * g["exit_days"].le(90).mean()
        else:
            row["Sold back %"] = float("nan")

        for col, name in HORIZON_COLS.items():
            row[name] = g[col].mean() * 100
            row[f"{name} median"] = g[col].median() * 100
            # How many of those buys actually have a closed window. Without this a
            # single resolved trade renders as a confident-looking average.
            row[f"{name} n"] = int(g[col].notna().sum())
        # dropna FIRST. (NaN > 0) is False, so an unresolved window counted as a
        # LOSS: on the default 10-session view -- where nothing can have resolved --
        # every client displayed a confident "Hit rate 0%", which reads as "this desk
        # never wins" when the truth is "not known yet". Blank is the honest answer.
        m21r = g["m21"].dropna()
        row["Hit rate 1 month"] = (m21r > 0).mean() * 100 if len(m21r) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sort by 1 month when it exists, but a narrow or recent date filter can leave
    # that column entirely unresolved -- ordering by an all-blank column is useless,
    # so fall back to the shortest horizon that has actually closed, then to size.
    for candidate in ("1 month", "2 weeks", "1 week", "2 months", "3 months"):
        if candidate in out.columns and out[candidate].notna().any():
            return out.sort_values(candidate, ascending=False)
    return out.sort_values("Buys", ascending=False)


def horizon_coverage(ev: pd.DataFrame) -> dict:
    """Share of buys in this slice whose forward window has actually CLOSED.

    Matters the moment a date filter is applied. Ask for 1-12 September and the
    3-month column cannot exist yet -- but `mean()` skips NaN silently, so it would
    quietly report the average of whichever few buys are old enough to have resolved.
    That is a look-ahead in reverse: the number would describe a biased sub-slice of
    the window while appearing to describe the window.
    """
    if ev.empty:
        return {name: 0.0 for name in HORIZON_COLS.values()}
    return {name: float(ev[col].notna().mean()) for col, name in HORIZON_COLS.items()}


def client_trades(ev: pd.DataFrame, client_key: str) -> pd.DataFrame:
    """Every stock this client bought in the slice, newest first."""
    if ev.empty or "client_key" not in ev.columns:
        return pd.DataFrame()
    g = ev[ev.client_key == client_key].copy()
    if g.empty:
        return g
    out = pd.DataFrame({
        "Date": g["trade_date"],
        "Symbol": g["symbol"],
        "Deal": g["deal_type"],
        "Qty (lakh)": g["qty_net"] / 1e5,
        "Value (Cr)": g["value_cr"],
        "Avg price": g["vwap"],
        "Last price": g["last_close"] if "last_close" in g else pd.NA,
        "Since buy %": g["since_pct"] * 100 if "since_pct" in g else pd.NA,
        "Nifty same span %": g["nifty_since_pct"] * 100 if "nifty_since_pct" in g else pd.NA,
        "Sessions since": g["sessions_since"] if "sessions_since" in g else pd.NA,
        "Price as of": pd.to_datetime(g["px_date"]) if "px_date" in g else pd.NaT,
        "% of 20d ADV": g["net_pct_adv"] * 100,
        "% of day delivery": g["pct_of_delivery"] if "pct_of_delivery" in g else pd.NA,
        "Exit after (days)": g["exit_days"] if "exit_days" in g else pd.NA,
        "Stock turnover (Cr)": g["turnover_lacs"] / 100,
    })
    for col, name in HORIZON_COLS.items():
        out[name] = g[col] * 100
    return out.sort_values("Date", ascending=False)


def stock_consensus(buys: pd.DataFrame, sells: pd.DataFrame | None = None,
                    exclude_hft: bool = False) -> pd.DataFrame:
    """One row per STOCK: how many different clients bought it in this window.

    The question this answers is "two desks bought the same name, is that a stronger
    signal?". Measured over 15,871 buys, the answer is no - agreement goes the WRONG
    way (see the caption this feeds). It is shipped as a description of who was
    active in a name, not as a ranking.

    Two things the count alone hides, so both are returned beside it:
      - HFT market makers are 27-45% of every multi-buyer cluster. Three "buyers"
        that are three market makers is inventory churn, not three opinions.
      - The sell side. NSE publishes both, and a name with 3 buyers and 4 sellers is
        being distributed, not accumulated. Net breadth is the honest version of the
        count, and its most negative cell is -2 (-4.82% / 1 month).
    """
    if buys.empty:
        return pd.DataFrame()
    b = buys[buys.bucket != "HFT_PROP"] if exclude_hft else buys
    if b.empty:
        return pd.DataFrame()

    def _types(g: pd.DataFrame) -> str:
        c = g.drop_duplicates("client_key")["bucket"].map(bucket_label).value_counts()
        return ", ".join(f"{n}x {lbl}" for lbl, n in c.items())

    rows = []
    for sym, g in b.groupby("symbol"):
        row = {
            "Symbol": sym,
            "Buyers": g["client_key"].nunique(),
            "Buys": len(g),
            "Days": g["trade_date"].nunique(),
            "First buy": g["trade_date"].min(),
            "Last buy": g["trade_date"].max(),
            "Who": _types(g),
            "Net bought (Cr)": (g["qty_net"] * g["vwap"]).sum() / 1e7,
            "Value (Cr)": g["value_cr"].sum(),
            "Median % of delivery": g.loc[g.deal_type == "BULK", "pct_of_delivery"].median()
                                    if (g.deal_type == "BULK").any() else float("nan"),
        }
        if "since_pct" in g.columns:
            row["Since buy median"] = g["since_pct"].median() * 100
            row["vs Nifty median"] = (g["since_pct"] - g["nifty_since_pct"]).median() * 100
        else:
            row["Since buy median"] = row["vs Nifty median"] = float("nan")
        for col, name in HORIZON_COLS.items():
            row[name] = g[col].mean() * 100
            row[f"{name} n"] = int(g[col].notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows)

    # Sell side over the SAME window, so crowding can be read against distribution.
    if sells is not None and not sells.empty:
        sl = sells[sells.symbol.isin(out.Symbol)]
        sc = sl.groupby("symbol")["client_key"].nunique().rename("Sellers")
        out = out.merge(sc, left_on="Symbol", right_index=True, how="left")
    if "Sellers" not in out.columns:
        out["Sellers"] = 0
    out["Sellers"] = out["Sellers"].fillna(0).astype(int)
    out["Net buyers"] = out["Buyers"] - out["Sellers"]
    return out.sort_values(["Buyers", "Value (Cr)"], ascending=False)


def stock_participants(buys: pd.DataFrame, sells: pd.DataFrame | None,
                       symbol: str) -> pd.DataFrame:
    """Every client on either side of one stock in the window, newest first."""
    parts = []
    for df, side in ((buys, "BOUGHT"), (sells, "SOLD")):
        if df is None or df.empty:
            continue
        g = df[df.symbol == symbol]
        if g.empty:
            continue
        parts.append(pd.DataFrame({
            "Date": g["trade_date"],
            "Side": side,
            "Client": g["client_name"],
            "Type": g["bucket"].map(bucket_label),
            "Deal": g["deal_type"],
            "Qty (lakh)": g["qty_net"].abs() / 1e5,
            "Net value (Cr)": (g["qty_net"].abs() * g["vwap"]) / 1e7,
            "Value (Cr)": g["value_cr"],
            "Avg price": g["vwap"],
            # Signed on the table it comes from; the Side column already carries
            # direction, so a negative here would just read as an error.
            "% of day delivery": (g["pct_of_delivery"].abs()
                                  if "pct_of_delivery" in g else pd.NA),
            "Since buy %": (g["since_pct"] * 100) if "since_pct" in g else pd.NA,
        }))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_values(["Date", "Side"], ascending=[False, True])


def repeat_buy_table(ev: pd.DataFrame) -> pd.DataFrame:
    """The "accumulation" story: same entity buying the same name again."""
    if ev.empty:
        return pd.DataFrame()
    d = ev.sort_values("trade_date").copy()
    d["prev"] = d.groupby(["client_key", "symbol"])["trade_date"].shift(1)
    rep = (d["trade_date"] - d["prev"]).dt.days.le(90).fillna(False)
    return pd.DataFrame([
        _hrow(d[rep], "Repeat buy, same name within 3 months"),
        _hrow(d[~rep], "First buy in that name"),
    ])


def recent_events(out: pd.DataFrame, days: int = 120) -> pd.DataFrame:
    """Deals recent enough that the 1-2 month window is still open or just closed."""
    if out.empty:
        return pd.DataFrame()
    cut = out["trade_date"].max() - pd.Timedelta(days=days)
    return out[out["trade_date"] >= cut].sort_values("trade_date", ascending=False)
