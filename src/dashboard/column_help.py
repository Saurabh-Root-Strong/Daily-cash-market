"""
Central column-help glossary + thin column_config wrappers.

Every table in the app routes its column_config through nc/tc/dc/cc/pc instead of
st.column_config.*. The wrappers auto-fill the `help=` tooltip from GLOSSARY by the
column label (so hovering any header shows "what is this / why we use it"), unless
an explicit help= is passed. Add a term here once and it applies app-wide.
"""
from __future__ import annotations

import streamlit as st

GLOSSARY: dict[str, str] = {
    # ── Options / OI ──────────────────────────────────────────────────────────
    "Strike": "Option strike price (the level the contract is struck at).",
    "CE OI": "Call Open Interest — outstanding call contracts at this strike. Why: a high-OI call strike acts as RESISTANCE (writers defend it).",
    "PE OI": "Put Open Interest — outstanding put contracts at this strike. Why: a high-OI put strike acts as SUPPORT.",
    "Call OI": "Total call open interest. Why: where call writers sit = overhead resistance.",
    "Put OI": "Total put open interest. Why: where put writers sit = downside support.",
    "CE Chg OI": "Change in call OI vs the prior session. Why: + = fresh call writing/buying, − = unwinding.",
    "PE Chg OI": "Change in put OI vs the prior session. Why: + = fresh put writing/buying, − = unwinding.",
    "Chg OI": "Change in open interest vs the prior session. Why: + = new positions (conviction), − = unwinding.",
    "Total OI": "Total open interest (calls + puts) at this strike.",
    "OI Contracts": "Open interest expressed in number of contracts.",
    "CE Price": "Call option last price (premium, ₹).",
    "PE Price": "Put option last price (premium, ₹).",
    "CE Price (₹)": "Call option last price (premium, ₹).",
    "PE Price (₹)": "Put option last price (premium, ₹).",
    "CE Vol": "Call traded volume (contracts) today.",
    "PE Vol": "Put traded volume (contracts) today.",
    "PCR": "Put/Call Ratio (open interest). Why: >1 = put-heavy (support/hedging), <0.7 = call-heavy (complacency, often near tops).",
    "ATM?": "Marks the at-the-money strike (closest to spot) — where gamma/hedging is most active.",
    "Max Pain": "The strike where option BUYERS lose most (writers profit most). Why: price often gravitates here into expiry.",
    "Max Pain?": "Flags the max-pain strike — the pin/gravity level into expiry.",
    "MP Dist%": "Distance of spot from max pain (%). Why: a large gap near expiry = stronger pin pull toward max pain.",
    "Top CE Strike": "The call strike with the most OI = the key resistance wall.",
    "Top PE Strike": "The put strike with the most OI = the key support wall.",
    "DTE": "Days To Expiry of the contract.",
    "Fut OI": "Futures open interest — outstanding futures contracts. Why: rising OI confirms conviction behind a move.",
    "Spot": "Current index/stock spot (cash) price.",
    "Basis%": "Futures premium/discount to spot, annualised %. Why: + = contango (bullish carry), − = backwardation (stress/bearish).",
    "Near OI": "Open interest in the near (current) month contract.",
    "Next OI": "Open interest in the next month contract.",
    "Far OI": "Open interest in the far month contract.",
    "Roll Signal": "Rollover read — whether positions are being carried to the next expiry, and the bias of that roll.",

    # ── Participant / FII-DII flows ───────────────────────────────────────────
    "Buy (Cr)": "Gross buy value (₹ Cr).",
    "Sell (Cr)": "Gross sell value (₹ Cr).",
    "Net (₹Cr)": "Net = buy − sell (₹ Cr). + = net buying, − = net selling.",
    "Net Flow (Cr)": "Net flow = buy − sell (₹ Cr). + = inflow, − = outflow.",
    "Gross Buy (₹Cr)": "Total buy value (₹ Cr).",
    "Gross Sell (₹Cr)": "Total sell value (₹ Cr).",
    "Buy Contracts": "Gross long positions opened (contracts).",
    "Sell Contracts": "Gross short positions opened (contracts).",
    "Net Contracts": "Net position = long − short (contracts). Why: the participant's net directional bet.",
    "Net Share %": "This participant's share of the total net flow.",
    "Covering": "Short covering — existing shorts being bought back. Why: forced covering creates upward pressure.",
    "Squeeze": "Short-squeeze setup — trapped shorts that may be forced to cover (explosive upside risk).",

    # ── Delivery / sector ─────────────────────────────────────────────────────
    "Deliv %": "Delivery % — shares actually taken to demat vs total traded. Why: high = investment/conviction, not just intraday churn.",
    "Today Deliv%": "Today's delivery % (shares taken to demat vs traded). Why: high = real accumulation.",
    "Deliv Ratio": "Today's delivery value ÷ its own 100-day average. Why: >1 = above-normal accumulation.",
    "DV Ratio": "Delivered-value ratio — today's delivery ₹ ÷ its 100-day daily average. Why: >1 = above-normal institutional buying.",
    "Vol Ratio": "Volume ratio — today's volume ÷ its average. Why: a spike confirms genuine participation behind a move.",
    "1W Deliv (Cr)": "Delivery value over the last 1 week (₹ Cr).",
    "2W Deliv (Cr)": "Delivery value over the last 2 weeks (₹ Cr).",
    "1M Deliv (Cr)": "Delivery value over the last 1 month (₹ Cr).",
    "3M Deliv (Cr)": "Delivery value over the last 3 months (₹ Cr).",
    "1W Price%": "Price return over the last 1 week.",
    "2W Price%": "Price return over the last 2 weeks.",
    "1M Price%": "Price return over the last 1 month.",
    "3M Price%": "Price return over the last 3 months.",
    "Z-Score": "How many standard deviations today's delivery value is above its 100-day mean. Why: ≥2 = top ~2.5% of days (statistically unusual).",
    "Score": "Composite signal score (0–100) ranking this row against the others on the same screen.",
    "Phase": "Rotation-clock phase — Leading / Improving / Weakening / Lagging (where institutional money is flowing).",
    "Phase on Signal Date": "The rotation phase as classified on the historical signal date (point-in-time).",
    "Flow Signal": "Plain-English institutional-flow label (money entering / exiting / sideways).",
    "Today Chg%": "Today's price change (%).",
    "Nifty%": "Nifty 50 return over the same period — the market benchmark to compare the sector against.",
    "Sector": "The sector this row belongs to.",
    "Sub-Sector": "The finer sub-sector / industry grouping.",
    "Index / Category": "The index or category grouping.",
    "Category": "Grouping category.",
    "Trading Days": "Number of actual trading days in the measurement window.",
    "Conviction": "Conviction tier of the signal (how strongly the evidence aligns).",
    "Deliv Chg vs Prior %": "Delivery value vs the prior equal-length period (%). + = accelerating accumulation.",

    # ── Prediction / backtest ─────────────────────────────────────────────────
    "Symbol": "Stock ticker symbol.",
    "Company": "Company name.",
    "Action": "Recommended action (e.g. Buy / Accumulate / Avoid).",
    "Return %": "Realised forward return over the test horizon (%).",
    "Signal": "The signal classification assigned to this row.",
    "Horizon": "The holding / forecast horizon for the signal.",
    "Actual Dir": "What the index ACTUALLY did the next day (realised direction).",
    "Correct?": "Whether the prediction matched the realised outcome.",
    "Conf.": "Confidence tier of the prediction.",
    "Outcome": "The realised outcome after the fact.",
    "Verdict": "The engine's overall verdict for this row.",
    "View": "Quick navigation / detail view link.",
    "SimP&L": "Simulated profit/loss from following the signal in the backtest.",
    "Date": "Calendar date.",
    "Close (₹)": "Closing price (₹).",
    "Forward Return (abs)": "Absolute price return over the forward window after the signal (%).",
    "vs Nifty50": "Return relative to Nifty 50 over the same window (sector/stock minus index).",
}


def _help(label, help):
    if help is not None:
        return help
    if not label:
        return None
    return GLOSSARY.get(label) or GLOSSARY.get(label.rstrip(" ?%")) or None


def nc(label=None, *, help=None, **kw):
    return st.column_config.NumberColumn(label, help=_help(label, help), **kw)


def tc(label=None, *, help=None, **kw):
    return st.column_config.TextColumn(label, help=_help(label, help), **kw)


def dc(label=None, *, help=None, **kw):
    return st.column_config.DateColumn(label, help=_help(label, help), **kw)


def cc(label=None, *, help=None, **kw):
    return st.column_config.CheckboxColumn(label, help=_help(label, help), **kw)


def pc(label=None, *, help=None, **kw):
    return st.column_config.ProgressColumn(label, help=_help(label, help), **kw)
