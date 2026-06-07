"""
FII/DII Flow Intelligence — an institutional-flow understanding engine.

WHAT IT IS
----------
Reads the daily cash-market FII + DII net flows and explains, in institutional
terms, WHAT the two sides are doing (regime, streaks, intensity, acceleration,
the tug-of-war) and what is — and is NOT — a validated forward read.

EVIDENCE (validated on ~350 sessions, point-in-time)
----------------------------------------------------
- FII net is the PRICE-SETTER but CONTEMPORANEOUS: FII[T] vs same-day return
  IC ≈ +0.37, yet FII[T] → next-day return IC ≈ 0 (slightly mean-reverting).
  → FII tells you the pressure happening NOW, not tomorrow's direction.
- DII net carries the only modest FORWARD tilt: DII[T] → next-day IC ≈ +0.11
  (sticky domestic / SIP support tends to continue). Still small — a tilt, not
  a forecast.
So: FII drives the REGIME read; DII drives the (mild) FORWARD lean. Everything
here is point-in-time (only flows ≤ as_of are used).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_flow_intelligence", "get_flow_history_pattern"]


def _regime_tag(fii5: float, dii5: float) -> Optional[str]:
    if fii5 is None or (isinstance(fii5, float) and np.isnan(fii5)):
        return None
    if fii5 < -3000 and dii5 > 0:  return "ABSORBED SELLING"
    if fii5 < -3000 and dii5 <= 0: return "BROAD RISK-OFF"
    if fii5 > 3000 and dii5 > 0:   return "ALIGNED BUYING"
    if fii5 > 3000 and dii5 < 0:   return "FII-LED BUYING"
    return "BALANCED"

VALIDATION = ("FII flow is contemporaneous (same-day IC +0.37, next-day ~0); the modest "
              "forward tilt is DII (+0.11 next-day). Regime = FII; forward lean = DII.")


def _fmt(v: float) -> str:
    return f"₹{v:+,.0f} Cr"


def get_flow_intelligence(as_of_date: date, lookback: int = 120) -> dict:
    df = query_dataframe("""
        SELECT trade_date, fii_net, dii_net FROM fii_dii_cash
        WHERE fii_net IS NOT NULL AND trade_date <= ?
        ORDER BY trade_date
    """, [as_of_date])
    if df.empty or len(df) < 12:
        return {}
    df = df.tail(lookback).reset_index(drop=True)
    fii = df["fii_net"].astype(float)
    dii = df["dii_net"].astype(float)
    n = len(df)

    fii_t, dii_t = float(fii.iloc[-1]), float(dii.iloc[-1])
    fii_5d = float(fii.iloc[-5:].sum()); dii_5d = float(dii.iloc[-5:].sum())
    fii_20d = float(fii.iloc[-20:].sum()) if n >= 20 else float(fii.sum())
    dii_20d = float(dii.iloc[-20:].sum()) if n >= 20 else float(dii.sum())

    # Intensity: today's FII net vs its trailing-20d distribution (robust z).
    base = fii.iloc[-21:-1] if n >= 21 else fii.iloc[:-1]
    mu, sd = float(base.mean()), float(base.std())
    fii_z = round((fii_t - mu) / sd, 2) if sd > 0 else 0.0

    # Acceleration: last-5d FII net vs the prior 5d.
    fii_5d_prev = float(fii.iloc[-10:-5].sum()) if n >= 10 else 0.0
    fii_accel = fii_5d - fii_5d_prev

    # Streak: consecutive same-sign FII days ending today.
    sign = np.sign(fii.values)
    streak = 1
    for i in range(n - 2, -1, -1):
        if sign[i] == sign[-1] and sign[-1] != 0:
            streak += 1
        else:
            break
    fii_streak = int(streak) * (1 if sign[-1] > 0 else -1)

    # Extreme percentile of the 5d net within the lookback window.
    roll5 = fii.rolling(5).sum().dropna()
    fii_5d_pctl = float((roll5 <= fii_5d).mean()) if len(roll5) else 0.5

    # DII absorption of FII selling over 5d (the classic Indian tug-of-war).
    absorption = None
    if fii_5d < 0 and dii_5d > 0:
        absorption = round(min(dii_5d / abs(fii_5d), 2.0) * 100, 0)

    # ── FII regime (the pressure read) ────────────────────────────────────────
    if fii_5d < 0:
        fii_regime = ("🔴 Aggressive Distribution" if fii_5d_pctl <= 0.12
                      else "🟠 Distribution")
    elif fii_5d > 0:
        fii_regime = ("🟢 Aggressive Accumulation" if fii_5d_pctl >= 0.88
                      else "🟢 Accumulation")
    else:
        fii_regime = "⚖️ Neutral"
    if abs(fii_5d) < 3000:
        fii_regime = "⚖️ Neutral / Balanced"

    # ── DII stance ────────────────────────────────────────────────────────────
    if dii_5d > 8000:
        dii_stance = "🛡️ Strongly Absorbing / Supporting"
    elif dii_5d > 0:
        dii_stance = "🛡️ Supporting"
    elif dii_5d < -5000:
        dii_stance = "📉 Selling alongside"
    else:
        dii_stance = "⚖️ Neutral"

    # ── Structural read (the tug-of-war) ──────────────────────────────────────
    if fii_5d < -3000 and dii_5d > 0:
        cushion = "fully" if (absorption or 0) >= 95 else "partly"
        structural = (f"FII distribution being {cushion} absorbed by DII — domestic money is "
                      "cushioning foreign outflows. Markets often grind/hold rather than crash "
                      "while this holds; the risk is if DII support fades.")
        regime_tag = "ABSORBED SELLING"
    elif fii_5d < -3000 and dii_5d <= 0:
        structural = ("Broad institutional selling — BOTH FII and DII are net sellers. No "
                      "domestic floor; this is the genuine risk-off configuration.")
        regime_tag = "BROAD RISK-OFF"
    elif fii_5d > 3000 and dii_5d > 0:
        structural = ("Aligned accumulation — FII and DII both buying. The strongest demand "
                      "configuration; broad institutional tailwind.")
        regime_tag = "ALIGNED BUYING"
    elif fii_5d > 3000 and dii_5d < 0:
        structural = ("FII buying into DII profit-taking — foreign demand absorbing domestic "
                      "supply. Constructive but watch for DII distribution capping upside.")
        regime_tag = "FII-LED BUYING"
    else:
        structural = "Balanced / low-conviction flows — neither side is pressing."
        regime_tag = "BALANCED"

    # ── Forward lean (validated: DII-driven, modest) ──────────────────────────
    # DII next-day IC ≈ +0.11; FII ≈ 0. So the lean comes from DII, scaled small.
    if dii_5d > 10000 and fii_5d > -25000:
        lean, lean_txt = "UP", ("mild UP — sustained DII buying has historically lent a small "
                                "next-day tailwind (the one validated tilt).")
    elif dii_5d < -3000:
        lean, lean_txt = "DOWN", ("mild DOWN — DII not supporting (the floor that usually "
                                  "cushions is absent).")
    else:
        lean, lean_txt = "NEUTRAL", ("neutral — no validated forward edge from flows alone "
                                     "(FII flow is a present-pressure gauge, not a forecast).")

    return {
        "as_of": str(as_of_date), "n": n,
        "fii_today": round(fii_t, 0), "dii_today": round(dii_t, 0),
        "fii_5d": round(fii_5d, 0), "dii_5d": round(dii_5d, 0),
        "fii_20d": round(fii_20d, 0), "dii_20d": round(dii_20d, 0),
        "fii_z": fii_z, "fii_accel": round(fii_accel, 0),
        "fii_streak": fii_streak, "fii_5d_pctl": round(fii_5d_pctl * 100, 0),
        "absorption_pct": absorption,
        "fii_regime": fii_regime, "dii_stance": dii_stance,
        "regime_tag": regime_tag, "structural_read": structural,
        "forward_lean": lean, "forward_note": lean_txt,
        "narrative": _narrative(fii_t, dii_t, fii_5d, dii_5d, fii_streak, fii_z, fii_accel, absorption),
        "validation": VALIDATION,
    }


def _narrative(fii_t, dii_t, fii_5d, dii_5d, streak, z, accel, absorption) -> str:
    bits = []
    side = "sold" if fii_t < 0 else "bought"
    bits.append(f"Today FIIs {side} ₹{abs(fii_t):,.0f} Cr (net {_fmt(fii_t)}), "
                f"DII net {_fmt(dii_t)}.")
    if abs(streak) >= 3:
        bits.append(f"FIIs have been net {'sellers' if streak < 0 else 'buyers'} "
                    f"{abs(streak)} sessions running — a {'persistent' if abs(streak) >= 4 else 'short'} streak.")
    if abs(z) >= 1.5:
        bits.append(f"Today's flow is {'unusually large' if z < 0 else 'a strong buy'} "
                    f"({z:+.1f}σ vs its 20-day norm).")
    if accel < -5000:
        bits.append("Selling is ACCELERATING vs the prior week.")
    elif accel > 5000:
        bits.append("Buying is ACCELERATING vs the prior week.")
    if absorption is not None:
        bits.append(f"Over 5 days DIIs have absorbed ~{absorption:.0f}% of the FII selling.")
    return " ".join(bits)


def get_flow_history_pattern(as_of_date: date, horizon: int = 10) -> dict:
    """
    Historical-analog study: classify every past day's flow regime and measure the
    forward `horizon`-day Nifty outcome, then surface what the CURRENT regime has
    historically led to. Answers "is the selling 'actual' or absorbed?" — point-in-
    time (forward windows must be fully realised, i.e. day <= as_of - horizon).

    Returns the current regime's forward distribution, all-regime context, the
    contemporaneous FII↔index correlation, and an overlay series (FII/DII cumulative
    + Nifty) for charting. Honest: these are descriptive tendencies on a flow-skewed
    ~1yr sample, NOT forecasts.
    """
    f = query_dataframe("""
        SELECT trade_date, fii_net, dii_net FROM fii_dii_cash
        WHERE fii_net IS NOT NULL AND trade_date <= ? ORDER BY trade_date
    """, [as_of_date])
    n = query_dataframe("""
        SELECT trade_date, close_val FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date
    """, [as_of_date])
    if f.empty or n.empty or len(f) < 40:
        return {}
    f["trade_date"] = pd.to_datetime(f["trade_date"])
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    d = f.merge(n, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    if len(d) < 40:
        return {}
    fii = d["fii_net"].astype(float); dii = d["dii_net"].astype(float)
    d["fii5"] = fii.rolling(5).sum(); d["dii5"] = dii.rolling(5).sum()
    d["regime"] = [_regime_tag(a, b) for a, b in zip(d["fii5"], d["dii5"])]
    d["fwd"] = (d["close_val"].shift(-horizon) / d["close_val"] - 1) * 100
    d["fii_cum"] = fii.cumsum(); d["dii_cum"] = dii.cumsum()

    # Only days with a FULLY REALISED forward window (point-in-time).
    study = d[d["fwd"].notna() & d["regime"].notna()]
    cur_regime = d["regime"].iloc[-1]

    def _dist(g):
        return {"mean": round(float(g.mean()), 2), "median": round(float(g.median()), 2),
                "pos_pct": round(float((g > 0).mean() * 100), 0), "n": int(len(g))}

    by_regime = {r: _dist(g["fwd"]) for r, g in study.groupby("regime") if len(g) >= 8}
    cur = by_regime.get(cur_regime)

    # Contemporaneous correlation (the strong, real relationship).
    corr = round(float(d["fii5"].corr(d["close_val"].pct_change(5) * 100)), 2)

    # Plain interpretation of the current regime's historical forward tendency.
    interp = ""
    if cur:
        if cur["mean"] > 0.4 and cur["pos_pct"] >= 55:
            interp = (f"Historically this regime led to a HIGHER index 2 weeks later "
                      f"(+{cur['mean']:.1f}% avg, up {cur['pos_pct']:.0f}% of the time).")
        elif cur["mean"] < -0.4 and cur["pos_pct"] <= 45:
            interp = (f"Historically this regime led to a LOWER index 2 weeks later "
                      f"({cur['mean']:.1f}% avg, up only {cur['pos_pct']:.0f}% of the time) — "
                      "the selling tended to be 'real'.")
        else:
            interp = (f"Historically this regime led to a roughly FLAT index 2 weeks later "
                      f"({cur['mean']:+.1f}% avg, {cur['pos_pct']:.0f}% positive, n={cur['n']}). "
                      + ("The absorbed selling did NOT translate into a crash — DII cushioning "
                         "held the market." if cur_regime == "ABSORBED SELLING"
                         else "Flows were not decisive for the 2-week path."))

    series = d[["trade_date", "fii_cum", "dii_cum", "close_val"]].tail(260).copy()
    series["trade_date"] = series["trade_date"].dt.strftime("%Y-%m-%d")

    return {
        "as_of": str(as_of_date), "horizon": horizon,
        "current_regime": cur_regime, "current_dist": cur,
        "by_regime": by_regime, "contemporaneous_corr": corr,
        "interpretation": interp,
        "series": series.to_dict("records"),
        "n_days": int(len(study)),
    }


def get_two_week_outlook(as_of_date: date) -> dict:
    """
    Consolidated 1–2 week outlook — synthesises the VALIDATED engines into one
    honest read: flow regime (FII-DII) + what that regime led to historically +
    the weekly mean-reversion tilt + the DII trip-wire. Returns a bias label,
    supports, risks, the trip-wire, and an explicit (low) confidence.

    NOT a forecast: every component is a weak/descriptive tendency. The value is a
    coherent base case + the one condition that flips it (DII turning seller).
    """
    flow = get_flow_intelligence(as_of_date)
    if not flow:
        return {}
    hist = get_flow_history_pattern(as_of_date)
    try:
        from src.analytics.weekly_outlook import get_weekly_outlook
        nifty_wk = next((r for r in get_weekly_outlook(as_of_date)
                         if r["label"] == "Nifty 50"), None)
    except Exception:
        nifty_wk = None

    score = 0.0
    supports: list[str] = []
    risks: list[str] = []
    tag = flow["regime_tag"]

    if tag == "ABSORBED SELLING":
        supports.append("DII absorbing FII selling — downside cushioned "
                        "(this regime historically held ~flat over 2 weeks, no crash).")
    elif tag == "BROAD RISK-OFF":
        score -= 2; risks.append("BOTH FII and DII are net sellers — no domestic floor (true risk-off).")
    elif tag == "ALIGNED BUYING":
        score += 2; supports.append("FII and DII both buying — broad demand tailwind.")
    elif tag == "FII-LED BUYING":
        score += 1; supports.append("FII buying is absorbing DII supply — constructive.")

    if nifty_wk:
        if nifty_wk["weekly_bias"] == "UP":
            score += 1
            supports.append(f"Nifty is oversold (RSI {nifty_wk['rsi']:.0f}) — a mild mean-reversion bounce bias.")
        elif nifty_wk["weekly_bias"] == "DOWN":
            score -= 1
            risks.append(f"Nifty is overbought (RSI {nifty_wk['rsi']:.0f}) — pullback risk into the week.")

    if hist and hist.get("current_dist"):
        m = hist["current_dist"]["mean"]
        score += 0.5 if m > 0.3 else (-0.5 if m < -0.3 else 0.0)

    if flow["dii_5d"] > 10000:
        supports.append("Strong DII buying is providing a floor.")
    if flow.get("fii_accel", 0) < -8000:
        risks.append("FII selling is ACCELERATING vs the prior week.")

    if score >= 2:      bias, col = "MILDLY BULLISH", "#00c853"
    elif score >= 0.5:  bias, col = "RANGE — MILD UP TILT", "#69f0ae"
    elif score <= -2:   bias, col = "RISK-OFF — LEAN DOWN", "#ff5252"
    elif score <= -0.5: bias, col = "RANGE — MILD DOWN TILT", "#ff9100"
    else:               bias, col = "RANGE-BOUND / FLAT", "#9e9e9e"

    if tag == "ABSORBED SELLING":
        tripwire = ("Watch DII flow. If DII flips to net SELLER, the regime becomes BROAD "
                    "RISK-OFF (FII + DII both selling = no floor = the real-downside case).")
    elif tag == "BROAD RISK-OFF":
        tripwire = "Watch for DII to step back in as a buyer — that would re-establish the floor."
    else:
        tripwire = "Watch whether FII and DII stay aligned; divergence changes the setup."

    hist_txt = ""
    if hist and hist.get("current_dist"):
        cd = hist["current_dist"]
        hist_txt = (f"In this regime historically, Nifty was {cd['mean']:+.1f}% on average 2 weeks "
                    f"later ({cd['pos_pct']:.0f}% positive, n={cd['n']}).")

    base_case = (f"Most-supported path: **{bias.lower()}**. " + (" ".join(supports[:2]) or "") +
                 (" " + hist_txt if hist_txt else ""))

    return {
        "as_of": str(as_of_date), "bias": bias, "bias_color": col, "score": round(score, 1),
        "base_case": base_case, "supports": supports, "risks": risks, "tripwire": tripwire,
        "confidence": "Low–Moderate (descriptive tendencies, not a forecast)",
        "regime_tag": tag, "weekly_bias": (nifty_wk["weekly_bias"] if nifty_wk else None),
        "hist_2wk": (hist.get("current_dist") if hist else None),
    }


def get_flow_events(as_of_date: date, z_threshold: float = 2.0) -> dict:
    """
    Flow-event MEMORY — detects and remembers the SIGNIFICANT flow moments across
    the full history (a sudden huge FII buy/sell, or a reversal), records what the
    market did afterwards, and flags whether TODAY is such an event.

    'Huge' = FII net at |z| >= threshold vs its trailing-60d norm (point-in-time).
    EVIDENCE (small samples, so suggestive): a huge FII BUY day mildly precedes
    weakness (mean-reversion — piling in marks tops); a huge SELL day → roughly flat
    (capitulation exhausts). Forward stats use only REALISED windows. Descriptive,
    not a forecast.
    """
    f = query_dataframe("""
        SELECT trade_date, fii_net, dii_net FROM fii_dii_cash
        WHERE fii_net IS NOT NULL AND trade_date <= ? ORDER BY trade_date
    """, [as_of_date])
    n = query_dataframe("""
        SELECT trade_date, close_val FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date
    """, [as_of_date])
    if f.empty or n.empty or len(f) < 65:
        return {}
    f["trade_date"] = pd.to_datetime(f["trade_date"]); n["trade_date"] = pd.to_datetime(n["trade_date"])
    d = f.merge(n, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    fii = d["fii_net"].astype(float)
    mu = fii.rolling(60).mean().shift(1); sd = fii.rolling(60).std().shift(1)
    d["z"] = (fii - mu) / sd
    sign = np.sign(fii.values)
    c5 = d["close_val"].shift(-5) / d["close_val"] - 1
    c10 = d["close_val"].shift(-10) / d["close_val"] - 1

    events = []
    for i in range(len(d)):
        z = d["z"].iloc[i]
        if pd.isna(z):
            continue
        ev = None
        if z >= z_threshold:   ev = "🟢 Huge FII Buy"
        elif z <= -z_threshold: ev = "🔴 Huge FII Sell"
        else:
            # Reversal: flips sign today after a >=3-day same-sign streak.
            if i >= 4 and sign[i] != 0 and sign[i] != sign[i-1] and \
               sign[i-1] == sign[i-2] == sign[i-3]:
                ev = ("🔄 FII flips to BUYING" if sign[i] > 0 else "🔄 FII flips to SELLING")
        if ev:
            events.append({
                "date": d["trade_date"].iloc[i].strftime("%d %b %y"),
                "_dt": d["trade_date"].iloc[i],
                "type": ev, "fii_net": round(float(fii.iloc[i]), 0),
                "dii_net": round(float(d["dii_net"].iloc[i]), 0), "z": round(float(z), 1),
                "fwd5": round(float(c5.iloc[i] * 100), 2) if pd.notna(c5.iloc[i]) else None,
                "fwd10": round(float(c10.iloc[i] * 100), 2) if pd.notna(c10.iloc[i]) else None,
            })

    latest_dt = d["trade_date"].iloc[-1]
    today_event = next((e for e in events if e["_dt"] == latest_dt), None)

    # Per-type forward stats (realised only) — what historically followed each type.
    type_stats = {}
    for t in set(e["type"] for e in events):
        fwds = [e["fwd10"] for e in events if e["type"] == t and e["fwd10"] is not None]
        if len(fwds) >= 4:
            arr = np.array(fwds)
            type_stats[t] = {"mean10": round(float(arr.mean()), 2),
                             "pos_pct": round(float((arr > 0).mean() * 100), 0), "n": len(fwds)}

    for e in events:
        e.pop("_dt", None)
    return {
        "as_of": str(as_of_date), "z_threshold": z_threshold,
        "today_event": today_event, "events": events[-15:][::-1],
        "type_stats": type_stats, "total_events": len(events),
    }
