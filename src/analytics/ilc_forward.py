"""
Index & Large Cap -- a PRE-REGISTERED forward test of its one surviving lead.

WHAT IS BEING TESTED
    Backtest, 2022-01 .. 2026-10 (scratch study, 13 Oct 2026): a ridge model on the
    numbers the tab prints (net score, each bucket's futures lean, options side,
    delivery z, futures OI %, carry spread) plus the index's own close-location and
    return, refit every 20 sessions on the past only, against the NEXT session's
    open -> close:
        Nifty 50    rank corr +0.12 (time-shifted null 95th +0.06, 0 of 300 beat it)
        Bank Nifty  rank corr +0.11
    Positive in 2024, 2025 and 2026. Trading only the top / bottom fifth after a
    0.05% round-trip cost: Nifty +0.08% per trade (60% hit, t 1.8, 277 trades),
    Bank Nifty +0.13% (t 2.4). Negative over the last 63 sessions on both. Adding
    Index Prediction (OFF composite, FII stance, FII legs) did not improve it.
    The two indices share their biggest banks, so they are not two confirmations.

    That is a lead, not an edge. The honest next step is to FREEZE the model and
    watch fresh sessions it has never seen. This module is that step.

THE RULES, FIXED BEFORE THE FIRST FORWARD ROW (also inside the model file)
    model     frozen coefficients in config/ilc_forward_model.json; never refit
    signal    the evening after the F&O file lands; written once, append-only,
              stamped with the time it was written
    on time   a row counts only if written before 09:15 IST on the next session
    missing   any input missing (e.g. the monthly settlement session, when the
              options read is switched off) = no forecast that day, never a guess
    outcome   next session's index open -> close (the part that can be acted on)
    verdict   PRIMARY: Nifty, rank corr of forecast vs outcome over on-time rows.
              Decided at 300 on-time sessions (about end-2027): PASS if rank corr
              t >= 2.0 AND the top/bottom-fifth trades net of 0.05% are positive.
              One interim look at 150: STOP if rank corr <= -0.05.
              Bank Nifty reported beside it, not counted as a second test.
    power     at a true rank corr of 0.12, t reaches 2 only after ~280 sessions;
              the per-trade figure needs far more. Nothing before then is evidence.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.forward_log import append_once, log_path, read_log
from src.data.repository import query_dataframe

MODEL_PATH = Path(__file__).resolve().parents[2] / "config" / "ilc_forward_model.json"
LOG_NAME = "ilc_lead"
OPEN_CUTOFF = (9, 15)                  # IST: a row must be written before the next open
COST_PCT = 0.05

_OPT_SIDE = {"put writing": 1.0, "put side heavier": 1.0, "balanced": 0.0,
             "call side heavier": -1.0, "call writing": -1.0}


def load_model(path: Path = MODEL_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _index_bar(index_name: str, trade_date: date) -> Optional[dict]:
    df = query_dataframe(
        "SELECT trade_date, high_val, low_val, close_val FROM index_data "
        "WHERE index_name = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 2",
        [index_name, trade_date])
    if len(df) < 2 or pd.to_datetime(df.iloc[0]["trade_date"]).date() != trade_date:
        return None
    t, y = df.iloc[0], df.iloc[1]
    h, l, c, pc = (float(t["high_val"]), float(t["low_val"]),
                   float(t["close_val"]), float(y["close_val"]))
    if not (pc > 0 and c > 0):
        return None
    clr = min(1.0, max(0.0, (c - l) / (h - l))) if h > l else None
    return dict(clr=clr, ret0=(c / pc - 1) * 100, close=c)


def features(trade_date: date, fno_symbol: str) -> tuple[dict, str]:
    """The model's inputs exactly as the tab computes them on `trade_date`.

    Returns (inputs, problem). `problem` is "" when every input is present.
    """
    from src.analytics.index_largecap import get_index_largecap, index_meta
    d = get_index_largecap(trade_date, fno_symbol)
    if not d.data_ok:
        return {}, d.note or "no constituent data"
    if d.is_expiry_session:
        return {}, "monthly settlement session: the options read is switched off"
    if len(d.rows) != 3:
        return {}, "basket does not have three buckets"
    x = {"net": d.net_score, "carry": d.carry_spread}
    for tag, b in zip("TNR", d.rows):
        x[f"{tag}_lean"] = None if b.fut_lean is None else float(b.fut_lean)
        x[f"{tag}_os"] = _OPT_SIDE.get(b.opt_read) if b.opt_read is not None else None
        x[f"{tag}_dz"] = b.deliv_z
        x[f"{tag}_foi"] = b.fut_oi_pct
    bar = _index_bar(index_meta(fno_symbol, trade_date)["index_name"], trade_date)
    if bar is None:
        return x, "index row missing"
    x["clr"], x["ret0"] = bar["clr"], bar["ret0"]
    missing = [k for k, v in x.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    if missing:
        return x, "missing: " + ", ".join(missing)
    return x, ""


def score(x: dict, m: dict) -> float:
    """Frozen ridge forecast of next open -> close, in %."""
    v = np.array([float(np.clip(x[k], -50, 50)) for k in m["features"]])
    z = (v - np.array(m["mu"])) / np.array(m["sd"])
    return float(z @ np.array(m["beta"]) + m["intercept"])


def side(s: float, m: dict) -> str:
    lo, hi = m["cuts"]
    return "LONG" if s >= hi else "SHORT" if s <= lo else "NONE"


def forecast(trade_date: date, fno_symbol: str, model: Optional[dict] = None) -> dict:
    """{status, score, side, inputs} for one index on one close."""
    model = model or load_model()
    m = model["indices"].get(fno_symbol)
    if m is None:
        return dict(status="not in the pre-registered set", score=None, side=None, inputs={})
    x, problem = features(trade_date, fno_symbol)
    if problem:
        return dict(status=f"no forecast: {problem}", score=None, side=None, inputs=x)
    s = score(x, m)
    return dict(status="ok", score=s, side=side(s, m), inputs=x)


def log_forecast(trade_date: date, fno_symbol: str, now: Optional[datetime] = None) -> dict:
    """Compute and append today's forecast (once). Returns what happened."""
    model = load_model()
    f = forecast(trade_date, fno_symbol, model)
    if f["status"] != "ok":
        return dict(written=False, **f)
    now = now or datetime.now()
    m = model["indices"][fno_symbol]
    row = {"trade_date": trade_date.isoformat(), "symbol": fno_symbol,
           "logged_at": now.isoformat(timespec="seconds"),
           "model_version": model["version"], "score": f"{f['score']:.6f}",
           "side": f["side"], "cut_low": f"{m['cuts'][0]:.6f}", "cut_high": f"{m['cuts'][1]:.6f}",
           "inputs": json.dumps({k: round(float(v), 6) for k, v in f["inputs"].items()},
                                sort_keys=True)}
    return dict(written=append_once(log_path(LOG_NAME), row), **f)


def _next_session_bar(index_name: str, after: date) -> Optional[dict]:
    df = query_dataframe(
        "SELECT trade_date, open_val, close_val FROM index_data "
        "WHERE index_name = ? AND trade_date > ? ORDER BY trade_date LIMIT 1",
        [index_name, after])
    if df.empty or not float(df.iloc[0]["open_val"] or 0) > 0:
        return None
    r = df.iloc[0]
    return dict(date=pd.to_datetime(r["trade_date"]).date(),
                oc=(float(r["close_val"]) / float(r["open_val"]) - 1) * 100)


def _next_session_date(after: date) -> Optional[date]:
    df = query_dataframe(
        "SELECT MIN(trade_date) AS d FROM daily_data WHERE trade_date > ? "
        "AND series IN ('EQ','SM','ST')", [after])
    v = df.iloc[0]["d"] if not df.empty else None
    return None if v is None or pd.isna(v) else pd.to_datetime(v).date()


def on_time(trade_date: date, logged_at: datetime, next_session: Optional[date]) -> Optional[bool]:
    """Written before 09:15 on the next session? None while that session is unknown.

    Unknown next session: if it is already past 09:15 on the next weekday the row
    is late no matter which day turns out to be the session; else undecided.
    """
    nxt = next_session
    if nxt is None:
        nxt = trade_date + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        cutoff = datetime.combine(nxt, datetime.min.time()).replace(
            hour=OPEN_CUTOFF[0], minute=OPEN_CUTOFF[1])
        return False if logged_at >= cutoff else None
    cutoff = datetime.combine(nxt, datetime.min.time()).replace(
        hour=OPEN_CUTOFF[0], minute=OPEN_CUTOFF[1])
    return logged_at < cutoff


def forward_record(fno_symbol: str = "NIFTY") -> dict:
    """The log joined with what happened next. Only on-time rows are scored."""
    from src.analytics.index_largecap import INDEX_BUCKETS
    log = read_log(log_path(LOG_NAME))
    out = {"rows": pd.DataFrame(), "n_logged": 0, "n_on_time": 0, "n_scored": 0,
           "late": 0, "ic": None, "ic_t": None, "trades": 0, "trade_net": None,
           "trade_hit": None, "trade_t": None}
    if log.empty:
        return out
    log = log[log["symbol"] == fno_symbol].copy()
    out["n_logged"] = len(log)
    if log.empty:
        return out
    name = INDEX_BUCKETS[fno_symbol]["index_name"]
    recs = []
    for r in log.itertuples():
        d = date.fromisoformat(r.trade_date)
        at = datetime.fromisoformat(r.logged_at)
        nb = _next_session_bar(name, d)
        ok = on_time(d, at, _next_session_date(d))
        recs.append(dict(trade_date=d, logged_at=at, side=r.side, score=float(r.score),
                         on_time=ok, next_date=nb["date"] if nb else None,
                         next_oc=nb["oc"] if nb else None))
    R = pd.DataFrame(recs).sort_values("trade_date")
    R["pnl_net"] = np.where(R.side == "LONG", R.next_oc - COST_PCT,
                            np.where(R.side == "SHORT", -R.next_oc - COST_PCT, np.nan))
    out["rows"] = R
    out["late"] = int((R.on_time == False).sum())                       # noqa: E712
    ok = R[(R.on_time == True) & R.next_oc.notna()]                     # noqa: E712
    out["n_on_time"] = int((R.on_time == True).sum())                   # noqa: E712
    out["n_scored"] = len(ok)
    if len(ok) >= 10:
        ic = float(ok.score.rank().corr(ok.next_oc.rank()))
        out["ic"], out["ic_t"] = ic, ic * np.sqrt(len(ok))
    tr = ok[ok.side.isin(["LONG", "SHORT"])]
    out["trades"] = len(tr)
    if len(tr):
        out["trade_net"] = float(tr.pnl_net.mean())
        out["trade_hit"] = float((tr.pnl_net > 0).mean() * 100)
        if len(tr) >= 10 and tr.pnl_net.std() > 0:
            out["trade_t"] = float(tr.pnl_net.mean() / (tr.pnl_net.std() / np.sqrt(len(tr))))
    return out
