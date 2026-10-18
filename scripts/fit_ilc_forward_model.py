"""Freeze the Index & Large Cap forward-test model. Run ONCE; never refit.

usage: python scripts/fit_ilc_forward_model.py NIFTY=<panel.parquet> BANKNIFTY=<panel.parquet>

Each panel is one row per session (DatetimeIndex) built from the LIVE tab functions
(get_index_largecap per date), with the input columns below plus `oc1`, the next
session's index open -> close in %. The panels used on 13 Oct 2026 were checked
against the code of that day: 0 of 3,408 label/number cells differed on 48 sampled
Nifty sessions. Refuses to overwrite an existing model file -- a forward test whose
model can be quietly refit is not a forward test.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "ilc_forward_model.json"
FEATURES = ["net", "T_lean", "N_lean", "R_lean", "T_os", "N_os", "R_os",
            "T_dz", "N_dz", "R_dz", "T_foi", "N_foi", "R_foi", "carry", "clr", "ret0"]
LAM = 50.0
OPT = {"put writing": 1, "put side heavier": 1, "balanced": 0,
       "call side heavier": -1, "call writing": -1}


def fit(panel: pd.DataFrame) -> dict:
    P = panel.copy()
    for t in "TNR":
        if f"{t}_os" not in P.columns:
            P[f"{t}_os"] = P[f"{t}_opt"].map(OPT)
    X = P[FEATURES].astype(float).clip(-50, 50)
    ok = X.notna().all(axis=1) & P["oc1"].notna()
    X, y = X[ok].values, P.loc[ok, "oc1"].astype(float).values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = float(y.mean())
    beta = np.linalg.solve(Z.T @ Z + LAM * np.eye(Z.shape[1]), Z.T @ (y - ym))
    pred = Z @ beta + ym
    lo, hi = np.quantile(pred, [0.2, 0.8])
    return dict(mu=mu.round(8).tolist(), sd=sd.round(8).tolist(), beta=beta.round(8).tolist(),
                intercept=round(ym, 8), cuts=[round(float(lo), 8), round(float(hi), 8)],
                features=FEATURES, lam=LAM, n_train=int(ok.sum()),
                train_from=str(P.index[ok].min().date()), train_to=str(P.index[ok].max().date()),
                in_sample_rank_corr=round(float(pd.Series(pred).rank().corr(pd.Series(y).rank())), 4))


def main():
    if OUT.exists():
        sys.exit(f"{OUT} already exists. The forward model is frozen; delete it only to "
                 "start a NEW forward test with a new version, never to refit this one.")
    idx = {}
    for arg in sys.argv[1:]:
        sym, path = arg.split("=", 1)
        idx[sym] = fit(pd.read_parquet(path))
    model = {
        "version": "ilc-lead-2026-10-13",
        "frozen_on": "2026-10-13",
        "what": "Ridge forecast of the NEXT session's index open->close from the Index & "
                "Large Cap numbers + the index's own close-location and return.",
        "preregistration": {
            "signal_time": "evening after the F&O file; append-only; first write wins",
            "on_time": "written before 09:15 IST on the next session; late rows never scored",
            "missing_inputs": "no forecast that day (monthly settlement: options read is off)",
            "outcome": "next session index open -> close, %",
            "primary": "NIFTY rank corr(score, outcome) over on-time sessions",
            "decision_at": "300 on-time NIFTY sessions",
            "pass": "rank-corr t >= 2.0 AND top/bottom-fifth trades net of 0.05% > 0",
            "interim": "one look at 150 on-time sessions: stop if rank corr <= -0.05",
            "secondary": "BANKNIFTY, reported beside, not a second test (shared banks)",
        },
        "backtest_before_freezing": {
            "NIFTY": "walk-forward rank corr +0.12 (shift-null 95th +0.06); fifth trades "
                     "+0.08%/trade net, 60% hit, t 1.8, 277 trades; last 63 sessions negative",
            "BANKNIFTY": "walk-forward rank corr +0.11; fifth trades +0.13%/trade net, t 2.4",
        },
        "indices": idx,
    }
    OUT.write_text(json.dumps(model, indent=1), encoding="utf-8")
    for s, m in idx.items():
        print(s, m["n_train"], m["train_from"], m["train_to"], "cuts", m["cuts"],
              "in-sample rank corr", m["in_sample_rank_corr"])


if __name__ == "__main__":
    main()
