"""
backtest_target_estimator.py — can positioning/flow data give a HONEST next-day target?

The cockpit's old directional target = (composite/20)·σ. Composite has ~0 next-day IC
(it's a same-day momentum echo), so that target was noise dressed as a forecast — now
suppressed sub-band. The user wants a real target back, even at mild confidence. This
script tests whether a target built from POSITIONING + FLOW data actually beats the
naive "tomorrow closes where it closed today" (spot) baseline, OUT OF SAMPLE.

Label (per index, per day T, pooled across 4 indices, σ-normalised so it pools):
    y = (close[T+1] − close[T]) / σ20_pts        # next-day move in realised-σ units
Baseline forecast = 0 (spot). A model only earns trust if held-out MAE < MAE(spot).

Point-in-time features at T (all known at the T close; price direction via the
settle≡prev-close identity, verified in project memory, so no look-ahead):
    gap        (open−prevclose)/prevclose             price action
    clrange    (close−low)/(high−low)                 close location in day range
    oi_flow    signed ΣΔOI/ΣOI on index futures        FUTIDX flow (rollover-clean, fwd exp)
    s_fut      Murphy matrix sign (fresh long/short)    FUTIDX direction×OI
    maxpain    (max_pain−spot)/spot · 1/DTE            OI gravity (near-expiry pin)
    wall_asym  (call_room − put_room)/spot             OI-wall asymmetry (room to run)
    fii_dlt    Δ FII net index-fut OI (day/day)        institutional positioning change
    pro_dlt    Δ Pro net index-fut OI (day/day)        the directional desk tell
    basket     turnover-wtd Σ sign(FUTSTK flow)         bottom-up constituent futures

Validation: pool rows, sort by date, 70/30 train/test split. Standardise on TRAIN,
ridge fit, report held-out MAE vs spot baseline (skill = 1−MAE/MAE0), directional IC
and sign hit-rate, plus drop-one-out ΔMAE per feature (what each component is worth).

Usage:  python scripts/backtest_target_estimator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
IDX = {  # fno_symbol -> index_data.index_name
    "NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
    "FINNIFTY": "Nifty Financial Services", "MIDCPNIFTY": "Nifty Midcap Select",
}
TRAIN_FRAC = 0.70
RIDGE_ALPHA = 1.0
FEATURES = ["gap", "clrange", "oi_flow", "s_fut", "maxpain",
            "wall_asym", "fii_dlt", "pro_dlt", "basket"]


# ── feature builders ──────────────────────────────────────────────────────────
def _index_frame(con, sym: str, name: str) -> pd.DataFrame:
    df = con.execute(
        "select trade_date, open_val o, high_val h, low_val l, close_val c, prev_close pc "
        "from index_data where index_name = ? and close_val is not null order by trade_date",
        [name],
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["fno_symbol"] = sym
    ret = df["c"].pct_change()
    df["sigma_pts"] = ret.rolling(20).std() * df["c"]
    df["next_c"] = df["c"].shift(-1)
    df["y"] = (df["next_c"] - df["c"]) / df["sigma_pts"]
    df["gap"] = (df["o"] - df["pc"]) / df["pc"]
    rng = (df["h"] - df["l"]).replace(0, np.nan)
    df["clrange"] = (df["c"] - df["l"]) / rng - 0.5      # centre at 0
    return df


def _fut_idx_flow(con, sym: str) -> pd.DataFrame:
    # FUTIDX total OI + ΔOI over forward (active) expiries; price dir via settle≡prevclose.
    df = con.execute(
        "select trade_date, sum(open_interest) oi, sum(chg_in_oi) doi, "
        "       sum(close_price*open_interest) c_w, sum(settle_price*open_interest) s_w "
        "from fno_bhavcopy where instrument='FUTIDX' and symbol=? and open_interest>0 "
        "  and expiry_date > trade_date group by trade_date order by trade_date",
        [sym],
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    pr_dir = np.sign(df["c_w"] - df["s_w"])              # OI-wtd fut day move sign
    oip = df["doi"] / df["oi"].replace(0, np.nan)
    df["oi_flow"] = oip * pr_dir
    df["s_fut"] = pr_dir * np.where(df["doi"] > 0, 1.0, 0.5)
    return df[["trade_date", "oi_flow", "s_fut"]]


def _maxpain_walls(con, sym: str) -> pd.DataFrame:
    # near-expiry OPTIDX chain → max-pain gravity + wall asymmetry, per date.
    opt = con.execute(
        "with ne as (select trade_date, min(expiry_date) e from fno_bhavcopy "
        "  where instrument='OPTIDX' and symbol=? and open_interest>0 and expiry_date>trade_date "
        "  group by trade_date) "
        "select f.trade_date, f.expiry_date, f.strike_price k, f.option_type ot, "
        "       sum(f.open_interest) oi "
        "from fno_bhavcopy f join ne on f.trade_date=ne.trade_date and f.expiry_date=ne.e "
        "where f.instrument='OPTIDX' and f.symbol=? and f.open_interest>0 "
        "group by f.trade_date, f.expiry_date, f.strike_price, f.option_type",
        [sym, sym],
    ).df()
    if opt.empty:
        return pd.DataFrame(columns=["trade_date", "max_pain", "top_call", "top_put", "dte"])
    opt["trade_date"] = pd.to_datetime(opt["trade_date"]).dt.date
    opt["expiry_date"] = pd.to_datetime(opt["expiry_date"]).dt.date
    rows = []
    for td, g in opt.groupby("trade_date"):
        strikes = np.sort(g["k"].unique())
        ce = g[g.ot == "CE"].set_index("k")["oi"]
        pe = g[g.ot == "PE"].set_index("k")["oi"]
        # max pain = strike minimising total writer payout
        pain = []
        for K in strikes:
            call_pay = float((np.maximum(K - strikes, 0) * ce.reindex(strikes).fillna(0).values).sum())
            put_pay = float((np.maximum(strikes - K, 0) * pe.reindex(strikes).fillna(0).values).sum())
            pain.append(call_pay + put_pay)
        mp = float(strikes[int(np.argmin(pain))])
        top_call = float(ce.idxmax()) if not ce.empty else np.nan
        top_put = float(pe.idxmax()) if not pe.empty else np.nan
        dte = max((g["expiry_date"].iloc[0] - td).days, 1)
        rows.append((td, mp, top_call, top_put, dte))
    return pd.DataFrame(rows, columns=["trade_date", "max_pain", "top_call", "top_put", "dte"])


def _participant_delta(con) -> pd.DataFrame:
    df = con.execute(
        "select trade_date, client_type, (fut_idx_long-fut_idx_short) net "
        "from fao_participant where data_type='OI' and client_type in ('FII','Pro') "
        "order by trade_date",
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    p = df.pivot_table(index="trade_date", columns="client_type", values="net", aggfunc="first").sort_index()
    out = pd.DataFrame({"trade_date": p.index})
    out["fii_dlt"] = p["FII"].diff().values
    out["pro_dlt"] = p["Pro"].diff().values
    return out


def _basket(con) -> pd.DataFrame:
    # market-wide FUTSTK turnover-weighted matrix-sign flow (proxy: no float weights table).
    df = con.execute(
        "with ne as (select symbol, trade_date, min(expiry_date) e from fno_bhavcopy "
        "  where instrument='FUTSTK' and open_interest>0 and expiry_date>trade_date "
        "  group by symbol, trade_date) "
        "select f.trade_date, "
        "       sum(case when (f.close_price-f.settle_price)>0 and f.chg_in_oi>0 then f.value_lacs "
        "                when (f.close_price-f.settle_price)<0 and f.chg_in_oi>0 then -f.value_lacs "
        "                when (f.close_price-f.settle_price)>0 and f.chg_in_oi<0 then 0.5*f.value_lacs "
        "                when (f.close_price-f.settle_price)<0 and f.chg_in_oi<0 then -0.5*f.value_lacs "
        "                else 0 end) num, sum(f.value_lacs) den "
        "from fno_bhavcopy f join ne on f.symbol=ne.symbol and f.trade_date=ne.trade_date "
        "  and f.expiry_date=ne.e "
        "where f.instrument='FUTSTK' and f.open_interest>0 group by f.trade_date",
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["basket"] = df["num"] / df["den"].replace(0, np.nan)
    return df[["trade_date", "basket"]]


# ── ridge + scoring ─────────────────────────────────────────────────────────────
def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[0, 0] -= alpha                                   # don't penalise intercept
    w = np.linalg.solve(A, Xs.T @ y)
    return w, mu, sd


def _ridge_pred(X, w, mu, sd):
    Xs = (X - mu) / sd
    return np.column_stack([np.ones(len(Xs)), Xs]) @ w


def _report(tag, ytr, Xtr, yte, Xte, feats):
    w, mu, sd = _ridge_fit(Xtr, ytr, RIDGE_ALPHA)
    pred = _ridge_pred(Xte, w, mu, sd)
    mae0 = float(np.mean(np.abs(yte)))                 # spot baseline (forecast 0)
    mae = float(np.mean(np.abs(yte - pred)))
    skill = 1 - mae / mae0
    ic = float(np.corrcoef(pred, yte)[0, 1]) if np.std(pred) > 1e-9 else 0.0
    m = np.abs(pred) > 0.1
    hit = float(np.mean(np.sign(pred[m]) == np.sign(yte[m]))) if m.sum() > 5 else float("nan")
    print(f"{tag:9} n_te={len(yte):>4}  MAE {mae:.3f} vs spot {mae0:.3f}  "
          f"skill {skill:+.1%}  IC {ic:+.3f}  dirHit {100*hit:4.1f}% (n={int(m.sum())})")
    return w, mu, sd, mae, mae0


def main():
    con = duckdb.connect(str(DB), read_only=True)
    part = _participant_delta(con)
    bask = _basket(con)
    frames = []
    for sym, name in IDX.items():
        d = _index_frame(con, sym, name)
        d = d.merge(_fut_idx_flow(con, sym), on="trade_date", how="left")
        mw = _maxpain_walls(con, sym)
        d = d.merge(mw, on="trade_date", how="left")
        d = d.merge(part, on="trade_date", how="left")
        d = d.merge(bask, on="trade_date", how="left")
        d["maxpain"] = (d["max_pain"] - d["c"]) / d["c"] / d["dte"]
        d["wall_asym"] = ((d["top_call"] - d["c"]) - (d["c"] - d["top_put"])) / d["c"]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["y"] + FEATURES).reset_index(drop=True)
    # winsorise label tails so a couple of gap days don't dominate MAE
    df["y"] = df["y"].clip(-4, 4)

    print(f"\nTarget-estimator backtest — {len(df)} pooled (index,day) rows, "
          f"{df.trade_date.min()} → {df.trade_date.max()}")
    print("Label = next-day move in σ units; baseline = spot (forecast 0).")
    print("=" * 92)

    dts = np.sort(df.trade_date.unique())
    cut = dts[int(len(dts) * TRAIN_FRAC)]
    tr, te = df[df.trade_date < cut], df[df.trade_date >= cut]
    Xtr, ytr = tr[FEATURES].to_numpy(float), tr["y"].to_numpy(float)
    Xte, yte = te[FEATURES].to_numpy(float), te["y"].to_numpy(float)

    print(f"train {len(tr)} (→{cut})   test {len(te)} ({cut}→)\n")
    w, mu, sd, mae, mae0 = _report("POOLED", ytr, Xtr, yte, Xte, FEATURES)

    print("\nStandardised ridge weights (sign = direction, |mag| = pull):")
    for f, wv in sorted(zip(FEATURES, w[1:]), key=lambda x: -abs(x[1])):
        print(f"   {f:10} {wv:+.4f}")

    print("\nDrop-one-out (held-out MAE rise when feature removed → its worth):")
    for i, f in enumerate(FEATURES):
        keep = [j for j in range(len(FEATURES)) if j != i]
        wj, muj, sdj = _ridge_fit(Xtr[:, keep], ytr, RIDGE_ALPHA)
        pj = _ridge_pred(Xte[:, keep], wj, muj, sdj)
        maej = float(np.mean(np.abs(yte - pj)))
        flag = "  <-- helps" if maej > mae else ("  (noise)" if maej < mae - 1e-4 else "")
        print(f"   drop {f:10} MAE {maej:.3f}  Δ {maej-mae:+.4f}{flag}")

    print("\nPer-index held-out:")
    for sym in IDX:
        s_te = te[te.fno_symbol == sym]
        if len(s_te) < 15:
            continue
        ps = _ridge_pred(s_te[FEATURES].to_numpy(float), w, mu, sd)
        ys = s_te["y"].to_numpy(float)
        mae0s, maes = np.mean(np.abs(ys)), np.mean(np.abs(ys - ps))
        ics = float(np.corrcoef(ps, ys)[0, 1]) if np.std(ps) > 1e-9 else 0.0
        print(f"   {sym:11} n={len(s_te):>3}  MAE {maes:.3f} vs {mae0s:.3f}  "
              f"skill {1-maes/mae0s:+.1%}  IC {ics:+.3f}")

    print("\nRead: positive skill = beats spot. Keep only drop-one-out '<-- helps' features;")
    print("'(noise)' features should be dropped. If pooled skill ≤ 0 → no honest target,")
    print("keep the cockpit suppressed sub-band + scenario levels only.")


if __name__ == "__main__":
    main()
