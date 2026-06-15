# Model Card — Daily_Cash_Market

A post-market NSE analytics platform (delivery, F&O, FII/DII, sector rotation,
index prediction). This card states **what each model is for, its measured
performance, and its limitations** so outputs are never trusted beyond their
evidence. It is decision-support, **not** an execution/trading system.

## Intended use & out-of-scope

- **In scope:** end-of-day situational awareness — where institutional delivery
  flow is concentrating, sector rotation phase, expected-move ranges, OI/level
  maps, regime context.
- **Out of scope:** automated order generation, intraday signals, position
  sizing, or any use that treats a directional "score" as a tradable edge.
  There is **no** risk engine (sizing, stops, drawdown control, kill switch).

## Components & measured performance

Measured over the logged history (377 trading days, 2024-12-02 → 2026-06-12;
1,508 index-prediction rows). Methodology: walk-forward, point-in-time,
forward-return as the out-of-sample outcome.

| Component | What it does | Measured result | Trust level |
|---|---|---|---|
| Index direction | Next-day UP/DOWN/SIDEWAYS per index | Pooled hit 50%, IC ≈ +0.019 (NIFTY 47%) | **None** — coin-flip; do not trade direction |
| Expected-move range | 68–75% next-day band | ~74.5% realized coverage vs ~75% intended | Calibrated, but ≈ trailing vol (table stakes, not alpha) |
| Sector rotation | Cross-sectional accumulation/rotation score | RS-2W factor IC ≈ +0.10 (walk-forward) | Plausible edge; **single-regime sample, not yet holdout-validated** |
| Sector memory | Analog recall of past setups | Persistent down-capture/accumulation | Leakage-controlled; small history |
| Index memory | Analog recall (Signal 24) | Standalone next-day IC ≈ −0.2 | Context only, not a forecast booster |

**Headline honesty:** the directional engine has no measured edge and the UI
says so (per-index "reality check"). The durable value is the range/levels/risk
map and the cross-sectional sector work.

## Data lineage

- **Source:** NSE EOD publications (bhavcopy, F&O UDiFF bhavcopy, FAO participant
  OI, FII derivatives stats, indices) + NSDL FPI. **Single vendor, no redundancy.**
- **Store:** DuckDB, point-in-time (queries key on `trade_date <= as_of`).
- **History:** ~377 F&O days / ~358 cash days (short by institutional standards).

## Known limitations / biases

1. **Survivorship + classification bias:** historical sector baskets use the
   *current* F&O universe and *current* `v_sector_master` sector labels applied
   to past dates. Delisted/removed names are absent; reclassified names use
   today's sector.
2. **Single regime:** the full sample is one broad regime — momentum-tilted
   factor weights may not hold out-of-sample.
3. **No corporate-action audit:** ~98.3% prev-close continuity; the residual is
   likely unadjusted corp actions.
4. **F&O format dependency:** ingestion tracks NSE's UDiFF format; an NSE format
   change requires a fetcher update (see `fno_bhavcopy_fetcher.py`).

## Validation status

- ✅ Point-in-time / leakage control (verified across analytics).
- ✅ Regression tests for parsers + signal classifiers (`tests/test_fno_signals.py`).
- ⚠️ **No locked holdout, no pre-registration, no multiple-testing correction.**
  Directional results are confirmatory only; sector-rotation edge needs a true
  out-of-sample + net-of-cost evaluation before it is relied upon.

## Versioning

Models evolve with the code. To reproduce a past call, check out the git commit
active on that prediction's date. (Planned: stamp each `prediction_log` row with
its producing git SHA for exact lineage.)
