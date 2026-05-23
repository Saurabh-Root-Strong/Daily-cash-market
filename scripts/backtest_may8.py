"""
Backtest: signals as of May 8, 2026 -> actual returns by May 19, 2026.

For each sector the rotation engine marked BUY on May 8:
  - Pull stocks for that sector
  - Compute conviction using same sector-relative quantile logic as dashboard
  - Keep Strong + Buying stocks only
  - Compare May 8 close vs May 19 close
  - Show P&L %
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from datetime import date

import pandas as pd

from src.analytics.sector_rotation import get_sector_rotation, get_sector_stocks_rotation
from src.data.repository import query_dataframe

SIGNAL_DATE  = date(2026, 5, 8)
CHECK_DATE   = date(2026, 5, 19)   # latest available date in DB
MIN_TURNOVER = 5.0


def get_close_prices(symbols: list, trade_date: date) -> dict:
    """Return {symbol: close_price} for the given date."""
    if not symbols:
        return {}
    ph = ", ".join("?" * len(symbols))
    df = query_dataframe(
        f"SELECT symbol, close_price FROM daily_data WHERE trade_date = ? AND symbol IN ({ph})",
        [trade_date] + list(symbols),
    )
    return {} if df.empty else dict(zip(df["symbol"], df["close_price"]))


def _assign_conviction(stocks: pd.DataFrame) -> pd.DataFrame:
    """Replicate dashboard's sector-relative conviction logic for INVEST sectors."""
    stocks = stocks.copy()
    valid_deliv = stocks["wtd_deliv_per"].dropna()
    if len(valid_deliv) >= 3:
        hi = float(valid_deliv.quantile(0.67))
        lo = float(valid_deliv.quantile(0.33))
    else:
        hi = float(valid_deliv.max())
        lo = float(valid_deliv.min())

    def _conv(r):
        d = float(r["wtd_deliv_per"]) if pd.notna(r["wtd_deliv_per"]) else 0.0
        p = float(r["price_chg_pct"]) if pd.notna(r["price_chg_pct"]) else 0.0
        if d >= hi and p < 0:
            return "Strong"
        elif d >= hi:
            return "Buying"
        elif d >= lo:
            return "Watch"
        else:
            return "Weak"

    stocks["conviction"] = stocks.apply(_conv, axis=1)
    return stocks


def main():
    print()
    print("=" * 75)
    print(f"  BACKTEST: Signals on {SIGNAL_DATE}  ->  Returns by {CHECK_DATE}")
    print("=" * 75)

    # 1. Sector rotation on signal date
    rotation = get_sector_rotation(SIGNAL_DATE, min_turnover_lacs=MIN_TURNOVER)
    buy_sectors = rotation[rotation["action"].str.startswith("BUY")].copy()

    print(f"\nBUY sectors on {SIGNAL_DATE}: {len(buy_sectors)}")
    for _, row in buy_sectors.sort_values("accum_score", ascending=False).iterrows():
        sig = row["signal"].encode("ascii", "replace").decode()
        print(f"  {row['sector']:35s}  score={row['accum_score']:4.0f}  {sig}")

    print()

    all_picks = []

    for _, sec_row in buy_sectors.iterrows():
        sector = sec_row["sector"]
        stocks = get_sector_stocks_rotation(sector, SIGNAL_DATE, min_turnover_lacs=MIN_TURNOVER)
        if stocks.empty:
            continue

        stocks = _assign_conviction(stocks)
        picks = stocks[stocks["conviction"].isin(["Strong", "Buying"])].copy()
        if picks.empty:
            continue

        syms          = picks["symbol"].tolist()
        prices_entry  = get_close_prices(syms, SIGNAL_DATE)
        prices_exit   = get_close_prices(syms, CHECK_DATE)

        for _, st in picks.iterrows():
            sym   = st["symbol"]
            entry = prices_entry.get(sym, 0.0)
            exit_ = prices_exit.get(sym, 0.0)
            pnl   = (exit_ - entry) / entry * 100 if (entry > 0 and exit_ > 0) else float("nan")

            all_picks.append({
                "sector":     sector,
                "symbol":     sym,
                "company":    str(st.get("company_name", ""))[:28],
                "conviction": st["conviction"],
                "deliv_pct":  st.get("wtd_deliv_per", 0.0),
                "ltp_may8":   entry,
                "ltp_may19":  exit_,
                "pnl_pct":    pnl,
            })

    if not all_picks:
        print("No picks found.")
        return

    # Sort: winners first, losers second, no-data last
    all_picks.sort(
        key=lambda r: (-r["pnl_pct"] if not math.isnan(r["pnl_pct"]) else -9999)
    )

    # 2. Print results table
    print(f"{'STOCK-LEVEL RESULTS':^75}")
    print(f"{'Symbol':10s}  {'Company':28s}  {'Sector':22s}  {'Conv':7s}  "
          f"{'Deliv%':>6s}  {'May 8':>7s}  {'May19':>7s}  {'P&L%':>8s}")
    print("-" * 112)

    winners = losers = no_data = 0
    total_pnl = 0.0

    for r in all_picks:
        if math.isnan(r["pnl_pct"]):
            pnl_str = "  NO DATA"
            no_data += 1
        elif r["pnl_pct"] >= 0:
            pnl_str = f"+{r['pnl_pct']:7.2f}%"
            winners += 1
            total_pnl += r["pnl_pct"]
        else:
            pnl_str = f"{r['pnl_pct']:8.2f}%"
            losers += 1
            total_pnl += r["pnl_pct"]

        print(f"{r['symbol']:10s}  {r['company']:28s}  {r['sector'][:22]:22s}  "
              f"{r['conviction']:7s}  {r['deliv_pct']:6.1f}%  "
              f"{r['ltp_may8']:7.2f}  {r['ltp_may19']:7.2f}  {pnl_str}")

    # 3. Summary
    valid = winners + losers
    print()
    print("=" * 75)
    print("  BACKTEST SUMMARY")
    print(f"  Period         : {SIGNAL_DATE}  ->  {CHECK_DATE}  (~8 trading days)")
    print(f"  Total picks    : {len(all_picks)}")
    print(f"  With data      : {valid}")
    print(f"  No data        : {no_data}")
    if valid:
        print(f"  Winners (>=0%) : {winners}  ({100*winners/valid:.0f}%)")
        print(f"  Losers (<0%)   : {losers}  ({100*losers/valid:.0f}%)")
        print(f"  Avg P&L        : {total_pnl/valid:+.2f}%")
        top3 = sorted([r for r in all_picks if not math.isnan(r["pnl_pct"])],
                      key=lambda r: -r["pnl_pct"])[:3]
        bot3 = sorted([r for r in all_picks if not math.isnan(r["pnl_pct"])],
                      key=lambda r: r["pnl_pct"])[:3]
        if top3:
            print(f"\n  Top 3 picks:")
            for r in top3:
                print(f"    {r['symbol']:10s}  {r['sector']:22s}  {r['pnl_pct']:+.2f}%")
        if bot3:
            print(f"\n  Worst 3 picks:")
            for r in bot3:
                print(f"    {r['symbol']:10s}  {r['sector']:22s}  {r['pnl_pct']:+.2f}%")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
