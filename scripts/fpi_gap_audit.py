"""Audit fpi_nsdl_flows for MISSING dates and MERGED (multi-day) prints."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, pandas as pd, numpy as np
pd.set_option('display.width', 220)
c = duckdb.connect('data/market_data.duckdb', read_only=True)

eq = c.execute("""SELECT trade_date, gross_purchase_cr, gross_sales_cr, net_investment_cr
                  FROM fpi_nsdl_flows WHERE category='Equity' ORDER BY trade_date""").df()
eq['trade_date'] = pd.to_datetime(eq['trade_date'])
td = c.execute(f"""SELECT DISTINCT trade_date FROM daily_data
                   WHERE trade_date >= '{eq['trade_date'].min():%Y-%m-%d}'
                     AND trade_date <= '{eq['trade_date'].max():%Y-%m-%d}'
                   ORDER BY 1""").df()
td['trade_date'] = pd.to_datetime(td['trade_date'])

print(f"FPI Equity rows: {len(eq)}  span {eq['trade_date'].min():%Y-%m-%d}..{eq['trade_date'].max():%Y-%m-%d}")
print(f"NSE trading days in that span: {len(td)}")
missing = sorted(set(td['trade_date']) - set(eq['trade_date']))
print(f"\n=== A. MISSING FROM NSDL ({len(missing)}) ===")
for m in missing:
    print(f"   {m:%Y-%m-%d} ({m:%a})")

print("\n=== B. MERGED / MULTI-DAY PRINTS (gross vs trailing-20 median) ===")
eq['turn'] = eq['gross_purchase_cr'] + eq['gross_sales_cr']
eq['med20'] = eq['turn'].rolling(20, min_periods=8).median().shift(1)
eq['x'] = eq['turn'] / eq['med20']
# a merged print follows a gap: mark rows whose PREVIOUS trading day is absent
tset = set(td['trade_date'])
prev_missing = []
for d in eq['trade_date']:
    prior = [x for x in sorted(tset) if x < d]
    prev_missing.append(bool(prior) and prior[-1] not in set(eq['trade_date']))
eq['prev_day_missing'] = prev_missing
flag = eq[eq['x'] >= 1.8].copy()
print(flag[['trade_date', 'gross_purchase_cr', 'gross_sales_cr', 'net_investment_cr',
            'med20', 'x', 'prev_day_missing']].round(2).to_string(index=False))

print("\n=== C. do the spikes line up with the gaps? ===")
print(f"   rows with x>=1.8            : {(eq['x']>=1.8).sum()}")
print(f"   of those, prev day missing  : {int((eq['x']>=1.8).mul(eq['prev_day_missing']).sum())}")
print(f"   gaps that produced a spike  : {int(eq['prev_day_missing'].mul(eq['x']>=1.8).sum())} of {int(eq['prev_day_missing'].sum())}")

print("\n=== D. first trading day of each month (is month-start structurally inflated?) ===")
eq['ym'] = eq['trade_date'].dt.to_period('M')
first = eq.groupby('ym').head(1)
print(first[['trade_date', 'gross_purchase_cr', 'gross_sales_cr', 'med20', 'x']].round(2).to_string(index=False))
rest = eq[~eq.index.isin(first.index)]
print(f"\n   median x, month-start rows : {first['x'].median():.2f}")
print(f"   median x, all other rows   : {rest['x'].median():.2f}")
