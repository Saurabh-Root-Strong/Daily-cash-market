"""Does a segment grouping explain 60-day return dispersion better than random labels?

usage: python scripts/validate_segments.py "<basic_industry>|<basic_industry>..." [days=400] [perms=200]
Results for each segmented sub-sector are recorded in config/segment_validation.csv.

Method
- Non-overlapping 60-trading-day log returns of liquid members (>= Rs 2 Cr/day).
- A stock is dropped only from the windows containing one of its split / bonus / demerger /
  rights ex-dates (unadjusted closes would fake a return there).
- R^2 of segment means vs the same statistic under random relabelings of identical group sizes;
  p = share of random runs >= actual.
- Reported twice: raw, and after regressing each window's returns on size (log median
  turnover over the prior 60 sessions, point-in-time) and prior-60-day momentum. A segment
  that only works raw is really a size/momentum effect. Several tests are run per
  sub-sector, so treat a lone p near 0.03 as unconfirmed.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

bis = sys.argv[1].split("|")
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PERMS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
DB = Path(__file__).resolve().parent.parent / 'data' / 'market_data.duckdb'
c = duckdb.connect(str(DB), read_only=True)
inlist = ','.join(repr(b) for b in bis)
seg = c.execute(f"""select symbol, segment, market_focus from stock_segments where symbol in
    (select symbol from v_stock_classification where basic_industry in ({inlist}))""").df().set_index('symbol')
liq = c.execute("""select symbol, avg(turnover_lacs)/100 cr from daily_data where series='EQ'
    and trade_date>(select max(trade_date)-90 from daily_data) group by 1""").df().set_index('symbol').cr
ca = c.execute(f"""select symbol, ex_date from corp_actions where kind in ('SPLIT','BONUS','DEMERGER','RIGHTS')
    and ex_date > (select max(trade_date) from daily_data) - {DAYS}""").df()
syms = ','.join(repr(s) for s in seg.index)
raw = c.execute(f"""select trade_date, symbol, close_price, turnover_lacs from daily_data where series='EQ'
    and trade_date >= (select max(trade_date) from daily_data) - {DAYS} and symbol in ({syms})""").df()
px = raw.pivot(index='trade_date', columns='symbol', values='close_price').sort_index()
tv = raw.pivot(index='trade_date', columns='symbol', values='turnover_lacs').sort_index()
keep = [s for s in px.columns if liq.get(s, 0) >= 2]
exd = pd.to_datetime(ca.ex_date)
dates = list(px.index[::20])
raw_w, ctl_w = [], []
for i in range(len(dates) - 3):
    a, b = dates[i], dates[i + 3]
    bad = set(ca[(exd > pd.Timestamp(a)) & (exd <= pd.Timestamp(b))].symbol)
    ret = (np.log(px.loc[b, keep]) - np.log(px.loc[a, keep])).dropna()
    ret = ret[~ret.index.isin(bad)]
    if len(ret) < 10:
        continue
    raw_w.append(ret)
    lo = px.index.get_loc(a)
    size = np.log(tv[keep].iloc[max(0, lo - 60):lo + 1].median() + 1).reindex(ret.index)
    mom = ((np.log(px.loc[a, keep]) - np.log(px.iloc[lo - 60][keep])).reindex(ret.index)
           if lo >= 60 else pd.Series(0.0, index=ret.index))
    X = pd.DataFrame({'c': 1.0, 'size': size, 'mom': mom}).fillna(0.0).values
    beta = np.linalg.lstsq(X, ret.values, rcond=None)[0]
    ctl_w.append(ret - X @ beta)


def r2(windows, labels):
    v = []
    for ret in windows:
        g = labels.reindex(ret.index)
        v.append(1 - ((ret - ret.groupby(g).transform('mean'))**2).sum() / ((ret - ret.mean())**2).sum())
    return float(np.mean(v))


rng = np.random.default_rng(0)
for col in ['segment', 'market_focus']:
    lab = seg[col].reindex(keep)
    for name, W in [('raw', raw_w), ('size+mom controlled', ctl_w)]:
        real = r2(W, lab)
        ctrl = [r2(W, pd.Series(rng.permutation(lab.values), index=keep)) for _ in range(PERMS)]
        p = (np.sum(np.array(ctrl) >= real) + 1) / (PERMS + 1)
        verdict = 'REAL' if p < 0.05 else 'not distinguishable from random'
        print(f"{col:13} {name:20} groups={lab.nunique():2}  explained={real:.1%}  "
              f"random={np.mean(ctrl):.1%} (95th {np.percentile(ctrl, 95):.1%})  p={p:.3f}  -> {verdict}")
print(f"window {DAYS}d: {len(raw_w)} non-overlapping 60-day windows, {len(keep)} stocks")
