"""Does a segment grouping explain 60-day return dispersion better than random labels?

usage: python scripts/validate_segments.py "<basic_industry>|<basic_industry>..." [days=400]
Results for each segmented sub-sector are recorded in config/segment_validation.csv.
A stock is dropped only from the 60-day windows that contain one of its split / bonus /
demerger / rights ex-dates (unadjusted closes would fake a return there).
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

bis = sys.argv[1].split("|")
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
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
px = c.execute(f"""select trade_date, symbol, close_price from daily_data where series='EQ'
    and trade_date >= (select max(trade_date) from daily_data) - {DAYS} and symbol in ({syms})""").df() \
    .pivot(index='trade_date', columns='symbol', values='close_price').sort_index()
keep = [s for s in px.columns if liq.get(s, 0) >= 2]
dates = list(px.index[::20])
windows = []
for i in range(len(dates) - 3):
    a, b = dates[i], dates[i + 3]
    exd = pd.to_datetime(ca.ex_date)
    bad = set(ca[(exd > pd.Timestamp(a)) & (exd <= pd.Timestamp(b))].symbol)
    ret = (np.log(px.loc[b, keep]) - np.log(px.loc[a, keep])).dropna()
    ret = ret[~ret.index.isin(bad)]
    if len(ret) >= 10:
        windows.append(ret)


def r2(labels):
    v = []
    for ret in windows:
        g = labels.reindex(ret.index)
        v.append(1 - ((ret - ret.groupby(g).transform('mean'))**2).sum() / ((ret - ret.mean())**2).sum())
    return float(np.mean(v))


rng = np.random.default_rng(0)
for col in ['segment', 'market_focus']:
    lab = seg[col].reindex(keep)
    real = r2(lab)
    ctrl = [r2(pd.Series(rng.permutation(lab.values), index=keep)) for _ in range(200)]
    p = (np.sum(np.array(ctrl) >= real) + 1) / (len(ctrl) + 1)
    verdict = 'REAL' if p < 0.05 else 'not distinguishable from random'
    print(f"{col:13} groups={lab.nunique():2}  explained={real:.1%}  random={np.mean(ctrl):.1%} "
          f"(95th {np.percentile(ctrl, 95):.1%})  p={p:.3f}  -> {verdict}")
print(f"window {DAYS}d: {len(windows)} non-overlapping 60-day windows, {len(keep)} stocks")
