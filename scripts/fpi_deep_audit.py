"""FPI subsystem deep audit — semantics, integrity, and whether the shipped
15-20 day outlook has any forward edge.

Q1 COVERAGE     gaps + merged prints in fpi_nsdl_flows AND fii_dii_cash
Q2 SEMANTICS    is the NSDL date a TRADE date or a SETTLEMENT/report date?
                cross-check against NSE's own provisional FII cash number at
                lag -1/0/+1. NSE's date is unambiguously the trade date.
Q3 CALIBRATION  does FPI equity net == FII cash net? (they measure almost the
                same thing; a level/scale break means one is not what it says)
Q4 EDGE (short) does the shipped outlook score predict the next 15-20 sessions,
                on the 85 days actually in the table?
Q5 EDGE (long)  same premise tested on 600 days of fii_dii_cash as a proxy, so
                the answer is not decided by a 4-month sample.
Q6 THRESHOLDS   are the +-12000/5000/1500 Cr score cuts calibrated to anything?
Inference: Newey-West at lag=h; forward windows overlap.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option('display.width', 230)
c = duckdb.connect('data/market_data.duckdb', read_only=True)

fpi = c.execute("""SELECT trade_date, category, gross_purchase_cr, gross_sales_cr,
                          net_investment_cr FROM fpi_nsdl_flows""").df()
fpi['trade_date'] = pd.to_datetime(fpi['trade_date'])
eq = fpi[fpi.category == 'Equity'].set_index('trade_date').sort_index()
cash = c.execute("SELECT trade_date, fii_net, fii_buy, fii_sell FROM fii_dii_cash").df()
cash['trade_date'] = pd.to_datetime(cash['trade_date'])
cash = cash.set_index('trade_date').sort_index()
nif = c.execute("""SELECT trade_date, close_val, pct_chg FROM index_data
                   WHERE index_name='Nifty 50'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date'])
nif = nif.set_index('trade_date').sort_index()
td = c.execute("SELECT DISTINCT trade_date FROM daily_data").df()
td['trade_date'] = pd.to_datetime(td['trade_date'])
tds = set(td['trade_date'])


def nw_t(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 25: return float('nan')
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float('nan')


print("=" * 92)
print("Q1. COVERAGE — gaps and merged prints in BOTH flow tables")
print("=" * 92)
for name, s in [("fpi_nsdl_flows (Equity)", eq), ("fii_dii_cash", cash)]:
    span = [d for d in sorted(tds) if s.index.min() <= d <= s.index.max()]
    miss = [d for d in span if d not in set(s.index)]
    turn = (s['gross_purchase_cr'] + s['gross_sales_cr']) if 'gross_purchase_cr' in s \
        else (s['fii_buy'] + s['fii_sell'])
    med = turn.rolling(20, min_periods=8).median().shift(1)
    x = turn / med
    print(f"\n  {name}: {len(s)} rows, {s.index.min():%Y-%m-%d}..{s.index.max():%Y-%m-%d}, "
          f"{len(span)} NSE sessions in span")
    print(f"    missing dates ({len(miss)}): {[f'{d:%Y-%m-%d}' for d in miss][:12]}")
    print(f"    turnover >=1.8x trailing median: {int((x >= 1.8).sum())} rows "
          f"-> {[f'{d:%Y-%m-%d}({v:.1f}x)' for d, v in x[x >= 1.8].items()][:8]}")

print("\n" + "=" * 92)
print("Q2. SEMANTICS — is the NSDL date a TRADE date or a SETTLEMENT/report date?")
print("=" * 92)
print("  NSE's fii_dii_cash date IS the trade date. If NSDL uses the same convention,")
print("  correlation peaks at lag 0. A peak at lag +1 means NSDL date = trade date + 1.")
j = pd.DataFrame({'fpi': eq['net_investment_cr'], 'nse': cash['fii_net']}).dropna()
print(f"\n  overlapping dates: {len(j)}")
for lag in (-2, -1, 0, 1, 2):
    a = eq['net_investment_cr']
    b = cash['fii_net'].shift(lag)          # lag>0 -> NSE value from `lag` days EARLIER
    k = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(k) < 20: continue
    print(f"    NSDL(t) vs NSE(t-{lag:+d}) : corr {k['a'].corr(k['b']):+.4f}   n={len(k)}")

print("\n  same test against Nifty's own return:")
for lag in (-2, -1, 0, 1, 2):
    k = pd.DataFrame({'a': eq['net_investment_cr'], 'b': nif['pct_chg'].shift(lag)}).dropna()
    if len(k) < 20: continue
    print(f"    NSDL(t) vs Nifty ret(t-{lag:+d}) : corr {k['a'].corr(k['b']):+.4f}   n={len(k)}")
print("\n  control — NSE cash vs Nifty return (known-aligned, sets the benchmark):")
for lag in (-1, 0, 1):
    k = pd.DataFrame({'a': cash['fii_net'], 'b': nif['pct_chg'].shift(lag)}).dropna()
    print(f"    NSE(t) vs Nifty ret(t-{lag:+d}) : corr {k['a'].corr(k['b']):+.4f}   n={len(k)}")

print("\n" + "=" * 92)
print("Q3. CALIBRATION — do the two sources agree on level?")
print("=" * 92)
if len(j) > 20:
    d = j['fpi'] - j['nse']
    print(f"  n={len(j)}  corr={j['fpi'].corr(j['nse']):+.4f}")
    print(f"  mean NSDL {j['fpi'].mean():+9.1f} Cr   mean NSE {j['nse'].mean():+9.1f} Cr")
    print(f"  mean diff {d.mean():+9.1f} Cr   sd of diff {d.std():9.1f} Cr")
    print(f"  regression NSDL = a + b*NSE : b={np.polyfit(j['nse'], j['fpi'], 1)[0]:+.3f}")

print("\n" + "=" * 92)
print("Q4. FORWARD EDGE of the SHIPPED outlook — on the 85 days actually in the table")
print("=" * 92)
piv = fpi.pivot_table(index='trade_date', columns='category',
                      values='net_investment_cr', aggfunc='sum', fill_value=0)
piv['equity_net'] = piv.get('Equity', 0)
piv['debt_net'] = piv.get('Debt', 0) + piv.get('Debt-VRR', 0)
piv['hybrid_net'] = piv.get('Hybrid', 0)
den = piv['equity_net'].abs() + piv['debt_net'].abs() + piv['hybrid_net'].abs() + 1e-9
piv['risk_score'] = piv['equity_net'] / den * 100


def outlook_score(eq15, dbt15, avg_risk):
    if eq15 > 12000: s = 3
    elif eq15 > 5000: s = 2
    elif eq15 > 1500: s = 1
    elif eq15 < -12000: s = -3
    elif eq15 < -5000: s = -2
    elif eq15 < -1500: s = -1
    else: s = 0
    if avg_risk > 60: r = 2
    elif avg_risk > 40: r = 1
    elif avg_risk < -20: r = -2
    elif avg_risk < 10: r = -1
    else: r = 0
    cf = -2 if (eq15 < -2000 and dbt15 < -2000) else 0
    return s + r + cf


piv['eq15'] = piv['equity_net'].rolling(15).sum()
piv['dbt15'] = piv['debt_net'].rolling(15).sum()
piv['risk15'] = piv['risk_score'].rolling(15).mean()
piv['score'] = [outlook_score(a, b, r) if pd.notna(a) else np.nan
                for a, b, r in zip(piv['eq15'], piv['dbt15'], piv['risk15'])]
nret = nif['pct_chg'].reindex(piv.index)
lgn = np.log1p(nif['pct_chg'] / 100.0)
for h in (5, 10, 15, 20):
    f = (np.expm1(lgn.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0).reindex(piv.index)
    k = pd.DataFrame({'s': piv['score'], 'f': f}).dropna()
    if len(k) < 25:
        print(f"  h={h:2d}: n={len(k)} — too few to test"); continue
    ic = k['s'].rank().corr(k['f'].rank())
    signed = np.sign(k['s']) * k['f']
    print(f"  h={h:2d}d  n={len(k):3d}  IC(spearman) {ic:+.3f}  "
          f"signed mean {signed.mean():+.3f}%  NW-t {nw_t(signed, h):+.2f}  "
          f"hit {(signed > 0).mean() * 100:.1f}%")

print("\n" + "=" * 92)
print("Q5. THE SAME PREMISE ON 600 DAYS — 'cumulative 15d foreign equity flow predicts")
print("    the next 15-20 sessions', tested on NSE's own FII cash series")
print("=" * 92)
cs = cash.copy()
cs['eq15'] = cs['fii_net'].rolling(15).sum()
for h in (5, 10, 15, 20):
    f = (np.expm1(lgn.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0).reindex(cs.index)
    k = pd.DataFrame({'s': cs['eq15'], 'f': f}).dropna()
    ic = k['s'].rank().corr(k['f'].rank())
    signed = np.sign(k['s']) * k['f']
    print(f"  h={h:2d}d  n={len(k):3d}  IC(spearman) {ic:+.3f}  "
          f"signed mean {signed.mean():+.3f}%  NW-t {nw_t(signed, h):+.2f}  "
          f"hit {(signed > 0).mean() * 100:.1f}%")
print("\n  quintiles of 15d cumulative FII cash flow -> next 20 sessions:")
f20 = (np.expm1(lgn.iloc[::-1].rolling(20).sum().iloc[::-1].shift(-1)) * 100.0).reindex(cs.index)
k = pd.DataFrame({'s': cs['eq15'], 'f': f20}).dropna()
k['q'] = pd.qcut(k['s'], 5, labels=False)
print(k.groupby('q').agg(n=('f', 'size'), flow_Cr=('s', 'mean'),
                         fwd20=('f', 'mean'), up=('f', lambda v: (v > 0).mean() * 100)).round(2).to_string())

print("\n" + "=" * 92)
print("Q6. THRESHOLDS — where do +-12000 / 5000 / 1500 Cr sit in the actual distribution?")
print("=" * 92)
for nm, s in [("fpi_nsdl 15d equity (85d table)", piv['eq15'].dropna()),
              ("fii_dii_cash 15d (600d)", cs['eq15'].dropna())]:
    q = s.quantile([.05, .25, .5, .75, .95])
    print(f"  {nm}: p5 {q.iloc[0]:>9.0f}  p25 {q.iloc[1]:>9.0f}  med {q.iloc[2]:>9.0f}  "
          f"p75 {q.iloc[3]:>9.0f}  p95 {q.iloc[4]:>9.0f}")
    for t in (1500, 5000, 12000):
        print(f"      |15d| > {t:>6}: {(s.abs() > t).mean() * 100:5.1f}% of days   "
              f"(>{t}: {(s > t).mean() * 100:4.1f}%,  <-{t}: {(s < -t).mean() * 100:4.1f}%)")

print("\n" + "=" * 92)
print("Q7. CONFIRMING THE OFF-BY-ONE — regression at the winning lag, outliers removed")
print("=" * 92)
turn = eq['gross_purchase_cr'] + eq['gross_sales_cr']
medt = turn.rolling(20, min_periods=8).median().shift(1)
clean = eq[(turn / medt < 1.8).reindex(eq.index).fillna(True)]
print(f"  dropped {len(eq)-len(clean)} merged/month-start rows -> n={len(clean)}")
for lag in (0, 1):
    k = pd.DataFrame({'a': clean['net_investment_cr'],
                      'b': cash['fii_net'].shift(lag)}).dropna()
    if len(k) < 20: continue
    b1, b0 = np.polyfit(k['b'], k['a'], 1)
    r = k['a'].corr(k['b'])
    resid = k['a'] - (b1 * k['b'] + b0)
    print(f"    NSDL(t) = a + b*NSE(t-{lag}):  b={b1:+.3f}  a={b0:+8.1f}  "
          f"corr={r:+.4f}  R2={r**2:.3f}  resid sd={resid.std():.0f} Cr")
print("\n  => b~1.0 and R2 near 1 at the TRUE alignment; the other lag is noise.")

print("\n" + "=" * 92)
print("Q8. WHAT THE OFF-BY-ONE DOES TO THE PAGE")
print("=" * 92)
k = pd.DataFrame({'nsdl': clean['net_investment_cr'],
                  'nse_same': cash['fii_net'],
                  'nse_prev': cash['fii_net'].shift(1)}).dropna()
wrong = (np.sign(k['nsdl']) != np.sign(k['nse_same'])).mean() * 100
right = (np.sign(k['nsdl']) != np.sign(k['nse_prev'])).mean() * 100
print(f"  If you read the NSDL row as 'what FPIs did TODAY':")
print(f"    sign disagrees with NSE's actual same-day FII cash on {wrong:.1f}% of days")
print(f"    sign disagrees with NSE's PREVIOUS day             on {right:.1f}% of days")
print(f"  n={len(k)}")
