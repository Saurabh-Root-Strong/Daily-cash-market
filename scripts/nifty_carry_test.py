"""The heavyweight-carry premise, tested WITHOUT needing index weights.

cap-weighted return = the index itself (exact, index_data)
equal-weighted return = mean of the 50 constituents (exact)
carry spread = index - equalweight  -> POSITIVE means heavyweights are carrying it,
NEGATIVE means the broad basket is beating the index. This is the Univest
screenshot's +0.24% vs +0.09% observation, computed exactly and point-in-time.

Q1 does the spread exist and how big is it?
Q2 does TODAY's spread predict TOMORROW / next week's Nifty?
Q3 does breadth (adv/dec among the 50) predict it?
Q4 control: is any of it just Nifty's own momentum?
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd
pd.set_option('display.width', 220)
c = duckdb.connect('data/market_data.duckdb', read_only=True)
SYMS = ("ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
        "BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL","CIPLA","COALINDIA","DRREDDY",
        "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
        "HINDUNILVR","ICICIBANK","ITC","INFY","INDIGO","JSWSTEEL","JIOFIN","KOTAKBANK",
        "LT","M&M","MARUTI","MAXHEALTH","NTPC","NESTLEIND","ONGC","POWERGRID",
        "RELIANCE","SBILIFE","SHRIRAMFIN","SBIN","SUNPHARMA","TCS","TATACONSUM",
        "TMPV","TATASTEEL","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO")
ph = ",".join("?"*len(SYMS))
d = c.execute(f"""SELECT trade_date, symbol,
    (close_price-prev_close)/NULLIF(prev_close,0)*100 r, deliv_per, turnover_lacs
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND prev_close>0 AND trade_date>='2022-01-01'""", list(SYMS)).df()
d['trade_date']=pd.to_datetime(d['trade_date'])
R=d.pivot_table('r','trade_date','symbol').sort_index()
DL=d.pivot_table('deliv_per','trade_date','symbol').sort_index()
nif=c.execute("SELECT trade_date,pct_chg FROM index_data WHERE index_name='Nifty 50' AND trade_date>='2022-01-01'").df()
nif['trade_date']=pd.to_datetime(nif['trade_date']); nif=nif.set_index('trade_date').sort_index()
ix=R.index.intersection(nif.index)
R=R.loc[ix]; DL=DL.loc[ix]; y=nif.loc[ix,'pct_chg'].astype(float)
cov=R.notna().sum(axis=1)
R=R[cov>=45]; DL=DL.loc[R.index]; y=y.loc[R.index]
ew=R.mean(axis=1)
spread=y-ew
adv=(R>0).sum(axis=1)/R.notna().sum(axis=1)*100
print(f"panel {len(R)} sessions {R.index.min():%Y-%m-%d}..{R.index.max():%Y-%m-%d}, >=45 names/day")
print(f"\nQ1. carry spread (index - equalweight), pp/day")
print(f"  mean {spread.mean():+.4f}  sd {spread.std():.4f}  "
      f"|spread|>0.25pp on {(spread.abs()>0.25).mean()*100:.1f}% of days")
print(f"  days index UP while <50% of the 50 advanced: "
      f"{((y>0)&(adv<50)).sum()} of {len(y)} ({((y>0)&(adv<50)).mean()*100:.1f}%)")
print(f"  days index DOWN while >50% advanced:         "
      f"{((y<0)&(adv>50)).sum()} ({((y<0)&(adv>50)).mean()*100:.1f}%)")

def nw(x,lag):
    x=np.asarray(pd.Series(x).dropna(),float);n=len(x)
    if n<25: return float('nan')
    e=x-x.mean();v=(e@e)/n
    for L in range(1,lag+1): v+=2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    return x.mean()/math.sqrt(v/n) if v>0 else float('nan')

lg=np.log1p(y/100.0)
print("\nQ2/Q3. FORWARD TEST — does today's carry/breadth predict the index?")
print(f"{'signal':<26}{'h':>3}{'IC':>9}{'signed%':>10}{'NW-t':>8}{'hit%':>8}")
for nm,sig in [('carry spread', spread), ('breadth adv%', adv-50),
               ('delivery avg (z)', (DL.mean(axis=1)-DL.mean(axis=1).rolling(60).mean())
                                     /DL.mean(axis=1).rolling(60).std()),
               ('CONTROL nifty ret', y)]:
    for h in (1,5,10):
        f=(np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1))*100.0)
        k=pd.DataFrame({'s':sig,'f':f}).dropna()
        if len(k)<50: continue
        ic=k['s'].rank().corr(k['f'].rank()); sg=np.sign(k['s'])*k['f']
        print(f"{nm:<26}{h:>3}{ic:>+9.3f}{sg.mean():>+10.3f}{nw(sg,h):>+8.2f}{(sg>0).mean()*100:>8.1f}")
