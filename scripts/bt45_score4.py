import sys,os,math
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd,numpy as np
pd.set_option('display.width',250)
R=pd.read_pickle(r'C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay45.pkl'); R['up']=R['actual_ret']>0
print("nulls:", {c:int(R[c].isna().sum()) for c in ['em','rlo','rhi','tgt','bo_up','bo_dn','wk_lo','wk_hi','side_band']})

print("\n=== V. SIDEWAYS 'break side' lean (composite>=3 -> up-lean, <=-3 -> down-lean) ===")
s=R[R['direction']=='SIDEWAYS'].copy()
lean=s[s['composite'].abs()>=3]
print(f"n={len(lean)} of {len(s)} sideways calls carry a printed lean; "
      f"lean-side matched next-day sign {(((lean['composite']>0)==lean['up']).mean()*100):.1f}%  "
      f"signed bps={(np.sign(lean['composite'])*lean['actual_ret']).mean()*100:+.2f}")

print("\n=== W. BREAKOUT trigger levels — did a break actually extend? ===")
b=R.dropna(subset=['bo_up','bo_dn'])
up_br=b[b['next_close']>b['bo_up']]; dn_br=b[b['next_close']<b['bo_dn']]
print(f"upside break-level exceeded on {len(up_br)}/{len(b)} days; downside {len(dn_br)}/{len(b)}")

print("\n=== X. WEEKLY (to-expiry) range coverage — did close stay inside until expiry? ===")
w=R.dropna(subset=['wk_lo','wk_hi','wk_exp']).copy()
w['wk_exp']=pd.to_datetime(w['wk_exp']).dt.date
import duckdb
c=duckdb.connect('data/market_data.duckdb',read_only=True)
NAMES={"NIFTY":"Nifty 50","BANKNIFTY":"Nifty Bank","FINNIFTY":"Nifty Financial Services","MIDCPNIFTY":"Nifty Midcap Select"}
hist={s:c.execute("SELECT trade_date,close_val,high_val,low_val FROM index_data WHERE index_name=? AND trade_date>='2026-06-01'",[n]).df().assign(trade_date=lambda d:pd.to_datetime(d['trade_date']).dt.date) for s,n in NAMES.items()}
res=[]
for r in w.itertuples():
    h=hist[r.sym]; seg=h[(h['trade_date']>r.trade_date)&(h['trade_date']<=r.wk_exp)]
    if seg.empty: continue
    res.append(dict(sym=r.sym, close_in=bool(seg['close_val'].between(r.wk_lo,r.wk_hi).all()),
                    exp_close_in=bool(r.wk_lo<=seg['close_val'].iloc[-1]<=r.wk_hi),
                    touch_out=bool((seg['high_val']>r.wk_hi).any() or (seg['low_val']<r.wk_lo).any()), n_days=len(seg)))
W=pd.DataFrame(res)
print(W.groupby('sym').agg(n=('close_in','size'),all_closes_inside=('close_in','mean'),
     expiry_close_inside=('exp_close_in','mean'),intraday_touched_out=('touch_out','mean')).round(3).to_string())
print("POOLED", W[['close_in','exp_close_in','touch_out']].mean().round(3).to_dict())
