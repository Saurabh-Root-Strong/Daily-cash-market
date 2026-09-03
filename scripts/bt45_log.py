"""45-session log-based backtest of the Index Prediction engine (real-time record)."""
import duckdb, numpy as np, pandas as pd
import math

c = duckdb.connect('data/market_data.duckdb', read_only=True)
df = c.execute("""
    SELECT trade_date, fno_symbol, direction_pred, confidence_pred, composite_score,
           spot_close, range_low, range_high, target_close, expected_move_pts,
           actual_return, direction_actual, was_correct, outcome_filled, created_at,
           hmm_state, memory_label
    FROM prediction_log ORDER BY trade_date
""").df()
df['trade_date'] = pd.to_datetime(df['trade_date'])

dates = sorted(df['trade_date'].unique())
last45 = dates[-46:-1]           # exclude the newest date (28-Aug: no outcome yet)
w = df[df['trade_date'].isin(last45)].copy()
print(f"window: {str(last45[0])[:10]} .. {str(last45[-1])[:10]}  sessions={len(last45)}  rows={len(w)}")
print(f"unfilled in window: {(~w['outcome_filled'].astype(bool)).sum()}")
print(f"live-logged (created_at > trade_date, real OOS): {(w['created_at'] > w['trade_date']).sum()}/{len(w)}")
print()

def binom_p(k, n, p=0.5):
    if not n: return float('nan')
    k=int(k); n=int(n)
    pmf=[math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(n+1)]
    obs=pmf[k]
    return min(1.0, sum(v for v in pmf if v<=obs*(1+1e-9)))

rows = []
for sym, g in w.groupby('fno_symbol'):
    g = g.dropna(subset=['actual_return'])
    n = len(g)
    # --- 3-class engine scoring (its own was_correct) ---
    acc3 = g['was_correct'].astype(bool).mean()*100
    # --- directional subset ---
    d = g[g['direction_pred'].isin(['UP','DOWN'])]
    nd = len(d)
    hit3 = d['was_correct'].astype(bool).mean()*100 if nd else np.nan
    sign = ((d['direction_pred']=='UP') == (d['actual_return']>0)).mean()*100 if nd else np.nan
    # --- sideways subset ---
    s = g[g['direction_pred']=='SIDEWAYS']
    sw = s['was_correct'].astype(bool).mean()*100 if len(s) else np.nan
    # --- base rates ---
    up_rate = (g['actual_return']>0).mean()*100
    # --- IC ---
    ic = g['composite_score'].rank().corr(g['actual_return'].rank())
    # --- range coverage ---
    cn = g.dropna(subset=['range_low','range_high','spot_close'])
    nxt = cn['spot_close']*(1+cn['actual_return']/100)
    cov = ((nxt>=cn['range_low'])&(nxt<=cn['range_high'])).mean()*100 if len(cn) else np.nan
    # --- target skill vs naive (spot unchanged) ---
    t = g.dropna(subset=['target_close','spot_close'])
    nt = t['spot_close']*(1+t['actual_return']/100)
    mae_t = (t['target_close']-nt).abs().mean()
    mae_0 = (t['spot_close']-nt).abs().mean()
    skill = (1-mae_t/mae_0)*100 if mae_0 else np.nan
    rows.append(dict(sym=sym, n=n, n_dir=nd, acc3=acc3, dir_hit3=hit3, dir_sign=sign,
                     p_sign=binom_p(round(sign/100*nd), nd) if nd else np.nan,
                     sw_n=len(s), sw_acc=sw, up_base=up_rate, IC=ic, cov1s=cov,
                     tgt_skill=skill, mae_t=mae_t, mae_0=mae_0))
r = pd.DataFrame(rows)
# pooled
g = w.dropna(subset=['actual_return'])
d = g[g['direction_pred'].isin(['UP','DOWN'])]
s = g[g['direction_pred']=='SIDEWAYS']
cn = g.dropna(subset=['range_low','range_high','spot_close']); nxt = cn['spot_close']*(1+cn['actual_return']/100)
t = g.dropna(subset=['target_close','spot_close']); ntc = t['spot_close']*(1+t['actual_return']/100)
sign_p = ((d['direction_pred']=='UP')==(d['actual_return']>0)).mean()*100 if len(d) else np.nan
r.loc[len(r)] = dict(sym='POOLED', n=len(g), n_dir=len(d), acc3=g['was_correct'].astype(bool).mean()*100,
    dir_hit3=d['was_correct'].astype(bool).mean()*100 if len(d) else np.nan, dir_sign=sign_p,
    p_sign=binom_p(round(sign_p/100*len(d)), len(d)) if len(d) else np.nan,
    sw_n=len(s), sw_acc=s['was_correct'].astype(bool).mean()*100 if len(s) else np.nan,
    up_base=(g['actual_return']>0).mean()*100,
    IC=g['composite_score'].rank().corr(g['actual_return'].rank()),
    cov1s=((nxt>=cn['range_low'])&(nxt<=cn['range_high'])).mean()*100,
    tgt_skill=(1-(t['target_close']-ntc).abs().mean()/(t['spot_close']-ntc).abs().mean())*100,
    mae_t=(t['target_close']-ntc).abs().mean(), mae_0=(t['spot_close']-ntc).abs().mean())
pd.set_option('display.width', 250)
print(r.round(2).to_string(index=False))
print()
print("--- verdict mix per index ---")
print(pd.crosstab(w['fno_symbol'], w['direction_pred']).to_string())
print()
print("--- actual outcome mix (band-classified) ---")
print(pd.crosstab(w['fno_symbol'], w['direction_actual']).to_string())
print()
print("--- confusion pooled (pred x actual) ---")
print(pd.crosstab(w['direction_pred'], w['direction_actual']).to_string())
print()
print("--- by confidence ---")
gg = w.dropna(subset=['actual_return'])
for k, grp in gg.groupby('confidence_pred'):
    dd = grp[grp['direction_pred'].isin(['UP','DOWN'])]
    print(f"{k:7s} n={len(grp):4d} acc3={grp['was_correct'].astype(bool).mean()*100:5.1f}%  "
          f"n_dir={len(dd):3d} sign={(((dd['direction_pred']=='UP')==(dd['actual_return']>0)).mean()*100 if len(dd) else float('nan')):5.1f}%")
