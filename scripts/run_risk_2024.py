# -*- coding: utf-8 -*-
"""Корректный риск-контур на 2024: baseline / optimized / optimized+EVT / optimized+EVT+CVaR.
Порядок: оптимизация базы -> EVT-оверлей модулирует базу -> CVaR-лимит экспозиции.
Запуск: python scripts/run_risk_2024.py <src_dir> <EXCHANGE> [SYMBOL]"""
import sys, pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from arch import arch_model
from src import ob_minute_io as io, backtest as bt, metrics, risk_overlay as ro
RES=pathlib.Path("results")
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]
P_DEF=bt.BTParams(window=60,cooldown=10)
P_OPT=bt.BTParams(window=120,cooldown=30,vol_filter=0.5); RP_OPT=bt.RegimeParams(1.0,3.0,0.03,True)

def causal_xi(ret, step=240):
    r=(ret.dropna()*100).values; idx=ret.dropna().index; n=len(r); ntr=int(0.3*n)
    res=arch_model(r[:ntr],mean="Constant",vol="GARCH",p=1,q=1,dist="t").fit(disp="off")
    mu,om,al,be=(res.params[k] for k in ["mu","omega","alpha[1]","beta[1]"]); eps=r-mu; s2=np.empty(n); s2[0]=np.var(eps[:ntr])
    for t in range(1,n): s2[t]=om+al*eps[t-1]**2+be*s2[t-1]
    return ro.rolling_gpd_xi(pd.Series(eps/np.sqrt(s2),index=idx),window=1440,step=step)

def main(src,ex,only):
    rows=[]
    for sym in (SYMS if only is None else [only]):
        df=io.load(ex,sym,src)
        if df is None or len(df)<3000: print("[skip]",ex,sym); continue
        print(f"[data] {ex} {sym}: {len(df)} мин, {df.index.normalize().nunique()} дней")
        xi=causal_xi(df["ret"]).reindex(df.index).ffill().fillna(0.0)
        bud=2.665*np.nanpercentile(df["ret"].rolling(1440,min_periods=300).std().dropna(),40)
        cfgs={"baseline":bt.backtest(df,False,P_DEF),
              "optimized":bt.backtest(df,False,P_OPT,base_rp=RP_OPT),
              "optimized+EVT":bt.backtest(df,True,P_OPT,xi_series=xi,overlay_base=RP_OPT),
              "optimized+EVT+CVaR":bt.backtest(df,True,P_OPT,xi_series=xi,overlay_base=RP_OPT,cvar_budget=bud)}
        for n,r in cfgs.items():
            s=metrics.summarize(r["equity"],r["pnl"]); s.update(exchange=ex,symbol=sym,cfg=n,trades=r["n_trades"],days=int(df.index.normalize().nunique())); rows.append(s)
    if rows:
        S=pd.DataFrame(rows); S.to_csv(RES/f"risk2024_{ex}{'_'+only if only else ''}.csv",index=False)
        print("\n",S[["symbol","days","cfg","total_return","max_drawdown","cvar_99","sharpe","trades"]].round(4).to_string(index=False))

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
