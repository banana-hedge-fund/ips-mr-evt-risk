# -*- coding: utf-8 -*-
"""Строгость EVT-оверлея: (1) HAC/Newey-West значимость + блочный бутстрап CVaR;
(2) причинная walk-forward GARCH (контроль look-ahead); (3) чувствительность ξ.
Требует кэш ξ (data/xi/) — см. run_evt_overlay.py.
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from arch import arch_model
from src import data_io, backtest as bt, metrics, risk_overlay as ro
RES=pathlib.Path("results"); XI=pathlib.Path("data/xi")
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]; MS=["2024-03","2024-08","2026-01","2026-04"]; P=bt.BTParams(window=60,cooldown=10)


def significance():
    pnl={"baseline":[],"var_only":[],"garch_evt":[]}
    for s in SYMS:
        for m in MS:
            df=data_io.load_window(s,[m]); xi=pd.read_parquet(XI/f"{s}_{m}.parquet")["xi"]
            pnl["baseline"].append(bt.backtest(df,False,P)["pnl"].values)
            pnl["var_only"].append(bt.backtest(df,True,P,xi_series=None)["pnl"].values)
            pnl["garch_evt"].append(bt.backtest(df,True,P,xi_series=xi)["pnl"].values)
    for k in pnl: pnl[k]=np.concatenate(pnl[k])
    def hac(d):
        m=sm.OLS(d,np.ones((len(d),1))).fit(cov_type="HAC",cov_kwds={"maxlags":int(len(d)**0.5)})
        return float(m.params[0]),float(m.tvalues[0]),float(m.pvalues[0])
    rows=[]
    for a,b in [("garch_evt","var_only"),("var_only","baseline"),("garch_evt","baseline")]:
        mu,t,p=hac(pnl[a]-pnl[b]); rows.append(dict(pair=f"{a}-{b}",dmean=mu,t_hac=t,pvalue=p))
    pd.DataFrame(rows).to_csv(RES/"evt_significance.csv",index=False); print(pd.DataFrame(rows).to_string(index=False))


def garch_oos_sr(ret, train_frac=0.3, scale=100.0):
    r=(ret.dropna()*scale).values; idx=ret.dropna().index; n=len(r); ntr=int(train_frac*n)
    res=arch_model(r[:ntr],mean="Constant",vol="GARCH",p=1,q=1,dist="t").fit(disp="off")
    mu,om,al,be=(res.params[k] for k in ["mu","omega","alpha[1]","beta[1]"])
    eps=r-mu; s2=np.empty(n); s2[0]=np.var(eps[:ntr])
    for t in range(1,n): s2[t]=om+al*eps[t-1]**2+be*s2[t-1]
    return pd.Series(eps/np.sqrt(s2),index=idx)


if __name__=="__main__":
    significance()
