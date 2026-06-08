# -*- coding: utf-8 -*-
"""Полный перепрогон на 1m-микроструктуре (любая биржа/символ; частичные данные).
H1/H3 (EVT) + H4 (baseline / optimized / GARCH-EVT-оверлей). Запуск:
  python scripts/run_all_2024.py <src_dir> [EXCHANGE]
"""
import sys, pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import ob_minute_io as io, econometrics as ec, evt, metrics, backtest as bt, risk_overlay as ro
RES=pathlib.Path("results"); RES.mkdir(exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]
P_DEF=bt.BTParams(window=60,cooldown=10)
P_OPT=bt.BTParams(window=120,cooldown=30,vol_filter=0.5); RP_OPT=bt.RegimeParams(1.0,3.0,0.03,True)

def analyze(ret):
    d=ec.descriptive_stats(ret); d.update(ec.hill_estimator(ret,tail="left"))
    r=ret.dropna().values; gpd=evt.fit_gpd_pot(-r,0.95); vc=evt.evt_var_cvar(gpd,0.99)
    d.update(gpd_xi=gpd["xi"],evt_var99=vc["var"],gauss_var99=evt.gaussian_var(r,0.99),hist_var99=evt.historical_var(r,0.99))
    bg=(r<-d["gauss_var99"]).sum(); d["kupiec_gauss_rate"]=evt.kupiec_pof(int(bg),len(r),0.99)["rate"]
    return d

def main(src_dir, exchange):
    econ=[]; summ=[]
    for sym in SYMS:
        df=io.load(exchange,sym,src_dir)
        if df is None or len(df)<2000: print(f"[skip] {exchange} {sym}"); continue
        ndays=df.index.normalize().nunique()
        print(f"[data] {exchange} {sym}: {len(df)} мин, {ndays} дней")
        a=analyze(df["ret"]); a.update(exchange=exchange,symbol=sym,minutes=len(df),days=ndays); econ.append(a)
        xi=ro.xi_series_for(df,window=1440,step=120)
        cfgs={"baseline":bt.backtest(df,False,P_DEF),
              "optimized":bt.backtest(df,False,P_OPT,base_rp=RP_OPT),
              "garch_evt":bt.backtest(df,True,P_DEF,xi_series=xi)}
        for name,r in cfgs.items():
            s=metrics.summarize(r["equity"],r["pnl"]); s.update(exchange=exchange,symbol=sym,cfg=name,trades=r["n_trades"]); summ.append(s)
    if econ: pd.DataFrame(econ).to_csv(RES/f"y2024_econ_{exchange}.csv",index=False)
    if summ:
        S=pd.DataFrame(summ); S.to_csv(RES/f"y2024_backtest_{exchange}.csv",index=False)
        g=S.groupby("cfg").agg(total=("total_return","mean"),sharpe=("sharpe","mean"),mdd=("max_drawdown","mean"),cvar=("cvar_99","mean"),trades=("trades","mean")).reset_index()
        print("\n=== BACKTEST (агрегат по конфигурации) ==="); print(g.round(4).to_string(index=False))
    print("\nГотово.")

if __name__=="__main__":
    src=sys.argv[1] if len(sys.argv)>1 else "data/micro_test"
    ex=sys.argv[2] if len(sys.argv)>2 else "BINANCE"
    main(src,ex)
