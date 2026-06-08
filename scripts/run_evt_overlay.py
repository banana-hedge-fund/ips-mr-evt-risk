# -*- coding: utf-8 -*-
"""Сравнение уровней риск-контроля: baseline / VaR-only / полный GARCH-EVT.

Сначала кэширует ξ по выборкам (GARCH + rolling GPD), затем гоняет 3 конфигурации.
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import data_io, backtest as bt, metrics, risk_overlay as ro

RES=pathlib.Path("results"); XI=pathlib.Path("data/xi"); XI.mkdir(parents=True,exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]; MBY={"2024":["2024-03","2024-08"],"2026":["2026-01","2026-04"]}
P=bt.BTParams(window=60,cooldown=10)


def ensure_xi(s,m,df):
    f=XI/f"{s}_{m}.parquet"
    if f.exists(): return pd.read_parquet(f)["xi"]
    xi=ro.xi_series_for(df,window=1440,step=120); xi.rename("xi").to_frame().to_parquet(f); return xi


def main():
    rows=[]
    for s in SYMS:
        for yr,ms in MBY.items():
            for m in ms:
                df=data_io.load_window(s,[m]); xi=ensure_xi(s,m,df)
                cfgs={"baseline":bt.backtest(df,False,P),
                      "var_only":bt.backtest(df,True,P,xi_series=None),
                      "garch_evt":bt.backtest(df,True,P,xi_series=xi)}
                for name,r in cfgs.items():
                    sm=metrics.summarize(r["equity"],r["pnl"]); sm.update(dict(cfg=name,symbol=s,year=yr,month=m,trades=r["n_trades"]))
                    rows.append(sm)
    D=pd.DataFrame(rows); D.to_csv(RES/"evt_overlay_long.csv",index=False)
    g=D.groupby(["cfg","year"]).agg(total=("total_return","mean"),sharpe=("sharpe","mean"),
        mdd=("max_drawdown","mean"),cvar=("cvar_99","mean"),trades=("trades","mean")).reset_index()
    g.to_csv(RES/"evt_overlay_agg.csv",index=False); print(g.round(4).to_string(index=False))


if __name__=="__main__":
    main()
