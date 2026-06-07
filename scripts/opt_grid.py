# -*- coding: utf-8 -*-
"""Grid-search классической MR (chunked). Запуск: python scripts/opt_grid.py <chunk> <nchunks>
Сетка: window×k_entry×d_stop×cooldown×vol_filter. Train=2024, Test=2026.
"""
import itertools, pathlib, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from src import data_io, backtest as bt, metrics

RES=pathlib.Path("results"); RES.mkdir(exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]; MBY={"2024":["2024-03","2024-08"],"2026":["2026-01","2026-04"]}
WINDOWS=[60,120,240]; KENTRY=[2.0,2.5,3.0]; DSTOP=[0.015,0.03,0.05]; COOL=[10,30]; VOLF=[-1e9,0.0,0.5,1.0]

DATA={}
for s in SYMS:
    for yr,ms in MBY.items():
        for m in ms:
            DATA[(s,m)]=data_io.load_window(s,[m])

def run_combo(c):
    w,k,d,cd,vf=c; rows=[]
    for (s,m),df in DATA.items():
        yr="2024" if m.startswith("2024") else "2026"
        r=bt.backtest(df,False,bt.BTParams(window=w,cooldown=cd,vol_filter=vf),bt.RegimeParams(1.0,k,d,True))
        sm=metrics.summarize(r["equity"],r["pnl"])
        rows.append(dict(window=w,k_entry=k,d_stop=d,cooldown=cd,vol_filter=vf,symbol=s,month=m,year=yr,
                         total=sm["total_return"],sharpe=sm["sharpe"],mdd=sm["max_drawdown"],
                         cvar=sm["cvar_99"],trades=r["n_trades"]))
    return rows

if __name__=="__main__":
    chunk=int(sys.argv[1]); nch=int(sys.argv[2])
    combos=list(itertools.product(WINDOWS,KENTRY,DSTOP,COOL,VOLF))
    mine=combos[chunk::nch]
    with Pool(processes=4) as pool:
        allrows=[]
        for rows in pool.imap_unordered(run_combo,mine): allrows.extend(rows)
    pd.DataFrame(allrows).to_csv(RES/f"opt_chunk_{chunk}.csv",index=False)
    print(f"chunk {chunk} done ({len(allrows)} rows)",flush=True)
