# -*- coding: utf-8 -*-
"""Поиск технической альфы на 1s-данных: OB-фильтр входов MR и направленная OB-стратегия.
Net под моделями исполнения: taker / maker / maker-rebate (с комиссиями и спредом).
Требует кэш data/hf/*_1s.parquet (см. run_hf_predict.py).
"""
import glob, pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scripts.run_hf_predict import PRICE, MICRO
RES=pathlib.Path("results")
TAKER=0.0004; MAKER=0.0001  # комиссия за сторону
HOR="fwd_5"; KZ=2.0

def nets(g, c):
    return dict(n=int(len(g)), hit=float(np.nanmean(g>0)),
        spread_only=float(np.nanmean(g-c)*1e4), taker=float(np.nanmean(g-c-2*TAKER)*1e4),
        maker=float(np.nanmean(g+c/2-2*MAKER)*1e4), maker_rebate=float(np.nanmean(g+c/2)*1e4))

def main():
    D=pd.concat([pd.read_parquet(f) for f in glob.glob("data/hf/*_1s.parquet")],ignore_index=True).dropna(subset=PRICE+MICRO+[HOR,"rel_spread_raw"])
    D["day"]=pd.to_datetime(D["day"]); days=np.sort(D["day"].unique()); split=days[int(0.6*len(days))]
    te=(D["day"]>=split).values
    fwd=D[HOR].values; sp=D["rel_spread_raw"].values; zmid=D["zmid"].values; imb=D["imb_l1"].values
    dirn=-np.sign(zmid); gross=dirn*fwd
    agree=((dirn>0)&(imb>0.5))|((dirn<0)&(imb<0.5))
    ent=(np.abs(zmid)>=KZ)&te
    rows=[{"config":"MR_no_filter",**nets(gross[ent],sp[ent])},
          {"config":"MR_OB_filter",**nets(gross[ent&agree],sp[ent&agree])}]
    R=pd.DataFrame(rows); R.to_csv(RES/"alpha_1s.csv",index=False)
    print(R.round(3).to_string(index=False))
    print("\nnet в б.п. на сделку; taker/maker комиссии 4/1 б.п. за сторону")

if __name__=="__main__":
    main()
