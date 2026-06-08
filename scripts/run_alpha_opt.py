# -*- coding: utf-8 -*-
"""Оптимизация микроструктурного edge (без комиссий): направленная OB-стратегия.
На 1s-данных: GB(prob) -> направление; неперекрывающиеся сделки шаг h; net = dir*fwd_h - spread.
Свип по h и margin; устойчивость по дням. Требует data/hf/*_1s.parquet.
ДИСКЛЕЙМЕР: research, не инвест-рекомендация; комиссии/импакт/adverse selection не учтены.
"""
import pathlib, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.run_hf_predict import PRICE, MICRO
RES=pathlib.Path("results"); FEATS=PRICE+MICRO

def eval_sym(sym, horizons=(5,10,20,30), margins=(0.2,0.3)):
    D=pd.read_parquet(f"data/hf/{sym}_1s.parquet"); D["day"]=pd.to_datetime(D["day"]); lm=np.log(D["mid"])
    days=np.sort(D["day"].unique()); split=days[int(0.6*len(days))]; rows=[]
    for h in horizons:
        D[f"fwd_{h}"]=lm.groupby(D["day"]).shift(-h)-lm
        d=D.dropna(subset=FEATS+[f"fwd_{h}","rel_spread_raw"]).reset_index(drop=True)
        tr=(d["day"]<split).values; te=(d["day"]>=split).values
        X=d[FEATS].values; y=(d[f"fwd_{h}"]>0).astype(int).values
        tri=np.random.RandomState(0).choice(np.where(tr)[0],min(250000,tr.sum()),replace=False)
        m=HistGradientBoostingClassifier(max_iter=120,max_depth=3,learning_rate=0.05,random_state=0).fit(X[tri],y[tri])
        p=m.predict_proba(X)[:,1]; fwd=d[f"fwd_{h}"].values; sp=d["rel_spread_raw"].values
        ti=np.where(te)[0]; ti=ti[ti%h==0]
        for mar in margins:
            sel=ti[np.abs(p[ti]-0.5)>=mar]
            if len(sel)<50: continue
            net=(np.sign(p[sel]-0.5)*fwd[sel]-sp[sel])
            rows.append(dict(symbol=sym,h=h,margin=mar,n=len(sel),net_bp=np.mean(net)*1e4,
                             ir_trade=np.mean(net)/np.std(net),total_bp=np.sum(net)*1e4))
    return pd.DataFrame(rows)

if __name__=="__main__":
    import glob
    syms=[pathlib.Path(f).stem.replace("_1s","") for f in glob.glob("data/hf/*_1s.parquet")]
    R=pd.concat([eval_sym(s) for s in syms],ignore_index=True)
    R.to_csv(RES/"alpha_opt.csv",index=False); print(R.round(3).to_string(index=False))
