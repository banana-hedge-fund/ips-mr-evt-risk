# -*- coding: utf-8 -*-
"""HF-предсказание знака движения mid на коротких горизонтах: цена vs +стакан.

Автоподхват субсекундных данных:
  1) <mounted>/orderbook_subsec/  (orderbook_1s_{SYM}_{YYYYMMDD}.parquet или *_100ms_*)
  2) data/ob_1s/ , data/ob_100ms/
  3) data/ob/{SYM}.parquet  — DRY-RUN на минутных данных (горизонты в БАКЕТАХ)
Горизонты задаются в БАКЕТАХ (для 1с-данных это секунды).
"""
import glob, pathlib, warnings, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

RES=pathlib.Path("results"); RES.mkdir(exist_ok=True)
SYMS=["BTCUSDT","ETHUSDT","DOGEUSDT"]
HORIZONS=[1,5,10]          # в бакетах
TSCOL="bucket_ts"

PRICE=["r1","mom_3","mom_10","mom_30","rvol","zmid"]
MICRO=["rel_spread","micro_dev","imb_l1","imb_ln","book_slope_bid","book_slope_ask",
       "cancel_rate","add_rate","update_rate","ofi_l1","ofi_ln","eff_spread",
       "trade_sign_imb","depth_imb","taker_buy_ratio","rv_mid"]

def _candidates():
    base=glob.glob("/sessions/*/mnt/*/orderbook_subsec")+glob.glob("/sessions/*/mnt/orderbook_subsec")
    base+=["data/ob_1s","data/ob_100ms"]
    return base

def load_symbol(sym):
    for d in _candidates():
        fs=sorted(glob.glob(f"{d}/orderbook_*_{sym}_*.parquet"))
        if fs:
            df=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
            ts=TSCOL if TSCOL in df.columns else ("minute_ts" if "minute_ts" in df.columns else df.columns[0])
            df[ts]=pd.to_datetime(df[ts],utc=True); df=df.rename(columns={ts:"ts"})
            return df.sort_values("ts").drop_duplicates("ts"), "subsec"
    p=f"data/ob/{sym}.parquet"
    if pathlib.Path(p).exists():
        df=pd.read_parquet(p); df["ts"]=pd.to_datetime(df["minute_ts"],utc=True)
        return df.sort_values("ts").drop_duplicates("ts"), "dryrun-1min"
    return None,None

def feats_labels(df):
    out=[]
    for day,g in df.groupby(df["ts"].dt.date):
        if len(g)<max(HORIZONS)+40: continue
        g=g.sort_values("ts").reset_index(drop=True)
        lm=np.log(g["mid"]); r1=lm.diff()
        f=pd.DataFrame(index=g.index)
        f["r1"]=r1; f["mom_3"]=r1.rolling(3).sum(); f["mom_10"]=r1.rolling(10).sum(); f["mom_30"]=r1.rolling(30).sum()
        f["rvol"]=r1.rolling(30).std()
        f["zmid"]=(lm-lm.rolling(30).mean())/lm.rolling(30).std()
        f["rel_spread"]=g.get("rel_spread"); f["micro_dev"]=(g["microprice"]-g["mid"])/g["mid"]
        f["imb_l1"]=g.get("imb_l1"); f["imb_ln"]=g.get("imb_ln")
        f["book_slope_bid"]=g.get("book_slope_bid"); f["book_slope_ask"]=g.get("book_slope_ask")
        f["cancel_rate"]=g.get("cancel_rate"); f["add_rate"]=g.get("add_rate"); f["update_rate"]=np.log1p(g.get("update_rate"))
        f["ofi_l1"]=g.get("ofi_l1"); f["ofi_ln"]=g.get("ofi_ln"); f["eff_spread"]=g.get("eff_spread")
        f["trade_sign_imb"]=g.get("trade_sign_imb")
        f["depth_imb"]=(g["depth_bid_n"]-g["depth_ask_n"])/(g["depth_bid_n"]+g["depth_ask_n"])
        f["taker_buy_ratio"]=g.get("taker_buy_ratio"); f["rv_mid"]=g.get("rv_mid")
        f["mid"]=g["mid"].values; f["rel_spread_raw"]=g.get("rel_spread"); f["day"]=str(day); f["ts"]=g["ts"].values
        for h in HORIZONS:
            f[f"fwd_{h}"]=np.log(g["mid"].shift(-h))-lm
        out.append(f)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def run():
    frames=[]; mode=None
    for s in SYMS:
        df,m=load_symbol(s)
        if df is None: continue
        mode=m; ff=feats_labels(df); ff["symbol"]=s; frames.append(ff)
    if not frames:
        print("НЕТ ДАННЫХ"); return
    D=pd.concat(frames,ignore_index=True).dropna(subset=PRICE+MICRO)
    days=np.sort(D["day"].unique()); split=days[int(0.6*len(days))]
    tr=D["day"]<split; te=D["day"]>=split
    print(f"режим={mode} | строк={len(D)} train={tr.sum()} test={te.sum()} | дней={len(days)} split={split}")
    rows=[]; rocs={}
    for h in HORIZONS:
        y=(D[f"fwd_{h}"]>0).astype(int).values
        for sname,feats in [("цена",PRICE),("цена+стакан",PRICE+MICRO)]:
            X=D[feats].values
            mdl=HistGradientBoostingClassifier(max_iter=150,max_depth=3,learning_rate=0.05,random_state=0)
            mdl.fit(X[tr.values],y[tr.values]); p=mdl.predict_proba(X[te.values])[:,1]
            auc=roc_auc_score(y[te.values],p); rows.append(dict(horizon=h,features=sname,auc=auc))
            if h==HORIZONS[0]: rocs[sname]=(y[te.values],p)
            if sname=="цена+стакан":
                fwd=D[f"fwd_{h}"].values[te.values]; sgn=np.where(p>=0.5,1,-1)
                gross=np.mean(sgn*fwd)*1e4; cost=np.nanmean(D["rel_spread_raw"].values[te.values])*1e4
                rows[-1].update(dict(gross_bps=gross,spread_bps=cost,net_bps=gross-cost))
    R=pd.DataFrame(rows); R.to_csv(RES/"hf_auc.csv",index=False)
    pd.set_option('display.width',200)
    print("\n=== AUC по горизонтам ==="); print(R.pivot(index="horizon",columns="features",values="auc").round(3).to_string())
    tr_df=R[R.features=="цена+стакан"][["horizon","gross_bps","spread_bps","net_bps"]].dropna()
    if len(tr_df): print("\n=== Торгуемость (+стакан), bps ===\n",tr_df.round(2).to_string(index=False))
    pv=R.pivot(index="horizon",columns="features",values="auc")
    pv.plot(marker="o"); plt.axhline(0.5,color='k',ls='--',lw=0.8)
    plt.xlabel("горизонт (бакетов)"); plt.ylabel("Test AUC"); plt.title(f"HF-предсказание mid ({mode})"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RES/"fig_hf_auc.png",dpi=120); plt.close()
    if rocs:
        plt.figure(figsize=(6,6))
        for sname,(yt,pp) in rocs.items():
            fpr,tpr,_=roc_curve(yt,pp); plt.plot(fpr,tpr,label=f"{sname} (AUC={roc_auc_score(yt,pp):.3f})")
        plt.plot([0,1],[0,1],'k--',lw=.8); plt.legend(loc="lower right"); plt.title(f"ROC, горизонт {HORIZONS[0]} ({mode})")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout(); plt.savefig(RES/"fig_hf_roc.png",dpi=120); plt.close()
    json.dump(dict(mode=mode,rows=rows),open(RES/"hf_summary.json","w"),ensure_ascii=False,indent=2)
    print("\nсохранено: hf_auc.csv, fig_hf_auc.png, fig_hf_roc.png")

if __name__=="__main__":
    run()
