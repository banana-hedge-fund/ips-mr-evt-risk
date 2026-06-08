# -*- coding: utf-8 -*-
"""OPT-5: мета-лейблинг на периоде со стаканом. Цена-сигнал = mid из фич стакана.

Сравнение двух наборов признаков: свечные (из mid+объёма) и свечные+микроструктура.
Ожидаемые файлы: data/ob/{SYMBOL}.parquet (консолидация orderbook_minute_*.csv).
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd

OBDIR = pathlib.Path("data/ob")

CANDLE = ["z","abs_z","rv","mom_5","mom_15","mom_60","vol_z","taker_ratio",
          "rng","ewma_slope","detrend","rsi","hour"]
OB = ["rel_spread","micro_dev","imb_l1","imb_ln","book_slope_bid","book_slope_ask",
      "cancel_rate","update_rate","add_rate","ofi_l1","ofi_ln","eff_spread",
      "vwap_dev","trade_sign_imb","depth_imb","rv_mid"]

PARAMS = dict(window=60, k_entry=2.0, d_stop=0.03, cooldown=10, fee=0.0002)


def _day_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    lm = np.log(g["mid"]); ret = lm.diff()
    f = pd.DataFrame(index=g.index)
    ma = lm.rolling(60).mean(); sd = lm.rolling(60).std()
    f["z"] = (lm - ma)/sd; f["abs_z"] = f["z"].abs(); f["ret"] = ret
    f["rv"] = ret.rolling(60).std()
    for w in (5,15,60): f[f"mom_{w}"] = ret.rolling(w).sum()
    lv = np.log1p(g["trade_volume"].clip(lower=0))
    f["vol_z"] = ((lv - lv.rolling(60).mean())/lv.rolling(60).std()).clip(-5,5)
    f["taker_ratio"] = g["taker_buy_ratio"].clip(0,1)
    f["rng"] = g["rv_mid"]
    ew = lm.ewm(span=30).mean(); f["ewma_slope"] = ew.diff(5); f["detrend"] = lm - ew
    up = ret.clip(lower=0).rolling(30).mean(); dn=(-ret.clip(upper=0)).rolling(30).mean()
    f["rsi"] = (up/(up+dn)).fillna(0.5)
    f["hour"] = g.index.hour + g.index.minute/60.0
    f["rel_spread"]=g["rel_spread"]; f["micro_dev"]=(g["microprice"]-g["mid"])/g["mid"]
    f["imb_l1"]=g["imb_l1"]; f["imb_ln"]=g["imb_ln"]
    f["book_slope_bid"]=g["book_slope_bid"]; f["book_slope_ask"]=g["book_slope_ask"]
    f["cancel_rate"]=g["cancel_rate"]; f["update_rate"]=np.log1p(g["update_rate"]); f["add_rate"]=g["add_rate"]
    f["ofi_l1"]=g["ofi_l1"]; f["ofi_ln"]=g["ofi_ln"]; f["eff_spread"]=g["eff_spread"]
    f["vwap_dev"]=g["vwap_dev"]; f["trade_sign_imb"]=g["trade_sign_imb"]
    f["depth_imb"]=(g["depth_bid_n"]-g["depth_ask_n"])/(g["depth_bid_n"]+g["depth_ask_n"])
    f["rv_mid"]=g["rv_mid"]
    f["mid"]=g["mid"]; f["crossed"]=g["crossed"]
    return f


def build_dataset(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(OBDIR/f"{symbol}.parquet").set_index("minute_ts").sort_index()
    p = PARAMS; allcols = CANDLE+OB; rows=[]
    for day, g in df.groupby(df.index.date):
        if len(g) < 120: continue
        f = _day_features(g)
        z = f["z"].values; mid = f["mid"].values; idx = f.index
        pos=0.0; entry=None; cool=0; ent_i=0
        for t in range(len(idx)):
            zt=z[t]
            if not np.isfinite(zt): continue
            if pos!=0.0:
                adverse=(mid[t]/entry-1.0)*np.sign(pos)
                reverted=(pos>0 and zt>=0) or (pos<0 and zt<=0)
                if adverse<=-p["d_stop"] or reverted:
                    d=np.sign(pos); net=d*(mid[t]/entry-1.0)-2*p["fee"]
                    fr=f.iloc[ent_i]
                    if not fr[allcols].isna().any():
                        row={k:float(fr[k]) for k in allcols}; row.update(dict(symbol=symbol,day=str(day),
                            direction=int(d),ret_net=float(net),label=int(net>0),entry_time=idx[ent_i]))
                        rows.append(row)
                    pos=0.0; entry=None; cool=p["cooldown"]
            if cool>0: cool-=1
            if pos==0.0 and cool==0 and int(f["crossed"].iloc[t])==0:
                if zt<=-p["k_entry"]: pos=1.0; entry=mid[t]; ent_i=t
                elif zt>=p["k_entry"]: pos=-1.0; entry=mid[t]; ent_i=t
    return pd.DataFrame(rows)
