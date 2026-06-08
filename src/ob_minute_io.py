# -*- coding: utf-8 -*-
"""Загрузка ПОДНЕВНЫХ 1m-таблиц микроструктуры (orderbook_minute_{EX}_{SYM}_{YYYYMMDD}.parquet).

Цена-сигнал = mid. Возвращает df, совместимый с пайплайном (close/ret/volume + OB-колонки).
Работает на ЧАСТИЧНЫХ данных — берёт все доступные подневные файлы.
"""
from __future__ import annotations
import glob, pathlib
import numpy as np, pandas as pd

def list_available(src_dir):
    out={}
    for f in glob.glob(f"{src_dir}/orderbook_minute_*_*.parquet"):
        name=pathlib.Path(f).stem.split("_")
        if len(name)>=5:
            ex,sym,date=name[2],name[3],name[4]; out.setdefault((ex,sym),[]).append(date)
    return {k:sorted(v) for k,v in out.items()}

def load(exchange, symbol, src_dir):
    fs=sorted(glob.glob(f"{src_dir}/orderbook_minute_{exchange}_{symbol}_*.parquet"))
    if not fs: return None
    df=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
    ts="minute_ts" if "minute_ts" in df.columns else "bucket_ts"
    df[ts]=pd.to_datetime(df[ts],utc=True)
    df=df.sort_values(ts).drop_duplicates(ts).set_index(ts)
    df=df[df.get("crossed",0)!=1]
    df["close"]=df["mid"]; df["high"]=df["mid"]; df["low"]=df["mid"]
    df["volume"]=pd.to_numeric(df.get("trade_volume"),errors="coerce").fillna(0.0)
    tbr=pd.to_numeric(df.get("taker_buy_ratio"),errors="coerce").fillna(0.5)
    df["taker_buy_volume"]=tbr*df["volume"]
    lm=np.log(df["close"])
    df["ret"]=lm.groupby(df.index.date).diff()
    return df.dropna(subset=["close"])
