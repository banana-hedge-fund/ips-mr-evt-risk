# -*- coding: utf-8 -*-
"""ML-расширение: признаки и мета-лейблинг MR-сделок.

Идея (meta-labeling, в духе Lopez de Prado): базовая MR-стратегия генерирует
сигналы входа; классификатор предсказывает, окажется ли сделка прибыльной,
и фильтрует заведомо убыточные входы — снижая убытки.

Фичи строятся только из информации, доступной НА МОМЕНТ входа (без заглядывания
вперёд): z-score, импульс, реализованная волатильность, режим, order-flow
imbalance (taker-buy из klines), относительный объём, внутрибаровый диапазон,
EWMA-тренд/детренд (разложение/сглаживание), RSI-подобный осциллятор, час суток.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .backtest import BTParams, BASELINE, compute_signals

FEATURES = [
    "z", "abs_z", "rv", "vol_ratio",
    "mom_5", "mom_15", "mom_60",
    "taker_ratio", "vol_z", "hl_range",
    "ewma_slope", "detrend", "rsi", "hour",
]


def build_features(df: pd.DataFrame, p: BTParams = BTParams()):
    s = compute_signals(df, p)
    logp = np.log(s["close"]); ret = s["ret"]
    f = pd.DataFrame(index=s.index)
    f["z"] = s["z"]
    f["abs_z"] = s["z"].abs()
    f["rv"] = s["rv"]
    f["vol_ratio"] = s["vol_ratio"].clip(0, 5)
    for w in (5, 15, 60):
        f[f"mom_{w}"] = ret.rolling(w).sum()
    if "taker_buy_volume" in df.columns:
        tb = df["taker_buy_volume"].rolling(60).sum()
        vv = df["volume"].rolling(60).sum().replace(0, np.nan)
        f["taker_ratio"] = (tb / vv).clip(0, 1)
    else:
        f["taker_ratio"] = 0.5
    vmean = df["volume"].rolling(60).mean(); vstd = df["volume"].rolling(60).std()
    f["vol_z"] = ((df["volume"] - vmean) / vstd).clip(-5, 5)
    f["hl_range"] = ((df["high"] - df["low"]) / df["close"]).rolling(15).mean()
    ew = logp.ewm(span=30).mean()
    f["ewma_slope"] = ew.diff(5)
    f["detrend"] = (logp - ew)
    up = ret.clip(lower=0).rolling(30).mean()
    dn = (-ret.clip(upper=0)).rolling(30).mean()
    f["rsi"] = (up / (up + dn)).fillna(0.5)
    f["hour"] = s.index.hour + s.index.minute / 60.0
    return s, f


def build_trade_dataset(df: pd.DataFrame, symbol="", month="", year="",
                        p: BTParams = BTParams(), rp=BASELINE) -> pd.DataFrame:
    """Симуляция baseline-сделок; на каждый вход — фичи и label (прибыльна ли сделка)."""
    s, feats = build_features(df, p)
    idx = s.index; z = s["z"].values; close = s["close"].values
    days = idx.normalize()
    rows = []
    pos = 0.0; pend = None; cool = 0
    fee_rt = 2 * p.fee

    def close_trade(exit_price, exit_time):
        d = pend["dir"]
        net = d * (exit_price / pend["entry_price"] - 1.0) - fee_rt
        fr = feats.iloc[pend["i"]]
        if fr[FEATURES].isna().any():
            return
        row = {k: float(fr[k]) for k in FEATURES}
        row.update({"direction": int(d), "ret_net": float(net), "label": int(net > 0),
                    "symbol": symbol, "month": month, "year": year,
                    "entry_time": idx[pend["i"]], "exit_time": exit_time})
        rows.append(row)

    for t in range(len(idx)):
        zt = z[t]; new_day = t > 0 and days[t] != days[t - 1]
        if new_day and pos != 0.0:
            close_trade(close[t], idx[t]); pos = 0.0; pend = None; cool = p.cooldown
        if not np.isfinite(zt):
            continue
        if pos != 0.0:
            adverse = (close[t] / pend["entry_price"] - 1.0) * np.sign(pos)
            reverted = (pos > 0 and zt >= 0) or (pos < 0 and zt <= 0)
            if adverse <= -rp.d_stop or reverted:
                close_trade(close[t], idx[t]); pos = 0.0; pend = None; cool = p.cooldown
        if cool > 0:
            cool -= 1
        if pos == 0.0 and cool == 0:
            if zt <= -rp.k_entry:
                pos = +rp.L
            elif zt >= rp.k_entry:
                pos = -rp.L
            if pos != 0.0:
                pend = {"i": t, "dir": np.sign(pos), "entry_price": close[t]}
    return pd.DataFrame(rows)
