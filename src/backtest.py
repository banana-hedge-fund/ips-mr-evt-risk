"""Упрощённый бэктест mean-reversion стратегии по пересечению свечей.

Упрощения (ускоренный режим, ИПС):
  - исполнение по цене закрытия свечи (close-to-close), БЕЗ учёта ликвидности/стакана;
  - издержки — только комиссия (по обороту);
  - сигнал: z-score цены относительно скользящего среднего (пересечение порогов);
  - intraday: принудительное закрытие позиции в конце каждых суток (ежедневный чекпоинт).

Конфигурации: baseline (base_rp или BASELINE) и adaptive (режимный риск-оверлей).
Параметры baseline можно передать через base_rp (для grid-search).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .evt import classify_regime


@dataclass
class RegimeParams:
    L: float
    k_entry: float
    d_stop: float
    allow_new: bool


ADAPTIVE_MAP = {
    "R_N": RegimeParams(1.0, 2.0, 0.05, True),
    "R_E": RegimeParams(0.5, 2.5, 0.03, True),
    "R_X": RegimeParams(0.1, 3.5, 0.015, False),
}
BASELINE = RegimeParams(1.0, 2.0, 0.05, True)


@dataclass
class BTParams:
    window: int = 60
    fee: float = 0.0002
    rv_window: int = 60
    rv_baseline: int = 1440
    var_window: int = 1440
    var_q: float = 0.99
    viol_window: int = 240
    cooldown: int = 10


def compute_signals(df: pd.DataFrame, p: BTParams) -> pd.DataFrame:
    out = df.copy()
    logp = np.log(out["close"])
    ma = logp.rolling(p.window).mean()
    sd = logp.rolling(p.window).std()
    out["z"] = (logp - ma) / sd
    out["rv"] = out["ret"].rolling(p.rv_window).std()
    out["rv_base"] = out["ret"].rolling(p.rv_baseline).std()
    out["vol_ratio"] = out["rv"] / out["rv_base"]
    var = -out["ret"].rolling(p.var_window).quantile(1 - p.var_q)
    out["breach"] = (out["ret"] < -var).astype(float)
    out["viol_rate"] = out["breach"].rolling(p.viol_window).mean()
    return out


def _regime_series(sig: pd.DataFrame) -> pd.Series:
    vr = sig["vol_ratio"].fillna(1.0).values
    viol = sig["viol_rate"].fillna(0.0).values
    reg = [classify_regime(float(v), 0.0, float(w)) for v, w in zip(vr, viol)]
    return pd.Series(reg, index=sig.index)


def backtest(df: pd.DataFrame, adaptive: bool, p: BTParams = BTParams(), base_rp: RegimeParams = None) -> dict:
    sig = compute_signals(df, p)
    regimes = _regime_series(sig) if adaptive else None
    base = base_rp if base_rp is not None else BASELINE

    idx = sig.index
    z = sig["z"].values
    ret = sig["ret"].values
    close = sig["close"].values
    days = idx.normalize()

    pos = 0.0
    entry_price = np.nan
    cool = 0
    pos_arr = np.zeros(len(idx))
    regime_arr = np.empty(len(idx), dtype=object)
    n_trades = 0

    for t in range(len(idx)):
        rp = ADAPTIVE_MAP[regimes.iloc[t]] if adaptive else base
        regime_arr[t] = regimes.iloc[t] if adaptive else "R_N"
        new_day = t > 0 and days[t] != days[t - 1]
        zt = z[t]

        if new_day and pos != 0.0:
            pos = 0.0; entry_price = np.nan; cool = p.cooldown

        if not np.isfinite(zt):
            pos_arr[t] = pos; continue

        if pos != 0.0:
            adverse = (close[t] / entry_price - 1.0) * np.sign(pos)
            reverted = (pos > 0 and zt >= 0) or (pos < 0 and zt <= 0)
            if adverse <= -rp.d_stop or reverted:
                pos = 0.0; entry_price = np.nan; cool = p.cooldown

        if cool > 0:
            cool -= 1
        if pos == 0.0 and rp.allow_new and cool == 0:
            if zt <= -rp.k_entry:
                pos = +rp.L; entry_price = close[t]; n_trades += 1
            elif zt >= rp.k_entry:
                pos = -rp.L; entry_price = close[t]; n_trades += 1

        pos_arr[t] = pos

    pos_s = pd.Series(pos_arr, index=idx)
    gross = pos_s.shift(1).fillna(0.0).values * ret
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos_arr])))
    cost = turnover * p.fee
    pnl = pd.Series(gross - cost, index=idx).fillna(0.0)
    equity = (1.0 + pnl).cumprod()

    return {"equity": equity, "pnl": pnl, "position": pos_s,
            "regime": pd.Series(regime_arr, index=idx), "signals": sig,
            "n_trades": int(n_trades)}
