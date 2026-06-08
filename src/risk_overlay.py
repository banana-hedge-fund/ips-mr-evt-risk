# -*- coding: utf-8 -*-
"""Полная вертикаль риск-контроля: GARCH -> стандартизированные остатки ->
скользящая POT/GPD-оценка ξ -> классификатор режима -> параметры стратегии.

Замыкает иерархию «MR — GARCH — EVT — VaR/CVaR» в боевом контуре.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats


def garch_std_resid(ret: pd.Series, scale: float = 100.0, model: str = "GARCH"):
    """GARCH(1,1)/EGARCH(1,1), t-распределение. Возвращает (станд. остатки, условная волат.)."""
    from arch import arch_model
    r = ret.dropna() * scale
    vol = "EGARCH" if model.upper() == "EGARCH" else "GARCH"
    res = arch_model(r, mean="Constant", vol=vol, p=1, q=1, dist="t").fit(disp="off")
    sr = pd.Series(np.asarray(res.std_resid), index=ret.dropna().index)
    cv = pd.Series(np.asarray(res.conditional_volatility) / scale, index=ret.dropna().index)
    return sr, cv


def rolling_gpd_xi(std_resid: pd.Series, window: int = 1440, step: int = 120, q: float = 0.95) -> pd.Series:
    """Скользящая POT/GPD-оценка индекса хвоста ξ на станд. остатках (левый хвост)."""
    sr = std_resid.values
    n = len(sr); xi = np.full(n, np.nan); last = np.nan
    for t in range(n):
        if t >= window and (t % step == 0):
            w = sr[t - window:t]; losses = -w[w < 0]
            if len(losses) >= 50:
                u = np.quantile(losses, q); exc = losses[losses > u] - u
                if len(exc) >= 20:
                    try:
                        xi_, _, _ = stats.genpareto.fit(exc, floc=0); last = float(xi_)
                    except Exception:
                        pass
        xi[t] = last
    return pd.Series(xi, index=std_resid.index).ffill().fillna(0.0)


def xi_series_for(df: pd.DataFrame, model: str = "GARCH", window: int = 1440, step: int = 120) -> pd.Series:
    sr, _ = garch_std_resid(df["ret"], model=model)
    xi = rolling_gpd_xi(sr, window=window, step=step)
    return xi.reindex(df.index).ffill().fillna(0.0)
