"""Риск-скорректированные метрики эффективности (CFA / quant finance)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_PER_YEAR = 365 * 24 * 60  # крипторынок 24/7


def _ann_factor(freq_per_year: int = MIN_PER_YEAR) -> float:
    return np.sqrt(freq_per_year)


def sharpe(pnl: pd.Series, freq_per_year: int = MIN_PER_YEAR) -> float:
    r = pnl.dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * _ann_factor(freq_per_year))


def sortino(pnl: pd.Series, freq_per_year: int = MIN_PER_YEAR) -> float:
    r = pnl.dropna()
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd) or len(r) < 2:
        return 0.0
    return float(r.mean() / dd * _ann_factor(freq_per_year))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def calmar(equity: pd.Series, pnl: pd.Series, freq_per_year: int = MIN_PER_YEAR) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    n_years = len(pnl) / freq_per_year
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    return float(cagr / mdd)


def tail_loss(pnl: pd.Series, q: float = 0.99) -> dict:
    r = pnl.dropna().values
    if len(r) == 0:
        return {"var": 0.0, "cvar": 0.0, "worst": 0.0}
    var = -np.quantile(r, 1 - q)
    cvar = -r[r <= -var].mean() if (r <= -var).any() else var
    return {"var": float(var), "cvar": float(cvar), "worst": float(-r.min())}


def summarize(equity: pd.Series, pnl: pd.Series, q: float = 0.99) -> dict:
    tl = tail_loss(pnl, q)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "sharpe": sharpe(pnl),
        "sortino": sortino(pnl),
        "calmar": calmar(equity, pnl),
        "max_drawdown": max_drawdown(equity),
        "var_99": tl["var"],
        "cvar_99": tl["cvar"],
        "worst_bar": tl["worst"],
        "n_bars": int(len(pnl)),
    }
