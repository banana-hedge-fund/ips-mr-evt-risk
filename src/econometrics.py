"""Эконометрика: дескриптивы, стационарность, хвосты, MR и волатильность.

Используется для проверки H1 (тяжёлые хвосты, асимметрия) и H2
(режимы волатильности).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss


def descriptive_stats(ret: pd.Series) -> dict:
    r = ret.dropna().values
    jb, jb_p = stats.jarque_bera(r)
    return {
        "n": int(len(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)),
        "skew": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "jarque_bera": float(jb),
        "jb_pvalue": float(jb_p),
    }


def stationarity_tests(series: pd.Series) -> dict:
    s = series.dropna().values
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    return {
        "adf_stat": float(adf_stat), "adf_pvalue": float(adf_p),
        "kpss_stat": float(kpss_stat), "kpss_pvalue": float(kpss_p),
    }


def hill_estimator(ret: pd.Series, tail: str = "left", k_frac: float = 0.05) -> dict:
    """Оценка индекса хвоста Хилла. Для left tail работаем с |отриц. доходностями|."""
    r = ret.dropna().values
    x = -r[r < 0] if tail == "left" else r[r > 0]
    x = np.sort(x[x > 0])[::-1]
    k = max(int(len(x) * k_frac), 10)
    k = min(k, len(x) - 1)
    xk = x[:k + 1]
    logs = np.log(xk[:-1]) - np.log(xk[-1])
    gamma = float(np.mean(logs))  # индекс хвоста (1/alpha)
    return {"hill_gamma": gamma, "tail_alpha": float(1.0 / gamma) if gamma > 0 else np.nan, "k": int(k)}


def ar1_halflife(spread: pd.Series) -> dict:
    """AR(1) на ряде; half-life возврата = -ln(2)/ln(phi)."""
    s = spread.dropna()
    y = s.values[1:]
    x = s.values[:-1]
    x1 = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    phi = float(beta[1])
    hl = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.nan
    return {"ar1_phi": phi, "half_life_bars": hl}


def fit_garch(ret: pd.Series, scale: float = 100.0, model: str = "GARCH"):
    """GARCH(1,1)/EGARCH(1,1) через arch. Возвращает (res, cond_vol в исходном масштабе)."""
    from arch import arch_model
    r = ret.dropna() * scale
    vol = "EGARCH" if model.upper() == "EGARCH" else "GARCH"
    am = arch_model(r, mean="Constant", vol=vol, p=1, q=1, dist="t")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / scale
    cond_vol.index = ret.dropna().index
    return res, cond_vol


def realized_vol(ret: pd.Series, window: int = 60) -> pd.Series:
    return ret.rolling(window).std()


def vol_regime_quartiles(ret: pd.Series, window: int = 60) -> pd.Series:
    rv = realized_vol(ret, window)
    q = pd.qcut(rv.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return q
