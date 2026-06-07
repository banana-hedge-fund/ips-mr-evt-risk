"""Теория экстремальных значений (POT/GPD), VaR/CVaR и тесты адекватности.

Проверка H1 (EVT vs гаусс) и H3 (режимы хвостового риска).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def fit_gpd_pot(losses: np.ndarray, threshold_q: float = 0.95) -> dict:
    """POT: подгонка GPD к превышениям порога. losses — положительные величины потерь."""
    losses = np.asarray(losses)
    losses = losses[np.isfinite(losses)]
    u = np.quantile(losses, threshold_q)
    exceed = losses[losses > u] - u
    if len(exceed) < 20:
        return {"threshold": float(u), "n_exceed": int(len(exceed)), "xi": np.nan, "beta": np.nan}
    xi, _, beta = stats.genpareto.fit(exceed, floc=0)
    return {"threshold": float(u), "n_exceed": int(len(exceed)),
            "xi": float(xi), "beta": float(beta), "n": int(len(losses)), "Nu": int((losses > u).sum())}


def evt_var_cvar(gpd: dict, q: float = 0.99) -> dict:
    """VaR/CVaR из POT-GPD (McNeil-Frey-Embrechts, гл. 7)."""
    xi, beta, u = gpd["xi"], gpd["beta"], gpd["threshold"]
    n, Nu = gpd.get("n"), gpd.get("Nu")
    if not np.isfinite(xi) or n is None:
        return {"var": np.nan, "cvar": np.nan}
    var = u + (beta / xi) * (((n / Nu) * (1 - q)) ** (-xi) - 1)
    cvar = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.nan
    return {"var": float(var), "cvar": float(cvar)}


def gaussian_var(ret: np.ndarray, q: float = 0.99) -> float:
    mu, sd = np.mean(ret), np.std(ret, ddof=1)
    return float(-(mu + sd * stats.norm.ppf(1 - q)))


def historical_var(ret: np.ndarray, q: float = 0.99) -> float:
    return float(-np.quantile(ret, 1 - q))


def kupiec_pof(violations: int, n: int, q: float = 0.99) -> dict:
    """Kupiec POF-тест (unconditional coverage)."""
    p = 1 - q
    pi = violations / n if n else 0.0
    if violations == 0 or violations == n:
        lr = np.nan
    else:
        lr = -2 * (np.log((1 - p) ** (n - violations) * p ** violations)
                   - np.log((1 - pi) ** (n - violations) * pi ** violations))
    pval = float(1 - stats.chi2.cdf(lr, 1)) if np.isfinite(lr) else np.nan
    return {"violations": int(violations), "n": int(n), "rate": float(pi),
            "expected_rate": float(p), "lr_pof": float(lr) if np.isfinite(lr) else np.nan, "pvalue": pval}


def christoffersen_independence(breaches: np.ndarray) -> dict:
    """Тест независимости Christoffersen (кластеризация нарушений)."""
    b = np.asarray(breaches).astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(b[:-1], b[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"lr_ind": np.nan, "pvalue": np.nan}
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    def _ll(a, b_, pr):
        if pr <= 0 or pr >= 1:
            return 0.0
        return a * np.log(1 - pr) + b_ * np.log(pr)
    lr = -2 * ((_ll(n00, n01, p) + _ll(n10, n11, p))
              - (_ll(n00, n01, p01) + _ll(n10, n11, p11)))
    pval = float(1 - stats.chi2.cdf(lr, 1)) if np.isfinite(lr) else np.nan
    return {"lr_ind": float(lr), "pvalue": pval, "n01": n01, "n11": n11}


def classify_regime(vol_ratio: float, xi: float, viol_rate: float) -> str:
    """Режим хвостового риска (согласно спецификации риск-модуля)."""
    if (vol_ratio >= 2.0) or (xi >= 0.35) or (viol_rate >= 0.05):
        return "R_X"
    if (vol_ratio >= 1.2) or (xi >= 0.20) or (viol_rate >= 0.02):
        return "R_E"
    return "R_N"
