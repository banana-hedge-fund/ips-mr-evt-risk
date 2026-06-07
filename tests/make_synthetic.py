"""Генератор синтетических минутных свечей для смоук-теста пайплайна.

НЕ используется в итоговом анализе — только для проверки кода без реальных данных.
GBM с меан-реверсионной компонентой и режимами волатильности + тяжёлые хвосты (t).
"""
import pathlib
import numpy as np
import pandas as pd


def make(symbol="TEST", month="2024-03", n_days=20, seed=0, out=None):
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="1min")
    # режимы волатильности
    vol = np.where((np.arange(n) // 1440) % 5 == 0, 0.0008, 0.0003)
    shocks = rng.standard_t(4, n) * vol
    # меан-реверсия в лог-цене (OU-подобный)
    logp = np.zeros(n)
    mu = np.log(100.0)
    theta = 0.02
    for t in range(1, n):
        logp[t] = logp[t-1] + theta * (mu - (logp[t-1] + mu) ) * 0 + shocks[t]
    price = 100.0 * np.exp(np.cumsum(shocks))
    # OHLC вокруг close
    close = price
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.0002, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.0002, n)))
    volu = rng.lognormal(3, 1, n)
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": volu}, index=idx)
    df.index.name = "dt"
    if out is None:
        out = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(out / f"{symbol}-1m-{month}.parquet", index=False)
    return out / f"{symbol}-1m-{month}.parquet"


if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]:
        for m in ["2024-03", "2024-08", "2026-01", "2026-04"]:
            p = make(sym, m, n_days=15, seed=hash((sym, m)) % 1000)
            print("wrote", p)
