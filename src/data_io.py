"""Загрузка и подготовка минутных свечей Binance USD-M Futures.

Ожидаемые файлы в data/raw/: {SYMBOL}-1m-{YYYY-MM}.parquet или одноимённые .zip
(месячные архивы клайнов с data.binance.vision). Функции приводят данные
к единому виду: DatetimeIndex (UTC, 1m), колонки OHLCV + лог-доходности.
"""
from __future__ import annotations
import io
import zipfile
import pathlib
import numpy as np
import pandas as pd

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _read_zip(path: pathlib.Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    first = raw[:20].decode("utf-8", "ignore")
    header = 0 if first.startswith("open_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header,
                     names=None if header == 0 else KLINE_COLS)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 1e15 else "ms"
    df = df.copy()
    df["dt"] = pd.to_datetime(ot, unit=unit, utc=True)
    keep = ["dt", "open", "high", "low", "close", "volume"]
    for opt in ["quote_volume", "count", "taker_buy_volume"]:
        if opt in df.columns:
            keep.append(opt)
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[keep]
    df = df.dropna(subset=["close"]).sort_values("dt").drop_duplicates("dt")
    return df.set_index("dt")


def load_symbol_month(symbol: str, month: str, raw_dir: pathlib.Path = RAW) -> pd.DataFrame:
    pq = raw_dir / f"{symbol}-1m-{month}.parquet"
    zp = raw_dir / f"{symbol}-1m-{month}.zip"
    if pq.exists():
        df = pd.read_parquet(pq)
        if "dt" in df.columns:
            df = df.set_index(pd.to_datetime(df["dt"], utc=True)).drop(columns=["dt"])
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    if zp.exists():
        return _normalize(_read_zip(zp))
    raise FileNotFoundError(f"Нет данных для {symbol} {month}: ожидался {pq.name} или {zp.name}")


def load_window(symbol: str, months: list[str], raw_dir: pathlib.Path = RAW) -> pd.DataFrame:
    parts = [load_symbol_month(symbol, m, raw_dir) for m in months]
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = add_returns(df)
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = np.log(df["close"]).diff()
    df = df.dropna(subset=["ret"])
    return df


def integrity_report(df: pd.DataFrame) -> dict:
    """Проверка целостности минутного ряда."""
    idx = df.index
    expected = pd.date_range(idx.min(), idx.max(), freq="1min", tz="UTC")
    missing = len(expected) - len(idx.intersection(expected))
    return {
        "rows": int(len(df)),
        "start": str(idx.min()),
        "end": str(idx.max()),
        "expected_minutes": int(len(expected)),
        "missing_minutes": int(missing),
        "missing_pct": round(100 * missing / max(len(expected), 1), 3),
        "dup_index": int(idx.duplicated().sum()),
    }
