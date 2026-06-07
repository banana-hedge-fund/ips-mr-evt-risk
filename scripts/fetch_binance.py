"""Скачивание минутных klines Binance USD-M Futures с data.binance.vision.

Запускается в GitHub Actions (раннер имеет полный доступ в интернет, в отличие
от песочницы). Сохраняет компактные parquet в data/raw/ для последующего
клонирования и анализа.
"""
import io
import sys
import zipfile
import pathlib
import urllib.request
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
MONTHS = ["2024-03", "2024-08", "2026-01", "2026-04"]
MARKET = "futures/um"
INTERVAL = "1m"
BASE = "https://data.binance.vision/data"

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

OUT = pathlib.Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)


def monthly_url(sym, month):
    return f"{BASE}/{MARKET}/monthly/klines/{sym}/{INTERVAL}/{sym}-{INTERVAL}-{month}.zip"


def daily_urls(sym, month):
    import calendar
    y, m = map(int, month.split("-"))
    ndays = calendar.monthrange(y, m)[1]
    for d in range(1, ndays + 1):
        day = f"{month}-{d:02d}"
        yield f"{BASE}/{MARKET}/daily/klines/{sym}/{INTERVAL}/{sym}-{INTERVAL}-{day}.zip"


def fetch_zip_csv(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    name = zf.namelist()[0]
    raw = zf.read(name)
    first = raw[:20].decode("utf-8", "ignore")
    header = 0 if first.startswith("open_time") else None
    df = pd.read_csv(io.BytesIO(raw), header=header, names=None if header == 0 else COLS)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_month(sym, month):
    try:
        df = fetch_zip_csv(monthly_url(sym, month))
        print(f"  monthly OK {sym} {month}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  monthly FAIL {sym} {month}: {e}; trying daily")
    parts = []
    for u in daily_urls(sym, month):
        try:
            parts.append(fetch_zip_csv(u))
        except Exception as e:
            print(f"    daily FAIL {u.split('/')[-1]}: {e}")
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    print(f"  daily OK {sym} {month}: {len(df)} rows ({len(parts)} days)")
    return df


def main():
    manifest = []
    for sym in SYMBOLS:
        for month in MONTHS:
            df = load_month(sym, month)
            if df is None or df.empty:
                manifest.append((sym, month, 0))
                continue
            ot = df["open_time"].astype("int64")
            unit = "us" if ot.iloc[0] > 1e15 else "ms"
            df["dt"] = pd.to_datetime(ot, unit=unit, utc=True)
            keep = ["dt", "open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume"]
            df = df[keep].sort_values("dt").reset_index(drop=True)
            for c in keep[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            path = OUT / f"{sym}-1m-{month}.parquet"
            df.to_parquet(path, index=False, compression="zstd")
            manifest.append((sym, month, len(df)))
            print(f"saved {path} ({len(df)} rows, {path.stat().st_size/1e6:.2f} MB)")
    man = pd.DataFrame(manifest, columns=["symbol", "month", "rows"])
    man.to_csv("data/raw/manifest.csv", index=False)
    print(man.to_string(index=False))
    if (man["rows"] == 0).any():
        print("WARNING: some month-symbol combos returned 0 rows", file=sys.stderr)


if __name__ == "__main__":
    main()
