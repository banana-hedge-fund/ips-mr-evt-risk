"""Оркестратор: прогон всех блоков анализа по символам и окнам.

Запуск:  python -m src.run_all
Результаты — в results/ (json, csv, png).
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import data_io, econometrics as ec, evt, metrics, backtest as bt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]
WINDOWS = {"2024": ["2024-03", "2024-08"], "2026": ["2026-01", "2026-04"]}


def analyze_returns(ret: pd.Series) -> dict:
    d = ec.descriptive_stats(ret)
    d.update(ec.stationarity_tests(ret))
    d.update(ec.hill_estimator(ret, tail="left"))
    losses = -ret.dropna().values
    gpd = evt.fit_gpd_pot(losses, 0.95)
    vc99 = evt.evt_var_cvar(gpd, 0.99)
    d["gpd_xi"] = gpd["xi"]
    d["evt_var99"] = vc99["var"]
    d["evt_cvar99"] = vc99["cvar"]
    d["gauss_var99"] = evt.gaussian_var(ret.dropna().values, 0.99)
    d["hist_var99"] = evt.historical_var(ret.dropna().values, 0.99)
    # backtest адекватности VaR (Kupiec) для гауссовой модели
    r = ret.dropna().values
    breaches = (r < -d["gauss_var99"]).astype(int)
    d["kupiec_gauss"] = evt.kupiec_pof(int(breaches.sum()), len(r), 0.99)
    breaches_evt = (r < -d["evt_var99"]).astype(int) if np.isfinite(d["evt_var99"]) else breaches * 0
    d["kupiec_evt"] = evt.kupiec_pof(int(breaches_evt.sum()), len(r), 0.99)
    return d


def run():
    summary_rows = []
    econ_rows = []
    for sym in SYMBOLS:
        for tag, months in WINDOWS.items():
            try:
                df = data_io.load_window(sym, months)
            except FileNotFoundError as e:
                print(f"[skip] {sym} {tag}: {e}")
                continue
            integ = data_io.integrity_report(df)
            print(f"[data] {sym} {tag}: {integ['rows']} баров, пропуски {integ['missing_pct']}%")

            ar = analyze_returns(df["ret"])
            ar.update({"symbol": sym, "window": tag, **{f"data_{k}": v for k, v in integ.items()}})
            econ_rows.append(ar)

            res_base = bt.backtest(df, adaptive=False)
            res_adap = bt.backtest(df, adaptive=True)
            for name, res in [("baseline", res_base), ("adaptive", res_adap)]:
                s = metrics.summarize(res["equity"], res["pnl"])
                s.update({"symbol": sym, "window": tag, "config": name, "n_trades": res["n_trades"]})
                summary_rows.append(s)

            # график equity
            fig, ax = plt.subplots(figsize=(9, 4))
            res_base["equity"].plot(ax=ax, label="baseline (без риск-менеджмента)")
            res_adap["equity"].plot(ax=ax, label="adaptive (EVT/VaR оверлей)")
            ax.set_title(f"{sym} {tag}: equity curve"); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(RESULTS / f"equity_{sym}_{tag}.png", dpi=110); plt.close(fig)

    summ = pd.DataFrame(summary_rows)
    econ = pd.DataFrame(econ_rows)
    if not summ.empty:
        summ.to_csv(RESULTS / "backtest_summary.csv", index=False)
        summ.to_json(RESULTS / "backtest_summary.json", orient="records", force_ascii=False, indent=2)
    if not econ.empty:
        econ.drop(columns=[c for c in econ.columns if c.startswith("kupiec")], errors="ignore")\
            .to_csv(RESULTS / "econometrics_summary.csv", index=False)
        with open(RESULTS / "econometrics_full.json", "w", encoding="utf-8") as f:
            json.dump(econ_rows, f, ensure_ascii=False, indent=2, default=str)
    print("\n=== BACKTEST SUMMARY ===")
    if not summ.empty:
        print(summ.to_string(index=False))
    print("\nГотово. Результаты в results/.")
    return summ, econ


if __name__ == "__main__":
    run()
