# results

Сводные выходы пайплайна (воспроизводятся командой `python -m src.run_all`).

| Файл | Содержание |
|---|---|
| `aggregate_econometrics_by_year.csv` | H1: асимметрия, эксцесс, GPD ξ, EVT/гаусс/ист. VaR99 по годам |
| `aggregate_by_year_config.csv` | H4: метрики baseline vs adaptive по годам |
| `h2_vol_regime.csv` | H2: Sharpe MR по квартилям волатильности |
| `backtest_summary.csv` | бэктест по каждой выборке (symbol × month × config) |
| `econometrics_summary.csv` | эконометрика по каждой выборке |

PNG-графики (equity_*, fig_*) генерируются тем же запуском и не хранятся в репо (.gitignore).
