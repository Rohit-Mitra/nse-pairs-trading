# NSE Pairs Trading  [click here](https://nse-pairs-trading.streamlit.app)

A statistical arbitrage backtester for NSE-listed equities. The system discovers cointegrated pairs across 14 sectors, trains a LightGBM classifier on spread features, and simulates a pairs-trading strategy on a held-out test period (2022–2025).

The codebase is a modular refactor of the original monolith script (`kothu_pair.py`). Numeric behavior is preserved from that baseline and guarded by regression tests.

## How it works

1. **Download** adjusted-close prices via Yahoo Finance (cached locally as parquet).
2. **Pair selection** — Engle–Granger / Johansen cointegration, ADF, half-life, and Hurst RS filters on the train window (2016–2020).
3. **Validation pass** — Re-filter pairs on the 2021 validation window; fallback to top half by EG p-value if too strict.
4. **Per-pair modeling** — Walk-forward accuracy (diagnostic), window optimization on validation PnL, final LightGBM train on train data.
5. **Backtest** — Z-score entry/exit with ML gating, capped position forward-fill, transaction costs.
6. **Report** — Console summary tables and CSV export for the dashboard.

## Architecture

```text
nse-pairs-trading/
├── config/
│   └── config.yaml          # Sectors, dates, thresholds, feature cols
├── dashboard/
│   └── app.py                # Reads CSVs from output dir (PAIRS_OUTPUT_DIR)
├── data/
│   └── cache/                # Parquet price cache (gitignored)
├── output/                   # Default backtest CSV output
├── scripts/
│   └── run_backtest.py       # CLI entry point
├── src/pairs_trading/
│   ├── config.py              # YAML → PairsTradingConfig
│   ├── data_loader.py         # yfinance download + cache
│   ├── stats.py                # Cointegration, ADF, half-life, OLS hedge
│   ├── pair_selection.py       # Sector / cross-sector pair filtering
│   ├── features.py             # Spread z-scores, ML target
│   ├── model.py                 # LightGBM train, WF accuracy, window opt
│   ├── backtest.py              # simulate(), risk_metrics()
│   ├── pipeline.py              # validation_filter + run_pipeline()
│   └── report.py                # CSV export + console tables
├── tests/
│   ├── golden/                  # Frozen baseline CSVs
│   ├── test_stats.py
│   ├── test_features.py
│   ├── test_backtest.py
│   └── test_regression.py       # Full pipeline vs golden
├── kothu_pair.py                # Original monolith (reference)
├── pytest.ini
└── requirements.txt
```


### Module responsibilities

| Module | Role |
|--------|------|
| `data_loader` | Fetch and cache NSE prices and Nifty 50 benchmark |
| `pair_selection` | Build sector combos, apply cointegration filters |
| `pipeline` | Orchestrate validation pass and per-pair train/test loop |
| `model` | LightGBM training, walk-forward accuracy, window selection |
| `backtest` | Position logic, PnL simulation, risk metrics |
| `report` | Write `cointegrated_pairs.csv`, `nse_pairs_summary.csv`, `nse_pairs_trades_detailed.csv` |

## Setup

Requires **Python 3.10+** (tested on 3.14).

```powershell
cd nse-pairs-trading

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
