from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.pairs_trading.config import PairsTradingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

LATE_LISTING_CUTOFF = pd.Timestamp("2016-06-01")
NIFTY_TICKER = "^NSEI"


def _cache_key(prefix: str, tickers: list[str], start: str, end: str) -> str:
    payload = "|".join([prefix, *sorted(tickers), start, end])
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(prefix: str, tickers: list[str], start: str, end: str) -> Path:
    return CACHE_DIR / f"{prefix}_{_cache_key(prefix, tickers, start, end)}.parquet"


def _extract_adj_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            return raw["Adj Close"].copy()
        return raw["Close"].copy()

    col = "Adj Close" if "Adj Close" in raw.columns else "Close"
    price_data = raw[[col]].copy()
    price_data.columns = tickers[:1]
    return price_data


def _clean_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    price_data = price_data.ffill().bfill()

    bad = [c for c in price_data.columns if price_data[c].isna().mean() > 0.05]
    if bad:
        price_data.drop(columns=bad, inplace=True)

    late = [
        c for c in price_data.columns
        if (f := price_data[c].first_valid_index()) and f > LATE_LISTING_CUTOFF
    ]
    if late:
        price_data.drop(columns=late, inplace=True)

    price_data.dropna(axis=1, how="all", inplace=True)
    return price_data.ffill().bfill()


def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, progress=True)
    price_data = _extract_adj_close(raw, tickers)
    return _clean_prices(price_data)


def load_prices(config: PairsTradingConfig) -> pd.DataFrame:
    tickers = config.ALL_TICKERS
    start, end = config.TRAIN_START, config.TEST_END
    path = _cache_path("prices", tickers, start, end)

    if path.is_file():
        return pd.read_parquet(path)

    price_data = _download_prices(tickers, start, end)

    path.parent.mkdir(parents=True, exist_ok=True)
    price_data.to_parquet(path)

    return price_data


def _download_nifty(start: str, end: str) -> pd.Series:
    nr = yf.download(NIFTY_TICKER, start=start, end=end, progress=False)
    nifty = nr["Adj Close"] if "Adj Close" in nr.columns else nr["Close"]
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.iloc[:, 0]
    return pd.Series(nifty.values, index=nifty.index).ffill().bfill()


def load_nifty(config: PairsTradingConfig) -> pd.Series | None:
    start, end = config.TRAIN_START, config.TEST_END
    path = _cache_path("nifty", [NIFTY_TICKER], start, end)

    if path.is_file():
        series = pd.read_parquet(path)
        # parquet round-trip may return DataFrame with one column
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series

    try:
        nifty = _download_nifty(start, end)
    except Exception:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    nifty.to_frame("nifty").to_parquet(path)

    return nifty