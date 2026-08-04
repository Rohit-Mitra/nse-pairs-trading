from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.pairs_trading.backtest import simulate
from src.pairs_trading.config import PairsTradingConfig, load_config


class _MockScaler:
    def transform(self, X):
        return np.zeros_like(X, dtype=float)


class _MockModel:
    def __init__(self, signals: list[int]):
        self._signals = np.array(signals, dtype=int)

    def predict(self, X):
        return self._signals

    def predict_proba(self, X):
        p = self._signals.astype(float)
        return np.column_stack([1 - p, p])


def _expected_positions(
    zscore_lag1: list[float],
    pred_signal: list[int],
    entry_z: float,
    exit_z: float,
    ffill_limit: int,
) -> np.ndarray:
    """Mirror simulate's position rules exactly (reference implementation)."""
    d = pd.DataFrame({"zscore_lag1": zscore_lag1, "pred_signal": pred_signal})
    d["position"] = 0
    d.loc[(d["zscore_lag1"] < -entry_z) & (d["pred_signal"] == 1), "position"] = 1
    d.loc[(d["zscore_lag1"] > entry_z) & (d["pred_signal"] == 0), "position"] = -1
    d.loc[d["zscore_lag1"].abs() < exit_z, "position"] = 0
    d["position"] = d["position"].replace(0, np.nan).ffill(limit=ffill_limit).fillna(0)
    return d["position"].to_numpy()


def _make_simulate_input(
    zscore_lag1: list[float],
    config: PairsTradingConfig,
) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(zscore_lag1))
    data = pd.DataFrame(index=idx)
    for col in config.FEATURE_COLS:
        data[col] = 0.0
    data["zscore_lag1"] = zscore_lag1
    data["zscore"] = zscore_lag1
    return data


@pytest.fixture
def config() -> PairsTradingConfig:
    return load_config()


def test_simulate_position_sequence(config: PairsTradingConfig):
    """
    Hand-built z-score path:
      - long entry when zscore_lag1 < -ENTRY_Z and model predicts 1
      - hold via ffill through neutral bars
      - short entry when zscore_lag1 > ENTRY_Z and model predicts 0
      - exit-zone rows (|z| < EXIT_Z) become 0 then forward-fill prior position
    """
    zscore_lag1 = [0.0, -1.5, -1.2, 0.8, 0.8, 1.5, 0.3]
    pred_signal = [0, 1, 1, 1, 1, 0, 0]

    data = _make_simulate_input(zscore_lag1, config)
    model = _MockModel(pred_signal)
    scaler = _MockScaler()

    y_ret = pd.Series(0.01, index=data.index)
    x_ret = pd.Series(0.005, index=data.index)

    result = simulate(data, model, scaler, beta=1.0, y_ret=y_ret, x_ret=x_ret, cap=1_000_000, config=config)

    expected = _expected_positions(
        zscore_lag1,
        pred_signal,
        config.entry_z,
        config.exit_z,
        config.ffill_limit,
    )
    assert result["position"].to_numpy().tolist() == expected.tolist()
    # long (1) through neutral bars, then flip to short (-1) with exit-zone ffill
    assert expected.tolist() == [0, 1, 1, 1, 1, -1, -1]


def test_simulate_ffill_capped_at_config_limit(config: PairsTradingConfig):
    """Position ffill must respect config.ffill_limit (FIX 5)."""
    cfg = replace(config, ffill_limit=2)
    zscore_lag1 = [0.0, -1.5, 0.8, 0.8, 0.8, 0.8]
    pred_signal = [0, 1, 0, 0, 0, 0]

    data = _make_simulate_input(zscore_lag1, cfg)
    result = simulate(
        data,
        _MockModel(pred_signal),
        _MockScaler(),
        beta=1.0,
        y_ret=pd.Series(0.0, index=data.index),
        x_ret=pd.Series(0.0, index=data.index),
        cap=100_000,
        config=cfg,
    )

    expected = _expected_positions(
        zscore_lag1,
        pred_signal,
        cfg.entry_z,
        cfg.exit_z,
        cfg.ffill_limit,
    )
    assert result["position"].to_numpy().tolist() == expected.tolist()
    # long held for 2 bars after entry, then flat
    assert expected.tolist() == [0, 1, 1, 1, 0, 0]


def test_simulate_uses_presliced_returns(config: PairsTradingConfig):
    """simulate must use the y_ret/x_ret arguments, not any global price frame."""
    zscore_lag1 = [0.0, -1.5, 0.8]
    pred_signal = [0, 1, 0]
    data = _make_simulate_input(zscore_lag1, config)

    y_ret = pd.Series([0.0, 0.0, 0.05], index=data.index)
    x_ret = pd.Series([0.0, 0.0, 0.0], index=data.index)

    result = simulate(
        data,
        _MockModel(pred_signal),
        _MockScaler(),
        beta=1.0,
        y_ret=y_ret,
        x_ret=x_ret,
        cap=100.0,
        config=config,
    )
    # position ffill: [0, 1, 1]; lagged position at bar 2 is 1
    assert result.loc[data.index[2], "pnl_gross"] == pytest.approx(5.0)
