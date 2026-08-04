from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs_trading.stats import adf_test, eg_test, half_life, ols_hedge

COINT_P = 0.05
ADF_P = 0.10
N = 500
SEED = 42


def _random_walk(n: int, rng: np.random.Generator) -> pd.Series:
    return pd.Series(rng.standard_normal(n).cumsum())


def _cointegrated_pair(n: int, rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    """Random walk x plus y = beta*x + stationary noise."""
    x = _random_walk(n, rng)
    spread = pd.Series(rng.standard_normal(n))
    y = 1.2 * x + spread
    return y, x


def _independent_random_walks(n: int, rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    return _random_walk(n, rng), _random_walk(n, rng)


def _mean_reverting_ar1(n: int, rng: np.random.Generator, phi: float = 0.90) -> pd.Series:
    """Stationary AR(1): x_t = phi * x_{t-1} + eps."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return pd.Series(x)


class TestCointegration:
    def test_cointegrated_pair_detected(self):
        rng = np.random.default_rng(SEED)
        y, x = _cointegrated_pair(N, rng)

        eg_p = eg_test(y, x)
        assert eg_p < COINT_P, f"Expected cointegration, got EG p={eg_p}"

        b, a = ols_hedge(y, x)
        spread = y - b * x - a
        adf_p = adf_test(spread)
        assert adf_p < ADF_P, f"Expected stationary spread, got ADF p={adf_p}"

    def test_independent_random_walks_not_cointegrated(self):
        rng = np.random.default_rng(SEED)
        y, x = _independent_random_walks(N, rng)

        eg_p = eg_test(y, x)
        assert eg_p >= COINT_P, f"Expected no cointegration, got EG p={eg_p}"


class TestHalfLife:
    def test_mean_reverting_series_finite_positive(self):
        rng = np.random.default_rng(SEED)
        s = _mean_reverting_ar1(N, rng, phi=0.90)
        hl = half_life(s)
        assert np.isfinite(hl)
        assert hl > 0
    def test_random_walk_infinite_half_life(self):
            LARGE_N = 8000
            half_lives = []
            for seed in range(15):
                rng = np.random.default_rng(seed)
                s = _random_walk(LARGE_N, rng)
                half_lives.append(half_life(s))
            median_hl = np.median(half_lives)
            assert median_hl == np.inf or median_hl > 400, (
                f"Expected negligible mean reversion at large N, "
                f"median half-life={median_hl}"
            )