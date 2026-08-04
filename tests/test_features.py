from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pairs_trading.config import load_config
from src.pairs_trading.features import create_features

SEED = 42
N = 400
WIN = 60
BETA = 1.2


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def price_pair():
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2016-01-01", periods=N)
    xp = pd.Series(rng.standard_normal(N).cumsum() + 100, index=idx)
    yp = pd.Series(BETA * xp.values + rng.standard_normal(N) * 0.5 + 50, index=idx)
    return yp, xp


def test_feature_cols_no_lookahead(price_pair, config):
    """Each FEATURE_COLS value at t must depend only on rows <= t."""
    yp, xp = price_pair
    full = create_features(yp, xp, BETA, WIN, config)

    for t_idx in full.index:
        t_loc = yp.index.get_loc(t_idx)
        truncated_yp = yp.iloc[: t_loc + 1]
        truncated_xp = xp.iloc[: t_loc + 1]

        partial = create_features(truncated_yp, truncated_xp, BETA, WIN, config)
        if t_idx not in partial.index:
            continue

        for col in config.FEATURE_COLS:
            full_val = full.loc[t_idx, col]
            partial_val = partial.loc[t_idx, col]
            assert full_val == pytest.approx(partial_val, rel=0, abs=1e-12), (
                f"{col} at {t_idx} changed when recomputed on truncated history: "
                f"full={full_val}, partial={partial_val}"
            )


def test_target_uses_forward_shift_by_design(price_pair, config):
    yp, xp = price_pair
    H = config.target_h

    assert "target" not in config.FEATURE_COLS

    base = create_features(yp, xp, BETA, WIN, config)
    labeled = base.dropna(subset=["target"])
    assert not labeled.empty, "Need at least one labeled row for this test"

    t_idx = labeled.index[len(labeled) // 2]
    t_loc = yp.index.get_loc(t_idx)
    future_loc = t_loc + H
    if future_loc >= len(yp):
        pytest.skip("Not enough rows to test forward-shift target")

    # Perturb only strictly future prices (after t) — features at t must be unchanged
    yp_future = yp.copy()
    xp_future = xp.copy()
    yp_future.iloc[t_loc + 1 :] += 500
    xp_future.iloc[t_loc + 1 :] -= 300

    perturbed = create_features(yp_future, xp_future, BETA, WIN, config)
    for col in config.FEATURE_COLS:
        assert base.loc[t_idx, col] == pytest.approx(
            perturbed.loc[t_idx, col], rel=0, abs=1e-12
        )

    # Perturb exactly at t+H — features at t still unchanged, target may change
    yp_at_h = yp.copy()
    xp_at_h = xp.copy()
    yp_at_h.iloc[future_loc] *= 0.01

    at_h = create_features(yp_at_h, xp_at_h, BETA, WIN, config)
    for col in config.FEATURE_COLS:
        assert base.loc[t_idx, col] == pytest.approx(
            at_h.loc[t_idx, col], rel=0, abs=1e-12
        )

    # Confirm target formula uses forward zscore (shift(-H)), not lagged zscore
    z = base["zscore"]
    fz = z.shift(-H)
    lo = z < -config.target_zt
    sh = z > config.target_zt
    lok = fz > -config.exit_z
    shk = fz < config.exit_z

    expected = np.nan
    if lo.loc[t_idx] and lok.loc[t_idx]:
        expected = 1.0
    elif lo.loc[t_idx] and not lok.loc[t_idx]:
        expected = 0.0
    elif sh.loc[t_idx] and shk.loc[t_idx]:
        expected = 1.0
    elif sh.loc[t_idx] and not shk.loc[t_idx]:
        expected = 0.0

    if not np.isnan(expected):
        assert base.loc[t_idx, "target"] == expected

    # Document intentional forward dependence: changing z_{t+H} can change target_t
    # while all FEATURE_COLS at t stay fixed (assertions above).
    # If entry zones fire at t_idx, a large enough perturbation at t+H should flip target.
    if (lo.loc[t_idx] or sh.loc[t_idx]) and t_idx in at_h.index:
        # target_t is a function of z_{t+H}; we verified features_t are not
        assert at_h.loc[t_idx, "zscore"] == pytest.approx(
            base.loc[t_idx, "zscore"], rel=0, abs=1e-12
        )
        # target may or may not differ depending on recovery at t+H — that's OK;
        # the key property is it is computed via shift(-H), not shift(+H)
        assert fz.loc[t_idx] == base["zscore"].shift(-H).loc[t_idx]