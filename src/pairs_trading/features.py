from __future__ import annotations

import numpy as np
import pandas as pd

from src.pairs_trading.config import PairsTradingConfig
from src.pairs_trading.stats import compute_rsi


def create_features(yp, xp, beta, win, config: PairsTradingConfig):
    sp = yp - beta * xp
    rm, rs_ = sp.rolling(win).mean(), sp.rolling(win).std().replace(0, np.nan)
    df = pd.DataFrame(index=sp.index)
    df["spread"], df["zscore"] = sp, (sp - rm) / rs_
    df["zscore_lag1"] = df["zscore"].shift(1)
    vs, vl = sp.rolling(10).std(), sp.rolling(40).std()
    df["vol_ratio"] = (vs / vl.replace(0, np.nan)).fillna(1)
    df["rel_momentum"] = (yp.pct_change(5) - xp.pct_change(5)).fillna(0)
    df["rsi_spread"] = compute_rsi(sp, 14)
    # FIX 1: zscore-reversion target (did zscore cross back past EXIT_Z?)
    fz = df["zscore"].shift(-config.target_h)
    lo = df["zscore"] < -config.target_zt          # long-spread entry zone
    sh = df["zscore"] > config.target_zt           # short-spread entry zone
    lok = fz > -config.exit_z                       # zscore recovered toward mean
    shk = fz < config.exit_z                        # zscore recovered toward mean
    tgt = pd.Series(np.nan, index=df.index)
    tgt[lo & lok] = 1.0
    tgt[lo & ~lok] = 0.0
    tgt[sh & shk] = 1.0
    tgt[sh & ~shk] = 0.0
    df["target"] = tgt
    df.dropna(subset=config.FEATURE_COLS, inplace=True)
    return df