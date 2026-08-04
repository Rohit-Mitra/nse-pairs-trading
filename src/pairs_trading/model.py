from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.pairs_trading.backtest import simulate
from src.pairs_trading.config import PairsTradingConfig
from src.pairs_trading.features import create_features
from src.pairs_trading.stats import ols_hedge

def build_model(fast: bool = False):
    return LGBMClassifier(
        n_estimators=100 if fast else 300,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=21,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )


def train_predict(
    tr: pd.DataFrame,
    config: PairsTradingConfig,
    feat_cols: list[str] | None = None,
    fast: bool = False,
):
    feat_cols = config.FEATURE_COLS if feat_cols is None else feat_cols
    tr = tr.dropna(subset=["target"])
    tr["ml_target"] = tr["target"].astype(int)
    if len(tr) < 50 or len(np.unique(tr["ml_target"])) < 2:
        return None, None, None
    sc = StandardScaler()
    X = sc.fit_transform(tr[feat_cols])
    y = tr["ml_target"]
    m = build_model(fast)
    m.fit(X, y)
    return m, sc, accuracy_score(y, m.predict(X))


def _pair_beta(price_data: pd.DataFrame, pair: tuple[str, str], config: PairsTradingConfig, cap_pair: float = 1.0) -> float: # default cap_pair is 1.0
    ys, xs = pair
    tr = price_data.loc[config.TRAIN_START : config.TRAIN_END].dropna()
    try:
        b, _ = ols_hedge(tr[ys], tr[xs])
    except Exception:
        b = 1.0
    return b * cap_pair


def walk_forward_accuracy(
    price_data: pd.DataFrame,
    pair: tuple[str, str],
    config: PairsTradingConfig,
) -> float:
    ys, xs = pair
    beta = _pair_beta(price_data, pair, config)

    fa = create_features(price_data[ys], price_data[xs], beta, 60, config)
    wf_acc = []
    for ts, te, vs_, ve in config.WF_FOLDS:
        t, v = fa.loc[ts:te].copy(), fa.loc[vs_:ve].copy()
        t = t.dropna(subset=["target"])
        t["ml_target"] = t["target"].astype(int)
        v = v.dropna(subset=["target"])
        if len(t) < 50 or len(v) < 20 or len(np.unique(t["ml_target"])) < 2:
            continue
        sc = StandardScaler()
        Xt = sc.fit_transform(t[config.FEATURE_COLS])
        m = build_model(True)
        m.fit(Xt, t["ml_target"])
        wf_acc.append(
            accuracy_score(
                v["target"].astype(int),
                m.predict(sc.transform(v[config.FEATURE_COLS])),
            )
        )
    return float(np.mean(wf_acc)) if wf_acc else 0.0


def optimize_window(
    price_data: pd.DataFrame,
    pair: tuple[str, str],
    config: PairsTradingConfig,
    cap_pair: float,
) -> int:
    ys, xs = pair
    beta = _pair_beta(price_data, pair, config, cap_pair)
    all_returns = price_data.pct_change()

    best_pnl, best_win = -np.inf, 60
    for w in config.window_opts:
        fd = create_features(price_data[ys], price_data[xs], beta, w, config)
        td = fd.loc[config.TRAIN_START : config.TRAIN_END].copy()
        vd = fd.loc[config.VAL_START : config.VAL_END].copy()
        if len(vd) < 30:
            continue
        m, sc, _ = train_predict(td, config, fast=True)
        if m is None:
            continue
        yr_v = all_returns[ys].loc[config.VAL_START : config.VAL_END]
        xr_v = all_returns[xs].loc[config.VAL_START : config.VAL_END]
        sp = simulate(vd, m, sc, beta, yr_v, xr_v, config.init_capital, config)
        p = sp["pnl_net"].sum()
        if p > best_pnl:
            best_pnl, best_win = p, w

    return best_win
