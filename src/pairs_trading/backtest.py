from __future__ import annotations

import numpy as np
import pandas as pd

from src.pairs_trading.config import PairsTradingConfig


def simulate(
    data: pd.DataFrame,
    model,
    scaler,
    beta: float,
    y_ret: pd.Series,
    x_ret: pd.Series,
    cap: float,
    config: PairsTradingConfig,
):
    d = data.copy()
    feat = scaler.transform(d[config.FEATURE_COLS])
    d["pred_signal"], d["pred_proba"] = model.predict(feat), model.predict_proba(feat)[:, 1]
    d["position"] = 0
    d.loc[(d["zscore_lag1"] < -config.entry_z) & (d["pred_signal"] == 1), "position"] = 1
    d.loc[(d["zscore_lag1"] > config.entry_z) & (d["pred_signal"] == 0), "position"] = -1
    d.loc[d["zscore_lag1"].abs() < config.exit_z, "position"] = 0
    d["position"] = d["position"].replace(0, np.nan).ffill(limit=config.ffill_limit).fillna(0)
    yr = y_ret.reindex(d.index).fillna(0)
    xr = x_ret.reindex(d.index).fillna(0)
    d["pnl_gross"] = d["position"].shift(1).fillna(0) * cap * (yr - beta * xr)
    d["txn_cost"] = d["position"].diff().fillna(0).abs() * cap * config.txn_cost
    d["pnl_net"] = d["pnl_gross"] - d["txn_cost"]
    d["cum_pnl"] = d["pnl_net"].cumsum()
    d["portfolio_value"] = cap + d["cum_pnl"]
    return d


def risk_metrics(pnl, cum):
    dr = pnl.dropna()
    c = cum.dropna()
    if len(dr) == 0:
        return {}
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = (c - c.cummax()).min()
    ar = dr.mean() * 252
    gp, gl = dr[dr > 0].sum(), dr[dr < 0].abs().sum()
    bp = (c < c.cummax()).astype(int)
    mdd = (bp * (bp.groupby((bp != bp.shift()).cumsum()).cumcount() + 1)).max()
    return {
        "Sharpe": round(sh, 3),
        "MaxDD": round(dd, 2),
        "MaxDD_Days": int(mdd),
        "Calmar": round(ar / abs(dd), 3) if dd != 0 else 0,
        "VaR95": round(dr.quantile(0.05), 2),
        "HitRate": round((dr > 0).mean(), 4),
        "ProfitFactor": round(gp / gl, 3) if gl > 0 else None,
        "AnnRet": round(ar, 2),
    }
