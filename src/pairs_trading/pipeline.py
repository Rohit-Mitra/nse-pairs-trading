# src/pairs_trading/pipeline.py
from dataclasses import dataclass
import pandas as pd
from src.pairs_trading.config import PairsTradingConfig
from src.pairs_trading.data_loader import load_prices
from src.pairs_trading.pair_selection import select_pairs
from src.pairs_trading.model import walk_forward_accuracy, optimize_window, train_predict
from src.pairs_trading.features import create_features
from src.pairs_trading.backtest import simulate, risk_metrics
from src.pairs_trading.stats import ols_hedge, adf_test, half_life, rs_exponent

VAL_MIN_ROWS = 50
VAL_ADF_P = 0.20  # hardcoded in monolith, not in config


def validation_filter(
    price_data: pd.DataFrame,
    pairs: list[tuple[str, str]],
    pair_info: list[dict],
    config: PairsTradingConfig,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Section #6 — preserve exact monolith logic + fallback."""
    val_pairs, val_info = [], []

    for i, (ys, xs) in enumerate(pairs):
        tr = price_data.loc[config.TRAIN_START : config.TRAIN_END, [ys, xs]].dropna()
        try:
            b, _ = ols_hedge(tr[ys], tr[xs])
        except Exception:
            b = 1.0

        vp = price_data.loc[config.VAL_START : config.VAL_END, [ys, xs]].dropna()
        if len(vp) < VAL_MIN_ROWS:
            continue

        vs = vp[ys] - b * vp[xs]  # note: no intercept, matches monolith
        if (
            adf_test(vs) < VAL_ADF_P
            or half_life(vs) < config.hl_max
            or rs_exponent(vs) < config.rs_thresh
        ):
            val_pairs.append((ys, xs))
            val_info.append(pair_info[i])

    # FIX 3 fallback: too few survivors → top half by EG_P
    if len(val_pairs) < max(2, len(pairs) // 4):
        sorted_pi = sorted(pair_info, key=lambda d: d["EG_P"])
        half = max(2, len(sorted_pi) // 2)
        val_info = sorted_pi[:half]
        val_pairs = [(d["Stock1"], d["Stock2"]) for d in val_info]

    if not val_pairs:
        val_pairs, val_info = pairs, pair_info

    return val_pairs, val_info


@dataclass(frozen=True)
class PipelineResult:
    pairs: pd.DataFrame
    summary: pd.DataFrame
    trades: pd.DataFrame          # concat of per-pair daily rows
    port_pnl: pd.Series           # sum of per-pair pnl_net (for portfolio Sharpe)
    n_pairs: int                  # post-validation count (for CAP_PAIR / win rate)


def run_pipeline(config: PairsTradingConfig) -> PipelineResult:
    price_data = load_prices(config)
    all_returns = price_data.pct_change()

    pairs, pair_info = select_pairs(price_data, config)
    pairs_df = pd.DataFrame(pair_info)

    pairs, pair_info = validation_filter(price_data, pairs, pair_info, config)
    n_pairs = len(pairs)
    cap_pair = config.init_capital / max(n_pairs, 1)

    summary_rows: list[dict] = []

    for ys, xs in pairs:
        sec = next(
            (d["Sector"] for d in pair_info if d["Stock1"] == ys and d["Stock2"] == xs),
            "?",
        )
        tr = price_data.loc[config.TRAIN_START : config.TRAIN_END].dropna()
        try:
            b, _ = ols_hedge(tr[ys], tr[xs])
        except Exception:
            b = 1.0

        wf = walk_forward_accuracy(price_data, (ys, xs), config)
        best_win = optimize_window(price_data, (ys, xs), config, cap_pair)  # see fix below

        ff = create_features(price_data[ys], price_data[xs], b, best_win, config)
        ft = ff.loc[config.TRAIN_START : config.TRAIN_END].copy()
        ftest = ff.loc[config.TEST_START : config.TEST_END].copy()
        fm, fsc, tacc = train_predict(ft, config, fast=False)
        if fm is None or len(ftest) < 20:
            continue

        yr_te = all_returns[ys].loc[config.TEST_START : config.TEST_END]
        xr_te = all_returns[xs].loc[config.TEST_START : config.TEST_END]
        res = simulate(ftest, fm, fsc, b, yr_te, xr_te, cap_pair, config)

        pnl = res["pnl_net"].sum()
        ret = pnl / cap_pair * 100
        ntrades = int((res["position"].diff() != 0).sum())
        met = risk_metrics(res["pnl_net"], res["cum_pnl"])
        trades_all.append(pd.DataFrame({
            "Pair": f"{ys}/{xs}", "Sector": sec, "Date": res.index,
            "Position": res["position"].values, "Spread": res["spread"].values,
            "ZScore": res["zscore"].values, "PnL_Net": res["pnl_net"].values,
            "Cum_PnL": res["cum_pnl"].values, "Value": res["portfolio_value"].values,
        }))
        port_pnl = port_pnl.add(res["pnl_net"].fillna(0), fill_value=0) if not port_pnl.empty else res["pnl_net"].fillna(0)
        summary_rows.append({
            "Pair": f"{ys}/{xs}",
            "Sector": sec,
            "Beta": round(b, 4),
            "Window": best_win,
            "Capital": cap_pair,
            "WF_Acc": round(wf, 4),
            "Train_Acc": round(tacc, 4),
            "Trades": ntrades,
            "Net_PnL": round(pnl, 2),
            "Return%": round(ret, 2),
            "EndValue": round(cap_pair + pnl, 2),
            "TxnCost": round(res["txn_cost"].sum(), 2),
            **met,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Net_PnL", ascending=False)
    return PipelineResult(pairs=pairs_df, summary=summary_df)