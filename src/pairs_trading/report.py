from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pairs_trading.config import PairsTradingConfig
from src.pairs_trading.pipeline import PipelineResult

PAIRS_CSV = "cointegrated_pairs.csv"
SUMMARY_CSV = "nse_pairs_summary.csv"
TRADES_CSV = "nse_pairs_trades_detailed.csv"


def export_results(result: PipelineResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.pairs.to_csv(output_dir / PAIRS_CSV, index=False)
    if not result.summary.empty:
        result.summary.to_csv(output_dir / SUMMARY_CSV, index=False)
    if not result.trades.empty:
        result.trades.to_csv(output_dir / TRADES_CSV, index=False)


def print_results(
    result: PipelineResult,
    config: PairsTradingConfig,
    *,
    nifty: pd.Series | None = None,
) -> None:
    print("\n" + "=" * 60 + "\nFINAL RESULTS\n" + "=" * 60)

    sdf = result.summary
    if sdf.empty:
        print("  ✗ No results.")
        print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)
        return

    tot = sdf["Net_PnL"].sum()
    ret = tot / config.init_capital * 100
    yrs = (pd.Timestamp(config.TEST_END) - pd.Timestamp(config.TEST_START)).days / 365.25
    ps = (
        result.port_pnl.mean() / result.port_pnl.std() * np.sqrt(252)
        if result.port_pnl.std() > 0
        else 0
    )

    print(f"\n{'Pair':<35} {'PnL':>12} {'Ret%':>8} {'Sharpe':>8} {'Trades':>7}")
    print("-" * 72)
    for _, r in sdf.iterrows():
        print(
            f"  {r['Pair']:<33} ₹{r['Net_PnL']:>10,.0f} {r['Return%']:>+7.1f}% "
            f"{r.get('Sharpe', 0):>7.2f} {r['Trades']:>6}"
        )

    print(f"\n{'─' * 60}")
    print(f"  Capital: ₹{config.init_capital:,.0f} → ₹{config.init_capital + tot:,.0f}")
    print(f"  Return: {ret:+.2f}% | Ann: {ret / yrs:+.2f}% | Sharpe: {ps:.3f}")
    print(f"  Pairs: {result.n_pairs} | Win rate: {(sdf['Net_PnL'] > 0).mean() * 100:.0f}%")

    print(f"\n  {'Sector':<18} {'#':>3} {'PnL':>12} {'Ret%':>8}")
    for sec in sorted(sdf["Sector"].unique()):
        sd = sdf[sdf["Sector"] == sec]
        sp, sa = sd["Net_PnL"].sum(), sd["Capital"].sum()
        print(f"  {sec:<18} {len(sd):>3} ₹{sp:>10,.0f} {sp / sa * 100:>+7.1f}%")

    if nifty is not None:
        ns = nifty.loc[config.TEST_START : config.TEST_END].dropna()
        if isinstance(ns, pd.DataFrame):
            ns = ns.iloc[:, 0]
        ns = pd.Series(ns.values, index=ns.index)
        nr_ = ns.pct_change().dropna()
        nret = (float(ns.iloc[-1]) / float(ns.iloc[0]) - 1) * 100
        nsh = nr_.mean() / nr_.std() * np.sqrt(252) if nr_.std() > 0 else 0
        print(f"\n  Nifty50: {nret:+.1f}% | Sharpe: {nsh:.3f}")
        print(f"  Alpha: {ret - nret:+.2f}% | Sharpe edge: {ps - nsh:+.3f}")

    print(f"\n✓ Saved: {PAIRS_CSV}, {SUMMARY_CSV}, {TRADES_CSV}")
    print("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)