#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairs_trading.config import PairsTradingConfig, load_config
from src.pairs_trading.data_loader import load_nifty
from src.pairs_trading.pipeline import run_pipeline
from src.pairs_trading.report import export_results, print_results


def _subset_config(
    config: PairsTradingConfig,
    *,
    fast: bool,
    sectors: list[str] | None,
) -> PairsTradingConfig:
    if sectors:
        chosen = sectors
    elif fast:
        chosen = list(config.sectors.keys())[:2]  # first 2 YAML sectors
    else:
        return config

    unknown = set(chosen) - set(config.sectors)
    if unknown:
        raise SystemExit(f"Unknown sector(s): {sorted(unknown)}")

    new_sectors = {k: config.sectors[k] for k in chosen}
    new_links = [
        (a, b) for a, b in config.cross_links if a in chosen and b in chosen
    ]
    return replace(config, sectors=new_sectors, cross_links=new_links)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NSE pairs trading backtest")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Limit to first 2 sectors for quick dev iteration",
    )
    p.add_argument(
        "--sectors",
        nargs="+",
        metavar="NAME",
        help="Run only these sectors (overrides --fast if both given)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory for CSV output (default: output/)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = _subset_config(config, fast=args.fast, sectors=args.sectors)

    result = run_pipeline(config)
    nifty = load_nifty(config)

    export_results(result, args.output_dir)
    print_results(result, config, nifty=nifty)


if __name__ == "__main__":
    main()