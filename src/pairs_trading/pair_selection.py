from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.pairs_trading.config import PairsTradingConfig
from src.pairs_trading.stats import (
    adf_test,
    eg_test,
    half_life,
    joh_test,
    ols_hedge,
    rs_exponent,
)

ADF_P = 0.10
MIN_TRAIN_ROWS = 200


def _build_combos(
    sectors: dict[str, list[str]],
    cross_links: list[tuple[str, str]],
    valid: set[str],
) -> list[tuple[str, str, str]]:
    combos: list[tuple[str, str, str]] = []

    for sec, tks in sectors.items():
        sv = [t for t in tks if t in valid]
        combos += [(a, b, sec) for a, b in combinations(sv, 2)]

    for s1, s2 in cross_links:
        t1 = [t for t in sectors.get(s1, []) if t in valid]
        t2 = [t for t in sectors.get(s2, []) if t in valid]
        combos += [(a, b, f"{s1}×{s2}") for a in t1 for b in t2 if a != b]

    return combos


def select_pairs(
    price_data: pd.DataFrame,
    config: PairsTradingConfig,
) -> tuple[list[tuple[str, str]], list[dict]]:
    valid = set(price_data.columns)
    combos = _build_combos(config.SECTORS, config.CROSS_LINKS, valid)

    pairs: list[tuple[str, str]] = []
    pair_info: list[dict] = []

    for s1, s2, sec in combos:
        sub = price_data.loc[config.TRAIN_START : config.TRAIN_END, [s1, s2]].dropna()
        if len(sub) < MIN_TRAIN_ROWS:
            continue

        y, x = sub[s1], sub[s2]
        eg_p = eg_test(y, x)
        eg_ok = eg_p < config.coint_p
        joh_ok = joh_test(y, x)
        if not eg_ok and not joh_ok:
            continue

        b, a = ols_hedge(y, x)
        sp = y - b * x - a
        if adf_test(sp) >= ADF_P:
            continue

        hl = half_life(sp)
        if not (config.hl_min <= hl <= config.hl_max):
            continue

        rs = rs_exponent(sp)
        if rs >= config.rs_thresh:
            continue

        pairs.append((s1, s2))
        pair_info.append(
            {
                "Stock1": s1,
                "Stock2": s2,
                "Sector": sec,
                "EG_P": round(eg_p, 5),
                "Johansen": joh_ok,
                "HalfLife": round(hl, 1),
                "RS_Exp": round(rs, 4),
                "Beta": round(b, 4),
            }
        )

    return pairs, pair_info