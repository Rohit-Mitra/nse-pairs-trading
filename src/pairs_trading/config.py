from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class DateRange:
    start: str
    end: str

    def __post_init__(self) -> None:
        _parse_date(self.start, "start")
        _parse_date(self.end, "end")
        if self.start > self.end:
            raise ValueError(f"DateRange start ({self.start}) must be <= end ({self.end})")

    @property
    def start_ts(self) -> date:
        return _parse_date(self.start, "start")

    @property
    def end_ts(self) -> date:
        return _parse_date(self.end, "end")


@dataclass(frozen=True)
class WFFold:
    train_start: str
    train_end: str
    val_start: str
    val_end: str

    def __post_init__(self) -> None:
        train = DateRange(self.train_start, self.train_end)
        val = DateRange(self.val_start, self.val_end)
        if train.end > val.start:
            raise ValueError(
                f"WF fold train end ({train.end}) must be <= val start ({val.start})"
            )

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.train_start, self.train_end, self.val_start, self.val_end)


@dataclass(frozen=True)
class PairsTradingConfig:
    sectors: dict[str, list[str]]
    dates: dict[str, DateRange]
    txn_cost: float
    coint_p: float
    entry_z: float
    exit_z: float
    ffill_limit: int
    window_opts: list[int]
    init_capital: int
    target_h: int
    target_zt: float
    hl_min: int
    hl_max: int
    rs_thresh: float
    cross_links: list[tuple[str, str]]
    wf_folds: list[WFFold]
    feature_cols: list[str]

    # --- compatibility aliases matching the monolith names ---
    @property
    def TRAIN_START(self) -> str:
        return self.dates["train"].start

    @property
    def TRAIN_END(self) -> str:
        return self.dates["train"].end

    @property
    def VAL_START(self) -> str:
        return self.dates["val"].start

    @property
    def VAL_END(self) -> str:
        return self.dates["val"].end

    @property
    def TEST_START(self) -> str:
        return self.dates["test"].start

    @property
    def TEST_END(self) -> str:
        return self.dates["test"].end

    @property
    def SECTORS(self) -> dict[str, list[str]]:
        return self.sectors

    @property
    def ALL_TICKERS(self) -> list[str]:
        return [t for tickers in self.sectors.values() for t in tickers]

    @property
    def WF_FOLDS(self) -> list[tuple[str, str, str, str]]:
        return [fold.as_tuple() for fold in self.wf_folds]

    @property
    def FEATURE_COLS(self) -> list[str]:
        return self.feature_cols

    @property
    def CROSS_LINKS(self) -> list[tuple[str, str]]:
        return self.cross_links

    def __post_init__(self) -> None:
        _validate_config(self)


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date for {label}: {value!r}") from exc


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value


def _require_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing or empty list: {key}")
    return value


def _parse_dates(raw: dict[str, Any]) -> dict[str, DateRange]:
    dates_raw = _require_mapping(raw, "dates")
    for split in ("train", "val", "test"):
        if split not in dates_raw:
            raise ValueError(f"Missing dates.{split}")
        split_raw = dates_raw[split]
        if not isinstance(split_raw, dict):
            raise ValueError(f"dates.{split} must be a mapping")
    train = DateRange(dates_raw["train"]["start"], dates_raw["train"]["end"])
    val = DateRange(dates_raw["val"]["start"], dates_raw["val"]["end"])
    test = DateRange(dates_raw["test"]["start"], dates_raw["test"]["end"])
    if train.end > val.start:
        raise ValueError("train.end must be <= val.start")
    if val.end > test.start:
        raise ValueError("val.end must be <= test.start")
    return {"train": train, "val": val, "test": test}


def _parse_sectors(raw: dict[str, Any]) -> dict[str, list[str]]:
    sectors_raw = _require_mapping(raw, "sectors")
    sectors: dict[str, list[str]] = {}
    for name, tickers in sectors_raw.items():
        if not isinstance(tickers, list) or not tickers:
            raise ValueError(f"sectors.{name} must be a non-empty list")
        if not all(isinstance(t, str) and t for t in tickers):
            raise ValueError(f"sectors.{name} contains invalid ticker(s)")
        sectors[name] = tickers
    return sectors


def _parse_cross_links(raw: dict[str, Any], sectors: dict[str, list[str]]) -> list[tuple[str, str]]:
    links_raw = _require_list(raw, "cross_links")
    links: list[tuple[str, str]] = []
    for item in links_raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid cross_links entry: {item!r}")
        s1, s2 = item
        if not isinstance(s1, str) or not isinstance(s2, str):
            raise ValueError(f"Invalid cross_links entry: {item!r}")
        if s1 not in sectors or s2 not in sectors:
            raise ValueError(f"cross_links sector not in sectors: ({s1}, {s2})")
        links.append((s1, s2))
    return links


def _parse_wf_folds(raw: dict[str, Any]) -> list[WFFold]:
    folds_raw = _require_list(raw, "wf_folds")
    folds: list[WFFold] = []
    for item in folds_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid wf_folds entry: {item!r}")
        try:
            folds.append(
                WFFold(
                    train_start=item["train_start"],
                    train_end=item["train_end"],
                    val_start=item["val_start"],
                    val_end=item["val_end"],
                )
            )
        except KeyError as exc:
            raise ValueError(f"wf_folds entry missing key: {exc}") from exc
    return folds


def _validate_config(cfg: PairsTradingConfig) -> None:
    if not (0 < cfg.coint_p <= 1):
        raise ValueError(f"coint_p must be in (0, 1], got {cfg.coint_p}")
    if cfg.txn_cost < 0:
        raise ValueError(f"txn_cost must be >= 0, got {cfg.txn_cost}")
    if cfg.entry_z <= cfg.exit_z:
        raise ValueError(f"entry_z ({cfg.entry_z}) must be > exit_z ({cfg.exit_z})")
    if cfg.ffill_limit <= 0:
        raise ValueError(f"ffill_limit must be > 0, got {cfg.ffill_limit}")
    if not cfg.window_opts or any(w <= 0 for w in cfg.window_opts):
        raise ValueError(f"window_opts must be positive integers, got {cfg.window_opts}")
    if cfg.init_capital <= 0:
        raise ValueError(f"init_capital must be > 0, got {cfg.init_capital}")
    if cfg.target_h <= 0:
        raise ValueError(f"target_h must be > 0, got {cfg.target_h}")
    if cfg.target_zt <= 0:
        raise ValueError(f"target_zt must be > 0, got {cfg.target_zt}")
    if cfg.hl_min <= 0 or cfg.hl_max <= 0 or cfg.hl_min >= cfg.hl_max:
        raise ValueError(f"Invalid half-life bounds: hl_min={cfg.hl_min}, hl_max={cfg.hl_max}")
    if not (0 < cfg.rs_thresh < 1):
        raise ValueError(f"rs_thresh must be in (0, 1), got {cfg.rs_thresh}")
    if not cfg.feature_cols:
        raise ValueError("feature_cols must be non-empty")


def load_config(path: Path | str | None = None) -> PairsTradingConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("config.yaml root must be a mapping")

    sectors = _parse_sectors(raw)
    dates = _parse_dates(raw)

    window_opts = _require_list(raw, "window_opts")
    if not all(isinstance(w, int) and w > 0 for w in window_opts):
        raise ValueError(f"window_opts must be positive integers, got {window_opts}")

    feature_cols = _require_list(raw, "feature_cols")
    if not all(isinstance(c, str) and c for c in feature_cols):
        raise ValueError(f"feature_cols contains invalid entries: {feature_cols}")

    return PairsTradingConfig(
        sectors=sectors,
        dates=dates,
        txn_cost=float(raw["txn_cost"]),
        coint_p=float(raw["coint_p"]),
        entry_z=float(raw["entry_z"]),
        exit_z=float(raw["exit_z"]),
        ffill_limit=int(raw["ffill_limit"]),
        window_opts=window_opts,
        init_capital=int(raw["init_capital"]),
        target_h=int(raw["target_h"]),
        target_zt=float(raw["target_zt"]),
        hl_min=int(raw["hl_min"]),
        hl_max=int(raw["hl_max"]),
        rs_thresh=float(raw["rs_thresh"]),
        cross_links=_parse_cross_links(raw, sectors),
        wf_folds=_parse_wf_folds(raw),
        feature_cols=feature_cols,
    )