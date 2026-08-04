from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.pairs_trading.config import load_config
from src.pairs_trading.pipeline import run_pipeline

GOLDEN = Path(__file__).parent / "golden"
RTOL = 1e-4
ATOL = 1e-2  # rupee-level metrics; tune if needed


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def pipeline_result(config):
    return run_pipeline(config)


def test_pairs_match_golden(pipeline_result):
    expected = pd.read_csv(GOLDEN / "cointegrated_pairs.csv")
    actual = pipeline_result.pairs

    # stable row order
    sort_cols = ["Stock1", "Stock2"]
    expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = actual.sort_values(sort_cols).reset_index(drop=True)

    assert_frame_equal(actual, expected, check_dtype=False, rtol=RTOL, atol=ATOL)


def test_summary_match_golden(pipeline_result):
    expected = pd.read_csv(GOLDEN / "nse_pairs_summary.csv")
    actual = pipeline_result.summary.reset_index(drop=True)
    expected = expected.reset_index(drop=True)

    # monolith sorts by Net_PnL descending — pipeline should too
    assert list(actual["Pair"]) == list(expected["Pair"])

    assert_frame_equal(actual, expected, check_dtype=False, rtol=RTOL, atol=ATOL)