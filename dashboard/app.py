import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

def get_output_dir() -> Path:
    # Set when launching: $env:PAIRS_OUTPUT_DIR = "output/dev"
    return Path(os.environ.get("PAIRS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

OUTPUT_DIR = get_output_dir()

summary = pd.read_csv(OUTPUT_DIR / "nse_pairs_summary.csv")
trades = pd.read_csv(OUTPUT_DIR / "nse_pairs_trades_detailed.csv")
pairs = pd.read_csv(OUTPUT_DIR / "cointegrated_pairs.csv")