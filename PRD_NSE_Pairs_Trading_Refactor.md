# PRD — NSE Pairs Trading System: Monolith → Structured Project

**Author:** Rohit | **Doc type:** Product Requirements Document
**Subject:** Refactoring `NSE_PAIRS_TRADING___LightGBM.py` (406-line single-file script) into a properly structured, vibe-codeable Python project

---

## 0. Context

The current script already *works* — it's a full pipeline (sector-based cointegration screening → LightGBM signal classification → walk-forward backtest → risk reporting → Nifty50 benchmark comparison) that has previously produced a validated, positive-alpha backtest. That's the hard part, and it's done.

What it isn't, is *structured*. Config, statistics, ML, simulation, and reporting all live in one file with global state, magic numbers, and no tests. That's fine for a research script you run top-to-bottom in one sitting. It's risky the moment you (or an AI agent in Cursor) start editing it — a one-line "fix" to `simulate()` can silently reintroduce lookahead bias 200 lines away from where you're looking.

**This PRD is for an engineering refactor, not a strategy redesign.** The target behavior is "same results, better structure," with room for extension afterward.

---

## 1. Project Objective

**Primary objective:** Convert the existing single-file backtester into a modular, tested, config-driven Python package that produces *numerically identical* results to the current script, structured well enough to extend, debug, and show off (GitHub/resume/portfolio, and it can double as supporting material for the NETCRYPT-2026 paper writeup).

**Why this matters (beyond "clean code"):**
- **Correctness under change.** Quant code fails silently. A modular structure with tests is what lets you catch a reintroduced bug before it costs you a wrong Sharpe ratio in a paper or a demo.
- **Vibe-coding safety.** Cursor's agent will happily "simplify" your `FIX 1–5` patches back to their buggy originals because it doesn't know their history. Structure + tests are the guardrails that make AI-assisted editing safe on financial logic.
- **Portfolio leverage.** "Built a 400-line script" reads very differently from "designed a modular backtesting package with a tested statistics layer, walk-forward validation, and a CI-friendly regression suite."

**Success criteria (Definition of Done for the whole project):**
1. Running the new pipeline end-to-end reproduces the original script's `nse_pairs_summary.csv` and `cointegrated_pairs.csv` within floating-point tolerance (this is your regression test, see M8).
2. No function is >~60 lines; no module mixes more than one concern (stats ≠ ML ≠ simulation ≠ I/O).
3. All non-trivial magic numbers (thresholds, dates, capital, tickers) live in one config file, not scattered in code.
4. Core statistical functions (`eg_test`, `half_life`, `rs_exponent`, feature/label logic) have unit tests with known synthetic inputs.
5. A single command runs the full pipeline; a single flag runs a fast/dev subset (e.g., 2 sectors) for quick iteration.
6. The existing Streamlit dashboard reads from the new package's outputs without modification to its logic.
7. README exists and a stranger (or a recruiter) can clone, install, and run it in under 10 minutes.

**Explicit non-goals (v1):**
- No new alpha, features, or strategy changes. Resist the urge to "improve" the model while refactoring — that's a separate project, done *after* the regression baseline is locked in.
- No live trading / broker integration.
- No cloud deployment beyond Streamlit Community Cloud (already exists).

---

## 2. Skills Required

You asked for this to map onto procedural, computational, analytical, and logical thinking — here's how each shows up in this specific project, plus the concrete tools.

| Thinking mode | What it looks like in this project |
|---|---|
| **Procedural thinking** | Defining the exact ordered pipeline (download → filter pairs → validate → engineer features → train → simulate → report) as discrete, callable steps instead of a script that runs top to bottom by accident of line order. |
| **Computational thinking** | *Decomposition* — splitting one file into `data`, `stats`, `features`, `model`, `backtest`, `report`. *Abstraction* — defining clean function signatures (inputs/outputs) between stages so each module doesn't need to know how the others work internally. *Pattern recognition* — noticing `eg_test`, `adf_test`, `joh_test` etc. all follow the same "try stat test, fall back to safe default on failure" pattern and can share a helper. |
| **Analytical thinking** | Reading Sharpe/MaxDD/Calmar/hit-rate output and deciding whether a filter (e.g. `RS_THRESH`, `HL_MAX`) is too strict or too loose; deciding whether a module boundary is in the right place. |
| **Logical thinking** | Preserving control-flow correctness during the refactor — e.g., making sure `create_features()`'s target label still only looks forward exactly `TARGET_H` bars (by design, for labeling) while every *feature* column stays strictly backward-looking (no lookahead). This is the single easiest thing to break silently. |

**Concrete tools/skills you'll use (you already have most of the ML/Python side from the original build):**
- Python packaging basics: `src/` layout, `__init__.py`, relative imports, virtual environments (already comfortable per your background).
- `pandas`/`numpy` (already have this), plus reading it *as a reviewer* of AI-generated diffs, not just a writer.
- `pytest` for unit + regression tests (new, but shallow learning curve — you'll only need `assert`, fixtures, and `pytest.approx`).
- Config management: a YAML file + a small dataclass/loader (new but simple).
- `argparse` or `Typer` for a CLI entrypoint (new but simple).
- Git discipline: small commits per milestone, so a bad refactor step is a one-command revert, not a debugging session.
- Prompt discipline for Cursor: writing scoped, single-module prompts and reading diffs before accepting — this is the actual "vibe coding" skill, covered in the companion guide.

---

## 3. Non-Functional Requirements (Guardrails)

These apply across every milestone — treat them as constraints on *how* you vibe code, not just what you build.

1. **Golden-baseline first.** Before touching any code, run the current script once, and save its three output CSVs untouched as your regression baseline. Nothing in this refactor is "done" until diffed against them.
2. **No silent numeric changes.** `random_state=42`, `TXN_COST`, thresholds, window options — none of these change value during the refactor. If a future milestone wants to tune them, that's a separate, explicit change with its own before/after comparison.
3. **One concern per module.** If you're editing `backtest.py` and find yourself writing a cointegration test, stop — it belongs in `stats.py`.
4. **Every AI-generated diff touching `stats.py`, `features.py`, or `backtest.py` gets manually read line-by-line before accepting.** These three files are where lookahead bias and off-by-one bugs hide. Everywhere else, a lighter review pass is fine.
5. **Reproducibility.** Same inputs → same outputs, every run. No unseeded randomness, no wall-clock-dependent logic in the core pipeline.

---

## 4. Target Architecture

```
nse-pairs-trading/
├── .cursor/rules/quant-safety.mdc   # Cursor guardrail rules (see companion guide)
├── .gitignore
├── README.md
├── pyproject.toml                    # deps + project metadata
├── config/
│   └── config.yaml                   # sectors, tickers, dates, thresholds, capital
├── data/cache/                       # gitignored parquet cache of yfinance pulls
├── src/pairs_trading/
│   ├── __init__.py
│   ├── config.py                     # loads config.yaml into typed objects
│   ├── data_loader.py                # yfinance download + local caching + cleaning
│   ├── stats.py                      # eg_test, joh_test, adf_test, half_life, rs_exponent, ols_hedge
│   ├── pair_selection.py             # combos, sector + cross-sector filtering cascade
│   ├── features.py                   # create_features, compute_rsi, labeling logic
│   ├── model.py                      # build_model, train_predict, walk-forward eval
│   ├── backtest.py                   # simulate(), risk_metrics()
│   ├── pipeline.py                   # orchestrates validation pass + train/test loop
│   └── report.py                     # console + CSV/markdown reporting
├── scripts/
│   └── run_backtest.py               # thin CLI entrypoint
├── tests/
│   ├── conftest.py
│   ├── test_stats.py
│   ├── test_features.py
│   └── golden/                       # frozen original-script outputs for regression diffing
│       ├── cointegrated_pairs.csv
│       └── nse_pairs_summary.csv
└── dashboard/
    └── app.py                        # existing Streamlit app, repointed at src/ outputs
```

Every folder maps 1:1 to a numbered section comment already in your original script (`# ── 2. STAT FUNCTIONS`, `# ── 5. FEATURES & HELPERS`, etc.) — you're not inventing new boundaries, you're formalizing the ones the script already implicitly has.

---

## 5. Key Features — Milestones

Each milestone is a self-contained, committable unit. Do not start M(n+1) until M(n)'s acceptance criteria pass.

### M0 — Repo Bootstrap & Golden Baseline
- **Objective:** Set up the project skeleton and freeze a known-good baseline before changing anything.
- **Tasks:** `git init`; create the folder tree from §4 (empty files OK); set up a virtual environment; `pip freeze` the working deps from the original script into `pyproject.toml`/`requirements.txt`; run the *original* script once and copy its 3 output CSVs into `tests/golden/`.
- **Deliverable:** Empty-but-structured repo, committed; golden CSVs committed to `tests/golden/`.
- **Acceptance criteria:** `git log` shows an initial commit; golden CSVs exist and are non-empty; `python -m venv` + install works from a clean clone.

### M1 — Config Extraction
- **Objective:** Pull every hardcoded constant (`SECTORS`, date ranges, `TXN_COST`, `ENTRY_Z`, `WINDOW_OPTS`, `CROSS_LINKS`, `WF_FOLDS`, `FEATURE_COLS`, etc.) out of code and into `config/config.yaml`, loaded via `config.py`.
- **Tasks:** Write `config.yaml`; write a small loader (dataclass or `pydantic` model, your call) in `config.py`; nothing else changes yet.
- **Deliverable:** `config.py` + `config.yaml`.
- **Acceptance criteria:** Loading the config in a Python shell reproduces every constant from the original script, same values, same types.

### M2 — Data Layer with Caching
- **Objective:** Isolate the `yfinance` download + cleaning logic (§3 in the original) into `data_loader.py`, adding a local parquet cache so you're not re-downloading the same 5-10 years of daily data every time you (or Cursor's agent) iterate.
- **Tasks:** Port the download/clean/`ffill`/`bfill`/bad-ticker-drop logic; add "check `data/cache/` first, else download and save" logic keyed by ticker list + date range.
- **Deliverable:** `data_loader.py` with a `load_prices(config) -> pd.DataFrame` function.
- **Acceptance criteria:** First run downloads and caches; second run with identical config is near-instant and returns an identical DataFrame (`pd.testing.assert_frame_equal` against the first run's output).

### M3 — Statistics Module + Unit Tests
- **Objective:** Port `eg_test`, `joh_test`, `adf_test`, `half_life`, `rs_exponent`, `ols_hedge`, `compute_rsi` verbatim into `stats.py`.
- **Tasks:** Copy logic as-is (no "improvements" yet); write `test_stats.py` with synthetic inputs — e.g., two artificially cointegrated random-walk series should pass `eg_test`/`adf_test`; two independent random walks should fail.
- **Deliverable:** `stats.py`, `test_stats.py`.
- **Acceptance criteria:** `pytest tests/test_stats.py` passes; functions called against a slice of real cached price data return the same values as the original script did on that slice (spot-check 2-3 pairs manually).

### M4 — Pair Selection Pipeline
- **Objective:** Port the combo-generation + filtering cascade (§4) into `pair_selection.py`, using `stats.py` functions.
- **Tasks:** Build `select_pairs(price_data, config) -> (pairs, pair_info)`, preserving the exact filter order and thresholds (EG-or-Johansen → ADF on spread → half-life bounds → RS exponent).
- **Deliverable:** `pair_selection.py`.
- **Acceptance criteria:** Output `pair_info` list matches `tests/golden/cointegrated_pairs.csv` row-for-row (same pairs, same EG_P/HalfLife/RS_Exp values within float tolerance).

### M5 — Feature Engineering & Labeling
- **Objective:** Port `create_features()` into `features.py`, with explicit attention to the FIX-1 zscore-reversion labeling logic.
- **Tasks:** Port as-is; add a test that asserts every column in `FEATURE_COLS` at row `t` depends only on data at or before `t` (no forward peeking), while confirming the `target` column intentionally uses `shift(-TARGET_H)` (labels are allowed to look forward; features are not — the test should check both).
- **Deliverable:** `features.py`, extended `test_features.py`.
- **Acceptance criteria:** Lookahead test passes; feature values on a known pair/window match the original script's output.

### M6 — Model Training & Walk-Forward Validation
- **Objective:** Port `build_model`, `train_predict`, and the walk-forward accuracy loop + window optimization loop into `model.py`.
- **Tasks:** Separate "train a model on a slice" (`train_predict`) from "search over `WINDOW_OPTS` using validation PnL" (a new `optimize_window()` function) — these were interleaved in the original loop.
- **Deliverable:** `model.py`.
- **Acceptance criteria:** For a fixed pair, the chosen `best_win` and train accuracy match the original script's printed output.

### M7 — Backtest Engine & Risk Metrics
- **Objective:** Port `simulate()` and `risk_metrics()` into `backtest.py`, preserving FIX-4 (no global state — returns passed explicitly) and FIX-5 (capped ffill on position).
- **Tasks:** Port as-is; add a test with a hand-constructed toy z-score series where you know the expected position sequence, to lock in entry/exit logic.
- **Deliverable:** `backtest.py`.
- **Acceptance criteria:** Toy-case test passes; running `simulate()` on a real pair's test-period features reproduces the original script's `pnl_net`, `cum_pnl` series exactly.

### M8 — Pipeline Orchestration + Golden Regression Test
- **Objective:** Wire M2–M7 together in `pipeline.py`, replicating §6 (validation-pass re-filtering with the OR→AND fallback logic) and §7 (per-pair train/test loop) exactly. This is the milestone where correctness is proven, not assumed.
- **Tasks:** Write `run_pipeline(config) -> (trades_df, summary_df)`; write `tests/test_regression.py` that runs the full pipeline and diffs `summary_df`/`pairs_df` against `tests/golden/*.csv` using `pd.testing.assert_frame_equal(..., rtol=1e-6)` or similar.
- **Deliverable:** `pipeline.py`, `test_regression.py`.
- **Acceptance criteria:** Regression test passes. **This is the milestone that unlocks everything after it** — until this passes, you don't have a validated refactor, you have a rewrite of unknown correctness.

### M9 — CLI, Reporting & Dashboard Integration
- **Objective:** Add a usable entrypoint and reconnect the existing Streamlit dashboard.
- **Tasks:** `scripts/run_backtest.py` with `argparse`/Typer flags (`--fast` for a 2-sector dev subset, `--sectors`, `--output-dir`); port the console/table printing + CSV export from §8 into `report.py`; update `dashboard/app.py` to read from the new package's output location instead of any hardcoded values.
- **Deliverable:** Working CLI; dashboard unchanged in behavior but repointed.
- **Acceptance criteria:** `python scripts/run_backtest.py --fast` runs in under a couple minutes on 2 sectors; full run matches golden baseline; Streamlit dashboard renders correctly against fresh output.

### M10 — Documentation & Polish
- **Objective:** Make the repo presentable to a stranger (recruiter, professor, paper co-reviewer).
- **Tasks:** README with architecture diagram (can be ASCII, from §4), setup instructions, sample output, link to the dashboard; pin dependency versions; add a `LICENSE`; clean up any leftover print-debugging.
- **Deliverable:** Final README, tagged `v1.0` release/commit.
- **Acceptance criteria:** A friend (or Cursor's agent, fresh context) can clone and run it successfully using only the README.

---

## 6. Stretch Goals (explicitly post-v1.0)

Only after M10 and only as separate, individually regression-tested changes:
- Additional features (order-book-style signals, sector-neutral z-scores).
- Model comparison (LightGBM vs. logistic regression baseline vs. XGBoost).
- Parameterizing transaction cost/slippage models.
- CI (GitHub Actions running `pytest` on push).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Cursor's agent "cleans up" a `try/except` that's silently guarding against yfinance flakiness or a degenerate stat-test input, changing behavior on edge cases | Golden regression test (M8) catches value changes; manual diff review on `stats.py`/`features.py`/`backtest.py` per NFR #4 |
| Refactor drifts into "improving" the strategy mid-way, making regression testing meaningless | Non-goals in §1; discipline to open a separate branch/issue for any strategy idea that comes up |
| yfinance rate-limiting or API changes break `data_loader.py` mid-project | Caching (M2) means you only need a successful download once per config; keep the original script's try/except fallback pattern |
| Scope creep turns a 1-2 week refactor into a rewrite | Milestone gating — don't start M(n+1) without M(n)'s acceptance criteria met |
