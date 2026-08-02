# ASX Trading Lab

This repository is organised as a compact trading research workspace for ASX experimentation.

It is built around three goals:

- keep a usable ASX daily dataset up to date
- make it easy to add new metrics without rewriting the whole pipeline
- provide a clean place to build and test stock ideas against the ASX200 or another benchmark

## Main Structure

- `main.py`
  - single hardcoded entrypoint for selecting which model to run
  - toggles the two benchmark test modes with a two-boolean list
- `scripts/`
  - day-to-day data and audit scripts
  - the single dataset-build script plus the dataset integrity audit
- `models/`
  - workspace for new models
  - includes the reference walk-forward metric model
- `metric_modules/`
  - optional custom metrics loaded automatically by the dataset builder
- `docs/`
  - structure notes

See `docs/architecture.md`.

## Core Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Build or refresh the ASX dataset:

```powershell
python scripts/build_asx_dataset.py --verbose
```

Quick sample run:

```powershell
python scripts/build_asx_dataset.py --max-tickers 25 --verbose
```

Run the selected model from the root entrypoint:

```powershell
python main.py
```

Edit `SELECTED_MODEL` and `SELECTED_TESTS` in `main.py` to change which model runs and whether the direct and hedged-short benchmark tests are generated.

When enabled, outputs are saved under:

- `results/direct/`
- `results/hedged/`

Each selected mode folder contains the backtest CSV outputs plus a mode-specific benchmark comparison CSV and SVG chart.

- `results/direct/`
  - stock-only long/short equity strategy results
  - compared against a plain ASX200 buy-and-hold benchmark
- `results/hedged/`
  - stock strategy results with the explicit ASX200 short hedge applied

Generate an autoplay HTML replay from the saved results:

```powershell
python scripts/render_strategy_replay.py --mode direct
```

The replay is saved to `results/<mode>/strategy_replay.html` and shows synced charts for equity, exposures, holdings mix, trades per day, and the live holdings list.

Run the existing data audit:

```powershell
python scripts/data_quality_audit.py
```

## Benchmark Testing Modes

The reusable benchmark-testing helpers live in `models/benchmarking.py`.

- Direct benchmark test
  - compare the model return stream directly with the ASX200 or another base
- Hedged short benchmark test
  - pair the stock idea with an equal-notional short benchmark leg to test pure outperformance

## Trading Cost Assumptions

The reference backtest now uses explicit IG-style ASX CFD assumptions instead of placeholder retail proxies:

- ASX share CFD leverage cap: `5:1`
- Margin requirement: `20%` of gross notional exposure
- Share CFD commission: `0.09%` per trade with a `A$7` minimum ticket charge
- Overnight funding: `4.0%` annualized on gross CFD notional, charged on a `360` day basis
- Friday funding: `3x` daily funding to cover the weekend
- Short stock borrow admin charge: `0.5%` annualized on short notional
- Benchmark hedge leg: modeled as an index CFD with funding but no per-trade commission
- Extra spread markup/slippage: `0` by default until you add broker-specific fill data

Backtest outputs now also include daily margin requirement and free-equity fields so you can see how much capital the strategy is tying up under CFD rules.

Shared non-model settings now live in `project_config.py`, including fees, benchmark symbols, file paths, output names, and portfolio sizing defaults.

## Reference Code

The existing reference strategy now lives in the models layer:

- `scripts/build_asx_dataset.py`
- `models/metric.py`
- `scripts/data_quality_audit.py`

The training-data build is now a single monolith script: `scripts/build_asx_dataset.py`.

## Notes

- By default, rows tagged by ASX as `Not Applic` or `Class Pend` are excluded.
- Current implemented metric coverage is listed in `docs/metric_coverage.md`.
- Price-derived metrics are point-in-time safe for the current backtest after signal lagging.
- Yahoo metadata fields such as shares outstanding, valuation ratios, forward EPS, and dividend snapshot fields are still snapshots, not true historical fundamentals.
