# Sharadar S&P 500 stock selector

This repository uses Sharadar exclusively. The dataset combines:

- `SEP` adjusted and unadjusted prices, including delisted securities;
- point-in-time S&P 500 membership snapshots and changes;
- `DAILY` valuation data such as P/E, P/B, P/S, EV/EBITDA, and market cap; and
- filing-date `ART` fundamentals for profitability, financial strength, cash flow,
  and shareholder yield.

The API key is read from `SHARADAR_API_KEY` in `.env` and is never written to an
output file.

## Build the dataset

```powershell
.\.venv\Scripts\python.exe .\scripts\build_sp500_dataset.py --years 10
```

Download the complete survivorship-safe history from Sharadar's first full
S&P 500 membership snapshot (31 March 1998):

```powershell
.\.venv\Scripts\python.exe .\scripts\build_sp500_dataset.py --full-history
```

For large extracts, download Sharadar's compressed full-table archives instead
of paginating the query API:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_sharadar_bulk.py
```

Build the merged full-history S&P 500 dataset from those local archives:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_sp500_from_bulk.py
```

Generated inputs (ignored by Git):

- `data/processed/sp500_daily_dataset.csv`
- `data/raw/universe/sp500_membership_history.csv`
- `data/raw/universe/sp500_events.csv`
- `data/raw/benchmark/sp500_spy_sharadar.csv` — SPY adjusted close from Sharadar,
  used as the S&P 500 total-return baseline in reports.

Refresh only the benchmark with:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_sp500_dataset.py --years 10 --benchmark-only
```

## Tune factor weights

```powershell
.\.venv\Scripts\python.exe .\scripts\tune_sp500_sharadar_weights.py --samples 10000
```

The tuner uses overlapping six-month portfolios for better sample coverage and
optimizes benchmark-relative compounded returns rather than raw return alone.
It also rewards the lower quartile and outperformance frequency while penalizing
instability across chronological folds. Portfolios crossing the split are purged,
and the final two years are reported as an untouched audit. Both optimized and
baseline eight-year reports are written below the selected output directory.

## Run and rank

Run the six-year, top-25, six-month-hold simulation:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py
```

The backtest runner defaults to the full-history dataset and matching full SPY
benchmark. For example, a 28-year run now only needs:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py --years 28 --output-dir results/sp500_selector_hold_28y_sharadar
```

Rank the latest point-in-time universe:

```powershell
.\.venv\Scripts\python.exe .\scripts\rank_sp500_portfolio.py --top 25
```

Add `--trading212-demo` to the ranking command to annotate instruments and demo
positions. This is read-only and never submits orders.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
