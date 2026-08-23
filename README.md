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

The tuner uses overlapping six-month portfolios for better sample coverage,
selects weights on the older development period, purges portfolios crossing the
split, and reports the final two years as an untouched audit. Outputs are written
to `results/sp500_sharadar_weight_tuning/`.

## Run and rank

Run the six-year, top-25, six-month-hold simulation:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py
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
