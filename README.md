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

## Default strategy: six-month baseline

The default ranks eligible S&P 500 stocks using the existing composite score,
which combines momentum, quality, value, and other factors. The backtest selects
the top 25 and reselects them every six months.

Run the six-year, top-25, six-month baseline simulation:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py
```

The backtest runner defaults to the full-history dataset, matching full SPY
benchmark, composite ranking, and six-month portfolio updates. RSI-40 remains
available with `--strategy rsi_mean_reversion`. Add `--holding-months 1` to run
that RSI strategy monthly, matching the version used in the exploratory
comparison. Pass `--strategy composite` to select the baseline explicitly.

For example, a 28-year six-month baseline run needs:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py --years 28 --output-dir results/sp500_selector_hold_28y_sharadar
```

Compare the six-month baseline with other selection rules and risk checks over the full history:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_adaptive_strategies.py --years 26
```

This research-only run compares composite rankings, 12-to-1 momentum, a market
and stock trend filter, RSI mean reversion thresholds, monthly rank persistence,
15/20/25% trailing drawdown exits, regime-based factor weights and exposure,
and a monthly strategy switch selected by trailing 252-session Sharpe. Signals
use the prior close and simulated trades fill at the next open. It writes the
full and recent two-year comparison, equity series, trade log, and strategy
switch log under `results/adaptive_strategy_research/`. It never submits orders.
Refresh the Sharadar dataset first if you want the report to reflect newer
market data.

Rank the latest point-in-time universe. Run this after refreshing the dataset to
check for new candidates each day:

```powershell
.\.venv\Scripts\python.exe .\scripts\rank_sp500_portfolio.py --top 25
```

Add `--trading212-demo` to the ranking command to annotate instruments and demo
positions. This is read-only and never submits orders.

The ranking command defaults to the composite score; pass
`--strategy rsi_mean_reversion` to filter for stocks with prior-close 14-day RSI
at or below 40. A low RSI indicates recent weakness; it does not mean a rebound
is certain.

## Trading 212 equity demo connection

The local Trading 212 client is locked to `https://demo.trading212.com`. The
published API terms prohibit algorithmic trading; use order submission only
within the written Trading 212 approval you received for this bot. This project
does not provide a live-host client. Trading 212's public API supports Invest
and Stocks ISA equity accounts only; it does not connect to the CFD Practice
account. The order smoke test below is for an equity demo only.

Create or review the API key while the Trading 212 app is in Practice mode.
Read permissions are enough for the connection check and dry run; the one-order
smoke test also requires order permission. Grant only the permissions needed by
the scripts. Store the key pair in the ignored `.env` file as
`212_API_KEY_ID` and `212_API_KEY_SECRET`. The secret is shown only once by
Trading 212. Check the connection without printing account details with:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_trading212_practice.py
```

To validate one stock ticker and preview a small fractional buy without sending
an order:

```powershell
.\.venv\Scripts\python.exe .\scripts\test_trading212_demo_order.py --ticker AAPL_US_EQ --quantity 0.01
```

After checking the preview and confirming the demo API key has order permission,
add `--submit`. The script asks you to type the exact demo buy confirmation,
sends one market order, and does not retry a POST if the result is uncertain.
The order API accepts share quantity rather than cash value, so verify the
estimated notional in Practice mode first. This smoke test checks one API order;
it does not run the full six-month portfolio rebalance.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
