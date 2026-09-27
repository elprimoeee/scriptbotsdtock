# Agent context

## Default meaning of “trading strategy”

When the user talks about their trading strategy, strategy results, or improving
the strategy, assume they mean this project's Sharadar S&P 500 point-in-time
selector backtest unless they explicitly name another strategy. Keep work
focused on evaluating or improving this existing setup rather than switching to
a generic or unrelated trading strategy.

The reference workflow uses the full-history Sharadar daily dataset, the
Sharadar SPY benchmark, and `scripts/run_sp500_sharadar_backtest.py`. Its
standard selector settings are the runner defaults: top 25 stocks, six-month
holding periods, and $20,000 initial capital. Preserve the point-in-time S&P 500
membership and Sharadar fundamentals context when discussing results.

Example invocation for this checkout:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sp500_sharadar_backtest.py `
  --years 26 `
  --input .\data\processed\sp500_daily_dataset_full.csv `
  --benchmark .\data\raw\benchmark\sp500_spy_sharadar_full.csv `
  --output-dir .\results\sp500_selector_hold_28y_sharadar
```

The `--years` value is only the requested backtest window and can vary; it does
not identify a different strategy. The output directory name may also use a
different year label. If working in a checkout where the runner or benchmark
paths differ, map these paths to the equivalent Sharadar S&P 500 backtest files
in that checkout.
