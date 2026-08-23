#!/usr/bin/env python3
"""Run the six-month selector on Sharadar's point-in-time S&P 500 universe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.selector_hold_backtest import TRADING_DAYS_PER_MONTH, run_selector_hold_backtest
from models.sp500_selector import rank_sp500_stocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/sp500_daily_dataset.csv"))
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--holding-months", type=int, default=6)
    parser.add_argument("--initial-capital", type=float, default=20_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sp500_selector_hold_6y_sharadar"))
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/raw/benchmark/sp500_spy_sharadar.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.years < 1 or args.top < 1 or args.holding_months < 1:
        raise SystemExit("Years, top count, and holding months must be positive")
    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}")
    if not args.benchmark.exists():
        raise SystemExit(
            f"Sharadar benchmark not found: {args.benchmark}. "
            "Run scripts/build_sp500_dataset.py --benchmark-only first."
        )
    dataset = pd.read_csv(args.input, low_memory=False)
    benchmark = pd.read_csv(args.benchmark, usecols=["date", "close"])
    result = run_selector_hold_backtest(
        dataset,
        constituents=None,
        years=args.years,
        top_n=args.top,
        holding_days=args.holding_months * TRADING_DAYS_PER_MONTH,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
        benchmark=benchmark,
        market_name="S&P 500 (Sharadar point-in-time)",
        benchmark_name="S&P 500 (SPY, Sharadar)",
        benchmark_symbol="SPY",
        currency_symbol="$",
        commission_rate=0.0,
        minimum_commission=0.0,
        ranking_function=rank_sp500_stocks,
        data_source="Sharadar SEP + DAILY + ART + historical S&P 500 membership",
        research_limitation=(
            "Historical Sharadar membership, active/delisted prices, and filing-date ART fundamentals are used. "
            "Weights were tuned on a purged development sample and checked on a held-out two-year audit. "
            "Final-session exits approximate delisting/acquisition proceeds; taxes, FX, borrow costs, and market impact are excluded."
        ),
    )
    print("Backtest complete")
    for key in ("final_equity", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "forced_exit_orders"):
        print(f"  {key}: {result.summary[key]}")
    print(f"Report: {result.report_path.resolve()}")


if __name__ == "__main__":
    main()
