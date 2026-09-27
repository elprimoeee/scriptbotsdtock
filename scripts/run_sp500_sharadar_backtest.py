#!/usr/bin/env python3
"""Run the six-month composite selector on Sharadar's point-in-time S&P 500 universe."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.selector_hold_backtest import TRADING_DAYS_PER_MONTH, run_selector_hold_backtest
from models.sp500_selector import (
    DEFAULT_MEAN_REVERSION_RSI_THRESHOLD,
    FUNDAMENTAL_COLUMNS,
    SP500SelectorConfig,
    rank_sp500_stocks,
)


class TerminalProgress:
    """Small dependency-free progress bar for long command-line runs."""

    def __init__(self, *, enabled: bool = True, width: int = 32) -> None:
        self.enabled = enabled
        self.width = width
        self._label: str | None = None
        self._finished_label: str | None = None
        self._last_percent = -1
        self._last_rendered = 0.0

    def update(self, label: str, completed: int, total: int) -> None:
        if not self.enabled:
            return
        total = max(int(total), 1)
        completed = min(max(int(completed), 0), total)
        percent = int(completed * 100 / total)
        now = time.monotonic()
        label_changed = label != self._label
        finished = completed >= total
        if finished and label == self._finished_label:
            return
        if (
            not label_changed
            and not finished
            and now - self._last_rendered < 0.1
        ):
            return
        if label_changed and self._label is not None:
            sys.stderr.write("\n")
        filled = int(self.width * completed / total)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stderr.write(f"\r[{bar}] {percent:3d}% {label}")
        sys.stderr.flush()
        self._label = label
        self._last_percent = percent
        self._last_rendered = now
        if finished:
            sys.stderr.write("\n")
            self._finished_label = label
            self._label = None
        elif label != self._finished_label:
            self._finished_label = None


class ProgressFile(io.RawIOBase):
    """Binary reader that reports CSV bytes consumed by pandas."""

    def __init__(self, path: Path, progress: TerminalProgress) -> None:
        super().__init__()
        self._file = path.open("rb", buffering=0)
        self._total = path.stat().st_size
        self._progress = progress
        self._progress.update("Loading dataset", 0, self._total)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        size = self._file.readinto(buffer)
        self._progress.update("Loading dataset", self._file.tell(), self._total)
        return size

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


def read_dataset(path: Path, progress: TerminalProgress) -> pd.DataFrame:
    """Read only columns used by the Sharadar selector and backtest."""

    columns = pd.read_csv(path, nrows=0).columns
    required = {"date", "ticker", "open", "close", "dollar_volume", "is_sp500"}
    missing = sorted(required.difference(columns))
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {', '.join(missing)}")
    wanted = required | {
        "adj_close",
        "market_cap",
        "market_cap_rank",
        *FUNDAMENTAL_COLUMNS,
    }
    usecols = [column for column in columns if column in wanted]
    raw = ProgressFile(path, progress)
    with io.BufferedReader(raw, buffer_size=2 * 1024 * 1024) as source:
        dataset = pd.read_csv(source, usecols=usecols, low_memory=False)
    progress.update("Loading dataset", path.stat().st_size, path.stat().st_size)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/sp500_daily_dataset_full.csv"),
    )
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--holding-months", type=int, default=6)
    parser.add_argument(
        "--strategy",
        choices=("rsi_mean_reversion", "composite"),
        default="composite",
        help="Default ranks eligible stocks with the composite factor score",
    )
    parser.add_argument("--initial-capital", type=float, default=20_000.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sp500_selector_hold_6y_sharadar"))
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/raw/benchmark/sp500_spy_sharadar_full.csv"))
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars",
    )
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
    progress = TerminalProgress(enabled=not args.no_progress)
    dataset = read_dataset(args.input, progress)
    benchmark = pd.read_csv(args.benchmark, usecols=["date", "close"])
    threshold = (
        DEFAULT_MEAN_REVERSION_RSI_THRESHOLD
        if args.strategy == "rsi_mean_reversion"
        else None
    )
    selector_config = SP500SelectorConfig(mean_reversion_rsi_threshold=threshold)
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
        ranking_function=lambda prices, *, ranking_dates=None: rank_sp500_stocks(
            prices, selector_config, ranking_dates=ranking_dates
        ),
        rank_rebalance_dates_only=True,
        progress=progress.update,
        data_source="Sharadar SEP + DAILY + ART + historical S&P 500 membership",
        research_limitation=(
            "Historical Sharadar membership, active/delisted prices, and filing-date ART fundamentals are used. "
            + (
                "The RSI-40 strategy was selected after exploratory comparisons; the recent comparison period was inspected and is not an untouched holdout. "
                if threshold is not None else
                "Composite factor weights were tuned on a purged development sample and checked on a held-out two-year audit. "
            )
            + "Signals use the prior close and simulated trades fill at the next open. Final-session exits approximate delisting/acquisition proceeds; taxes, FX, borrow costs, and market impact are excluded."
        ),
    )
    print("Backtest complete")
    for key in ("final_equity", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe_ratio", "forced_exit_orders"):
        print(f"  {key}: {result.summary[key]}")
    print(f"Report: {result.report_path.resolve()}")


if __name__ == "__main__":
    main()
