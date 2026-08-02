#!/usr/bin/env python3
"""
Audit the processed ASX dataset and strategy outputs for data integrity and leakage risk.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from build_asx_dataset import compute_price_metrics, format_history_df, ticker_storage_stem


PRICE_METRIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dollar_volume",
    "avg_volume_30d",
    "avg_volume_90d",
    "daily_return",
    "weekly_return",
    "monthly_return",
    "quarterly_return",
    "yearly_return",
    "cumulative_return",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m_ex_1m",
    "price_vs_50d_ma",
    "price_vs_200d_ma",
    "daily_volatility",
    "volatility_30d",
    "volatility_90d",
    "atr_14d",
    "downside_volatility_90d",
    "max_drawdown",
]

SNAPSHOT_META_COLS = [
    "shares_outstanding",
    "float_shares",
    "enterprise_value",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "price_to_sales",
    "ev_to_ebitda",
    "eps_trailing",
    "eps_forward",
    "dividend_rate",
    "dividend_yield",
    "payout_ratio",
    "ex_dividend_date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit processed ASX data and backtest outputs.")
    parser.add_argument(
        "--dataset-csv",
        default="data/processed/asx_daily_dataset.csv",
        help="Processed dataset CSV to audit.",
    )
    parser.add_argument(
        "--price-dir",
        default="data/raw/prices",
        help="Directory containing raw price CSV files.",
    )
    parser.add_argument(
        "--strategy-dir",
        default="data/processed/strategy",
        help="Directory containing strategy output CSV files.",
    )
    parser.add_argument(
        "--sample-tickers",
        type=int,
        default=25,
        help="How many tickers to recompute from raw prices for validation.",
    )
    parser.add_argument(
        "--abs-tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for recomputed metric checks.",
    )
    return parser.parse_args()


def add_issue(issue_map, severity: str, message: str) -> None:
    issue_map[severity].append(message)


def choose_sample_tickers(dataset_df, sample_size: int) -> list[str]:
    counts = dataset_df.groupby("ticker").size().sort_values(ascending=False)
    return counts.head(max(1, int(sample_size))).index.tolist()


def load_price_history(pd, price_path: Path, ticker: str):
    raw_df = pd.read_csv(price_path)
    if {"date", "open", "high", "low", "close", "adj_close", "volume"}.issubset(raw_df.columns):
        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date
        for col in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        raw_df["ticker"] = ticker
        raw_df = raw_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        return raw_df.reset_index(drop=True)
    return format_history_df(pd, raw_df, ticker)


def compare_numeric_series(np, left, right, tolerance: float) -> tuple[bool, float, int]:
    left_num = left.astype("float64")
    right_num = right.astype("float64")
    left_arr = left_num.to_numpy()
    right_arr = right_num.to_numpy()
    nan_mask = np.isnan(left_arr) & np.isnan(right_arr)
    diff = np.abs(left_arr - right_arr)
    diff[nan_mask] = 0.0
    diff[np.isnan(diff)] = float("inf")
    failures = int((diff > tolerance).sum())
    max_diff = float(diff.max()) if len(diff) else 0.0
    return failures == 0, max_diff, failures


def audit_dataset(np, pd, dataset_csv: Path, issues) -> pd.DataFrame:
    if not dataset_csv.exists():
        raise SystemExit(f"Dataset file not found: {dataset_csv}")

    dataset = pd.read_csv(dataset_csv, parse_dates=["date"], low_memory=False)
    if dataset.empty:
        add_issue(issues, "fail", f"Dataset is empty: {dataset_csv}")
        return dataset

    if int(dataset.duplicated(["date", "ticker"]).sum()) > 0:
        add_issue(issues, "fail", "Dataset has duplicate `date,ticker` rows.")
    if dataset["date"].max().date() > date.today():
        add_issue(issues, "fail", "Dataset contains future-dated rows.")

    null_price_rows = dataset[["open", "high", "low", "close", "adj_close"]].isna().any(axis=1).sum()
    if int(null_price_rows) > 0:
        add_issue(issues, "fail", f"Dataset has {int(null_price_rows)} rows with null OHLC/adj_close values.")

    by_ticker = dataset.sort_values(["ticker", "date"])
    non_monotonic = by_ticker.groupby("ticker")["date"].apply(lambda s: not s.is_monotonic_increasing).sum()
    if int(non_monotonic) > 0:
        add_issue(issues, "fail", f"{int(non_monotonic)} tickers have non-monotonic date order.")

    for col in SNAPSHOT_META_COLS:
        if col not in dataset.columns:
            continue
        varying = dataset.groupby("ticker")[col].nunique(dropna=True)
        has_values = dataset.groupby("ticker")[col].apply(lambda s: s.notna().any())
        suspicious = int(((varying <= 1) & has_values).sum())
        if suspicious > 0:
            add_issue(
                issues,
                "warn",
                f"{col} is snapshot-like for {suspicious} tickers and should be treated as non-point-in-time history.",
            )

    return dataset


def audit_recomputed_metrics(np, pd, dataset_df, price_dir: Path, sample_tickers: int, tolerance: float, issues) -> None:
    tickers = choose_sample_tickers(dataset_df, sample_tickers)
    checked = 0
    for ticker in tickers:
        price_path = price_dir / f"{ticker_storage_stem(ticker)}.csv"
        if not price_path.exists():
            add_issue(issues, "fail", f"Missing raw price file for sampled ticker {ticker}.")
            continue

        formatted = load_price_history(pd, price_path, ticker)
        recomputed = compute_price_metrics(pd, formatted)
        processed = dataset_df.loc[dataset_df["ticker"] == ticker, ["date", "ticker"] + PRICE_METRIC_COLS].copy()
        processed["date"] = pd.to_datetime(processed["date"], errors="coerce").dt.date
        processed["ticker"] = processed["ticker"].astype(str).str.upper()
        recomputed["ticker"] = recomputed["ticker"].astype(str).str.upper()
        merged = processed.merge(
            recomputed[["date", "ticker"] + PRICE_METRIC_COLS],
            on=["date", "ticker"],
            suffixes=("_processed", "_recomputed"),
            how="inner",
        )
        if merged.empty:
            add_issue(issues, "fail", f"No overlapping processed/raw rows for sampled ticker {ticker}.")
            continue

        for col in PRICE_METRIC_COLS:
            ok, max_diff, failures = compare_numeric_series(
                np=np,
                left=merged[f"{col}_processed"],
                right=merged[f"{col}_recomputed"],
                tolerance=tolerance,
            )
            if not ok:
                add_issue(
                    issues,
                    "fail",
                    f"Recomputed metric mismatch for {ticker} column `{col}`: {failures} rows exceed tolerance, max diff={max_diff:.6g}.",
                )
                break
        checked += 1

    if checked == 0:
        add_issue(issues, "fail", "No sampled tickers were validated against raw price files.")


def audit_strategy_outputs(np, pd, dataset_df, strategy_dir: Path, issues) -> None:
    picks_path = strategy_dir / "strategy_daily_top20.csv"
    simulation_path = strategy_dir / "strategy_backtest_simulation.csv"

    if not picks_path.exists() or not simulation_path.exists():
        add_issue(issues, "warn", "Strategy output files are missing, so chronology checks were skipped.")
        return

    picks = pd.read_csv(
        picks_path,
        parse_dates=["date", "train_start_date", "train_end_date"],
        low_memory=False,
    )
    simulation = pd.read_csv(simulation_path, parse_dates=["date"], low_memory=False)

    if int(picks.duplicated(["date", "ticker"]).sum()) > 0:
        add_issue(issues, "fail", "strategy_daily_top20.csv has duplicate `date,ticker` rows.")
    if int((picks["train_end_date"] >= picks["date"]).sum()) > 0:
        add_issue(issues, "fail", "Strategy picks include rows where `train_end_date >= date`.")
    if int(simulation.duplicated(["date"]).sum()) > 0:
        add_issue(issues, "fail", "strategy_backtest_simulation.csv has duplicate dates.")

    signal_price_column = None
    if "signal_close" in picks.columns:
        signal_price_column = "signal_close"
    elif "signal_adj_close" in picks.columns:
        signal_price_column = "signal_adj_close"

    if signal_price_column is not None:
        if int((picks[signal_price_column] < 0.20).sum()) > 0:
            add_issue(issues, "warn", "Some stored picks have lagged signal prices below the default tradability threshold.")
    else:
        add_issue(issues, "warn", "strategy_daily_top20.csv does not include lagged tradability fields from the backtest.")

    returns = dataset_df[["date", "ticker", "close"]].copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
    returns = returns.sort_values(["ticker", "date"]).reset_index(drop=True)
    returns["dataset_next_return"] = (
        returns.groupby("ticker", sort=False)["close"].shift(-1) / returns["close"]
    ) - 1.0
    joined = picks.merge(
        returns[["date", "ticker", "dataset_next_return"]],
        on=["date", "ticker"],
        how="left",
    )
    if "next_day_return" in joined.columns:
        ok, max_diff, failures = compare_numeric_series(
            np=np,
            left=joined["next_day_return"],
            right=joined["dataset_next_return"],
            tolerance=1e-10,
        )
        if not ok:
            add_issue(
                issues,
                "fail",
                f"Strategy pick next-day returns do not match dataset returns: {failures} rows, max diff={max_diff:.6g}.",
            )


def print_summary(dataset_df, issues) -> None:
    print("Dataset audit summary")
    if not dataset_df.empty:
        print(f"rows: {len(dataset_df):,}")
        print(f"tickers: {dataset_df['ticker'].nunique():,}")
        print(f"date range: {dataset_df['date'].min().date()} -> {dataset_df['date'].max().date()}")

    if issues["pass"]:
        for message in issues["pass"]:
            print(f"PASS: {message}")
    for severity in ["warn", "fail"]:
        for message in issues[severity]:
            print(f"{severity.upper()}: {message}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_csv = (root / args.dataset_csv).resolve()
    price_dir = (root / args.price_dir).resolve()
    strategy_dir = (root / args.strategy_dir).resolve()

    issues = defaultdict(list)
    dataset_df = audit_dataset(np=np, pd=pd, dataset_csv=dataset_csv, issues=issues)
    if not dataset_df.empty:
        audit_recomputed_metrics(
            np=np,
            pd=pd,
            dataset_df=dataset_df,
            price_dir=price_dir,
            sample_tickers=args.sample_tickers,
            tolerance=float(args.abs_tolerance),
            issues=issues,
        )
        audit_strategy_outputs(
            np=np,
            pd=pd,
            dataset_df=dataset_df,
            strategy_dir=strategy_dir,
            issues=issues,
        )

    if not issues["warn"] and not issues["fail"]:
        issues["pass"].append("No integrity or leakage issues were detected by the automated audit.")

    print_summary(dataset_df=dataset_df, issues=issues)
    if issues["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
