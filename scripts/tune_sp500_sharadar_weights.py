#!/usr/bin/env python3
"""Tune S&P 500 selector weights using only point-in-time Sharadar data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.selector_hold_backtest import _prepare_prices, run_selector_hold_backtest
from models.sp500_selector import (
    DEFAULT_SP500_WEIGHTS,
    FUNDAMENTAL_COLUMNS,
    SP500SelectorConfig,
    rank_sp500_stocks,
)


FEATURE_COLUMNS = tuple(DEFAULT_SP500_WEIGHTS)
PRICE_COLUMNS = (
    "date", "ticker", "open", "close", "adj_close", "dollar_volume",
    "is_sp500", "market_cap_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/sp500_daily_dataset.csv"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--audit-years", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--holding-days", type=int, default=126)
    parser.add_argument("--anchor-days", type=int, default=21,
                        help="Spacing between overlapping training portfolios")
    parser.add_argument("--max-weight", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sp500_sharadar_weight_tuning"))
    parser.add_argument("--backtest-years", type=int, default=6)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/raw/benchmark/sp500_spy_sharadar.csv"))
    return parser.parse_args()


def constrained_samples(count: int, features: int, seed: int, max_weight: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    gathered = 0
    while gathered < count:
        # Alpha below one explores useful sparse factor mixes as well as blended models.
        batch = rng.dirichlet(np.full(features, 0.8), size=max(count, 2_000))
        batch = batch[batch.max(axis=1) <= max_weight]
        accepted.append(batch)
        gathered += len(batch)
    return np.vstack(accepted)[:count]


def build_periods(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    holding_days: int,
    anchor_days: int,
) -> list[dict[str, object]]:
    sessions = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    price_history: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ticker, rows in prices.groupby("ticker", sort=False):
        ordered = rows.sort_values("date")
        price_history[str(ticker)] = (
            ordered["date"].to_numpy(dtype="datetime64[ns]"),
            ordered["execution_open"].to_numpy(dtype=float),
            ordered["mark_close"].to_numpy(dtype=float),
        )

    rank_days = {pd.Timestamp(day): rows for day, rows in ranked.groupby("date", sort=False)}
    periods: list[dict[str, object]] = []
    for offset in range(0, len(sessions) - holding_days, anchor_days):
        start = pd.Timestamp(sessions[offset])
        end = pd.Timestamp(sessions[offset + holding_days])
        day = rank_days.get(start)
        if day is None or day.empty:
            continue
        feature_rows: list[np.ndarray] = []
        forward_returns: list[float] = []
        tickers: list[str] = []
        for row in day.itertuples(index=False):
            ticker = str(row.ticker)
            history = price_history.get(ticker)
            entry = float(row.execution_open)
            if history is None or not np.isfinite(entry) or entry <= 0:
                continue
            dates, opens, marks = history
            target = np.datetime64(end.to_datetime64())
            exit_offset = int(np.searchsorted(dates, target, side="right") - 1)
            if exit_offset < 0 or dates[exit_offset] < np.datetime64(start.to_datetime64()):
                continue
            exit_price = (
                opens[exit_offset] if dates[exit_offset] == target else marks[exit_offset]
            )
            if not np.isfinite(exit_price) or exit_price <= 0:
                continue
            feature_rows.append(np.asarray([getattr(row, name) for name in FEATURE_COLUMNS], dtype=float))
            forward_returns.append(exit_price / entry - 1.0)
            tickers.append(ticker)
        if len(tickers) >= 10:
            periods.append({
                "date": start,
                "end_date": end,
                "tickers": np.asarray(tickers),
                "features": np.nan_to_num(np.vstack(feature_rows), nan=0.5),
                "returns": np.asarray(forward_returns),
            })
    return periods


def evaluate(periods: list[dict[str, object]], weights: np.ndarray, top_n: int) -> np.ndarray:
    output = np.zeros((len(weights), len(periods)), dtype=float)
    for period_index, period in enumerate(periods):
        features = period["features"]
        returns = period["returns"]
        assert isinstance(features, np.ndarray) and isinstance(returns, np.ndarray)
        scores = features @ weights.T
        pick_count = min(top_n, len(returns))
        selected = np.argpartition(scores, -pick_count, axis=0)[-pick_count:, :]
        output[:, period_index] = returns[selected].mean(axis=0)
    return output


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.audit_years < 1 or args.top < 1:
        raise SystemExit("Samples, audit years, and top count must be positive")
    if not args.benchmark.exists():
        raise SystemExit(
            f"Sharadar benchmark not found: {args.benchmark}. "
            "Run scripts/build_sp500_dataset.py --benchmark-only first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = pd.read_csv(args.benchmark, usecols=["date", "close"])

    header = pd.read_csv(args.input, nrows=0).columns
    wanted = list(dict.fromkeys([*PRICE_COLUMNS, *FUNDAMENTAL_COLUMNS]))
    missing = sorted(set(PRICE_COLUMNS).difference(header))
    if missing:
        raise SystemExit(f"Sharadar dataset is missing: {', '.join(missing)}")
    dataset = pd.read_csv(args.input, usecols=[column for column in wanted if column in header], low_memory=False)
    prices = _prepare_prices(dataset)
    ranked = rank_sp500_stocks(prices)
    periods = build_periods(ranked, prices, args.holding_days, args.anchor_days)
    if not periods:
        raise SystemExit("No eligible tuning periods were produced")

    baseline = np.asarray([DEFAULT_SP500_WEIGHTS[name] for name in FEATURE_COLUMNS])
    supplied = np.asarray([
        0.32, 0.26, 0.14, 0.20, 0.03, 0.02, 0.03, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    supplied /= supplied.sum()
    random_weights = constrained_samples(args.samples, len(FEATURE_COLUMNS), args.seed, args.max_weight)
    weights = np.vstack([baseline, supplied, random_weights])
    labels = ["current_sharadar", "user_price_only"] + [f"sample_{index:05d}" for index in range(args.samples)]
    returns = evaluate(periods, weights, args.top)

    last_end = max(period["end_date"] for period in periods)
    audit_start = pd.Timestamp(last_end) - pd.DateOffset(years=args.audit_years)
    development = np.asarray([
        index for index, period in enumerate(periods) if period["end_date"] < audit_start
    ])
    audit = np.asarray([
        index for index, period in enumerate(periods) if period["date"] >= audit_start
    ])
    if len(development) < 12 or len(audit) < 3:
        raise SystemExit("Not enough purged development/audit periods for robust tuning")

    folds = [fold for fold in np.array_split(development, 4) if len(fold)]
    fold_means = np.column_stack([returns[:, fold].mean(axis=1) for fold in folds])
    development_mean = returns[:, development].mean(axis=1)
    # The stability penalty discourages a candidate that wins in only one market regime.
    robust_score = development_mean - 0.50 * fold_means.std(axis=1)
    winner = int(np.argmax(robust_score))

    candidate_rows = []
    for index, label in enumerate(labels):
        row: dict[str, object] = {
            "label": label,
            "development_mean_6m_return_pct": development_mean[index] * 100.0,
            "development_fold_std_pct": fold_means[index].std() * 100.0,
            "robust_score_pct": robust_score[index] * 100.0,
            "audit_mean_6m_return_pct": returns[index, audit].mean() * 100.0,
            "audit_median_6m_return_pct": np.median(returns[index, audit]) * 100.0,
        }
        row.update({name: weights[index, column] for column, name in enumerate(FEATURE_COLUMNS)})
        candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows).sort_values("robust_score_pct", ascending=False)
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)

    rescored = ranked.copy()
    rescored["selection_score"] = 100.0 * (
        rescored.loc[:, FEATURE_COLUMNS].fillna(0.5).to_numpy(dtype=float) @ weights[winner]
    )
    rescored["selection_rank"] = rescored.groupby("date")["selection_score"].rank(
        ascending=False, method="first"
    ).astype("int64")
    backtest = run_selector_hold_backtest(
        dataset,
        constituents=None,
        years=args.backtest_years,
        top_n=args.top,
        holding_days=args.holding_days,
        initial_capital=20_000.0,
        output_dir=args.output_dir / f"production_backtest_{args.backtest_years}y",
        benchmark=benchmark,
        market_name="S&P 500 (Sharadar point-in-time)",
        benchmark_name="S&P 500 (SPY, Sharadar)",
        benchmark_symbol="SPY",
        currency_symbol="$",
        commission_rate=0.0,
        minimum_commission=0.0,
        ranking_function=lambda _prices: rescored,
        data_source="Sharadar SEP + DAILY + ART + historical S&P 500 membership",
        research_limitation=(
            "Weights were selected on a purged development sample and reported on a held-out audit sample. "
            "Historical Sharadar membership, adjusted prices, and filing-date fundamentals are used."
        ),
    )

    summary = {
        "data_source": "Sharadar only (SEP, DAILY, ART and point-in-time membership)",
        "feature_columns": FEATURE_COLUMNS,
        "samples": args.samples,
        "seed": args.seed,
        "period_start": periods[0]["date"].date().isoformat(),
        "period_end": periods[-1]["end_date"].date().isoformat(),
        "audit_start": audit_start.date().isoformat(),
        "development_periods": len(development),
        "audit_periods": len(audit),
        "winner": labels[winner],
        "recommended_weights": {
            name: float(weights[winner, column]) for column, name in enumerate(FEATURE_COLUMNS)
        },
        "winner_development_mean_6m_return_pct": float(development_mean[winner] * 100.0),
        "winner_development_fold_means_pct": [float(value * 100.0) for value in fold_means[winner]],
        "winner_audit_mean_6m_return_pct": float(returns[winner, audit].mean() * 100.0),
        "winner_audit_median_6m_return_pct": float(np.median(returns[winner, audit]) * 100.0),
        "baseline_development_mean_6m_return_pct": float(development_mean[0] * 100.0),
        "baseline_audit_mean_6m_return_pct": float(returns[0, audit].mean() * 100.0),
        "production_backtest": backtest.summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
