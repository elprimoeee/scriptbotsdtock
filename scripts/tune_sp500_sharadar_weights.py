#!/usr/bin/env python3
"""Tune S&P 500 selector weights against SPY using point-in-time Sharadar data.

The final audit window is never used to choose weights. The development
objective rewards benchmark-relative compound return, frequent outperformance,
and stability across chronological folds.
"""

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

from models.selector_hold_backtest import (
    _prepare_prices,
    build_rebalance_dates,
    run_selector_hold_backtest,
)
from models.sp500_selector import (
    DEFAULT_SP500_WEIGHTS,
    FUNDAMENTAL_COLUMNS,
    rank_sp500_stocks,
)


FEATURE_COLUMNS = tuple(DEFAULT_SP500_WEIGHTS)
PRICE_COLUMNS = (
    "date", "ticker", "open", "close", "adj_close", "dollar_volume",
    "is_sp500", "market_cap", "market_cap_rank",
)
LEGACY_WEIGHTS = {
    "momentum_12m_ex_1m_rank": 0.027233,
    "momentum_6m_rank": 0.004208,
    "momentum_3m_rank": 0.155180,
    "trend_quality_rank": 0.040071,
    "downside_risk_rank": 0.048900,
    "drawdown_rank": 0.005191,
    "liquidity_rank": 0.181538,
    "size_rank": 0.026393,
    "quality_rank": 0.020410,
    "value_rank": 0.063227,
    "financial_strength_rank": 0.106469,
    "earnings_yield_rank": 0.012693,
    "fcf_yield_rank": 0.013838,
    "shareholder_yield_rank": 0.294649,
    "risk_adjusted_momentum_6m_rank": 0.0,
}
PRICE_ONLY_WEIGHTS = {
    "momentum_12m_ex_1m_rank": 0.32,
    "momentum_6m_rank": 0.26,
    "momentum_3m_rank": 0.14,
    "trend_quality_rank": 0.20,
    "downside_risk_rank": 0.03,
    "drawdown_rank": 0.02,
    "liquidity_rank": 0.03,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/sp500_daily_dataset.csv"))
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--refine-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--audit-years", type=int, default=2)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--holding-days", type=int, default=126)
    parser.add_argument(
        "--anchor-days", type=int, default=21,
        help="Spacing between overlapping tuning portfolios",
    )
    parser.add_argument("--min-weight", type=float, default=-0.20)
    parser.add_argument("--max-weight", type=float, default=0.45)
    parser.add_argument("--max-l1", type=float, default=1.80)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sp500_sharadar_weight_tuning"))
    parser.add_argument("--backtest-years", type=int, default=8)
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("data/raw/benchmark/sp500_spy_sharadar.csv"),
    )
    return parser.parse_args()


def _weight_vector(values: dict[str, float]) -> np.ndarray:
    vector = np.asarray([values.get(name, 0.0) for name in FEATURE_COLUMNS], dtype=float)
    total = float(vector.sum())
    if not np.isfinite(vector).all() or total <= 0.0:
        raise ValueError("Weight vectors must be finite and have a positive sum")
    return vector / total


def _valid_weights(
    values: np.ndarray,
    *,
    min_weight: float,
    max_weight: float,
    max_l1: float,
) -> np.ndarray:
    return (
        (values.min(axis=1) >= min_weight)
        & (values.max(axis=1) <= max_weight)
        & (np.abs(values).sum(axis=1) <= max_l1)
    )


def constrained_samples(
    count: int,
    features: int,
    seed: int,
    min_weight: float,
    max_weight: float,
    max_l1: float,
    centers: np.ndarray,
) -> np.ndarray:
    """Draw positive, signed, and baseline-centred mixes that sum to one."""

    if count == 0:
        return np.empty((0, features), dtype=float)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    gathered = 0
    rounds = 0
    while gathered < count:
        rounds += 1
        if rounds > 1_000:
            raise RuntimeError("Unable to sample weights under the requested constraints")
        batch_size = max(2_000, count - gathered)
        mode = rounds % 3
        if mode == 0:
            batch = rng.dirichlet(np.full(features, 0.65), size=batch_size)
        elif mode == 1:
            batch = rng.normal(1.0 / features, 0.105, size=(batch_size, features))
            batch += ((1.0 - batch.sum(axis=1)) / features)[:, None]
        else:
            base = centers[rng.integers(0, len(centers), size=batch_size)]
            scale = rng.choice((0.025, 0.05, 0.09), size=batch_size)
            noise = rng.normal(size=(batch_size, features))
            noise -= noise.mean(axis=1)[:, None]
            batch = base + noise * scale[:, None]
        batch = batch[_valid_weights(
            batch,
            min_weight=min_weight,
            max_weight=max_weight,
            max_l1=max_l1,
        )]
        if len(batch):
            accepted.append(batch)
            gathered += len(batch)
    return np.vstack(accepted)[:count]


def refined_samples(
    count: int,
    centers: np.ndarray,
    seed: int,
    min_weight: float,
    max_weight: float,
    max_l1: float,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, centers.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    gathered = 0
    while gathered < count:
        batch_size = max(2_000, count - gathered)
        base = centers[rng.integers(0, len(centers), size=batch_size)]
        scale = rng.choice((0.008, 0.015, 0.03, 0.05), size=batch_size)
        noise = rng.normal(size=base.shape)
        noise -= noise.mean(axis=1)[:, None]
        batch = base + noise * scale[:, None]
        batch = batch[_valid_weights(
            batch,
            min_weight=min_weight,
            max_weight=max_weight,
            max_l1=max_l1,
        )]
        if len(batch):
            accepted.append(batch)
            gathered += len(batch)
    return np.vstack(accepted)[:count]


def _benchmark_by_session(
    benchmark: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> dict[pd.Timestamp, float]:
    frame = benchmark.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    series = frame.dropna().drop_duplicates("date", keep="last").set_index("date")["close"].sort_index()
    aligned = series.reindex(sessions, method="ffill", tolerance=pd.Timedelta(days=4))
    return {pd.Timestamp(date): float(value) for date, value in aligned.dropna().items()}


def build_periods(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    holding_days: int,
    anchor_days: int,
) -> list[dict[str, object]]:
    sessions = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    benchmark_prices = _benchmark_by_session(benchmark, sessions)
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
        start_benchmark = benchmark_prices.get(start)
        end_benchmark = benchmark_prices.get(end)
        if (
            day is None
            or day.empty
            or start_benchmark is None
            or end_benchmark is None
            or start_benchmark <= 0.0
        ):
            continue
        feature_rows: list[np.ndarray] = []
        forward_returns: list[float] = []
        tickers: list[str] = []
        target = np.datetime64(end.to_datetime64())
        for row in day.itertuples(index=False):
            ticker = str(row.ticker)
            history = price_history.get(ticker)
            entry = float(row.execution_open)
            if history is None or not np.isfinite(entry) or entry <= 0.0:
                continue
            dates, opens, marks = history
            exit_offset = int(np.searchsorted(dates, target, side="right") - 1)
            if exit_offset < 0 or dates[exit_offset] < np.datetime64(start.to_datetime64()):
                continue
            exit_price = opens[exit_offset] if dates[exit_offset] == target else marks[exit_offset]
            if not np.isfinite(exit_price) or exit_price <= 0.0:
                continue
            feature_rows.append(np.asarray(
                [getattr(row, name) for name in FEATURE_COLUMNS], dtype=np.float32
            ))
            forward_returns.append(exit_price / entry - 1.0)
            tickers.append(ticker)
        if len(tickers) >= 10:
            periods.append({
                "date": start,
                "end_date": end,
                "tickers": np.asarray(tickers),
                "features": np.nan_to_num(np.vstack(feature_rows), nan=0.5),
                "returns": np.asarray(forward_returns, dtype=np.float32),
                "benchmark_return": end_benchmark / start_benchmark - 1.0,
            })
    return periods


def evaluate(
    periods: list[dict[str, object]],
    weights: np.ndarray,
    top_n: int,
    batch_size: int,
) -> np.ndarray:
    """Evaluate candidates in bounded-memory batches."""

    output = np.zeros((len(weights), len(periods)), dtype=np.float32)
    for batch_start in range(0, len(weights), batch_size):
        batch_end = min(batch_start + batch_size, len(weights))
        batch = weights[batch_start:batch_end].astype(np.float32, copy=False)
        for period_index, period in enumerate(periods):
            features = period["features"]
            returns = period["returns"]
            assert isinstance(features, np.ndarray) and isinstance(returns, np.ndarray)
            scores = features @ batch.T
            pick_count = min(top_n, len(returns))
            selected = np.argpartition(scores, -pick_count, axis=0)[-pick_count:, :]
            output[batch_start:batch_end, period_index] = returns[selected].mean(axis=0)
        print(f"Evaluated {batch_end:,}/{len(weights):,} candidates", file=sys.stderr)
    return output


def candidate_metrics(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    development: np.ndarray,
    audit: np.ndarray,
) -> dict[str, np.ndarray]:
    active = np.log1p(np.clip(returns, -0.999, None)) - np.log1p(benchmark_returns)[None, :]
    development_active = active[:, development]
    audit_active = active[:, audit]
    folds = [fold for fold in np.array_split(development, 4) if len(fold)]
    fold_means = np.column_stack([active[:, fold].mean(axis=1) for fold in folds])
    mean_active = development_active.mean(axis=1)
    lower_quartile = np.quantile(development_active, 0.25, axis=1)
    outperformance_rate = (development_active > 0.0).mean(axis=1)
    fold_std = fold_means.std(axis=1)
    # Mean active compounding drives the area between the curves. The lower
    # quartile, hit-rate, and fold terms favour a lead that persists across time.
    robust_score = (
        mean_active
        + 0.35 * lower_quartile
        + 0.01 * (outperformance_rate - 0.50)
        - 0.35 * fold_std
    )
    return {
        "robust_score": robust_score,
        "development_mean_active": mean_active,
        "development_active_q25": lower_quartile,
        "development_outperformance_rate": outperformance_rate,
        "development_fold_std": fold_std,
        "development_strategy_mean": returns[:, development].mean(axis=1),
        "audit_mean_active": audit_active.mean(axis=1),
        "audit_active_q25": np.quantile(audit_active, 0.25, axis=1),
        "audit_outperformance_rate": (audit_active > 0.0).mean(axis=1),
        "audit_strategy_mean": returns[:, audit].mean(axis=1),
        "fold_means": fold_means,
    }


def candidates_frame(
    labels: list[str],
    weights: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "label": labels,
        "robust_score_pct": metrics["robust_score"] * 100.0,
        "development_mean_active_log_return_pct": metrics["development_mean_active"] * 100.0,
        "development_active_q25_pct": metrics["development_active_q25"] * 100.0,
        "development_outperformance_rate_pct": metrics["development_outperformance_rate"] * 100.0,
        "development_fold_std_pct": metrics["development_fold_std"] * 100.0,
        "development_mean_6m_return_pct": metrics["development_strategy_mean"] * 100.0,
        "audit_mean_active_log_return_pct": metrics["audit_mean_active"] * 100.0,
        "audit_active_q25_pct": metrics["audit_active_q25"] * 100.0,
        "audit_outperformance_rate_pct": metrics["audit_outperformance_rate"] * 100.0,
        "audit_mean_6m_return_pct": metrics["audit_strategy_mean"] * 100.0,
    })
    return pd.concat(
        [frame, pd.DataFrame(weights, columns=FEATURE_COLUMNS)], axis=1
    ).sort_values("robust_score_pct", ascending=False).reset_index(drop=True)


def rescore(ranked: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    output = ranked.copy()
    output["selection_score"] = 100.0 * (
        output.loc[:, FEATURE_COLUMNS].fillna(0.5).to_numpy(dtype=float) @ weights
    )
    output["selection_rank"] = output.groupby("date")["selection_score"].rank(
        ascending=False, method="first"
    ).astype("int64")
    return output.sort_values(["date", "selection_rank", "ticker"])


def curve_metrics(daily: pd.DataFrame, initial_capital: float) -> dict[str, float]:
    valid = daily.dropna(subset=["equity", "benchmark_equity"])
    gap = valid["equity"] - valid["benchmark_equity"]
    relative_wealth = valid["equity"] / valid["benchmark_equity"] - 1.0
    return {
        "mean_equity_gap": float(gap.mean()),
        "mean_equity_gap_pct_of_initial": float(gap.mean() / initial_capital * 100.0),
        "days_ahead_pct": float((gap > 0.0).mean() * 100.0),
        "worst_equity_gap": float(gap.min()),
        "final_relative_wealth_pct": float(relative_wealth.iloc[-1] * 100.0),
    }


def _run_backtest(
    dataset: pd.DataFrame,
    benchmark: pd.DataFrame,
    ranked: pd.DataFrame,
    output_dir: Path,
    years: int,
    top_n: int,
    holding_days: int,
    limitation: str,
):
    return run_selector_hold_backtest(
        dataset,
        constituents=None,
        years=years,
        top_n=top_n,
        holding_days=holding_days,
        initial_capital=20_000.0,
        output_dir=output_dir,
        benchmark=benchmark,
        market_name="S&P 500 (Sharadar point-in-time)",
        benchmark_name="S&P 500 (SPY, Sharadar)",
        benchmark_symbol="SPY",
        currency_symbol="$",
        commission_rate=0.0,
        minimum_commission=0.0,
        ranking_function=lambda _prices: ranked,
        data_source="Sharadar SEP + DAILY + ART + historical S&P 500 membership",
        research_limitation=limitation,
    )


def main() -> None:
    args = parse_args()
    if (
        args.samples < 1
        or args.refine_samples < 0
        or args.audit_years < 1
        or args.top < 1
        or args.holding_days < 1
        or args.anchor_days < 1
        or args.batch_size < 1
    ):
        raise SystemExit("Sample counts, years, top count, holding days, anchor days, and batch size are invalid")
    if args.min_weight >= args.max_weight or args.max_l1 < 1.0:
        raise SystemExit("Weight bounds are invalid")
    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}")
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
    dataset = pd.read_csv(
        args.input,
        usecols=[column for column in wanted if column in header],
        low_memory=False,
    )
    prices = _prepare_prices(dataset)
    sessions = pd.DatetimeIndex(prices["date"].unique()).sort_values()
    tuning_dates = sessions[0:len(sessions) - args.holding_days:args.anchor_days]
    requested_start = sessions[-1] - pd.DateOffset(years=args.backtest_years)
    production_sessions = sessions[sessions >= requested_start]
    production_dates = build_rebalance_dates(production_sessions, args.holding_days)
    ranking_dates = sorted(
        set(pd.Timestamp(value) for value in tuning_dates)
        | set(production_dates)
        | {pd.Timestamp(sessions[-1])}
    )
    print(f"Ranking {len(ranking_dates)} decision dates", file=sys.stderr)
    ranked = rank_sp500_stocks(prices, ranking_dates=ranking_dates)
    periods = build_periods(
        ranked, prices, benchmark, args.holding_days, args.anchor_days
    )
    if not periods:
        raise SystemExit("No eligible tuning periods were produced")

    last_end = max(pd.Timestamp(period["end_date"]) for period in periods)
    audit_start = last_end - pd.DateOffset(years=args.audit_years)
    development = np.asarray([
        index for index, period in enumerate(periods)
        if pd.Timestamp(period["end_date"]) < audit_start
    ])
    audit = np.asarray([
        index for index, period in enumerate(periods)
        if pd.Timestamp(period["date"]) >= audit_start
    ])
    if len(development) < 12 or len(audit) < 3:
        raise SystemExit("Not enough purged development/audit periods for robust tuning")
    benchmark_returns = np.asarray(
        [period["benchmark_return"] for period in periods], dtype=float
    )

    named = {
        "current_user_weights": _weight_vector(DEFAULT_SP500_WEIGHTS),
        "legacy_tuner_weights": _weight_vector(LEGACY_WEIGHTS),
        "price_only_weights": _weight_vector(PRICE_ONLY_WEIGHTS),
        "equal_weights": np.full(len(FEATURE_COLUMNS), 1.0 / len(FEATURE_COLUMNS)),
    }
    named_weights = np.vstack(list(named.values()))
    random_weights = constrained_samples(
        args.samples,
        len(FEATURE_COLUMNS),
        args.seed,
        args.min_weight,
        args.max_weight,
        args.max_l1,
        named_weights,
    )
    initial_weights = np.vstack([named_weights, random_weights])
    initial_labels = list(named) + [f"sample_{index:05d}" for index in range(args.samples)]
    initial_returns = evaluate(periods, initial_weights, args.top, args.batch_size)
    initial_metrics = candidate_metrics(
        initial_returns, benchmark_returns, development, audit
    )

    seed_count = min(32, len(initial_weights))
    seed_indices = np.argsort(initial_metrics["robust_score"])[-seed_count:]
    refinements = refined_samples(
        args.refine_samples,
        initial_weights[seed_indices],
        args.seed + 1,
        args.min_weight,
        args.max_weight,
        args.max_l1,
    )
    if len(refinements):
        refinement_returns = evaluate(periods, refinements, args.top, args.batch_size)
        all_weights = np.vstack([initial_weights, refinements])
        all_returns = np.vstack([initial_returns, refinement_returns])
        labels = initial_labels + [
            f"refinement_{index:05d}" for index in range(len(refinements))
        ]
    else:
        all_weights = initial_weights
        all_returns = initial_returns
        labels = initial_labels
    metrics = candidate_metrics(all_returns, benchmark_returns, development, audit)
    winner = int(np.argmax(metrics["robust_score"]))
    candidates = candidates_frame(labels, all_weights, metrics)
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)

    limitation = (
        "Weights were selected only on a purged development sample using benchmark-relative "
        "returns, then evaluated once on the final held-out audit period. Historical Sharadar "
        "membership, adjusted prices, and filing-date fundamentals are used."
    )
    winner_ranked = rescore(ranked, all_weights[winner])
    baseline_ranked = rescore(ranked, named["current_user_weights"])
    winner_backtest = _run_backtest(
        dataset,
        benchmark,
        winner_ranked,
        args.output_dir / f"optimized_backtest_{args.backtest_years}y",
        args.backtest_years,
        args.top,
        args.holding_days,
        limitation,
    )
    baseline_backtest = _run_backtest(
        dataset,
        benchmark,
        baseline_ranked,
        args.output_dir / f"baseline_backtest_{args.backtest_years}y",
        args.backtest_years,
        args.top,
        args.holding_days,
        limitation,
    )

    baseline_index = 0
    summary = {
        "data_source": "Sharadar only (SEP, DAILY, ART and point-in-time membership)",
        "objective": (
            "development mean active log return + 0.35 * active-return lower quartile "
            "+ outperformance-rate bonus - 0.35 * chronological-fold instability"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "samples": args.samples,
        "refine_samples": args.refine_samples,
        "seed": args.seed,
        "period_start": pd.Timestamp(periods[0]["date"]).date().isoformat(),
        "period_end": pd.Timestamp(periods[-1]["end_date"]).date().isoformat(),
        "audit_start": audit_start.date().isoformat(),
        "development_periods": len(development),
        "purged_periods": len(periods) - len(development) - len(audit),
        "audit_periods": len(audit),
        "winner": labels[winner],
        "recommended_weights": {
            name: float(all_weights[winner, column])
            for column, name in enumerate(FEATURE_COLUMNS)
        },
        "winner_development_mean_active_log_return_pct": float(metrics["development_mean_active"][winner] * 100.0),
        "winner_development_active_q25_pct": float(metrics["development_active_q25"][winner] * 100.0),
        "winner_development_outperformance_rate_pct": float(metrics["development_outperformance_rate"][winner] * 100.0),
        "winner_development_fold_means_pct": [float(value * 100.0) for value in metrics["fold_means"][winner]],
        "winner_audit_mean_active_log_return_pct": float(metrics["audit_mean_active"][winner] * 100.0),
        "winner_audit_active_q25_pct": float(metrics["audit_active_q25"][winner] * 100.0),
        "winner_audit_outperformance_rate_pct": float(metrics["audit_outperformance_rate"][winner] * 100.0),
        "baseline_development_mean_active_log_return_pct": float(metrics["development_mean_active"][baseline_index] * 100.0),
        "baseline_audit_mean_active_log_return_pct": float(metrics["audit_mean_active"][baseline_index] * 100.0),
        "optimized_backtest": winner_backtest.summary,
        "optimized_curve": curve_metrics(winner_backtest.daily_equity, 20_000.0),
        "baseline_backtest": baseline_backtest.summary,
        "baseline_curve": curve_metrics(baseline_backtest.daily_equity, 20_000.0),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
