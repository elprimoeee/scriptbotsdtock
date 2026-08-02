"""Reusable benchmark comparison modes for model backtests."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class BenchmarkMode(str, Enum):
    DIRECT = "direct"
    HEDGED_SHORT_BENCHMARK = "hedged_short_benchmark"


def _normalize_returns(frame: pd.DataFrame, date_column: str, return_column: str) -> pd.DataFrame:
    normalized = frame[[date_column, return_column]].copy()
    normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce").dt.normalize()
    normalized[return_column] = pd.to_numeric(normalized[return_column], errors="coerce").fillna(0.0)
    normalized = normalized.dropna(subset=[date_column]).sort_values(date_column)
    normalized = normalized.drop_duplicates(subset=[date_column], keep="last").reset_index(drop=True)
    return normalized


def build_direct_benchmark_test(
    strategy_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    initial_capital: float = 10_000.0,
    strategy_return_column: str = "daily_return",
    benchmark_return_column: str = "benchmark_daily_return",
) -> pd.DataFrame:
    strategy = _normalize_returns(strategy_daily, "date", strategy_return_column)
    benchmark = _normalize_returns(benchmark_daily, "date", benchmark_return_column)
    merged = strategy.merge(benchmark, on="date", how="inner")
    merged["strategy_equity"] = initial_capital * (1.0 + merged[strategy_return_column]).cumprod()
    merged["benchmark_equity"] = initial_capital * (1.0 + merged[benchmark_return_column]).cumprod()
    merged["relative_outperformance"] = (
        merged["strategy_equity"] / merged["benchmark_equity"]
    ) - 1.0
    merged["test_mode"] = BenchmarkMode.DIRECT.value
    return merged


def build_hedged_short_benchmark_test(
    position_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    initial_capital: float = 10_000.0,
    strategy_return_column: str = "position_return",
    notional_column: str = "notional",
    benchmark_return_column: str = "benchmark_daily_return",
) -> pd.DataFrame:
    positions = position_daily[["date", strategy_return_column, notional_column]].copy()
    positions["date"] = pd.to_datetime(positions["date"], errors="coerce").dt.normalize()
    positions[strategy_return_column] = pd.to_numeric(
        positions[strategy_return_column], errors="coerce"
    ).fillna(0.0)
    positions[notional_column] = pd.to_numeric(positions[notional_column], errors="coerce").fillna(0.0)
    positions = positions.dropna(subset=["date"]).sort_values("date")
    positions = positions.groupby("date", as_index=False).agg(
        {
            strategy_return_column: "sum",
            notional_column: "sum",
        }
    )

    benchmark = _normalize_returns(benchmark_daily, "date", benchmark_return_column)
    merged = positions.merge(benchmark, on="date", how="left")
    merged[benchmark_return_column] = merged[benchmark_return_column].fillna(0.0)

    capital_base = float(initial_capital) if initial_capital else 1.0
    merged["strategy_pnl"] = merged[strategy_return_column] * merged[notional_column]
    merged["benchmark_hedge_pnl"] = -merged[benchmark_return_column] * merged[notional_column]
    merged["combined_pnl"] = merged["strategy_pnl"] + merged["benchmark_hedge_pnl"]
    merged["combined_return"] = merged["combined_pnl"] / capital_base
    merged["hedged_equity"] = capital_base + merged["combined_pnl"].cumsum()
    merged["test_mode"] = BenchmarkMode.HEDGED_SHORT_BENCHMARK.value
    return merged
