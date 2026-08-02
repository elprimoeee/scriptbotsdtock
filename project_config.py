"""Shared project configuration for backtests, fees, data paths, and output structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    input_csv: Path = Path("data/processed/asx_daily_dataset.csv")
    output_dir: Path = Path("results")
    benchmark_cache_csv: Path = Path("data/raw/benchmark/asx200_benchmark.csv")


@dataclass(frozen=True)
class BenchmarkConfig:
    symbol: str = "^AXJO"
    chart_file: str = "strategy_vs_benchmark.svg"
    benchmark_daily_file: str = "strategy_vs_benchmark_daily.csv"
    benchmark_test_file: str = "benchmark_test.csv"
    direct_results_dir: str = "direct"
    hedged_results_dir: str = "hedged"


@dataclass(frozen=True)
class PortfolioConfig:
    min_price: float = 0.50
    min_dollar_volume: float = 200_000.0
    initial_capital: float = 20_000.0
    slippage_per_trade: float = 0.0005
    top_metric_count: int = 8
    top_rank_count: int = 20
    sell_rank_count: int = 40
    min_active_positions: int = 8
    max_active_positions: int = 8
    confidence_weight_floor: float = 0.0
    confidence_weight_spread: float = 1.0
    equality_coherence: float = 0.5
    position_rebalance_threshold: float = 0.05
    switch_safety_buffer: float = 0.0025
    hedge_beta_window_days: int = 63
    hedge_beta_min_obs: int = 21
    invest_start_delay_months: int = 36


@dataclass(frozen=True)
class CfdCostConfig:
    trading_days_per_year: float = 252.0
    funding_day_count: float = 360.0
    friday_funding_multiplier: float = 3.0
    asx_share_cfd_max_leverage: float = 5.0
    asx_share_cfd_commission_rate: float = 0.0009
    asx_share_cfd_min_commission_aud: float = 7.0
    asx_share_cfd_overnight_funding_annual_rate: float = 0.04
    asx_share_cfd_short_borrow_admin_annual_rate: float = 0.005
    index_cfd_commission_rate: float = 0.0
    index_cfd_min_commission_aud: float = 0.0
    index_cfd_overnight_funding_annual_rate: float = 0.04

    @property
    def asx_share_cfd_margin_requirement(self) -> float:
        return 1.0 / self.asx_share_cfd_max_leverage


PATHS = PathConfig()
BENCHMARKS = BenchmarkConfig()
PORTFOLIO = PortfolioConfig()
CFD_COSTS = CfdCostConfig()
