#!/usr/bin/env python3
"""Rules-based monthly momentum backtest for ASX stocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.benchmarking import build_direct_benchmark_test
from project_config import BENCHMARKS, PATHS, PORTFOLIO


INVERSE_ETF_LEVERAGE = 3.0
INVERSE_ETF_TICKER = "ASX200_3X_SHORT_ETF"
SIGNAL_LAG_DAYS = 1
MOMENTUM_LOOKBACK_DAYS = 90
HOLDING_PERIOD_DAYS = 180
TOP_MOMENTUM_GROUP_SIZE = 20
SELL_RANK_THRESHOLD = 30
BUY_COUNT = 10
TARGET_POSITION_WEIGHT = 0.10
MAX_POSITION_WEIGHT = 0.15
MAX_SECTOR_WEIGHT = 0.30
MIN_DOLLAR_VOLUME = 1_000_000.0
COMMISSION_RATE = 0.001
MIN_COMMISSION_AUD = 8.0
EXPECTED_RETURN_COLUMN = f"expected_forward_return_{HOLDING_PERIOD_DAYS}d"
EXPECTED_EDGE_COLUMN = f"expected_edge_{HOLDING_PERIOD_DAYS}d"
POSITION_EXPECTED_RETURN_COLUMN = f"position_expected_return_{HOLDING_PERIOD_DAYS}d"


@dataclass
class MetricRunResult:
    output_dir: Path
    summary: dict[str, object]
    saved_paths: dict[str, Path]


def load_data(input_csv: Path) -> pd.DataFrame:
    usecols = ["date", "ticker", "close", "dollar_volume", "sector", "price_vs_200d_ma"]
    df = pd.read_csv(input_csv, usecols=usecols, parse_dates=["date"], low_memory=False)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["sector"] = df["sector"].fillna("Unknown").astype(str)
    for column in ("close", "dollar_volume", "price_vs_200d_ma"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped_close = df.groupby("ticker", sort=False)["close"]
    df["next_return"] = (grouped_close.shift(-1) / df["close"]) - 1.0
    df["target_forward_return"] = (grouped_close.shift(-HOLDING_PERIOD_DAYS) / df["close"]) - 1.0
    df["momentum_90d"] = grouped_close.pct_change(MOMENTUM_LOOKBACK_DAYS).astype("float32")

    signal_cols = ["close", "dollar_volume", "price_vs_200d_ma", "momentum_90d"]
    shifted = df.groupby("ticker", sort=False)[signal_cols].shift(SIGNAL_LAG_DAYS)
    df["signal_close"] = shifted["close"].astype("float32")
    df["signal_dollar_volume"] = shifted["dollar_volume"].astype("float32")
    df["signal_price_vs_200d_ma"] = shifted["price_vs_200d_ma"].astype("float32")
    df["signal_momentum_90d"] = shifted["momentum_90d"].astype("float32")

    value_cols = [
        "next_return",
        "target_forward_return",
        "signal_close",
        "signal_dollar_volume",
        "signal_price_vs_200d_ma",
        "signal_momentum_90d",
    ]
    df[value_cols] = df[value_cols].replace([np.inf, -np.inf], np.nan)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def build_monthly_rebalance_dates(date_values) -> pd.DatetimeIndex:
    dates = pd.Series(pd.to_datetime(date_values, errors="coerce")).dropna().drop_duplicates().sort_values()
    if dates.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates.groupby(dates.dt.to_period("M")).min().tolist())


def prepare_benchmark_frame(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    prepared = benchmark_df[["date", "close"]].copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    prepared["next_benchmark_return"] = (prepared["close"].shift(-1) / prepared["close"]) - 1.0
    return prepared


def build_candidate_frame(day: pd.DataFrame) -> pd.DataFrame:
    if day.empty:
        return day.copy()

    ranked = day[
        [
            "date",
            "ticker",
            "sector",
            "next_return",
            "target_forward_return",
            "signal_close",
            "signal_dollar_volume",
            "signal_price_vs_200d_ma",
            "signal_momentum_90d",
        ]
    ].copy()
    ranked = ranked.dropna(subset=["signal_momentum_90d", "signal_dollar_volume", "signal_price_vs_200d_ma"])
    ranked = ranked[
        (ranked["signal_close"].fillna(0.0) > 0.0)
        & (ranked["signal_dollar_volume"].fillna(0.0) >= MIN_DOLLAR_VOLUME)
        & (ranked["signal_price_vs_200d_ma"].fillna(-1.0) > 0.0)
    ].copy()
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(["signal_momentum_90d", "ticker"], ascending=[False, True]).reset_index(drop=True)
    ranked["probability_rank"] = np.arange(1, len(ranked) + 1, dtype="int64")
    ranked["probability_up"] = 1.0 - ((ranked["probability_rank"] - 1) / float(len(ranked)))
    ranked["score_percentile"] = ranked["probability_up"].astype(float)
    ranked["metric_score"] = ranked["signal_momentum_90d"].astype(float)
    ranked["directional_edge"] = ranked["metric_score"] - float(ranked["metric_score"].median())
    ranked[EXPECTED_RETURN_COLUMN] = pd.to_numeric(ranked["target_forward_return"], errors="coerce")
    ranked[EXPECTED_EDGE_COLUMN] = ranked[EXPECTED_RETURN_COLUMN] - float(ranked[EXPECTED_RETURN_COLUMN].median(skipna=True))
    ranked[POSITION_EXPECTED_RETURN_COLUMN] = ranked[EXPECTED_RETURN_COLUMN]
    return ranked


def compute_rank_ic(sample: pd.DataFrame) -> float:
    valid = sample[["metric_score", "target_forward_return"]].dropna()
    if len(valid) < 2:
        return float("nan")
    value = valid["metric_score"].rank(method="average").corr(
        valid["target_forward_return"].rank(method="average"),
        method="pearson",
    )
    return float(value) if pd.notna(value) else float("nan")


def prepare_strategy_data(df: pd.DataFrame):
    grouped_days = {pd.Timestamp(date_value): sample.copy() for date_value, sample in df.groupby("date", sort=True)}
    rebalance_dates = build_monthly_rebalance_dates(df["date"])
    month_ends = {
        period: pd.Timestamp(sample.max()).normalize()
        for period, sample in df["date"].groupby(df["date"].dt.to_period("M"))
    }
    ranked_lookup: dict[pd.Timestamp, pd.DataFrame] = {}
    metric_rows = []
    period_rows = []
    first_valid_date: pd.Timestamp | None = None
    for date_value in rebalance_dates:
        ranked = build_candidate_frame(grouped_days.get(pd.Timestamp(date_value), pd.DataFrame()))
        if ranked.empty:
            continue
        date_value = pd.Timestamp(date_value).normalize()
        ranked_lookup[date_value] = ranked
        if first_valid_date is None:
            first_valid_date = date_value
        month_period = date_value.to_period("M")
        rank_ic = compute_rank_ic(ranked)
        period_rows.append(
            {
                "rebalance_month": str(month_period),
                "train_start_date": None,
                "train_end_date": None,
                "test_start_date": date_value.date().isoformat(),
                "test_end_date": month_ends[month_period].date().isoformat(),
                "selected_metric_count": 1,
                "eligible_count": int(len(ranked)),
                "top_group_count": int(min(SELL_RANK_THRESHOLD, len(ranked))),
                "buy_count": int(min(BUY_COUNT, len(ranked))),
                "test_rank_ic": rank_ic,
                "avg_top20_momentum": float(ranked.head(TOP_MOMENTUM_GROUP_SIZE)["metric_score"].mean()),
                "avg_top20_forward_return": float(
                    ranked.head(TOP_MOMENTUM_GROUP_SIZE)["target_forward_return"].dropna().mean()
                ),
            }
        )
        metric_rows.append(
            {
                "rebalance_month": str(month_period),
                "metric": "momentum_90d",
                "selected_for_model": True,
                "weight": 1.0,
                "direction": 1.0,
                "signed_weight": 1.0,
                "coefficient": 1.0,
                "abs_coefficient": 1.0,
                "rank_ic": rank_ic,
                "eligible_count": int(len(ranked)),
            }
        )
    if first_valid_date is None:
        raise SystemExit("No eligible monthly rebalance dates were produced. Backtest cannot run.")
    return (
        pd.DataFrame(metric_rows),
        df.loc[df["date"] >= first_valid_date].copy(),
        pd.DataFrame(period_rows),
        ranked_lookup,
    )


def trade_commission_fraction(trade_notional: float, equity_base: float) -> float:
    if equity_base <= 0.0 or trade_notional <= 1e-12:
        return 0.0
    return max(float(trade_notional) * COMMISSION_RATE, MIN_COMMISSION_AUD) / float(equity_base)


def commission_cost_from_weight_changes(
    previous_weights: dict[str, float],
    executed_weights: dict[str, float],
    *,
    equity_base: float,
) -> float:
    total = 0.0
    for ticker in set(previous_weights) | set(executed_weights):
        delta = abs(float(executed_weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0)))
        if delta > 1e-12:
            total += trade_commission_fraction(delta * equity_base, equity_base)
    return total


def update_holding_meta(
    holdings: set[str],
    holding_meta: dict[str, dict[str, object]],
    ranked: pd.DataFrame | None,
) -> None:
    if ranked is None or ranked.empty:
        return
    lookup = ranked.set_index("ticker", drop=False)
    for ticker in holdings:
        if ticker not in lookup.index:
            continue
        row = lookup.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        meta = holding_meta.get(ticker, {})
        meta.update(
            {
                "sector": str(row["sector"]),
                "metric_score": float(row["metric_score"]),
                "probability_up": float(row["probability_up"]),
                "probability_rank": int(row["probability_rank"]),
                "score_percentile": float(row["score_percentile"]),
                "is_in_buy_top_rank": int(int(row["probability_rank"]) <= BUY_COUNT),
                "is_in_sell_top_rank": int(int(row["probability_rank"]) <= SELL_RANK_THRESHOLD),
                EXPECTED_RETURN_COLUMN: float(row[EXPECTED_RETURN_COLUMN]) if pd.notna(row[EXPECTED_RETURN_COLUMN]) else None,
                EXPECTED_EDGE_COLUMN: float(row[EXPECTED_EDGE_COLUMN]) if pd.notna(row[EXPECTED_EDGE_COLUMN]) else 0.0,
                POSITION_EXPECTED_RETURN_COLUMN: (
                    float(row[POSITION_EXPECTED_RETURN_COLUMN])
                    if pd.notna(row[POSITION_EXPECTED_RETURN_COLUMN])
                    else None
                ),
                "directional_edge": float(row["directional_edge"]),
            }
        )
        holding_meta[ticker] = meta


def sector_weights_for_holdings(
    holdings: set[str],
    holding_meta: dict[str, dict[str, object]],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for ticker in holdings:
        sector = str(holding_meta.get(ticker, {}).get("sector", "Unknown"))
        weights[sector] = weights.get(sector, 0.0) + TARGET_POSITION_WEIGHT
    return weights


def run_backtest(
    test_df: pd.DataFrame,
    ranked_lookup: dict[pd.Timestamp, pd.DataFrame],
    stock_side_mode: str = "long_only",
):
    if stock_side_mode != "long_only":
        raise ValueError(f"Unsupported stock_side_mode: {stock_side_mode}")

    holdings: set[str] = set()
    holding_days: dict[str, int] = {}
    holding_meta: dict[str, dict[str, object]] = {}
    current_weights: dict[str, float] = {}
    portfolio_value = float(PORTFOLIO.initial_capital)
    simulation_rows = []
    picks_rows = []

    grouped_days = list(test_df.groupby("date", sort=True))
    for date_value, day in tqdm(
        grouped_days,
        total=len(grouped_days),
        desc="Backtest days (long_only)",
        dynamic_ncols=True,
    ):
        date_value = pd.Timestamp(date_value).normalize()
        day = day.copy()
        day_lookup = day.set_index("ticker", drop=False)
        previous_weights = current_weights.copy()
        previous_holdings = set(holdings)
        ranked = ranked_lookup.get(date_value)
        new_entries: set[str] = set()

        if ranked is not None:
            top_group = set(ranked.head(SELL_RANK_THRESHOLD)["ticker"].tolist())
            next_holdings = set(holdings)
            for ticker in list(holdings):
                if (
                    holding_days.get(ticker, 0) >= HOLDING_PERIOD_DAYS
                    or ticker not in top_group
                    or ticker not in day_lookup.index
                ):
                    next_holdings.discard(ticker)
                    holding_days.pop(ticker, None)
                    holding_meta.pop(ticker, None)

            update_holding_meta(next_holdings, holding_meta, ranked)
            sector_weights = sector_weights_for_holdings(next_holdings, holding_meta)
            for row in ranked.itertuples(index=False):
                if len(next_holdings) >= BUY_COUNT:
                    break
                if row.ticker in next_holdings:
                    continue
                sector_name = str(row.sector)
                if sector_weights.get(sector_name, 0.0) + TARGET_POSITION_WEIGHT > MAX_SECTOR_WEIGHT + 1e-12:
                    continue
                next_holdings.add(row.ticker)
                holding_days[row.ticker] = 0
                holding_meta[row.ticker] = {"entry_date": date_value, "sector": sector_name}
                new_entries.add(row.ticker)
                sector_weights[sector_name] = sector_weights.get(sector_name, 0.0) + TARGET_POSITION_WEIGHT

            holdings = next_holdings
            update_holding_meta(holdings, holding_meta, ranked)
            target_weights = {ticker: TARGET_POSITION_WEIGHT for ticker in holdings}
        else:
            target_weights = {ticker: weight for ticker, weight in previous_weights.items() if ticker in holdings}

        executed_weights = {
            ticker: float(target_weights.get(ticker, 0.0))
            for ticker in holdings
            if float(target_weights.get(ticker, 0.0)) > 1e-12
        }
        held_rows = day.loc[day["ticker"].isin(executed_weights)].copy()
        if not held_rows.empty:
            held_rows["next_day_return_filled"] = pd.to_numeric(held_rows["next_return"], errors="coerce").fillna(0.0)
            held_rows = held_rows.reset_index(drop=True)

        all_tickers = set(previous_weights) | set(executed_weights)
        buy_turnover = float(
            sum(max(executed_weights.get(t, 0.0) - previous_weights.get(t, 0.0), 0.0) for t in all_tickers)
        )
        sell_turnover = float(
            sum(max(previous_weights.get(t, 0.0) - executed_weights.get(t, 0.0), 0.0) for t in all_tickers)
        )
        turnover_fraction = float(
            sum(abs(executed_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in all_tickers)
        )
        buys = int(sum(1 for t in all_tickers if (executed_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) > 1e-12))
        sells = int(sum(1 for t in all_tickers if (previous_weights.get(t, 0.0) - executed_weights.get(t, 0.0)) > 1e-12))
        commission_cost_fraction = commission_cost_from_weight_changes(
            previous_weights,
            executed_weights,
            equity_base=portfolio_value,
        )

        return_map = {ticker: 0.0 for ticker in executed_weights}
        if not held_rows.empty:
            return_map.update(
                {
                    row.ticker: float(row.next_day_return_filled)
                    for row in held_rows.itertuples(index=False)
                }
            )
        gross_stock_return = float(sum(executed_weights[ticker] * return_map.get(ticker, 0.0) for ticker in executed_weights))
        total_cost_fraction = commission_cost_fraction
        net_multiplier = (1.0 + gross_stock_return) * max(0.0, 1.0 - total_cost_fraction)
        portfolio_value = max(0.0, portfolio_value * net_multiplier)
        current_weights = executed_weights.copy()

        pre_increment_days = {ticker: holding_days.get(ticker, 0) for ticker in holdings}
        for ticker in list(holdings):
            holding_days[ticker] = holding_days.get(ticker, 0) + 1

        if not held_rows.empty:
            held_rows["signed_position_weight"] = held_rows["ticker"].map(executed_weights).astype(float)
            held_rows["position_weight"] = held_rows["signed_position_weight"]
            for row in held_rows.itertuples(index=False):
                meta = holding_meta.get(row.ticker, {})
                picks_rows.append(
                    {
                        "date": date_value.date().isoformat(),
                        "ticker": row.ticker,
                        "probability_up": float(meta.get("probability_up", 0.0)),
                        "probability_rank": int(meta.get("probability_rank", BUY_COUNT + 1)),
                        "confidence_score": float(meta.get("score_percentile", 0.0)),
                        "target_position_weight": TARGET_POSITION_WEIGHT,
                        "position_weight": float(row.position_weight),
                        "target_signed_position_weight": TARGET_POSITION_WEIGHT,
                        "signed_position_weight": float(row.signed_position_weight),
                        "position_side": "long",
                        "target_position_side": "long",
                        "directional_edge": float(meta.get("directional_edge", meta.get("metric_score", 0.0))),
                        EXPECTED_EDGE_COLUMN: meta.get(EXPECTED_EDGE_COLUMN),
                        EXPECTED_RETURN_COLUMN: meta.get(EXPECTED_RETURN_COLUMN),
                        POSITION_EXPECTED_RETURN_COLUMN: meta.get(POSITION_EXPECTED_RETURN_COLUMN),
                        "metric_score": float(
                            meta.get("metric_score", row.signal_momentum_90d if pd.notna(row.signal_momentum_90d) else 0.0)
                        ),
                        "signal_close": float(row.signal_close) if pd.notna(row.signal_close) else None,
                        "signal_dollar_volume": float(row.signal_dollar_volume) if pd.notna(row.signal_dollar_volume) else None,
                        "score_percentile": float(meta.get("score_percentile", 0.0)),
                        "rebalance_month": str(date_value.to_period("M")),
                        "train_start_date": None,
                        "train_end_date": None,
                        "test_start_date": date_value.date().isoformat(),
                        "test_end_date": date_value.date().isoformat(),
                        "is_in_buy_top_rank": int(meta.get("is_in_buy_top_rank", 0)),
                        "is_in_sell_top_rank": int(meta.get("is_in_sell_top_rank", 0)),
                        "is_new_buy": int(row.ticker in new_entries),
                        "out_of_top_rank_days": 0,
                        "holding_days": int(pre_increment_days.get(row.ticker, 0)),
                        "next_day_return": float(row.next_return) if pd.notna(row.next_return) else None,
                        "next_day_return_filled": float(row.next_day_return_filled),
                    }
                )

        weight_values = np.asarray(list(executed_weights.values()), dtype="float64")
        gross_exposure = float(weight_values.sum()) if len(weight_values) else 0.0
        simulation_rows.append(
            {
                "date": date_value.date().isoformat(),
                "gross_stock_return": gross_stock_return,
                "long_stock_return": gross_stock_return,
                "short_stock_return": 0.0,
                "gross_daily_return": gross_stock_return,
                "daily_return": net_multiplier - 1.0,
                "slippage_cost_fraction": 0.0,
                "commission_cost_fraction": commission_cost_fraction,
                "overnight_funding_cost_fraction": 0.0,
                "short_borrow_cost_fraction": 0.0,
                "total_cost_fraction": total_cost_fraction,
                "turnover_fraction": turnover_fraction,
                "stock_turnover_fraction": turnover_fraction,
                "buy_weight_turnover": buy_turnover,
                "sell_weight_turnover": sell_turnover,
                "portfolio_value": portfolio_value,
                "cumulative_return": (portfolio_value / PORTFOLIO.initial_capital) - 1.0,
                "holdings_count": len(executed_weights),
                "buys": buys,
                "sells": sells,
                "buy_top_rank_count": BUY_COUNT,
                "sell_top_rank_count": SELL_RANK_THRESHOLD,
                "min_fill_buys": int(max(0, len(new_entries) - max(0, BUY_COUNT - len(previous_holdings)))),
                "avg_holding_days": float(np.mean(list(holding_days.values()))) if holding_days else 0.0,
                "missing_returns_count": int(held_rows["next_return"].isna().sum()) if not held_rows.empty else 0,
                "weight_rebalances": int(sum(1 for t in all_tickers if abs(executed_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) > 1e-12)),
                "position_flips": 0,
                "long_positions_count": len(executed_weights),
                "short_positions_count": 0,
                "long_stock_exposure": gross_exposure,
                "short_stock_exposure": 0.0,
                "net_stock_exposure": gross_exposure,
                "gross_stock_exposure": gross_exposure,
                "benchmark_short_weight": 0.0,
                "avg_position_weight": float(weight_values.mean()) if len(weight_values) else 0.0,
                "min_position_weight": float(weight_values.min()) if len(weight_values) else 0.0,
                "max_position_weight": float(weight_values.max()) if len(weight_values) else 0.0,
                "position_weight_variance": float(weight_values.var()) if len(weight_values) else 0.0,
                "stock_margin_requirement_fraction": gross_exposure,
                "benchmark_margin_requirement_fraction": 0.0,
                "total_margin_requirement_fraction": gross_exposure,
                "free_equity_fraction": max(0.0, 1.0 - gross_exposure),
                "margin_call_flag": 0,
                "rebalance_flag": int(ranked is not None),
            }
        )

    return simulation_rows, picks_rows


def summarize_results(
    simulation_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    benchmark_daily_df: pd.DataFrame | None = None,
):
    daily = simulation_df["daily_return"].astype(float)
    gross_daily = simulation_df["gross_daily_return"].astype(float)
    equity = simulation_df["portfolio_value"].astype(float)
    n = len(simulation_df)
    total_return = float(simulation_df["cumulative_return"].iloc[-1]) if n else 0.0
    gross_total_return = float((1.0 + gross_daily).prod() - 1.0) if n else 0.0
    annualized_return = float((1.0 + total_return) ** (252.0 / n) - 1.0) if n else 0.0
    annualized_volatility = float(daily.std(ddof=0) * (252.0 ** 0.5)) if n else 0.0
    position_sizes = picks_df["position_weight"].astype(float) if not picks_df.empty else pd.Series(dtype="float64")
    benchmark_total_return = 0.0
    benchmark_annualized_return = 0.0
    benchmark_sharpe = 0.0
    if benchmark_daily_df is not None and not benchmark_daily_df.empty:
        benchmark_daily = benchmark_daily_df["benchmark_daily_return"].astype(float)
        benchmark_total_return = float((1.0 + benchmark_daily).prod() - 1.0)
        benchmark_annualized_return = float((1.0 + benchmark_total_return) ** (252.0 / len(benchmark_daily_df)) - 1.0)
        benchmark_vol = float(benchmark_daily.std(ddof=0) * (252.0 ** 0.5))
        benchmark_sharpe = float(benchmark_annualized_return / benchmark_vol) if benchmark_vol > 0 else 0.0
    if "benchmark_commission_cost_fraction" in simulation_df.columns:
        extra_commission = pd.to_numeric(
            simulation_df["benchmark_commission_cost_fraction"],
            errors="coerce",
        ).fillna(0.0)
    else:
        extra_commission = pd.Series(0.0, index=simulation_df.index, dtype="float64")
    return {
        "trading_days": n,
        "total_return": total_return,
        "gross_total_return": gross_total_return,
        "slippage_impact": gross_total_return - total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(annualized_return / annualized_volatility) if annualized_volatility > 0 else 0.0,
        "max_drawdown": float(((equity / equity.cummax()) - 1.0).min()) if n else 0.0,
        "win_rate": float((daily > 0).mean()) if n else 0.0,
        "min_holdings": int(simulation_df["holdings_count"].min()) if n else 0,
        "max_holdings": int(simulation_df["holdings_count"].max()) if n else 0,
        "avg_holdings": float(simulation_df["holdings_count"].mean()) if n else 0.0,
        "total_buys": int(simulation_df["buys"].sum()) if n else 0,
        "total_sells": int(simulation_df["sells"].sum()) if n else 0,
        "total_trades": int(simulation_df["buys"].sum() + simulation_df["sells"].sum()) if n else 0,
        "trade_days": int(((simulation_df["buys"] + simulation_df["sells"]) > 0).sum()) if n else 0,
        "avg_daily_turnover": float(simulation_df["turnover_fraction"].mean()) if n else 0.0,
        "avg_slippage_cost_fraction": 0.0,
        "avg_commission_cost_fraction": float(
            (simulation_df["commission_cost_fraction"].astype(float) + extra_commission).mean()
        ) if n else 0.0,
        "avg_overnight_funding_cost_fraction": 0.0,
        "avg_short_borrow_cost_fraction": 0.0,
        "avg_total_cost_fraction": float(simulation_df["total_cost_fraction"].astype(float).mean()) if n else 0.0,
        "best_day": float(daily.max()) if n else 0.0,
        "worst_day": float(daily.min()) if n else 0.0,
        "missing_return_days": int((simulation_df["missing_returns_count"] > 0).sum()) if n else 0,
        "avg_position_weight": float(position_sizes.mean()) if not position_sizes.empty else 0.0,
        "min_position_weight": float(position_sizes.min()) if not position_sizes.empty else 0.0,
        "max_position_weight": float(position_sizes.max()) if not position_sizes.empty else 0.0,
        "position_weight_variance": float(position_sizes.var(ddof=0)) if not position_sizes.empty else 0.0,
        "avg_daily_max_position_weight": float(simulation_df["max_position_weight"].mean()) if n else 0.0,
        "avg_long_positions": float(simulation_df["long_positions_count"].mean()) if n else 0.0,
        "avg_short_positions": 0.0,
        "avg_gross_stock_exposure": float(simulation_df["gross_stock_exposure"].mean()) if n else 0.0,
        "avg_net_stock_exposure": float(simulation_df["net_stock_exposure"].mean()) if n else 0.0,
        "avg_long_stock_exposure": float(simulation_df["long_stock_exposure"].mean()) if n else 0.0,
        "avg_short_stock_exposure": 0.0,
        "avg_total_margin_requirement_fraction": float(simulation_df["total_margin_requirement_fraction"].mean()) if n else 0.0,
        "avg_free_equity_fraction": float(simulation_df["free_equity_fraction"].mean()) if n else 0.0,
        "margin_call_days": 0,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "benchmark_sharpe": benchmark_sharpe,
        "excess_total_return": total_return - benchmark_total_return,
        "excess_annualized_return": annualized_return - benchmark_annualized_return,
    }


def load_or_fetch_benchmark(symbol: str, cache_csv: Path, start_date, end_date) -> pd.DataFrame:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_csv.parent.resolve()))
    if cache_csv.exists():
        cache = pd.read_csv(cache_csv, parse_dates=["date"])
        if cache["date"].dt.tz is not None:
            cache["date"] = cache["date"].dt.tz_localize(None)
        cache["date"] = cache["date"].dt.normalize()
        cache["close"] = pd.to_numeric(cache["close"], errors="coerce")
        cache = cache.dropna(subset=["date", "close"]).copy()
    else:
        cache = pd.DataFrame()
    needs_fetch = cache.empty or not (
        cache["date"].min() <= pd.Timestamp(start_date)
        and cache["date"].max() >= (pd.Timestamp(end_date) - pd.Timedelta(days=7))
    )
    if needs_fetch:
        try:
            hist = yf.Ticker(symbol).history(
                start=(pd.Timestamp(start_date) - pd.Timedelta(days=10)).date().isoformat(),
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=5)).date().isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
            ).reset_index()
            cache = hist.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]].copy()
            if cache["date"].dt.tz is not None:
                cache["date"] = cache["date"].dt.tz_localize(None)
            cache["date"] = cache["date"].dt.normalize()
            cache["close"] = pd.to_numeric(cache["close"], errors="coerce")
            cache = cache.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
            cache.to_csv(cache_csv, index=False)
        except Exception:
            if cache.empty:
                raise
    return cache[(cache["date"] >= pd.Timestamp(start_date)) & (cache["date"] <= pd.Timestamp(end_date))].copy()


def build_benchmark_daily(simulation_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    merged = simulation_df[
        [
            "date",
            "portfolio_value",
            "gross_stock_return",
            "daily_return",
            "long_stock_return",
            "long_stock_exposure",
            "total_cost_fraction",
        ]
    ].copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.merge(
        prepare_benchmark_frame(benchmark_df)[["date", "next_benchmark_return"]],
        on="date",
        how="left",
    ).sort_values("date").reset_index(drop=True)
    merged["benchmark_daily_return"] = merged["next_benchmark_return"].fillna(0.0).astype(float)
    merged["stock_gross_return"] = pd.to_numeric(merged["gross_stock_return"], errors="coerce").fillna(0.0)
    merged["stock_total_cost_fraction"] = pd.to_numeric(merged["total_cost_fraction"], errors="coerce").fillna(0.0)
    long_exposure = pd.to_numeric(merged["long_stock_exposure"], errors="coerce").fillna(0.0)
    merged["long_book_daily_return"] = np.where(
        long_exposure > 0.0,
        pd.to_numeric(merged["long_stock_return"], errors="coerce").fillna(0.0) / long_exposure,
        0.0,
    )
    trailing_long = merged["long_book_daily_return"].shift(1)
    trailing_benchmark = merged["benchmark_daily_return"].shift(1)
    rolling_cov = trailing_long.rolling(
        window=PORTFOLIO.hedge_beta_window_days,
        min_periods=PORTFOLIO.hedge_beta_min_obs,
    ).cov(trailing_benchmark)
    rolling_var = trailing_benchmark.rolling(
        window=PORTFOLIO.hedge_beta_window_days,
        min_periods=PORTFOLIO.hedge_beta_min_obs,
    ).var()
    portfolio_beta = (rolling_cov / rolling_var.replace(0.0, np.nan)).clip(lower=0.0).fillna(1.0)
    portfolio_beta = np.where(long_exposure > 0.0, portfolio_beta, 0.0)
    merged["inverse_etf_weight"] = (portfolio_beta * long_exposure / INVERSE_ETF_LEVERAGE).clip(lower=0.0, upper=1.0)
    merged["stock_capital_weight"] = (1.0 - merged["inverse_etf_weight"]).clip(lower=0.0, upper=1.0)
    merged["inverse_etf_daily_return"] = (-INVERSE_ETF_LEVERAGE * merged["benchmark_daily_return"]).clip(lower=-1.0)

    previous_inverse_weight = 0.0
    strategy_value = float(PORTFOLIO.initial_capital)
    starting_equity = []
    strategy_returns = []
    strategy_values = []
    hedge_trade_flags = []
    hedge_commissions = []
    for row in merged.itertuples(index=False):
        current_start = float(strategy_value)
        target_inverse_weight = float(row.inverse_etf_weight)
        hedge_commission_fraction = trade_commission_fraction(
            abs(target_inverse_weight - previous_inverse_weight) * current_start,
            current_start,
        )
        combined_daily_return = (
            float(row.stock_capital_weight) * float(row.stock_gross_return)
            - float(row.stock_total_cost_fraction)
            + float(target_inverse_weight) * float(row.inverse_etf_daily_return)
            - hedge_commission_fraction
        )
        strategy_value = max(0.0, current_start * (1.0 + combined_daily_return))
        starting_equity.append(current_start)
        strategy_returns.append(combined_daily_return)
        strategy_values.append(strategy_value)
        hedge_trade_flags.append(int(abs(target_inverse_weight - previous_inverse_weight) > 1e-12))
        hedge_commissions.append(hedge_commission_fraction)
        previous_inverse_weight = target_inverse_weight

    merged["starting_equity"] = starting_equity
    merged["inverse_etf_trade_flag"] = hedge_trade_flags
    merged["hedge_commission_cost_fraction"] = hedge_commissions
    merged["benchmark_commission_cost_fraction"] = merged["hedge_commission_cost_fraction"]
    merged["benchmark_overnight_funding_cost_fraction"] = 0.0
    merged["benchmark_short_borrow_cost_fraction"] = 0.0
    merged["daily_return"] = strategy_returns
    merged["combined_return"] = merged["daily_return"]
    merged["strategy_portfolio_value"] = strategy_values
    merged["hedged_equity"] = merged["strategy_portfolio_value"]
    merged["benchmark_portfolio_value"] = PORTFOLIO.initial_capital * (1.0 + merged["benchmark_daily_return"]).cumprod()
    merged["excess_daily_return"] = merged["daily_return"] - merged["benchmark_daily_return"]
    merged["inverse_etf_ticker"] = INVERSE_ETF_TICKER
    merged["test_mode"] = "stocks_plus_inverse_etf"
    return merged


def build_yearly_summary(simulation_df: pd.DataFrame, benchmark_daily_df: pd.DataFrame) -> pd.DataFrame:
    merged = benchmark_daily_df.copy()
    merged["year"] = pd.to_datetime(merged["date"]).dt.year
    sim_year = pd.to_datetime(simulation_df["date"]).dt.year
    rows = []
    for year, sample in merged.groupby("year", sort=True):
        strategy_equity = sample["strategy_portfolio_value"].astype(float)
        benchmark_equity = sample["benchmark_portfolio_value"].astype(float)
        rows.append(
            {
                "year": int(year),
                "trade_days": int(len(sample)),
                "strategy_return": float((1.0 + sample["daily_return"].astype(float)).prod() - 1.0),
                "benchmark_return": float((1.0 + sample["benchmark_daily_return"].astype(float)).prod() - 1.0),
                "excess_return": float(
                    (1.0 + sample["daily_return"].astype(float)).prod()
                    - (1.0 + sample["benchmark_daily_return"].astype(float)).prod()
                ),
                "avg_daily_turnover": float(simulation_df.loc[sim_year == year, "turnover_fraction"].astype(float).mean()),
                "strategy_max_drawdown": float(((strategy_equity / strategy_equity.cummax()) - 1.0).min()),
                "benchmark_max_drawdown": float(((benchmark_equity / benchmark_equity.cummax()) - 1.0).min()),
            }
        )
    return pd.DataFrame(rows)


def build_base_benchmark_daily(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    base_benchmark = prepare_benchmark_frame(benchmark_df)[["date", "next_benchmark_return"]].copy()
    base_benchmark["benchmark_daily_return"] = base_benchmark.pop("next_benchmark_return").fillna(0.0)
    return base_benchmark


def build_direct_benchmark_daily(simulation_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    strategy_daily = simulation_df[["date", "daily_return"]].copy()
    strategy_daily["date"] = pd.to_datetime(strategy_daily["date"], errors="coerce").dt.normalize()
    strategy_daily["daily_return"] = pd.to_numeric(strategy_daily["daily_return"], errors="coerce").fillna(0.0)
    benchmark_daily = build_base_benchmark_daily(benchmark_df)
    merged = strategy_daily.merge(benchmark_daily, on="date", how="left").sort_values("date").reset_index(drop=True)
    merged["benchmark_daily_return"] = merged["benchmark_daily_return"].fillna(0.0)
    merged["strategy_portfolio_value"] = PORTFOLIO.initial_capital * (1.0 + merged["daily_return"]).cumprod()
    merged["benchmark_portfolio_value"] = PORTFOLIO.initial_capital * (1.0 + merged["benchmark_daily_return"]).cumprod()
    merged["excess_daily_return"] = merged["daily_return"] - merged["benchmark_daily_return"]
    return merged


def _line_path(points):
    return f"M {points[0][0]:.2f} {points[0][1]:.2f}" + "".join(f" L {x:.2f} {y:.2f}" for x, y in points[1:])


def write_equity_curve_svg(
    curve_df,
    *,
    strategy_column: str,
    benchmark_column: str | None,
    title: str,
    strategy_label: str,
    benchmark_label: str | None,
    out_svg: Path,
):
    chart_columns = ["date", strategy_column]
    if benchmark_column is not None:
        chart_columns.append(benchmark_column)
    merged = curve_df[chart_columns].copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    merged["strategy_norm"] = pd.to_numeric(merged[strategy_column], errors="coerce").astype(float)
    value_columns = ["strategy_norm"]
    if benchmark_column is not None:
        merged["benchmark_norm"] = pd.to_numeric(merged[benchmark_column], errors="coerce").astype(float)
        value_columns.append("benchmark_norm")
    merged = merged.dropna(subset=value_columns).reset_index(drop=True)
    if merged.empty:
        raise ValueError("Cannot write an equity curve chart from an empty dataset.")

    width, height, left, right, top, bottom = 1200, 700, 80, 30, 30, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = float(merged[value_columns].min().min())
    y_max = float(merged[value_columns].max().max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.06
    y_min -= y_pad
    y_max += y_pad

    def sx(index):
        return left if len(merged) <= 1 else left + (index / (len(merged) - 1)) * plot_w

    def sy(value):
        return top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    strategy_pts = [(sx(i), sy(v)) for i, v in enumerate(merged["strategy_norm"].tolist())]
    benchmark_pts = (
        [(sx(i), sy(v)) for i, v in enumerate(merged["benchmark_norm"].tolist())]
        if benchmark_column is not None
        else None
    )
    date_start = merged["date"].iloc[0].date().isoformat()
    date_end = merged["date"].iloc[-1].date().isoformat()
    tick_lines = []
    tick_labels = []
    for i in range(7):
        val = y_min + (i / 6) * (y_max - y_min)
        y = sy(val)
        tick_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />')
        tick_labels.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#374151">{val:,.0f}</text>')
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left}" y="18" font-size="18" font-family="Arial" fill="#111827">{title} ({date_start} to {date_end})</text>',
        *tick_lines,
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#6b7280" stroke-width="1.2" />',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#6b7280" stroke-width="1.2" />',
        *tick_labels,
    ]
    if benchmark_pts is not None:
        svg_lines.append(f'<path d="{_line_path(benchmark_pts)}" fill="none" stroke="#2563eb" stroke-width="2.2" />')
    svg_lines.extend(
        [
            f'<path d="{_line_path(strategy_pts)}" fill="none" stroke="#dc2626" stroke-width="2.2" />',
            f'<text x="{left}" y="{height-45}" font-size="12" fill="#374151">Start: {date_start}</text>',
            f'<text x="{width-right}" y="{height-45}" text-anchor="end" font-size="12" fill="#374151">End: {date_end}</text>',
            f'<circle cx="{left+10}" cy="{height-20}" r="4" fill="#dc2626" />',
            f'<text x="{left+20}" y="{height-16}" font-size="12" fill="#111827">{strategy_label}</text>',
        ]
    )
    if benchmark_pts is not None and benchmark_label:
        svg_lines.extend(
            [
                f'<circle cx="{left+110}" cy="{height-20}" r="4" fill="#2563eb" />',
                f'<text x="{left+120}" y="{height-16}" font-size="12" fill="#111827">{benchmark_label}</text>',
            ]
        )
    svg_lines.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg_lines), encoding="utf-8")


def save_mode_results(
    mode_dir: Path,
    *,
    metric_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    period_summary_df: pd.DataFrame,
    yearly_summary_df: pd.DataFrame,
    benchmark_daily_df: pd.DataFrame,
    benchmark_test_df: pd.DataFrame,
    chart_title: str,
    benchmark_label: str | None,
    chart_source_df=None,
    chart_strategy_column: str = "strategy_portfolio_value",
    chart_benchmark_column: str | None = "benchmark_portfolio_value",
    chart_strategy_label: str = "Strategy",
):
    mode_dir.mkdir(parents=True, exist_ok=True)
    metric_path = mode_dir / "metric_return_correlations.csv"
    picks_path = mode_dir / "strategy_daily_top20.csv"
    simulation_path = mode_dir / "strategy_backtest_simulation.csv"
    periods_path = mode_dir / "strategy_walkforward_periods.csv"
    yearly_path = mode_dir / "strategy_yearly_summary.csv"
    benchmark_daily_path = mode_dir / BENCHMARKS.benchmark_daily_file
    benchmark_test_path = mode_dir / BENCHMARKS.benchmark_test_file
    chart_path = mode_dir / BENCHMARKS.chart_file
    metric_df.to_csv(metric_path, index=False)
    picks_df.to_csv(picks_path, index=False)
    simulation_df.to_csv(simulation_path, index=False)
    period_summary_df.to_csv(periods_path, index=False)
    yearly_summary_df.to_csv(yearly_path, index=False)
    benchmark_daily_df.to_csv(benchmark_daily_path, index=False)
    benchmark_test_df.to_csv(benchmark_test_path, index=False)
    write_equity_curve_svg(
        benchmark_daily_df if chart_source_df is None else chart_source_df,
        strategy_column=chart_strategy_column,
        benchmark_column=chart_benchmark_column,
        title=chart_title,
        strategy_label=chart_strategy_label,
        benchmark_label=benchmark_label,
        out_svg=chart_path,
    )
    return {
        "metric_return_correlations": metric_path,
        "strategy_daily_top20": picks_path,
        "strategy_backtest_simulation": simulation_path,
        "strategy_walkforward_periods": periods_path,
        "strategy_yearly_summary": yearly_path,
        "strategy_vs_benchmark_daily": benchmark_daily_path,
        "benchmark_test": benchmark_test_path,
        "strategy_vs_benchmark_chart": chart_path,
    }


def execute_metric_model(
    *,
    run_direct_benchmark_test: bool = False,
    run_hedged_short_benchmark_test: bool = False,
) -> MetricRunResult:
    if not run_direct_benchmark_test and not run_hedged_short_benchmark_test:
        raise SystemExit("Enable at least one benchmark test to save results.")

    input_csv = PATHS.input_csv.resolve()
    output_root = PATHS.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    metric_df, test_df, period_summary_df, ranked_lookup = prepare_strategy_data(df)
    simulation_rows, picks_rows = run_backtest(test_df, ranked_lookup, stock_side_mode="long_only")
    simulation_df = pd.DataFrame(simulation_rows)
    picks_df = pd.DataFrame(picks_rows)
    if simulation_df.empty:
        raise SystemExit("The monthly momentum strategy produced no simulation rows.")

    test_start_date = pd.to_datetime(simulation_df["date"]).min()
    test_end_date = pd.to_datetime(simulation_df["date"]).max()
    benchmark_series = load_or_fetch_benchmark(
        BENCHMARKS.symbol,
        PATHS.benchmark_cache_csv.resolve(),
        test_start_date,
        test_end_date,
    )

    saved_paths = {}
    mode_summaries = {}
    if run_direct_benchmark_test:
        direct_benchmark_daily_df = build_direct_benchmark_daily(simulation_df, benchmark_series)
        direct_benchmark_test_df = build_direct_benchmark_test(
            simulation_df[["date", "daily_return"]],
            build_base_benchmark_daily(benchmark_series),
            initial_capital=PORTFOLIO.initial_capital,
        )
        direct_yearly_summary_df = build_yearly_summary(simulation_df, direct_benchmark_daily_df)
        saved_paths[BENCHMARKS.direct_results_dir] = save_mode_results(
            output_root / BENCHMARKS.direct_results_dir,
            metric_df=metric_df,
            picks_df=picks_df,
            simulation_df=simulation_df,
            period_summary_df=period_summary_df,
            yearly_summary_df=direct_yearly_summary_df,
            benchmark_daily_df=direct_benchmark_daily_df,
            benchmark_test_df=direct_benchmark_test_df,
            chart_title="Monthly 6-Month Momentum Strategy vs ASX200",
            benchmark_label="ASX200",
        )
        mode_summaries[BENCHMARKS.direct_results_dir] = summarize_results(
            simulation_df,
            picks_df,
            direct_benchmark_daily_df,
        )

    if run_hedged_short_benchmark_test:
        hedged_benchmark_daily_df = build_benchmark_daily(simulation_df, benchmark_series)
        hedged_benchmark_test_df = hedged_benchmark_daily_df.copy()
        hedged_yearly_summary_df = build_yearly_summary(simulation_df, hedged_benchmark_daily_df)
        hedged_summary_df = simulation_df.copy()
        hedged_summary_df["daily_return"] = hedged_benchmark_daily_df["daily_return"].to_numpy(dtype="float64")
        hedged_summary_df["gross_daily_return"] = (
            hedged_benchmark_daily_df["stock_capital_weight"].to_numpy(dtype="float64")
            * hedged_benchmark_daily_df["stock_gross_return"].to_numpy(dtype="float64")
        ) + (
            hedged_benchmark_daily_df["inverse_etf_weight"].to_numpy(dtype="float64")
            * hedged_benchmark_daily_df["inverse_etf_daily_return"].to_numpy(dtype="float64")
        )
        hedged_summary_df["portfolio_value"] = hedged_benchmark_daily_df["strategy_portfolio_value"].to_numpy(dtype="float64")
        hedged_summary_df["cumulative_return"] = (
            hedged_summary_df["portfolio_value"].astype(float) / float(PORTFOLIO.initial_capital)
        ) - 1.0
        hedged_summary_df["total_cost_fraction"] = (
            simulation_df["total_cost_fraction"].to_numpy(dtype="float64")
            + hedged_benchmark_daily_df["benchmark_commission_cost_fraction"].to_numpy(dtype="float64")
        )
        hedged_summary_df["commission_cost_fraction"] = simulation_df["commission_cost_fraction"].to_numpy(dtype="float64")
        hedged_summary_df["benchmark_commission_cost_fraction"] = hedged_benchmark_daily_df[
            "benchmark_commission_cost_fraction"
        ].to_numpy(dtype="float64")
        hedged_summary_df["benchmark_overnight_funding_cost_fraction"] = 0.0
        hedged_summary_df["benchmark_short_borrow_cost_fraction"] = 0.0
        saved_paths[BENCHMARKS.hedged_results_dir] = save_mode_results(
            output_root / BENCHMARKS.hedged_results_dir,
            metric_df=metric_df,
            picks_df=picks_df,
            simulation_df=hedged_summary_df,
            period_summary_df=period_summary_df,
            yearly_summary_df=hedged_yearly_summary_df,
            benchmark_daily_df=hedged_benchmark_daily_df,
            benchmark_test_df=hedged_benchmark_test_df,
            chart_title="Monthly 6-Month Momentum + 3x Inverse ETF",
            benchmark_label=None,
            chart_source_df=hedged_benchmark_daily_df,
            chart_strategy_column="hedged_equity",
            chart_benchmark_column=None,
            chart_strategy_label="Momentum + Inverse ETF",
        )
        mode_summaries[BENCHMARKS.hedged_results_dir] = summarize_results(
            hedged_summary_df,
            picks_df,
            hedged_benchmark_daily_df,
        )

    summary_mode = BENCHMARKS.direct_results_dir if run_direct_benchmark_test else BENCHMARKS.hedged_results_dir
    summary = mode_summaries[summary_mode]
    average_rank_ic = float(pd.to_numeric(period_summary_df["test_rank_ic"], errors="coerce").mean())
    print("Backtest completed.")
    print(f"Input: {input_csv}")
    print(f"Monthly rebalance periods: {len(period_summary_df)}")
    print(f"Results root: {output_root}")
    print(
        "Benchmark tests: "
        f"direct={run_direct_benchmark_test}, hedged_short_benchmark={run_hedged_short_benchmark_test}"
    )
    print(f"Average monthly rank IC: {average_rank_ic:.6f}")
    print(f"Total return: {summary['total_return']:.4%}")
    print(f"Benchmark total return: {summary['benchmark_total_return']:.4%}")
    print(f"Excess total return: {summary['excess_total_return']:.4%}")
    print(f"Average position size: {summary['avg_position_weight']:.4%}")
    print(f"Minimum position size: {summary['min_position_weight']:.4%}")
    print(f"Maximum position size: {summary['max_position_weight']:.4%}")
    print(f"Average daily turnover: {summary['avg_daily_turnover']:.4%}")
    print(f"Average long positions: {summary['avg_long_positions']:.2f}")
    print(f"Average daily commission cost: {summary['avg_commission_cost_fraction']:.4%}")
    for mode_name, mode_paths in saved_paths.items():
        print(f"Saved [{mode_name}]: {output_root / mode_name}")
        for path in mode_paths.values():
            print(f"Saved: {path}")
    return MetricRunResult(output_dir=output_root, summary=mode_summaries, saved_paths=saved_paths)


def main():
    execute_metric_model()


if __name__ == "__main__":
    main()
