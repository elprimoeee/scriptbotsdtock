#!/usr/bin/env python3
"""Compare point-in-time selector and market-adaptive strategy variants.

All signals use information available by the previous close and trade at the
next session open. Results are research output only; this script places no orders.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sp500_selector import rank_sp500_stocks
from project_config import PORTFOLIO
from scripts.run_sp500_sharadar_backtest import TerminalProgress, read_dataset


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Variant:
    name: str
    rebalance_days: int
    method: str
    parameter: float = 0.0


VARIANTS = (
    Variant("baseline_6m", 126, "composite"),
    Variant("monthly_composite", 21, "composite"),
    Variant("momentum_12_1", 21, "momentum"),
    Variant("trend_filter", 21, "trend"),
    Variant("mean_reversion_rsi30", 21, "mean_reversion", 30.0),
    Variant("mean_reversion_rsi35", 21, "mean_reversion", 35.0),
    Variant("mean_reversion_rsi40", 21, "mean_reversion", 40.0),
    Variant("rank_persistence_30", 21, "rank_persistence", 30.0),
    Variant("rank_persistence_50", 21, "rank_persistence", 50.0),
    Variant("rank_persistence_75", 21, "rank_persistence", 75.0),
    Variant("risk_overlay_15pct", 126, "risk_overlay", 0.15),
    Variant("risk_overlay_20pct", 126, "risk_overlay", 0.20),
    Variant("risk_overlay_25pct", 126, "risk_overlay", 0.25),
    Variant("regime_adaptive_40pct", 21, "regime_adaptive", 0.40),
    Variant("regime_adaptive_65pct", 21, "regime_adaptive", 0.65),
    Variant("regime_adaptive_100pct", 21, "regime_adaptive", 1.00),
    Variant("performance_adaptive_252d", 21, "performance_adaptive", 252.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/sp500_daily_dataset_full.csv"),
    )
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("data/raw/benchmark/sp500_spy_sharadar_full.csv"),
    )
    parser.add_argument("--years", type=int, default=26)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--initial-capital", type=float, default=20_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0)
    parser.add_argument("--minimum-commission", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/adaptive_strategy_research"),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _market_features(benchmark: pd.DataFrame) -> pd.DataFrame:
    frame = benchmark[["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
    daily_return = frame["close"].pct_change(fill_method=None)
    frame["sma200"] = frame["close"].rolling(200, min_periods=200).mean()
    frame["return_63d"] = frame["close"].pct_change(63, fill_method=None)
    frame["volatility_63d"] = daily_return.rolling(63, min_periods=63).std() * np.sqrt(252.0)
    frame["volatility_limit"] = frame["volatility_63d"].rolling(
        756, min_periods=252
    ).quantile(0.75).shift(1)
    frame["risk_on"] = (
        (frame["close"] > frame["sma200"])
        & (frame["return_63d"] > 0.0)
        & (frame["volatility_63d"] <= frame["volatility_limit"])
    )
    # Rebalance-date signals are known at the previous close, then traded at open.
    frame["risk_on_for_open"] = frame["risk_on"].shift(1).fillna(False).astype(bool)
    frame["regime_for_open"] = np.where(frame["risk_on_for_open"], "risk_on", "risk_off")
    return frame


def _rsi_at_events(price_source: pd.DataFrame, event_dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = price_source[["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.replace(
        r"\.AX$", "", regex=True
    )
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "close"]).sort_values(
        ["ticker", "date"]
    )
    delta = frame.groupby("ticker", sort=False)["close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.groupby(frame["ticker"], sort=False).transform(
        lambda values: values.rolling(14, min_periods=14).mean()
    )
    average_loss = losses.groupby(frame["ticker"], sort=False).transform(
        lambda values: values.rolling(14, min_periods=14).mean()
    )
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(average_loss > 0.0, 100.0)
    rsi = rsi.where(average_gain > 0.0, 0.0)
    frame["rsi14_decision"] = rsi.groupby(frame["ticker"], sort=False).shift(1)
    events = frame[frame["date"].isin(event_dates)][
        ["date", "ticker", "rsi14_decision"]
    ]
    return events.drop_duplicates(["date", "ticker"], keep="last")


def _prepare_prices(price_source: pd.DataFrame) -> pd.DataFrame:
    frame = price_source[["date", "ticker", "open", "close", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.replace(
        r"\.AX$", "", regex=True
    )
    for column in ("open", "close", "adj_close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    adjustment = (frame["adj_close"] / frame["close"]).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)
    frame["execution_open"] = frame["open"] * adjustment
    frame["mark_close"] = frame["adj_close"].fillna(frame["close"])
    frame = frame.dropna(subset=["date", "ticker", "execution_open", "mark_close", "close"])
    return frame.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    ).sort_values(["date", "ticker"]).reset_index(drop=True)


def _add_adaptive_scores(ranked: pd.DataFrame) -> pd.DataFrame:
    frame = ranked.copy()
    on_weights = {
        "momentum_12m_ex_1m_rank": 0.35,
        "momentum_6m_rank": 0.25,
        "momentum_3m_rank": 0.15,
        "quality_rank": 0.10,
        "financial_strength_rank": 0.05,
        "value_rank": 0.05,
        "liquidity_rank": 0.05,
    }
    off_weights = {
        "quality_rank": 0.25,
        "financial_strength_rank": 0.25,
        "downside_risk_rank": 0.20,
        "value_rank": 0.15,
        "liquidity_rank": 0.10,
        "momentum_12m_ex_1m_rank": 0.05,
    }
    frame["adaptive_risk_on_score"] = sum(
        frame[column].fillna(0.5) * weight for column, weight in on_weights.items()
    )
    frame["adaptive_risk_off_score"] = sum(
        frame[column].fillna(0.5) * weight for column, weight in off_weights.items()
    )
    return frame


def _metrics(
    equity: pd.Series,
    benchmark_close: pd.Series,
    initial_value: float,
) -> dict[str, float]:
    equity = pd.to_numeric(equity, errors="coerce").dropna()
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    years = max((len(equity) - 1) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    total_return = float(equity.iloc[-1] / initial_value - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    daily_vol = float(returns.std(ddof=0))
    drawdown = equity / equity.cummax() - 1.0
    benchmark_close = benchmark_close.reindex(equity.index).ffill().dropna()
    benchmark_cagr = 0.0
    excess_cagr = cagr
    if len(benchmark_close) > 1:
        benchmark_years = max((len(benchmark_close) - 1) / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
        benchmark_cagr = float(
            (benchmark_close.iloc[-1] / benchmark_close.iloc[0]) ** (1.0 / benchmark_years) - 1.0
        )
        excess_cagr = cagr - benchmark_cagr
    return {
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "annual_volatility_pct": daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0,
        "sharpe_ratio": float(returns.mean() / daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_vol > 0.0 else 0.0,
        "max_drawdown_pct": abs(float(drawdown.min())) * 100.0,
        "benchmark_cagr_pct": benchmark_cagr * 100.0,
        "excess_cagr_pct": excess_cagr * 100.0,
    }


def _conditional_metrics(
    equity: pd.Series,
    risk_on: pd.Series,
    *,
    state: bool,
) -> dict[str, float]:
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    state_mask = risk_on.reindex(returns.index).fillna(False).astype(bool).eq(state)
    selected = returns.loc[state_mask]
    if selected.empty:
        return {"cagr_pct": 0.0, "sharpe_ratio": 0.0}
    volatility = float(selected.std(ddof=0))
    total = float((1.0 + selected).prod() - 1.0)
    cagr = (1.0 + total) ** (TRADING_DAYS_PER_YEAR / len(selected)) - 1.0
    sharpe = (
        float(selected.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if volatility > 0.0 else 0.0
    )
    return {"cagr_pct": cagr * 100.0, "sharpe_ratio": sharpe}


class PortfolioSimulation:
    """Small daily-close simulator shared by each strategy variant."""

    def __init__(
        self,
        variant: Variant,
        sessions: pd.DatetimeIndex,
        price_days: dict[pd.Timestamp, pd.DataFrame],
        rank_days: dict[pd.Timestamp, pd.DataFrame],
        regime_by_date: dict[pd.Timestamp, bool],
        final_price_dates: dict[str, pd.Timestamp],
        performance_choice_by_date: dict[pd.Timestamp, str] | None = None,
        *,
        top_n: int,
        initial_capital: float,
        commission_rate: float,
        minimum_commission: float,
        stop_pct: float,
    ) -> None:
        self.variant = variant
        self.sessions = sessions
        self.price_days = price_days
        self.rank_days = rank_days
        self.regime_by_date = regime_by_date
        self.final_price_dates = final_price_dates
        self.performance_choice_by_date = performance_choice_by_date or {}
        self.top_n = top_n
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.minimum_commission = minimum_commission
        self.stop_pct = stop_pct
        self.cash = float(initial_capital)
        self.positions: dict[str, int] = {}
        self.marks: dict[str, float] = {}
        self.latest_closes: dict[str, float] = {}
        self.peak_closes: dict[str, float] = {}
        self.equity_rows: list[dict[str, object]] = []
        self.trade_rows: list[dict[str, object]] = []
        self.gross_traded = 0.0
        self.estimated_slippage = 0.0
        self.total_commission = 0.0

    def _equity_at_previous_close(self) -> float:
        return self.cash + sum(
            shares * self.marks.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )

    def _commission(self, value: float) -> float:
        if value <= 0.0:
            return 0.0
        return max(self.minimum_commission, value * self.commission_rate)

    def _sell(
        self,
        ticker: str,
        day: pd.DataFrame,
        date: pd.Timestamp,
        *,
        at_close: bool = False,
        action: str = "SELL",
    ) -> None:
        if ticker not in self.positions or ticker not in day.index:
            return
        shares = self.positions.pop(ticker)
        base_price = float(
            day.loc[ticker, "mark_close"] if at_close else day.loc[ticker, "execution_open"]
        )
        fill = base_price * (1.0 - PORTFOLIO.slippage_per_trade)
        value = shares * fill
        commission = self._commission(value)
        self.cash += value - commission
        self.total_commission += commission
        self.gross_traded += shares * base_price
        self.estimated_slippage += shares * base_price * PORTFOLIO.slippage_per_trade
        self.trade_rows.append({
            "date": date, "strategy": self.variant.name, "action": action,
            "ticker": ticker, "shares": shares, "price": fill,
            "value": value, "commission": commission,
        })
        self.marks.pop(ticker, None)
        self.latest_closes.pop(ticker, None)
        self.peak_closes.pop(ticker, None)

    def _buy(
        self,
        ticker: str,
        value_budget: float,
        day: pd.DataFrame,
        date: pd.Timestamp,
    ) -> None:
        if ticker in self.positions or ticker not in day.index:
            return
        base_price = float(day.loc[ticker, "execution_open"])
        if not np.isfinite(base_price) or base_price <= 0.0:
            return
        fill = base_price * (1.0 + PORTFOLIO.slippage_per_trade)
        shares = max(0, int((value_budget - self.minimum_commission) / fill))
        while shares > 0:
            value = shares * fill
            commission = self._commission(value)
            if value + commission <= self.cash:
                break
            shares -= 1
        if shares <= 0:
            return
        value = shares * fill
        commission = self._commission(value)
        self.cash -= value + commission
        self.total_commission += commission
        self.gross_traded += shares * base_price
        self.estimated_slippage += shares * base_price * PORTFOLIO.slippage_per_trade
        self.positions[ticker] = shares
        self.trade_rows.append({
            "date": date, "strategy": self.variant.name, "action": "BUY",
            "ticker": ticker, "shares": shares, "price": fill,
            "value": value, "commission": commission,
        })

    def _targets(self, date: pd.Timestamp, excluded: set[str]) -> tuple[list[str], float]:
        frame = self.rank_days.get(date)
        if frame is None or frame.empty:
            return [], 1.0
        frame = frame[~frame["ticker"].isin(excluded)].copy()
        method = self.variant.method
        mean_reversion_threshold = self.variant.parameter
        if method == "performance_adaptive":
            chosen = self.performance_choice_by_date.get(date, "baseline_6m")
            method = {
                "baseline_6m": "composite",
                "monthly_composite": "composite",
                "momentum_12_1": "momentum",
                "trend_filter": "trend",
                "mean_reversion_rsi35": "mean_reversion",
            }.get(chosen, "composite")
            mean_reversion_threshold = 35.0
        exposure = 1.0

        if method == "momentum":
            frame = frame.sort_values(
                ["decision_momentum_12m_ex_1m", "selection_score"],
                ascending=False,
            )
        elif method == "trend":
            if not self.regime_by_date.get(date, False):
                return [], 0.0
            frame = frame[frame["decision_price_vs_200d_ma"] > 0.0]
            frame = frame.sort_values("selection_score", ascending=False)
        elif method == "mean_reversion":
            frame = frame[frame["rsi14_decision"] <= mean_reversion_threshold]
            frame = frame.sort_values("selection_score", ascending=False)
        elif method == "rank_persistence":
            current = [ticker for ticker in self.positions if ticker in set(frame["ticker"])]
            rank_map = frame.set_index("ticker")["selection_rank"]
            kept = [ticker for ticker in current if rank_map.get(ticker, np.inf) <= self.variant.parameter]
            additions = frame.loc[~frame["ticker"].isin(kept)].sort_values(
                "selection_rank", ascending=True
            )["ticker"].tolist()
            return (kept + additions)[: self.top_n], 1.0
        elif method == "regime_adaptive":
            risk_on = self.regime_by_date.get(date, False)
            score = "adaptive_risk_on_score" if risk_on else "adaptive_risk_off_score"
            exposure = 1.0 if risk_on else self.variant.parameter
            frame = frame.sort_values(score, ascending=False)
        else:
            frame = frame.sort_values("selection_score", ascending=False)

        return frame["ticker"].head(self.top_n).tolist(), exposure

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        scheduled = set(self.sessions[:: self.variant.rebalance_days])
        for date in self.sessions:
            date = pd.Timestamp(date)
            day = self.price_days.get(date)
            if day is None:
                day = pd.DataFrame()

            stopped: set[str] = set()
            if self.variant.method == "risk_overlay":
                for ticker in list(self.positions):
                    previous_close = self.latest_closes.get(ticker)
                    high_water = self.peak_closes.get(ticker)
                    if (
                    previous_close is not None and high_water is not None
                        and previous_close <= high_water * (1.0 - self.stop_pct)
                    ):
                        self._sell(ticker, day, date)
                        stopped.add(ticker)

            if date in scheduled and not day.empty:
                targets, exposure = self._targets(date, stopped)
                if self.variant.method == "rank_persistence":
                    desired = set(targets)
                    for ticker in list(self.positions):
                        if ticker not in desired:
                            self._sell(ticker, day, date)
                    new_targets = [ticker for ticker in targets if ticker not in self.positions]
                    equity = self._equity_at_previous_close()
                    budget = equity * exposure / max(len(targets), 1)
                    for ticker in new_targets:
                        self._buy(ticker, budget, day, date)
                else:
                    for ticker in list(self.positions):
                        self._sell(ticker, day, date)
                    equity = self.cash
                    target_budget = equity * exposure / max(len(targets), 1) if targets else 0.0
                    for ticker in targets:
                        self._buy(ticker, target_budget, day, date)

            if not day.empty and "mark_close" in day:
                for ticker in list(self.positions):
                    if ticker not in day.index:
                        continue
                    self.marks[ticker] = float(day.loc[ticker, "mark_close"])
                    current_close = float(day.loc[ticker, "close"])
                    self.latest_closes[ticker] = current_close
                    self.peak_closes[ticker] = max(
                        self.peak_closes.get(ticker, current_close), current_close
                    )
                # Realize delisted/acquired names on their last observed session,
                # matching the existing six-month backtest's terminal-price rule.
                for ticker in list(self.positions):
                    if (
                        self.final_price_dates.get(ticker) == date
                        and date != self.sessions[-1]
                    ):
                        self._sell(ticker, day, date, at_close=True, action="FORCED_EXIT")

            equity = self._equity_at_previous_close()
            self.equity_rows.append({
                "date": date,
                "equity": equity,
                "cash": self.cash,
                "positions_count": len(self.positions),
                "cash_pct": self.cash / equity * 100.0 if equity > 0.0 else 100.0,
            })

        daily = pd.DataFrame(self.equity_rows).set_index("date")
        trades = pd.DataFrame(self.trade_rows)
        return daily, trades, {
            "trade_count": len(trades),
            "gross_traded_value": self.gross_traded,
            "estimated_slippage_cost": self.estimated_slippage,
            "commission_cost": self.total_commission,
            "average_cash_pct": float(daily["cash_pct"].mean()),
            "ending_cash_pct": float(daily["cash_pct"].iloc[-1]),
        }


def _rolling_strategy_choices(
    sessions: pd.DatetimeIndex,
    simulation_rows: dict[str, pd.DataFrame],
    lookback_days: int,
) -> tuple[dict[pd.Timestamp, str], pd.DataFrame]:
    """Choose the highest trailing Sharpe strategy using prior-session returns only."""

    candidates = (
        "baseline_6m",
        "monthly_composite",
        "momentum_12_1",
        "trend_filter",
        "mean_reversion_rsi35",
    )
    returns = {
        name: simulation_rows[name]["equity"].pct_change(fill_method=None).fillna(0.0)
        for name in candidates
    }
    choices: dict[pd.Timestamp, str] = {}
    log_rows: list[dict[str, object]] = []
    for index, date in enumerate(sessions):
        start = max(1, index - lookback_days)
        sample_end = index  # excludes today's open-to-close return from the choice
        scores: dict[str, float] = {}
        if sample_end - start >= 63:
            for name in candidates:
                values = returns[name].iloc[start:sample_end]
                volatility = float(values.std(ddof=0))
                scores[name] = (
                    float(values.mean() / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
                    if volatility > 0.0 else -np.inf
                )
        selected = max(scores, key=scores.get) if scores else "baseline_6m"
        choices[pd.Timestamp(date)] = selected
        if index == 0 or index % 21 == 0:
            log_rows.append({
                "date": pd.Timestamp(date),
                "selected_strategy": selected,
                "lookback_days": lookback_days,
                "selection_sharpe": scores.get(selected),
            })
    return choices, pd.DataFrame(log_rows)


def _simulate_all(
    ranked: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    *,
    years: int,
    top_n: int,
    initial_capital: float,
    commission_rate: float,
    minimum_commission: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    latest = prices["date"].max()
    requested_start = latest - pd.DateOffset(years=years)
    sessions = pd.DatetimeIndex(
        prices.loc[prices["date"] >= requested_start, "date"].unique()
    ).sort_values()
    if sessions.empty:
        raise ValueError("No stock price sessions in requested period")
    final_price_dates = prices.groupby("ticker", sort=False)["date"].max().to_dict()
    prices = prices[prices["date"].isin(sessions)].copy()
    price_days = {
        pd.Timestamp(date): day.set_index("ticker")
        for date, day in prices.groupby("date", sort=False)
    }
    rank_days = {
        pd.Timestamp(date): day.sort_values("selection_rank")
        for date, day in ranked.groupby("date", sort=False)
    }
    market = market.set_index("date").reindex(sessions).ffill()
    regime_by_date = market["risk_on_for_open"].fillna(False).astype(bool).to_dict()
    simulation_rows: dict[str, pd.DataFrame] = {}
    trade_frames: list[pd.DataFrame] = []
    details: dict[str, dict[str, object]] = {}

    variants = [variant for variant in VARIANTS if variant.method != "performance_adaptive"]
    adaptive_variant = next(
        variant for variant in VARIANTS if variant.method == "performance_adaptive"
    )
    for variant in variants:
        stop_pct = variant.parameter if variant.method == "risk_overlay" else 0.0
        simulation = PortfolioSimulation(
            variant, sessions, price_days, rank_days, regime_by_date, final_price_dates,
            top_n=top_n,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            stop_pct=stop_pct,
        )
        daily, trades, extra = simulation.run()
        simulation_rows[variant.name] = daily
        if not trades.empty:
            trade_frames.append(trades)
        details[variant.name] = extra

    performance_choices, choice_log = _rolling_strategy_choices(
        sessions, simulation_rows, int(adaptive_variant.parameter)
    )
    simulation = PortfolioSimulation(
        adaptive_variant, sessions, price_days, rank_days, regime_by_date,
        final_price_dates, performance_choices,
        top_n=top_n,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        stop_pct=0.0,
    )
    daily, trades, extra = simulation.run()
    simulation_rows[adaptive_variant.name] = daily
    if not trades.empty:
        trade_frames.append(trades)
    details[adaptive_variant.name] = extra
    variants.append(adaptive_variant)

    benchmark_close = market["close"].copy()
    risk_on_states = market["risk_on_for_open"].fillna(False).astype(bool)
    recent_requested_start = latest - pd.DateOffset(years=2)
    recent_date = pd.Timestamp(sessions[sessions >= recent_requested_start][0])
    summary_rows: list[dict[str, object]] = []
    equity_output = pd.DataFrame(index=sessions)
    equity_output["benchmark_close"] = benchmark_close
    for variant in variants:
        daily = simulation_rows[variant.name]
        equity_output[variant.name] = daily["equity"]
        all_metrics = _metrics(daily["equity"], benchmark_close, initial_capital)
        recent_daily = daily.loc[daily.index >= recent_date]
        recent_metrics = _metrics(
            recent_daily["equity"], benchmark_close.loc[recent_daily.index],
            float(recent_daily["equity"].iloc[0]),
        )
        risk_on_metrics = _conditional_metrics(
            daily["equity"], risk_on_states, state=True
        )
        risk_off_metrics = _conditional_metrics(
            daily["equity"], risk_on_states, state=False
        )
        summary_rows.append({
            "strategy": variant.name,
            "method": variant.method,
            "rebalance_days": variant.rebalance_days,
            "parameter": variant.parameter,
            **{f"full_{key}": value for key, value in all_metrics.items()},
            **{f"recent_{key}": value for key, value in recent_metrics.items()},
            **{f"risk_on_{key}": value for key, value in risk_on_metrics.items()},
            **{f"risk_off_{key}": value for key, value in risk_off_metrics.items()},
            **details[variant.name],
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["recent_sharpe_ratio", "recent_max_drawdown_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    metadata = {
        "period_start": sessions[0].date().isoformat(),
        "period_end": sessions[-1].date().isoformat(),
        "recent_period_start": recent_date.date().isoformat(),
        "years_requested": years,
        "top_n": top_n,
        "initial_capital": initial_capital,
        "slippage_per_side_pct": PORTFOLIO.slippage_per_trade * 100.0,
        "commission_rate": commission_rate,
        "minimum_commission": minimum_commission,
        "strategies_tested": len(variants),
        "winner_by_recent_sharpe": str(summary.iloc[0]["strategy"]),
        "winner_recent_sharpe": float(summary.iloc[0]["recent_sharpe_ratio"]),
        "winner_recent_cagr_pct": float(summary.iloc[0]["recent_cagr_pct"]),
        "winner_recent_excess_cagr_pct": float(summary.iloc[0]["recent_excess_cagr_pct"]),
    }
    return summary, equity_output, {
        "metadata": metadata,
        "trades": trades,
        "strategy_choices": choice_log,
    }


def main() -> None:
    args = parse_args()
    if args.years < 3 or args.top < 1 or args.initial_capital <= 0.0:
        raise SystemExit("Use at least 3 years, a positive top count, and positive capital")
    if not args.input.exists() or not args.benchmark.exists():
        raise SystemExit("Both the Sharadar dataset and matching SPY benchmark must exist")

    progress = TerminalProgress(enabled=not args.no_progress)
    dataset = read_dataset(args.input, progress)
    benchmark = pd.read_csv(args.benchmark, usecols=["date", "close"])
    market = _market_features(benchmark)

    date_values = pd.to_datetime(dataset["date"], errors="coerce").dt.normalize()
    available_dates = pd.DatetimeIndex(date_values.dropna().unique()).sort_values()
    latest_price_date = pd.Timestamp(available_dates[-1])
    requested_start = latest_price_date - pd.DateOffset(years=args.years)
    simulation_sessions = available_dates[available_dates >= requested_start]
    event_dates = pd.DatetimeIndex(
        sorted(set(simulation_sessions[::21]).union({simulation_sessions[-1]}))
    )

    progress.update("Ranking monthly decision dates", 0, 1)
    ranked = rank_sp500_stocks(dataset, ranking_dates=event_dates)
    progress.update("Ranking monthly decision dates", 1, 1)
    if ranked.empty:
        raise SystemExit("The selector produced no ranked stocks for the requested period")

    price_source = dataset[["date", "ticker", "open", "close", "adj_close"]].copy()
    del dataset
    rsi = _rsi_at_events(price_source, event_dates)
    ranked = ranked.merge(rsi, on=["date", "ticker"], how="left")
    ranked = _add_adaptive_scores(ranked)
    prices = _prepare_prices(price_source)
    del price_source

    progress.update("Comparing strategy variants", 0, len(VARIANTS))
    # Each variant is simulated independently with the same history, point-in-time
    # rankings, next-open execution, 5 bp slippage per side, and requested commissions.
    comparison, equity, extra = _simulate_all(
        ranked,
        prices,
        market,
        years=args.years,
        top_n=args.top,
        initial_capital=args.initial_capital,
        commission_rate=args.commission_rate,
        minimum_commission=args.minimum_commission,
    )
    progress.update("Comparing strategy variants", len(VARIANTS), len(VARIANTS))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    equity.index.name = "date"
    equity.to_csv(output_dir / "daily_equity_comparison.csv")
    trades = extra["trades"]
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(output_dir / "trades.csv", index=False)
    choices = extra["strategy_choices"]
    if isinstance(choices, pd.DataFrame):
        choices.to_csv(output_dir / "strategy_switch_log.csv", index=False)

    current = market.dropna(subset=["close"]).iloc[-1]
    latest_market_date = pd.Timestamp(current["date"])
    current_state = "risk_on" if bool(current["risk_on"]) else "risk_off"
    metadata = extra["metadata"]
    metadata.update({
        "price_data_as_of": latest_price_date.date().isoformat(),
        "benchmark_data_as_of": latest_market_date.date().isoformat(),
        "latest_market_regime": current_state,
        "latest_spy_close": float(current["close"]),
        "latest_spy_sma200": float(current["sma200"])
        if pd.notna(current["sma200"]) else None,
        "latest_spy_return_63d_pct": float(current["return_63d"] * 100.0)
        if pd.notna(current["return_63d"]) else None,
        "latest_spy_volatility_63d_pct": float(current["volatility_63d"] * 100.0)
        if pd.notna(current["volatility_63d"]) else None,
        "limitations": [
            "Latest local price and benchmark data are through the dates reported above; this is not a live market feed.",
            "The recent two-year comparison was inspected while expanding variants; it is not an untouched holdout and selection may overfit this period.",
            "Taxes, FX, market impact, and live order execution are not modeled.",
            "Regime variants use predefined risk-on/risk-off factor weights and 40%, 65%, or 100% exposure in risk-off periods; weights are not fitted to the recent comparison window.",
            "Comparing 17 variants creates multiple-comparison risk, even though each rolling strategy choice uses only prior-session data.",
            "The rolling strategy switch chooses among prior strategy returns using the trailing 252-session Sharpe ratio and changes monthly.",
            "RSI strategies are constrained by the selector's normal liquidity, volatility, and 200-day drawdown eligibility filters.",
        ],
    })
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    display_columns = [
        "strategy", "full_cagr_pct", "full_sharpe_ratio", "full_max_drawdown_pct",
        "recent_cagr_pct", "recent_sharpe_ratio", "recent_max_drawdown_pct",
        "recent_excess_cagr_pct", "risk_on_cagr_pct", "risk_on_sharpe_ratio",
        "risk_off_cagr_pct", "risk_off_sharpe_ratio", "trade_count", "average_cash_pct",
    ]
    print(f"Price history: {metadata['period_start']} to {metadata['period_end']}")
    print(f"Recent comparison begins: {metadata['recent_period_start']}")
    print(f"Latest market regime ({metadata['benchmark_data_as_of']}): {current_state}")
    print(comparison[display_columns].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"Highest recent Sharpe among variants: {metadata['winner_by_recent_sharpe']}")
    print(f"Saved comparison: {(output_dir / 'strategy_comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
