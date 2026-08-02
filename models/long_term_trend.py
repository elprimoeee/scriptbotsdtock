#!/usr/bin/env python3
"""Slow-moving trend strategy intended for multi-year stock ownership."""

from __future__ import annotations

import pandas as pd

from models.rsi_momentum_bot import RSIBuySignal, SIGNAL_LAG_DAYS


LONG_TERM_MIN_HOLDING_DAYS = 126  # approximately six trading months
LONG_TERM_BUY_COOLDOWN_DAYS = 63
LONG_TERM_FAST_MA_DAYS = 50
LONG_TERM_SLOW_MA_DAYS = 200
LONG_TERM_TREND_SLOPE_DAYS = 21
LONG_TERM_MAX_DRAWDOWN_FROM_HIGH = -0.30
LONG_TERM_HARD_STOP_VS_200D = -0.15


class LongTermTrendSignal(RSIBuySignal):
    """Buy durable uptrends and exit only when the long-term trend breaks."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Long-Term Trend"
        self.description = (
            "Long-term trend following using 50/200-day averages, annual "
            "momentum, controlled drawdown, and confirmed structural exits"
        )

    def prepare_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare_features(dataset)
        grouped_close = df.groupby("ticker")["close"]
        df["ma_50d"] = grouped_close.transform(
            lambda values: values.rolling(LONG_TERM_FAST_MA_DAYS).mean()
        )
        df["ma_200d_slope_1m"] = df.groupby("ticker")["ma_200d"].transform(
            lambda values: values / values.shift(LONG_TERM_TREND_SLOPE_DAYS) - 1.0
        )
        df["high_252d"] = grouped_close.transform(
            lambda values: values.rolling(252).max()
        )
        df["drawdown_from_1y_high"] = df["close"] / df["high_252d"] - 1.0
        df["long_term_warmup_ready"] = (
            df["warmup_ready"]
            & df[
                [
                    "ma_50d",
                    "ma_200d_slope_1m",
                    "high_252d",
                    "drawdown_from_1y_high",
                ]
            ].notna().all(axis=1)
        )
        return df

    def score(self, dataset: pd.DataFrame) -> pd.Series:
        """Rank persistent returns while penalising large peak drawdowns."""
        annual_momentum = dataset["return_1y"].clip(-0.50, 1.00)
        trend_strength = dataset["price_vs_200d_ma"].clip(-0.25, 0.50)
        drawdown_quality = (1.0 + dataset["drawdown_from_1y_high"]).clip(0.0, 1.0)
        return (
            annual_momentum * 0.45
            + trend_strength * 0.35
            + drawdown_quality * 0.20
        )


def generate_long_term_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate one-day-lagged entries and slow structural exits."""
    frames: list[pd.DataFrame] = []
    ordered = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for _, ticker_data in ordered.groupby("ticker", sort=False):
        ticker_data = ticker_data.copy().reset_index(drop=True)
        market_regime = ticker_data.get(
            "market_regime_ok",
            pd.Series(True, index=ticker_data.index),
        ).fillna(False)
        durable_trend = (
            ticker_data["long_term_warmup_ready"]
            & (ticker_data["close"] > ticker_data["ma_200d"])
            & (ticker_data["ma_50d"] > ticker_data["ma_200d"])
            & (ticker_data["ma_200d_slope_1m"] > 0.0)
            & (ticker_data["return_1y"] > 0.0)
            & (
                ticker_data["drawdown_from_1y_high"]
                >= LONG_TERM_MAX_DRAWDOWN_FROM_HIGH
            )
            & market_regime
        )
        entry_quality = ticker_data["rsi"].between(40.0, 70.0)
        previous_durable_trend = (
            durable_trend.shift(1).fillna(False).astype(bool)
        )
        new_uptrend = durable_trend & ~previous_durable_trend & entry_quality
        pullback_recovery = (
            durable_trend
            & ticker_data["rsi"].between(40.0, 65.0)
            & (ticker_data["close"].shift(1) <= ticker_data["ma_50d"].shift(1))
            & (ticker_data["close"] > ticker_data["ma_50d"])
        )

        below_slow_average = ticker_data["close"] < ticker_data["ma_200d"]
        confirmed_breakdown = (
            below_slow_average
            & below_slow_average.shift(5).fillna(False).astype(bool)
            & (ticker_data["ma_50d"] < ticker_data["ma_200d"])
            & (ticker_data["ma_200d_slope_1m"] < 0.0)
        )
        hard_risk_break = (
            ticker_data["price_vs_200d_ma"] < LONG_TERM_HARD_STOP_VS_200D
        )
        confirmed_breakdown_event = (
            confirmed_breakdown
            & ~confirmed_breakdown.shift(1).fillna(False).astype(bool)
        )
        hard_risk_break_event = (
            hard_risk_break
            & ~hard_risk_break.shift(1).fillna(False).astype(bool)
        )

        ticker_data["long_term_trend_qualified"] = durable_trend
        raw_buy_signal = new_uptrend | pullback_recovery
        ticker_data["raw_long_term_buy"] = raw_buy_signal
        ticker_data["raw_long_term_sell"] = (
            confirmed_breakdown_event | hard_risk_break_event
        )
        ticker_data["risk_exit_signal"] = hard_risk_break_event.shift(
            SIGNAL_LAG_DAYS
        ).fillna(False).astype(bool)
        ticker_data["buy_signal"] = (
            raw_buy_signal
            .shift(SIGNAL_LAG_DAYS)
            .fillna(False)
            .astype(float)
        )
        ticker_data["sell_signal"] = (
            ticker_data["raw_long_term_sell"]
            .shift(SIGNAL_LAG_DAYS)
            .fillna(False)
            .astype(float)
        )
        frames.append(ticker_data)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
