from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.rsi_momentum_bot import RSIBuySignal
from models.long_term_trend import (
    LONG_TERM_MIN_HOLDING_DAYS,
    LongTermTrendSignal,
    generate_long_term_signals,
)
from scripts.run_single_stock import (
    MAX_TICKER_EXPOSURE_PCT,
    calculate_performance_metrics,
    simple_backtest,
)


class RSIIndicatorTests(unittest.TestCase):
    def test_rsi_is_100_when_prices_only_rise(self) -> None:
        prices = pd.Series(np.arange(1.0, 40.0))
        rsi = RSIBuySignal._calculate_rsi(prices)
        self.assertEqual(float(rsi.iloc[-1]), 100.0)

    def test_warmup_rows_remain_ineligible(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        close = pd.Series(np.linspace(10.0, 15.0, len(dates)))
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": "TEST",
                "close": close,
                "dollar_volume": 1_000_000.0,
            }
        )
        features = RSIBuySignal().prepare_features(frame)
        self.assertFalse(bool(features.loc[100, "warmup_ready"]))
        self.assertTrue(bool(features.iloc[-1]["warmup_ready"]))
        self.assertTrue(pd.isna(features.loc[10, "momentum_3m"]))


class BacktestTests(unittest.TestCase):
    def _signals(self) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=35)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "close": 10.0,
                "buy_signal": 0.0,
                "sell_signal": 0.0,
                "risk_exit_signal": False,
                "volatility": 0.01,
                "range_upper": 0.05,
                "range_lower": -0.05,
                "current_return": 0.0,
            }
        )
        frame.loc[1, "buy_signal"] = 1.0
        frame.loc[25, "sell_signal"] = 1.0
        return frame

    def test_costs_and_exposure_cap_are_applied(self) -> None:
        result = simple_backtest(self._signals())
        buy = next(trade for trade in result["trades"] if trade["action"] == "BUY")
        self.assertGreater(buy["commission"], 0.0)
        self.assertGreater(buy["price"], 10.0)
        self.assertLessEqual(
            buy["value"] / result["initial_capital"],
            MAX_TICKER_EXPOSURE_PCT,
        )
        self.assertGreater(result["total_fees"], 0.0)

    def test_drawdown_uses_running_peak(self) -> None:
        history = [
            {"portfolio_value": 100.0, "shares": 1},
            {"portfolio_value": 80.0, "shares": 1},
            {"portfolio_value": 120.0, "shares": 1},
            {"portfolio_value": 90.0, "shares": 1},
        ]
        metrics = calculate_performance_metrics(history, [], [], 100.0)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 25.0)

    def test_immaterial_exposure_top_up_is_skipped(self) -> None:
        frame = self._signals()
        frame.loc[5, "buy_signal"] = 1.0
        result = simple_backtest(frame, buy_cooldown_days=1)
        buys = [trade for trade in result["trades"] if trade["action"] == "BUY"]
        self.assertEqual(len(buys), 1)

    def test_profitable_mature_lot_exits_without_indicator_sell(self) -> None:
        frame = self._signals()
        frame["sell_signal"] = 0.0
        frame.loc[21:, ["open", "close"]] = 11.0

        result = simple_backtest(frame)
        sells = [trade for trade in result["trades"] if trade["action"] == "SELL"]

        self.assertEqual(len(sells), 1)
        self.assertEqual(pd.Timestamp(sells[0]["date"]), frame.loc[21, "date"])
        self.assertIn("profit-first exit outside indicator bands", sells[0]["sizing_reason"])
        self.assertEqual(result["win_rate_pct"], 100.0)

    def test_original_sell_signal_remains_loss_fallback(self) -> None:
        frame = self._signals()
        frame.loc[25:, ["open", "close"]] = 9.0

        result = simple_backtest(frame)
        sells = [trade for trade in result["trades"] if trade["action"] == "SELL"]

        self.assertEqual(len(sells), 1)
        self.assertEqual(pd.Timestamp(sells[0]["date"]), frame.loc[25, "date"])
        self.assertIn("original exit signal", sells[0]["sizing_reason"])
        self.assertEqual(result["win_rate_pct"], 0.0)


class LongTermStrategyTests(unittest.TestCase):
    def test_features_and_signals_are_lagged(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=360)
        close = np.concatenate(
            [
                np.linspace(10.0, 13.0, 300),
                np.linspace(13.0, 12.4, 20),
                np.linspace(12.4, 14.5, 40),
            ]
        )
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": "TEST",
                "open": close,
                "close": close,
                "dollar_volume": 1_000_000.0,
            }
        )
        features = LongTermTrendSignal().generate_signals(frame)
        signals = generate_long_term_signals(features)
        self.assertTrue(bool(signals.iloc[-1]["long_term_warmup_ready"]))
        expected = signals["raw_long_term_buy"].shift(1).fillna(False).astype(float)
        pd.testing.assert_series_equal(
            signals["buy_signal"],
            expected,
            check_names=False,
        )

    def test_long_term_minimum_hold_is_configurable(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=150)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "close": 10.0,
                "buy_signal": 0.0,
                "sell_signal": 0.0,
                "risk_exit_signal": False,
                "volatility": 0.01,
                "range_upper": 0.05,
                "range_lower": -0.05,
                "current_return": 0.0,
            }
        )
        frame.loc[1, "buy_signal"] = 1.0
        frame.loc[30, "sell_signal"] = 1.0
        frame.loc[130, "sell_signal"] = 1.0
        result = simple_backtest(
            frame,
            minimum_holding_days=LONG_TERM_MIN_HOLDING_DAYS,
        )
        sells = [trade for trade in result["trades"] if trade["action"] == "SELL"]
        self.assertEqual(len(sells), 1)
        self.assertEqual(pd.Timestamp(sells[0]["date"]), dates[130])


if __name__ == "__main__":
    unittest.main()
