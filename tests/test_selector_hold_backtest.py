from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from models.selector_hold_backtest import (
    _prepare_prices,
    build_rebalance_dates,
    run_selector_hold_backtest,
)


class SelectorHoldBacktestTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> pd.DataFrame:
        dates = pd.bdate_range("2022-01-03", periods=800)
        rows = []
        for offset, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
            raw = np.linspace(10.0 + offset, 18.0 + offset, len(dates))
            raw *= 1.0 + (0.01 + offset * 0.002) * np.sin(np.arange(len(dates)) / (9 + offset))
            adjustment = 1.05
            for date, close in zip(dates, raw):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "open": close * 0.998,
                        "close": close,
                        "adj_close": close * adjustment,
                        "dollar_volume": 5_000_000.0 + offset * 500_000.0,
                        "market_cap": (offset + 1) * 10_000_000_000.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_rebalance_dates_use_trading_sessions(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=300)
        rebalances = build_rebalance_dates(dates, 126)
        self.assertEqual(rebalances, [dates[0], dates[126], dates[252]])

    def test_adjusted_prices_are_used_for_execution_and_marks(self) -> None:
        prepared = _prepare_prices(self._dataset().head(1))
        row = prepared.iloc[0]
        self.assertAlmostEqual(row["mark_close"] / row["close"], 1.05)
        self.assertAlmostEqual(row["execution_open"] / row["open"], 1.05)

    def test_full_strategy_rebalances_and_writes_report(self) -> None:
        with TemporaryDirectory() as directory:
            source = self._dataset()
            dates = pd.DatetimeIndex(source["date"].drop_duplicates().sort_values())
            benchmark = pd.DataFrame(
                {"date": dates, "close": np.linspace(7_000.0, 8_000.0, len(dates))}
            )
            result = run_selector_hold_backtest(
                source,
                constituents=["AAA", "BBB", "CCC", "DDD"],
                years=2,
                top_n=2,
                holding_days=126,
                initial_capital=20_000.0,
                output_dir=Path(directory),
                benchmark=benchmark,
            )
            self.assertTrue(result.report_path.exists())
            self.assertGreater(len(result.rebalances), 1)
            self.assertGreater(len(result.trades), 0)
            self.assertEqual(result.summary["holding_trading_days"], 126)
            self.assertIn("benchmark_total_return_pct", result.summary)
            self.assertTrue(result.daily_equity["benchmark_equity"].notna().all())
            actions = set(result.trades["action"])
            self.assertEqual(actions, {"BUY", "SELL"})

    def test_position_is_realized_on_its_final_price_session(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=6)
        rows = []
        for ticker, ticker_dates in (("AAA", dates[:3]), ("BBB", dates)):
            for day in ticker_dates:
                rows.append({
                    "date": day, "ticker": ticker, "open": 10.0, "close": 10.0,
                    "adj_close": 10.0, "dollar_volume": 10_000_000.0,
                })
        source = pd.DataFrame(rows)

        def fixed_ranking(frame: pd.DataFrame) -> pd.DataFrame:
            ranked = frame.copy()
            ranked["selection_rank"] = ranked.groupby("date")["ticker"].rank(method="first").astype(int)
            ranked["membership_source"] = "test_point_in_time"
            return ranked

        with TemporaryDirectory() as directory:
            result = run_selector_hold_backtest(
                source, constituents=None, years=1, top_n=2, holding_days=126,
                initial_capital=20_000.0, output_dir=Path(directory),
                ranking_function=fixed_ranking, commission_rate=0.0,
                minimum_commission=0.0,
            )
        forced = result.trades[result.trades["action"] == "FORCED_EXIT"]
        self.assertEqual(forced["ticker"].tolist(), ["AAA"])
        self.assertEqual(result.summary["forced_exit_orders"], 1)

    def test_ranking_can_be_limited_to_rebalance_dates(self) -> None:
        source = self._dataset()
        requested_dates: list[pd.Timestamp] = []

        def dated_ranking(
            frame: pd.DataFrame, *, ranking_dates: list[pd.Timestamp]
        ) -> pd.DataFrame:
            requested_dates.extend(ranking_dates)
            ranked = frame[frame["date"].isin(ranking_dates)].copy()
            ranked["selection_rank"] = ranked.groupby("date")["ticker"].rank(
                method="first"
            ).astype(int)
            ranked["membership_source"] = "test_point_in_time"
            return ranked

        with TemporaryDirectory() as directory:
            result = run_selector_hold_backtest(
                source,
                constituents=None,
                years=2,
                top_n=2,
                holding_days=126,
                initial_capital=20_000.0,
                output_dir=Path(directory),
                ranking_function=dated_ranking,
                rank_rebalance_dates_only=True,
            )

        sessions = pd.DatetimeIndex(result.daily_equity["date"])
        expected_dates = set(build_rebalance_dates(sessions, 126)) | {sessions[-1]}
        self.assertEqual(set(requested_dates), expected_dates)


if __name__ == "__main__":
    unittest.main()
