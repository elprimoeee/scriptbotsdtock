from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.sharadar_client import SharadarClient, SharadarCredentials
from models.sp500_selector import DEFAULT_SP500_WEIGHTS, SP500SelectorConfig, rank_sp500_stocks
from scripts.build_sp500_dataset import (
    build_membership,
    date_windows,
    merge_point_in_time_fundamentals,
    ticker_batches,
)
from scripts.rank_sp500_portfolio import normalized_symbol


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


class SharadarPipelineTests(unittest.TestCase):
    def test_production_weights_include_sharadar_fundamentals(self) -> None:
        for factor in (
            "quality_rank", "value_rank", "financial_strength_rank",
            "earnings_yield_rank", "fcf_yield_rank", "shareholder_yield_rank",
            "risk_adjusted_momentum_6m_rank",
        ):
            self.assertIn(factor, DEFAULT_SP500_WEIGHTS)
        self.assertGreater(abs(DEFAULT_SP500_WEIGHTS["quality_rank"]), 0.0)
        self.assertGreater(abs(DEFAULT_SP500_WEIGHTS["value_rank"]), 0.0)
        self.assertGreater(abs(DEFAULT_SP500_WEIGHTS["financial_strength_rank"]), 0.0)

    def test_sp500_weights_are_normalized_automatically(self) -> None:
        weights = {name: value * 100.0 for name, value in DEFAULT_SP500_WEIGHTS.items()}
        config = SP500SelectorConfig(weights=weights)

        self.assertAlmostEqual(sum(config.normalized_weights.values()), 1.0)
        default_total = sum(DEFAULT_SP500_WEIGHTS.values())
        for name, value in DEFAULT_SP500_WEIGHTS.items():
            self.assertAlmostEqual(
                config.normalized_weights[name], value / default_total
            )

    def test_ticker_batches_respect_api_character_limit(self) -> None:
        batches = ticker_batches(["AAAA", "BBBB", "CCCC", "DDDD"], max_count=10, max_characters=10)
        self.assertEqual(batches, [["AAAA", "BBBB"], ["CCCC", "DDDD"]])
        self.assertTrue(all(len(",".join(batch)) <= 10 for batch in batches))
        self.assertEqual(len(ticker_batches([f"T{i}" for i in range(35)], max_count=50)[0]), 30)

    def test_date_windows_are_inclusive_and_non_overlapping(self) -> None:
        self.assertEqual(
            date_windows("1998-03-31", "2026-08-30", years=20),
            [("1998-03-31", "2018-03-30"), ("2018-03-31", "2026-08-30")],
        )

    def test_trading212_and_class_share_symbols_normalize_consistently(self) -> None:
        self.assertEqual(normalized_symbol("AAPL_US_EQ"), "AAPL")
        self.assertEqual(normalized_symbol("BRK.B"), "BRKB")
        self.assertEqual(normalized_symbol("BRKb_US_EQ"), "BRKB")

    def test_client_paginates_and_does_not_expose_key_in_repr(self) -> None:
        client = SharadarClient(SharadarCredentials("top-secret"), page_size=2)
        responses = [
            _Response("ticker,date\nAAA,2024-01-01\nBBB,2024-01-01\n"),
            _Response("ticker,date\nCCC,2024-01-01\n"),
        ]
        with patch("models.sharadar_client.urlopen", side_effect=responses) as mocked:
            rows = client.table("stocks")
        self.assertEqual([row["ticker"] for row in rows], ["AAA", "BBB", "CCC"])
        self.assertEqual(mocked.call_count, 2)
        self.assertNotIn("top-secret", repr(client))

    def test_membership_uses_snapshot_then_effective_changes(self) -> None:
        events = pd.DataFrame(
            [
                {"date": "2024-01-01", "action": "historical", "ticker": "AAA"},
                {"date": "2024-01-01", "action": "historical", "ticker": "BBB"},
                {"date": "2024-01-08", "action": "removed", "ticker": "AAA"},
                {"date": "2024-01-08", "action": "added", "ticker": "CCC"},
            ]
        )
        membership = build_membership(
            pd.DatetimeIndex(["2024-01-02", "2024-01-05", "2024-01-08"]), events
        )
        by_date = membership.groupby("date")["ticker"].apply(set)
        self.assertEqual(by_date[pd.Timestamp("2024-01-05")], {"AAA", "BBB"})
        self.assertEqual(by_date[pd.Timestamp("2024-01-08")], {"BBB", "CCC"})

    def test_art_fundamental_is_visible_only_from_filing_date(self) -> None:
        prices = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05"]),
                "close": [10.0, 10.5, 11.0],
            }
        )
        daily = pd.DataFrame(
            {"ticker": ["AAA"], "date": ["2024-02-05"], "marketcap": [123.0]}
        )
        art = pd.DataFrame(
            {
                "ticker": ["AAA"], "date": ["2024-02-02"],
                "calendardate": ["2023-12-31"], "reportperiod": ["2023-12-31"],
                "dimension": ["ART"], "roe": [0.2],
            }
        )
        merged = merge_point_in_time_fundamentals(prices, daily, art)
        self.assertTrue(pd.isna(merged.loc[merged["date"] == pd.Timestamp("2024-02-01"), "roe"]).all())
        self.assertEqual(float(merged.loc[merged["date"] == pd.Timestamp("2024-02-05"), "roe"].iloc[0]), 0.2)
        self.assertEqual(float(merged.loc[merged["date"] == pd.Timestamp("2024-02-05"), "market_cap"].iloc[0]), 123_000_000.0)

    def test_selector_excludes_nonmembers_and_uses_fundamentals(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=320)
        rows = []
        for ticker, member, roe, pe in (("GOOD", True, 0.3, 12.0), ("WEAK", True, 0.02, 40.0), ("OUT", False, 0.5, 5.0)):
            close = np.linspace(20.0, 30.0, len(dates))
            for day, value in zip(dates, close):
                rows.append({
                    "date": day, "ticker": ticker, "open": value, "close": value,
                    "dollar_volume": 10_000_000.0, "market_cap": 10_000_000_000.0,
                    "is_sp500": member, "roe": roe, "roic": roe, "roa": roe,
                    "gross_margin": roe, "net_margin": roe, "fcf": roe * 1e9,
                    "debt_equity": 1.0 / roe, "pe": pe, "pb": pe / 2,
                    "ps": pe / 3, "ev_ebitda": pe,
                })
        ranked = rank_sp500_stocks(pd.DataFrame(rows), SP500SelectorConfig(max_volatility_percentile=1.0))
        scaled_weights = {
            name: value * 100.0 for name, value in DEFAULT_SP500_WEIGHTS.items()
        }
        scaled = rank_sp500_stocks(
            pd.DataFrame(rows),
            SP500SelectorConfig(
                max_volatility_percentile=1.0,
                weights=scaled_weights,
            ),
        )
        np.testing.assert_allclose(ranked["selection_score"], scaled["selection_score"])
        latest = ranked[ranked["date"] == ranked["date"].max()]
        self.assertEqual(set(latest["ticker"]), {"GOOD", "WEAK"})
        self.assertLess(int(latest.set_index("ticker").loc["GOOD", "selection_rank"]),
                        int(latest.set_index("ticker").loc["WEAK", "selection_rank"]))


if __name__ == "__main__":
    unittest.main()
