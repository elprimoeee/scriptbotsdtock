from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.stock_selector import (
    StockSelectorConfig,
    DEFAULT_PRICE_WEIGHTS,
    latest_shortlist,
    rank_stocks,
)


class StockSelectorTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=320)
        rows = []
        settings = {
            "BIG": (10.0, 100_000_000_000.0, 5_000_000.0),
            "MID": (10.0, 20_000_000_000.0, 5_000_000.0),
            "LOWVOL": (10.0, 5_000_000_000.0, 100_000.0),
            "OUT": (10.0, 200_000_000_000.0, 5_000_000.0),
        }
        for ticker, (start, market_cap, dollar_volume) in settings.items():
            close = np.linspace(start, start * 1.4, len(dates))
            for date, price in zip(dates, close):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "close": price,
                        "dollar_volume": dollar_volume,
                        "market_cap": market_cap,
                    }
                )
        return pd.DataFrame(rows)

    def test_weights_match_the_production_selector_and_sum_to_one(self) -> None:
        self.assertAlmostEqual(DEFAULT_PRICE_WEIGHTS["size_rank"], 0.0)
        self.assertAlmostEqual(sum(DEFAULT_PRICE_WEIGHTS.values()), 1.0)

    def test_constituent_and_liquidity_filters_are_enforced(self) -> None:
        ranked = rank_stocks(
            self._dataset(),
            constituents=["BIG.AX", "MID", "LOWVOL"],
        )
        latest = latest_shortlist(ranked)
        self.assertEqual(set(latest["ticker"]), {"BIG", "MID"})
        self.assertNotIn("OUT", set(latest["ticker"]))
        self.assertNotIn("LOWVOL", set(latest["ticker"]))

    def test_zero_size_weight_leaves_otherwise_equal_scores_tied(self) -> None:
        ranked = rank_stocks(
            self._dataset(),
            constituents=["BIG", "MID"],
        )
        latest = latest_shortlist(ranked)
        scores = latest.set_index("ticker")["selection_score"]
        self.assertAlmostEqual(scores["BIG"], scores["MID"])

    def test_latest_decision_uses_previous_trading_day(self) -> None:
        dataset = self._dataset()
        big_rows = dataset.index[dataset["ticker"] == "BIG"]
        previous_close = float(dataset.loc[big_rows[-2], "close"])
        dataset.loc[big_rows[-1], "close"] = 999.0
        ranked = rank_stocks(dataset, constituents=["BIG", "MID"])
        latest = latest_shortlist(ranked)
        big = latest.loc[latest["ticker"] == "BIG"].iloc[0]
        self.assertAlmostEqual(float(big["decision_close"]), previous_close)

    def test_proxy_requires_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "No index membership"):
            rank_stocks(self._dataset())

        config = StockSelectorConfig(market_cap_proxy_count=2)
        ranked = rank_stocks(
            self._dataset(),
            allow_market_cap_proxy=True,
            config=config,
        )
        self.assertFalse(ranked.empty)
        self.assertEqual(set(ranked["membership_source"]), {"market_cap_proxy"})


if __name__ == "__main__":
    unittest.main()
