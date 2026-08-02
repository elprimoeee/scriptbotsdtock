"""Starter template for a simple ranking model."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base import TradingModel


class RankingModelTemplate(TradingModel):
    def __init__(self) -> None:
        super().__init__(
            name="ranking_model_template",
            description="Example ranking model that ranks stocks by a small set of metrics.",
            required_columns=("ticker", "date", "momentum_1m", "volatility_30d"),
        )

    def score(self, dataset: pd.DataFrame) -> pd.Series:
        # Higher 1-month momentum is good, lower volatility is good.
        return dataset["momentum_1m"].fillna(0.0) - dataset["volatility_30d"].fillna(0.0)
