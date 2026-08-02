"""Shared interface for trading models."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TradingModel:
    """Base class for simple ranking models."""

    name: str
    description: str
    required_columns: tuple[str, ...] = field(default_factory=tuple)

    def validate_columns(self, dataset: pd.DataFrame) -> None:
        missing = [column for column in self.required_columns if column not in dataset.columns]
        if missing:
            raise KeyError(
                f"Model `{self.name}` is missing required dataset columns: {', '.join(missing)}"
            )

    def prepare_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(dataset)
        return dataset.copy()

    def score(self, dataset: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Override `score()` in your model subclass.")

    def generate_signals(self, dataset: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare_features(dataset)
        scores = self.score(prepared)
        if len(scores) != len(prepared):
            raise ValueError(
                f"Model `{self.name}` returned {len(scores)} scores for {len(prepared)} rows."
            )
        signals = prepared.copy()
        signals["model_score"] = scores.to_numpy()
        return signals
