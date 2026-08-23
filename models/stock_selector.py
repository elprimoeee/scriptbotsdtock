"""Reusable lagged price, risk, liquidity, and size selector features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PRICE_WEIGHTS = {
    "momentum_12m_ex_1m_rank": 0.32,
    "momentum_6m_rank": 0.26,
    "momentum_3m_rank": 0.14,
    "trend_quality_rank": 0.2,
    "downside_risk_rank": 0.03,
    "drawdown_rank": 0.02,
    "liquidity_rank": 0.03,
    "size_rank": 0,
}


@dataclass(frozen=True)
class StockSelectorConfig:
    """Eligibility and weighting settings for the stock selector."""

    min_history_days: int = 252
    min_price: float = 1.0
    min_median_dollar_volume_60d: float = 2_000_000.0
    min_price_vs_200d_ma: float = -0.25
    max_volatility_percentile: float = 0.90
    decision_lag_days: int = 1
    market_cap_proxy_count: int = 200
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_PRICE_WEIGHTS.copy())

    def __post_init__(self) -> None:
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Stock selector weights must sum to 1.0")
        if self.decision_lag_days < 0:
            raise ValueError("decision_lag_days cannot be negative")


def normalize_tickers(values: Iterable[object]) -> set[str]:
    """Return normalized symbols, accepting legacy symbols with an .AX suffix."""

    normalized = set()
    for value in values:
        if pd.isna(value):
            continue
        ticker = str(value).strip().upper()
        if ticker.endswith(".AX"):
            ticker = ticker[:-3]
        if ticker:
            normalized.add(ticker)
    return normalized


def _ticker_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str.replace(r"\.AX$", "", regex=True)


def _cross_sectional_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool = True,
) -> pd.Series:
    """Percentile-rank a feature independently on each decision date."""

    return frame.groupby("date")[column].rank(
        pct=True,
        method="average",
        ascending=higher_is_better,
    )


def prepare_selector_features(dataset: pd.DataFrame, config: StockSelectorConfig) -> pd.DataFrame:
    """Calculate selector features and lag them before any ranking decision."""

    required = {"date", "ticker", "close", "dollar_volume"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise KeyError(f"Selector dataset is missing columns: {', '.join(missing)}")

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = _ticker_code(frame["ticker"])
    for column in ("close", "dollar_volume", "market_cap", "market_cap_rank"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", "close"]).sort_values(
        ["ticker", "date"]
    ).drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)

    grouped_close = frame.groupby("ticker", sort=False)["close"]
    returns = grouped_close.pct_change(fill_method=None)
    frame["history_days"] = frame.groupby("ticker", sort=False).cumcount() + 1
    frame["momentum_12m_ex_1m"] = (
        grouped_close.shift(21) / grouped_close.shift(252)
    ) - 1.0
    frame["momentum_6m"] = grouped_close.pct_change(126, fill_method=None)
    frame["momentum_3m"] = grouped_close.pct_change(63, fill_method=None)
    frame["ma_50d"] = grouped_close.transform(lambda values: values.rolling(50).mean())
    frame["ma_200d"] = grouped_close.transform(lambda values: values.rolling(200).mean())
    frame["price_vs_200d_ma"] = (frame["close"] / frame["ma_200d"]) - 1.0
    frame["ma_50d_vs_200d"] = (frame["ma_50d"] / frame["ma_200d"]) - 1.0
    frame["trend_quality"] = (
        frame["price_vs_200d_ma"].clip(-0.25, 0.25)
        + frame["ma_50d_vs_200d"].clip(-0.15, 0.15)
    )
    frame["volatility_30d"] = returns.groupby(frame["ticker"], sort=False).transform(
        lambda values: values.rolling(30).std()
    )
    negative_returns = returns.where(returns < 0.0, 0.0)
    frame["downside_volatility_90d"] = negative_returns.groupby(
        frame["ticker"], sort=False
    ).transform(lambda values: values.rolling(90).std())
    rolling_peak = grouped_close.transform(lambda values: values.rolling(252).max())
    frame["drawdown_252d"] = (frame["close"] / rolling_peak) - 1.0
    frame["median_dollar_volume_60d"] = frame.groupby("ticker", sort=False)[
        "dollar_volume"
    ].transform(lambda values: values.rolling(60).median())

    if "market_cap" not in frame.columns:
        frame["market_cap"] = np.nan
    if "market_cap_rank" not in frame.columns:
        frame["market_cap_rank"] = frame.groupby("date")["market_cap"].rank(
            ascending=False, method="dense"
        )

    decision_columns = [
        "close",
        "history_days",
        "momentum_12m_ex_1m",
        "momentum_6m",
        "momentum_3m",
        "trend_quality",
        "price_vs_200d_ma",
        "volatility_30d",
        "downside_volatility_90d",
        "drawdown_252d",
        "median_dollar_volume_60d",
        "market_cap",
        "market_cap_rank",
    ]
    lagged = frame.groupby("ticker", sort=False)[decision_columns].shift(
        config.decision_lag_days
    )
    for column in decision_columns:
        frame[f"decision_{column}"] = lagged[column]
    return frame.replace([np.inf, -np.inf], np.nan)


def rank_stocks(
    dataset: pd.DataFrame,
    *,
    constituents: Iterable[object] | None = None,
    allow_market_cap_proxy: bool = False,
    config: StockSelectorConfig | None = None,
) -> pd.DataFrame:
    """Return eligible stock-days with a 0-100 cross-sectional score."""

    config = config or StockSelectorConfig()
    frame = prepare_selector_features(dataset, config)

    constituent_codes = normalize_tickers([] if constituents is None else constituents)
    if constituent_codes:
        frame = frame[frame["ticker"].isin(constituent_codes)].copy()
        frame["membership_source"] = "constituent_file"
    elif "is_member" in frame.columns:
        membership_values = frame["is_member"]
        if membership_values.dtype == object:
            membership_values = membership_values.astype(str).str.strip().str.lower().isin(
                {"1", "true", "yes", "y"}
            )
        else:
            membership_values = membership_values.fillna(False).astype(bool)
        frame = frame[membership_values].copy()
        frame["membership_source"] = "point_in_time_column"
    elif allow_market_cap_proxy:
        frame = frame[
            frame["decision_market_cap_rank"].le(config.market_cap_proxy_count)
        ].copy()
        frame["membership_source"] = "market_cap_proxy"
    else:
        raise ValueError(
            "No index membership supplied. Provide a constituent list or use "
            "allow_market_cap_proxy=True explicitly."
        )

    frame["volatility_percentile"] = frame.groupby("date")[
        "decision_volatility_30d"
    ].rank(pct=True, method="average")
    eligible = (
        (frame["decision_history_days"] >= config.min_history_days)
        & (frame["decision_close"] >= config.min_price)
        & (
            frame["decision_median_dollar_volume_60d"]
            >= config.min_median_dollar_volume_60d
        )
        & (frame["decision_price_vs_200d_ma"] >= config.min_price_vs_200d_ma)
        & (frame["volatility_percentile"] <= config.max_volatility_percentile)
    )
    ranked = frame.loc[eligible].copy()
    if ranked.empty:
        ranked["selection_score"] = pd.Series(dtype=float)
        ranked["selection_rank"] = pd.Series(dtype="Int64")
        return ranked

    ranked["momentum_12m_ex_1m_rank"] = _cross_sectional_rank(
        ranked, "decision_momentum_12m_ex_1m"
    )
    ranked["momentum_6m_rank"] = _cross_sectional_rank(ranked, "decision_momentum_6m")
    ranked["momentum_3m_rank"] = _cross_sectional_rank(ranked, "decision_momentum_3m")
    ranked["trend_quality_rank"] = _cross_sectional_rank(
        ranked, "decision_trend_quality"
    )
    ranked["downside_risk_rank"] = _cross_sectional_rank(
        ranked, "decision_downside_volatility_90d", higher_is_better=False
    )
    ranked["drawdown_rank"] = _cross_sectional_rank(ranked, "decision_drawdown_252d")
    ranked["liquidity_rank"] = _cross_sectional_rank(
        ranked, "decision_median_dollar_volume_60d"
    )
    ranked["size_rank"] = _cross_sectional_rank(ranked, "decision_market_cap")

    ranked["selection_score"] = 100.0 * sum(
        ranked[column].fillna(0.0) * weight
        for column, weight in config.weights.items()
    )
    ranked["selection_rank"] = ranked.groupby("date")["selection_score"].rank(
        ascending=False, method="first"
    ).astype("int64")
    return ranked.sort_values(["date", "selection_rank", "ticker"]).reset_index(drop=True)


def latest_shortlist(ranked: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Return the best stocks on the latest available ranked date."""

    if ranked.empty:
        return ranked.copy()
    latest_date = ranked["date"].max()
    return ranked.loc[ranked["date"] == latest_date].nsmallest(
        limit, "selection_rank"
    ).reset_index(drop=True)
