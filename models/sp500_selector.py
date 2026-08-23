"""Point-in-time S&P 500 ranking with price, quality, and value factors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.stock_selector import StockSelectorConfig, prepare_selector_features


DEFAULT_SP500_WEIGHTS = {
    "momentum_12m_ex_1m_rank": 0.027233,
    "momentum_6m_rank": 0.004208,
    "momentum_3m_rank": 0.155180,
    "trend_quality_rank": 0.040071,
    "downside_risk_rank": 0.048900,
    "drawdown_rank": 0.005191,
    "liquidity_rank": 0.181538,
    "size_rank": 0.026393,
    "quality_rank": 0.020410,
    "value_rank": 0.063227,
    "financial_strength_rank": 0.106469,
    "earnings_yield_rank": 0.012693,
    "fcf_yield_rank": 0.013838,
    "shareholder_yield_rank": 0.294649,
}
FUNDAMENTAL_COLUMNS = (
    "market_cap", "pe", "pb", "ps", "ev_ebitda", "roe", "roic", "roa",
    "gross_margin", "net_margin", "ebitda_margin", "current_ratio",
    "asset_turnover", "fcf", "debt_equity", "dividend_yield",
)


@dataclass(frozen=True)
class SP500SelectorConfig:
    min_history_days: int = 252
    min_price: float = 3.0
    min_median_dollar_volume_60d: float = 5_000_000.0
    min_price_vs_200d_ma: float = -0.25
    max_volatility_percentile: float = 0.90
    decision_lag_days: int = 1
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_SP500_WEIGHTS.copy())

    def __post_init__(self) -> None:
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("S&P 500 selector weights must sum to 1.0")


def _rank(frame: pd.DataFrame, column: str, higher: bool = True) -> pd.Series:
    return frame.groupby("date")[column].rank(pct=True, method="average", ascending=higher)


def rank_sp500_stocks(dataset: pd.DataFrame, config: SP500SelectorConfig | None = None) -> pd.DataFrame:
    """Rank only members on each date; fundamentals must already be point-in-time."""

    if "is_sp500" not in dataset:
        raise KeyError("S&P 500 dataset is missing the point-in-time is_sp500 column")
    config = config or SP500SelectorConfig()
    base_config = StockSelectorConfig(
        min_history_days=config.min_history_days, min_price=config.min_price,
        min_median_dollar_volume_60d=config.min_median_dollar_volume_60d,
        min_price_vs_200d_ma=config.min_price_vs_200d_ma,
        max_volatility_percentile=config.max_volatility_percentile,
        decision_lag_days=config.decision_lag_days,
    )
    frame = prepare_selector_features(dataset, base_config)
    source = dataset.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.normalize()
    source["ticker"] = source["ticker"].astype(str).str.strip().str.upper()
    available = [column for column in FUNDAMENTAL_COLUMNS if column in source]
    extras = source[["date", "ticker", "is_sp500", *available]].drop_duplicates(
        ["date", "ticker"], keep="last"
    )
    frame = frame.merge(extras, on=["date", "ticker"], how="left", suffixes=("", "_pit"))
    for column in available:
        candidate = f"{column}_pit"
        if candidate in frame:
            frame[column] = frame[candidate]
        frame[f"decision_{column}"] = frame.groupby("ticker", sort=False)[column].shift(config.decision_lag_days)

    membership = frame["is_sp500"]
    if membership.dtype == object:
        membership = membership.astype(str).str.lower().isin({"1", "true", "yes", "y"})
    else:
        membership = membership.fillna(False).astype(bool)
    frame = frame[membership].copy()
    frame["membership_source"] = "sharadar_point_in_time"
    frame["volatility_percentile"] = frame.groupby("date")["decision_volatility_30d"].rank(pct=True, method="average")
    eligible = (
        (frame["decision_history_days"] >= config.min_history_days)
        & (frame["decision_close"] >= config.min_price)
        & (frame["decision_median_dollar_volume_60d"] >= config.min_median_dollar_volume_60d)
        & (frame["decision_price_vs_200d_ma"] >= config.min_price_vs_200d_ma)
        & (frame["volatility_percentile"] <= config.max_volatility_percentile)
    )
    ranked = frame.loc[eligible].copy()
    if ranked.empty:
        return ranked.assign(selection_score=pd.Series(dtype=float))

    ranked["momentum_12m_ex_1m_rank"] = _rank(ranked, "decision_momentum_12m_ex_1m")
    ranked["momentum_6m_rank"] = _rank(ranked, "decision_momentum_6m")
    ranked["momentum_3m_rank"] = _rank(ranked, "decision_momentum_3m")
    ranked["trend_quality_rank"] = _rank(ranked, "decision_trend_quality")
    ranked["downside_risk_rank"] = _rank(ranked, "decision_downside_volatility_90d", False)
    ranked["drawdown_rank"] = _rank(ranked, "decision_drawdown_252d")
    ranked["liquidity_rank"] = _rank(ranked, "decision_median_dollar_volume_60d")
    ranked["size_rank"] = _rank(ranked, "decision_market_cap")

    quality_parts = []
    for column in (
        "roe", "roic", "roa", "gross_margin", "net_margin", "ebitda_margin",
        "asset_turnover",
    ):
        decision = f"decision_{column}"
        if decision in ranked and ranked[decision].notna().any():
            quality_parts.append(_rank(ranked, decision))
    ranked["quality_rank"] = pd.concat(quality_parts, axis=1).mean(axis=1) if quality_parts else 0.5

    value_parts = []
    for column in ("pe", "pb", "ps", "ev_ebitda"):
        decision = f"decision_{column}"
        if decision in ranked:
            positive = ranked[decision].where(ranked[decision] > 0)
            if positive.notna().any():
                temp = f"_{column}_positive"
                ranked[temp] = positive
                value_parts.append(_rank(ranked, temp, False))
    positive_pe = ranked.get("decision_pe", pd.Series(np.nan, index=ranked.index)).where(
        lambda values: values > 0
    )
    ranked["_earnings_yield"] = 1.0 / positive_pe
    ranked["earnings_yield_rank"] = _rank(ranked, "_earnings_yield")

    market_cap = ranked.get(
        "decision_market_cap", pd.Series(np.nan, index=ranked.index)
    ).where(lambda values: values > 0)
    ranked["_fcf_yield"] = ranked.get(
        "decision_fcf", pd.Series(np.nan, index=ranked.index)
    ) / market_cap
    ranked["fcf_yield_rank"] = _rank(ranked, "_fcf_yield")
    if ranked["_fcf_yield"].notna().any():
        value_parts.append(ranked["fcf_yield_rank"])
    ranked["value_rank"] = pd.concat(value_parts, axis=1).mean(axis=1) if value_parts else 0.5

    strength_parts = []
    if "decision_current_ratio" in ranked and ranked["decision_current_ratio"].notna().any():
        strength_parts.append(_rank(ranked, "decision_current_ratio"))
    if "decision_debt_equity" in ranked and ranked["decision_debt_equity"].notna().any():
        strength_parts.append(_rank(ranked, "decision_debt_equity", False))
    ranked["financial_strength_rank"] = (
        pd.concat(strength_parts, axis=1).mean(axis=1) if strength_parts else 0.5
    )
    if "decision_dividend_yield" in ranked and ranked["decision_dividend_yield"].notna().any():
        ranked["shareholder_yield_rank"] = _rank(ranked, "decision_dividend_yield")
    else:
        ranked["shareholder_yield_rank"] = 0.5
    ranked["selection_score"] = 100.0 * sum(
        ranked[column].fillna(0.5) * weight for column, weight in config.weights.items()
    )
    ranked["selection_rank"] = ranked.groupby("date")["selection_score"].rank(
        ascending=False, method="first"
    ).astype("int64")
    return ranked.drop(columns=[c for c in ranked if c.startswith("_")]).sort_values(
        ["date", "selection_rank", "ticker"]
    ).reset_index(drop=True)
