#!/usr/bin/env python3
"""RSI momentum bot with dynamic range trading strategy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base import TradingModel
from models.benchmarking import build_direct_benchmark_test
from project_config import CFD_COSTS, PORTFOLIO, PathConfig, BenchmarkConfig, PortfolioConfig


# Configuration
RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 5
MOMENTUM_LOOKBACK_DAYS = 63  # ~3 months
ANNUAL_LOOKBACK_DAYS = 252  # ~1 year
SIGNAL_LAG_DAYS = 1
DECISION_LAG_DAYS = 1
RANGE_MULTIPLIER = 0.5  # If gained 5%, range is +/-2.5%
RANGE_QUANTILE = 0.70
RANGE_MIN_OFFSET = 0.03
RANGE_MAX_OFFSET = 0.30
TOP_STOCKS = 200
VOLATILITY_WINDOW = 20
ASX200_RANK_CUTOFF = 200
MIN_DOLLAR_VOLUME = 500_000.0
MIN_PRICE = 0.50
MAX_VOLATILITY_PERCENTILE = 0.90
MIN_PRICE_VS_200D_MA = -0.25
STOP_PRICE_VS_200D_MA = -0.35
MAX_BUY_RSI = 62.0
MIN_BUY_RSI = 20.0
MIN_BUY_MOMENTUM_3M = -0.45
MIN_WEEKLY_RELATIVE_MOMENTUM = 0.0
RSI_PULLBACK_REENTRY = 40.0
MAX_WEEKLY_UNDERPERFORMANCE_VOL = 1.0
INITIAL_CAPITAL = 20_000.0
MIN_HOLDING_TRADING_DAYS = 20
MAX_TICKER_EXPOSURE_PCT = 0.15
MAX_LOTS_PER_TICKER = 3
BUY_COOLDOWN_TRADING_DAYS = 10
MIN_UPPER_REENTRY_SELL_RSI = 60.0
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


class RSIBuySignal(TradingModel):
    """Calculate RSI-based momentum signals."""
    
    def __init__(self):
        super().__init__(
            name="RSI Momentum Bot",
            description="RSI-based momentum trading with dynamic range and confirmation logic",
            required_columns=("close", "dollar_volume")
        )
    
    def prepare_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Prepare RSI, volatility, and range features."""
        self.validate_columns(dataset)
        df = dataset.copy()
        
        # Ensure proper sorting
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Calculate RSI for each ticker
        df["rsi"] = df.groupby("ticker")["close"].transform(self._calculate_rsi)
        
        # Calculate volatility (std dev of returns over window)
        df["volatility"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.pct_change().rolling(VOLATILITY_WINDOW).std())
        )
        
        # Calculate 3-month momentum
        df["momentum_3m"] = df.groupby("ticker")["close"].transform(
            lambda x: (x / x.shift(MOMENTUM_LOOKBACK_DAYS)) - 1.0
        )

        # Compare the latest trading week with the weekly return implied by
        # the trailing year. Positive values mean short-term acceleration.
        df["return_1w"] = df.groupby("ticker")["close"].transform(
            lambda x: (x / x.shift(WEEKLY_LOOKBACK_DAYS)) - 1.0
        )
        
        # Calculate 1-year return
        df["return_1y"] = df.groupby("ticker")["close"].transform(
            lambda x: (x / x.shift(ANNUAL_LOOKBACK_DAYS)) - 1.0
        )
        valid_annual_return = df["return_1y"].where(df["return_1y"] > -1.0)
        df["yearly_implied_weekly_return"] = np.power(1.0 + valid_annual_return, 1.0 / 52.0) - 1.0
        df["weekly_relative_momentum"] = df["return_1w"] - df["yearly_implied_weekly_return"]

        df["ma_200d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(200).mean()
        )
        df["price_vs_200d_ma"] = (df["close"] / df["ma_200d"]) - 1.0

        if "benchmark_close" in df.columns:
            df["benchmark_momentum_3m"] = df.groupby("ticker")[
                "benchmark_close"
            ].transform(lambda x: (x / x.shift(MOMENTUM_LOOKBACK_DAYS)) - 1.0)
            df["benchmark_ma_200d"] = df.groupby("ticker")[
                "benchmark_close"
            ].transform(lambda x: x.rolling(200).mean())
            df["relative_strength_3m"] = (
                df["momentum_3m"] - df["benchmark_momentum_3m"]
            )
            df["market_regime_ok"] = df["benchmark_close"] >= df["benchmark_ma_200d"]
        else:
            df["benchmark_momentum_3m"] = np.nan
            df["benchmark_ma_200d"] = np.nan
            df["relative_strength_3m"] = np.nan
            df["market_regime_ok"] = True
        
        # Original range: half the absolute trailing one-year return.
        df["range_upper"] = abs(df["return_1y"]) * RANGE_MULTIPLIER
        df["range_lower"] = -abs(df["return_1y"]) * RANGE_MULTIPLIER
        
        # Current position relative to range (0 = middle, 1 = at upper, -1 = at lower)
        df["current_return"] = df["momentum_3m"]
        df["position_in_range"] = np.where(
            df["range_upper"] != 0,
            df["current_return"] / df["range_upper"],
            0
        )
        
        # Do not turn unavailable history into a valid zero-valued feature.
        required_history = [
            "rsi",
            "volatility",
            "momentum_3m",
            "return_1y",
            "weekly_relative_momentum",
            "ma_200d",
            "price_vs_200d_ma",
        ]
        df["warmup_ready"] = df[required_history].notna().all(axis=1)
        if "benchmark_close" in df.columns:
            df["warmup_ready"] &= df[
                ["benchmark_momentum_3m", "benchmark_ma_200d", "relative_strength_3m"]
            ].notna().all(axis=1)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=RSI_PERIOD):
        """Calculate RSI indicator."""
        if len(prices) < period:
            return pd.Series(np.nan, index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.where(loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
        rsi = rsi.mask((gain == 0) & (loss == 0), 50.0)
        return rsi
    
    def score(self, dataset: pd.DataFrame) -> pd.Series:
        """Generate momentum score based on RSI and volatility."""
        df = dataset.copy()
        
        # RSI component: favor middle range (50 is neutral)
        rsi_score = 1.0 - (abs(df["rsi"] - 50) / 50)
        
        # Volatility component: penalize high volatility
        volatility_percentile = df.groupby("date")["volatility"].transform(
            lambda x: x.rank(pct=True)
        )
        volatility_score = 1.0 - volatility_percentile
        
        # Combine scores
        combined_score = (rsi_score * 0.5 + volatility_score * 0.5)
        
        return combined_score


@dataclass
class RSIBotResult:
    """Results from RSI bot backtest."""
    output_dir: Path
    positions_df: pd.DataFrame
    daily_pnl: pd.DataFrame
    summary: dict
    saved_paths: dict


def filter_low_volatility_stocks(df: pd.DataFrame, top_n: int = TOP_STOCKS) -> pd.DataFrame:
    """Mark ASX200-quality candidates using only prior-day data."""
    result = df.sort_values(["ticker", "date"]).copy()
    result["decision_volatility"] = result.groupby("ticker")["volatility"].shift(DECISION_LAG_DAYS)
    result["decision_dollar_volume"] = result.groupby("ticker")["dollar_volume"].shift(DECISION_LAG_DAYS)
    result["decision_close"] = result.groupby("ticker")["close"].shift(DECISION_LAG_DAYS)
    result["decision_price_vs_200d_ma"] = result.groupby("ticker")["price_vs_200d_ma"].shift(DECISION_LAG_DAYS)

    if "market_cap_rank" in result.columns:
        result["decision_market_cap_rank"] = result.groupby("ticker")["market_cap_rank"].shift(DECISION_LAG_DAYS)
        asx200_filter = result["decision_market_cap_rank"].le(ASX200_RANK_CUTOFF)
    else:
        result["decision_market_cap_rank"] = np.nan
        asx200_filter = True

    result["volatility_percentile"] = result.groupby("date")["decision_volatility"].rank(pct=True)
    eligible = (
        result["decision_volatility"].notna()
        & result["decision_dollar_volume"].notna()
        & (result["decision_close"] >= MIN_PRICE)
        & (result["decision_dollar_volume"] >= MIN_DOLLAR_VOLUME)
        & (result["decision_price_vs_200d_ma"] >= MIN_PRICE_VS_200D_MA)
        & (result["volatility_percentile"] <= MAX_VOLATILITY_PERCENTILE)
        & asx200_filter
    )
    result["volatility_rank"] = np.nan
    result.loc[eligible, "volatility_rank"] = result.loc[eligible].groupby("date")["decision_volatility"].rank(
        method="first",
        ascending=True,
    )
    result["is_low_volatility_candidate"] = result["volatility_rank"].le(top_n).fillna(False)
    return result.reset_index(drop=True)


def generate_buy_sell_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate confirmed multi-timeframe buy/sell signals.
    - Buy only after momentum crosses back above the lower range
      or RSI recovers from a pullback within an established uptrend
    - Weekly momentum rejects abnormally weak pullback entries, but a single
      weak week never forces an exit
    - Sell after an upper-range reversal only when RSI is elevated; overbought
      reversals and risk breaks remain independent exits
    Signals are lagged one trading day so close-based signals are not
    executed at the same close that created them.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    signals = []
    
    for ticker in df["ticker"].unique():
        ticker_data = df[df["ticker"] == ticker].copy().reset_index(drop=True)
        
        # Determine price position relative to range
        ticker_data["above_range"] = ticker_data["current_return"] > ticker_data["range_upper"]
        ticker_data["below_range"] = ticker_data["current_return"] < ticker_data["range_lower"]
        ticker_data["in_range"] = ~(ticker_data["above_range"] | ticker_data["below_range"])
        ticker_data["rsi_rising"] = ticker_data["rsi"] > ticker_data["rsi"].shift(1)
        ticker_data["rsi_falling"] = ticker_data["rsi"] < ticker_data["rsi"].shift(1)
        ticker_data["weekly_outperforming_year"] = (
            ticker_data["weekly_relative_momentum"] > MIN_WEEKLY_RELATIVE_MOMENTUM
        )
        ticker_data["weekly_underperforming_year"] = (
            ticker_data["weekly_relative_momentum"] < -MIN_WEEKLY_RELATIVE_MOMENTUM
        )
        ticker_data["weekly_momentum_acceptable"] = (
            ticker_data["weekly_relative_momentum"]
            > -(ticker_data["volatility"] * MAX_WEEKLY_UNDERPERFORMANCE_VOL)
        )
        ticker_data["confirmed_lower_reentry"] = (
            ticker_data["below_range"].shift(1).fillna(False)
            & (ticker_data["current_return"] >= ticker_data["range_lower"])
        )
        ticker_data["confirmed_upper_reentry"] = (
            ticker_data["above_range"].shift(1).fillna(False)
            & (ticker_data["current_return"] <= ticker_data["range_upper"])
        )
        ticker_data["risk_break"] = (
            (ticker_data["price_vs_200d_ma"] < STOP_PRICE_VS_200D_MA)
            | (ticker_data["momentum_3m"] < MIN_BUY_MOMENTUM_3M)
        )
        
        # Signal generation
        ticker_data["buy_signal"] = 0.0
        ticker_data["sell_signal"] = 0.0
        
        range_reentry_buy = (
            (
                ticker_data["confirmed_lower_reentry"]
                | (ticker_data["below_range"] & ticker_data["rsi_rising"])
            )
            & ticker_data["rsi_rising"]
            & ticker_data["rsi"].between(MIN_BUY_RSI, MAX_BUY_RSI)
            & (ticker_data["momentum_3m"] >= MIN_BUY_MOMENTUM_3M)
            & (ticker_data["price_vs_200d_ma"] >= MIN_PRICE_VS_200D_MA)
        )
        trend_pullback_buy = (
            (ticker_data["rsi"].shift(1) <= RSI_PULLBACK_REENTRY)
            & (ticker_data["rsi"] > RSI_PULLBACK_REENTRY)
            & (ticker_data["close"] > ticker_data["ma_200d"])
            & (ticker_data["momentum_3m"] > 0.0)
            & ticker_data["weekly_momentum_acceptable"]
        )
        raw_buy_signal = (range_reentry_buy | trend_pullback_buy).astype(float)
        raw_sell_signal = (
            ticker_data["confirmed_upper_reentry"]
            | ((ticker_data["rsi"] >= 70.0) & ticker_data["rsi_falling"])
            | ticker_data["risk_break"]
        ).astype(float)
        ticker_data["buy_signal"] = raw_buy_signal.shift(SIGNAL_LAG_DAYS).fillna(0.0)
        ticker_data["sell_signal"] = raw_sell_signal.shift(SIGNAL_LAG_DAYS).fillna(0.0)
        
        signals.append(ticker_data)
    
    result = pd.concat(signals, ignore_index=True)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def latest_strict_recommendations(df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Return the latest buy/sell/watch list ranked by signal quality."""
    signals = generate_buy_sell_signals(df)
    latest_date = signals["date"].max()
    latest = signals[signals["date"] == latest_date].copy()
    latest = latest[latest["is_low_volatility_candidate"]].copy()
    latest["signal_type"] = np.select(
        [latest["buy_signal"] > 0, latest["sell_signal"] > 0],
        ["BUY", "SELL"],
        default="WATCH",
    )
    latest["strict_score"] = (
        latest["model_score"].fillna(0)
        + (1.0 - latest["volatility_percentile"].fillna(1.0))
        + latest["price_vs_200d_ma"].clip(-0.25, 0.25).fillna(0)
        + np.where(latest["below_range"], 0.25, 0.0)
    )
    latest["signal_priority"] = latest["signal_type"].map({"BUY": 0, "SELL": 1, "WATCH": 2})
    columns = [
        "date",
        "ticker",
        "signal_type",
        "strict_score",
        "close",
        "rsi",
        "momentum_3m",
        "return_1w",
        "return_1y",
        "yearly_implied_weekly_return",
        "weekly_relative_momentum",
        "price_vs_200d_ma",
        "volatility",
        "volatility_rank",
        "decision_market_cap_rank",
        "dollar_volume",
    ]
    if latest.empty:
        return latest.reindex(columns=columns)
    latest = latest.sort_values(
        ["signal_priority", "strict_score", "volatility_rank"],
        ascending=[True, False, True],
    )
    return latest[columns].head(limit).reset_index(drop=True)


def backtest_rsi_bot(
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    position_size_pct: float = 0.02,  # 2% per position
) -> RSIBotResult:
    """Run a cost-aware, profit-first backtest with original-signal fallback."""
    
    df = generate_buy_sell_signals(df)
    
    # Initialize tracking
    positions = {}  # {ticker: [{units, entry_index, entry_date}, ...]}
    cash = initial_capital
    equity_curve = []
    position_log = []
    total_fees = 0.0
    realized_trade_pnls: list[float] = []
    last_buy_index: dict[str, int] = {}
    
    dates = sorted(df["date"].unique())
    
    for date_index, date in enumerate(tqdm(dates, desc="Running backtest")):
        date_data = df[df["date"] == date].set_index("ticker")
        
        # Execute sells
        for ticker in list(positions.keys()):
            if ticker in date_data.index:
                row = date_data.loc[ticker]
                risk_exit = bool(row.get("risk_exit_signal", False))
                eligible_lots = [
                    lot for lot in positions[ticker]
                    if risk_exit
                    or date_index - lot["entry_index"] >= MIN_HOLDING_TRADING_DAYS
                ]
                market_price = row.get("open", row["close"])
                price = market_price * (1.0 - PORTFOLIO.slippage_per_trade)
                profitable_lots = []
                for lot in eligible_lots:
                    entry_cost = (
                        lot["units"] * lot["entry_price"]
                        + lot.get("entry_commission", 0.0)
                    )
                    projected_proceeds = lot["units"] * price
                    projected_commission = max(
                        CFD_COSTS.asx_share_cfd_min_commission_aud,
                        projected_proceeds * CFD_COSTS.asx_share_cfd_commission_rate,
                    )
                    if projected_proceeds - projected_commission > entry_cost:
                        profitable_lots.append(lot)

                original_exit = row["sell_signal"] > 0
                lots_to_sell = eligible_lots if original_exit else profitable_lots
                if lots_to_sell:
                    units = sum(lot["units"] for lot in lots_to_sell)
                    proceeds = units * price
                    commission = max(
                        CFD_COSTS.asx_share_cfd_min_commission_aud,
                        proceeds * CFD_COSTS.asx_share_cfd_commission_rate,
                    )
                    cash += proceeds - commission
                    total_fees += commission
                    entry_cost = sum(
                        lot["units"] * lot["entry_price"]
                        + lot.get("entry_commission", 0.0)
                        for lot in lots_to_sell
                    )
                    realized_pnl = proceeds - commission - entry_cost
                    realized_trade_pnls.append(realized_pnl)
                    
                    position_log.append({
                        "date": date,
                        "signal_lag_days": SIGNAL_LAG_DAYS,
                        "ticker": ticker,
                        "action": "SELL",
                        "units": units,
                        "price": price,
                        "value": proceeds,
                        "commission": commission,
                        "realized_pnl": realized_pnl,
                        "exit_reason": (
                            "original_signal_fallback"
                            if original_exit
                            else "profit_first_outside_bands"
                        ),
                        "cash_after_trade": cash,
                    })
                    
                    eligible_ids = {id(lot) for lot in lots_to_sell}
                    positions[ticker] = [
                        lot for lot in positions[ticker] if id(lot) not in eligible_ids
                    ]
                    if not positions[ticker]:
                        del positions[ticker]
        
        # Execute buys
        if "is_low_volatility_candidate" in date_data:
            buy_candidates = date_data[
                (date_data["buy_signal"] > 0)
                & date_data["is_low_volatility_candidate"]
            ].copy()
        else:
            buy_candidates = date_data[date_data["buy_signal"] > 0].copy()
        
        if len(buy_candidates) > 0 and cash > 0:
            for ticker in buy_candidates.index:
                row = buy_candidates.loc[ticker]
                existing_lots = positions.get(ticker, [])
                if len(existing_lots) >= MAX_LOTS_PER_TICKER:
                    continue
                if date_index - last_buy_index.get(ticker, -BUY_COOLDOWN_TRADING_DAYS) < BUY_COOLDOWN_TRADING_DAYS:
                    continue
                market_price = row.get("open", row["close"])
                price = market_price * (1.0 + PORTFOLIO.slippage_per_trade)
                portfolio_value = cash
                for held_ticker, lots in positions.items():
                    if held_ticker in date_data.index:
                        portfolio_value += (
                            sum(lot["units"] for lot in lots)
                            * date_data.loc[held_ticker, "close"]
                        )
                held_value = sum(lot["units"] for lot in existing_lots) * row["close"]
                exposure_room = max(
                    0.0,
                    portfolio_value * MAX_TICKER_EXPOSURE_PCT - held_value,
                )
                trade_budget = min(
                    cash - CFD_COSTS.asx_share_cfd_min_commission_aud,
                    portfolio_value * position_size_pct,
                    exposure_room,
                )
                if price > 0 and trade_budget > 0:
                    units = int(trade_budget / price)
                    if units > 0:
                        cost = units * price
                        commission = max(
                            CFD_COSTS.asx_share_cfd_min_commission_aud,
                            cost * CFD_COSTS.asx_share_cfd_commission_rate,
                        )
                        if cost + commission <= cash:
                            cash -= cost + commission
                            total_fees += commission
                            last_buy_index[ticker] = date_index
                            positions.setdefault(ticker, []).append({
                                "units": units,
                                "entry_index": date_index,
                                "entry_date": date,
                                "entry_price": price,
                                "entry_commission": commission,
                            })
                                
                            position_log.append({
                                "date": date,
                                "signal_lag_days": SIGNAL_LAG_DAYS,
                                "ticker": ticker,
                                "action": "BUY",
                                "units": units,
                                "price": price,
                                "value": cost,
                                "commission": commission,
                                "cash_after_trade": cash,
                            })
        
        # Calculate portfolio value
        portfolio_value = cash
        for ticker in positions:
            if ticker in date_data.index:
                units = sum(lot["units"] for lot in positions[ticker])
                portfolio_value += units * date_data.loc[ticker, "close"]
        
        equity_curve.append({
            "date": date,
            "equity": portfolio_value,
            "cash": cash,
            "positions_count": len(positions)
        })
    
    # Create results dataframes
    equity_df = pd.DataFrame(equity_curve)
    positions_df = pd.DataFrame(position_log)
    
    # Calculate summary metrics
    total_return = (equity_df["equity"].iloc[-1] / initial_capital - 1) * 100
    running_peak = equity_df["equity"].cummax()
    max_drawdown = ((equity_df["equity"] / running_peak) - 1.0).min() * -100.0
    
    summary = {
        "initial_capital": initial_capital,
        "final_equity": equity_df["equity"].iloc[-1],
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "total_trades": len(positions_df),
        "buy_count": len(positions_df[positions_df["action"] == "BUY"]),
        "sell_count": len(positions_df[positions_df["action"] == "SELL"]),
        "final_cash": equity_df["cash"].iloc[-1],
        "open_positions": equity_df["positions_count"].iloc[-1],
        "minimum_holding_trading_days": MIN_HOLDING_TRADING_DAYS,
        "total_fees": total_fees,
        "slippage_bps": PORTFOLIO.slippage_per_trade * 10_000.0,
        "realized_sell_count": len(realized_trade_pnls),
        "winning_sell_count": sum(pnl > 0.0 for pnl in realized_trade_pnls),
        "win_rate_pct": (
            100.0 * sum(pnl > 0.0 for pnl in realized_trade_pnls) / len(realized_trade_pnls)
            if realized_trade_pnls
            else 0.0
        ),
    }
    
    output_dir = Path("results/rsi_bot")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = RSIBotResult(
        output_dir=output_dir,
        positions_df=positions_df,
        daily_pnl=equity_df,
        summary=summary,
        saved_paths={}
    )
    
    return result


def plot_strategy(
    df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    output_dir: Path,
    sample_tickers: list[str] | None = None
) -> dict[str, Path]:
    """Create visualization plots."""
    saved_paths = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity_curve["date"], equity_curve["equity"], linewidth=2, label="Portfolio Value")
    ax.fill_between(equity_curve["date"], equity_curve["equity"], alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("RSI Bot - Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    equity_path = output_dir / "equity_curve.png"
    plt.savefig(equity_path, dpi=100)
    plt.close()
    saved_paths["equity_curve"] = equity_path
    
    # 2. Sample stock performance with range
    if sample_tickers is None:
        sample_tickers = df["ticker"].unique()[:5]
    
    sample_data = df[df["ticker"].isin(sample_tickers)].sort_values(["ticker", "date"])
    
    fig, axes = plt.subplots(len(sample_tickers), 1, figsize=(14, 4*len(sample_tickers)))
    if len(sample_tickers) == 1:
        axes = [axes]
    
    for idx, ticker in enumerate(sample_tickers):
        ticker_data = sample_data[sample_data["ticker"] == ticker].copy()
        ax = axes[idx]
        
        # Plot momentum relative to range
        ax.plot(ticker_data["date"], ticker_data["current_return"] * 100, 
                label="3M Momentum", linewidth=2, color="blue")
        
        # Plot range bands
        ax.fill_between(
            ticker_data["date"],
            ticker_data["range_lower"] * 100,
            ticker_data["range_upper"] * 100,
            alpha=0.2, color="green", label="Trading Range"
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        
        # Mark buy/sell signals
        buys = ticker_data[ticker_data["buy_signal"] > 0]
        sells = ticker_data[ticker_data["sell_signal"] > 0]
        
        ax.scatter(buys["date"], buys["current_return"] * 100, color="green", 
                   s=100, marker="^", label="Buy Signal", zorder=5)
        ax.scatter(sells["date"], sells["current_return"] * 100, color="red", 
                   s=100, marker="v", label="Sell Signal", zorder=5)
        
        ax.set_ylabel(f"{ticker} Return (%)")
        ax.set_title(f"{ticker} - RSI Momentum vs Trading Range")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    sample_path = output_dir / "sample_stocks_ranges.png"
    plt.savefig(sample_path, dpi=100)
    plt.close()
    saved_paths["sample_stocks"] = sample_path
    
    return saved_paths


def _json_number(value: object) -> float | int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _records_for_json(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    records = []
    if df.empty:
        return records
    for row in df[columns].itertuples(index=False):
        item = {}
        for column, value in zip(columns, row):
            if column == "date":
                item[column] = pd.Timestamp(value).strftime("%Y-%m-%d")
            elif column in {"ticker", "action", "signal_type"}:
                item[column] = "" if pd.isna(value) else str(value)
            else:
                item[column] = _json_number(value)
        records.append(item)
    return records


def _metric_card(label: str, value: str) -> str:
    return f"""
                <div class="metric">
                    <span>{label}</span>
                    <strong>{value}</strong>
                </div>"""


def create_interactive_rsi_bot_report(
    result: RSIBotResult,
    recommendations_df: pd.DataFrame,
    signals_df: pd.DataFrame,
) -> Path:
    """Create an interactive Plotly HTML report matching the single-stock style."""

    output_file = result.output_dir / "rsi_bot_report.html"
    latest_date = pd.Timestamp(signals_df["date"].max()).strftime("%Y-%m-%d")
    start_date = pd.Timestamp(signals_df["date"].min()).strftime("%Y-%m-%d")
    selected = signals_df[signals_df["is_low_volatility_candidate"]].copy()
    latest_candidates = selected[selected["date"] == selected["date"].max()].copy()
    buy_count = int((recommendations_df["signal_type"] == "BUY").sum()) if not recommendations_df.empty else 0
    sell_count = int((recommendations_df["signal_type"] == "SELL").sum()) if not recommendations_df.empty else 0
    watch_count = int((recommendations_df["signal_type"] == "WATCH").sum()) if not recommendations_df.empty else 0

    equity_rows = _records_for_json(
        result.daily_pnl,
        ["date", "equity", "cash", "positions_count"],
    )
    trade_columns = ["date", "ticker", "action", "units", "price", "value", "cash_after_trade"]
    trade_rows = _records_for_json(result.positions_df, trade_columns) if not result.positions_df.empty else []
    rec_columns = [
        "date",
        "ticker",
        "signal_type",
        "strict_score",
        "close",
        "rsi",
        "momentum_3m",
        "return_1y",
        "price_vs_200d_ma",
        "volatility_rank",
        "decision_market_cap_rank",
        "dollar_volume",
    ]
    rec_rows = _records_for_json(recommendations_df, rec_columns)
    candidate_rows = _records_for_json(
        latest_candidates.sort_values("model_score", ascending=False).head(40),
        ["date", "ticker", "model_score", "rsi", "momentum_3m", "volatility_rank"],
    )

    metric_cards = "".join(
        [
            _metric_card("Latest dataset date", latest_date),
            _metric_card("Candidates today", f"{len(latest_candidates):,}"),
            _metric_card("BUY / SELL / WATCH", f"{buy_count} / {sell_count} / {watch_count}"),
            _metric_card("Total return", f"{result.summary['total_return_pct']:+.2f}%"),
            _metric_card("Max drawdown", f"{result.summary['max_drawdown_pct']:.2f}%"),
            _metric_card("Final equity", f"${result.summary['final_equity']:,.2f}"),
            _metric_card("Trades", f"{result.summary['total_trades']:,}"),
            _metric_card("Open positions", f"{result.summary['open_positions']:,}"),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSI Momentum Bot Report</title>
    <script src="{PLOTLY_CDN}"></script>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f5f7fa;
            --panel: #ffffff;
            --text: #18212b;
            --muted: #657381;
            --line: #d9e1e8;
            --accent: #0c7c59;
            --blue: #2364aa;
            --red: #c0392b;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: "Segoe UI", Arial, sans-serif;
        }}
        header {{
            background: #111820;
            color: white;
            padding: 22px 28px;
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: flex-end;
        }}
        h1 {{
            font-size: 26px;
            margin: 0 0 6px;
            letter-spacing: 0;
        }}
        .subtitle {{
            color: #bcc8d2;
            font-size: 13px;
        }}
        .period {{
            color: #dce5ec;
            font-size: 13px;
            text-align: right;
            white-space: nowrap;
        }}
        main {{
            max-width: 1480px;
            margin: 0 auto;
            padding: 18px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 16px;
        }}
        .metric {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
            min-height: 72px;
        }}
        .metric span {{
            display: block;
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .metric strong {{
            display: block;
            font-size: 18px;
            line-height: 1.2;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }}
        .chart-panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }}
        .chart-panel.wide {{ grid-column: 1 / -1; }}
        .chart {{ width: 100%; height: 420px; }}
        .chart.compact {{ height: 360px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 14px;
        }}
        caption {{
            text-align: left;
            font-weight: 700;
            padding: 14px 12px;
        }}
        th, td {{
            padding: 10px 12px;
            border-top: 1px solid var(--line);
            text-align: left;
            font-size: 13px;
            white-space: nowrap;
        }}
        th {{
            color: var(--muted);
            font-weight: 650;
        }}
        .table-wrap {{ overflow-x: auto; }}
        .BUY {{ color: var(--accent); font-weight: 700; }}
        .SELL {{ color: var(--red); font-weight: 700; }}
        .WATCH {{ color: var(--blue); font-weight: 700; }}
        @media (max-width: 900px) {{
            header {{ display: block; }}
            .period {{
                text-align: left;
                margin-top: 10px;
                white-space: normal;
            }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>RSI Momentum Bot Report</h1>
            <div class="subtitle">ASX candidate selection, latest buy/sell/watch list, and portfolio backtest.</div>
        </div>
        <div class="period">
            {start_date} to {latest_date}<br>
            {signals_df['ticker'].nunique():,} tickers loaded
        </div>
    </header>
    <main>
        <section class="metrics">
            {metric_cards}
        </section>

        <section class="grid">
            <div class="chart-panel wide"><div id="equityChart" class="chart"></div></div>
            <div class="chart-panel"><div id="cashChart" class="chart compact"></div></div>
            <div class="chart-panel"><div id="positionChart" class="chart compact"></div></div>
            <div class="chart-panel wide"><div id="recommendationChart" class="chart"></div></div>
        </section>

        <div class="table-wrap">
            <table id="recommendationsTable">
                <caption>Latest Recommendations</caption>
                <thead>
                    <tr>
                        <th>Signal</th>
                        <th>Ticker</th>
                        <th>Score</th>
                        <th>Price</th>
                        <th>RSI</th>
                        <th>3M Mom</th>
                        <th>1Y Return</th>
                        <th>vs 200D</th>
                        <th>Vol Rank</th>
                        <th>Mkt Cap Rank</th>
                        <th>Dollar Vol</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="table-wrap">
            <table id="tradesTable">
                <caption>Trades</caption>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Action</th>
                        <th>Ticker</th>
                        <th>Units</th>
                        <th>Price</th>
                        <th>Value</th>
                        <th>Cash After</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </main>

    <script>
        const equityRows = {json.dumps(equity_rows)};
        const tradeRows = {json.dumps(trade_rows)};
        const recRows = {json.dumps(rec_rows)};
        const candidateRows = {json.dumps(candidate_rows)};
        const initialCapital = {float(result.summary["initial_capital"])};

        const config = {{
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d", "autoScale2d"]
        }};
        const baseLayout = {{
            paper_bgcolor: "white",
            plot_bgcolor: "white",
            margin: {{ l: 62, r: 24, t: 52, b: 54 }},
            hovermode: "x unified",
            legend: {{ orientation: "h", y: 1.12, x: 0 }},
            font: {{ family: "Segoe UI, Arial, sans-serif", color: "#18212b" }},
            xaxis: {{ gridcolor: "#edf1f4", rangeslider: {{ visible: false }} }},
            yaxis: {{ gridcolor: "#edf1f4", zerolinecolor: "#ccd6df" }}
        }};
        function layout(title, yTitle, extra = {{}}) {{
            return {{
                ...baseLayout,
                title: {{ text: title, x: 0.02, xanchor: "left" }},
                yaxis: {{ ...baseLayout.yaxis, title: yTitle, ...(extra.yaxis || {{}}) }},
                xaxis: {{ ...baseLayout.xaxis, ...(extra.xaxis || {{}}) }},
                shapes: extra.shapes || [],
            }};
        }}

        Plotly.newPlot("equityChart", [{{
            x: equityRows.map(row => row.date),
            y: equityRows.map(row => row.equity),
            type: "scatter",
            mode: "lines",
            name: "Equity",
            line: {{ color: "#0c7c59", width: 2 }},
            fill: "tozeroy",
            fillcolor: "rgba(12, 124, 89, 0.12)",
            hovertemplate: "%{{x}}<br>Equity: $%{{y:,.2f}}<extra></extra>"
        }}], layout("Portfolio Equity", "Equity ($)", {{
            xaxis: {{ rangeslider: {{ visible: true, thickness: 0.08 }} }},
            shapes: [{{ type: "line", xref: "paper", x0: 0, x1: 1, y0: initialCapital, y1: initialCapital, line: {{ color: "#c0392b", width: 1, dash: "dash" }} }}]
        }}), config);

        Plotly.newPlot("cashChart", [{{
            x: equityRows.map(row => row.date),
            y: equityRows.map(row => row.cash),
            type: "scatter",
            mode: "lines",
            name: "Cash",
            line: {{ color: "#2364aa", width: 2 }},
            fill: "tozeroy",
            fillcolor: "rgba(35, 100, 170, 0.12)",
            hovertemplate: "%{{x}}<br>Cash: $%{{y:,.2f}}<extra></extra>"
        }}], layout("Cash Balance", "Cash ($)"), config);

        Plotly.newPlot("positionChart", [{{
            x: equityRows.map(row => row.date),
            y: equityRows.map(row => row.positions_count),
            type: "scatter",
            mode: "lines",
            name: "Positions",
            line: {{ color: "#b26a00", width: 2, shape: "hv" }},
            fill: "tozeroy",
            fillcolor: "rgba(230, 126, 34, 0.18)",
            hovertemplate: "%{{x}}<br>Positions: %{{y:,.0f}}<extra></extra>"
        }}], layout("Open Positions", "Count"), config);

        Plotly.newPlot("recommendationChart", [{{
            x: candidateRows.map(row => row.ticker),
            y: candidateRows.map(row => row.model_score),
            type: "bar",
            name: "Model score",
            marker: {{ color: "#2364aa" }},
            customdata: candidateRows.map(row => [row.rsi, row.momentum_3m, row.volatility_rank]),
            hovertemplate: "%{{x}}<br>Score: %{{y:.3f}}<br>RSI: %{{customdata[0]:.1f}}<br>3M Mom: %{{customdata[1]:+.2%}}<br>Vol Rank: %{{customdata[2]:.0f}}<extra></extra>"
        }}], layout("Top Current Candidates", "Model Score", {{ xaxis: {{ tickangle: -45 }} }}), config);

        function money(value) {{
            if (value === null || value === undefined) return "";
            return "$" + value.toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
        }}
        function pct(value) {{
            if (value === null || value === undefined) return "";
            return (value * 100).toFixed(2) + "%";
        }}
        function num(value, digits = 2) {{
            if (value === null || value === undefined) return "";
            return value.toLocaleString(undefined, {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
        }}

        const recBody = document.querySelector("#recommendationsTable tbody");
        if (recRows.length === 0) {{
            recBody.innerHTML = '<tr><td colspan="11">No current candidates passed the filters.</td></tr>';
        }} else {{
            recBody.innerHTML = recRows.map(row => `
                <tr>
                    <td class="${{row.signal_type}}">${{row.signal_type}}</td>
                    <td>${{row.ticker}}</td>
                    <td>${{num(row.strict_score, 3)}}</td>
                    <td>${{money(row.close)}}</td>
                    <td>${{num(row.rsi, 1)}}</td>
                    <td>${{pct(row.momentum_3m)}}</td>
                    <td>${{pct(row.return_1y)}}</td>
                    <td>${{pct(row.price_vs_200d_ma)}}</td>
                    <td>${{num(row.volatility_rank, 0)}}</td>
                    <td>${{num(row.decision_market_cap_rank, 0)}}</td>
                    <td>${{money(row.dollar_volume)}}</td>
                </tr>
            `).join("");
        }}

        const tradeBody = document.querySelector("#tradesTable tbody");
        if (tradeRows.length === 0) {{
            tradeBody.innerHTML = '<tr><td colspan="7">No trades were generated.</td></tr>';
        }} else {{
            tradeBody.innerHTML = tradeRows.slice().reverse().map(row => `
                <tr>
                    <td>${{row.date}}</td>
                    <td class="${{row.action}}">${{row.action}}</td>
                    <td>${{row.ticker}}</td>
                    <td>${{num(row.units, 0)}}</td>
                    <td>${{money(row.price)}}</td>
                    <td>${{money(row.value)}}</td>
                    <td>${{money(row.cash_after_trade)}}</td>
                </tr>
            `).join("");
        }}
    </script>
</body>
</html>
"""
    output_file.write_text(html, encoding="utf-8")
    return output_file


def execute_rsi_bot_model(
    run_direct_benchmark_test: bool = True,
    run_hedged_short_benchmark_test: bool = False,
) -> RSIBotResult:
    """Main execution function."""
    
    print("\n" + "="*60)
    print("RSI MOMENTUM BOT - STARTING EXECUTION")
    print("="*60)
    
    # Load configuration
    paths = PathConfig()
    
    # Load data
    print(f"\nLoading data from {paths.input_csv}...")
    available_columns = pd.read_csv(paths.input_csv, nrows=0).columns.tolist()
    usecols = ["date", "ticker", "close", "dollar_volume"]
    if "market_cap_rank" in available_columns:
        usecols.append("market_cap_rank")
    df = pd.read_csv(
        paths.input_csv,
        usecols=usecols,
        parse_dates=["date"],
        low_memory=False
    )
    
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce").astype("float32")
    if "market_cap_rank" in df.columns:
        df["market_cap_rank"] = pd.to_numeric(df["market_cap_rank"], errors="coerce").astype("float32")
    df = df.dropna(subset=["close", "dollar_volume"])
    df = df[df["close"] >= MIN_PRICE]
    
    print(f"Loaded {len(df)} records for {df['ticker'].nunique()} tickers")
    
    # Generate signals
    print("\nGenerating RSI signals and features...")
    model = RSIBuySignal()
    signals_df = model.generate_signals(df)
    
    # Filter for low volatility stocks
    print(f"\nFiltering for top {TOP_STOCKS} low volatility stocks...")
    filtered_df = filter_low_volatility_stocks(signals_df, top_n=TOP_STOCKS)
    candidate_days = int(filtered_df["is_low_volatility_candidate"].sum())
    candidate_tickers = int(filtered_df.loc[filtered_df["is_low_volatility_candidate"], "ticker"].nunique())
    print(f"Marked {candidate_days} stock-days across {candidate_tickers} stocks")
    
    # Run backtest
    print("\nRunning backtest...")
    result = backtest_rsi_bot(filtered_df, initial_capital=INITIAL_CAPITAL)

    print("\nRanking latest recommendations...")
    recommendations_df = latest_strict_recommendations(filtered_df)
    plotted_signals_df = generate_buy_sell_signals(filtered_df)

    # Generate plots
    print("\nGenerating visualizations...")
    sample_tickers = filtered_df["ticker"].unique()[:8]
    plot_paths = plot_strategy(plotted_signals_df, result.daily_pnl, result.output_dir, sample_tickers)
    result.saved_paths.update(plot_paths)
    
    # Save results
    print("\nSaving results...")
    result.positions_df.to_csv(result.output_dir / "positions.csv", index=False)
    result.daily_pnl.to_csv(result.output_dir / "daily_pnl.csv", index=False)
    recommendations_df.to_csv(result.output_dir / "latest_recommendations.csv", index=False)
    report_path = create_interactive_rsi_bot_report(
        result=result,
        recommendations_df=recommendations_df,
        signals_df=plotted_signals_df,
    )
    result.saved_paths["positions"] = result.output_dir / "positions.csv"
    result.saved_paths["daily_pnl"] = result.output_dir / "daily_pnl.csv"
    result.saved_paths["latest_recommendations"] = result.output_dir / "latest_recommendations.csv"
    result.saved_paths["html_report"] = report_path
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    for key, value in result.summary.items():
        if "pct" in key:
            print(f"{key:.<40} {value:>10.2f}%")
        elif "capital" in key or "equity" in key or "cash" in key:
            print(f"{key:.<40} ${value:>10,.2f}")
        else:
            print(f"{key:.<40} {value:>10}")
    
    print("\nSaved files:")
    for name, path in result.saved_paths.items():
        print(f"  - {name}: {path}")
    if recommendations_df.empty:
        print("\nLatest recommendations: none")
    else:
        print("\nLatest recommendations:")
        print(recommendations_df.to_string(index=False))
    
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    result = execute_rsi_bot_model(
        run_direct_benchmark_test=True,
        run_hedged_short_benchmark_test=False
    )
