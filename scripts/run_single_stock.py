#!/usr/bin/env python3
"""Run RSI bot backtest on a single stock (any exchange)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rsi_momentum_bot import (
    RSIBuySignal,
    generate_buy_sell_signals,
    MIN_HOLDING_TRADING_DAYS,
)
from models.long_term_trend import (
    LONG_TERM_BUY_COOLDOWN_DAYS,
    LONG_TERM_MIN_HOLDING_DAYS,
    LongTermTrendSignal,
    generate_long_term_signals,
)


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MIN_POSITION_PCT = 0.08
MAX_POSITION_PCT = 0.35
MAX_TICKER_EXPOSURE_PCT = 0.35
MAX_OPEN_LOTS = 3
BUY_COOLDOWN_TRADING_DAYS = 10
PYRAMID_CONFIRMATION_RETURN = 0.02
SLIPPAGE_BPS = 5.0
COMMISSION_RATE = 0.0009
MIN_COMMISSION = 7.0
ANNUAL_RISK_FREE_RATE = 0.04
CALM_DAILY_VOLATILITY = 0.015
HIGH_DAILY_VOLATILITY = 0.045
WIDE_RANGE_THRESHOLD = 0.75
DEEP_RANGE_BREAK_THRESHOLD = 0.40
ASX_TICKER_ALIASES = {
    "CBA": "CBA.AX",
}


def normalize_session_dates(values: pd.Series) -> pd.Series:
    """Drop timezone metadata without shifting the exchange trading date."""

    def normalize(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        return timestamp.normalize()

    return values.map(normalize)


def resolve_yahoo_ticker(ticker: str) -> str:
    """Resolve common ASX tickers to their Yahoo Finance symbols."""

    normalized = ticker.upper()
    return ASX_TICKER_ALIASES.get(normalized, normalized)


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    cache_dir = Path(".cache/yfinance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir.resolve()))
    requested_ticker = ticker.upper()
    yahoo_ticker = resolve_yahoo_ticker(requested_ticker)
    local_file = Path("data/downloaded_stocks") / f"{yahoo_ticker}.csv"
    if local_file.exists():
        print(f"Loading {requested_ticker} data from {local_file}...")
        df = pd.read_csv(local_file)
        df = df.rename(columns=str.lower)
        df["date"] = normalize_session_dates(df["date"])
        df["ticker"] = yahoo_ticker
        for column in ["open", "high", "low", "close", "adj_close", "volume"]:
            if column in df:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "adj_close" in df and df["adj_close"].notna().any():
            adjustment = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan)
            for column in ["open", "high", "low", "close"]:
                if column in df:
                    df[column] = df[column] * adjustment
            df["close"] = df["adj_close"]
        if "open" not in df:
            df["open"] = df["close"]
        if "dollar_volume" not in df.columns:
            df["dollar_volume"] = df["close"] * df["volume"]
        df = df[["date", "open", "close", "volume", "ticker", "dollar_volume"]].copy()
        df["open"] = pd.to_numeric(df["open"], errors="coerce").astype("float32")
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
        df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce").astype("float32")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

        if period != "max":
            period_days = {
                "1d": 1,
                "5d": 5,
                "1mo": 31,
                "3mo": 93,
                "6mo": 186,
                "1y": 366,
                "2y": 366 * 2,
                "5y": 366 * 5,
                "10y": 366 * 10,
                "ytd": None,
            }
            if period == "ytd":
                start = pd.Timestamp.now().replace(month=1, day=1)
                df = df[df["date"] >= start]
            elif period in period_days:
                cutoff = df["date"].max() - pd.Timedelta(days=period_days[period])
                df = df[df["date"] >= cutoff]

        print(f"Loaded {len(df)} days of data for {requested_ticker} ({yahoo_ticker})")
        return df.reset_index(drop=True)

    print(f"Fetching {requested_ticker} data from Yahoo Finance as {yahoo_ticker} ({period})...")
    
    stock = yf.Ticker(yahoo_ticker)
    data = stock.history(period=period, auto_adjust=True)
    
    if data.empty:
        raise ValueError(f"No data found for {requested_ticker} ({yahoo_ticker})")
    
    # Reset index to make date a column
    data = data.reset_index()
    
    # Rename columns to lowercase
    data.columns = data.columns.str.lower()
    
    df = pd.DataFrame({
        "date": data["date"],
        "open": data["open"],
        "close": data["close"],
        "volume": data["volume"],
        "ticker": yahoo_ticker,
    })
    
    # Add dollar volume (close * volume)
    df["dollar_volume"] = df["close"] * df["volume"]
    
    df["close"] = df["close"].astype("float32")
    df["dollar_volume"] = df["dollar_volume"].astype("float32")
    df["date"] = normalize_session_dates(df["date"])
    
    print(f"Downloaded {len(df)} days of data for {requested_ticker} ({yahoo_ticker})")
    return df


def attach_market_context(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Attach an adjusted ASX 200 series for regime and relative-strength tests."""

    try:
        benchmark = yf.Ticker("^AXJO").history(period=period, auto_adjust=True)
        if benchmark.empty:
            return df
        benchmark = benchmark.reset_index()
        benchmark.columns = benchmark.columns.str.lower()
        benchmark = benchmark[["date", "close"]].rename(
            columns={"close": "benchmark_close"}
        )
        benchmark["date"] = normalize_session_dates(benchmark["date"])
        return pd.merge_asof(
            df.sort_values("date"),
            benchmark.sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=4),
        )
    except Exception as exc:
        print(f"Warning: ASX 200 context unavailable: {exc}")
        return df


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if pd.isna(value):
        return lower
    return max(lower, min(upper, value))


def _row_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def risk_adjusted_position_pct(row: pd.Series) -> tuple[float, float, str]:
    """Size a buy from current risk data instead of using a fixed dollar amount."""

    volatility = _row_float(row, "volatility")
    range_upper = abs(_row_float(row, "range_upper"))
    range_lower = _row_float(row, "range_lower")
    current_return = _row_float(row, "current_return")

    volatility_risk = _bounded(
        (volatility - CALM_DAILY_VOLATILITY)
        / (HIGH_DAILY_VOLATILITY - CALM_DAILY_VOLATILITY)
    )
    range_risk = _bounded(range_upper / WIDE_RANGE_THRESHOLD)
    downside_break = max(0.0, range_lower - current_return)
    break_risk = _bounded(downside_break / DEEP_RANGE_BREAK_THRESHOLD)

    risk_score = _bounded(
        (volatility_risk * 0.50)
        + (range_risk * 0.30)
        + (break_risk * 0.20)
    )
    position_pct = MAX_POSITION_PCT - (
        risk_score * (MAX_POSITION_PCT - MIN_POSITION_PCT)
    )

    reason = (
        f"vol {volatility * 100:.2f}%, "
        f"range +/-{range_upper * 100:.1f}%, "
        f"break {downside_break * 100:.1f}%"
    )
    return position_pct, risk_score, reason


def trade_cost(notional: float) -> float:
    """Return configured brokerage for one side of a trade."""

    return max(MIN_COMMISSION, notional * COMMISSION_RATE) if notional > 0 else 0.0


def execution_price(row: pd.Series, side: str) -> float:
    """Execute an already-lagged signal at the session open with slippage."""

    market_open = _row_float(row, "open", _row_float(row, "close"))
    direction = 1.0 if side.upper() == "BUY" else -1.0
    return market_open * (1.0 + direction * SLIPPAGE_BPS / 10_000.0)


def calculate_performance_metrics(
    portfolio_history: list[dict],
    trades: list[dict],
    realized_returns: list[float],
    initial_capital: float,
) -> dict:
    """Calculate risk, exposure, turnover, and trade-quality statistics."""

    history = pd.DataFrame(portfolio_history)
    if history.empty:
        return {}
    equity = history["portfolio_value"].astype(float)
    daily_returns = equity.pct_change().fillna(0.0)
    years = max(len(history) / 252.0, 1.0 / 252.0)
    total_return = equity.iloc[-1] / initial_capital - 1.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
    daily_std = daily_returns.std(ddof=0)
    annual_volatility = daily_std * math.sqrt(252.0)
    excess_daily = daily_returns - ANNUAL_RISK_FREE_RATE / 252.0
    sharpe = (
        excess_daily.mean() / daily_std * math.sqrt(252.0)
        if daily_std > 0
        else 0.0
    )
    downside = excess_daily.clip(upper=0.0)
    downside_deviation = math.sqrt(float((downside ** 2).mean())) * math.sqrt(252.0)
    sortino = (
        (cagr - ANNUAL_RISK_FREE_RATE) / downside_deviation
        if downside_deviation > 0
        else 0.0
    )
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = abs(float(drawdown.min()))
    wins = [value for value in realized_returns if value > 0]
    losses = [value for value in realized_returns if value < 0]
    gross_loss = abs(sum(losses))
    return {
        "cagr_pct": cagr * 100.0,
        "annual_volatility_pct": annual_volatility * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": cagr / max_drawdown if max_drawdown > 0 else 0.0,
        "market_exposure_pct": float((history["shares"] > 0).mean() * 100.0),
        "turnover_x": (
            sum(float(trade["value"]) for trade in trades) / float(equity.mean())
        ),
        "completed_lots": len(realized_returns),
        "win_rate_pct": len(wins) / len(realized_returns) * 100.0 if realized_returns else 0.0,
        "profit_factor": sum(wins) / gross_loss if gross_loss > 0 else 0.0,
        "average_lot_return_pct": (
            float(np.mean(realized_returns)) * 100.0 if realized_returns else 0.0
        ),
        "total_fees": sum(float(trade.get("commission", 0.0)) for trade in trades),
        "slippage_bps": SLIPPAGE_BPS,
    }


STRATEGIES = {
    "medium-term": {
        "label": "Medium-Term RSI",
        "model": RSIBuySignal,
        "signal_generator": generate_buy_sell_signals,
        "minimum_holding_days": MIN_HOLDING_TRADING_DAYS,
        "buy_cooldown_days": BUY_COOLDOWN_TRADING_DAYS,
        "profit_first_exits": True,
    },
    "long-term": {
        "label": "Long-Term Trend",
        "model": LongTermTrendSignal,
        "signal_generator": generate_long_term_signals,
        "minimum_holding_days": LONG_TERM_MIN_HOLDING_DAYS,
        "buy_cooldown_days": LONG_TERM_BUY_COOLDOWN_DAYS,
        "profit_first_exits": False,
    },
}
STRATEGY_ALIASES = {
    "medium": "medium-term",
    "medium_term": "medium-term",
    "long": "long-term",
    "long_term": "long-term",
}


def normalize_strategy(strategy: str) -> str:
    """Return a canonical strategy key or explain the valid choices."""
    normalized = strategy.strip().lower()
    normalized = STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in STRATEGIES:
        choices = ", ".join(STRATEGIES)
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of: {choices}")
    return normalized


def analyze_single_stock(
    ticker: str,
    period: str = "5y",
    strategy: str = "medium-term",
) -> dict:
    """Analyze and backtest a single stock."""
    strategy = normalize_strategy(strategy)
    strategy_config = STRATEGIES[strategy]
    
    # Fetch data
    df = attach_market_context(fetch_stock_data(ticker, period), period)
    
    # Generate signals
    print(f"\nGenerating {strategy_config['label']} features for {ticker}...")
    model = strategy_config["model"]()
    signals_df = model.generate_signals(df)
    
    # Generate buy/sell signals
    print("Generating buy/sell signals...")
    signals_df = strategy_config["signal_generator"](signals_df)
    
    # Get latest stats
    latest = signals_df.iloc[-1]
    
    # Count signals
    buy_signals = (signals_df["buy_signal"] > 0).sum()
    sell_signals = (signals_df["sell_signal"] > 0).sum()
    
    stats = {
        "ticker": ticker,
        "strategy": strategy,
        "strategy_name": strategy_config["label"],
        "data_start": signals_df["date"].min(),
        "data_end": signals_df["date"].max(),
        "trading_days": len(signals_df),
        "latest_date": latest["date"],
        "latest_price": float(latest["close"]),
        "rsi": float(latest["rsi"]),
        "volatility": float(latest["volatility"]),
        "momentum_3m": float(latest["momentum_3m"]),
        "return_1w": float(latest["return_1w"]),
        "yearly_implied_weekly_return": float(latest["yearly_implied_weekly_return"]),
        "weekly_relative_momentum": float(latest["weekly_relative_momentum"]),
        "return_1y": float(latest["return_1y"]),
        "range_lower": float(latest["range_lower"]),
        "range_upper": float(latest["range_upper"]),
        "current_return": float(latest["current_return"]),
        "total_buy_signals": buy_signals,
        "total_sell_signals": sell_signals,
    }
    
    return {
        "stats": stats,
        "data": signals_df,
    }


def simple_backtest(
    signals_df: pd.DataFrame,
    initial_capital: float = 10_000,
    minimum_holding_days: int = MIN_HOLDING_TRADING_DAYS,
    buy_cooldown_days: int = BUY_COOLDOWN_TRADING_DAYS,
    profit_first_exits: bool = True,
) -> dict:
    """Run a lagged next-open, cost-aware, profit-first lot backtest.

    Once a lot has completed its minimum hold, it may be sold outside the
    indicator bands as soon as the trade is profitable after both entry and
    estimated exit commission.  The strategy's original sell signal remains
    the fallback, so a weakening position is not held merely to manufacture a
    higher win rate.
    """

    cash = initial_capital
    lots = []
    trades = []
    portfolio_values = []
    realized_returns = []
    last_buy_index = -buy_cooldown_days

    for idx, row in signals_df.iterrows():
        current_price = float(row["close"])
        risk_exit = bool(row.get("risk_exit_signal", False))
        eligible_lots = [
            lot for lot in lots
            if risk_exit or idx - lot["entry_index"] >= minimum_holding_days
        ]
        fill_price = execution_price(row, "SELL")
        profitable_lots = []
        if profit_first_exits:
            for lot in eligible_lots:
                entry_cost = (
                    lot["shares"] * lot["entry_price"]
                    + lot.get("entry_commission", 0.0)
                )
                projected_proceeds = lot["shares"] * fill_price
                projected_net_proceeds = projected_proceeds - trade_cost(projected_proceeds)
                if projected_net_proceeds > entry_cost:
                    profitable_lots.append(lot)

        original_exit = row["sell_signal"] > 0
        lots_to_sell = eligible_lots if original_exit else profitable_lots
        if lots_to_sell:
            sold_shares = sum(lot["shares"] for lot in lots_to_sell)
            proceeds = sold_shares * fill_price
            commission = trade_cost(proceeds)
            cash += proceeds - commission
            for lot in lots_to_sell:
                commission_share = commission * lot["shares"] / sold_shares
                entry_cost = (
                    lot["shares"] * lot["entry_price"]
                    + lot.get("entry_commission", 0.0)
                )
                lot_net_proceeds = lot["shares"] * fill_price - commission_share
                realized_returns.append(lot_net_proceeds / entry_cost - 1.0)
            trades.append({
                "date": row["date"],
                "action": "SELL",
                "price": fill_price,
                "shares": sold_shares,
                "value": proceeds,
                "commission": commission,
                "position_pct": None,
                "risk_score": None,
                "sizing_reason": (
                    f"{'original exit signal' if original_exit else 'profit-first exit outside indicator bands'}; "
                    f"{len(lots_to_sell)} matured lot(s), "
                    + (
                        "immediate risk exit"
                        if risk_exit
                        else f"minimum hold {minimum_holding_days} trading days"
                    )
                ),
            })
            eligible_ids = {id(lot) for lot in lots_to_sell}
            lots = [lot for lot in lots if id(lot) not in eligible_ids]

        can_buy = (
            row["buy_signal"] > 0
            and cash > 0
            and len(lots) < MAX_OPEN_LOTS
            and idx - last_buy_index >= buy_cooldown_days
        )
        if can_buy:
            held_shares = sum(lot["shares"] for lot in lots)
            portfolio_value = cash + (held_shares * current_price)
            position_pct, risk_score, sizing_reason = risk_adjusted_position_pct(row)
            fill_price = execution_price(row, "BUY")
            current_exposure = held_shares * current_price
            exposure_room = max(
                0.0,
                portfolio_value * MAX_TICKER_EXPOSURE_PCT - current_exposure,
            )
            investment = min(portfolio_value * position_pct, exposure_room, cash)
            if investment < portfolio_value * MIN_POSITION_PCT:
                investment = 0.0
            new_shares = int(investment / fill_price)
            while new_shares > 0:
                cost = new_shares * fill_price
                commission = trade_cost(cost)
                if cost + commission <= cash:
                    break
                new_shares -= 1

            if new_shares > 0:
                cost = new_shares * fill_price
                commission = trade_cost(cost)
                cash -= cost + commission
                lots.append({
                    "shares": new_shares,
                    "entry_index": idx,
                    "entry_date": row["date"],
                    "entry_price": fill_price,
                    "entry_commission": commission,
                })
                last_buy_index = idx
                trades.append({
                    "date": row["date"],
                    "action": "BUY",
                    "price": fill_price,
                    "shares": new_shares,
                    "value": cost,
                    "commission": commission,
                    "position_pct": position_pct,
                    "risk_score": risk_score,
                    "sizing_reason": sizing_reason,
                })

        held_shares = sum(lot["shares"] for lot in lots)
        portfolio_value = cash + (held_shares * current_price)
        portfolio_values.append({
            "date": row["date"],
            "portfolio_value": portfolio_value,
            "shares": held_shares,
            "cash": cash,
        })
    
    # Mark open lots to market; do not create a fake end-of-test sale.
    final_price = signals_df.iloc[-1]["close"]
    final_value = cash + sum(lot["shares"] for lot in lots) * final_price
    total_return = ((final_value / initial_capital) - 1) * 100
    metrics = calculate_performance_metrics(
        portfolio_values,
        trades,
        realized_returns,
        initial_capital,
    )
    first_price = float(signals_df.iloc[0]["close"])
    buy_hold_return = final_price / first_price - 1.0
    years = max(len(signals_df) / 252.0, 1.0 / 252.0)
    buy_hold_cagr = (1.0 + buy_hold_return) ** (1.0 / years) - 1.0
    benchmark_values = (
        signals_df["benchmark_close"].dropna()
        if "benchmark_close" in signals_df
        else pd.Series(dtype=float)
    )
    benchmark_return = (
        float(benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1.0)
        if len(benchmark_values) > 1
        else float("nan")
    )

    results = {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "trades": trades,
        "portfolio_history": portfolio_values,
        "buy_count": len([t for t in trades if t["action"] == "BUY"]),
        "sell_count": len([t for t in trades if t["action"] == "SELL"]),
        "minimum_holding_trading_days": minimum_holding_days,
        "open_lots": len(lots),
        "max_ticker_exposure_pct": MAX_TICKER_EXPOSURE_PCT * 100.0,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "buy_hold_cagr_pct": buy_hold_cagr * 100.0,
        "asx200_return_pct": benchmark_return * 100.0,
        "excess_vs_buy_hold_pct": total_return - buy_hold_return * 100.0,
        **metrics,
    }

    return results


def print_stats(stats: dict) -> None:
    """Print analysis statistics."""
    print("\n" + "="*70)
    print(f"{stats.get('strategy_name', 'Medium-Term RSI').upper()} ANALYSIS - {stats['ticker']}")
    print("="*70)
    print(f"Period:            {stats['data_start'].date()} -> {stats['data_end'].date()}")
    print(f"Trading Days:      {stats['trading_days']}")
    print(f"\nLatest ({stats['latest_date'].date()}):")
    print(f"  Price:           ${stats['latest_price']:.2f}")
    print(f"  RSI(14):         {stats['rsi']:.1f} {'(Overbought)' if stats['rsi'] > 70 else '(Oversold)' if stats['rsi'] < 30 else '(Neutral)'}")
    print(f"  3M Momentum:     {stats['momentum_3m']*100:+.2f}%")
    print(f"  1W Return:       {stats['return_1w']*100:+.2f}%")
    print(f"  1W vs 1Y Pace:   {stats['weekly_relative_momentum']*100:+.2f}%")
    print(f"  1Y Return:       {stats['return_1y']*100:+.2f}%")
    print(f"  Volatility:      {stats['volatility']*100:.2f}%")
    print(f"  Trading Range:   {stats['range_lower']*100:+.2f}% to {stats['range_upper']*100:+.2f}%")
    print(f"  Current Pos:     {stats['current_return']*100:+.2f}% {'(BELOW range)' if stats['current_return'] < stats['range_lower'] else '(ABOVE range)' if stats['current_return'] > stats['range_upper'] else '(IN range)'}")
    print(f"\nSignals Generated:")
    print(f"  Buy Signals:     {stats['total_buy_signals']}")
    print(f"  Sell Signals:    {stats['total_sell_signals']}")
    print("="*70 + "\n")


def print_backtest_results(results: dict) -> None:
    """Print backtest results."""
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(f"Initial Capital:   ${results['initial_capital']:,.2f}")
    print(f"Final Value:       ${results['final_value']:,.2f}")
    print(f"Total Return:      {results['total_return_pct']:+.2f}%")
    print(f"Buy & Hold:        {results['buy_hold_return_pct']:+.2f}%")
    if not pd.isna(results["asx200_return_pct"]):
        print(f"ASX 200:           {results['asx200_return_pct']:+.2f}%")
    print(f"Excess Return:     {results['excess_vs_buy_hold_pct']:+.2f}%")
    print(f"Max Drawdown:      {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe / Sortino:  {results['sharpe_ratio']:.2f} / {results['sortino_ratio']:.2f}")
    print(f"Cost-aware Wins:   {results['win_rate_pct']:.1f}% of completed lots")
    print(f"Fees:              ${results['total_fees']:,.2f}")
    print(f"Buy/Hold Trades:   {results['buy_count']} buys, {results['sell_count']} sells")
    print(f"Minimum Hold:      {results['minimum_holding_trading_days']} trading days per buy")
    print(f"Open Buy Lots:     {results['open_lots']}")
    
    if len(results["trades"]) > 0:
        print(f"\nRecent Trades:")
        for trade in results["trades"][-10:]:  # Show last 10
            position_pct = trade.get("position_pct")
            sizing = (
                f" | size {position_pct * 100:4.1f}% | {trade.get('sizing_reason', '')}"
                if position_pct is not None
                else f" | {trade.get('sizing_reason', '')}"
            )
            print(f"  {trade['date'].date()} - {trade['action']:4s} "
                  f"{trade['shares']:6.0f} @ ${trade['price']:8.2f} = ${trade['value']:10,.2f}{sizing}")
        if len(results["trades"]) > 10:
            print(f"  ... and {len(results['trades']) - 10} more trades")
    
    print("="*70 + "\n")


def _clean_number(value: object) -> float | None:
    """Convert pandas/numpy values into JSON-safe numbers."""

    if pd.isna(value):
        return None
    return float(value)


def _chart_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Return compact, JSON-safe chart rows."""

    records = []
    for row in df[columns].itertuples(index=False):
        item = {}
        for column, value in zip(columns, row):
            if column == "date":
                item[column] = pd.Timestamp(value).strftime("%Y-%m-%d")
            else:
                item[column] = _clean_number(value)
        records.append(item)
    return records


def _summary_card(label: str, value: str) -> str:
    return f"""
                <div class="metric">
                    <span>{label}</span>
                    <strong>{value}</strong>
                </div>"""


def create_interactive_report(ticker: str, signals_df: pd.DataFrame, backtest_results: dict, stats: dict) -> Path:
    """Create an interactive Plotly HTML report."""

    output_dir = Path("results/single_stock")
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy = stats.get("strategy", "medium-term")
    suffix = "" if strategy == "medium-term" else f"_{strategy.replace('-', '_')}"
    output_file = output_dir / f"{ticker}{suffix}_report.html"
    strategy_name = stats.get("strategy_name", "Medium-Term RSI")

    chart_columns = [
        "date",
        "close",
        "rsi",
        "volatility",
        "current_return",
        "range_lower",
        "range_upper",
        "buy_signal",
        "sell_signal",
    ]
    for optional_column in ["ma_50d", "ma_200d"]:
        if optional_column in signals_df.columns:
            chart_columns.append(optional_column)
    if "benchmark_close" in signals_df.columns:
        chart_columns.append("benchmark_close")
    chart_data = _chart_records(signals_df, chart_columns)
    portfolio_history = _chart_records(
        pd.DataFrame(backtest_results["portfolio_history"]),
        ["date", "portfolio_value", "shares", "cash"],
    )
    trades = [
        {
            "date": pd.Timestamp(trade["date"]).strftime("%Y-%m-%d"),
            "action": trade["action"],
            "price": float(trade["price"]),
            "shares": int(trade["shares"]),
            "value": float(trade["value"]),
            "commission": float(trade.get("commission", 0.0)),
            "position_pct": (
                None
                if trade.get("position_pct") is None
                else float(trade["position_pct"])
            ),
            "risk_score": (
                None
                if trade.get("risk_score") is None
                else float(trade["risk_score"])
            ),
            "sizing_reason": trade.get("sizing_reason", ""),
        }
        for trade in backtest_results["trades"]
    ]

    range_status = (
        "BELOW range"
        if stats["current_return"] < stats["range_lower"]
        else "ABOVE range"
        if stats["current_return"] > stats["range_upper"]
        else "IN range"
    )
    metric_cards = "".join(
        [
            _summary_card("Latest price", f"${stats['latest_price']:,.2f}"),
            _summary_card("RSI(14)", f"{stats['rsi']:.1f}"),
            _summary_card("3M momentum", f"{stats['momentum_3m'] * 100:+.2f}%"),
            _summary_card("1W vs 1Y pace", f"{stats['weekly_relative_momentum'] * 100:+.2f}%"),
            _summary_card("1Y return", f"{stats['return_1y'] * 100:+.2f}%"),
            _summary_card("Volatility", f"{stats['volatility'] * 100:.2f}%"),
            _summary_card("Range status", range_status),
            _summary_card("Buy signals", f"{stats['total_buy_signals']:,}"),
            _summary_card("Sell signals", f"{stats['total_sell_signals']:,}"),
            _summary_card("Backtest return", f"{backtest_results['total_return_pct']:+.2f}%"),
            _summary_card("Buy-and-hold", f"{backtest_results['buy_hold_return_pct']:+.2f}%"),
            _summary_card(
                "ASX 200",
                (
                    f"{backtest_results['asx200_return_pct']:+.2f}%"
                    if not pd.isna(backtest_results["asx200_return_pct"])
                    else "Unavailable"
                ),
            ),
            _summary_card("Excess vs hold", f"{backtest_results['excess_vs_buy_hold_pct']:+.2f}%"),
            _summary_card("CAGR", f"{backtest_results['cagr_pct']:+.2f}%"),
            _summary_card("Max drawdown", f"{backtest_results['max_drawdown_pct']:.2f}%"),
            _summary_card("Sharpe", f"{backtest_results['sharpe_ratio']:.2f}"),
            _summary_card("Sortino", f"{backtest_results['sortino_ratio']:.2f}"),
            _summary_card("Market exposure", f"{backtest_results['market_exposure_pct']:.1f}%"),
            _summary_card("Win rate", f"{backtest_results['win_rate_pct']:.1f}%"),
            _summary_card("Total fees", f"${backtest_results['total_fees']:,.2f}"),
            _summary_card("Final value", f"${backtest_results['final_value']:,.2f}"),
            _summary_card(
                "Minimum hold",
                f"{backtest_results['minimum_holding_trading_days']} trading days",
            ),
            _summary_card("Open buy lots", f"{backtest_results['open_lots']:,}"),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ticker} Interactive {strategy_name} Analysis</title>
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

        .chart-panel.wide {{
            grid-column: 1 / -1;
        }}

        .chart {{
            width: 100%;
            height: 420px;
        }}

        .chart.compact {{
            height: 360px;
        }}

        .trades {{
            width: 100%;
            border-collapse: collapse;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            margin-top: 14px;
        }}

        .trades caption {{
            text-align: left;
            font-weight: 700;
            padding: 14px 12px;
        }}

        .trades th,
        .trades td {{
            padding: 10px 12px;
            border-top: 1px solid var(--line);
            text-align: left;
            font-size: 13px;
        }}

        .trades th {{
            color: var(--muted);
            font-weight: 650;
        }}

        @media (max-width: 900px) {{
            header {{
                display: block;
            }}

            .period {{
                text-align: left;
                margin-top: 10px;
                white-space: normal;
            }}

            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>{ticker} Interactive {strategy_name} Analysis</h1>
            <div class="subtitle">Hover, zoom, pan, box-select, and use the built-in Plotly toolbar on every graph.</div>
        </div>
        <div class="period">
            {stats['data_start'].date()} to {stats['data_end'].date()}<br>
            {stats['trading_days']:,} trading days
        </div>
    </header>

    <main>
        <section class="metrics">
            {metric_cards}
        </section>

        <section class="grid">
            <div class="chart-panel wide"><div id="priceChart" class="chart"></div></div>
            <div class="chart-panel"><div id="rsiChart" class="chart compact"></div></div>
            <div class="chart-panel"><div id="volatilityChart" class="chart compact"></div></div>
            <div class="chart-panel wide"><div id="momentumChart" class="chart"></div></div>
            <div class="chart-panel wide"><div id="portfolioChart" class="chart"></div></div>
        </section>

        <table class="trades" id="tradesTable">
            <caption>Trades</caption>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Action</th>
                    <th>Shares</th>
                    <th>Price</th>
                    <th>Value</th>
                    <th>Fee</th>
                    <th>Size</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </main>

    <script>
        const rows = {json.dumps(chart_data)};
        const portfolio = {json.dumps(portfolio_history)};
        const trades = {json.dumps(trades)};
        const initialCapital = {float(backtest_results["initial_capital"])};

        const dates = rows.map(row => row.date);
        const close = rows.map(row => row.close);
        const buyHold = rows.map(row => initialCapital * row.close / rows[0].close);
        const firstBenchmark = rows.find(row => row.benchmark_close !== null)?.benchmark_close;
        const asxHold = rows.map(row =>
            firstBenchmark && row.benchmark_close !== null
                ? initialCapital * row.benchmark_close / firstBenchmark
                : null
        );
        const rsi = rows.map(row => row.rsi);
        const volatilityPct = rows.map(row => row.volatility === null ? null : row.volatility * 100);
        const momentumPct = rows.map(row => row.current_return === null ? null : row.current_return * 100);
        const lowerPct = rows.map(row => row.range_lower === null ? null : row.range_lower * 100);
        const upperPct = rows.map(row => row.range_upper === null ? null : row.range_upper * 100);
        const buys = rows.filter(row => row.buy_signal > 0);
        const sells = rows.filter(row => row.sell_signal > 0);
        const trendTraces = rows.some(row => row.ma_50d !== undefined) ? [
            {{
                x: dates,
                y: rows.map(row => row.ma_50d),
                type: "scatter",
                mode: "lines",
                name: "50-day average",
                line: {{ color: "#2364aa", width: 1.4 }},
                hovertemplate: "%{{x}}<br>50-day: $%{{y:,.2f}}<extra></extra>"
            }},
            {{
                x: dates,
                y: rows.map(row => row.ma_200d),
                type: "scatter",
                mode: "lines",
                name: "200-day average",
                line: {{ color: "#d97706", width: 1.8 }},
                hovertemplate: "%{{x}}<br>200-day: $%{{y:,.2f}}<extra></extra>"
            }}
        ] : [];

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

        const config = {{
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d", "autoScale2d"]
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

        Plotly.newPlot("priceChart", [
            {{
                x: dates,
                y: close,
                type: "scatter",
                mode: "lines",
                name: "Close",
                line: {{ color: "#18212b", width: 2 }},
                fill: "tozeroy",
                fillcolor: "rgba(24, 33, 43, 0.08)",
                hovertemplate: "%{{x}}<br>Close: $%{{y:,.2f}}<extra></extra>"
            }},
            ...trendTraces,
            {{
                x: buys.map(row => row.date),
                y: buys.map(row => row.close),
                type: "scatter",
                mode: "markers",
                name: "Buy",
                marker: {{ color: "#0c7c59", size: 11, symbol: "triangle-up" }},
                hovertemplate: "%{{x}}<br>Buy: $%{{y:,.2f}}<extra></extra>"
            }},
            {{
                x: sells.map(row => row.date),
                y: sells.map(row => row.close),
                type: "scatter",
                mode: "markers",
                name: "Sell",
                marker: {{ color: "#c0392b", size: 11, symbol: "triangle-down" }},
                hovertemplate: "%{{x}}<br>Sell: $%{{y:,.2f}}<extra></extra>"
            }}
        ], layout("Price Action", "Price ($)", {{ xaxis: {{ rangeslider: {{ visible: true, thickness: 0.08 }} }} }}), config);

        Plotly.newPlot("rsiChart", [
            {{
                x: dates,
                y: rsi,
                type: "scatter",
                mode: "lines",
                name: "RSI(14)",
                line: {{ color: "#2364aa", width: 2 }},
                hovertemplate: "%{{x}}<br>RSI: %{{y:.1f}}<extra></extra>"
            }}
        ], layout("Relative Strength Index", "RSI", {{
            yaxis: {{ range: [0, 100] }},
            shapes: [
                {{ type: "rect", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 100, fillcolor: "rgba(192, 57, 43, 0.10)", line: {{ width: 0 }} }},
                {{ type: "rect", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 30, fillcolor: "rgba(12, 124, 89, 0.10)", line: {{ width: 0 }} }},
                {{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 70, line: {{ color: "#c0392b", width: 1, dash: "dash" }} }},
                {{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 30, y1: 30, line: {{ color: "#0c7c59", width: 1, dash: "dash" }} }}
            ]
        }}), config);

        Plotly.newPlot("volatilityChart", [
            {{
                x: dates,
                y: volatilityPct,
                type: "scatter",
                mode: "lines",
                name: "Volatility",
                line: {{ color: "#b26a00", width: 2 }},
                fill: "tozeroy",
                fillcolor: "rgba(230, 126, 34, 0.22)",
                hovertemplate: "%{{x}}<br>Volatility: %{{y:.2f}}%<extra></extra>"
            }}
        ], layout("Volatility", "Volatility (%)"), config);

        Plotly.newPlot("momentumChart", [
            {{
                x: dates,
                y: upperPct,
                type: "scatter",
                mode: "lines",
                name: "Range upper",
                line: {{ color: "rgba(12, 124, 89, 0.30)", width: 1 }},
                hovertemplate: "%{{x}}<br>Upper: %{{y:.2f}}%<extra></extra>"
            }},
            {{
                x: dates,
                y: lowerPct,
                type: "scatter",
                mode: "lines",
                name: "Trading range",
                line: {{ color: "rgba(12, 124, 89, 0.30)", width: 1 }},
                fill: "tonexty",
                fillcolor: "rgba(12, 124, 89, 0.16)",
                hovertemplate: "%{{x}}<br>Lower: %{{y:.2f}}%<extra></extra>"
            }},
            {{
                x: dates,
                y: momentumPct,
                type: "scatter",
                mode: "lines",
                name: "3M momentum",
                line: {{ color: "#2364aa", width: 2 }},
                hovertemplate: "%{{x}}<br>Momentum: %{{y:.2f}}%<extra></extra>"
            }},
            {{
                x: buys.map(row => row.date),
                y: buys.map(row => row.current_return * 100),
                type: "scatter",
                mode: "markers",
                name: "Buy",
                marker: {{ color: "#0c7c59", size: 12, symbol: "triangle-up" }},
                hovertemplate: "%{{x}}<br>Buy signal: %{{y:.2f}}%<extra></extra>"
            }},
            {{
                x: sells.map(row => row.date),
                y: sells.map(row => row.current_return * 100),
                type: "scatter",
                mode: "markers",
                name: "Sell",
                marker: {{ color: "#c0392b", size: 12, symbol: "triangle-down" }},
                hovertemplate: "%{{x}}<br>Sell signal: %{{y:.2f}}%<extra></extra>"
            }}
        ], layout("3-Month Momentum vs Trading Range", "Return (%)", {{
            shapes: [{{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ color: "#657381", width: 1, dash: "dash" }} }}]
        }}), config);

        Plotly.newPlot("portfolioChart", [
            {{
                x: portfolio.map(row => row.date),
                y: portfolio.map(row => row.portfolio_value),
                type: "scatter",
                mode: "lines",
                name: "Portfolio value",
                line: {{ color: "#0c7c59", width: 2 }},
                fill: "tozeroy",
                fillcolor: "rgba(12, 124, 89, 0.12)",
                customdata: portfolio.map(row => [row.cash, row.shares]),
                hovertemplate: "%{{x}}<br>Value: $%{{y:,.2f}}<br>Cash: $%{{customdata[0]:,.2f}}<br>Shares: %{{customdata[1]:,.0f}}<extra></extra>"
            }},
            {{
                x: dates,
                y: buyHold,
                type: "scatter",
                mode: "lines",
                name: "Buy-and-hold",
                line: {{ color: "#d97706", width: 2, dash: "dash" }},
                hovertemplate: "%{{x}}<br>Buy-and-hold: $%{{y:,.2f}}<extra></extra>"
            }},
            {{
                x: dates,
                y: asxHold,
                type: "scatter",
                mode: "lines",
                name: "ASX 200",
                line: {{ color: "#2364aa", width: 2, dash: "dot" }},
                hovertemplate: "%{{x}}<br>ASX 200: $%{{y:,.2f}}<extra></extra>"
            }}
        ], layout("Backtest Portfolio Value", "Portfolio Value ($)", {{
            shapes: [{{ type: "line", xref: "paper", x0: 0, x1: 1, y0: initialCapital, y1: initialCapital, line: {{ color: "#c0392b", width: 1, dash: "dash" }} }}]
        }}), config);

        const tbody = document.querySelector("#tradesTable tbody");
        if (trades.length === 0) {{
            tbody.innerHTML = '<tr><td colspan="8">No trades were generated for this period.</td></tr>';
        }} else {{
            tbody.innerHTML = trades.slice().reverse().map(trade => `
                <tr>
                    <td>${{trade.date}}</td>
                    <td>${{trade.action}}</td>
                    <td>${{trade.shares.toLocaleString()}}</td>
                    <td>$${{trade.price.toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</td>
                    <td>$${{trade.value.toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</td>
                    <td>$${{trade.commission.toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }})}}</td>
                    <td>${{trade.position_pct === null ? "" : (trade.position_pct * 100).toFixed(1) + "%"}}</td>
                    <td>${{trade.sizing_reason || ""}}</td>
                </tr>
            `).join("");
        }}
    </script>
</body>
</html>
"""

    output_file.write_text(html, encoding="utf-8")
    print(f"Interactive report saved to {output_file}")
    return output_file


def plot_analysis(ticker: str, signals_df: pd.DataFrame, backtest_results: dict, stats: dict | None = None) -> Path:
    """Backward-compatible wrapper that now writes an interactive HTML report."""

    if stats is None:
        latest = signals_df.iloc[-1]
        stats = {
            "ticker": ticker,
            "data_start": signals_df["date"].min(),
            "data_end": signals_df["date"].max(),
            "trading_days": len(signals_df),
            "latest_date": latest["date"],
            "latest_price": float(latest["close"]),
            "rsi": float(latest["rsi"]),
            "volatility": float(latest["volatility"]),
            "momentum_3m": float(latest["momentum_3m"]),
            "return_1y": float(latest["return_1y"]),
            "range_lower": float(latest["range_lower"]),
            "range_upper": float(latest["range_upper"]),
            "current_return": float(latest["current_return"]),
            "total_buy_signals": int((signals_df["buy_signal"] > 0).sum()),
            "total_sell_signals": int((signals_df["sell_signal"] > 0).sum()),
        }
    return create_interactive_report(ticker, signals_df, backtest_results, stats)


def main():
    """Main execution."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_single_stock.py <TICKER> [period] [strategy]")
        print("Example: python run_single_stock.py NVDA")
        print("Example: python run_single_stock.py CBA 3y")
        print("Example: python run_single_stock.py TLS.AX 10y long-term")
        print("\nValid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    period = sys.argv[2] if len(sys.argv) > 2 else "5y"
    strategy = normalize_strategy(sys.argv[3]) if len(sys.argv) > 3 else "medium-term"
    
    try:
        # Analyze stock
        analysis = analyze_single_stock(ticker, period, strategy)
        print_stats(analysis["stats"])
        
        # Run backtest
        print("Running backtest...")
        config = STRATEGIES[strategy]
        backtest_results = simple_backtest(
            analysis["data"],
            initial_capital=10_000,
            minimum_holding_days=config["minimum_holding_days"],
            buy_cooldown_days=config["buy_cooldown_days"],
            profit_first_exits=config["profit_first_exits"],
        )
        print_backtest_results(backtest_results)
        
        # Create interactive report
        print("Creating interactive report...")
        report_path = plot_analysis(ticker, analysis["data"], backtest_results, analysis["stats"])
        
        print(f"\nComplete! Open {report_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
