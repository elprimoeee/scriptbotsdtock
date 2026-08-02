#!/usr/bin/env python3
"""Daily update script for RSI bot - updates calculations and generates fresh signals."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rsi_momentum_bot import (
    RSIBuySignal,
    filter_low_volatility_stocks,
    generate_buy_sell_signals,
    plot_strategy,
    TOP_STOCKS,
    MIN_DOLLAR_VOLUME,
)
from project_config import PathConfig


def get_latest_data() -> pd.DataFrame:
    """Load the latest dataset."""
    paths = PathConfig()
    
    print(f"Loading data from {paths.input_csv}...")
    usecols = ["date", "ticker", "close", "dollar_volume"]
    df = pd.read_csv(
        paths.input_csv,
        usecols=usecols,
        parse_dates=["date"],
        low_memory=False
    )
    
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce").astype("float32")
    df = df.dropna(subset=["close", "dollar_volume"])
    df = df[df["close"] > 0.5]
    
    print(f"Loaded {len(df)} records for {df['ticker'].nunique()} tickers")
    return df


def generate_daily_report(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Generate daily trading signals and report."""
    
    if output_dir is None:
        output_dir = Path("results/rsi_bot/daily")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating RSI signals and features...")
    model = RSIBuySignal()
    signals_df = model.generate_signals(df)
    
    # Filter for low volatility
    print(f"Filtering for top {TOP_STOCKS} low volatility stocks...")
    filtered_df = filter_low_volatility_stocks(signals_df, top_n=TOP_STOCKS)
    print(f"Selected {filtered_df['ticker'].nunique()} stocks")
    
    # Generate buy/sell signals
    print("Generating buy/sell signals...")
    signals_df = generate_buy_sell_signals(filtered_df)
    
    # Get today's signals
    latest_date = signals_df["date"].max()
    today_signals = signals_df[signals_df["date"] == latest_date].copy()
    
    # Identify buy and sell recommendations
    buy_recommendations = today_signals[today_signals["buy_signal"] > 0].sort_values("rsi")
    sell_recommendations = today_signals[today_signals["sell_signal"] > 0].sort_values("rsi", ascending=False)
    
    # Generate report
    report = {
        "date": latest_date,
        "timestamp": datetime.now(),
        "buy_signals": len(buy_recommendations),
        "sell_signals": len(sell_recommendations),
        "buy_stocks": buy_recommendations.to_dict("records") if len(buy_recommendations) > 0 else [],
        "sell_stocks": sell_recommendations.to_dict("records") if len(sell_recommendations) > 0 else [],
        "total_tracked_stocks": len(today_signals),
    }
    
    return report


def print_daily_report(report: dict) -> None:
    """Pretty print the daily report."""
    print("\n" + "="*80)
    print(f"RSI BOT - DAILY REPORT ({report['date'].date()})")
    print("="*80)
    
    print(f"\nBUY SIGNALS: {report['buy_signals']} stocks")
    if report['buy_signals'] > 0:
        print("-" * 80)
        print(f"{'Ticker':<8} {'RSI':<8} {'1W Rel':<10} {'3M Mom':<10} {'1Y Return':<12} {'Price':<10}")
        print("-" * 80)
        for stock in report['buy_stocks'][:10]:  # Show top 10
            print(f"{stock.get('ticker', 'N/A'):<8} {stock.get('rsi', 0):<8.2f} "
                  f"{stock.get('weekly_relative_momentum', 0)*100:<10.2f}% "
                  f"{stock.get('momentum_3m', 0)*100:<10.2f}% {stock.get('return_1y', 0)*100:<12.2f}% "
                  f"${stock.get('close', 0):<10.2f}")
        if report['buy_signals'] > 10:
            print(f"... and {report['buy_signals'] - 10} more")
    
    print(f"\nSELL SIGNALS: {report['sell_signals']} stocks")
    if report['sell_signals'] > 0:
        print("-" * 80)
        print(f"{'Ticker':<8} {'RSI':<8} {'1W Rel':<10} {'3M Mom':<10} {'1Y Return':<12} {'Price':<10}")
        print("-" * 80)
        for stock in report['sell_stocks'][:10]:  # Show top 10
            print(f"{stock.get('ticker', 'N/A'):<8} {stock.get('rsi', 0):<8.2f} "
                  f"{stock.get('weekly_relative_momentum', 0)*100:<10.2f}% "
                  f"{stock.get('momentum_3m', 0)*100:<10.2f}% {stock.get('return_1y', 0)*100:<12.2f}% "
                  f"${stock.get('close', 0):<10.2f}")
        if report['sell_signals'] > 10:
            print(f"... and {report['sell_signals'] - 10} more")
    
    print(f"\nTotal Tracked Stocks: {report['total_tracked_stocks']}")
    print("="*80 + "\n")


def save_daily_report(report: dict, output_dir: Path) -> Path:
    """Save daily report to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine buy and sell signals
    all_signals = []
    
    for stock in report['buy_stocks']:
        stock['signal_type'] = 'BUY'
        all_signals.append(stock)
    
    for stock in report['sell_stocks']:
        stock['signal_type'] = 'SELL'
        all_signals.append(stock)
    
    if len(all_signals) > 0:
        signals_df = pd.DataFrame(all_signals)
        date_str = report['date'].strftime("%Y-%m-%d")
        output_file = output_dir / f"signals_{date_str}.csv"
        signals_df.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")
        return output_file
    
    return None


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("RSI BOT - DAILY UPDATE")
    print("="*80)
    
    # Load latest data
    df = get_latest_data()
    
    # Generate report
    report = generate_daily_report(df)
    
    # Print report
    print_daily_report(report)
    
    # Save report
    output_dir = Path("results/rsi_bot/daily")
    save_daily_report(report, output_dir)
    
    print("Daily update complete!")


if __name__ == "__main__":
    main()
