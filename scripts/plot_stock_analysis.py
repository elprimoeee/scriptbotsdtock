#!/usr/bin/env python3
"""Detailed stock analysis and visualization tool for RSI bot."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rsi_momentum_bot import RSIBuySignal
from project_config import PathConfig


def load_and_prepare_data(ticker: str) -> pd.DataFrame:
    """Load and prepare data for a specific ticker."""
    paths = PathConfig()
    
    usecols = ["date", "ticker", "close", "dollar_volume"]
    df = pd.read_csv(
        paths.input_csv,
        usecols=usecols,
        parse_dates=["date"],
        low_memory=False
    )
    
    df = df[df["ticker"].str.upper() == ticker.upper()].copy()
    if len(df) == 0:
        raise ValueError(f"Ticker {ticker} not found in dataset")
    
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce").astype("float32")
    df = df.sort_values("date").reset_index(drop=True)
    
    return df


def analyze_stock(ticker: str, output_dir: Path | None = None) -> dict:
    """Generate detailed analysis for a single stock."""
    
    if output_dir is None:
        output_dir = Path("results/rsi_bot/stock_analysis")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {ticker}...")
    
    # Load data
    df = load_and_prepare_data(ticker)
    
    # Generate signals
    model = RSIBuySignal()
    signals_df = model.generate_signals(df)
    signals_df = signals_df[signals_df["ticker"].str.upper() == ticker.upper()]
    
    if len(signals_df) == 0:
        print(f"No signals generated for {ticker}")
        return None
    
    # Calculate stats
    latest = signals_df.iloc[-1]
    
    stats = {
        "ticker": ticker,
        "latest_date": latest["date"],
        "latest_price": float(latest["close"]),
        "rsi": float(latest["rsi"]),
        "volatility": float(latest["volatility"]),
        "momentum_3m": float(latest["momentum_3m"]),
        "return_1y": float(latest["return_1y"]),
        "range_lower": float(latest["range_lower"]),
        "range_upper": float(latest["range_upper"]),
        "position_in_range": float(latest["position_in_range"]),
        "in_range": float(latest["current_return"]) > float(latest["range_lower"]) and float(latest["current_return"]) < float(latest["range_upper"]),
        "above_range": float(latest["current_return"]) > float(latest["range_upper"]),
        "below_range": float(latest["current_return"]) < float(latest["range_lower"]),
    }
    
    return {
        "stats": stats,
        "data": signals_df,
    }


def plot_stock_analysis(ticker: str, analysis_data: dict, output_dir: Path) -> Path:
    """Create comprehensive visualization for a stock."""
    
    if analysis_data is None:
        return None
    
    data = analysis_data["data"]
    stats = analysis_data["stats"]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Price with ranges
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data["date"], data["close"], linewidth=2, label="Close Price", color="black")
    
    # Add price range bands (10% and 20%)
    ma20 = data["close"].rolling(20).mean()
    ax1.fill_between(data["date"], ma20 * 0.95, ma20 * 1.05, alpha=0.2, color="yellow", label="±5% Band")
    
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} - Price Action")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI with overbought/oversold
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data["date"], data["rsi"], linewidth=2, color="blue", label="RSI(14)")
    ax2.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)")
    ax2.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax2.fill_between(data["date"], 70, 100, alpha=0.1, color="red")
    ax2.fill_between(data["date"], 0, 30, alpha=0.1, color="green")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.set_title("Relative Strength Index (14-period)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Volatility
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data["date"], data["volatility"] * 100, linewidth=2, color="orange")
    ax3.fill_between(data["date"], 0, data["volatility"] * 100, alpha=0.3, color="orange")
    ax3.set_ylabel("Volatility (%)")
    ax3.set_title("Volatility (20-period rolling std)")
    ax3.grid(True, alpha=0.3)
    
    # 4. Momentum vs Range
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(data["date"], data["current_return"] * 100, linewidth=2, label="3M Momentum", color="blue")
    
    # Plot range bands
    ax4.fill_between(
        data["date"],
        data["range_lower"] * 100,
        data["range_upper"] * 100,
        alpha=0.2, color="green", label="Trading Range"
    )
    
    ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax4.set_ylabel("Return (%)")
    ax4.set_xlabel("Date")
    ax4.set_title("3-Month Momentum vs Trading Range")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics box
    stats_text = (
        f"Latest Price: ${stats['latest_price']:.2f}\n"
        f"RSI(14): {stats['rsi']:.1f}\n"
        f"3M Momentum: {stats['momentum_3m']*100:+.2f}%\n"
        f"1Y Return: {stats['return_1y']*100:+.2f}%\n"
        f"Volatility: {stats['volatility']*100:.2f}%\n"
        f"Range: {stats['range_lower']*100:+.2f}% / {stats['range_upper']*100:+.2f}%\n"
        f"Status: {'BELOW' if stats['below_range'] else 'ABOVE' if stats['above_range'] else 'IN'} RANGE"
    )
    
    fig.text(0.98, 0.97, stats_text, transform=fig.transFigure, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    
    plt.suptitle(f"{ticker} - Detailed Technical Analysis", fontsize=14, fontweight='bold')
    
    output_file = output_dir / f"{ticker}_analysis.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {output_file}")
    return output_file


def print_stock_stats(stats: dict) -> None:
    """Print stock statistics."""
    print("\n" + "="*60)
    print(f"STOCK ANALYSIS: {stats['ticker']}")
    print("="*60)
    print(f"Date:              {stats['latest_date'].date()}")
    print(f"Price:             ${stats['latest_price']:.2f}")
    print(f"RSI(14):           {stats['rsi']:.1f} {'(Overbought)' if stats['rsi'] > 70 else '(Oversold)' if stats['rsi'] < 30 else '(Neutral)'}")
    print(f"3M Momentum:       {stats['momentum_3m']*100:+.2f}%")
    print(f"1Y Return:         {stats['return_1y']*100:+.2f}%")
    print(f"Volatility:        {stats['volatility']*100:.2f}%")
    print(f"Trading Range:     {stats['range_lower']*100:+.2f}% to {stats['range_upper']*100:+.2f}%")
    print(f"Position in Range: {stats['position_in_range']*100:+.1f}%")
    
    if stats['below_range']:
        print(f"STATUS:            🔴 BELOW RANGE - Consider buying on reversal")
    elif stats['above_range']:
        print(f"STATUS:            🟢 ABOVE RANGE - Consider selling on reversal")
    else:
        print(f"STATUS:            🟡 IN RANGE - Wait for breakout or breakdown")
    
    print("="*60 + "\n")


def main():
    """Main execution."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_stock_analysis.py <TICKER> [<TICKER2> ...]")
        print("Example: python plot_stock_analysis.py CBA BHP NAB")
        sys.exit(1)
    
    tickers = sys.argv[1:]
    output_dir = Path("results/rsi_bot/stock_analysis")
    
    for ticker in tickers:
        try:
            analysis_data = analyze_stock(ticker, output_dir)
            
            if analysis_data:
                print_stock_stats(analysis_data["stats"])
                plot_stock_analysis(ticker, analysis_data, output_dir)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")


if __name__ == "__main__":
    main()
