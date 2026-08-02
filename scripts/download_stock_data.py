#!/usr/bin/env python3
"""Download stock data from Yahoo Finance and save to CSV."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str,
    period: str = "5y",
    output_dir: Path | None = None,
) -> Path:
    """Download stock data and save to CSV."""
    
    if output_dir is None:
        output_dir = Path("data/downloaded_stocks")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {ticker} ({period})...")
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    print(f"Got {len(df)} rows of data")
    print(f"Columns: {df.columns.tolist()}")
    
    # Reset index to make date a column
    df = df.reset_index()
    print(f"After reset_index: {df.shape}")
    print(f"Columns now: {df.columns.tolist()}")
    
    # Rename columns to lowercase
    df.columns = df.columns.str.lower()
    print(f"After lowercase: {df.columns.tolist()}")
    
    # Select and rename columns
    output_df = pd.DataFrame()
    output_df["date"] = df["date"]
    output_df["open"] = df["open"]
    output_df["high"] = df["high"]
    output_df["low"] = df["low"]
    output_df["close"] = df["close"]
    output_df["volume"] = df["volume"]
    output_df["adj_close"] = df.get("adj close", df["close"])
    output_df["ticker"] = ticker.upper()
    
    # Add dollar volume
    output_df["dollar_volume"] = output_df["close"] * output_df["volume"]
    
    # Save to CSV
    output_file = output_dir / f"{ticker.upper()}.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Downloaded {len(output_df)} rows")
    print(f"📊 Date range: {output_df['date'].min()} to {output_df['date'].max()}")
    print(f"💾 Saved to: {output_file}")
    
    return output_file


def main():
    """Main execution."""
    
    if len(sys.argv) < 2:
        print("Usage: python download_stock_data.py <TICKER> [period]")
        print("\nExamples:")
        print("  python download_stock_data.py NVDA")
        print("  python download_stock_data.py NVDA 2y")
        print("  python download_stock_data.py CBA 10y")
        print("\nValid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    period = sys.argv[2] if len(sys.argv) > 2 else "5y"
    
    try:
        output_file = download_stock_data(ticker, period)
        print(f"\n📥 Data ready at: {output_file}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
