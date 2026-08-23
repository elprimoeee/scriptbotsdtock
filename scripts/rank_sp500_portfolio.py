#!/usr/bin/env python3
"""Rank the latest point-in-time S&P 500 universe and annotate Trading 212 demo holdings."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sp500_selector import rank_sp500_stocks
from models.trading212_client import Trading212Credentials, Trading212PracticeClient


def normalized_symbol(value: object) -> str:
    """Normalize Sharadar and Trading 212 identifiers for a conservative join."""

    base = str(value).strip().upper().split("_", 1)[0]
    return re.sub(r"[^A-Z0-9]", "", base)


def annotate_trading212(ranking: pd.DataFrame, client: Trading212PracticeClient) -> pd.DataFrame:
    instruments = client.instruments()
    positions = client.positions()
    instrument_rows = []
    for item in instruments:
        if item.get("currencyCode") != "USD" or item.get("type") != "STOCK":
            continue
        instrument_rows.append({
            "symbol_key": normalized_symbol(item.get("ticker", "")),
            "trading212_ticker": item.get("ticker"),
            "trading212_name": item.get("name") or item.get("shortName"),
        })
    instrument_frame = pd.DataFrame(instrument_rows).drop_duplicates("symbol_key", keep="first")
    held: dict[str, float] = {}
    for position in positions:
        instrument = position.get("instrument") or {}
        key = normalized_symbol(instrument.get("ticker", ""))
        held[key] = held.get(key, 0.0) + float(position.get("quantity") or 0.0)
    output = ranking.copy()
    output["symbol_key"] = output["ticker"].map(normalized_symbol)
    output = output.merge(instrument_frame, on="symbol_key", how="left")
    output["trading212_demo_quantity"] = output["symbol_key"].map(held).fillna(0.0)
    output["held_in_trading212_demo"] = output["trading212_demo_quantity"] > 0
    output["tradable_on_trading212"] = output["trading212_ticker"].notna()
    return output.drop(columns="symbol_key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/sp500_daily_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/sp500_latest_ranking.csv"))
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--trading212-demo", action="store_true", help="Read demo instruments and positions; never places orders")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be positive")
    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}. Run scripts/build_sp500_dataset.py first.")
    dataset = pd.read_csv(args.input, low_memory=False)
    ranking = rank_sp500_stocks(dataset)
    if ranking.empty:
        raise SystemExit("No eligible S&P 500 stocks were found")
    latest_date = ranking["date"].max()
    latest = ranking.loc[ranking["date"] == latest_date].sort_values("selection_rank")
    if args.trading212_demo:
        client = Trading212PracticeClient(Trading212Credentials.from_env_file(args.env_file))
        latest = annotate_trading212(latest, client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(args.output, index=False)
    print(f"Saved {len(latest):,} ranked stocks for {pd.Timestamp(latest_date).date()} to {args.output}")
    columns = ["selection_rank", "ticker", "selection_score", "quality_rank", "value_rank"]
    if args.trading212_demo:
        columns += ["tradable_on_trading212", "held_in_trading212_demo"]
    print(latest.head(args.top)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
