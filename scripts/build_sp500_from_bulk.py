#!/usr/bin/env python3
"""Build the full-history S&P 500 dataset from local Sharadar bulk archives."""

from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_sp500_dataset import (
    ART_FIELDS,
    DAILY_FIELDS,
    FULL_HISTORY_START,
    PRICE_FIELDS,
    build_membership,
    merge_point_in_time_fundamentals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-dir", type=Path, default=Path("data/raw/sharadar_bulk"))
    parser.add_argument("--start", default=FULL_HISTORY_START.date().isoformat())
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--filtered-dir", type=Path,
        default=Path("data/raw/sharadar_bulk/sp500_filtered"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/sp500_daily_dataset_full.csv"),
    )
    parser.add_argument(
        "--membership-output", type=Path,
        default=Path("data/raw/universe/sp500_membership_history_full.csv"),
    )
    parser.add_argument(
        "--events-output", type=Path,
        default=Path("data/raw/universe/sp500_events_full.csv"),
    )
    return parser.parse_args()


def _filter_archive(
    archive: Path,
    output: Path,
    *,
    fields: tuple[str, ...],
    tickers: set[str],
    start: str,
    end: str,
    chunk_size: int,
    dimension: str | None = None,
) -> tuple[int, int]:
    """Stream a market-wide bulk archive into a much smaller project slice."""

    partial = output.with_suffix(output.suffix + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    output_rows = 0
    wrote_header = False
    with partial.open("w", newline="", encoding="utf-8") as destination:
        for chunk in pd.read_csv(
            archive, usecols=list(fields), chunksize=chunk_size, low_memory=False,
        ):
            input_rows += len(chunk)
            ticker_values = chunk["ticker"].astype(str).str.strip().str.upper()
            mask = ticker_values.isin(tickers) & chunk["date"].between(start, end)
            if dimension is not None:
                mask &= chunk["dimension"].eq(dimension)
            selected = chunk.loc[mask].copy()
            if not selected.empty:
                selected["ticker"] = ticker_values.loc[mask]
                selected.to_csv(destination, index=False, header=not wrote_header)
                wrote_header = True
                output_rows += len(selected)
            if input_rows % (chunk_size * 10) == 0:
                print(
                    f"{archive.stem}: scanned {input_rows:,}, kept {output_rows:,}",
                    flush=True,
                )
    if not wrote_header:
        pd.DataFrame(columns=fields).to_csv(partial, index=False)
    os.replace(partial, output)
    print(
        f"{archive.stem}: saved {output_rows:,} of {input_rows:,} rows to {output}",
        flush=True,
    )
    return input_rows, output_rows


def _require_archives(bulk_dir: Path) -> dict[str, Path]:
    archives = {
        table: bulk_dir / f"{table}.csv.zip"
        for table in ("stocks", "daily", "fundamentals", "sp500")
    }
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing Sharadar bulk archives: " + ", ".join(missing)
            + ". Run scripts/download_sharadar_bulk.py first."
        )
    return archives


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if start > end:
        raise SystemExit("--start must not be after --end")
    archives = _require_archives(args.bulk_dir)

    events = pd.read_csv(archives["sp500"])
    event_dates = pd.to_datetime(events["date"], errors="coerce")
    snapshot_start = start - pd.DateOffset(months=6)
    events = events.loc[event_dates.between(snapshot_start, end)].copy()
    events["ticker"] = events["ticker"].astype(str).str.strip().str.upper()
    tickers = set(events["ticker"].dropna())
    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.events_output, index=False)
    print(
        f"Membership history contains {len(events):,} rows and {len(tickers):,} tickers",
        flush=True,
    )

    filtered_prices = args.filtered_dir / "stocks_sp500_full.csv"
    filtered_daily = args.filtered_dir / "daily_sp500_full.csv"
    filtered_art = args.filtered_dir / "fundamentals_art_sp500_full.csv"
    _filter_archive(
        archives["stocks"], filtered_prices, fields=PRICE_FIELDS,
        tickers=tickers, start=start.date().isoformat(), end=end.date().isoformat(),
        chunk_size=args.chunk_size,
    )
    _filter_archive(
        archives["daily"], filtered_daily, fields=DAILY_FIELDS,
        tickers=tickers, start=start.date().isoformat(), end=end.date().isoformat(),
        chunk_size=args.chunk_size,
    )
    art_start = start - pd.DateOffset(years=2)
    _filter_archive(
        archives["fundamentals"], filtered_art, fields=ART_FIELDS,
        tickers=tickers, start=art_start.date().isoformat(), end=end.date().isoformat(),
        chunk_size=args.chunk_size, dimension="ART",
    )

    print("Loading filtered stock prices", flush=True)
    prices = pd.read_csv(filtered_prices, low_memory=False)
    for column in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices = prices.rename(columns={"closeadj": "adj_close", "closeunadj": "unadjusted_close"})
    prices["dollar_volume"] = prices["close"] * prices["volume"]
    prices = prices.dropna(subset=["ticker", "date", "close"])

    print("Expanding point-in-time membership", flush=True)
    membership = build_membership(pd.DatetimeIndex(prices["date"].drop_duplicates()), events)
    args.membership_output.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(args.membership_output, index=False)
    prices = prices.merge(membership, on=["date", "ticker"], how="left")
    prices["is_sp500"] = prices["is_sp500"].fillna(False).astype(bool)

    print("Loading filtered DAILY valuations", flush=True)
    daily = pd.read_csv(filtered_daily, low_memory=False)
    print("Loading filtered ART fundamentals", flush=True)
    art = pd.read_csv(filtered_art, low_memory=False)
    print("Merging prices, valuations, and filing-date fundamentals", flush=True)
    dataset = merge_point_in_time_fundamentals(prices, daily, art)
    dataset["market_cap_rank"] = dataset.groupby("date")["market_cap"].rank(
        ascending=False, method="dense"
    )
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(
        f"Saved {len(dataset):,} rows for {dataset['ticker'].nunique():,} tickers to {args.output}",
        flush=True,
    )
    print(f"Point-in-time member rows: {int(dataset['is_sp500'].sum()):,}", flush=True)


if __name__ == "__main__":
    main()
