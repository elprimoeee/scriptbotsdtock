#!/usr/bin/env python3
"""Build a survivorship-safe S&P 500 dataset from Sharadar."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sharadar_client import SharadarClient, SharadarCredentials


PRICE_FIELDS = ("ticker", "date", "open", "high", "low", "close", "volume", "closeadj", "closeunadj")
DAILY_FIELDS = ("ticker", "date", "marketcap", "ev", "evebit", "evebitda", "pb", "pe", "ps")
ART_FIELDS = (
    "ticker", "date", "calendardate", "reportperiod", "dimension", "roe", "roic",
    "roa", "grossmargin", "netmargin", "ebitdamargin", "currentratio", "de",
    "assetturnover", "fcf", "revenueusd", "netinccmnusd", "epsusd", "divyield",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--end", default=date.today().isoformat(), help="Inclusive YYYY-MM-DD end date")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--output", type=Path, default=Path("data/processed/sp500_daily_dataset.csv"))
    parser.add_argument("--membership-output", type=Path, default=Path("data/raw/universe/sp500_membership_history.csv"))
    parser.add_argument("--events-output", type=Path, default=Path("data/raw/universe/sp500_events.csv"))
    parser.add_argument("--benchmark-output", type=Path,
                        default=Path("data/raw/benchmark/sp500_spy_sharadar.csv"))
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Refresh only the Sharadar SPY benchmark")
    return parser.parse_args()


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def build_spy_benchmark(client: SharadarClient, start: pd.Timestamp,
                        end: pd.Timestamp) -> pd.DataFrame:
    """Return a total-return S&P 500 proxy sourced from Sharadar SEP."""

    rows = client.table(
        "funds", fields=PRICE_FIELDS, tickers=["SPY"],
        start=start.date().isoformat(), end=end.date().isoformat(),
    )
    benchmark = _frame(rows)
    if benchmark.empty:
        raise ValueError("Sharadar returned no SPY benchmark prices")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce").dt.normalize()
    benchmark["close"] = pd.to_numeric(benchmark["closeadj"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date", "close"])[["date", "close"]]
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark["ticker"] = "SPY"
    benchmark["data_source"] = "Sharadar SEP adjusted close"
    return benchmark


def ticker_batches(tickers: list[str], max_count: int, max_characters: int = 200) -> list[list[str]]:
    """Respect Sharadar's count and comma-separated ticker length limits."""

    max_count = min(max_count, 30)  # Sharadar rejects ticker filters with more than 30 values.
    batches: list[list[str]] = []
    current: list[str] = []
    current_length = 0
    for ticker in tickers:
        added_length = len(ticker) + (1 if current else 0)
        if current and (len(current) >= max_count or current_length + added_length > max_characters):
            batches.append(current)
            current = []
            current_length = 0
            added_length = len(ticker)
        if len(ticker) > max_characters:
            raise ValueError(f"Ticker exceeds Sharadar's {max_characters}-character filter limit")
        current.append(ticker)
        current_length += added_length
    if current:
        batches.append(current)
    return batches


def _fetch_batches(client: SharadarClient, table: str, tickers: list[str], *,
                   fields: tuple[str, ...], start: str, end: str, batch_size: int,
                   **filters: object) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    completed = 0
    for batch in ticker_batches(tickers, batch_size):
        print(f"{table}: {completed + 1}-{completed + len(batch)} of {len(tickers)}")
        rows = client.table(table, fields=fields, tickers=batch, start=start, end=end, **filters)
        if rows:
            frames.append(_frame(rows))
        completed += len(batch)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)


def build_membership(price_dates: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    """Expand snapshots and effective-date changes onto actual market sessions."""

    event_frame = events.copy()
    event_frame["date"] = pd.to_datetime(event_frame["date"], errors="coerce").dt.normalize()
    event_frame["ticker"] = event_frame["ticker"].astype(str).str.strip().str.upper()
    event_frame["action"] = event_frame["action"].astype(str).str.strip().str.lower()
    event_frame = event_frame.dropna(subset=["date"]).sort_values(["date", "action", "ticker"])
    snapshots = event_frame[event_frame["action"].isin({"historical", "current"})]
    first_session = pd.Timestamp(price_dates.min())
    snapshot_dates = snapshots.loc[snapshots["date"] <= first_session, "date"]
    if snapshot_dates.empty:
        raise ValueError("Sharadar returned no S&P 500 snapshot on or before the first price session")
    seed_date = snapshot_dates.max()
    members = set(snapshots.loc[snapshots["date"] == seed_date, "ticker"])
    relevant = event_frame[event_frame["date"] > seed_date]
    by_date = {day: rows for day, rows in relevant.groupby("date")}
    event_dates = sorted(by_date)
    event_index = 0
    rows: list[dict[str, object]] = []
    for session in price_dates.sort_values().unique():
        session = pd.Timestamp(session)
        while event_index < len(event_dates) and event_dates[event_index] <= session:
            day = by_date[event_dates[event_index]]
            historical = day[day["action"].isin({"historical", "current"})]
            if not historical.empty:
                members = set(historical["ticker"])
            for ticker in day.loc[day["action"] == "removed", "ticker"]:
                members.discard(ticker)
            for ticker in day.loc[day["action"] == "added", "ticker"]:
                members.add(ticker)
            event_index += 1
        rows.extend({"date": session, "ticker": ticker, "is_sp500": True} for ticker in members)
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def merge_point_in_time_fundamentals(prices: pd.DataFrame, daily: pd.DataFrame,
                                     art: pd.DataFrame) -> pd.DataFrame:
    output = prices.copy()
    for frame in (output, daily, art):
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        if "ticker" in frame:
            frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    daily = daily.rename(columns={"marketcap": "market_cap_millions", "evebit": "ev_ebit", "evebitda": "ev_ebitda"})
    for column in daily.columns.difference(["ticker", "date"]):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if "market_cap_millions" in daily:
        daily["market_cap"] = daily["market_cap_millions"] * 1_000_000.0
    output = output.merge(daily, on=["ticker", "date"], how="left")

    art = art.rename(columns={
        "date": "fundamental_date", "calendardate": "fundamental_calendar_date",
        "reportperiod": "fundamental_report_period", "grossmargin": "gross_margin",
        "netmargin": "net_margin", "ebitdamargin": "ebitda_margin",
        "currentratio": "current_ratio", "de": "debt_equity",
        "assetturnover": "asset_turnover", "revenueusd": "revenue_usd",
        "netinccmnusd": "net_income_common_usd", "epsusd": "eps_usd",
        "divyield": "dividend_yield",
    })
    for column in ("fundamental_date", "fundamental_calendar_date", "fundamental_report_period"):
        art[column] = pd.to_datetime(art[column], errors="coerce").dt.normalize()
    text_columns = {"ticker", "fundamental_date", "fundamental_calendar_date", "fundamental_report_period", "dimension"}
    for column in art.columns.difference(list(text_columns)):
        art[column] = pd.to_numeric(art[column], errors="coerce")
    art = art.dropna(subset=["ticker", "fundamental_date"]).sort_values(
        ["fundamental_date", "ticker"]
    ).drop_duplicates(["ticker", "fundamental_date"], keep="last")
    output = pd.merge_asof(
        output.sort_values(["date", "ticker"]),
        art.sort_values(["fundamental_date", "ticker"]),
        left_on="date", right_on="fundamental_date", by="ticker",
        direction="backward", allow_exact_matches=True,
    )
    return output.sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.years < 1 or args.batch_size < 1:
        raise SystemExit("--years and --batch-size must be positive")
    end = pd.Timestamp(args.end).normalize()
    start = end - pd.DateOffset(years=args.years)
    snapshot_start = start - pd.DateOffset(months=6)
    client = SharadarClient(SharadarCredentials.from_env_file(args.env_file))

    benchmark = build_spy_benchmark(client, start, end)
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(args.benchmark_output, index=False)
    print(f"Saved {len(benchmark):,} Sharadar SPY benchmark rows to {args.benchmark_output}")
    if args.benchmark_only:
        return

    events = _frame(client.table("sp500", start=snapshot_start.date().isoformat(), end=end.date().isoformat()))
    if events.empty:
        raise SystemExit("Sharadar returned no S&P 500 membership history")
    tickers = sorted(set(events["ticker"].dropna().astype(str).str.upper()))
    print(f"Membership history contains {len(tickers)} current and former tickers")
    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.events_output, index=False)

    prices = _fetch_batches(client, "stocks", tickers, fields=PRICE_FIELDS,
                            start=start.date().isoformat(), end=end.date().isoformat(),
                            batch_size=args.batch_size)
    if prices.empty:
        raise SystemExit("Sharadar returned no stock prices")
    for column in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices = prices.rename(columns={"closeadj": "adj_close", "closeunadj": "unadjusted_close"})
    prices["dollar_volume"] = prices["close"] * prices["volume"]
    prices = prices.dropna(subset=["ticker", "date", "close"])

    membership = build_membership(pd.DatetimeIndex(prices["date"].drop_duplicates()), events)
    args.membership_output.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(args.membership_output, index=False)
    prices = prices.merge(membership, on=["date", "ticker"], how="left")
    prices["is_sp500"] = prices["is_sp500"].fillna(False).astype(bool)

    daily = _fetch_batches(client, "daily", tickers, fields=DAILY_FIELDS,
                           start=start.date().isoformat(), end=end.date().isoformat(),
                           batch_size=args.batch_size)
    art = _fetch_batches(client, "fundamentals", tickers, fields=ART_FIELDS,
                         start=(start - pd.DateOffset(years=2)).date().isoformat(),
                         end=end.date().isoformat(), batch_size=args.batch_size,
                         dimension="ART")
    dataset = merge_point_in_time_fundamentals(prices, daily, art)
    dataset["market_cap_rank"] = dataset.groupby("date")["market_cap"].rank(ascending=False, method="dense")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {len(dataset):,} rows for {dataset['ticker'].nunique():,} tickers to {args.output}")
    print(f"Point-in-time member rows: {int(dataset['is_sp500'].sum()):,}")


if __name__ == "__main__":
    main()
