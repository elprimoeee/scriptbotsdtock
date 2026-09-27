#!/usr/bin/env python3
"""Prepare or submit the Sharadar selector basket to Trading 212 Practice.

This script is permanently restricted to the Trading 212 demo Equity API. It
uses fresh Sharadar prices and point-in-time S&P 500 data. Order submission is
one-shot; an ambiguous response must be checked in Trading 212 before retrying.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sp500_selector import (
    DEFAULT_MEAN_REVERSION_RSI_THRESHOLD,
    FUNDAMENTAL_COLUMNS,
    SP500SelectorConfig,
    rank_sp500_stocks,
)
from models.sharadar_client import SharadarClient, SharadarCredentials
from models.trading212_client import (
    Trading212Credentials,
    Trading212Error,
    Trading212PracticeClient,
)
from scripts.build_sp500_dataset import (
    ART_FIELDS,
    DAILY_FIELDS,
    PRICE_FIELDS,
    merge_point_in_time_fundamentals,
)


DATASET_PATH = PROJECT_ROOT / "data/processed/sp500_daily_dataset_full.csv"
DEFAULT_PLAN = PROJECT_ROOT / "results/trading212_demo_strategy_plan.json"
DEFAULT_CSV = PROJECT_ROOT / "results/trading212_demo_strategy_plan.csv"
ACCOUNT_BUDGET_AUD = 5_000.0
DEPLOYMENT_FRACTION = 0.80
# A$1.45 per US$1 leaves room above the latest reference USD/AUD rate and the
# broker's currency conversion fee. The remaining 20% of cash is unallocated.
USD_TO_AUD_BUDGET_RATE = 1.45
MAX_POSITIONS = 25
HOLDING_MONTHS = 6
VERIFIED_TICKER_CHANGE_ISINS = {
    # Trading 212 metadata retains the pre-change order symbol but advertises
    # the current symbol in shortName. ISINs guard each alias against mismatch.
    "ELV": "US0367521038",   # Elevance Health, listed as ANTM_US_EQ
    "CPAY": "US2199481068",  # Corpay, listed as FLT_US_EQ
    "XYZ": "US8522341036",   # Block, listed as SQ_US_EQ
    "WELL": "US95040Q1040",  # Welltower, listed as HCN_US_EQ
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--submit", action="store_true",
        help="Submit the previously prepared plan as market buys to the demo Equity API",
    )
    actions.add_argument(
        "--resume", action="store_true",
        help="Resume a partially submitted plan after verifying its accepted orders",
    )
    return parser.parse_args()


def symbol_key(value: object) -> str:
    base = str(value or "").strip().upper().split("_", 1)[0]
    return re.sub(r"[^A-Z0-9]", "", base)


def ticker_batches(tickers: Iterable[str], max_count: int = 30,
                   max_characters: int = 200) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    length = 0
    for ticker in sorted(set(tickers)):
        addition = len(ticker) + (1 if current else 0)
        if current and (len(current) >= max_count or length + addition > max_characters):
            batches.append(current)
            current, length = [], 0
            addition = len(ticker)
        if len(ticker) > max_characters:
            raise RuntimeError(f"Ticker exceeds Sharadar's filter limit: {ticker}")
        current.append(ticker)
        length += addition
    if current:
        batches.append(current)
    return batches


def query_batches(client: SharadarClient, table: str, tickers: list[str], *,
                  fields: tuple[str, ...], start: str, end: str,
                  **filters: object) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    batches = ticker_batches(tickers)
    for index, batch in enumerate(batches, start=1):
        print(f"Sharadar {table}: batch {index}/{len(batches)}", flush=True)
        records.extend(client.table(
            table,
            fields=fields,
            tickers=batch,
            start=start,
            end=end,
            **filters,
        ))
    return pd.DataFrame.from_records(records, columns=list(fields))


def read_recent_local(current_tickers: set[str], start_date: pd.Timestamp,
                      end_date: pd.Timestamp) -> tuple[pd.DataFrame, set[str], pd.Timestamp]:
    if not DATASET_PATH.is_file():
        raise SystemExit(f"Required Sharadar dataset is missing: {DATASET_PATH}")
    header = pd.read_csv(DATASET_PATH, nrows=0).columns
    wanted = {
        "date", "ticker", "open", "close", "volume", "dollar_volume",
        "adj_close", "is_sp500", "market_cap", "market_cap_rank",
        *FUNDAMENTAL_COLUMNS,
    }
    usecols = [column for column in header if column in wanted]
    required = {"date", "ticker", "close", "dollar_volume", "is_sp500"}
    if not required.issubset(usecols):
        raise SystemExit("The full Sharadar dataset is missing required selector fields")

    chunks: list[pd.DataFrame] = []
    ticker_history: set[str] = set()
    latest = pd.Timestamp.min
    for chunk in pd.read_csv(DATASET_PATH, usecols=usecols, chunksize=250_000,
                             low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        chunk["ticker"] = chunk["ticker"].astype(str).str.strip().str.upper()
        chunk = chunk.loc[
            chunk["ticker"].isin(current_tickers)
            & chunk["date"].between(start_date, end_date)
        ].copy()
        if not chunk.empty:
            ticker_history.update(chunk["ticker"].dropna().astype(str))
            latest = max(latest, chunk["date"].max())
            chunks.append(chunk)
    if not chunks:
        raise SystemExit("No recent local rows matched the current S&P 500 constituents")
    return pd.concat(chunks, ignore_index=True), ticker_history, latest


def fetch_current_members(client: SharadarClient) -> list[str]:
    rows = client.table(
        "sp500", fields=("date", "ticker", "action"), action="current",
        sort="ticker.asc",
    )
    tickers = sorted({
        str(row.get("ticker", "")).strip().upper()
        for row in rows if row.get("ticker")
    })
    if not 490 <= len(tickers) <= 510:
        raise SystemExit(f"Unexpected current Sharadar S&P 500 size: {len(tickers)}")
    return tickers


def refresh_strategy_frame() -> tuple[pd.DataFrame, pd.Timestamp, int]:
    sharadar = SharadarClient(SharadarCredentials.from_env_file(PROJECT_ROOT / ".env"))
    current_tickers = fetch_current_members(sharadar)
    today = pd.Timestamp.now(tz="Australia/Sydney").tz_localize(None).normalize()
    history_start = today - pd.DateOffset(days=430)
    dataset, local_tickers, local_latest = read_recent_local(
        set(current_tickers), history_start, today,
    )
    new_tickers = sorted(set(current_tickers) - local_tickers)
    refresh_start = (local_latest + pd.Timedelta(days=1)).date().isoformat()
    refresh_end = today.date().isoformat()
    print(
        f"Local prices through {local_latest.date()}; refreshing through {refresh_end}. "
        f"Current members: {len(current_tickers)}; new tickers needing history: {len(new_tickers)}",
        flush=True,
    )

    update_prices = query_batches(
        sharadar, "stocks", current_tickers, fields=PRICE_FIELDS,
        start=refresh_start, end=refresh_end,
    )
    update_daily = query_batches(
        sharadar, "daily", current_tickers, fields=DAILY_FIELDS,
        start=refresh_start, end=refresh_end,
    )
    if new_tickers:
        older_prices = query_batches(
            sharadar, "stocks", new_tickers, fields=PRICE_FIELDS,
            start=history_start.date().isoformat(),
            end=local_latest.date().isoformat(),
        )
        older_daily = query_batches(
            sharadar, "daily", new_tickers, fields=DAILY_FIELDS,
            start=history_start.date().isoformat(),
            end=local_latest.date().isoformat(),
        )
        update_prices = pd.concat([older_prices, update_prices], ignore_index=True)
        update_daily = pd.concat([older_daily, update_daily], ignore_index=True)
    if update_prices.empty:
        raise SystemExit("Sharadar returned no newer stock prices")

    art = query_batches(
        sharadar, "fundamentals", current_tickers, fields=ART_FIELDS,
        start="1998-01-01", end=refresh_end, dimension="ART",
    )
    for column in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
        if column in update_prices:
            update_prices[column] = pd.to_numeric(update_prices[column], errors="coerce")
    update_prices["date"] = pd.to_datetime(update_prices["date"], errors="coerce").dt.normalize()
    update_prices["ticker"] = update_prices["ticker"].astype(str).str.strip().str.upper()
    update_prices = update_prices.rename(
        columns={"closeadj": "adj_close", "closeunadj": "unadjusted_close"}
    )
    update_prices["dollar_volume"] = update_prices["close"] * update_prices["volume"]
    update_prices = update_prices.dropna(subset=["ticker", "date", "close"])

    fresh = merge_point_in_time_fundamentals(update_prices, update_daily, art)
    fresh["is_sp500"] = False
    last_date = fresh["date"].max()
    fresh.loc[
        (fresh["date"] == last_date) & fresh["ticker"].isin(current_tickers),
        "is_sp500",
    ] = True
    dataset = pd.concat([dataset, fresh], ignore_index=True, sort=False)
    dataset = dataset.sort_values(["ticker", "date"]).drop_duplicates(
        ["ticker", "date"], keep="last"
    )
    dataset["market_cap_rank"] = dataset.groupby("date")["market_cap"].rank(
        ascending=False, method="dense"
    )

    expected_last = today - pd.offsets.BDay(1)
    if (expected_last - last_date).days > 3:
        raise SystemExit(
            f"Fresh Sharadar prices are stale: latest session {last_date.date()}, "
            f"expected near {expected_last.date()}"
        )
    latest_members = set(
        dataset.loc[(dataset["date"] == last_date) & dataset["is_sp500"], "ticker"]
    )
    if len(latest_members) < 490:
        raise SystemExit(
            f"Only {len(latest_members)} current S&P 500 stocks have a price row on "
            f"{last_date.date()}"
        )
    print(
        f"Fresh point-in-time ranking date: {last_date.date()} "
        f"({len(latest_members)} current constituents with prices)",
        flush=True,
    )
    return dataset, last_date, len(current_tickers)


def get_instrument_map(client: Trading212PracticeClient) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for item in client.instruments():
        if item.get("type") != "STOCK" or item.get("currencyCode") != "USD":
            continue
        keys = {
            symbol_key(item.get("ticker")),
            symbol_key(item.get("shortName")),
        } - {""}
        for key in keys:
            candidates.setdefault(key, []).append(item)
    return {
        key: items[0]
        for key, items in candidates.items()
        if len(items) == 1
    }


def write_plan_csv(manifest: dict[str, Any]) -> None:
    submitted_by_ticker = {
        str(item["ticker"]): item for item in manifest.get("submitted_orders", [])
    }
    rows = []
    for order in manifest.get("orders", []):
        row = dict(order)
        submitted = submitted_by_ticker.get(str(order["trading212_ticker"]))
        row["order_id"] = submitted.get("id") if submitted else None
        row["order_status"] = submitted.get("status") if submitted else None
        row["submission_state"] = "accepted" if submitted else "not_submitted"
        rows.append(row)
    csv_path = Path(manifest.get("csv_path") or DEFAULT_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def prepare_plan(plan_path: Path, csv_path: Path) -> None:
    dataset, signal_date, current_member_count = refresh_strategy_frame()
    config = SP500SelectorConfig(
        mean_reversion_rsi_threshold=DEFAULT_MEAN_REVERSION_RSI_THRESHOLD
    )
    ranking = rank_sp500_stocks(
        dataset, config, ranking_dates=[signal_date],
    ).sort_values(["selection_rank", "ticker"])
    if ranking.empty:
        raise SystemExit("The current RSI-40 selector returned no eligible stocks")
    ranking = ranking.head(MAX_POSITIONS).copy()

    account_client = Trading212PracticeClient(
        Trading212Credentials.from_env_file(PROJECT_ROOT / ".env")
    )
    account = account_client.account_summary()
    if account.get("currency") != "AUD":
        raise SystemExit(
            f"Expected the verified AUD demo account; got {account.get('currency')!r}"
        )
    available = float((account.get("cash") or {}).get("availableToTrade") or 0.0)
    if available + 1e-9 < ACCOUNT_BUDGET_AUD:
        raise SystemExit(
            f"Demo cash is {available:.2f} AUD, below the requested {ACCOUNT_BUDGET_AUD:.2f} AUD"
        )
    if account_client.positions():
        raise SystemExit("Demo account has open positions; refusing to layer this basket")
    if account_client.pending_orders():
        raise SystemExit("Demo account has pending orders; refusing to create duplicates")

    instruments = get_instrument_map(account_client)
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    for row in ranking.to_dict("records"):
        source = str(row["ticker"]).strip().upper()
        item = instruments.get(symbol_key(source))
        expected_isin = VERIFIED_TICKER_CHANGE_ISINS.get(source)
        if item is not None and expected_isin and item.get("isin") != expected_isin:
            item = None
        if item is None:
            names.append(source)
        else:
            row["trading212_ticker"] = item.get("ticker")
            row["trading212_name"] = item.get("name") or item.get("shortName")
            row["instrument_currency"] = item.get("currencyCode")
            rows.append(row)
    if names:
        raise SystemExit(
            "Selected strategy stocks do not all have an unambiguous USD Trading 212 "
            f"stock instrument; no orders prepared: {', '.join(names)}"
        )
    if not rows:
        raise SystemExit("The strategy basket is empty")

    deploy_aud = min(ACCOUNT_BUDGET_AUD, available) * DEPLOYMENT_FRACTION
    per_position_aud = deploy_aud / len(rows)
    per_position_usd = per_position_aud / USD_TO_AUD_BUDGET_RATE
    plan_orders: list[dict[str, Any]] = []
    for row in rows:
        close = float(row.get("close") or 0.0)
        if not math.isfinite(close) or close <= 0:
            raise SystemExit(f"Invalid fresh close for {row['ticker']}")
        quantity = math.floor((per_position_usd / close) * 10_000.0) / 10_000.0
        if not math.isfinite(quantity) or quantity <= 0:
            raise SystemExit(f"Invalid order quantity for {row['ticker']}")
        plan_orders.append({
            "rank": int(row["selection_rank"]),
            "source_ticker": str(row["ticker"]),
            "trading212_ticker": str(row["trading212_ticker"]),
            "name": str(row["trading212_name"]),
            "instrument_currency": "USD",
            "signal_rsi14": float(row["decision_rsi14"]),
            "signal_date": signal_date.date().isoformat(),
            "reference_close_usd": close,
            "target_aud": per_position_aud,
            "estimated_aud": quantity * close * USD_TO_AUD_BUDGET_RATE,
            "quantity": quantity,
        })

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "environment": "demo",
        "product": "equity_invest_or_isa_not_cfd",
        "mode": "prepared_not_submitted",
        "strategy": "Sharadar point-in-time S&P 500 selector, RSI-40, top 25, six-month hold",
        "signal_date": signal_date.date().isoformat(),
        "current_constituent_count": current_member_count,
        "account_currency": "AUD",
        "available_cash_aud_at_prepare": available,
        "requested_budget_aud": ACCOUNT_BUDGET_AUD,
        "deployment_fraction": DEPLOYMENT_FRACTION,
        "planned_budget_aud": deploy_aud,
        "unallocated_reserve_aud": ACCOUNT_BUDGET_AUD - deploy_aud,
        "usd_to_aud_sizing_rate_with_buffer": USD_TO_AUD_BUDGET_RATE,
        "order_type": "market",
        "extended_hours": False,
        "order_queue_note": "Market is closed on Sunday; accepted orders may queue for the next US session.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path.resolve()),
        "orders": plan_orders,
        "submitted_orders": [],
    }
    plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plan_csv(manifest)
    print(json.dumps({
        "environment": manifest["environment"],
        "product": manifest["product"],
        "mode": manifest["mode"],
        "signal_date": manifest["signal_date"],
        "strategy": manifest["strategy"],
        "account_currency": "AUD",
        "available_cash_aud": round(available, 2),
        "planned_budget_aud": round(deploy_aud, 2),
        "cash_reserve_aud": round(ACCOUNT_BUDGET_AUD - deploy_aud, 2),
        "order_count": len(plan_orders),
        "orders": [
            {"rank": x["rank"], "ticker": x["trading212_ticker"],
             "quantity": x["quantity"], "reference_close_usd": x["reference_close_usd"],
             "target_aud": round(x["target_aud"], 2), "rsi14": round(x["signal_rsi14"], 2)}
            for x in plan_orders
        ],
        "plan_file": str(plan_path.resolve()),
        "csv_file": str(csv_path.resolve()),
    }, indent=2))
    print("Preview only: no Trading 212 order was sent. Use --submit after reviewing this plan.")


def submit_plan(plan_path: Path) -> None:
    if not plan_path.is_file():
        raise SystemExit(f"Prepared plan does not exist: {plan_path}")
    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest.get("environment") != "demo" or manifest.get("product") != "equity_invest_or_isa_not_cfd":
        raise SystemExit("Plan is not locked to the Trading 212 equity demo environment")
    if manifest.get("mode") != "prepared_not_submitted":
        raise SystemExit(f"Plan is not in a submit-ready state: {manifest.get('mode')}")
    signal_date = pd.Timestamp(manifest["signal_date"]).normalize()
    today = pd.Timestamp.now(tz="Australia/Sydney").tz_localize(None).normalize()
    if (today - signal_date).days > 3:
        raise SystemExit(
            f"Plan signal date {signal_date.date()} is too old to submit on {today.date()}"
        )
    orders = manifest.get("orders") or []
    if not 1 <= len(orders) <= MAX_POSITIONS:
        raise SystemExit(f"Unexpected order count in plan: {len(orders)}")
    for order in orders:
        qty = float(order.get("quantity") or 0.0)
        if not math.isfinite(qty) or qty <= 0 or order.get("instrument_currency") != "USD":
            raise SystemExit("Plan has an invalid quantity or unsupported currency")

    client = Trading212PracticeClient(
        Trading212Credentials.from_env_file(PROJECT_ROOT / ".env")
    )
    account = client.account_summary()
    if account.get("currency") != manifest.get("account_currency"):
        raise SystemExit("Demo account currency changed after plan preparation")
    available = float((account.get("cash") or {}).get("availableToTrade") or 0.0)
    if available + 1e-9 < float(manifest.get("planned_budget_aud") or 0.0):
        raise SystemExit("Available demo cash is below the prepared budget")
    if client.positions():
        raise SystemExit("Demo account has open positions; refusing to duplicate the basket")
    if client.pending_orders():
        raise SystemExit("Demo account has pending orders; refusing to duplicate the basket")

    manifest["mode"] = "submitting_demo_orders"
    manifest["submission_started_at"] = datetime.now(timezone.utc).isoformat()
    plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Submitting {len(orders)} BUY market orders to the Trading 212 demo Equity account. "
        "The API is quantity-based; no live-money endpoint is used.",
        flush=True,
    )
    for order in orders:
        try:
            response = client.market_order(
                ticker=order["trading212_ticker"],
                quantity=float(order["quantity"]),
            )
        except Trading212Error as exc:
            manifest["mode"] = "submission_incomplete_check_demo_history_before_retry"
            manifest["submission_error"] = str(exc)
            manifest["submission_finished_at"] = datetime.now(timezone.utc).isoformat()
            plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(
                f"Stopped after {len(manifest['submitted_orders'])} accepted order(s). "
                f"Check the Practice order history before retrying. {exc}"
            ) from None
        if not isinstance(response, dict) or not response.get("id"):
            manifest["mode"] = "submission_incomplete_unknown_order_outcome"
            manifest["submission_error"] = "Response omitted order ID"
            plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(
                f"Stopped after {len(manifest['submitted_orders'])} accepted order(s): "
                "the latest outcome is uncertain. Check Practice history before retrying."
            )
        accepted = {
            "id": int(response["id"]),
            "ticker": order["trading212_ticker"],
            "quantity": float(order["quantity"]),
            "status": response.get("status"),
            "created_at": response.get("createdAt"),
        }
        manifest["submitted_orders"].append(accepted)
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_plan_csv(manifest)
        print(json.dumps(accepted), flush=True)

    manifest["mode"] = "submitted_waiting_for_market_or_fill"
    manifest["submission_finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        pending = client.pending_orders()
        pending_by_id = {int(item["id"]): item for item in pending if item.get("id") is not None}
        manifest["post_submit_status_check"] = {
            "complete": True,
            "orders_found": len([
                order for order in manifest["submitted_orders"]
                if order["id"] in pending_by_id
            ]),
            "statuses": {
                str(order["id"]): pending_by_id[order["id"]].get("status")
                for order in manifest["submitted_orders"]
                if order["id"] in pending_by_id
            },
        }
    except Trading212Error as exc:
        manifest["post_submit_status_check"] = {
            "complete": False,
            "error": str(exc),
            "note": "All listed POST requests returned order IDs; do not resubmit.",
        }
    plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plan_csv(manifest)
    print(json.dumps({
        "environment": "demo",
        "mode": manifest["mode"],
        "accepted_count": len(manifest["submitted_orders"]),
        "signal_date": manifest["signal_date"],
        "status_check": manifest.get("post_submit_status_check"),
        "plan_file": str(plan_path.resolve()),
    }, indent=2))


def resume_plan(plan_path: Path) -> None:
    if not plan_path.is_file():
        raise SystemExit(f"Prepared plan does not exist: {plan_path}")
    manifest = json.loads(plan_path.read_text(encoding="utf-8"))
    if manifest.get("environment") != "demo" or manifest.get("product") != "equity_invest_or_isa_not_cfd":
        raise SystemExit("Plan is not locked to the Trading 212 equity demo environment")
    if manifest.get("mode") != "submission_incomplete_check_demo_history_before_retry":
        raise SystemExit(f"Plan is not in the recoverable state: {manifest.get('mode')}")
    failure = str(manifest.get("submission_error", ""))
    if "quantity-precision-mismatch" not in failure:
        raise SystemExit(
            "Only a definitive quantity-precision rejection can be resumed automatically"
        )

    client = Trading212PracticeClient(
        Trading212Credentials.from_env_file(PROJECT_ROOT / ".env")
    )
    account = client.account_summary()
    if account.get("currency") != manifest.get("account_currency"):
        raise SystemExit("Demo account currency changed after plan preparation")
    pending = client.pending_orders()
    pending_by_id = {
        int(item["id"]): item
        for item in pending if item.get("id") is not None
    }
    submitted = manifest.get("submitted_orders") or []
    accepted_ids = {int(item["id"]) for item in submitted}
    if not accepted_ids:
        raise SystemExit("There are no accepted orders to verify")
    missing_ids = accepted_ids - set(pending_by_id)
    extra_ids = set(pending_by_id) - accepted_ids
    if missing_ids:
        raise SystemExit(
            "Previously accepted order(s) are no longer pending. Check their current "
            "status and Practice positions manually; no additional orders were sent."
        )
    if extra_ids:
        raise SystemExit(
            "The demo account has an unrecognized pending order; no additional orders were sent."
        )
    accepted_tickers = {str(item["ticker"]) for item in submitted}
    positions = client.positions()
    unexpected_positions = [
        str((item.get("instrument") or {}).get("ticker", ""))
        for item in positions
        if str((item.get("instrument") or {}).get("ticker", "")) not in accepted_tickers
    ]
    if unexpected_positions:
        raise SystemExit(
            "The demo account has unexpected open positions; no additional orders were sent."
        )

    account_cash = float((account.get("cash") or {}).get("availableToTrade") or 0.0)
    planned = manifest.get("orders") or []
    remaining = [
        item for item in planned
        if str(item["trading212_ticker"]) not in accepted_tickers
    ]
    remaining_budget = sum(float(item.get("target_aud") or 0.0) for item in remaining)
    if account_cash + 1e-9 < remaining_budget:
        raise SystemExit(
            f"Available demo cash ({account_cash:.2f} AUD) is below the remaining "
            f"planned allocation ({remaining_budget:.2f} AUD)"
        )

    # The first 4-decimal quantity was accepted, but the next stock rejected 4
    # decimals and reported a precision limit. Round the remaining quantities
    # down to two decimals, which is conservative and keeps every order below
    # its planned notional. Any further rejection stops the batch again.
    adjustments = []
    for item in remaining:
        original = float(item["quantity"])
        adjusted = math.floor((original + 1e-10) * 100.0) / 100.0
        if not math.isfinite(adjusted) or adjusted <= 0:
            raise SystemExit(f"Precision adjustment made {item['source_ticker']} quantity zero")
        item["quantity_before_precision_adjustment"] = original
        item["quantity"] = adjusted
        item["estimated_aud"] = (
            adjusted * float(item["reference_close_usd"]) * USD_TO_AUD_BUDGET_RATE
        )
        adjustments.append({
            "ticker": item["trading212_ticker"],
            "old_quantity": original,
            "new_quantity": adjusted,
        })

    manifest["mode"] = "submitting_resumed_demo_orders"
    manifest["precision_adjustments"] = adjustments
    manifest["resume_started_at"] = datetime.now(timezone.utc).isoformat()
    plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plan_csv(manifest)
    print(json.dumps({
        "environment": "demo",
        "mode": "resume_preview",
        "already_accepted": len(submitted),
        "remaining_orders": len(remaining),
        "remaining_planned_allocation_aud": round(remaining_budget, 2),
        "precision": "remaining quantities rounded down to 2 decimals",
        "adjustments": adjustments,
    }, indent=2), flush=True)

    for item in remaining:
        try:
            response = client.market_order(
                ticker=item["trading212_ticker"], quantity=float(item["quantity"])
            )
        except Trading212Error as exc:
            manifest["mode"] = "submission_incomplete_check_demo_history_before_retry"
            manifest["submission_error"] = str(exc)
            manifest["resume_finished_at"] = datetime.now(timezone.utc).isoformat()
            plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            write_plan_csv(manifest)
            raise SystemExit(
                f"Stopped after {len(manifest['submitted_orders'])} accepted order(s). "
                f"Check Practice history before retrying. {exc}"
            ) from None
        if not isinstance(response, dict) or not response.get("id"):
            manifest["mode"] = "submission_incomplete_unknown_order_outcome"
            manifest["submission_error"] = "Response omitted order ID"
            plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            write_plan_csv(manifest)
            raise SystemExit(
                f"Stopped after {len(manifest['submitted_orders'])} accepted order(s): "
                "latest outcome is uncertain. Check Practice history before retrying."
            )
        accepted = {
            "id": int(response["id"]),
            "ticker": item["trading212_ticker"],
            "quantity": float(item["quantity"]),
            "status": response.get("status"),
            "created_at": response.get("createdAt"),
        }
        manifest["submitted_orders"].append(accepted)
        plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        write_plan_csv(manifest)
        print(json.dumps(accepted), flush=True)

    manifest["mode"] = "submitted_waiting_for_market_or_fill"
    manifest["resume_finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        refreshed = client.pending_orders()
        refreshed_by_id = {
            int(item["id"]): item for item in refreshed if item.get("id") is not None
        }
        manifest["post_submit_status_check"] = {
            "complete": True,
            "orders_found": len([
                item for item in manifest["submitted_orders"]
                if int(item["id"]) in refreshed_by_id
            ]),
            "statuses": {
                str(item["id"]): refreshed_by_id[int(item["id"])].get("status")
                for item in manifest["submitted_orders"]
                if int(item["id"]) in refreshed_by_id
            },
        }
    except Trading212Error as exc:
        manifest["post_submit_status_check"] = {
            "complete": False,
            "error": str(exc),
            "note": "All listed POST requests returned order IDs; do not resubmit.",
        }
    plan_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_plan_csv(manifest)
    print(json.dumps({
        "environment": "demo",
        "mode": manifest["mode"],
        "accepted_count": len(manifest["submitted_orders"]),
        "signal_date": manifest["signal_date"],
        "status_check": manifest.get("post_submit_status_check"),
        "plan_file": str(plan_path.resolve()),
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.submit:
        submit_plan(args.plan)
    elif args.resume:
        resume_plan(args.plan)
    else:
        prepare_plan(args.plan, args.csv)


if __name__ == "__main__":
    main()
