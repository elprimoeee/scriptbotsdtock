#!/usr/bin/env python3
"""Dry-run or confirm one market buy in Trading 212's equity demo (not CFD)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trading212_client import (
    Trading212Credentials,
    Trading212Error,
    Trading212PracticeClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Exact Trading 212 stock ticker, e.g. AAPL_US_EQ")
    parser.add_argument("--quantity", required=True, type=float, help="Small positive fractional share quantity")
    parser.add_argument("--submit", action="store_true", help="Submit one order to the demo account after interactive confirmation")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.quantity) or args.quantity <= 0:
        raise SystemExit("This smoke test accepts a finite positive quantity (buy only).")

    client = Trading212PracticeClient(Trading212Credentials.from_env_file(args.env_file))
    ticker = args.ticker.strip().upper()
    instruments = client.instruments()
    instrument = next((item for item in instruments if item.get("ticker") == ticker), None)
    if instrument is None or instrument.get("type") != "STOCK":
        raise SystemExit("Ticker was not found as a Trading 212 stock instrument.")

    summary = client.account_summary()
    mode = "submit" if args.submit else "dry-run"
    preview = {
        "environment": "demo",
        "product": "equity_invest_or_isa_not_cfd",
        "mode": mode,
        "side": "BUY",
        "ticker": ticker,
        "name": instrument.get("name") or instrument.get("shortName"),
        "instrument_currency": instrument.get("currencyCode"),
        "account_currency": summary.get("currency"),
        "quantity": args.quantity,
        "extended_hours": False,
    }

    if not args.submit:
        print(json.dumps(preview, indent=2, sort_keys=True))
        print("Dry run only: no order was sent. Add --submit to continue to the confirmation prompt.")
        print("The API order is quantity-based and this preview has no live price estimate; verify notional in Practice mode before submitting.")
        return

    confirmation = f"BUY {ticker} {args.quantity:g} EQUITY DEMO"
    print(json.dumps(preview, indent=2, sort_keys=True))
    print("This is a market order. It may slip, and if the exchange is closed Trading 212 may queue it for the next session.")
    typed = input(f"To submit exactly this demo order, type {confirmation!r}: ").strip()
    if typed != confirmation:
        raise SystemExit("Confirmation did not match; no order was sent.")

    # Do not retry this POST. A timeout can leave the order's outcome uncertain.
    placed = client.market_order(ticker=ticker, quantity=args.quantity)
    if not isinstance(placed, dict) or not placed.get("id"):
        raise SystemExit(
            "The order response did not include an ID. Check demo order history before retrying."
        )
    order_id = int(placed["id"])
    try:
        current = client.order_by_id(order_id)
    except Trading212Error as exc:
        partial_result = {
            "environment": "demo",
            "order_id": order_id,
            "ticker": ticker,
            "quantity": args.quantity,
            "status": placed.get("status"),
            "status_check": "failed",
        }
        print(json.dumps(partial_result, indent=2, sort_keys=True))
        raise SystemExit(
            "Order was submitted and has the ID shown above. Do not resubmit; check its status in Practice mode."
        ) from exc
    result = {
        "environment": "demo",
        "order_id": order_id,
        "ticker": current.get("ticker", ticker),
        "side": current.get("side", "BUY"),
        "quantity": current.get("quantity", args.quantity),
        "filled_quantity": current.get("filledQuantity"),
        "status": current.get("status", placed.get("status")),
        "currency": current.get("currency", placed.get("currency")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
