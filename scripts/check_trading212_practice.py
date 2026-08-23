#!/usr/bin/env python3
"""Verify the Trading 212 practice connection without placing orders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trading212_client import Trading212Credentials, Trading212PracticeClient


def main() -> None:
    credentials = Trading212Credentials.from_env_file(PROJECT_ROOT / ".env")
    client = Trading212PracticeClient(credentials)
    account = client.account_summary()
    positions = client.positions()
    pending = client.pending_orders()
    instruments = client.instruments()

    output = {
        "connected": True,
        "environment": "demo",
        "account": account,
        "positions_count": len(positions),
        "pending_orders_count": len(pending),
        "instrument_count": len(instruments),
        "aud_instruments": sum(item.get("currencyCode") == "AUD" for item in instruments),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
