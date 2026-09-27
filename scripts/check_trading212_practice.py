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
    if not isinstance(account, dict):
        raise SystemExit("Trading 212 demo returned an unexpected account response")

    output = {
        "connected": True,
        "environment": "demo",
        "product": "equity_invest_or_isa",
        "read_only": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
