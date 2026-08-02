#!/usr/bin/env python3
"""Compatibility entry point for single-stock interactive reports."""

from __future__ import annotations

import sys

from run_single_stock import main as run_single_stock_main


def main() -> None:
    """Create the interactive report through the single-stock pipeline."""

    if len(sys.argv) < 2:
        print("Usage: python scripts/create_html_report.py <TICKER> [period]")
        print("Example: python scripts/create_html_report.py NVDA")
        raise SystemExit(1)

    run_single_stock_main()


if __name__ == "__main__":
    main()
