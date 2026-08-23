"""Shared portfolio execution configuration."""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class PortfolioConfig:
    slippage_per_trade: float = 0.0005

PORTFOLIO = PortfolioConfig()
