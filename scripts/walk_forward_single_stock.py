#!/usr/bin/env python3
"""Walk-forward and parameter-stability evaluation for the RSI bot."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rsi_momentum_bot import (  # noqa: E402
    MIN_BUY_MOMENTUM_3M,
    MIN_BUY_RSI,
    MIN_PRICE_VS_200D_MA,
    RSIBuySignal,
    SIGNAL_LAG_DAYS,
    generate_buy_sell_signals,
)
from scripts.run_single_stock import (  # noqa: E402
    attach_market_context,
    fetch_stock_data,
    simple_backtest,
)


RSI_REENTRIES = (38.0, 40.0, 42.0)
MAX_BUY_RSIS = (58.0, 62.0, 66.0)
TRAIN_DAYS = 252 * 3
TEST_DAYS = 126


def parameterized_signals(
    prepared: pd.DataFrame,
    rsi_reentry: float,
    max_buy_rsi: float,
) -> pd.DataFrame:
    """Rebuild only the entry event while retaining the production exit."""

    frame = prepared.sort_values(["ticker", "date"]).copy()
    range_entry = (
        (frame["confirmed_lower_reentry"] | (frame["below_range"] & frame["rsi_rising"]))
        & frame["rsi_rising"]
        & frame["rsi"].between(MIN_BUY_RSI, max_buy_rsi)
        & (frame["momentum_3m"] >= MIN_BUY_MOMENTUM_3M)
        & (frame["price_vs_200d_ma"] >= MIN_PRICE_VS_200D_MA)
        & frame["market_regime_ok"].fillna(False)
        & (frame["relative_strength_3m"].fillna(0.0) >= -0.02)
    )
    trend_entry = (
        (frame.groupby("ticker")["rsi"].shift(1) <= rsi_reentry)
        & (frame["rsi"] > rsi_reentry)
        & (frame["close"] > frame["ma_200d"])
        & (frame["momentum_3m"] > 0.0)
        & frame["weekly_momentum_acceptable"]
        & frame["market_regime_ok"].fillna(False)
        & (frame["relative_strength_3m"].fillna(0.0) >= 0.0)
    )
    condition = (
        (range_entry | trend_entry)
        & frame["warmup_ready"].fillna(False)
    )
    event = condition & ~condition.groupby(frame["ticker"]).shift(1).fillna(False)
    frame["buy_signal"] = (
        event.groupby(frame["ticker"]).shift(SIGNAL_LAG_DAYS).fillna(False).astype(float)
    )
    return frame


def objective(result: dict) -> float:
    """Favor excess return while penalising drawdown."""

    return (
        float(result["excess_vs_buy_hold_pct"])
        - 0.5 * float(result["max_drawdown_pct"])
    )


def run_walk_forward(ticker: str, period: str = "10y") -> dict:
    data = attach_market_context(fetch_stock_data(ticker, period), period)
    features = RSIBuySignal().generate_signals(data)
    production = generate_buy_sell_signals(features)
    variants = {
        (reentry, max_rsi): parameterized_signals(production, reentry, max_rsi)
        for reentry, max_rsi in product(RSI_REENTRIES, MAX_BUY_RSIS)
    }

    stability_rows = []
    for (reentry, max_rsi), frame in variants.items():
        result = simple_backtest(frame)
        stability_rows.append(
            {
                "rsi_reentry": reentry,
                "max_buy_rsi": max_rsi,
                "return_pct": result["total_return_pct"],
                "excess_vs_hold_pct": result["excess_vs_buy_hold_pct"],
                "max_drawdown_pct": result["max_drawdown_pct"],
                "sharpe": result["sharpe_ratio"],
                "trades": result["buy_count"] + result["sell_count"],
                "objective": objective(result),
            }
        )

    fold_rows = []
    fold = 1
    for test_start in range(TRAIN_DAYS, len(production), TEST_DAYS):
        test_end = min(test_start + TEST_DAYS, len(production))
        if test_end - test_start < TEST_DAYS // 2:
            break
        train_start = max(0, test_start - TRAIN_DAYS)
        scored = []
        for parameters, frame in variants.items():
            train_result = simple_backtest(frame.iloc[train_start:test_start])
            scored.append((objective(train_result), parameters))
        _, best = max(scored, key=lambda item: item[0])
        test_frame = variants[best].iloc[test_start:test_end]
        test_result = simple_backtest(test_frame)
        fold_rows.append(
            {
                "fold": fold,
                "train_start": production.iloc[train_start]["date"],
                "train_end": production.iloc[test_start - 1]["date"],
                "test_start": production.iloc[test_start]["date"],
                "test_end": production.iloc[test_end - 1]["date"],
                "selected_rsi_reentry": best[0],
                "selected_max_buy_rsi": best[1],
                "test_return_pct": test_result["total_return_pct"],
                "test_buy_hold_pct": test_result["buy_hold_return_pct"],
                "test_excess_pct": test_result["excess_vs_buy_hold_pct"],
                "test_max_drawdown_pct": test_result["max_drawdown_pct"],
                "test_sharpe": test_result["sharpe_ratio"],
                "test_trades": test_result["buy_count"] + test_result["sell_count"],
            }
        )
        fold += 1

    output_dir = Path("results/robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("^", "INDEX_")
    stability_path = output_dir / f"{safe_ticker}_parameter_stability.csv"
    folds_path = output_dir / f"{safe_ticker}_walk_forward.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    compounded_strategy = 1.0
    compounded_hold = 1.0
    for row in fold_rows:
        compounded_strategy *= 1.0 + row["test_return_pct"] / 100.0
        compounded_hold *= 1.0 + row["test_buy_hold_pct"] / 100.0
    summary = {
        "ticker": ticker,
        "folds": len(fold_rows),
        "out_of_sample_return_pct": float((compounded_strategy - 1.0) * 100.0),
        "out_of_sample_buy_hold_pct": float((compounded_hold - 1.0) * 100.0),
        "out_of_sample_excess_pct": float(
            (compounded_strategy - compounded_hold) * 100.0
        ),
        "stability_csv": str(stability_path),
        "walk_forward_csv": str(folds_path),
    }
    summary_path = output_dir / f"{safe_ticker}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/walk_forward_single_stock.py TICKER [period]")
    ticker = sys.argv[1].upper()
    period = sys.argv[2] if len(sys.argv) > 2 else "10y"
    print(json.dumps(run_walk_forward(ticker, period), indent=2))


if __name__ == "__main__":
    main()
