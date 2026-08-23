"""Fixed-period buy-and-hold portfolio driven by the stock selector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from models.stock_selector import rank_stocks
from project_config import PORTFOLIO


TRADING_DAYS_PER_MONTH = 21


@dataclass
class SelectorHoldResult:
    summary: dict[str, object]
    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    rebalances: pd.DataFrame
    latest_ranking: pd.DataFrame
    report_path: Path


def _commission(
    notional: float,
    rate: float = 0.0,
    minimum: float = 0.0,
) -> float:
    if notional <= 0.0:
        return 0.0
    return max(
        minimum,
        notional * rate,
    )


def _prepare_prices(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "close", "dollar_volume"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise KeyError(f"Combined backtest dataset is missing: {', '.join(missing)}")
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = (
        frame["ticker"].astype(str).str.upper().str.replace(r"\.AX$", "", regex=True)
    )
    for column in ("open", "close", "adj_close", "dollar_volume", "market_cap", "market_cap_rank"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    adjustment = (
        (frame["adj_close"] / frame["close"]).replace([np.inf, -np.inf], np.nan)
        if "adj_close" in frame
        else pd.Series(1.0, index=frame.index)
    )
    adjustment = adjustment.fillna(1.0)
    frame["execution_open"] = frame["open"] * adjustment
    frame["mark_close"] = frame["adj_close"] if "adj_close" in frame else frame["close"]
    return frame.dropna(subset=["date", "ticker", "execution_open", "mark_close"]).sort_values(
        ["ticker", "date"]
    ).drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)


def build_rebalance_dates(dates: Iterable[object], holding_days: int) -> list[pd.Timestamp]:
    """Take the first session and then every holding_days-th trading session."""

    if holding_days < 1:
        raise ValueError("holding_days must be positive")
    sessions = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="coerce")).dropna().unique().sort_values()
    return [pd.Timestamp(value) for value in sessions[::holding_days]]


def _metrics(equity: pd.Series, initial_capital: float) -> dict[str, float]:
    daily_returns = equity.pct_change().fillna(0.0)
    years = max((len(equity) - 1) / 252.0, 1.0 / 252.0)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
    standard_deviation = float(daily_returns.std(ddof=0))
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "annual_volatility_pct": standard_deviation * math.sqrt(252.0) * 100.0,
        "sharpe_ratio": (
            float(daily_returns.mean() / standard_deviation * math.sqrt(252.0))
            if standard_deviation > 0.0 else 0.0
        ),
        "max_drawdown_pct": abs(float(drawdown.min())) * 100.0,
    }


def _write_report(
    output_dir: Path,
    summary: dict[str, object],
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    rebalances: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    holding_months = max(1, round(int(summary["holding_trading_days"]) / TRADING_DAYS_PER_MONTH))
    holding_label = f"{holding_months}-Month"
    market_name = str(summary.get("market_name", "S&P 500"))
    benchmark_name = str(summary.get("benchmark_name", "S&P 500 (SPY)"))
    currency_symbol = str(summary.get("currency_symbol", "A$"))
    report_path = output_dir / f"selector_{holding_months}m_hold_report.html"
    report_daily = daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d")).copy()
    for column in ("equity", "drawdown_pct", "benchmark_equity", "benchmark_drawdown_pct"):
        if column in report_daily:
            report_daily[column] = report_daily[column].map(
                lambda value: None if pd.isna(value) else float(value)
            ).astype(object)
    daily_rows = report_daily.to_dict("records")
    trade_rows = trades.copy()
    if not trade_rows.empty:
        trade_rows["date"] = pd.to_datetime(trade_rows["date"]).dt.strftime("%Y-%m-%d")
    rebalance_rows = rebalances.copy()
    if not rebalance_rows.empty:
        rebalance_rows["date"] = pd.to_datetime(rebalance_rows["date"]).dt.strftime("%Y-%m-%d")
    card_values = [
            ("Period", f"{summary['period_start']} to {summary['period_end']}"),
            ("Strategy return", f"{summary['total_return_pct']:+.2f}%"),
            ("CAGR", f"{summary['cagr_pct']:+.2f}%"),
            ("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%"),
            ("Final equity", f"{currency_symbol}{summary['final_equity']:,.2f}"),
            ("Rebalances", f"{summary['rebalance_count']:,}"),
            ("Orders", f"{summary['total_orders']:,}"),
            ("Fees", f"{currency_symbol}{summary['total_fees']:,.2f}"),
    ]
    if "benchmark_total_return_pct" in summary:
        card_values[2:2] = [
            (f"{benchmark_name} return", f"{summary['benchmark_total_return_pct']:+.2f}%"),
            ("Excess return", f"{summary['excess_total_return_pct']:+.2f}%"),
        ]
    cards = "".join(
        f"<div class='card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in card_values
    )
    research_limitation = str(summary.get(
        "research_limitation",
        "Current index membership and current shares-outstanding metadata are applied historically. "
        "Results contain survivorship and size look-ahead bias. Adjusted prices are used for portfolio returns.",
    ))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{market_name} Selector — {holding_label} Hold</title><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script><style>
body{{margin:0;background:#f5f7fa;color:#17212b;font-family:Segoe UI,Arial,sans-serif}}header{{background:#111820;color:white;padding:24px 30px}}main{{max-width:1400px;margin:auto;padding:18px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.card,.panel{{background:white;border:1px solid #d9e1e8;border-radius:9px;padding:14px}}.card span{{display:block;color:#667582;font-size:12px;margin-bottom:8px}}.card strong{{font-size:18px}}.warning{{margin:16px 0;padding:13px;border-left:4px solid #d97706;background:#fff7ed}}.grid{{display:grid;grid-template-columns:2fr 1fr;gap:14px;margin-top:14px}}.chart{{height:420px}}table{{width:100%;border-collapse:collapse;background:white;margin-top:14px}}th,td{{padding:9px;border-bottom:1px solid #e3e8ed;text-align:left;font-size:13px}}th{{color:#607080}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<header><h1>{market_name} Selector — {holding_label} Buy & Hold</h1><div>Top {summary['selector_top_n']} equal-weight portfolio, reselected every {summary['holding_trading_days']} trading sessions</div></header><main>
<div class='warning'><strong>Research limitation:</strong> {research_limitation}</div>
<section class='cards'>{cards}</section><section class='grid'><div class='panel'><div id='equity' class='chart'></div></div><div class='panel'><div id='drawdown' class='chart'></div></div></section>
<table><caption><h2>Rebalances</h2></caption><thead><tr><th>Date</th><th>Selected</th><th>Tickers</th><th>Equity after trades</th></tr></thead><tbody id='rebalances'></tbody></table>
<table><caption><h2>Orders</h2></caption><thead><tr><th>Date</th><th>Action</th><th>Ticker</th><th>Shares</th><th>Price</th><th>Value</th><th>Commission</th></tr></thead><tbody id='trades'></tbody></table>
<script>const daily={json.dumps(daily_rows)};const trades={json.dumps(trade_rows.to_dict('records'))};const rebalances={json.dumps(rebalance_rows.to_dict('records'))};
const cfg={{responsive:true,displaylogo:false}};const benchmark=daily.filter(x=>x.benchmark_equity!==null&&x.benchmark_equity!==undefined);
const equityTraces=[{{x:daily.map(x=>x.date),y:daily.map(x=>x.equity),mode:'lines',name:'Selector portfolio',line:{{color:'#087f5b',width:2.5}}}}];
if(benchmark.length) equityTraces.push({{x:benchmark.map(x=>x.date),y:benchmark.map(x=>x.benchmark_equity),mode:'lines',name:{json.dumps(benchmark_name)},line:{{color:'#2563eb',width:2}}}});
Plotly.newPlot('equity',equityTraces,{{title:'Portfolio vs {benchmark_name}',yaxis:{{title:'Growth of {currency_symbol}{summary['initial_capital']:,.0f}'}},legend:{{orientation:'h'}},hovermode:'x unified',margin:{{t:50}}}},cfg);
const drawdownTraces=[{{x:daily.map(x=>x.date),y:daily.map(x=>x.drawdown_pct),mode:'lines',name:'Selector portfolio',line:{{color:'#c0392b'}}}}];
if(benchmark.length) drawdownTraces.push({{x:benchmark.map(x=>x.date),y:benchmark.map(x=>x.benchmark_drawdown_pct),mode:'lines',name:{json.dumps(benchmark_name)},line:{{color:'#2563eb'}}}});
Plotly.newPlot('drawdown',drawdownTraces,{{title:'Drawdown comparison',yaxis:{{title:'%'}},legend:{{orientation:'h'}},hovermode:'x unified',margin:{{t:50}}}},cfg);
document.querySelector('#rebalances').innerHTML=rebalances.map(x=>`<tr><td>${{x.date}}</td><td>${{x.selected_count}}</td><td>${{x.tickers}}</td><td>{currency_symbol}${{Number(x.equity_after_trades).toLocaleString(undefined,{{maximumFractionDigits:2}})}}</td></tr>`).join('');
document.querySelector('#trades').innerHTML=trades.slice().reverse().map(x=>`<tr><td>${{x.date}}</td><td>${{x.action}}</td><td>${{x.ticker}}</td><td>${{x.shares}}</td><td>{currency_symbol}${{Number(x.price).toFixed(2)}}</td><td>{currency_symbol}${{Number(x.value).toFixed(2)}}</td><td>{currency_symbol}${{Number(x.commission).toFixed(2)}}</td></tr>`).join('');</script></main></body></html>"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_selector_hold_backtest(
    dataset: pd.DataFrame,
    *,
    constituents: Iterable[object] | None,
    allow_market_cap_proxy: bool = False,
    years: int = 5,
    top_n: int = 25,
    holding_days: int = 126,
    initial_capital: float = 20_000.0,
    output_dir: Path = Path("results/selector_hold_5y"),
    benchmark: pd.DataFrame | None = None,
    market_name: str = "S&P 500",
    benchmark_name: str = "S&P 500 (SPY, Sharadar)",
    benchmark_symbol: str = "SPY",
    currency_symbol: str = "$",
    commission_rate: float = 0.0,
    minimum_commission: float = 0.0,
    ranking_function: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    data_source: str = "local_dataset",
    research_limitation: str | None = None,
) -> SelectorHoldResult:
    """Buy selector leaders, hold six months, then fully reselect."""

    prices = _prepare_prices(dataset)
    ranked = (
        ranking_function(prices)
        if ranking_function is not None
        else rank_stocks(
            prices,
            constituents=constituents,
            allow_market_cap_proxy=allow_market_cap_proxy,
        )
    )
    latest_date = prices["date"].max()
    requested_start = latest_date - pd.DateOffset(years=years)
    sessions = pd.DatetimeIndex(prices.loc[prices["date"] >= requested_start, "date"].unique()).sort_values()
    if sessions.empty:
        raise ValueError("No sessions exist in the requested backtest window")
    rebalance_dates = set(build_rebalance_dates(sessions, holding_days))
    price_days = {pd.Timestamp(date): day.set_index("ticker") for date, day in prices.loc[prices["date"].isin(sessions)].groupby("date")}
    rank_days = {pd.Timestamp(date): day.sort_values("selection_rank") for date, day in ranked.loc[ranked["date"].isin(sessions)].groupby("date")}
    final_price_dates = prices.groupby("ticker")["date"].max().to_dict()

    cash = float(initial_capital)
    positions: dict[str, dict[str, float | int | pd.Timestamp]] = {}
    last_marks: dict[str, float] = {}
    trade_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    completed_returns: list[float] = []
    total_fees = 0.0

    for date in sessions:
        date = pd.Timestamp(date)
        day = price_days[date]
        for ticker, row in day.iterrows():
            last_marks[ticker] = float(row["mark_close"])

        if date in rebalance_dates:
            # Full liquidation makes the six-month holding period explicit and auditable.
            for ticker, position in list(positions.items()):
                if ticker not in day.index:
                    continue
                fill = float(day.loc[ticker, "execution_open"]) * (1.0 - PORTFOLIO.slippage_per_trade)
                shares = int(position["shares"])
                value = shares * fill
                commission = _commission(value, commission_rate, minimum_commission)
                cash += value - commission
                total_fees += commission
                entry_cost = float(position["entry_cost"])
                completed_returns.append((value - commission) / entry_cost - 1.0)
                trade_rows.append({"date": date, "action": "SELL", "ticker": ticker, "shares": shares, "price": fill, "value": value, "commission": commission})
                del positions[ticker]

            candidates = rank_days.get(date, pd.DataFrame()).head(top_n)
            candidates = candidates[candidates["ticker"].isin(day.index)].copy() if not candidates.empty else candidates
            candidate_tickers = candidates["ticker"].tolist() if not candidates.empty else []
            target_value = cash / len(candidate_tickers) if candidate_tickers else 0.0
            for ticker in candidate_tickers:
                fill = float(day.loc[ticker, "execution_open"]) * (1.0 + PORTFOLIO.slippage_per_trade)
                shares = max(0, int((target_value - minimum_commission) / fill))
                while shares > 0:
                    value = shares * fill
                    commission = _commission(value, commission_rate, minimum_commission)
                    if value + commission <= cash:
                        break
                    shares -= 1
                if shares <= 0:
                    continue
                value = shares * fill
                commission = _commission(value, commission_rate, minimum_commission)
                cash -= value + commission
                total_fees += commission
                positions[ticker] = {"shares": shares, "entry_cost": value + commission, "entry_date": date}
                trade_rows.append({"date": date, "action": "BUY", "ticker": ticker, "shares": shares, "price": fill, "value": value, "commission": commission})

            marked_equity = cash + sum(int(position["shares"]) * last_marks.get(ticker, 0.0) for ticker, position in positions.items())
            rebalance_rows.append({"date": date, "selected_count": len(positions), "tickers": ", ".join(positions), "equity_after_trades": marked_equity})

        # A delisted/acquired security must not remain frozen at its final quote forever.
        # Sharadar adjusted prices include distributions and spinoff adjustments; realize
        # the final marked value on the security's last observed session.
        for ticker, position in list(positions.items()):
            if final_price_dates.get(ticker) != date or date == sessions[-1] or ticker not in day.index:
                continue
            fill = float(day.loc[ticker, "mark_close"]) * (1.0 - PORTFOLIO.slippage_per_trade)
            shares = int(position["shares"])
            value = shares * fill
            commission = _commission(value, commission_rate, minimum_commission)
            cash += value - commission
            total_fees += commission
            entry_cost = float(position["entry_cost"])
            completed_returns.append((value - commission) / entry_cost - 1.0)
            trade_rows.append({"date": date, "action": "FORCED_EXIT", "ticker": ticker, "shares": shares, "price": fill, "value": value, "commission": commission})
            del positions[ticker]

        equity = cash + sum(int(position["shares"]) * last_marks.get(ticker, 0.0) for ticker, position in positions.items())
        daily_rows.append({"date": date, "equity": equity, "cash": cash, "positions_count": len(positions)})

    daily = pd.DataFrame(daily_rows)
    daily["drawdown_pct"] = (daily["equity"] / daily["equity"].cummax() - 1.0) * 100.0
    trades = pd.DataFrame(trade_rows)
    rebalances = pd.DataFrame(rebalance_rows)
    latest_ranking = ranked.loc[ranked["date"] == ranked["date"].max()].sort_values("selection_rank")
    summary: dict[str, object] = {
        "period_start": sessions[0].date().isoformat(),
        "period_end": sessions[-1].date().isoformat(),
        "years_requested": years,
        "selector_top_n": top_n,
        "holding_trading_days": holding_days,
        "initial_capital": initial_capital,
        "market_name": market_name,
        "benchmark_name": benchmark_name,
        "currency_symbol": currency_symbol,
        "commission_rate": commission_rate,
        "minimum_commission": minimum_commission,
        "data_source": data_source,
        "research_limitation": research_limitation,
        "rebalance_count": len(rebalances),
        "total_orders": len(trades),
        "buy_orders": int((trades["action"] == "BUY").sum()) if not trades.empty else 0,
        "sell_orders": int((trades["action"] == "SELL").sum()) if not trades.empty else 0,
        "total_fees": total_fees,
        "completed_positions": len(completed_returns),
        "win_rate_pct": float(np.mean(np.array(completed_returns) > 0.0) * 100.0) if completed_returns else 0.0,
        "open_positions": len(positions),
        "forced_exit_orders": int((trades["action"] == "FORCED_EXIT").sum()) if not trades.empty else 0,
        "membership_source": str(ranked["membership_source"].iloc[-1]) if not ranked.empty else "unknown",
        **_metrics(daily["equity"], initial_capital),
    }
    if benchmark is not None and not benchmark.empty:
        benchmark_frame = benchmark[["date", "close"]].copy()
        benchmark_frame["date"] = pd.to_datetime(
            benchmark_frame["date"], errors="coerce"
        ).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
        benchmark_frame["close"] = pd.to_numeric(benchmark_frame["close"], errors="coerce")
        benchmark_frame = benchmark_frame.dropna().sort_values("date").drop_duplicates("date")
        daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
        daily = pd.merge_asof(
            daily.sort_values("date"),
            benchmark_frame,
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=4),
        )
        valid_benchmark = daily["close"].dropna()
        if len(valid_benchmark) > 1:
            base_close = float(valid_benchmark.iloc[0])
            daily["benchmark_equity"] = initial_capital * daily["close"] / base_close
            daily["benchmark_drawdown_pct"] = (
                daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1.0
            ) * 100.0
            benchmark_metrics = _metrics(daily["benchmark_equity"].dropna(), initial_capital)
            summary.update(
                {
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_total_return_pct": benchmark_metrics["total_return_pct"],
                    "benchmark_cagr_pct": benchmark_metrics["cagr_pct"],
                    "benchmark_max_drawdown_pct": benchmark_metrics["max_drawdown_pct"],
                    "excess_total_return_pct": (
                        float(summary["total_return_pct"])
                        - benchmark_metrics["total_return_pct"]
                    ),
                    "excess_cagr_pct": float(summary["cagr_pct"]) - benchmark_metrics["cagr_pct"],
                }
            )
        daily = daily.drop(columns=["close"], errors="ignore")
    if "benchmark_equity" not in daily:
        daily["benchmark_equity"] = np.nan
        daily["benchmark_drawdown_pct"] = np.nan
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_equity.csv", index=False)
    trades.to_csv(output_dir / "trades.csv", index=False)
    rebalances.to_csv(output_dir / "rebalances.csv", index=False)
    latest_ranking.to_csv(output_dir / "latest_full_ranking.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    report_path = _write_report(output_dir, summary, daily, trades, rebalances)
    return SelectorHoldResult(summary, daily, trades, rebalances, latest_ranking, report_path)
