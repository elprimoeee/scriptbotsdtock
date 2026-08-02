#!/usr/bin/env python3
"""
Build and maintain a free ASX daily dataset.

This is the one monolith script for training-data generation.

Data sources:
- ASX listed companies directory CSV (free, official)
- Yahoo Finance via yfinance (free daily OHLCV + limited fundamentals)
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ASX_DIRECTORY_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
AU_TZ = ZoneInfo("Australia/Sydney")
NON_EQUITY_GICS = {"Not Applic", "Class Pend"}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    inputs: tuple[str, ...]
    compute: object


class MetricRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    @property
    def definitions(self) -> dict[str, MetricDefinition]:
        return dict(self._definitions)

    def register(self, name: str, description: str, inputs: tuple[str, ...], compute) -> None:
        if name in self._definitions:
            raise ValueError(f"Metric `{name}` is already registered.")
        self._definitions[name] = MetricDefinition(
            name=name,
            description=description,
            inputs=tuple(inputs),
            compute=compute,
        )

    def apply(self, pd, dataset):
        extended = dataset.copy()
        for definition in self._definitions.values():
            missing = [column for column in definition.inputs if column not in extended.columns]
            if missing:
                raise KeyError(
                    f"Metric `{definition.name}` is missing required columns: {', '.join(missing)}"
                )
            result = definition.compute(extended.copy())
            if not isinstance(result, pd.Series):
                raise TypeError(
                    f"Metric `{definition.name}` must return a pandas Series, got {type(result)!r}."
                )
            if len(result) != len(extended):
                raise ValueError(
                    f"Metric `{definition.name}` returned {len(result)} rows for a dataset with "
                    f"{len(extended)} rows."
                )
            extended[definition.name] = result.to_numpy()
        return extended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 7-year daily ASX data, compile the dataset, and auto-load custom metrics."
    )
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument(
        "--years",
        type=int,
        default=7,
        help="History window size in years (default: 7).",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional max tickers for testing.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Optional comma-separated ASX codes to run (e.g. BHP,CBA,WBC).",
    )
    parser.add_argument(
        "--extra-tickers-file",
        default=None,
        help=(
            "Optional CSV with at least a `ticker` column to append (for delisted/manual "
            "coverage). Optional columns: company_name,gics_industry_group."
        ),
    )
    parser.add_argument(
        "--include-non-equity",
        action="store_true",
        help="Include rows where ASX GICS group is Not Applic/Class Pend.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore existing raw files and re-pull the full history window.",
    )
    parser.add_argument(
        "--refresh-meta",
        action="store_true",
        help="Force refresh of cached metadata JSON files.",
    )
    parser.add_argument(
        "--meta-max-age-days",
        type=int,
        default=30,
        help="Days before metadata cache is considered stale.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.08,
        help="Sleep between ticker requests to reduce throttling.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip network downloads and only compile from existing raw files.",
    )
    parser.add_argument(
        "--skip-custom-metrics",
        action="store_true",
        help="Skip loading custom metric files from `metric_modules/`.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress logs.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    ts = datetime.now(AU_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {message}")


def import_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import requests
        import yfinance as yf
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            "Missing dependency: "
            f"{missing}. Install with `pip install -r requirements.txt`."
        ) from exc
    return np, pd, requests, yf


def subtract_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # Handles Feb 29.
        return d.replace(month=2, day=28, year=d.year - years)


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "raw_universe": base_dir / "raw" / "universe",
        "raw_prices": base_dir / "raw" / "prices",
        "raw_meta": base_dir / "raw" / "meta",
        "processed": base_dir / "processed",
        "logs": base_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def clean_text_cell(value):
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def normalize_scalar(value):
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def fetch_asx_universe(pd, requests, output_csv: Path):
    response = requests.get(ASX_DIRECTORY_URL, timeout=30)
    response.raise_for_status()
    text = response.text
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("company name,asx code,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find header row in ASX CSV response.")
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text), dtype=str)
    df = df.rename(
        columns={
            "Company name": "company_name",
            "ASX code": "ticker",
            "GICS industry group": "gics_industry_group",
        }
    )
    for col in ["company_name", "ticker", "gics_industry_group"]:
        if col in df.columns:
            df[col] = df[col].map(clean_text_cell)
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].str.upper()
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    return df


def filter_universe(df, include_non_equity: bool):
    if include_non_equity:
        return df.copy()
    gics = df["gics_industry_group"].fillna("").str.strip()
    return df.loc[~gics.isin(NON_EQUITY_GICS)].copy()


def parse_ticker_list(tickers_arg: str | None) -> set[str] | None:
    if not tickers_arg:
        return None
    values = [t.strip().upper() for t in tickers_arg.split(",")]
    values = [t for t in values if t]
    return set(values) if values else None


def apply_extra_tickers(pd, universe_df, extra_tickers_file: str | None):
    if not extra_tickers_file:
        return universe_df
    path = Path(extra_tickers_file)
    if not path.exists():
        raise SystemExit(f"Extra tickers file not found: {path}")
    extra = pd.read_csv(path, dtype=str)
    if "ticker" not in extra.columns:
        raise SystemExit("Extra tickers CSV must include a `ticker` column.")
    for col in ["ticker", "company_name", "gics_industry_group"]:
        if col not in extra.columns:
            extra[col] = None
    extra = extra[["ticker", "company_name", "gics_industry_group"]].copy()
    extra["ticker"] = extra["ticker"].map(clean_text_cell).str.upper()
    extra["company_name"] = extra["company_name"].map(clean_text_cell)
    extra["gics_industry_group"] = extra["gics_industry_group"].map(clean_text_cell)
    extra = extra.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"])
    merged = pd.concat([universe_df, extra], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ticker"], keep="first")
    return merged


def ticker_symbol(ticker: str) -> str:
    return f"{ticker}.AX"


def ticker_storage_stem(ticker: str) -> str:
    upper = ticker.upper()
    if upper in WINDOWS_RESERVED_NAMES:
        return f"__{upper}"
    return upper


def ticker_from_storage_stem(stem: str) -> str:
    upper = stem.upper()
    if upper.startswith("__"):
        maybe_reserved = upper[2:]
        if maybe_reserved in WINDOWS_RESERVED_NAMES:
            return maybe_reserved
    return upper


def safe_read_csv(pd, path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def format_history_df(pd, raw_df, ticker_code: str):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.reset_index().copy()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    rename_map = {
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    df = df.rename(columns=rename_map)
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()
    if "dividends" not in df.columns:
        df["dividends"] = 0.0
    if "stock_splits" not in df.columns:
        df["stock_splits"] = 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    df["dividends"] = df["dividends"].fillna(0.0)
    df["stock_splits"] = df["stock_splits"].fillna(0.0)
    df = df.dropna(subset=["date", "open", "high", "low", "close", "adj_close"]).copy()
    df["ticker"] = ticker_code
    keep_cols = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    df = df[keep_cols].sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def extract_meta_from_yf(ticker_obj, fallback_company_name: str | None, gics_group: str | None):
    info = {}
    fast_info = {}
    try:
        fast_info = dict(ticker_obj.fast_info)
    except Exception:
        fast_info = {}
    try:
        info = ticker_obj.get_info() or {}
    except Exception:
        info = {}

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or fallback_company_name
    )
    metadata = {
        "company_name": normalize_scalar(company_name),
        "sector": normalize_scalar(info.get("sector") or gics_group),
        "industry": normalize_scalar(info.get("industry") or gics_group),
        "quote_type": normalize_scalar(info.get("quoteType")),
        "currency": normalize_scalar(info.get("currency") or fast_info.get("currency")),
        "country": normalize_scalar(info.get("country")),
        "shares_outstanding": normalize_scalar(
            fast_info.get("shares") or info.get("sharesOutstanding")
        ),
        "float_shares": normalize_scalar(info.get("floatShares")),
        "market_cap_hint": normalize_scalar(
            fast_info.get("marketCap") or info.get("marketCap")
        ),
        "enterprise_value": normalize_scalar(info.get("enterpriseValue")),
        "trailing_pe": normalize_scalar(info.get("trailingPE")),
        "forward_pe": normalize_scalar(info.get("forwardPE")),
        "price_to_book": normalize_scalar(info.get("priceToBook")),
        "price_to_sales": normalize_scalar(info.get("priceToSalesTrailing12Months")),
        "ev_to_ebitda": normalize_scalar(info.get("enterpriseToEbitda")),
        "eps_trailing": normalize_scalar(info.get("trailingEps")),
        "eps_forward": normalize_scalar(info.get("forwardEps")),
        "dividend_rate": normalize_scalar(info.get("dividendRate")),
        "dividend_yield": normalize_scalar(info.get("dividendYield")),
        "payout_ratio": normalize_scalar(info.get("payoutRatio")),
        "ex_dividend_date": normalize_scalar(info.get("exDividendDate")),
        "fetched_at": datetime.now(AU_TZ).isoformat(),
    }
    return metadata


def metadata_is_stale(meta_path: Path, max_age_days: int) -> bool:
    if not meta_path.exists():
        return True
    modified = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=AU_TZ)
    return datetime.now(AU_TZ) - modified > timedelta(days=max_age_days)


def load_metadata(meta_path: Path):
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_metadata(meta_path: Path, payload: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def cap_bucket(market_cap):
    if market_cap is None or (isinstance(market_cap, float) and math.isnan(market_cap)):
        return None
    if market_cap >= 200_000_000_000:
        return "mega"
    if market_cap >= 10_000_000_000:
        return "large"
    if market_cap >= 2_000_000_000:
        return "mid"
    if market_cap >= 300_000_000:
        return "small"
    return "micro"


def compute_price_metrics(pd, price_df):
    df = price_df.sort_values("date").copy()
    # Use the raw close so ex-dividend adjustments do not leak into point-in-time features.
    price = df["close"]
    returns = price.pct_change()

    df["dollar_volume"] = df["close"] * df["volume"]
    df["vwap_proxy"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["avg_volume_30d"] = df["volume"].rolling(30).mean()
    df["avg_volume_90d"] = df["volume"].rolling(90).mean()

    df["daily_return"] = returns
    df["weekly_return"] = price.pct_change(5)
    df["monthly_return"] = price.pct_change(21)
    df["quarterly_return"] = price.pct_change(63)
    df["yearly_return"] = price.pct_change(252)
    first_price = price.iloc[0] if not price.empty else None
    df["cumulative_return"] = (price / first_price) - 1 if first_price else None

    df["momentum_1m"] = price.pct_change(21)
    df["momentum_3m"] = price.pct_change(63)
    df["momentum_6m"] = price.pct_change(126)
    df["momentum_12m_ex_1m"] = (price.shift(21) / price.shift(252)) - 1
    ma50 = price.rolling(50).mean()
    ma200 = price.rolling(200).mean()
    df["price_vs_50d_ma"] = (price / ma50) - 1
    df["price_vs_200d_ma"] = (price / ma200) - 1

    df["daily_volatility"] = returns.abs()
    df["volatility_30d"] = returns.rolling(30).std()
    df["volatility_90d"] = returns.rolling(90).std()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14d"] = tr.rolling(14).mean()
    down_returns = returns.where(returns < 0, 0.0)
    df["downside_volatility_90d"] = down_returns.rolling(90).std()

    rolling_peak = price.cummax()
    drawdown = (price / rolling_peak) - 1
    df["max_drawdown"] = drawdown.cummin()
    return df


def load_metric_modules(metric_modules_dir: Path) -> list[Path]:
    if not metric_modules_dir.exists():
        return []
    return sorted(
        path
        for path in metric_modules_dir.glob("*.py")
        if path.name not in {"__init__.py"}
    )


def apply_custom_metric_modules(pd, dataset, metric_modules_dir: Path, verbose: bool = False):
    module_paths = load_metric_modules(metric_modules_dir)
    if not module_paths:
        return dataset, []

    registry = MetricRegistry()
    loaded = []
    for module_path in module_paths:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load metric module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        register = getattr(module, "register_metrics", None)
        if register is None:
            raise AttributeError(
                f"Metric module `{module_path.name}` must define `register_metrics(registry)`."
            )
        register(registry)
        loaded.append(module_path.stem)
        if verbose:
            log(f"Loaded custom metric module: {module_path.name}")

    if not registry.definitions:
        return dataset, loaded

    return registry.apply(pd, dataset), loaded


def compile_dataset(
    pd,
    paths: dict[str, Path],
    universe_df,
    metric_modules_dir: Path | None = None,
    verbose: bool = False,
):
    price_dir = paths["raw_prices"]
    meta_dir = paths["raw_meta"]
    files = sorted(price_dir.glob("*.csv"))
    if not files:
        raise RuntimeError("No raw price files found to compile.")

    universe_lookup = universe_df.set_index("ticker").to_dict("index")
    all_frames = []
    for path in files:
        ticker = ticker_from_storage_stem(path.stem)
        price_df = pd.read_csv(path)
        if price_df.empty:
            continue
        price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce", utc=True)
        price_df = price_df.dropna(subset=["date"]).copy()
        if price_df.empty:
            continue
        if "ticker" in price_df.columns and not price_df["ticker"].empty:
            ticker = str(price_df["ticker"].iloc[0]).upper()
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
        ]
        for col in numeric_cols:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        if "volume" in price_df.columns:
            price_df["volume"] = price_df["volume"].fillna(0.0)
        for col in ["dividends", "stock_splits"]:
            if col in price_df.columns:
                price_df[col] = price_df[col].fillna(0.0)
        price_df["date"] = price_df["date"].dt.date
        price_df = price_df.dropna(subset=["date", "open", "high", "low", "close", "adj_close"]).copy()
        price_df = compute_price_metrics(pd, price_df)

        u = universe_lookup.get(ticker, {})
        meta = load_metadata(meta_dir / f"{ticker_storage_stem(ticker)}.json")
        shares_outstanding = meta.get("shares_outstanding")
        float_shares = meta.get("float_shares")
        enterprise_value = meta.get("enterprise_value")

        price_df["company_name"] = meta.get("company_name") or u.get("company_name")
        price_df["gics_industry_group"] = u.get("gics_industry_group")
        price_df["sector"] = meta.get("sector") or u.get("gics_industry_group")
        price_df["industry"] = meta.get("industry") or u.get("gics_industry_group")

        price_df["shares_outstanding"] = shares_outstanding
        price_df["float_shares"] = float_shares
        price_df["enterprise_value"] = enterprise_value

        price_df["market_cap"] = (
            price_df["close"] * price_df["shares_outstanding"]
            if shares_outstanding
            else meta.get("market_cap_hint")
        )
        price_df["free_float_market_cap"] = (
            price_df["close"] * price_df["float_shares"] if float_shares else None
        )
        price_df["turnover_ratio"] = (
            price_df["volume"] / price_df["shares_outstanding"]
            if shares_outstanding
            else None
        )
        price_df["cap_category"] = price_df["market_cap"].map(cap_bucket)

        price_df["trailing_pe"] = meta.get("trailing_pe")
        price_df["forward_pe"] = meta.get("forward_pe")
        price_df["price_to_book"] = meta.get("price_to_book")
        price_df["price_to_sales"] = meta.get("price_to_sales")
        price_df["ev_to_ebitda"] = meta.get("ev_to_ebitda")
        price_df["eps_trailing"] = meta.get("eps_trailing")
        price_df["eps_forward"] = meta.get("eps_forward")
        price_df["dividend_rate"] = meta.get("dividend_rate")
        price_df["dividend_yield"] = meta.get("dividend_yield")
        price_df["payout_ratio"] = meta.get("payout_ratio")
        price_df["ex_dividend_date"] = meta.get("ex_dividend_date")
        all_frames.append(price_df)

    if not all_frames:
        raise RuntimeError("No compiled frames were generated.")

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.date
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    dataset["market_cap_rank"] = (
        dataset.groupby("date")["market_cap"].rank(ascending=False, method="dense")
    )

    loaded_modules = []
    if metric_modules_dir is not None:
        dataset, loaded_modules = apply_custom_metric_modules(
            pd=pd,
            dataset=dataset,
            metric_modules_dir=metric_modules_dir,
            verbose=verbose,
        )

    out_csv = paths["processed"] / "asx_daily_dataset.csv"
    dataset.to_csv(out_csv, index=False)
    return out_csv, len(dataset), loaded_modules


def main():
    args = parse_args()
    np, pd, requests, yf = import_dependencies()
    _ = np  # Imported for dependency validation and numeric coercion.

    project_root = Path(__file__).resolve().parents[1]
    metric_modules_dir = project_root / "metric_modules"
    base_dir = Path(args.data_dir).resolve()
    paths = ensure_dirs(base_dir)

    today = datetime.now(AU_TZ).date()
    start_date = subtract_years(today, args.years)
    end_date_exclusive = today + timedelta(days=1)

    if args.verbose:
        log(f"Base directory: {base_dir}")
        log(f"Window: {start_date} to {today} (Australia/Sydney)")

    universe_path = paths["raw_universe"] / "asx_listed_companies.csv"
    if args.skip_download and universe_path.exists():
        universe_df = pd.read_csv(universe_path, dtype=str)
    else:
        log("Fetching ASX listed companies directory...")
        universe_df = fetch_asx_universe(pd, requests, universe_path)
        log(f"Universe rows downloaded: {len(universe_df):,}")
    universe_df = apply_extra_tickers(pd, universe_df, args.extra_tickers_file)

    filtered_universe = filter_universe(universe_df, args.include_non_equity)
    requested_tickers = parse_ticker_list(args.tickers)
    if requested_tickers:
        filtered_universe = filtered_universe[
            filtered_universe["ticker"].isin(requested_tickers)
        ].copy()
    if args.max_tickers:
        filtered_universe = filtered_universe.head(args.max_tickers).copy()

    tickers = filtered_universe["ticker"].tolist()
    if not tickers:
        raise SystemExit("No tickers selected after filtering.")

    log(f"Tickers selected: {len(tickers):,}")

    failures = []
    download_rows = 0
    if not args.skip_download:
        for i, ticker in enumerate(tickers, start=1):
            symbol = ticker_symbol(ticker)
            file_stem = ticker_storage_stem(ticker)
            price_path = paths["raw_prices"] / f"{file_stem}.csv"
            meta_path = paths["raw_meta"] / f"{file_stem}.json"
            existing = safe_read_csv(pd, price_path)
            fetch_start = start_date
            if (
                existing is not None
                and not existing.empty
                and not args.full_refresh
                and "date" in existing.columns
            ):
                existing_dates = pd.to_datetime(existing["date"], errors="coerce").dropna()
                if not existing_dates.empty:
                    last_date = existing_dates.max().date()
                    fetch_start = max(start_date, last_date - timedelta(days=3))

            if args.verbose or i % 50 == 0 or i == 1:
                log(f"[{i}/{len(tickers)}] {ticker} ({symbol}) from {fetch_start}")

            ticker_obj = yf.Ticker(symbol)
            try:
                raw_hist = ticker_obj.history(
                    start=fetch_start.isoformat(),
                    end=end_date_exclusive.isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                )
            except Exception as exc:
                failures.append({"ticker": ticker, "error": str(exc)})
                continue

            new_df = format_history_df(pd, raw_hist, ticker)
            if existing is not None and not existing.empty and not args.full_refresh:
                existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.date
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.dropna(subset=["date"]).sort_values("date")
                combined = combined.drop_duplicates(subset=["date"], keep="last")
            else:
                combined = new_df

            if combined.empty:
                failures.append({"ticker": ticker, "error": "No history returned"})
                continue

            combined = combined[combined["date"] >= start_date].copy()
            combined.to_csv(price_path, index=False)
            download_rows += len(new_df)

            if args.refresh_meta or metadata_is_stale(meta_path, args.meta_max_age_days):
                fallback_name = filtered_universe.loc[
                    filtered_universe["ticker"] == ticker, "company_name"
                ].iloc[0]
                fallback_gics = filtered_universe.loc[
                    filtered_universe["ticker"] == ticker, "gics_industry_group"
                ].iloc[0]
                try:
                    meta_payload = extract_meta_from_yf(
                        ticker_obj, fallback_name, fallback_gics
                    )
                    save_metadata(meta_path, meta_payload)
                except Exception as exc:
                    failures.append({"ticker": ticker, "error": f"Meta fetch failed: {exc}"})

            time.sleep(max(args.pause_seconds, 0.0))

        log(f"Raw ticker downloads finished. New rows fetched: {download_rows:,}")

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = paths["logs"] / "download_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        log(f"Failures logged: {len(failures)} -> {fail_path}")

    log("Compiling combined dataset...")
    out_csv, rows, loaded_modules = compile_dataset(
        pd=pd,
        paths=paths,
        universe_df=filtered_universe,
        metric_modules_dir=None if args.skip_custom_metrics else metric_modules_dir,
        verbose=args.verbose,
    )
    log(f"Compiled rows: {rows:,}")
    if loaded_modules:
        log(f"Custom metric modules applied: {', '.join(loaded_modules)}")
    elif not args.skip_custom_metrics:
        log("No custom metric modules found.")
    log(f"Dataset written: {out_csv}")

    log("Done.")


if __name__ == "__main__":
    main()
