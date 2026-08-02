"""Build self-contained HTML strategy replays from saved backtest results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from project_config import BENCHMARKS, PATHS


DEFAULT_REPLAY_FILE = "strategy_replay.html"
MODE_LABELS = {
    BENCHMARKS.direct_results_dir: "Direct",
    BENCHMARKS.hedged_results_dir: "Hedged",
}


@dataclass(frozen=True)
class ReplayArtifact:
    mode: str
    output_path: Path
    frame_count: int
    start_date: str
    end_date: str


def _normalize_dates(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    normalized = frame.copy()
    normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce").dt.normalize()
    normalized = normalized.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    return normalized


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _safe_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def _mode_dir(mode: str, results_root: Path) -> Path:
    if mode not in MODE_LABELS:
        valid = ", ".join(sorted(MODE_LABELS))
        raise ValueError(f"Unsupported mode `{mode}`. Expected one of: {valid}")
    return results_root / mode


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required replay input: {csv_path}")
    return pd.read_csv(csv_path)


def _build_holdings_lookup(picks_df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    holdings_df = _normalize_dates(picks_df)
    signed_column = "signed_position_weight" if "signed_position_weight" in holdings_df.columns else "position_weight"
    holdings_df[signed_column] = pd.to_numeric(holdings_df[signed_column], errors="coerce").fillna(0.0)
    holdings_df["position_weight"] = pd.to_numeric(
        holdings_df.get("position_weight", holdings_df[signed_column]),
        errors="coerce",
    ).fillna(0.0)
    holdings_df = holdings_df[holdings_df[signed_column].abs() > 1e-12].copy()
    if holdings_df.empty:
        return {}

    holdings_df["probability_up"] = pd.to_numeric(holdings_df.get("probability_up"), errors="coerce")
    holdings_df["metric_score"] = pd.to_numeric(holdings_df.get("metric_score"), errors="coerce")
    holdings_df["holding_days"] = pd.to_numeric(holdings_df.get("holding_days"), errors="coerce").fillna(0.0)
    holdings_df["ticker"] = holdings_df["ticker"].astype(str).str.upper()
    position_side = holdings_df["position_side"] if "position_side" in holdings_df.columns else pd.Series("", index=holdings_df.index)
    holdings_df["position_side"] = position_side.astype(str).str.lower()
    holdings_df["abs_weight"] = holdings_df[signed_column].abs()

    grouped: dict[str, list[dict[str, object]]] = {}
    for date_value, sample in holdings_df.groupby("date", sort=True):
        ordered = sample.sort_values(["abs_weight", "ticker"], ascending=[False, True])
        grouped[pd.Timestamp(date_value).date().isoformat()] = [
            {
                "ticker": row.ticker,
                "side": row.position_side or ("long" if getattr(row, signed_column) >= 0.0 else "short"),
                "weight": _safe_float(row.position_weight),
                "signed_weight": _safe_float(getattr(row, signed_column)),
                "probability_up": _safe_float(row.probability_up),
                "metric_score": _safe_float(row.metric_score),
                "holding_days": _safe_float(row.holding_days),
            }
            for row in ordered.itertuples(index=False)
        ]
    return grouped


def build_replay_payload(mode: str, results_root: Path | None = None) -> dict[str, object]:
    results_dir = _mode_dir(mode, Path(results_root or PATHS.output_dir))
    simulation_path = results_dir / "strategy_backtest_simulation.csv"
    picks_path = results_dir / "strategy_daily_top20.csv"
    benchmark_daily_path = results_dir / BENCHMARKS.benchmark_daily_file

    simulation_df = _normalize_dates(_load_csv(simulation_path))
    if simulation_df.empty:
        raise ValueError(f"Replay input is empty: {simulation_path}")

    holdings_lookup = _build_holdings_lookup(_load_csv(picks_path))

    benchmark_lookup: dict[str, dict[str, float | None]] = {}
    if benchmark_daily_path.exists():
        benchmark_df = _normalize_dates(_load_csv(benchmark_daily_path))
        benchmark_df["strategy_portfolio_value"] = pd.to_numeric(
            benchmark_df.get("strategy_portfolio_value"),
            errors="coerce",
        )
        benchmark_df["benchmark_portfolio_value"] = pd.to_numeric(
            benchmark_df.get("benchmark_portfolio_value"),
            errors="coerce",
        )
        benchmark_df["benchmark_daily_return"] = pd.to_numeric(
            benchmark_df.get("benchmark_daily_return"),
            errors="coerce",
        )
        benchmark_lookup = {
            row.date.date().isoformat(): {
                "strategy_portfolio_value": None if pd.isna(row.strategy_portfolio_value) else float(row.strategy_portfolio_value),
                "benchmark_portfolio_value": None if pd.isna(row.benchmark_portfolio_value) else float(row.benchmark_portfolio_value),
                "benchmark_daily_return": None if pd.isna(row.benchmark_daily_return) else float(row.benchmark_daily_return),
            }
            for row in benchmark_df.itertuples(index=False)
        }

    numeric_columns = [
        "portfolio_value",
        "cumulative_return",
        "holdings_count",
        "buys",
        "sells",
        "avg_holding_days",
        "turnover_fraction",
        "stock_turnover_fraction",
        "long_positions_count",
        "short_positions_count",
        "long_stock_exposure",
        "short_stock_exposure",
        "net_stock_exposure",
        "gross_stock_exposure",
        "benchmark_short_weight",
        "total_margin_requirement_fraction",
        "free_equity_fraction",
        "margin_call_flag",
    ]
    for column in numeric_columns:
        if column in simulation_df.columns:
            simulation_df[column] = pd.to_numeric(simulation_df[column], errors="coerce")

    frames: list[dict[str, object]] = []
    for row in simulation_df.itertuples(index=False):
        date_key = row.date.date().isoformat()
        benchmark_row = benchmark_lookup.get(date_key, {})
        benchmark_weight = _safe_float(getattr(row, "benchmark_short_weight", 0.0))
        gross_stock_exposure = _safe_float(getattr(row, "gross_stock_exposure", 0.0))
        effective_gross_exposure = gross_stock_exposure + abs(benchmark_weight)
        frames.append(
            {
                "date": date_key,
                "portfolio_value": _safe_float(getattr(row, "portfolio_value", 0.0)),
                "cumulative_return": _safe_float(getattr(row, "cumulative_return", 0.0)),
                "holdings_count": _safe_int(getattr(row, "holdings_count", 0)),
                "long_positions_count": _safe_int(getattr(row, "long_positions_count", 0)),
                "short_positions_count": _safe_int(getattr(row, "short_positions_count", 0)),
                "buys": _safe_int(getattr(row, "buys", 0)),
                "sells": _safe_int(getattr(row, "sells", 0)),
                "trade_count": _safe_int(getattr(row, "buys", 0)) + _safe_int(getattr(row, "sells", 0)),
                "avg_holding_days": _safe_float(getattr(row, "avg_holding_days", 0.0)),
                "turnover_fraction": _safe_float(getattr(row, "turnover_fraction", 0.0)),
                "stock_turnover_fraction": _safe_float(getattr(row, "stock_turnover_fraction", 0.0)),
                "long_stock_exposure": _safe_float(getattr(row, "long_stock_exposure", 0.0)),
                "short_stock_exposure": _safe_float(getattr(row, "short_stock_exposure", 0.0)),
                "net_stock_exposure": _safe_float(getattr(row, "net_stock_exposure", 0.0)),
                "gross_stock_exposure": gross_stock_exposure,
                "benchmark_short_weight": benchmark_weight,
                "effective_gross_exposure": effective_gross_exposure,
                "capital_committed_fraction": 1.0 - _safe_float(getattr(row, "free_equity_fraction", 0.0)),
                "total_margin_requirement_fraction": _safe_float(
                    getattr(row, "total_margin_requirement_fraction", 0.0)
                ),
                "free_equity_fraction": _safe_float(getattr(row, "free_equity_fraction", 0.0)),
                "margin_call_flag": _safe_int(getattr(row, "margin_call_flag", 0)),
                "strategy_portfolio_value": benchmark_row.get("strategy_portfolio_value"),
                "benchmark_portfolio_value": benchmark_row.get("benchmark_portfolio_value"),
                "benchmark_daily_return": benchmark_row.get("benchmark_daily_return"),
                "holdings": holdings_lookup.get(date_key, []),
            }
        )

    return {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "title": f"{MODE_LABELS[mode]} Strategy Replay",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "frame_count": len(frames),
        "start_date": frames[0]["date"],
        "end_date": frames[-1]["date"],
        "frames": frames,
    }


def _payload_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


def _payloads_json(payloads: dict[str, dict[str, object]]) -> str:
    return json.dumps(payloads, separators=(",", ":")).replace("</", "<\\/")


def build_multi_replay_html(payloads: list[dict[str, object]]) -> str:
    if not payloads:
        raise ValueError("At least one replay payload is required.")

    title = "Strategy Replay" if len(payloads) > 1 else str(payloads[0]["title"])
    payload_map = {str(payload["mode"]): payload for payload in payloads}
    payloads_json = _payloads_json(payload_map)
    mode_order_json = json.dumps([str(payload["mode"]) for payload in payloads], separators=(",", ":"))
    show_mode_tabs = len(payloads) > 1
    mode_tabs_markup = ""
    if show_mode_tabs:
        mode_tabs_markup = """
      <div class="control-row mode-switch-row">
        <span class="timeline-label">Mode</span>
        <div class="mode-tabs" id="modeTabs"></div>
      </div>"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --panel: rgba(255, 251, 245, 0.86);
      --panel-strong: #fffdf8;
      --ink: #1f2933;
      --muted: #5f6c77;
      --accent: #0f766e;
      --shadow: 0 18px 44px rgba(33, 38, 45, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 26%),
        linear-gradient(180deg, #f8f4eb 0%, #f0eadf 100%);
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 18px;
    }}
    .hero, .controls, .panel {{
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px;
      align-items: end;
      flex-wrap: wrap;
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(246,240,230,0.88));
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.8rem, 3vw, 2.8rem);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .hero-meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(110px, 1fr));
      gap: 10px;
      min-width: min(100%, 420px);
    }}
    .meta-card, .stat-card {{
      padding: 12px 14px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid rgba(36, 48, 58, 0.08);
    }}
    .meta-card span, .stat-card span {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .controls {{
      display: grid;
      gap: 14px;
      padding: 18px 20px;
    }}
    .control-row {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .mode-switch-row {{
      justify-content: space-between;
    }}
    .mode-tabs {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .mode-tab {{
      border: 1px solid rgba(15, 118, 110, 0.14);
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      color: var(--ink);
      background: rgba(255, 253, 248, 0.88);
      transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
    }}
    .mode-tab[data-active="true"] {{
      color: white;
      background: linear-gradient(135deg, #0f766e, #155e75);
      border-color: transparent;
    }}
    button, select {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
    }}
    button {{
      color: white;
      background: linear-gradient(135deg, #0f766e, #155e75);
    }}
    select {{ background: #fffdf8; color: var(--ink); }}
    input[type="range"] {{ width: min(100%, 880px); accent-color: var(--accent); }}
    .timeline-label {{ font-weight: 700; min-width: 120px; }}
    .layout {{
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 2fr) minmax(320px, 0.95fr);
    }}
    .charts {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .chart-panel {{ padding: 8px 10px 10px; overflow: hidden; }}
    .chart-panel canvas {{
      display: block;
      width: 100%;
      height: 248px;
    }}
    .side-column {{ display: grid; gap: 18px; align-content: start; }}
    .snapshot, .holdings {{ padding: 18px 20px 20px; }}
    .snapshot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 14px;
    }}
    .section-head h2 {{ margin: 0; font-size: 1rem; letter-spacing: -0.03em; }}
    .section-head span {{ color: var(--muted); font-size: 0.88rem; }}
    .holdings-list {{ display: grid; gap: 10px; }}
    .holding-row {{
      display: grid;
      grid-template-columns: 68px minmax(0, 1fr) 76px;
      align-items: center;
      gap: 10px;
      padding: 8px 0;
    }}
    .ticker {{ font-weight: 700; letter-spacing: 0.03em; }}
    .bar-wrap {{
      position: relative;
      height: 14px;
      border-radius: 999px;
      background: rgba(36, 48, 58, 0.08);
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      border-radius: 999px;
    }}
    .bar.long {{ background: linear-gradient(90deg, #0f766e, #14b8a6); }}
    .bar.short {{ background: linear-gradient(90deg, #dc2626, #fb7185); }}
    .weight {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .footer-note {{
      color: var(--muted);
      font-size: 0.86rem;
      padding: 0 4px 4px;
    }}
    @media (max-width: 1120px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 760px) {{
      .charts {{ grid-template-columns: 1fr; }}
      .snapshot-grid, .hero-meta {{ grid-template-columns: 1fr; }}
      .holding-row {{ grid-template-columns: 58px minmax(0, 1fr) 68px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero panel">
      <div>
        <h1 id="title">{title}</h1>
        <p>Autoplay replay of equity, exposure, holdings mix, leverage, and trading activity from the saved results.</p>
      </div>
      <div class="hero-meta">
        <div class="meta-card"><span>Date</span><strong id="currentDate">-</strong></div>
        <div class="meta-card"><span>Frame</span><strong id="frameLabel">-</strong></div>
        <div class="meta-card"><span>Mode</span><strong id="modeLabel">-</strong></div>
      </div>
    </section>
    <section class="controls panel">
{mode_tabs_markup}
      <div class="control-row">
        <button id="playButton" type="button">Play</button>
        <button id="stepBackButton" type="button">Back</button>
        <button id="stepForwardButton" type="button">Forward</button>
        <label>
          <span class="timeline-label">Speed</span>
          <select id="speedSelect">
            <option value="0.35">0.35x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1.0x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2.0x</option>
            <option value="4">4.0x</option>
          </select>
        </label>
      </div>
      <div class="control-row">
        <span class="timeline-label">Timeline</span>
        <input id="timeline" type="range" min="0" max="0" value="0" step="1" />
      </div>
    </section>
    <div class="layout">
      <section class="charts">
        <article class="panel chart-panel"><canvas id="equityChart"></canvas></article>
        <article class="panel chart-panel"><canvas id="exposureChart"></canvas></article>
        <article class="panel chart-panel"><canvas id="positionsChart"></canvas></article>
        <article class="panel chart-panel"><canvas id="tradesChart"></canvas></article>
      </section>
      <aside class="side-column">
        <section class="panel snapshot">
          <div class="section-head">
            <h2>Current Snapshot</h2>
            <span id="snapshotReturn">-</span>
          </div>
          <div class="snapshot-grid" id="snapshotGrid"></div>
        </section>
        <section class="panel holdings">
          <div class="section-head">
            <h2>Current Holdings</h2>
            <span id="holdingsCountLabel">-</span>
          </div>
          <div class="holdings-list" id="holdingsList"></div>
        </section>
      </aside>
    </div>
    <div class="footer-note">Open this file locally in a browser. Use space to play or pause, and left/right arrows to step through dates.</div>
  </div>
  <script>
    const payloads = {payloads_json};
    const modeOrder = {mode_order_json};
    const currentIndexByMode = Object.fromEntries(modeOrder.map((mode) => [mode, 0]));
    let activeMode = modeOrder[0];
    let payload = payloads[activeMode];
    let frames = payload.frames;
    let currentIndex = 0;
    let isPlaying = false;
    let speedMultiplier = 1;
    let lastTickMs = 0;
    let chartRenderers = [];

    const frameMsBase = 150;
    const timeline = document.getElementById("timeline");
    const playButton = document.getElementById("playButton");
    const speedSelect = document.getElementById("speedSelect");
    const frameLabel = document.getElementById("frameLabel");
    const currentDate = document.getElementById("currentDate");
    const modeLabel = document.getElementById("modeLabel");
    const modeTabs = document.getElementById("modeTabs");
    const snapshotReturn = document.getElementById("snapshotReturn");
    const snapshotGrid = document.getElementById("snapshotGrid");
    const holdingsList = document.getElementById("holdingsList");
    const holdingsCountLabel = document.getElementById("holdingsCountLabel");

    function currency(value) {{
      return new Intl.NumberFormat("en-AU", {{
        style: "currency",
        currency: "AUD",
        maximumFractionDigits: 0
      }}).format(value ?? 0);
    }}

    function percent(value, digits = 1) {{
      return `${{((value ?? 0) * 100).toFixed(digits)}}%`;
    }}

    function integer(value) {{
      return new Intl.NumberFormat("en-AU").format(value ?? 0);
    }}

    function prepareCanvas(canvas) {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(220, Math.floor(rect.height));
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {{
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
      }}
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, width, height }};
    }}

    function findRange(frameSeries, seriesDefs) {{
      const values = [];
      seriesDefs.forEach((series) => {{
        frameSeries.forEach((frame) => {{
          const value = frame[series.key];
          if (value !== null && value !== undefined && Number.isFinite(value)) {{
            values.push(value);
          }}
        }});
      }});
      let min = Math.min(...values);
      let max = Math.max(...values);
      if (!Number.isFinite(min) || !Number.isFinite(max)) {{
        min = 0;
        max = 1;
      }}
      if (min === max) {{
        max = min + 1;
      }}
      const pad = (max - min) * 0.12;
      return {{ min: min - pad, max: max + pad }};
    }}

    function drawChart(canvas, definition, frameIndex) {{
      const {{ ctx, width, height }} = prepareCanvas(canvas);
      const padding = {{ top: 28, right: 10, bottom: 30, left: 48 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const {{ min, max }} = findRange(frames, definition.series);
      const xAt = (index) => padding.left + (frames.length <= 1 ? 0 : (index / Math.max(1, frames.length - 1)) * plotWidth);
      const yAt = (value) => padding.top + (1 - ((value - min) / (max - min))) * plotHeight;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(36, 48, 58, 0.11)";
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i += 1) {{
        const value = min + ((max - min) * i) / 4;
        const y = yAt(value);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.fillStyle = "#5f6c77";
        ctx.font = "12px Segoe UI";
        ctx.textAlign = "right";
        ctx.fillText(definition.tickFormatter(value), padding.left - 8, y + 4);
      }}

      ctx.strokeStyle = "rgba(36, 48, 58, 0.18)";
      ctx.beginPath();
      ctx.moveTo(padding.left, padding.top);
      ctx.lineTo(padding.left, height - padding.bottom);
      ctx.lineTo(width - padding.right, height - padding.bottom);
      ctx.stroke();

      ctx.fillStyle = "#1f2933";
      ctx.font = "700 15px Segoe UI";
      ctx.textAlign = "left";
      ctx.fillText(definition.title, padding.left, 20);

      let legendX = padding.left;
      definition.series.forEach((series) => {{
        const currentValue = frames[frameIndex][series.key];
        ctx.fillStyle = series.color;
        ctx.fillRect(legendX, 24, 10, 10);
        ctx.fillStyle = "#5f6c77";
        ctx.font = "12px Segoe UI";
        ctx.fillText(`${{series.label}} ${{series.valueFormatter(currentValue)}}`, legendX + 16, 34);
        legendX += 168;
      }});

      definition.series.forEach((series) => {{
        ctx.beginPath();
        let hasPoint = false;
        frames.forEach((frame, index) => {{
          const value = frame[series.key];
          if (!Number.isFinite(value)) {{
            return;
          }}
          const x = xAt(index);
          const y = yAt(value);
          if (!hasPoint) {{
            ctx.moveTo(x, y);
            hasPoint = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }});
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2.2;
        ctx.stroke();

        const currentValue = frames[frameIndex][series.key];
        if (Number.isFinite(currentValue)) {{
          ctx.beginPath();
          ctx.arc(xAt(frameIndex), yAt(currentValue), 4.2, 0, Math.PI * 2);
          ctx.fillStyle = series.color;
          ctx.fill();
        }}
      }});

      const cursorX = xAt(frameIndex);
      ctx.strokeStyle = "rgba(15, 118, 110, 0.38)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, padding.top);
      ctx.lineTo(cursorX, height - padding.bottom);
      ctx.stroke();

      ctx.fillStyle = "#5f6c77";
      ctx.font = "12px Segoe UI";
      ctx.textAlign = "left";
      ctx.fillText(payload.start_date, padding.left, height - 10);
      ctx.textAlign = "right";
      ctx.fillText(payload.end_date, width - padding.right, height - 10);
    }}

    function buildChartRenderer(canvasId, definition) {{
      const canvas = document.getElementById(canvasId);
      const render = () => drawChart(canvas, definition, currentIndex);
      const observer = new ResizeObserver(render);
      observer.observe(canvas);
      return render;
    }}

    function renderModeTabs() {{
      if (!modeTabs) {{
        return;
      }}
      modeTabs.innerHTML = modeOrder.map((mode) => {{
        const modePayload = payloads[mode];
        const isActive = mode === activeMode ? "true" : "false";
        return `<button class="mode-tab" type="button" data-mode="${{mode}}" data-active="${{isActive}}">${{modePayload.mode_label}}</button>`;
      }}).join("");
      modeTabs.querySelectorAll("[data-mode]").forEach((button) => {{
        button.addEventListener("click", () => switchMode(button.dataset.mode));
      }});
    }}

    function syncActivePayload() {{
      payload = payloads[activeMode];
      frames = payload.frames;
      currentIndex = clampIndex(currentIndexByMode[activeMode] ?? 0);
    }}

    function renderSnapshot(frame) {{
      document.getElementById("title").textContent = payload.title;
      currentDate.textContent = frame.date;
      frameLabel.textContent = `${{integer(currentIndex + 1)}} / ${{integer(frames.length)}}`;
      modeLabel.textContent = payload.mode_label;
      snapshotReturn.textContent = `Cumulative return: ${{percent(frame.cumulative_return, 2)}}`;
      const items = [
        ["Portfolio", currency(frame.portfolio_value)],
        ["Benchmark", frame.benchmark_portfolio_value == null ? "-" : currency(frame.benchmark_portfolio_value)],
        ["Gross Exposure", percent(frame.effective_gross_exposure, 1)],
        ["Net Exposure", percent(frame.net_stock_exposure, 1)],
        ["Margin Used", percent(frame.total_margin_requirement_fraction, 1)],
        ["Free Equity", percent(frame.free_equity_fraction, 1)],
        ["Trades Today", integer(frame.trade_count)],
        ["Turnover", percent(frame.turnover_fraction, 1)],
        ["Holdings", integer(frame.holdings_count)],
        ["Long vs Short", `${{integer(frame.long_positions_count)}} / ${{integer(frame.short_positions_count)}}`],
        ["Avg Hold Days", frame.avg_holding_days.toFixed(1)],
        ["Margin Call", frame.margin_call_flag ? "Yes" : "No"]
      ];
      snapshotGrid.innerHTML = items.map(([label, value]) => `
        <div class="stat-card">
          <span>${{label}}</span>
          <strong>${{value}}</strong>
        </div>
      `).join("");
    }}

    function renderHoldings(frame) {{
      const holdings = frame.holdings || [];
      holdingsCountLabel.textContent = `${{integer(holdings.length)}} live positions`;
      if (!holdings.length) {{
        holdingsList.innerHTML = `<div class="footer-note">No active holdings were saved for this date.</div>`;
        return;
      }}
      const maxWeight = Math.max(...holdings.map((holding) => Math.abs(holding.signed_weight)), 0.0001);
      holdingsList.innerHTML = holdings.slice(0, 12).map((holding) => {{
        const width = Math.max(6, (Math.abs(holding.signed_weight) / maxWeight) * 100);
        const side = holding.signed_weight >= 0 ? "long" : "short";
        return `
          <div class="holding-row" title="${{holding.ticker}} | ${{side}} | weight ${{percent(Math.abs(holding.signed_weight), 2)}} | score ${{holding.metric_score.toFixed(3)}}">
            <div class="ticker">${{holding.ticker}}</div>
            <div class="bar-wrap"><div class="bar ${{side}}" style="width:${{width}}%"></div></div>
            <div class="weight">${{percent(holding.signed_weight, 2)}}</div>
          </div>
        `;
      }}).join("");
    }}

    function clampIndex(index) {{
      return Math.max(0, Math.min(frames.length - 1, index));
    }}

    function renderAll() {{
      if (!frames.length) {{
        return;
      }}
      const frame = frames[currentIndex];
      timeline.value = String(currentIndex);
      timeline.max = String(Math.max(0, frames.length - 1));
      renderSnapshot(frame);
      renderHoldings(frame);
      chartRenderers.forEach((render) => render());
      renderModeTabs();
    }}

    function setIndex(nextIndex) {{
      currentIndex = clampIndex(nextIndex);
      currentIndexByMode[activeMode] = currentIndex;
      renderAll();
    }}

    function switchMode(nextMode) {{
      if (!payloads[nextMode] || nextMode === activeMode) {{
        return;
      }}
      activeMode = nextMode;
      syncActivePayload();
      togglePlay(false);
      renderAll();
    }}

    function togglePlay(forceState = null) {{
      isPlaying = forceState == null ? !isPlaying : Boolean(forceState);
      playButton.textContent = isPlaying ? "Pause" : "Play";
      if (isPlaying) {{
        lastTickMs = 0;
        window.requestAnimationFrame(animationStep);
      }}
    }}

    function animationStep(timestampMs) {{
      if (!isPlaying) {{
        return;
      }}
      if (!lastTickMs) {{
        lastTickMs = timestampMs;
      }}
      const elapsed = timestampMs - lastTickMs;
      if (elapsed >= frameMsBase / speedMultiplier) {{
        lastTickMs = timestampMs;
        setIndex(currentIndex >= frames.length - 1 ? 0 : currentIndex + 1);
      }}
      window.requestAnimationFrame(animationStep);
    }}

    function installEvents() {{
      timeline.addEventListener("input", () => {{
        togglePlay(false);
        setIndex(Number(timeline.value));
      }});
      playButton.addEventListener("click", () => togglePlay());
      document.getElementById("stepBackButton").addEventListener("click", () => {{
        togglePlay(false);
        setIndex(currentIndex - 1);
      }});
      document.getElementById("stepForwardButton").addEventListener("click", () => {{
        togglePlay(false);
        setIndex(currentIndex + 1);
      }});
      speedSelect.addEventListener("change", () => {{
        speedMultiplier = Number(speedSelect.value) || 1;
      }});
      window.addEventListener("keydown", (event) => {{
        if (event.code === "Space") {{
          event.preventDefault();
          togglePlay();
        }} else if (event.code === "ArrowRight") {{
          event.preventDefault();
          togglePlay(false);
          setIndex(currentIndex + 1);
        }} else if (event.code === "ArrowLeft") {{
          event.preventDefault();
          togglePlay(false);
          setIndex(currentIndex - 1);
        }}
      }});
    }}

    function initCharts() {{
      chartRenderers = [
        buildChartRenderer("equityChart", {{
          title: "Equity Curve",
          tickFormatter: (value) => currency(value),
          series: [
            {{ key: "portfolio_value", label: "Strategy", color: "#0f766e", valueFormatter: (value) => currency(value) }},
            {{ key: "benchmark_portfolio_value", label: "Benchmark", color: "#d97706", valueFormatter: (value) => value == null ? "-" : currency(value) }}
          ]
        }}),
        buildChartRenderer("exposureChart", {{
          title: "Exposure And Margin",
          tickFormatter: (value) => percent(value, 0),
          series: [
            {{ key: "effective_gross_exposure", label: "Gross", color: "#0f766e", valueFormatter: (value) => percent(value, 1) }},
            {{ key: "net_stock_exposure", label: "Net", color: "#155e75", valueFormatter: (value) => percent(value, 1) }},
            {{ key: "total_margin_requirement_fraction", label: "Margin", color: "#d97706", valueFormatter: (value) => percent(value, 1) }},
            {{ key: "free_equity_fraction", label: "Free", color: "#7c3aed", valueFormatter: (value) => percent(value, 1) }}
          ]
        }}),
        buildChartRenderer("positionsChart", {{
          title: "Holdings Mix",
          tickFormatter: (value) => integer(Math.max(0, Math.round(value))),
          series: [
            {{ key: "holdings_count", label: "Total", color: "#0f766e", valueFormatter: (value) => integer(value) }},
            {{ key: "long_positions_count", label: "Long", color: "#166534", valueFormatter: (value) => integer(value) }},
            {{ key: "short_positions_count", label: "Short", color: "#b91c1c", valueFormatter: (value) => integer(value) }}
          ]
        }}),
        buildChartRenderer("tradesChart", {{
          title: "Trades Per Day",
          tickFormatter: (value) => integer(Math.max(0, Math.round(value))),
          series: [
            {{ key: "trade_count", label: "Total", color: "#0f766e", valueFormatter: (value) => integer(value) }},
            {{ key: "buys", label: "Buys", color: "#166534", valueFormatter: (value) => integer(value) }},
            {{ key: "sells", label: "Sells", color: "#b91c1c", valueFormatter: (value) => integer(value) }}
          ]
        }})
      ];
    }}

    syncActivePayload();
    installEvents();
    initCharts();
    renderAll();
  </script>
</body>
</html>
"""


def build_replay_html(payload: dict[str, object]) -> str:
    return build_multi_replay_html([payload])


def write_replay_html(
    mode: str,
    *,
    results_root: Path | None = None,
    output_path: Path | None = None,
    modes: list[str] | tuple[str, ...] | None = None,
) -> ReplayArtifact:
    resolved_results_root = Path(results_root or PATHS.output_dir)
    if mode == "all":
        selected_modes = sorted(modes) if modes is not None else sorted(MODE_LABELS)
        invalid_modes = [item_mode for item_mode in selected_modes if item_mode not in MODE_LABELS]
        if invalid_modes:
            valid = ", ".join(sorted(MODE_LABELS))
            invalid = ", ".join(sorted(invalid_modes))
            raise ValueError(f"Unsupported mode(s) `{invalid}`. Expected only: {valid}")
        if not selected_modes:
            raise ValueError("At least one replay mode is required when mode=`all`.")

        payloads = [
            build_replay_payload(mode=item_mode, results_root=resolved_results_root)
            for item_mode in selected_modes
        ]
        final_path = Path(output_path) if output_path is not None else resolved_results_root / DEFAULT_REPLAY_FILE
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(build_multi_replay_html(payloads), encoding="utf-8")
        return ReplayArtifact(
            mode=mode,
            output_path=final_path,
            frame_count=sum(int(payload["frame_count"]) for payload in payloads),
            start_date=min(str(payload["start_date"]) for payload in payloads),
            end_date=max(str(payload["end_date"]) for payload in payloads),
        )

    payload = build_replay_payload(mode=mode, results_root=resolved_results_root)
    results_dir = _mode_dir(mode, resolved_results_root)
    final_path = Path(output_path) if output_path is not None else results_dir / DEFAULT_REPLAY_FILE
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(build_replay_html(payload), encoding="utf-8")
    return ReplayArtifact(
        mode=mode,
        output_path=final_path,
        frame_count=int(payload["frame_count"]),
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]),
    )
