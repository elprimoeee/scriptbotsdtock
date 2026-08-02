#!/usr/bin/env python3
"""Render autoplay HTML replays from saved strategy result folders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.replay import MODE_LABELS, write_replay_html
from project_config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML replay for saved strategy results.",
    )
    parser.add_argument(
        "--mode",
        choices=[*sorted(MODE_LABELS), "all"],
        default="all",
        help="Which saved results mode to render. Default writes a combined multi-mode replay.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PATHS.output_dir,
        help="Root directory containing the mode result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = write_replay_html(
        mode=args.mode,
        results_root=args.results_root,
        output_path=args.output,
    )
    print(
        f"[{artifact.mode}] replay saved to {artifact.output_path} "
        f"({artifact.frame_count} frames, {artifact.start_date} -> {artifact.end_date})"
    )


if __name__ == "__main__":
    main()
