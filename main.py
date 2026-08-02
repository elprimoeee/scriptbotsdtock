from __future__ import annotations

from pathlib import Path

from models.metric import execute_metric_model
from models.rsi_momentum_bot import execute_rsi_bot_model
from models.replay import write_replay_html
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for environments without tqdm installed
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


SELECTED_MODEL = "rsi_bot"
SELECTED_TESTS = [True, True]


MODEL_RUNNERS = {
    "metric_model": execute_metric_model,
    "rsi_bot": execute_rsi_bot_model,
}


def main() -> None:
    if len(SELECTED_TESTS) != 2:
        raise SystemExit("SELECTED_TESTS must contain exactly two booleans: [direct, hedged_short_benchmark].")

    direct_benchmark_test, hedged_short_benchmark_test = [bool(flag) for flag in SELECTED_TESTS]
    runner = MODEL_RUNNERS.get(SELECTED_MODEL)
    if runner is None:
        available_models = ", ".join(sorted(MODEL_RUNNERS))
        raise SystemExit(f"Unknown model `{SELECTED_MODEL}`. Available models: {available_models}")

    print(f"Selected model: {SELECTED_MODEL}")
    print(
        "Selected tests: "
        f"direct={direct_benchmark_test}, hedged_short_benchmark={hedged_short_benchmark_test}"
    )
    run_result = runner(
        run_direct_benchmark_test=direct_benchmark_test,
        run_hedged_short_benchmark_test=hedged_short_benchmark_test,
    )
    selected_modes = sorted(run_result.saved_paths)
    artifact = write_replay_html(
        mode="all",
        results_root=run_result.output_dir,
        modes=selected_modes,
    )
    print(
        f"[{artifact.mode}] replay saved to {artifact.output_path} "
        f"({artifact.frame_count} frames, {artifact.start_date} -> {artifact.end_date})"
    )

    for mode in tqdm(
        selected_modes,
        total=len(selected_modes),
        desc="Replay cleanup",
        dynamic_ncols=True,
    ):
        stale_replay_path = Path(run_result.output_dir) / mode / "strategy_replay.html"
        if stale_replay_path.exists():
            stale_replay_path.unlink()
            print(f"[{mode}] removed stale per-mode replay at {stale_replay_path}")


if __name__ == "__main__":
    main()
