"""Plot the per-prompt E2E and TTFT by colocated start order of a single pd_interference run, from send and in engine time, to OUTPUT_DIR/<name>/plot/{from_send,engine_time}/latency_by_start_order.png.

Run with the conserve env active; the argument names the experiment, defaulting to test:
    python plot_interference_single.py
    python plot_interference_single.py test
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interference_metrics import run_properties
from plot_interference_sweep import INK, OUTPUT_DIR, SURFACE, TIMINGS, _plot_by_start_order, latencies_in_start_order


def plot_latency_by_start_order(timing, experiment_name="test"):
    """Plot the per-prompt E2E and TTFT by timing and colocated start order of the run experiment_name to its plot/<timing>/latency_by_start_order.png."""
    run_dir = OUTPUT_DIR / experiment_name
    e2e, ttft = latencies_in_start_order(run_dir / "data", timing)
    properties = run_properties(run_dir / "data" / "colocated")
    timing_label = timing.replace("_", " ")
    figure, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=SURFACE)
    _plot_by_start_order(axes[0], e2e, f"E2E {timing_label}")
    _plot_by_start_order(axes[1], ttft, f"TTFT {timing_label}")
    figure.suptitle(
        f"{experiment_name}: per-prompt latency by colocated start order at {properties['prompt_length']} input and {properties['output_length']} output tokens",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    plot_dir = run_dir / "plot" / timing
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "latency_by_start_order.png"
    figure.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    for timing in TIMINGS:
        plot_latency_by_start_order(timing, *sys.argv[1:2])
