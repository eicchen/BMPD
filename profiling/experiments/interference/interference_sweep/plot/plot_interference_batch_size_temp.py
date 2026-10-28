"""Temporary: plot a single pd_interference run's per-prompt E2E and TTFT by colocated start order, colocated and disaggregated side by side, over the mean batch size and KV cache usage of the colocated request and of the disaggregated prefill and decode, from send and in engine time, to OUTPUT_DIR/<name>/plot/{from_send,engine_time}/batch_size_by_start_order.png, and its colocated, prefill and decode requests over time above their run's batch size and KV cache usage per engine step to plot/request_timeline.png.

Run with the conserve env active; the argument names the experiment, defaulting to test:
    python plot_interference_batch_size_temp.py
    python plot_interference_batch_size_temp.py test
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interference_metrics import RUN_NAMES, run_properties
from plot_interference_sweep import COLORS, INK, MARK, MUTED, OUTPUT_DIR, SECONDARY, SURFACE, TIMINGS, _in_start_order, _style, latencies_in_start_order

BATCH_SIZE_COLOR = "#1baf7a"
KV_USAGE_COLOR = "#4a3aa7"
STEP_QUANTITIES = {
    "batch_size": lambda step: step["ctx_reqs"] + step["gen_reqs"],
    "kv_usage": lambda step: step["kv_usage"],
}
RUN_KIND_COLORS = {"colocated": COLORS["colocated"], "disaggregated prefill": COLORS["disaggregated"], "disaggregated decode": COLORS["disaggregated"]}


# Plot data


def _steps(run_dir, quantity):
    """Start, end and quantity, batch_size (context plus generation requests) or kv_usage (fraction of KV cache blocks held), of each engine step of run_dir, one row per step."""
    steps = map(json.loads, next(run_dir.glob("gpu*/steps.jsonl")).read_text().splitlines())
    return np.array([(step["start"], step["end"], STEP_QUANTITIES[quantity](step)) for step in steps])


def _window(request, metric, timing):
    """Start and end of request's metric, e2e or ttft, by timing."""
    return request.sent if timing == "from_send" else request.first_step, request.done if metric == "e2e" else request.first_token


def _time_average(steps, window_start, window_end):
    """Time-averaged quantity of steps from window_start to window_end, idle time counting as 0."""
    starts, ends, step_values = steps.T
    return (np.clip(np.minimum(ends, window_end) - np.maximum(starts, window_start), 0, None) * step_values).sum() / (window_end - window_start)


def _time_averages(steps, run_requests, metric, timing):
    """Time-averaged quantity of steps over the window of each of run_requests' metric, e2e or ttft, by timing."""
    return np.array([_time_average(steps, *_window(request, metric, timing)) for request in run_requests])


def means_in_start_order(data_dir, quantity, metric, timing):
    """Per-prompt mean quantity, batch_size or kv_usage, by run kind and phase over the window its metric, e2e or ttft, measures by timing, in colocated start order; the disaggregated prefill's window is its E2E."""
    colocated_steps, prefill_steps, decode_steps = (_steps(data_dir / run_name, quantity) for run_name in RUN_NAMES)
    prompts = _in_start_order(data_dir)
    return {
        "colocated": {"colocated": _time_averages(colocated_steps, [prompt.colocated for prompt in prompts], metric, timing)},
        "disaggregated": {
            "prefill": _time_averages(prefill_steps, [prompt.disaggregated.prefill for prompt in prompts], "e2e", timing),
            "decode": _time_averages(decode_steps, [prompt.disaggregated.decode for prompt in prompts], metric, timing),
        },
    }


def _run_kinds(data_dir):
    """Each run kind's requests of data_dir in colocated start order, with the engine steps' batch size and KV cache usage of its run."""
    prompts = _in_start_order(data_dir)
    run_kind_requests = {
        "colocated": [prompt.colocated for prompt in prompts],
        "disaggregated prefill": [prompt.disaggregated.prefill for prompt in prompts],
        "disaggregated decode": [prompt.disaggregated.decode for prompt in prompts],
    }
    return {
        run_kind: (run_requests, _steps(data_dir / run_name, "batch_size"), _steps(data_dir / run_name, "kv_usage"))
        for (run_kind, run_requests), run_name in zip(run_kind_requests.items(), RUN_NAMES)
    }


# Plots


def _area_style(phase, color):
    """fill_between style of phase's area in color, hatched for the disaggregated prefill."""
    return dict(facecolor="none", edgecolor=color, hatch="//", lw=0) if phase == "prefill" else dict(color=color, alpha=0.4, lw=0)


def _shade_means(axis, means_by_phase, color):
    """Shade each phase's means, in colocated start order, as an area on axis in color."""
    for phase, means in means_by_phase.items():
        axis.fill_between(np.arange(1, len(means) + 1), means, **_area_style(phase, color))


def _mean_area_legend():
    """Legend handles and labels of the mean batch size and KV cache usage areas, filled for the colocated request and decode and hatched for the prefill, with their overlap."""
    batch_area, kv_area = Patch(**_area_style("decode", BATCH_SIZE_COLOR)), Patch(**_area_style("decode", KV_USAGE_COLOR))
    return (
        [batch_area, kv_area, (batch_area, kv_area), Patch(**_area_style("prefill", BATCH_SIZE_COLOR)), Patch(**_area_style("prefill", KV_USAGE_COLOR))],
        ["mean batch size", "mean KV cache usage", "overlap", "prefill mean batch size", "prefill mean KV cache usage"],
    )


def _style_right_axis(axis, y_label, color):
    """Label the right-hand axis with y_label, mute its ticks and mark its spine in color."""
    axis.set_ylabel(y_label, color=SECONDARY)
    axis.tick_params(colors=MUTED, length=0)
    axis.spines[:].set_visible(False)
    axis.spines.right.set(visible=True, color=color, lw=2)


def _kv_usage_axis(axis, y_label):
    """Right-hand twin of axis for KV cache usage from 0 to 100%, labelled y_label."""
    kv_axis = axis.twinx()
    kv_axis.set_ylim(0, 1)
    kv_axis.set_yticks(np.linspace(0, 1, 5))
    kv_axis.yaxis.set_major_formatter(PercentFormatter(1))
    _style_right_axis(kv_axis, y_label, KV_USAGE_COLOR)
    return kv_axis


def _plot_with_means(axis, latencies, batch_sizes_by_phase, kv_usages_by_phase, color, request_count, title):
    """Plot latencies in seconds, already in colocated start order, against it on axis, over each phase's mean batch sizes and KV cache usages shaded on right-hand axes from 0 to request_count and 0 to 100%."""
    start_order = np.arange(1, len(latencies) + 1)
    batch_axis = axis.twinx()
    _shade_means(batch_axis, batch_sizes_by_phase, BATCH_SIZE_COLOR)
    batch_axis.set_ylim(0, request_count)
    batch_axis.set_yticks(np.linspace(0, request_count, 5))
    _style_right_axis(batch_axis, "Mean batch size (requests)", BATCH_SIZE_COLOR)
    kv_axis = _kv_usage_axis(axis, "Mean KV cache usage")
    kv_axis.spines.right.set_position(("axes", 1.1))
    _shade_means(kv_axis, kv_usages_by_phase, KV_USAGE_COLOR)
    axis.set_zorder(kv_axis.get_zorder() + 1)
    axis.patch.set_visible(False)
    axis.plot(start_order, latencies, color=color, **MARK)
    _style(axis, title, "Prompt/turn")
    axis.tick_params(labelleft=True)
    axis.set_ylabel("Latency (s)", color=SECONDARY)


def plot_batch_size_by_start_order(timing, experiment_name="test"):
    """Plot the per-prompt E2E (top) and TTFT (bottom) by timing and colocated start order of the run experiment_name, colocated and disaggregated side by side, over their mean batch sizes and KV cache usages, to its plot/<timing>/batch_size_by_start_order.png."""
    run_dir = OUTPUT_DIR / experiment_name
    properties = run_properties(run_dir / "data" / "colocated")
    timing_label = timing.replace("_", " ")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=SURFACE, sharey="row")
    for row, (metric, latencies) in enumerate(zip(("e2e", "ttft"), latencies_in_start_order(run_dir / "data", timing))):
        batch_sizes = means_in_start_order(run_dir / "data", "batch_size", metric, timing)
        kv_usages = means_in_start_order(run_dir / "data", "kv_usage", metric, timing)
        for column, (run_kind, color) in enumerate(COLORS.items()):
            _plot_with_means(
                axes[row, column],
                latencies[run_kind],
                batch_sizes[run_kind],
                kv_usages[run_kind],
                color,
                int(properties["request_count"]),
                f"{run_kind.capitalize()} {metric.upper()} {timing_label}",
            )
    figure.legend(*_mean_area_legend(), loc="lower center", ncol=5, frameon=False, labelcolor=SECONDARY)
    figure.suptitle(
        f"{experiment_name}: per-prompt latency, mean batch size and mean KV cache usage by prompt/turn order at {properties['prompt_length']} input and {properties['output_length']} output tokens",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    plot_dir = run_dir / "plot" / timing
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "batch_size_by_start_order.png"
    figure.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


def _plot_request_timeline(axis, run_requests, run_start, color, title):
    """Plot each of run_requests, in colocated start order, as a bar from its first step to its first token and a lighter one on to its done, in seconds since run_start, on axis."""
    start_order = np.arange(1, len(run_requests) + 1)
    first_steps, first_tokens, dones = (np.array([getattr(request, field) for request in run_requests]) - run_start for field in ("first_step", "first_token", "done"))
    axis.barh(start_order, first_tokens - first_steps, left=first_steps, height=0.7, color=color, label="first step to first token")
    axis.barh(start_order, dones - first_tokens, left=first_tokens, height=0.7, color=color, alpha=0.35, label="first token to done")
    _style(axis, "", "")
    axis.tick_params(labelleft=True)
    axis.set_title(title, loc="left", color=INK, pad=24)
    axis.set_ylabel("Prompt/turn order", color=SECONDARY)


def _shade_steps(axis, steps, run_start, color):
    """Shade the quantity of each of steps, in seconds since run_start, as an area on axis in color, returning the area."""
    starts, ends, step_values = steps.T
    return axis.fill_between(np.column_stack((starts, ends)).ravel() - run_start, np.repeat(step_values, 2), color=color, alpha=0.4, lw=0)


def _plot_steps_over_time(axis, batch_size_steps, kv_usage_steps, run_start, request_count):
    """Plot the batch size and KV cache usage of each engine step, in seconds since run_start, as areas on axis from 0 to request_count and on a right-hand axis from 0 to 100%, returning both areas."""
    batch_area = _shade_steps(axis, batch_size_steps, run_start, BATCH_SIZE_COLOR)
    _style(axis, "", "Seconds since the run's first send")
    axis.tick_params(labelleft=True)
    axis.set_ylim(0, request_count)
    axis.set_yticks(np.linspace(0, request_count, 5))
    axis.set_ylabel("Batch size (requests)", color=SECONDARY)
    kv_area = _shade_steps(_kv_usage_axis(axis, "KV cache usage"), kv_usage_steps, run_start, KV_USAGE_COLOR)
    return batch_area, kv_area


def plot_request_timeline(experiment_name="test"):
    """Plot the colocated, disaggregated prefill and disaggregated decode requests of the run experiment_name side by side, each as bars over time above its run's batch size and KV cache usage per engine step, with its first finish marked, to its plot/request_timeline.png."""
    run_dir = OUTPUT_DIR / experiment_name
    properties = run_properties(run_dir / "data" / "colocated")
    figure, axes = plt.subplots(2, len(RUN_KIND_COLORS), figsize=(24, 10), facecolor=SURFACE, sharex="col", sharey="row", height_ratios=(3, 1))
    for column, (run_kind, (run_requests, batch_size_steps, kv_usage_steps)) in enumerate(_run_kinds(run_dir / "data").items()):
        run_start = min(request.sent for request in run_requests)
        _plot_request_timeline(axes[0, column], run_requests, run_start, RUN_KIND_COLORS[run_kind], run_kind.capitalize())
        batch_area, kv_area = _plot_steps_over_time(axes[1, column], batch_size_steps, kv_usage_steps, run_start, int(properties["request_count"]))
        for axis in axes[:, column]:
            axis.axvline(min(request.done for request in run_requests) - run_start, color=MUTED, ls="--", lw=1, label="first finish")
        axes[0, column].legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, labelcolor=SECONDARY)
    figure.legend(
        handles=[batch_area, kv_area, (batch_area, kv_area)],
        labels=["batch size", "KV cache usage", "overlap"],
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=SECONDARY,
    )
    figure.suptitle(
        f"{experiment_name}: requests over time, batch size and KV cache usage per engine step at {properties['prompt_length']} input and {properties['output_length']} output tokens",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    plot_path = run_dir / "plot" / "request_timeline.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    for timing in TIMINGS:
        plot_batch_size_by_start_order(timing, *sys.argv[1:2])
    plot_request_timeline(*sys.argv[1:2])
