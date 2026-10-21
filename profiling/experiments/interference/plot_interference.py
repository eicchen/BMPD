"""Measure queue time, TTFT, TPOT, E2E and GPU busy time of an interference sweep or single run, write them to metrics.csv, and plot them.

Run from this directory with the conserve env active; an argument names the experiment, read from OUTPUT_DIR/<name>/data with outputs written to OUTPUT_DIR/<name>/plot, and defaults to sweep:
    python plot_interference.py
    python plot_interference.py smoke_test
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pd_mix import OUTPUT_DIR
from metric import SUFFIX

MEASUREMENTS = ("queue", "ttft", "tpot", "e2e")
STATISTICS = ("mean", "p95")
PANELS = (
    ("ttft", "mean", "Mean TTFT (s)", 1),
    ("tpot", "mean", "Mean TPOT (ms)", 1000),
    ("tpot", "p95", "p95 TPOT (ms)", 1000),
    ("e2e", "mean", "Mean E2E (s)", 1),
)
COLORS = {"decode": "#2a78d6", "prefill": "#eb6834"}
SURFACE, INK, SECONDARY, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARK = dict(lw=2, marker="o", ms=8, mec=SURFACE, mew=2)


# Measurements (queue time, TTFT, TPOT, E2E, GPU busy time)


def token_times(log_dir):
    """When the engine emitted each request's tokens, from the step-end events in log_dir's core.jsonl."""
    emission_times = defaultdict(list)
    for line in (log_dir / "core.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "step_end":
            for request_id in event["finished_request_ids"]:  # every request with output this step, not only finished ones
                emission_times[SUFFIX.sub("", request_id)].append(datetime.fromisoformat(event["timestamp"]).timestamp())
    return emission_times


def queue(request_result):
    """Seconds from send to the request's first engine step: time queued in vLLM, and for decode requests their KV load."""
    gpu_request = request_result["gpu_requests"][0]
    return gpu_request["first_step"] - gpu_request["sent"]


def ttft(request_result):
    """Seconds from the request's first engine step to its first token, excluding time queued."""
    gpu_request = request_result["gpu_requests"][0]
    return gpu_request["first_token"] - gpu_request["first_step"]


def tpot(request_result, emission_times):
    """Seconds between consecutive tokens of the request, from the engine's step timestamps."""
    gpu_request = request_result["gpu_requests"][0]
    return list(np.diff(emission_times[f"cmpl-{request_result['id']}-0"][: gpu_request["completion_tokens"]]))


def e2e(request_result):
    """Seconds from the request's first engine step to its last token, excluding time queued."""
    gpu_request = request_result["gpu_requests"][0]
    return gpu_request["done"] - gpu_request["first_step"]


def measure(run_dir):
    """Every queue, TTFT, TPOT and E2E value in seconds of one run directory, grouped by request kind."""
    request_results = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    emission_times = token_times(run_dir / f"gpu{request_results[0]['gpu_requests'][0]['gpu']}")
    measurements = defaultdict(lambda: defaultdict(list))
    for request_result in request_results:
        kind_measurements = measurements[request_result["request_kind"]]
        kind_measurements["queue"].append(queue(request_result))
        kind_measurements["ttft"].append(ttft(request_result))
        kind_measurements["tpot"] += tpot(request_result, emission_times)
        kind_measurements["e2e"].append(e2e(request_result))
    return measurements


def busy_time(run_dir):
    """Seconds the engine spent running steps in one run directory, from its steps.jsonl."""
    return sum(step["end"] - step["start"] for steps_file in run_dir.glob("gpu*/steps.jsonl") for step in map(json.loads, steps_file.read_text().splitlines()))


def _statistics(values):
    """{"mean", "p95"} of values, NaN when there are none."""
    return {"mean": np.mean(values), "p95": np.percentile(values, 95)} if values else dict.fromkeys(STATISTICS, np.nan)


def _properties(run_dir):
    """run_dir's run.properties as a dict."""
    return dict(line.split("=", 1) for line in (run_dir / "run.properties").read_text().splitlines())


def _runs(directory):
    """(decode ratio, run kind, measurements, busy time) of the single run in directory, or of every run of the sweep in directory, by decode ratio."""
    run_dirs = [directory] if (directory / "requests.jsonl").exists() else [requests_file.parent for requests_file in directory.glob("*/*/requests.jsonl")]
    return sorted((float(_properties(run_dir)["decode_ratio"]), run_dir.name, measure(run_dir), busy_time(run_dir)) for run_dir in run_dirs)


def table(experiment_name):
    """Write the mean and p95 of queue time, TTFT, TPOT and E2E, and the run's GPU busy time, for each run and request kind of the experiment to OUTPUT_DIR/experiment_name/plot/metrics.csv."""
    plot_dir = OUTPUT_DIR / experiment_name / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with open(plot_dir / "metrics.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "decode_ratio",
                "run_kind",
                "request_kind",
                "requests",
                "gpu_busy_s",
                *(f"{measurement}_{statistic}_s" for measurement in MEASUREMENTS for statistic in STATISTICS),
            ],
        )
        for decode_ratio, run_kind, measurements, run_busy_time in _runs(OUTPUT_DIR / experiment_name / "data"):
            for request_kind, kind_measurements in measurements.items():
                statistics = {measurement: _statistics(kind_measurements[measurement]) for measurement in MEASUREMENTS}
                writer.writerow(
                    [
                        decode_ratio,
                        run_kind,
                        request_kind,
                        len(kind_measurements["ttft"]),
                        run_busy_time,
                        *(statistics[measurement][statistic] for measurement in MEASUREMENTS for statistic in STATISTICS),
                    ],
                )


def _style(axis, title, x_label):
    """Apply recessive axes, x_label, a y axis from 0 and a left-aligned title."""
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK)
    axis.set_xlabel(x_label, color=SECONDARY)
    axis.set_ylim(bottom=0)
    axis.grid(color=GRID, lw=1)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE)


def _description(properties):
    """Request count, arrivals and request lengths of a run, from its properties."""
    arrival = "all sent at once" if properties["arrival_rate"] == "None" else f"at {properties['arrival_rate']} req/s"
    return (
        f"{properties['request_count']} requests {arrival}, "
        f"decode {properties['decode_prompt_length']} → {properties['decode_output_length']} tokens from stored KV, "
        f"prefill {properties['prefill_prompt_length']} → {properties['prefill_output_length']} tokens"
    )


def plot(experiment_name):
    """Plot TTFT, TPOT (mean and p95), E2E and GPU busy time of a sweep against decode ratio, mixed runs solid and solo runs faded, to OUTPUT_DIR/experiment_name/plot/plot.png."""
    sweep_dir = OUTPUT_DIR / experiment_name / "data"
    series = defaultdict(list)
    colocated_busy, separate_busy = {}, defaultdict(float)
    for decode_ratio, run_kind, measurements, run_busy_time in _runs(sweep_dir):
        for request_kind, kind_measurements in measurements.items():
            served = "alone" if len(measurements) == 1 else "mixed"
            series[request_kind, served].append((decode_ratio, {measurement: _statistics(kind_measurements[measurement]) for measurement in MEASUREMENTS}))
        if run_kind == "mixed":
            colocated_busy[decode_ratio] = run_busy_time
        if len(measurements) == 1:
            separate_busy[decode_ratio] += run_busy_time
    decode_ratios = sorted({decode_ratio for points in series.values() for decode_ratio, _ in points})
    figure, axes = plt.subplots(1, len(PANELS) + 1, figsize=(25, 4.5), sharex=True, facecolor=SURFACE)
    for axis, (measurement, statistic, title, scale) in zip(axes, PANELS):
        for (request_kind, served), points in sorted(series.items(), reverse=True):
            point_ratios = [decode_ratio for decode_ratio, _ in points]
            point_values = [statistics[measurement][statistic] * scale for _, statistics in points]
            axis.plot(point_ratios, point_values, color=COLORS[request_kind], alpha=1 if served == "mixed" else 0.4, label=f"{request_kind} {served}", **MARK)
        _style(axis, title, "Decode ratio")
    for label, busy_by_ratio, alpha in (("mixed run busy", colocated_busy, 1), ("decode alone + prefill alone busy", separate_busy, 0.4)):
        busy_ratios = sorted(busy_by_ratio)
        axes[-1].plot(busy_ratios, [busy_by_ratio[decode_ratio] for decode_ratio in busy_ratios], color=SECONDARY, alpha=alpha, label=label, **MARK)
    _style(axes[-1], "GPU busy time (s)", "Decode ratio")
    for axis in axes:
        axis.set_xticks(decode_ratios, [f"{decode_ratio:g}" for decode_ratio in decode_ratios])
        axis.margins(x=0.15)
    legend_entries = [entry for axis in (axes[0], axes[-1]) for entry in zip(*axis.get_legend_handles_labels())]
    figure.legend(*zip(*legend_entries), loc="upper right", ncol=len(legend_entries), frameon=False, labelcolor=SECONDARY)
    figure.suptitle(f"Decode ratio sweep: {_description(_properties(next(sweep_dir.glob('*/*/run.properties')).parent))}", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    plot_dir = OUTPUT_DIR / experiment_name / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "plot.png", dpi=150)


def plot_run(experiment_name):
    """Plot a single run's request timeline (queued faded, served solid) and its engine step durations (steps with prefill work orange) against time since the first send, to OUTPUT_DIR/experiment_name/plot/plot.png."""
    run_dir = OUTPUT_DIR / experiment_name / "data"
    request_results = sorted(
        (json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()),
        key=lambda request_result: request_result["gpu_requests"][0]["sent"],
    )
    steps = [json.loads(line) for line in (run_dir / f"gpu{request_results[0]['gpu_requests'][0]['gpu']}" / "steps.jsonl").read_text().splitlines()]
    first_send = request_results[0]["gpu_requests"][0]["sent"]
    figure, (timeline_axis, step_axis) = plt.subplots(1, 2, figsize=(16, 4.5), facecolor=SURFACE)
    for send_order, request_result in enumerate(request_results):
        gpu_request = request_result["gpu_requests"][0]
        color = COLORS[request_result["request_kind"]]
        timeline_axis.plot([gpu_request["sent"] - first_send, gpu_request["first_step"] - first_send], [send_order] * 2, color=color, alpha=0.4, lw=2)
        timeline_axis.plot([gpu_request["first_step"] - first_send, gpu_request["done"] - first_send], [send_order] * 2, color=color, lw=2)
    _style(timeline_axis, "Requests in send order", "Seconds since first send")
    timeline_axis.legend(
        handles=[
            Line2D([], [], color=COLORS[request_kind], alpha=alpha, lw=2, label=f"{request_kind} {segment}")
            for request_kind in sorted({request_result["request_kind"] for request_result in request_results})
            for segment, alpha in (("queued", 0.4), ("served", 1))
        ],
        loc="upper left",
        frameon=False,
        labelcolor=SECONDARY,
    )
    for has_prefill, label in ((False, "decode-only step"), (True, "step with prefill work")):
        kind_steps = [step for step in steps if (step["ctx_tokens"] > step["ctx_reqs"]) == has_prefill]
        step_axis.scatter(
            [step["start"] - first_send for step in kind_steps],
            [(step["end"] - step["start"]) * 1000 for step in kind_steps],
            s=10,
            color=COLORS["prefill" if has_prefill else "decode"],
            alpha=0.6,
            linewidths=0,
            label=label,
        )
    _style(step_axis, "Engine step duration (ms)", "Seconds since first send")
    figure.legend(*step_axis.get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, labelcolor=SECONDARY)
    properties = _properties(run_dir)
    figure.suptitle(f"{experiment_name}, decode ratio {properties['decode_ratio']}: {_description(properties)}", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    plot_dir = OUTPUT_DIR / experiment_name / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "plot.png", dpi=150)


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    table(experiment_name)
    if (OUTPUT_DIR / experiment_name / "data" / "requests.jsonl").exists():
        plot_run(experiment_name)
    else:
        plot(experiment_name)
