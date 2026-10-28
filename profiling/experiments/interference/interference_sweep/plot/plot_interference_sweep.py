"""Plot heatmaps of the mean and p95 interference latency per prompt (colocated minus disaggregated E2E), with its percentage slowdown, of a pd_interference sweep over input and output length, each from send and in engine time, to OUTPUT_DIR/<name>/<max_num_seqs>/plot/{from_send,engine_time}/interference_heatmap.png, and the per-prompt latency by colocated start order of one input and output length against the rest of the sweep to plot/{from_send,engine_time}/interference_by_start_order_<input>_<output>.png.

Run from this directory with the conserve env active; arguments name the experiment, its max_num_seqs and the input and output length of the start-order plot, defaulting to length_sweep1, None, 4096 and 512:
    python plot_interference_sweep.py
    python plot_interference_sweep.py length_sweep1 None 4096 512
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pd_interference import OUTPUT_DIR
from interference_metrics import STATISTICS, load_prompts, run_properties

PANELS = (("mean", "Mean per prompt"), ("p95", "p95"))
TIMINGS = ("from_send", "engine_time")
COLORS = {"colocated": "#2a78d6", "disaggregated": "#eb6834"}
LENGTH_RAMP = LinearSegmentedColormap.from_list("length_ramp", ("#86b6ef", "#0d366b"))
SURFACE, INK, SECONDARY, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MARK = dict(lw=2, marker="o", ms=8, mec=SURFACE, mew=2)


# Plot data


def _sweep_lengths(sweep_dir):
    """Input and output lengths of the sweep in sweep_dir, ascending."""
    input_lengths = sorted(int(path.name) for path in sweep_dir.iterdir() if path.name.isdigit())
    output_lengths = sorted(int(path.name) for path in (sweep_dir / str(input_lengths[0])).iterdir())
    return input_lengths, output_lengths


def interference_grids(sweep_dir, timing):
    """Input lengths, output lengths, and each PANELS statistic of the per-prompt colocated minus disaggregated E2E by timing in seconds and in percent of disaggregated, indexed by output then input length."""
    input_lengths, output_lengths = _sweep_lengths(sweep_dir)
    grids = {statistic: np.zeros((len(output_lengths), len(input_lengths))) for statistic, _ in PANELS}
    slowdowns = {statistic: np.zeros((len(output_lengths), len(input_lengths))) for statistic, _ in PANELS}
    for input_index, input_length in enumerate(input_lengths):
        for output_index, output_length in enumerate(output_lengths):
            prompts = load_prompts(sweep_dir / str(input_length) / str(output_length) / "data")
            prompt_interference = np.array([getattr(prompt, f"interference_{timing}") for prompt in prompts])
            disaggregated_e2e = np.array([getattr(prompt.disaggregated, f"e2e_{timing}") for prompt in prompts])
            for statistic, _ in PANELS:
                grids[statistic][output_index, input_index] = STATISTICS[statistic](prompt_interference)
                slowdowns[statistic][output_index, input_index] = STATISTICS[statistic](prompt_interference / disaggregated_e2e * 100)
    return input_lengths, output_lengths, grids, slowdowns


def _in_start_order(data_dir):
    """data_dir's prompts sorted by when their colocated requests first ran on the engine."""
    return sorted(load_prompts(data_dir), key=lambda prompt: prompt.colocated.first_step)


def _interference_in_start_order(data_dir, timing):
    """Per-prompt interference by timing of data_dir, in colocated start order."""
    return np.array([getattr(prompt, f"interference_{timing}") for prompt in _in_start_order(data_dir)])


def latencies_in_start_order(data_dir, timing):
    """Per-prompt E2E and TTFT by timing and run kind of data_dir, in colocated start order."""
    prompts = _in_start_order(data_dir)
    return (
        {run_kind: np.array([getattr(getattr(prompt, run_kind), f"e2e_{timing}") for prompt in prompts]) for run_kind in ("colocated", "disaggregated")},
        {run_kind: np.array([getattr(getattr(prompt, run_kind), f"ttft_{timing}") for prompt in prompts]) for run_kind in ("colocated", "disaggregated")},
    )


def start_order_series(sweep_dir, input_length, output_length, timing):
    """Per-prompt E2E and TTFT by timing and run kind of the input_length and output_length cell, and per-prompt E2E interference by timing of every cell sharing its output length and of every cell sharing its input length, keyed by the swept length, each in its cell's colocated start order."""
    input_lengths, output_lengths = _sweep_lengths(sweep_dir)
    return (
        *latencies_in_start_order(sweep_dir / str(input_length) / str(output_length) / "data", timing),
        {length: _interference_in_start_order(sweep_dir / str(length) / str(output_length) / "data", timing) for length in input_lengths},
        {length: _interference_in_start_order(sweep_dir / str(input_length) / str(length) / "data", timing) for length in output_lengths},
    )


# ---------------


# Plots


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


def plot_interference_heatmap(timing, experiment_name="length_sweep1", max_num_seqs="None"):
    """Plot the PANELS heatmaps by timing of the sweep at max_num_seqs side by side to its plot/<timing>/interference_heatmap.png."""
    sweep_dir = OUTPUT_DIR / experiment_name / max_num_seqs
    input_lengths, output_lengths, grids, slowdowns = interference_grids(sweep_dir, timing)
    properties = run_properties(sweep_dir / str(input_lengths[0]) / str(output_lengths[0]) / "data" / "colocated")
    arrival = "all sent at once" if properties["arrival_rate"] in ("0", "None") else f"at {properties['arrival_rate']} req/s"
    figure, axes = plt.subplots(1, len(PANELS), figsize=(16, 5.5), facecolor=SURFACE)
    for axis, (statistic, title) in zip(axes, PANELS):
        grid = grids[statistic]
        norm = SymLogNorm(0.1, vmin=grid.min(), vmax=grid.max())
        image = axis.imshow(grid, cmap="plasma", norm=norm, origin="lower", aspect="auto")
        for (output_index, input_index), value in np.ndenumerate(grid):
            axis.text(input_index, output_index, f"{value:.2f}\n{slowdowns[statistic][output_index, input_index]:+.1f}%", ha="center", va="center", color="white" if norm(value) < 0.6 else INK)
        axis.set_xticks(range(len(input_lengths)), input_lengths)
        axis.set_yticks(range(len(output_lengths)), output_lengths)
        axis.set_xlabel("Input tokens", color=SECONDARY)
        axis.set_ylabel("Output tokens", color=SECONDARY)
        axis.set_title(title, loc="left", color=INK)
        axis.tick_params(colors=MUTED, length=0)
        axis.spines[:].set_visible(False)
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Interference latency (s)", color=SECONDARY)
        colorbar.ax.tick_params(colors=MUTED)
        colorbar.outline.set_visible(False)
    figure.suptitle(
        f"{experiment_name}: colocated minus disaggregated E2E {timing.replace('_', ' ')} per prompt, {properties['request_count']} prompts per cell {arrival}, max_num_seqs {max_num_seqs}",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    plot_dir = sweep_dir / "plot" / timing
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / "interference_heatmap.png", dpi=150)
    print(f"Wrote {plot_dir / 'interference_heatmap.png'}")


def _plot_by_start_order(axis, latencies, title):
    """Plot the colocated and disaggregated latencies in seconds, already in colocated start order, against it on axis."""
    for run_kind, color in COLORS.items():
        axis.plot(np.arange(1, len(latencies[run_kind]) + 1), latencies[run_kind], color=color, label=run_kind, **MARK)
    _style(axis, title, "Prompt/turn order")
    axis.set_ylabel(f"{title} (s)", color=SECONDARY)
    axis.legend(frameon=False, labelcolor=SECONDARY)


def _plot_interference_by_start_order(axis, interference_by_length, highlighted, title, length_name):
    """Plot each swept length's per-prompt E2E interference in interference_by_length, already in its cell's colocated start order, against it on axis, with the highlighted one on top."""
    for (length, prompt_interference), color in zip(interference_by_length.items(), LENGTH_RAMP(np.linspace(0, 1, len(interference_by_length)))):
        line_style = {**MARK, "lw": 3, "color": INK, "zorder": 3} if length == highlighted else {"lw": 1.5, "color": color}
        axis.plot(np.arange(1, len(prompt_interference) + 1), prompt_interference, label=length, **line_style)
    _style(axis, title, "Turn Count")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.autoscale(axis="y")
    axis.set_ylabel("E2E interference (s)", color=SECONDARY)
    axis.legend(frameon=False, labelcolor=SECONDARY, title=length_name).get_title().set_color(SECONDARY)


def plot_by_start_order(timing, experiment_name="length_sweep1", max_num_seqs="None", input_length="8192", output_length="512"):
    """Plot the per-prompt E2E and TTFT by timing and colocated start order of the input_length and output_length cell of the sweep at max_num_seqs, and the per-prompt E2E interference by timing and colocated start order of every cell sharing its output or input length with it highlighted, to its plot/<timing>/interference_by_start_order_<input_length>_<output_length>.png."""
    sweep_dir = OUTPUT_DIR / experiment_name / max_num_seqs
    input_length, output_length = int(input_length), int(output_length)
    e2e, ttft, interference_by_input_length, interference_by_output_length = start_order_series(sweep_dir, input_length, output_length, timing)
    timing_label = timing.replace("_", " ")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=SURFACE)
    _plot_by_start_order(axes[0, 0], e2e, f"E2E {timing_label}")
    _plot_by_start_order(axes[0, 1], ttft, f"TTFT {timing_label}")
    _plot_interference_by_start_order(axes[1, 0], interference_by_input_length, input_length, f"E2E interference {timing_label} at {output_length} output tokens", "Input tokens")
    _plot_interference_by_start_order(axes[1, 1], interference_by_output_length, output_length, f"E2E interference {timing_label} at {input_length} input tokens", "Output tokens")
    figure.suptitle(
        f"{experiment_name}: per-prompt latency by prompt/turn order at {input_length} input and {output_length} output tokens, max_num_seqs {max_num_seqs}",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    plot_dir = sweep_dir / "plot" / timing
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"interference_by_start_order_{input_length}_{output_length}.png"
    figure.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    for timing in TIMINGS:
        plot_interference_heatmap(timing, *sys.argv[1:3])
        # plot_by_start_order(timing, *sys.argv[1:5])
        for input_len in [512, 1024, 2048, 4096, 8192, 16384]:
            plot_by_start_order(timing, *sys.argv[1:3], input_length=str(input_len))
