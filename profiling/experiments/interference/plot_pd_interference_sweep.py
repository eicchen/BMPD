"""Plot heatmaps of the mean and p95 interference latency per request (colocated minus disaggregated full E2E), with its percentage slowdown, of a pd_interference sweep over input and output length, to OUTPUT_DIR/<name>/<max_num_seqs>/plot/interference_heatmap.png, and the latency by send order of one input and output length against the rest of the sweep to plot/interference_by_send_order_<input>_<output>.png.

Run from this directory with the conserve env active; arguments name the experiment, its max_num_seqs and the input and output length of the send-order plot, defaulting to length_sweep1, None, 4096 and 512:
    python plot_pd_interference_sweep.py
    python plot_pd_interference_sweep.py length_sweep1 None 4096 512
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

from pd_interference import OUTPUT_DIR
from plot_interference import INK, MARK, MUTED, SECONDARY, SURFACE, _properties, _style
from summarize_pd_interference import RUN_NAMES, STATISTICS, _gpu_requests, e2e_latencies

PANELS = (("mean", "Mean per request"), ("p95", "p95"))
COLORS = {"colocated": "#2a78d6", "disaggregated": "#eb6834"}
LENGTH_RAMP = LinearSegmentedColormap.from_list("length_ramp", ("#86b6ef", "#0d366b"))


def _sweep_lengths(sweep_dir):
    """Input and output lengths of the sweep in sweep_dir, ascending."""
    input_lengths = sorted(int(path.name) for path in sweep_dir.iterdir() if path.name.isdigit())
    output_lengths = sorted(int(path.name) for path in (sweep_dir / str(input_lengths[0])).iterdir())
    return input_lengths, output_lengths


def interference_grids(sweep_dir):
    """Input lengths, output lengths, and each PANELS statistic of the per-request colocated minus disaggregated full E2E in seconds and in percent of disaggregated, indexed by output then input length."""
    input_lengths, output_lengths = _sweep_lengths(sweep_dir)
    grids = {statistic: np.zeros((len(output_lengths), len(input_lengths))) for statistic, _ in PANELS}
    slowdowns = {statistic: np.zeros((len(output_lengths), len(input_lengths))) for statistic, _ in PANELS}
    for input_index, input_length in enumerate(input_lengths):
        for output_index, output_length in enumerate(output_lengths):
            e2e = e2e_latencies(sweep_dir / str(input_length) / str(output_length) / "data")
            interference = e2e["colocated full"] - e2e["disaggregated full"]
            for statistic, _ in PANELS:
                grids[statistic][output_index, input_index] = STATISTICS[statistic](interference)
                slowdowns[statistic][output_index, input_index] = STATISTICS[statistic](interference / e2e["disaggregated full"] * 100)
    return input_lengths, output_lengths, grids, slowdowns


def plot(experiment_name="length_sweep1", max_num_seqs="None"):
    """Plot the PANELS heatmaps of the sweep at max_num_seqs side by side to its plot/interference_heatmap.png."""
    sweep_dir = OUTPUT_DIR / experiment_name / max_num_seqs
    input_lengths, output_lengths, grids, slowdowns = interference_grids(sweep_dir)
    properties = _properties(sweep_dir / str(input_lengths[0]) / str(output_lengths[0]) / "data" / "colocated")
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
        f"{experiment_name}: colocated minus disaggregated E2E per request, {properties['request_count']} requests per cell {arrival}, max_num_seqs {max_num_seqs}",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    plot_dir = sweep_dir / "plot"
    plot_dir.mkdir(exist_ok=True)
    figure.savefig(plot_dir / "interference_heatmap.png", dpi=150)
    print(f"Wrote {plot_dir / 'interference_heatmap.png'}")


def ttft_latencies(data_dir):
    """Time to first token per request from send, colocated against its disaggregated prefill."""
    colocated, prefill = (_gpu_requests(data_dir / run_name) for run_name in RUN_NAMES[:2])
    return {
        run_kind: np.array([gpu_requests[index]["first_token"] - gpu_requests[index]["sent"] for index in sorted(colocated)])
        for run_kind, gpu_requests in (("colocated", colocated), ("disaggregated", prefill))
    }


def _plot_by_send_order(axis, latencies, title):
    """Plot the colocated and disaggregated latencies in seconds against send order on axis."""
    for run_kind, color in COLORS.items():
        axis.plot(np.arange(1, len(latencies[run_kind]) + 1), latencies[run_kind], color=color, label=run_kind, **MARK)
    _style(axis, title, "Send order")
    axis.set_ylabel(f"{title} (s)", color=SECONDARY)
    axis.legend(frameon=False, labelcolor=SECONDARY)


def _plot_interference_by_send_order(axis, sweep_dir, cells, highlighted, title, length_name):
    """Plot the per-request colocated minus disaggregated full E2E of each (input length, output length) cell in cells, keyed by its swept length, against send order on axis, with the highlighted one on top."""
    for (length, (input_length, output_length)), color in zip(cells.items(), LENGTH_RAMP(np.linspace(0, 1, len(cells)))):
        e2e = e2e_latencies(sweep_dir / str(input_length) / str(output_length) / "data")
        interference = e2e["colocated full"] - e2e["disaggregated full"]
        line_style = {**MARK, "lw": 3, "color": INK, "zorder": 3} if length == highlighted else {"lw": 1.5, "color": color}
        axis.plot(np.arange(1, len(interference) + 1), interference, label=length, **line_style)
    _style(axis, title, "Send order")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.autoscale(axis="y")
    axis.set_ylabel("E2E interference (s)", color=SECONDARY)
    axis.legend(frameon=False, labelcolor=SECONDARY, title=length_name).get_title().set_color(SECONDARY)


def plot_by_send_order(experiment_name="length_sweep1", max_num_seqs="None", input_length="4096", output_length="512"):
    """Plot the E2E and TTFT by send order of the input_length and output_length cell of the sweep at max_num_seqs, and the E2E interference by send order of every cell sharing its output or input length with it highlighted, to its plot/interference_by_send_order_<input_length>_<output_length>.png."""
    sweep_dir = OUTPUT_DIR / experiment_name / max_num_seqs
    input_length, output_length = int(input_length), int(output_length)
    input_lengths, output_lengths = _sweep_lengths(sweep_dir)
    data_dir = sweep_dir / str(input_length) / str(output_length) / "data"
    e2e = e2e_latencies(data_dir)
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=SURFACE)
    _plot_by_send_order(axes[0, 0], {run_kind: e2e[f"{run_kind} full"] for run_kind in COLORS}, "E2E")
    _plot_by_send_order(axes[0, 1], ttft_latencies(data_dir), "TTFT")
    _plot_interference_by_send_order(axes[1, 0], sweep_dir, {length: (length, output_length) for length in input_lengths}, input_length, f"E2E interference at {output_length} output tokens", "Input tokens")
    _plot_interference_by_send_order(axes[1, 1], sweep_dir, {length: (input_length, length) for length in output_lengths}, output_length, f"E2E interference at {input_length} input tokens", "Output tokens")
    figure.suptitle(
        f"{experiment_name}: latency by send order at {input_length} input and {output_length} output tokens, max_num_seqs {max_num_seqs}",
        x=0.01,
        ha="left",
        color=INK,
    )
    figure.tight_layout()
    plot_dir = sweep_dir / "plot"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / f"interference_by_send_order_{input_length}_{output_length}.png"
    figure.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    plot(*sys.argv[1:3])
    plot_by_send_order(*sys.argv[1:5])
