"""Write the interference report of a pd_interference experiment (latency per run, colocated against disaggregated, engine step mix and end-to-end latency) to interference.md, and its latency per run to interference.csv.

Run from this directory with the conserve env active; an argument names the experiment, read from OUTPUT_DIR/<name>/data with outputs written to OUTPUT_DIR/<name>/plot, and defaults to smoke_test:
    python summarize_pd_interference.py
    python summarize_pd_interference.py smoke_test
"""

import csv
import sys

import numpy as np

from pd_interference import OUTPUT_DIR
from interference_metrics import RUN_NAMES, STATISTICS, load_prompts, load_requests, run_properties, step_kinds

LATENCIES = (("queue", "mean"), ("queue", "p95"), ("ttft_engine_time", "mean"), ("ttft_engine_time", "p95"), ("tpot", "mean"), ("tpot", "p50"), ("tpot", "p95"), ("e2e_engine_time", "mean"))


# Report data


def measure_latencies(data_dir):
    """Request count and LATENCIES in seconds of each run, NaN for TPOT of a run that generates one token per request."""
    run_latencies = {}
    for run_name in RUN_NAMES:
        run_requests = load_requests(data_dir / run_name)
        values = {
            "queue": [request.queue for request in run_requests],
            "ttft_engine_time": [request.ttft_engine_time for request in run_requests],
            "tpot": [gap for request in run_requests for gap in request.tpot],
            "e2e_engine_time": [request.e2e_engine_time for request in run_requests],
        }
        run_latencies[run_name] = {
            "requests": len(run_requests),
            **{f"{metric}_{statistic}_s": STATISTICS[statistic](values[metric]) if values[metric] else np.nan for metric, statistic in LATENCIES},
        }
    return run_latencies


def comparisons(run_latencies):
    """Colocated against the disaggregated prefill run for queue and TTFT and the decode run for TPOT, as (metric, colocated, disaggregated, colocated change in percent) rows."""
    compared_latencies = (
        ("TTFT s", "mean", 1, "ttft_engine_time", "disaggregated/prefill"),
        ("TTFT s", "p95", 1, "ttft_engine_time", "disaggregated/prefill"),
        ("queue s", "mean", 1, "queue", "disaggregated/prefill"),
        ("TPOT ms", "mean", 1e3, "tpot", "disaggregated/decode"),
        ("TPOT ms", "p50", 1e3, "tpot", "disaggregated/decode"),
        ("TPOT ms", "p95", 1e3, "tpot", "disaggregated/decode"),
    )
    comparison_rows = []
    for label, statistic, scale, metric, disaggregated_run in compared_latencies:
        colocated_value = run_latencies["colocated"][f"{metric}_{statistic}_s"] * scale
        disaggregated_value = run_latencies[disaggregated_run][f"{metric}_{statistic}_s"] * scale
        comparison_rows.append((f"{label} {statistic}", colocated_value, disaggregated_value, (colocated_value / disaggregated_value - 1) * 100))
    return comparison_rows


def step_statistics(data_dir):
    """(run, step kind, steps, mean ms, p95 ms, mean prompt tokens) of each kind of engine step that did work in each run."""
    step_rows = []
    for run_name in RUN_NAMES:
        for step_kind, kind_steps in step_kinds(data_dir / run_name).items():
            if kind_steps:
                durations = np.array([step["end"] - step["start"] for step in kind_steps]) * 1e3
                step_rows.append((run_name, step_kind, len(kind_steps), durations.mean(), np.percentile(durations, 95), np.mean([step["ctx_tokens"] for step in kind_steps])))
    return step_rows


def e2e_statistics(prompts):
    """STATISTICS of the colocated and disaggregated E2E from send and in engine time, the mean interference, and the interference per prompt."""
    prompt_interference_from_send = np.array([prompt.interference_from_send for prompt in prompts])
    labelled_latencies = {
        "colocated from send": [prompt.colocated.e2e_from_send for prompt in prompts],
        "disaggregated from send": [prompt.disaggregated.e2e_from_send for prompt in prompts],
        "colocated engine time": [prompt.colocated.e2e_engine_time for prompt in prompts],
        "disaggregated engine time": [prompt.disaggregated.e2e_engine_time for prompt in prompts],
    }
    return {label: [function(values) for function in STATISTICS.values()] for label, values in labelled_latencies.items()}, prompt_interference_from_send.mean(), prompt_interference_from_send


def e2e_totals(prompts):
    """Total and per-prompt E2E from send, colocated and disaggregated, with their difference and the colocated change in percent."""
    colocated = [prompt.colocated.e2e_from_send for prompt in prompts]
    disaggregated = [prompt.disaggregated.e2e_from_send for prompt in prompts]
    totals = {}
    for label, aggregate in (("Total run", np.sum), ("Per request", np.mean)):
        colocated_value, disaggregated_value = aggregate(colocated), aggregate(disaggregated)
        totals[label] = (colocated_value, disaggregated_value, colocated_value - disaggregated_value, (colocated_value / disaggregated_value - 1) * 100)
    return totals


# ---------------


# Report sections


def _number(value):
    """value to 3 decimals, - if NaN."""
    return "-" if np.isnan(value) else f"{value:.3f}"


def _table(header, rows):
    """header and rows as a markdown table."""
    return "\n".join(f"| {' | '.join(map(str, cells))} |" for cells in (header, ["---"] * len(header), *rows))


def latency_section(run_latencies):
    """Queue, TTFT, TPOT and engine E2E of each run"""
    header = ["run", "n", "queue mean (s)", "queue p95 (s)", "TTFT mean (s)", "TTFT p95 (s)", "TPOT mean (ms)", "TPOT p50 (ms)", "TPOT p95 (ms)", "E2E mean (s)"]
    table_rows = [
        [
            run_name,
            latencies["requests"],
            *(_number(latencies[f"{metric}_{statistic}_s"] * (1e3 if metric == "tpot" else 1)) for metric, statistic in LATENCIES),
        ]
        for run_name, latencies in run_latencies.items()
    ]
    return f"## Latency per run\n\nQueue time runs from send to the request's first engine step; TTFT and E2E run from that step.\n\n{_table(header, table_rows)}"


def comparison_section(comparison_rows):
    """Markdown table of the colocated against disaggregated comparison_rows."""
    table_rows = [[metric, _number(colocated_value), _number(disaggregated_value), f"{change:+.0f}%"] for metric, colocated_value, disaggregated_value, change in comparison_rows]
    return f"## Colocated against disaggregated\n\n{_table(['metric', 'colocated', 'disaggregated', 'change'], table_rows)}"


def steps_section(step_rows):
    """Markdown table of the engine step_rows."""
    table_rows = [[run_name, step_kind, steps, _number(mean), _number(p95), f"{prompt_tokens:.0f}"] for run_name, step_kind, steps, mean, p95, prompt_tokens in step_rows]
    return f"## Engine steps\n\n{_table(['run', 'step kind', 'steps', 'mean (ms)', 'p95 (ms)', 'mean prompt tokens'], table_rows)}"


def e2e_section(e2e_by_statistic, interference_mean, request_interference):
    """Markdown table of the end-to-end latency statistics and the colocated penalty from send by prompt index."""
    table_rows = [[name, *map(_number, values)] for name, values in e2e_by_statistic.items()]
    return (
        "## End-to-end latency (s)\n\nDisaggregated adds each prompt's prefill and decode.\n\n"
        f"{_table(['', *STATISTICS], table_rows)}\n\n"
        f"colocated - disaggregated from send: mean {interference_mean:.2f} s, by prompt index {np.round(request_interference, 1).tolist()}"
    )


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "smoke_test"
    data_dir = OUTPUT_DIR / experiment_name / "data"
    plot_dir = OUTPUT_DIR / experiment_name / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_latencies = measure_latencies(data_dir)
    prompts = load_prompts(data_dir)
    colocated_properties = run_properties(data_dir / "colocated")
    with open(plot_dir / "interference.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, ["run", "requests", *(f"{metric}_{statistic}_s" for metric, statistic in LATENCIES)])
        writer.writeheader()
        writer.writerows({"run": run_name, **latencies} for run_name, latencies in run_latencies.items())
    sections = (
        f"# {experiment_name}",
        f"{colocated_properties['request_count']} prompts, input length {colocated_properties['prompt_length']} tokens, output length {colocated_properties['output_length']} tokens, max_num_seqs {colocated_properties.get('max_num_seqs')}.",
        latency_section(run_latencies),
        comparison_section(comparisons(run_latencies)),
        steps_section(step_statistics(data_dir)),
        e2e_section(*e2e_statistics(prompts)),
    )
    (plot_dir / "interference.md").write_text("\n\n".join(sections) + "\n")
    print(f"Wrote {plot_dir / 'interference.md'}, {plot_dir / 'interference.csv'}")
    for label, (colocated_value, disaggregated_value, difference, change) in e2e_totals(prompts).items():
        print(f"{label} E2E: colocated {colocated_value:.2f} s, disaggregated {disaggregated_value:.2f} s, difference {difference:.2f} s ({change:+.1f}%)")
