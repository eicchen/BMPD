"""Write the interference report of a pd_interference experiment (latency per run, colocated against disaggregated, engine step mix and end-to-end latency) to interference.md, and its latency per run to interference.csv.

Run from this directory with the conserve env active; an argument names the experiment, read from OUTPUT_DIR/<name>/data with outputs written to OUTPUT_DIR/<name>/plot, and defaults to smoke_test:
    python summarize_pd_interference.py
    python summarize_pd_interference.py smoke_test
"""

import csv
import json
import sys

import numpy as np

from pd_interference import OUTPUT_DIR
from plot_interference import _properties, measure

RUN_NAMES = ("colocated", "disaggregated/prefill", "disaggregated/decode")
STATISTICS = {"mean": np.mean, "p50": np.median, "p95": lambda values: np.percentile(values, 95), "max": np.max}
LATENCIES = (("queue", "mean"), ("queue", "p95"), ("ttft", "mean"), ("ttft", "p95"), ("tpot", "mean"), ("tpot", "p50"), ("tpot", "p95"), ("e2e", "mean"))


def _number(value):
    """value to 3 decimals, - if NaN."""
    return "-" if np.isnan(value) else f"{value:.3f}"


def _table(header, rows):
    """header and rows as a markdown table."""
    return "\n".join(f"| {' | '.join(map(str, cells))} |" for cells in (header, ["---"] * len(header), *rows))


def _gpu_requests(run_dir):
    """Each request's GPU request in run_dir, keyed by its prompt index."""
    return {
        request_result["id"].split("-", 1)[1]: request_result["gpu_requests"][0]
        for request_result in map(json.loads, (run_dir / "requests.jsonl").read_text().splitlines())
    }


def measure_latencies(data_dir):
    """Request count and LATENCIES in seconds of each run, NaN for TPOT of a run that generates one token per request."""
    run_latencies = {}
    for run_name in RUN_NAMES:
        for values in measure(data_dir / run_name).values():
            run_latencies[run_name] = {
                "requests": len(values["queue"]),
                **{f"{metric}_{statistic}_s": STATISTICS[statistic](values[metric]) if values[metric] else np.nan for metric, statistic in LATENCIES},
            }
    return run_latencies


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
    return f"## Latency per run\n\n{_table(header, table_rows)}"


def comparison_section(data_dir):
    """Colocated against the disaggregated prefill phase for queue and TTFT and the decode phase for TPOT and decode time, with the colocated change."""
    colocated = measure(data_dir / "colocated")["colocated"]
    prefill = measure(data_dir / "disaggregated/prefill")["prefill"]
    decode = measure(data_dir / "disaggregated/decode")["decode"]
    comparisons = (
        ("TTFT s", "mean", 1, "ttft", prefill),
        ("TTFT s", "p95", 1, "ttft", prefill),
        ("queue s", "mean", 1, "queue", prefill),
        ("TPOT ms", "mean", 1e3, "tpot", decode),
        ("TPOT ms", "p50", 1e3, "tpot", decode),
        ("TPOT ms", "p95", 1e3, "tpot", decode),
        ("decode time s", "mean", 1, "e2e", decode),
    )
    table_rows = []
    for label, statistic, scale, metric, disaggregated in comparisons:
        colocated_value = STATISTICS[statistic](colocated[metric]) * scale
        disaggregated_value = STATISTICS[statistic](disaggregated[metric]) * scale
        table_rows.append([f"{label} {statistic}", _number(colocated_value), _number(disaggregated_value), f"{(colocated_value / disaggregated_value - 1) * 100:+.0f}%"])
    return f"## Colocated against disaggregated\n\n{_table(['metric', 'colocated', 'disaggregated', 'change'], table_rows)}"


def steps_section(data_dir):
    """Count, mean and p95 duration and mean prompt tokens of the engine steps that did work in each run, split into decode-only, mixed and prefill-only."""
    table_rows = []
    for run_name in RUN_NAMES:
        steps = [
            step
            for steps_file in (data_dir / run_name).glob("gpu*/steps.jsonl")
            for step in map(json.loads, steps_file.read_text().splitlines())
            if step["ctx_tokens"] or step["gen_reqs"]
        ]
        step_kinds = {
            "decode-only": [step for step in steps if step["gen_reqs"] and not step["ctx_tokens"]],
            "mixed": [step for step in steps if step["gen_reqs"] and step["ctx_tokens"]],
            "prefill-only": [step for step in steps if step["ctx_tokens"] and not step["gen_reqs"]],
        }
        for step_kind, kind_steps in step_kinds.items():
            if kind_steps:
                durations = np.array([step["end"] - step["start"] for step in kind_steps]) * 1e3
                table_rows.append([run_name, step_kind, len(kind_steps), _number(durations.mean()), _number(np.percentile(durations, 95)), f"{np.mean([step['ctx_tokens'] for step in kind_steps]):.0f}"])
    return f"## Engine steps\n\n{_table(['run', 'step kind', 'steps', 'mean (ms)', 'p95 (ms)', 'mean prompt tokens'], table_rows)}"


def e2e_latencies(data_dir):
    """End-to-end latency per request, colocated against its disaggregated prefill plus decode, full from send and engine only from first step."""
    colocated, prefill, decode = (_gpu_requests(data_dir / run_name) for run_name in RUN_NAMES)
    prompt_indices = sorted(colocated)
    e2e = {}
    for start, scope in (("sent", "full"), ("first_step", "engine")):
        e2e[f"colocated {scope}"] = np.array([colocated[index]["done"] - colocated[index][start] for index in prompt_indices])
        e2e[f"disaggregated {scope}"] = np.array([prefill[index]["done"] - prefill[index][start] + decode[index]["done"] - decode[index][start] for index in prompt_indices])
    return e2e


def e2e_section(e2e):
    """Markdown table of the end-to-end latencies in e2e and the colocated full penalty by arrival order."""
    table_rows = [[name, *(_number(function(values)) for function in STATISTICS.values())] for name, values in e2e.items()]
    penalty = e2e["colocated full"] - e2e["disaggregated full"]
    return (
        "## End-to-end latency (s)\n\nDisaggregated adds each prompt's prefill and decode.\n\n"
        f"{_table(['', *STATISTICS], table_rows)}\n\n"
        f"colocated - disaggregated full: mean {penalty.mean():.2f} s, by arrival order {np.round(penalty, 1).tolist()}"
    )


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "smoke_test"
    data_dir = OUTPUT_DIR / experiment_name / "data"
    plot_dir = OUTPUT_DIR / experiment_name / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_latencies = measure_latencies(data_dir)
    e2e = e2e_latencies(data_dir)
    colocated_properties = _properties(data_dir / "colocated")
    with open(plot_dir / "interference.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, ["run", "requests", *(f"{metric}_{statistic}_s" for metric, statistic in LATENCIES)])
        writer.writeheader()
        writer.writerows({"run": run_name, **latencies} for run_name, latencies in run_latencies.items())
    sections = (
        f"# {experiment_name}",
        f"{colocated_properties['request_count']} prompts, input length {colocated_properties['prompt_length']} tokens, output length {colocated_properties['output_length']} tokens, max_num_seqs {colocated_properties.get('max_num_seqs')}.",
        latency_section(run_latencies),
        comparison_section(data_dir),
        steps_section(data_dir),
        e2e_section(e2e),
    )
    (plot_dir / "interference.md").write_text("\n\n".join(sections) + "\n")
    print(f"Wrote {plot_dir / 'interference.md'}, {plot_dir / 'interference.csv'}")
    for label, aggregate in (("Total run", np.sum), ("Per Turn", np.mean)):
        colocated_value, disaggregated_value = aggregate(e2e["colocated full"]), aggregate(e2e["disaggregated full"])
        print(f"{label} E2E: colocated {colocated_value:.2f} s, disaggregated {disaggregated_value:.2f} s, difference {colocated_value - disaggregated_value:.2f} s ({(colocated_value / disaggregated_value - 1) * 100:+.1f}%)")
