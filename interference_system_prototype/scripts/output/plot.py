"""Write the mean and p95 metrics of the colocated, disaggregated and prototype runs in RUN to RUN/metrics.csv, and the prototype's split by where its turns decoded to RUN/split_turns.csv and by where its conversations ran to RUN/split_conv.csv.

Run from this directory with the conserve env active:
    python plot.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

for d in ("src", "helpers"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / d))

from metric import SUFFIX
from runs import OUTPUT_DIR

RUN = OUTPUT_DIR / "test0"
SYSTEMS = ("colocated", "disagg", "prototype")
METRICS = ("queue", "ttft", "e2e", "tpot")
STATS = ("mean", "p95")
COLS = (*METRICS, "tbt_s1", "tbt_s2", "handoff")


def _rows(system):
    """Request rows of system's run in RUN."""
    return [json.loads(line) for line in (RUN / system / "requests.jsonl").read_text().splitlines()]


def _roles(system):
    """{gpu: role} of system's run in RUN, from its run.properties."""
    props = dict(line.split("=", 1) for line in (RUN / system / "run.properties").read_text().splitlines())
    return {int(gpu): role for gpu, role in json.loads(props["engines"]).items()}


def _gpu_request_times(row, times):
    """Emission times of each GPU request's tokens that reached the user, from its engine's token times."""
    return [times[gpu_request["gpu"]][f"cmpl-{row['id']}-0"][: gpu_request["completion_tokens"]] for gpu_request in row["gpu_requests"]]


def _metrics(row, times):
    """Queue time before admission, TTFT and end-to-end latency from admission, and the measured gaps between the tokens the user sees, as lists of seconds, of a request row."""
    gpu_requests = row["gpu_requests"]
    shown = [b is None or b["prompt_tokens"] != a["prompt_tokens"] for a, b in zip(gpu_requests, gpu_requests[1:] + [None])]
    seen = [t for gpu_request_times, s in zip(_gpu_request_times(row, times), shown) if s for t in gpu_request_times]
    admitted = gpu_requests[0]["first_step"]
    return {
        "queue": [admitted - gpu_requests[0]["sent"]],
        "ttft": [gpu_requests[shown.index(True)]["first_token"] - admitted],
        "e2e": [gpu_requests[-1]["done"] - admitted],
        "tpot": list(np.diff(seen)),
    }


def _pooled(rows, times):
    """{metric: every value of it over rows}."""
    values = defaultdict(list)
    for r in rows:
        for m, v in _metrics(r, times).items():
            values[m] += v
    return values


def _stats(values):
    """Mean and p95 of values, blank if there are none."""
    return (np.mean(values), np.percentile(values, 95)) if values else ("", "")


def _token_times(log_dir):
    """{request id: times its tokens were emitted} from the step outputs in log_dir's core.jsonl."""
    times = defaultdict(list)
    for line in (log_dir / "core.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["event"] == "step_end":
            for rid in rec["finished_request_ids"]:  # every request with output this step, not only finished ones
                times[SUFFIX.sub("", rid)].append(datetime.fromisoformat(rec["timestamp"]).timestamp())
    return times


def _times(system, rows):
    """{gpu: {request id: token emission times}} of each GPU in system's run."""
    return {gpu: _token_times(RUN / system / f"gpu{gpu}") for gpu in {gpu_request["gpu"] for r in rows for gpu_request in r["gpu_requests"]}}


def _group_stats(rows, times, roles, legs):
    """Mean and p95 of every metric over rows, and if legs of the time between tokens on S1 and S2 and the handoff gap, in COLS order."""
    values = _pooled(rows, times)
    for r in rows if legs else ():
        tokens = _gpu_request_times(r, times)
        for gpu_request, t in zip(r["gpu_requests"], tokens):
            values["tbt_s1" if roles[gpu_request["gpu"]] == "kv_producer" else "tbt_s2"] += list(np.diff(t))
        if len(tokens) == 2:
            values["handoff"].append(tokens[1][0] - tokens[0][-1])
    return [x for m in COLS for x in _stats(values[m])]


def summary():
    """Write each system's mean and p95 of every metric to RUN/metrics.csv."""
    with open(RUN / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["system", *(f"{m}_{stat}_s" for m in METRICS for stat in STATS)])
        for system in SYSTEMS:
            rows = _rows(system)
            values = _pooled(rows, _times(system, rows))
            writer.writerow([system, *(x for m in METRICS for x in _stats(values[m]))])


def split():
    """Write mean and p95 metrics, time between tokens on S1 and S2, and handoff gap of the prototype's turns by where they decoded, S1 and S2 told apart by GPU role, then of the colocated run in total and over the turns the prototype finished on S1, then of the prototype in total, to RUN/split_turns.csv."""
    proto, col = _rows("prototype"), _rows("colocated")
    pt, ct = _times("prototype", proto), _times("colocated", col)
    roles = _roles("prototype")
    on_s1 = [r for r in proto if len(r["gpu_requests"]) == 1 and roles[r["gpu_requests"][0]["gpu"]] == "kv_producer"]
    on_s2 = [r for r in proto if len(r["gpu_requests"]) == 1 and roles[r["gpu_requests"][0]["gpu"]] == "kv_consumer"]
    groups = [
        ("colocated_total", col, ct, False),
        ("prototype_total", proto, pt, True),
        ("colocated_s1_turns", [r for r in col if r["id"] in {s["id"] for s in on_s1}], ct, False),
        ("whole_on_s1", on_s1, pt, True),
        ("whole_on_s2", on_s2, pt, True),
        ("s1_transfered_mid-decode", [r for r in proto if len(r["gpu_requests"]) == 2 and r["gpu_requests"][0]["completion_tokens"] > 1], pt, True),
        ("s1_transfered_after_prefill", [r for r in proto if len(r["gpu_requests"]) == 2 and r["gpu_requests"][0]["completion_tokens"] == 1], pt, True),
        
    ]
    with open(RUN / "split_turns.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "turns", *(f"{m}_{stat}_s" for m in COLS for stat in STATS)])
        for name, group, times, on_s1_s2 in groups:
            writer.writerow([name, len(group), *_group_stats(group, times, roles, on_s1_s2)])


def by_conversation():
    """Write mean and p95 metrics, time between tokens on S1 and S2, and handoff gap of the prototype's conversations by where they ran: entirely on S1, on S2 after their first turn, moved on a later turn, and the latter's turns before and after the move, to RUN/split_conv.csv."""
    proto = _rows("prototype")
    pt, roles = _times("prototype", proto), _roles("prototype")
    convs = defaultdict(list)
    for r in sorted(proto, key=lambda r: int(r["id"].split("-")[1])):
        convs[r["id"].split("-")[0]].append(r)
    groups = defaultdict(list)
    for turns in convs.values():
        moved = next((i for i, r in enumerate(turns) if len(r["gpu_requests"]) == 2), None)
        if moved is None:
            groups["entirely_on_s1"].append(turns)
        elif moved == 0:
            groups["s2_after_t1"].append(turns)
        else:
            groups["moved_mid_run"].append(turns)
            groups["moved_mid_run_s1"].append(turns[:moved])
            groups["moved_mid_run_s2"].append(turns[moved + 1 :])
    with open(RUN / "split_conv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "conversations", "turns", *(f"{m}_{stat}_s" for m in COLS for stat in STATS)])
        for name in ("entirely_on_s1", "s2_after_t1", "moved_mid_run", "moved_mid_run_s1", "moved_mid_run_s2"):
            rows = [r for turns in groups[name] for r in turns]
            writer.writerow([name, len(groups[name]), len(rows), *_group_stats(rows, pt, roles, True)])


if __name__ == "__main__":
    summary()
    split()
    by_conversation()
