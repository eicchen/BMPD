"""Parse a run's engine logs into per-GPU-request fields, per-step rows and a run summary, and write them into the run's directory."""

import json
import re
from bisect import bisect_right
from datetime import datetime

SUFFIX = re.compile(r"-[0-9a-f]{8}$")
HOLD = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\].*\[conserve_pd\] hold (?:released|timeout) req=(\S+) .*?waited=([\d.]+)s")
ITERATION = re.compile(r"Iteration\(\d+\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, (\d+) generation tokens")
COUNTS = ("ctx_reqs", "ctx_tokens", "gen_reqs", "gen_tokens")
STEP_FIELDS = ("start", "end", *COUNTS, "kv_usage", "waiting")
GPU_REQUEST_FIELDS = ("arrived", "first_step", "hold_start", "hold_end", "mixed_wait")


# Raw logs


def _read_jsonl(path):
    """Records of a JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _epoch(iso):
    """Epoch seconds of an ISO timestamp."""
    return datetime.fromisoformat(iso).timestamp()


def _arrivals(log_dir):
    """When each request reached the engine, from the request_start events in log_dir's engine.jsonl."""
    return {event["request_id"]: _epoch(event["timestamp"]) for event in _read_jsonl(log_dir / "engine.jsonl") if event["event"] == "request_start"}


def _steps(log_dir, server_log):
    """Engine steps from log_dir's core.jsonl, each step_end timed from the step_start just before it, with the context and generation counts of every step that ran requests from server_log's Iteration lines."""
    steps = []
    step_start = None
    for event in _read_jsonl(log_dir / "core.jsonl"):
        if event["event"] == "step_start":
            step_start = _epoch(event["timestamp"])  # async scheduling: opens the pass that collects the batch scheduled one pass earlier
            continue
        steps.append(
            {
                "start": step_start,
                "end": _epoch(event["timestamp"]),
                "request_ids": [SUFFIX.sub("", request_id) for request_id in event["executed_request_ids"]],
                **dict.fromkeys(COUNTS, 0),
                "kv_usage": event["kv_cache_usage"],
            },
        )
    running_steps = [step for step in steps if step["request_ids"]]
    for step, counts in zip(running_steps, ITERATION.findall(server_log), strict=True):
        step.update(zip(COUNTS, map(int, counts)))
    return steps


def _holds(server_log):
    """Start and end of each held request's hold, from server_log's conserve_pd hold lines."""
    holds = {}
    for logged_at, request_id, waited in HOLD.findall(server_log):
        hold_end = datetime.strptime(logged_at, "%Y-%m-%d %H:%M:%S,%f").timestamp()
        holds[SUFFIX.sub("", request_id)] = {"hold_start": hold_end - float(waited), "hold_end": hold_end}
    return holds


# Derived fields


def _first_steps(steps):
    """Start of the first step that ran each request."""
    first_steps = {}
    for step in steps:
        for request_id in step["request_ids"]:
            first_steps.setdefault(request_id, step["start"])
    return first_steps


def _waiting(steps, arrivals, first_steps):
    """Requests that had arrived but not yet started at each step's start."""
    arrival_times = sorted(arrivals.values())
    first_step_times = sorted(first_steps.values())
    return [bisect_right(arrival_times, step["start"]) - bisect_right(first_step_times, step["start"]) for step in steps]


def _mixed_steps(steps):
    """Steps that ran both context and generation work."""
    return [step for step in steps if step["ctx_reqs"] and step["gen_reqs"]]


def _mixed_wait(mixed_steps, arrived, first_step):
    """Seconds of mixed_steps that overlap the wait from arrived to first_step."""
    return sum((max(0.0, min(step["end"], first_step) - max(step["start"], arrived)) for step in mixed_steps), 0.0)


def _engine_fields(log_dir):
    """The steps of log_dir's engine logs with their waiting counts, and each request's GPU_REQUEST_FIELDS."""
    server_log = (log_dir / "server.log").read_text()
    steps = _steps(log_dir, server_log)
    arrivals = _arrivals(log_dir)
    first_steps = _first_steps(steps)
    for step, waiting in zip(steps, _waiting(steps, arrivals, first_steps)):
        step["waiting"] = waiting
    holds = _holds(server_log)
    mixed_steps = _mixed_steps(steps)
    gpu_request_fields = {}
    for request_id, arrived in arrivals.items():
        first_step = first_steps.get(request_id)
        gpu_request_fields[request_id] = {
            "arrived": arrived,
            "first_step": first_step,
            **holds.get(request_id, {"hold_start": None, "hold_end": None}),
            "mixed_wait": _mixed_wait(mixed_steps, arrived, first_step) if first_step else None,
        }
    return steps, gpu_request_fields


# Run files


def parse_engine_logs(request_results, engine_roles, output_dir):
    """Add each GPU request's GPU_REQUEST_FIELDS to request_results from the engine logs in output_dir, and write requests.jsonl, each GPU's steps.jsonl and summary.properties."""
    summary = {"kv_producer_colocated": sum(len(request_result["gpu_requests"]) == 1 and engine_roles[request_result["gpu_requests"][0]["gpu"]] == "kv_producer" for request_result in request_results)}
    fields_by_gpu_request = {}
    for gpu in engine_roles:
        log_dir = output_dir / f"gpu{gpu}"
        steps, gpu_request_fields = _engine_fields(log_dir)
        (log_dir / "steps.jsonl").write_text("".join(json.dumps({field: step[field] for field in STEP_FIELDS}) + "\n" for step in steps))
        mixed_steps = _mixed_steps(steps)
        summary[f"gpu{gpu}.mixed_steps"] = len(mixed_steps)
        summary[f"gpu{gpu}.mixed_s"] = round(sum(step["end"] - step["start"] for step in mixed_steps), 3)
        fields_by_gpu_request |= {(gpu, request_id): fields for request_id, fields in gpu_request_fields.items()}
    for request_result in request_results:
        for gpu_request in request_result["gpu_requests"]:
            gpu_request |= fields_by_gpu_request.get((gpu_request["gpu"], f"cmpl-{request_result['id']}-0"), dict.fromkeys(GPU_REQUEST_FIELDS))
    (output_dir / "requests.jsonl").write_text("".join(json.dumps(request_result) + "\n" for request_result in request_results))
    (output_dir / "summary.properties").write_text("".join(f"{key}={value}\n" for key, value in summary.items()))
