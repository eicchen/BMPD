"""Requests and prompts of interference runs with their latency metrics, and the engine steps of each run."""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

import common  # puts the prototype's metric module on sys.path
from metric import SUFFIX

RUN_NAMES = ("colocated", "disaggregated/prefill", "disaggregated/decode")
STATISTICS = {"mean": np.mean, "p50": np.median, "p95": lambda values: np.percentile(values, 95), "max": np.max}


# Run records


def run_properties(run_dir):
    """run_dir's run.properties as a dict."""
    return dict(line.split("=", 1) for line in (run_dir / "run.properties").read_text().splitlines())


def token_times(log_dir):
    """When the engine emitted each request's tokens, from the step-end events in log_dir's core.jsonl."""
    emission_times = defaultdict(list)
    for line in (log_dir / "core.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "step_end":
            timestamp = event["timestamp"]
            for request_id in event["finished_request_ids"]:  # every request with output this step, not only finished ones
                emission_times[SUFFIX.sub("", request_id)].append(timestamp if isinstance(timestamp, float) else datetime.fromisoformat(timestamp).timestamp())
    return emission_times


# ---------------


# Requests and prompts


@dataclass
class Request:
    """One request of a run: its timestamps and the engine's emission time of each of its tokens."""

    prompt_index: str
    sent: float
    first_step: float
    first_token: float
    done: float
    emissions: list

    @property
    def queue(self):
        """Seconds from send to the first engine step: time queued in vLLM, and for decode requests their KV load."""
        return self.first_step - self.sent

    @property
    def ttft_from_send(self):
        """Seconds from send to the first token."""
        return self.first_token - self.sent

    @property
    def ttft_engine_time(self):
        """Seconds from the first engine step to the first token."""
        return self.first_token - self.first_step

    @property
    def tpot(self):
        """Seconds between consecutive tokens, from the engine's step timestamps."""
        return list(np.diff(self.emissions))

    @property
    def e2e_from_send(self):
        """Seconds from send to the last token."""
        return self.done - self.sent

    @property
    def e2e_engine_time(self):
        """Seconds from the first engine step to the last token."""
        return self.done - self.first_step


@dataclass
class Prompt:
    """One prompt of a cell: its colocated request against its disaggregated prefill and decode."""

    index: str
    colocated: Request
    disaggregated: "Disaggregated"

    @property
    def interference_from_send(self):
        """Seconds the colocated request took beyond the disaggregated prefill plus decode, from send."""
        return self.colocated.e2e_from_send - self.disaggregated.e2e_from_send

    @property
    def interference_engine_time(self):
        """Seconds the colocated request took beyond the disaggregated prefill plus decode, in engine time."""
        return self.colocated.e2e_engine_time - self.disaggregated.e2e_engine_time


# ---------------


# Disaggregated prefill and decode


@dataclass
class Disaggregated:
    """One prompt's disaggregated prefill and decode requests, served in separate runs."""

    prefill: Request
    decode: Request

    @property
    def ttft_from_send(self):
        """The prefill's E2E plus the decode's TTFT from send."""
        return self.prefill.e2e_from_send + self.decode.ttft_from_send

    @property
    def ttft_engine_time(self):
        """The prefill's E2E plus the decode's TTFT in engine time."""
        return self.prefill.e2e_engine_time + self.decode.ttft_engine_time

    @property
    def e2e_from_send(self):
        """The prefill's plus the decode's E2E from send."""
        return self.prefill.e2e_from_send + self.decode.e2e_from_send

    @property
    def e2e_engine_time(self):
        """The prefill's plus the decode's E2E in engine time."""
        return self.prefill.e2e_engine_time + self.decode.e2e_engine_time


# ---------------


# Loading


def load_requests(run_dir):
    """run_dir's requests, with their token emission times from its core.jsonl."""
    request_results = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    emission_times = token_times(run_dir / f"gpu{request_results[0]['gpu_requests'][0]['gpu']}")
    run_requests = []
    for request_result in request_results:
        gpu_request = request_result["gpu_requests"][0]
        run_requests.append(
            Request(
                prompt_index=request_result["id"].split("-", 1)[1],
                sent=gpu_request["sent"],
                first_step=gpu_request["first_step"],
                first_token=gpu_request["first_token"],
                done=gpu_request["done"],
                emissions=emission_times[f"cmpl-{request_result['id']}-0"][: gpu_request["completion_tokens"]],
            ),
        )
    return run_requests


def load_prompts(data_dir):
    """The prompts of the cell in data_dir, in prompt index order."""
    colocated, prefill, decode = ({request.prompt_index: request for request in load_requests(data_dir / run_name)} for run_name in RUN_NAMES)
    return [Prompt(index, colocated[index], Disaggregated(prefill[index], decode[index])) for index in sorted(colocated)]


# Engine steps


def step_kinds(run_dir):
    """Engine steps of run_dir that did work, split into decode-only, mixed and prefill-only by whether they computed more than one prompt token per context request."""
    steps = [
        step
        for steps_file in run_dir.glob("gpu*/steps.jsonl")
        for step in map(json.loads, steps_file.read_text().splitlines())
        if step["ctx_tokens"] or step["gen_reqs"]
    ]
    return {
        "decode-only": [step for step in steps if step["ctx_tokens"] <= step["ctx_reqs"]],
        "mixed": [step for step in steps if step["ctx_tokens"] > step["ctx_reqs"] and step["gen_reqs"]],
        "prefill-only": [step for step in steps if step["ctx_tokens"] > step["ctx_reqs"] and not step["gen_reqs"]],
    }
