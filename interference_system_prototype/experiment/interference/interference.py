"""Interference between decode requests served from stored KV and prefill requests on one colocated engine, swept over the decode ratio.

Run from this directory on a GPU node with the conserve env active:
    python -c "from interference import *; write_decode_file(build_decode_requests(*DECODE_LENGTHS, REQUEST_COUNT))"
    python interference.py
    python -c "from interference import *; run(SWEEP_DIR / 'test', DECODE_LENGTHS, PREFILL_LENGTHS, 16, ARRIVAL_RATE, 0.5)"
"""

import asyncio
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests

PROTOTYPE_DIR = Path(__file__).resolve().parents[2]
for scripts_subdir in ("src", "output"):
    sys.path.insert(0, str(PROTOTYPE_DIR / "scripts" / scripts_subdir))
sys.path.insert(0, str(PROTOTYPE_DIR.parent / "profiling" / "scripts"))

from metric import annotate
from system import CHUNK_SIZE, ENGINE_PORT, L1_SIZE_GB, LMCACHE_HTTP_PORT, client, complete, launch_engine, launch_lmcache, wait_engine
from token_generation import DATA_FILE
from config import GPU_MEM_UTIL, GPU_TYPE, MAX_MODEL_LEN, MODEL, MODEL_DIR

EXPERIMENT = Path(__file__).stem
GPU = 0
DECODE_LENGTHS = (2048, 512)
PREFILL_LENGTHS = (2048, 1)
REQUEST_COUNT = 128
WARMUP_REQUEST_COUNT = 16
ARRIVAL_RATE = 2.0
DECODE_RATIOS = (0, 0.25, 0.5, 0.75, 1)
ARRIVAL_SEED = 0
MIX_SEED = 0
PROMPT_REGION_START = {"decode": 0.0, "prefill": 0.5}  # fraction of the token stream
KV_ROOT = MODEL_DIR.parent / "interference_kv"
SWEEP_DIR = PROTOTYPE_DIR / "output" / EXPERIMENT
ENGINE_LOGS = ("core.jsonl", "server.log", "engine.jsonl")


def _write_properties(path, properties):
    """Write properties to path as key=value lines."""
    path.write_text("".join(f"{key}={value}\n" for key, value in properties.items()))


def _log(message, depth=0):
    """Print message after the time, indented by depth levels."""
    print(f"{datetime.now():%H:%M:%S} {'    ' * depth}{message}", flush=True)


# Setup (launching vLLM and LMCache)


@contextmanager
def launch(output_dir, gpu, kv_dir):
    """Start an LMCache server backed by the KV files in kv_dir and one vLLM engine on gpu that loads from it; both shut down when the with block exits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        _log(f"Launching LMCache server and vLLM on GPU {gpu}...", 1)
        processes.append(launch_lmcache([gpu], output_dir, kv_dir))
        processes.append(launch_engine(gpu, "kv_both", output_dir / f"gpu{gpu}", "LMCacheMPConnector"))
        wait_engine(processes[-1], gpu)
        _log("LMCache server and vLLM ready", 1)
        yield
    finally:
        _log("Terminating vLLM and LMCache server...", 1)
        for process in reversed(processes):
            process.terminate()
            process.wait()
        _log("LMCache server and vLLM terminated", 1)


def _vllm_requests(gpu):
    """Running plus waiting requests of the engine on gpu, from its metrics."""
    metrics = requests.get(f"http://localhost:{ENGINE_PORT + gpu}/metrics").text
    return sum(float(value) for value in re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} (\S+)$", metrics, re.M))


def _lmcache_stores_in_flight():
    """Keys waiting to be stored plus store tasks in flight in the LMCache server."""
    store_controller = requests.get(f"http://localhost:{LMCACHE_HTTP_PORT}/status").json()["storage_manager"]["store_controller"]
    return store_controller["pending_keys_count"] + store_controller["in_flight_task_count"]


def wipe_engine(gpu):
    """Wait until vLLM on gpu holds no requests and LMCache has no stores in flight, then empty LMCache's L1 so the next run starts with no cached KV."""
    while _vllm_requests(gpu) or _lmcache_stores_in_flight():
        time.sleep(0.5)
    requests.post(f"http://localhost:{LMCACHE_HTTP_PORT}/cache/clear")
    _log("Wiped vLLM and LMCache", 1)


# Decode KV file handling


def decode_kv_dir(prompt_length):
    """Directory of the decode KV files for prompts of prompt_length."""
    return KV_ROOT / f"{EXPERIMENT}_{prompt_length}"


def write_decode_file(decode_requests, gpu=GPU):
    """Prefill each decode prompt once so LMCache saves its KV as files, then record decode.properties and make the directory read-only so runs can't add to it."""
    prompt_length = len(decode_requests[0]["prompt"])
    kv_dir = decode_kv_dir(prompt_length)
    kv_dir.mkdir(parents=True, exist_ok=True)
    kv_dir.chmod(0o770)
    file_count = len(decode_requests) * (prompt_length // CHUNK_SIZE)
    with launch(kv_dir / "logs", gpu, kv_dir):
        asyncio.run(send([{**request, "max_tokens": 1, "send_time": 0} for request in decode_requests], gpu))
        _log(f"Waiting for {file_count} KV files in {kv_dir}...", 1)
        while len(list(kv_dir.glob("*.data"))) < file_count or any(kv_dir.glob("*.tmp")):
            time.sleep(1)
    token_properties = dict(line.split("=", 1) for line in DATA_FILE.with_suffix(".properties").read_text().splitlines())
    _write_properties(
        kv_dir / "decode.properties",
        {
            "prompt_length": prompt_length,
            "request_count": len(decode_requests),
            "file_count": file_count,
            "chunk_size": CHUNK_SIZE,
            "prompt_region_start": PROMPT_REGION_START["decode"],
            "token_file": DATA_FILE.name,
            "token_sha256": token_properties["sha256"],
            "model": MODEL,
            "gpu_type": GPU_TYPE,
            "vllm_version": version("vllm"),
            "lmcache_version": version("lmcache"),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    )
    kv_dir.chmod(0o550)


def ensure_decode_file(prompt_length, request_count, gpu=GPU):
    """Return the decode KV directory for prompts of prompt_length, first writing it with write_decode_file if it holds fewer than request_count decode prompts."""
    kv_dir = decode_kv_dir(prompt_length)
    properties_path = kv_dir / "decode.properties"
    stored_count = int(dict(line.split("=", 1) for line in properties_path.read_text().splitlines())["request_count"]) if properties_path.exists() else 0
    if stored_count < request_count:
        _log(f"{kv_dir} holds {stored_count} of the {request_count} decode prompts needed, writing them...", 1)
        write_decode_file(build_decode_requests(prompt_length, 1, request_count), gpu)
    return kv_dir


def load_decode_file(decode_requests):
    """Copy the decode prompts' KV from the files into LMCache's CPU memory before a run, so decode requests don't read from disk mid-run; return chunks found and expected."""
    prefetch_url = f"http://localhost:{LMCACHE_HTTP_PORT}/cache/prefetches"
    prefetch_ids = [
        requests.post(
            prefetch_url,
            json={"model_name": MODEL, "world_size": 1, "token_ids": request["prompt"]},
        ).json()["request_id"]
        for request in decode_requests
    ]
    found_chunks = total_chunks = 0
    for prefetch_id in prefetch_ids:
        while (status := requests.get(f"{prefetch_url}/{prefetch_id}").json())["status"] == "pending":
            time.sleep(0.1)
        found_chunks += status["found_keys"]
        total_chunks += status["total_keys"]
    return found_chunks, total_chunks


# Data serving (prefill and decode parameters feeding the above, and the mix)


def prompts(request_kind, prompt_length, request_count):
    """First request_count prompts of prompt_length from request_kind's fixed region of the token stream, so decode prompts and their stored KV don't depend on the prefill settings."""
    token_stream = np.load(DATA_FILE, mmap_mode="r")
    region_start = int(PROMPT_REGION_START[request_kind] * len(token_stream))
    return [
        token_stream[region_start + index * prompt_length : region_start + (index + 1) * prompt_length].tolist()
        for index in range(request_count)
    ]


def build_decode_requests(prompt_length, output_length, request_count):
    """Decode requests: prompts whose KV is in the decode files, each generating output_length tokens."""
    return [
        {"id": f"decode-{index:04d}", "request_kind": "decode", "prompt": prompt, "max_tokens": output_length}
        for index, prompt in enumerate(prompts("decode", prompt_length, request_count))
    ]


def build_prefill_requests(prompt_length, output_length, request_count):
    """Prefill requests: prompts with no stored KV, each generating output_length tokens."""
    return [
        {"id": f"prefill-{index:04d}", "request_kind": "prefill", "prompt": prompt, "max_tokens": output_length}
        for index, prompt in enumerate(prompts("prefill", prompt_length, request_count))
    ]


def mix(decode_requests, prefill_requests, decode_ratio, arrival_rate, served_kinds=("decode", "prefill")):
    """One run's request list: each slot is a decode or prefill request picked at decode_ratio, with a Poisson send time at arrival_rate (all at time 0 if arrival_rate is None); solo runs keep only served_kinds."""
    request_count = len(decode_requests)
    send_times = np.cumsum(np.random.default_rng(ARRIVAL_SEED).exponential(1 / arrival_rate, request_count)) if arrival_rate else np.zeros(request_count)
    decode_slots = set(np.random.default_rng(MIX_SEED).permutation(request_count)[: round(decode_ratio * request_count)].tolist())
    slot_requests = (decode_requests[slot] if slot in decode_slots else prefill_requests[slot] for slot in range(request_count))
    return [
        {**request, "send_time": float(send_time)}
        for request, send_time in zip(slot_requests, send_times)
        if request["request_kind"] in served_kinds
    ]


async def send(mixed_requests, gpu):
    """Send each request to the engine on gpu at its send time and wait for all of them; return one result per request with its timings and token counts."""
    engine_client = client(gpu)

    async def send_request(request):
        await asyncio.sleep(request["send_time"])
        _log(f"{request['id']} start", 2)
        gpu_request = await complete(engine_client, request)
        _log(f"{request['id']} end ({gpu_request['done'] - gpu_request['sent']:.2f} s)", 2)
        return {
            "id": request["id"],
            "request_kind": request["request_kind"],
            "gpu_requests": [{"gpu": gpu, **gpu_request}],
        }

    try:
        return await asyncio.gather(*(send_request(request) for request in mixed_requests))
    finally:
        await engine_client.close()


# Run (single runs and sweeps)


def check(request_results):
    """Sanity counts for a run: decode requests that loaded their whole prompt from LMCache, and prefill requests that loaded nothing."""
    decode_gpu_requests = [request_result["gpu_requests"][0] for request_result in request_results if request_result["request_kind"] == "decode"]
    prefill_gpu_requests = [request_result["gpu_requests"][0] for request_result in request_results if request_result["request_kind"] == "prefill"]
    return {
        "decode_results": len(decode_gpu_requests),
        "decode_full_loads": sum(
            gpu_request["cached_tokens"] == min(gpu_request["prompt_tokens"] - 1, gpu_request["prompt_tokens"] // CHUNK_SIZE * CHUNK_SIZE)
            for gpu_request in decode_gpu_requests
        ),
        "prefill_results": len(prefill_gpu_requests),
        "prefill_no_loads": sum(gpu_request["cached_tokens"] == 0 for gpu_request in prefill_gpu_requests),
    }


def _serve_run(
    output_dir,
    server_dir,
    decode_lengths,
    prefill_lengths,
    request_count,
    arrival_rate,
    decode_ratio,
    served_kinds,
    gpu,
):
    """One run on servers already running with logs in server_dir: wipe them, load the decode KV, send the requests, then save this run's slice of the engine logs, the per-request timings and run.properties to output_dir."""
    mixed_requests = mix(
        build_decode_requests(*decode_lengths, request_count),
        build_prefill_requests(*prefill_lengths, request_count),
        decode_ratio,
        arrival_rate,
        served_kinds,
    )
    decode_requests = [request for request in mixed_requests if request["request_kind"] == "decode"]
    server_logs, run_logs = server_dir / f"gpu{gpu}", output_dir / f"gpu{gpu}"
    wipe_engine(gpu)
    log_starts = {name: (server_logs / name).stat().st_size if (server_logs / name).exists() else 0 for name in ENGINE_LOGS}
    _log(f"Loading KV of {len(decode_requests)} decode prompts into L1...", 1)
    found_chunks, total_chunks = load_decode_file(decode_requests)
    _log(f"Loaded {found_chunks}/{total_chunks} chunks", 1)
    request_results = asyncio.run(send(mixed_requests, gpu))
    run_logs.mkdir(parents=True, exist_ok=True)
    for name, start in log_starts.items():
        with open(server_logs / name, "rb") as server_log:
            server_log.seek(start)
            (run_logs / name).write_bytes(server_log.read())
    annotate(request_results, {gpu: "kv_both"}, output_dir)
    _write_properties(
        output_dir / "run.properties",
        {
            "decode_prompt_length": decode_lengths[0],
            "decode_output_length": decode_lengths[1],
            "prefill_prompt_length": prefill_lengths[0],
            "prefill_output_length": prefill_lengths[1],
            "request_count": request_count,
            "arrival_rate": arrival_rate,
            "decode_ratio": decode_ratio,
            "served_kinds": ",".join(served_kinds),
            "arrival_seed": ARRIVAL_SEED,
            "mix_seed": MIX_SEED,
            "decode_kv_dir": decode_kv_dir(decode_lengths[0]),
            "server_dir": server_dir,
            "found_chunks": found_chunks,
            "total_chunks": total_chunks,
            **check(request_results),
            "gpu": gpu,
            "model": MODEL,
            "gpu_type": GPU_TYPE,
            "gpu_mem_util": GPU_MEM_UTIL,
            "max_model_len": MAX_MODEL_LEN,
            "l1_size_gb": L1_SIZE_GB,
            "chunk_size": CHUNK_SIZE,
            "vllm_version": version("vllm"),
            "lmcache_version": version("lmcache"),
            "openai_version": version("openai"),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return request_results


def run(
    output_dir,
    decode_lengths,
    prefill_lengths,
    request_count,
    arrival_rate,
    decode_ratio,
    served_kinds=("decode", "prefill"),
    gpu=GPU,
):
    """One run on freshly launched servers: decode requests (KV loaded from the files) and prefill requests (full prefill) share one engine at decode_ratio and arrival_rate; logs, per-request timings and run.properties go to output_dir."""
    kv_dir = ensure_decode_file(decode_lengths[0], request_count, gpu)
    with launch(output_dir / "servers", gpu, kv_dir):
        return _serve_run(
            output_dir,
            output_dir / "servers",
            decode_lengths,
            prefill_lengths,
            request_count,
            arrival_rate,
            decode_ratio,
            served_kinds,
            gpu,
        )


def sweep(
    sweep_dir=SWEEP_DIR,
    decode_ratios=DECODE_RATIOS,
    run_solos=True,
    decode_lengths=DECODE_LENGTHS,
    prefill_lengths=PREFILL_LENGTHS,
    request_count=REQUEST_COUNT,
    arrival_rate=ARRIVAL_RATE,
    gpu=GPU,
):
    """Run every decode ratio into sweep_dir/<ratio>/ on one launch of the servers, after a discarded warm-up run: a mixed run, plus each kind served alone at the ratios between 0 and 1 as its baseline without interference."""
    server_dir = sweep_dir / "servers"
    _log(f"Sweep into {sweep_dir}")
    kv_dir = ensure_decode_file(decode_lengths[0], max(request_count, WARMUP_REQUEST_COUNT), gpu)
    with launch(server_dir, gpu, kv_dir):
        _log(f"[warm-up] decode_ratio=0.5 mixed, {WARMUP_REQUEST_COUNT} requests")
        _serve_run(
            sweep_dir / "warmup",
            server_dir,
            decode_lengths,
            prefill_lengths,
            WARMUP_REQUEST_COUNT,
            arrival_rate,
            0.5,
            ("decode", "prefill"),
            gpu,
        )
        for index, decode_ratio in enumerate(decode_ratios, 1):
            served_by_run_kind = {"mixed": ("decode", "prefill")}
            if run_solos and 0 < decode_ratio < 1:
                served_by_run_kind |= {"decode_only": ("decode",), "prefill_only": ("prefill",)}
            for run_kind, served_kinds in served_by_run_kind.items():
                _log(f"[{index}/{len(decode_ratios)}] decode_ratio={decode_ratio:g} {run_kind}")
                _serve_run(
                    sweep_dir / f"{decode_ratio:g}" / run_kind,
                    server_dir,
                    decode_lengths,
                    prefill_lengths,
                    request_count,
                    arrival_rate,
                    decode_ratio,
                    served_kinds,
                    gpu,
                )


if __name__ == "__main__":
    # run(output_dir, decode_lengths, prefill_lengths, request_count, arrival_rate, decode_ratio, served_kinds, gpu)
    # - output_dir: directory the run's engine logs, requests.jsonl and run.properties go into; the full server logs go into output_dir/servers
    # - decode_lengths: (prompt_length, output_length) of decode requests; the KV of the first request_count decode prompts is written to decode_kv_dir(prompt_length) first if it isn't stored yet
    # - prefill_lengths: (prompt_length, output_length) of prefill requests; output_length 1 makes them pure prefills
    # - request_count: number of request slots, each filled by a decode or a prefill request
    # - arrival_rate: requests per second, at seeded Poisson send times; None sends every request at time 0
    # - decode_ratio: fraction of slots that are decode, 0 to 1
    # - served_kinds: kinds actually sent; ("decode", "prefill") by default, ("decode",) or ("prefill",) for a solo run
    # - gpu: GPU id, GPU by default
    run(SWEEP_DIR / "smoke_test", DECODE_LENGTHS, PREFILL_LENGTHS, 16, ARRIVAL_RATE, 0.5)
    # sweep()
