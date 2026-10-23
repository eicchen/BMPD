"""Shared parts of the interference experiments: the servers and their reset between runs, the decode KV files, request serving and run records."""

import asyncio
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[3]
for scripts_subdir in ("src", "output"):
    sys.path.insert(0, str(ROOT / "interference_system_prototype" / "scripts" / scripts_subdir))
sys.path.insert(0, str(ROOT / "profiling" / "scripts"))

from launch_interference import GPU, KV_DIR, SCHEDULER_CLS, SCHEDULING_POLICY, SERVER_DIR, _log, launch, log_indent, ping
from system import CHUNK_SIZE, ENGINE_PORT, L1_SIZE_GB, LMCACHE_HTTP_PORT, client, complete
from token_generation import DATA_FILE
from config import GPU_MEM_UTIL, GPU_TYPE, MAX_MODEL_LEN, MODEL, MODEL_DIR

OUTPUT_ROOT = ROOT / "profiling" / "output" / "interference"
ARRIVAL_SEED = 0
ENGINE_LOGS = ("core.jsonl", "server.log", "engine.jsonl")


# Setup (resetting vLLM and LMCache between runs)


def _vllm_requests(gpu):
    """Running plus waiting requests of the engine on gpu, from its metrics."""
    metrics = requests.get(f"http://localhost:{ENGINE_PORT + gpu}/metrics").text
    return sum(float(value) for value in re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} (\S+)$", metrics, re.M))


def _lmcache_stores_in_flight():
    """Keys waiting to be stored plus store tasks in flight in the LMCache server."""
    store_controller = requests.get(f"http://localhost:{LMCACHE_HTTP_PORT}/status").json()["storage_manager"]["store_controller"]
    return store_controller["pending_keys_count"] + store_controller["in_flight_task_count"]


def wipe_engine(gpu):
    """Wait until vLLM on gpu holds no requests and LMCache has no stores in flight, then empty vLLM's prefix cache and LMCache's L1 so the next run starts with no cached KV."""
    while _vllm_requests(gpu) or _lmcache_stores_in_flight():
        time.sleep(0.5)
    requests.post(f"http://localhost:{ENGINE_PORT + gpu}/reset_prefix_cache")
    requests.post(f"http://localhost:{LMCACHE_HTTP_PORT}/cache/clear")
    _log("Wiped vLLM and LMCache")


@contextmanager
def servers(launch_engine, output_dir, gpu, max_num_seqs=None):
    """Yield the log directory of fresh servers in output_dir/servers, capped at max_num_seqs requests per step if given, if launch_engine, else of the running servers of launch_interference.py, with KV_DIR read-only so runs can't store KV files and emptied at the end."""
    if max_num_seqs and not launch_engine:
        raise ValueError("max_num_seqs needs launch_engine=True; the running servers keep the cap they were launched with")
    server_dir = output_dir / "servers" if launch_engine else SERVER_DIR
    with launch(server_dir, gpu, KV_DIR, max_num_seqs) if launch_engine else nullcontext():
        ping(gpu)
        KV_DIR.chmod(0o550)
        try:
            yield server_dir
        finally:
            delete_decode_file()


# Decode KV file handling


def write_decode_file(decode_requests, gpu=GPU):
    """Prefill each decode prompt once on the running servers, at its send time if it has one, so LMCache saves its KV as files in KV_DIR, then make KV_DIR read-only so runs can't add to it; return the prefills' results."""
    file_count = len(decode_requests) * (len(decode_requests[0]["prompt"]) // CHUNK_SIZE)
    KV_DIR.chmod(0o770)
    request_results = asyncio.run(send([{"send_time": 0, **request, "max_tokens": 1} for request in decode_requests], gpu))
    _log(f"Waiting for {file_count} KV files in {KV_DIR}...")
    while len(list(KV_DIR.glob("*.data"))) < file_count or any(KV_DIR.glob("*.tmp")):
        time.sleep(1)
    KV_DIR.chmod(0o550)
    return request_results


def delete_decode_file():
    """Delete the decode KV files in KV_DIR and make it writable again."""
    KV_DIR.chmod(0o770)
    for path in KV_DIR.glob("*.data"):
        path.unlink()
    _log(f"Deleted decode KV files in {KV_DIR}")


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


# Data serving


def send_times(arrival_rate, request_count):
    """Seeded Poisson send times at arrival_rate requests per second, or all 0 if arrival_rate is None."""
    return np.cumsum(np.random.default_rng(ARRIVAL_SEED).exponential(1 / arrival_rate, request_count)) if arrival_rate else np.zeros(request_count)


async def serve(engine_client, request, gpu):
    """Send request to the engine on gpu through engine_client; return its result with timings and token counts."""
    _log(f"{request['id']} start")
    gpu_request = await complete(engine_client, request)
    _log(f"{request['id']} end ({gpu_request['done'] - gpu_request['sent']:.2f} s)")
    return {
        "id": request["id"],
        "request_kind": request["request_kind"],
        "gpu_requests": [{"gpu": gpu, **gpu_request}],
    }


async def send(timed_requests, gpu):
    """Send each request to the engine on gpu at its send time and wait for all of them; return one result per request."""
    engine_client = client(gpu)

    async def send_request(request):
        await asyncio.sleep(request["send_time"])
        return await serve(engine_client, request, gpu)

    try:
        with log_indent():
            return await asyncio.gather(*(send_request(request) for request in timed_requests))
    finally:
        await engine_client.close()


# Run records


@contextmanager
def engine_logs(server_dir, output_dir, gpu):
    """Copy what the engine on gpu logs into server_dir during the with block to output_dir."""
    server_logs, run_logs = server_dir / f"gpu{gpu}", output_dir / f"gpu{gpu}"
    log_starts = {name: (server_logs / name).stat().st_size if (server_logs / name).exists() else 0 for name in ENGINE_LOGS}
    yield
    run_logs.mkdir(parents=True, exist_ok=True)
    for name, start in log_starts.items():
        with open(server_logs / name, "rb") as server_log:
            server_log.seek(start)
            (run_logs / name).write_bytes(server_log.read())


def check(request_results, request_kinds):
    """Sanity counts for a run per request kind: its requests, and those that loaded as expected from LMCache, their whole prompt for decode and nothing for every other kind."""
    counts = {}
    for request_kind in request_kinds:
        gpu_requests = [request_result["gpu_requests"][0] for request_result in request_results if request_result["request_kind"] == request_kind]
        counts[f"{request_kind}_results"] = len(gpu_requests)
        if request_kind == "decode":
            counts["decode_full_loads"] = sum(
                gpu_request["cached_tokens"] == min(gpu_request["prompt_tokens"] - 1, gpu_request["prompt_tokens"] // CHUNK_SIZE * CHUNK_SIZE)
                for gpu_request in gpu_requests
            )
        else:
            counts[f"{request_kind}_no_loads"] = sum(gpu_request["cached_tokens"] == 0 for gpu_request in gpu_requests)
    return counts


def write_properties(path, properties):
    """Write properties to path as key=value lines."""
    path.write_text("".join(f"{key}={value}\n" for key, value in properties.items()))


def base_properties(gpu, server_dir, max_num_seqs=None):
    """Properties every run records: arrival seed, token file, servers, GPU, model, engine settings, versions and creation time."""
    return {
        "arrival_seed": ARRIVAL_SEED,
        "token_file": DATA_FILE.name,
        "token_sha256": dict(line.split("=", 1) for line in DATA_FILE.with_suffix(".properties").read_text().splitlines())["sha256"],
        "server_dir": server_dir,
        "gpu": gpu,
        "model": MODEL,
        "gpu_type": GPU_TYPE,
        "gpu_mem_util": GPU_MEM_UTIL,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": max_num_seqs,
        "scheduling_policy": SCHEDULING_POLICY,
        "scheduler_cls": SCHEDULER_CLS,
        "l1_size_gb": L1_SIZE_GB,
        "chunk_size": CHUNK_SIZE,
        "vllm_version": version("vllm"),
        "lmcache_version": version("lmcache"),
        "openai_version": version("openai"),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
