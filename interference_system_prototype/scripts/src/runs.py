import asyncio
import json
import logging
import sys
import threading
from datetime import datetime
from importlib.metadata import version

import requests

import prototype
from baselines import Colocated, Disagg
from load import ROOT, get_rec_turns
from metric import annotate
from system import CHUNK_SIZE, L1_SIZE_GB, LMCACHE_HTTP_PORT, launch_engine, launch_lmcache, progress, wait_engine

sys.path.insert(0, str(ROOT / "config"))
from config import GPU_MEM_UTIL, GPU_TYPE, MAX_MODEL_LEN, MODEL

OUTPUT_DIR = ROOT / "interference_system_prototype" / "output"


def _watch_l1(stop):
    """Log the LMCache server's L1 usage to progress every 30 s until stop is set."""
    while not stop.wait(30):
        l1 = requests.get(f"http://localhost:{LMCACHE_HTTP_PORT}/status").json()["storage_manager"]["l1_manager"]
        progress.info(
            "lmcache L1 %.1f/%.0f GB, %d objects",
            l1["memory_used_bytes"] / (1 << 30),
            l1["memory_configured_bytes"] / (1 << 30),
            l1["total_object_count"],
        )


def _run(recs, engines, replay, out, connector=None):
    """Serve recs with replay on engines {gpu: role} using KV connector, plus an LMCache server if any engine is not colocated, logging into out; write and return the request rows."""
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "progress.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    progress.addHandler(handler)
    progress.setLevel(logging.INFO)
    stop = threading.Event()
    procs = []
    try:
        if any(role != "colocated" for role in engines.values()):
            print(f"{datetime.now():%H:%M:%S} Launching LMCache server on GPUs {list(engines)}...", flush=True)
            procs.append(("LMCache server", launch_lmcache(list(engines), out)))
            print(f"{datetime.now():%H:%M:%S} LMCache server ready", flush=True)
            threading.Thread(target=_watch_l1, args=(stop,), daemon=True).start()
        started = {}
        for gpu, role in engines.items():
            print(f"{datetime.now():%H:%M:%S} Launching vLLM {role} on GPU {gpu}...", flush=True)
            started[gpu] = launch_engine(gpu, role, out / f"gpu{gpu}", connector)
            procs.append((f"vLLM on GPU {gpu}", started[gpu]))
        for gpu, proc in started.items():
            wait_engine(proc, gpu)
            print(f"{datetime.now():%H:%M:%S} vLLM on GPU {gpu} ready", flush=True)
        print(f"{datetime.now():%H:%M:%S} All engines healthy, replaying {get_rec_turns()} requests", flush=True)
        progress.info("replaying %d conversations, %d requests", len(recs), get_rec_turns())
        rows = asyncio.run(replay(recs, engines))
    finally:
        stop.set()
        progress.removeHandler(handler)
        handler.close()
        for name, proc in reversed(procs):
            print(f"{datetime.now():%H:%M:%S} Terminating {name}...", flush=True)
            proc.terminate()
            proc.wait()
    annotate(rows, engines, out)
    props = {
        "system": f"{replay.__module__}.{replay.__qualname__}",
        "requests": get_rec_turns(),
        "engines": json.dumps(engines),
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
    }
    (out / "run.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
    return rows


def run_disagg(recs, engines, out):
    return _run(recs, engines, Disagg.replay, out, "LMCacheMPConnector")


def run_col(recs, gpus, out):
    return _run(recs, {gpu: "colocated" for gpu in gpus}, Colocated.replay, out)


def run_proto(recs, engines, out):
    return _run(recs, engines, prototype.replay, out, "LMCacheAdaptiveConnector")
