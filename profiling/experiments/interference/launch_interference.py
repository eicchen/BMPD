"""Launch the LMCache server and vLLM engine the interference runs share, and keep them up until Ctrl-C.

Run from this directory on a GPU node with the conserve env active, then call the run functions of pd_mix.py or pd_interference.py from another terminal on the same node:
    python launch_interference.py
"""

import os
import signal
import sys
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "interference_system_prototype" / "scripts" / "src"))

from system import ENGINE_PORT, LMCACHE_HTTP_PORT, launch_engine, launch_lmcache, wait_engine
from config import MODEL_DIR

GPU = 0
SERVER_DIR = ROOT / "profiling" / "output" / "interference" / "servers"
KV_DIR = MODEL_DIR.parent / "interference_kv" / os.environ.get("SLURM_JOB_ID", "local")
SCHEDULING_POLICY = "priority"
SCHEDULER_CLS = "in_order_scheduler.InOrderScheduler"
# SCHEDULER_CLS = None
LOG_DEPTH = 0


def _log(message):
    """Print message after the time, indented by LOG_DEPTH levels."""
    print(f"{datetime.now():%H:%M:%S} {'    ' * LOG_DEPTH}{message}", flush=True)


@contextmanager
def log_indent():
    """Indent every _log in the with block by one more level."""
    global LOG_DEPTH
    LOG_DEPTH += 1
    try:
        yield
    finally:
        LOG_DEPTH -= 1


@contextmanager
def launch(output_dir, gpu, kv_dir, max_num_seqs=None):
    """Start an LMCache server backed by the KV files in kv_dir and one vLLM engine on gpu that loads from it, capped at max_num_seqs requests per step if given; both shut down when the with block exits."""
    with suppress(requests.ConnectionError):
        ping(gpu)
        raise RuntimeError(f"Servers already up on GPU {gpu}; stop launch_interference.py or pass launch_engine=False")
    output_dir.mkdir(parents=True, exist_ok=True)
    kv_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        _log(f"Launching LMCache server and vLLM on GPU {gpu}...")
        processes.append(launch_lmcache([gpu], output_dir, kv_dir))
        processes.append(launch_engine(gpu, "kv_both", output_dir / f"gpu{gpu}", "LMCacheMPConnector", max_num_seqs, SCHEDULING_POLICY, SCHEDULER_CLS))
        wait_engine(processes[-1], gpu)
        _log("LMCache server and vLLM ready")
        yield
    finally:
        _log("Terminating vLLM and LMCache server...")
        for process in reversed(processes):
            process.terminate()
            process.wait()
        _log("LMCache server and vLLM terminated")


def ping(gpu=GPU):
    """Raise unless vLLM on gpu and the LMCache server answer their health checks."""
    requests.get(f"http://localhost:{ENGINE_PORT + gpu}/health").raise_for_status()
    requests.get(f"http://localhost:{LMCACHE_HTTP_PORT}/healthcheck").raise_for_status()


if __name__ == "__main__":
    with launch(SERVER_DIR, GPU, KV_DIR):
        signal.pause()
