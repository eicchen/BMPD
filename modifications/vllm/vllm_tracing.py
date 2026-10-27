"""vLLM general plugin that writes BMPD's request and engine-step event logs, enabled by vllm serve --engine-log-file and --core-log-file."""

# Logged events, one JSON line each with "event" and "timestamp" (wall clock when written).
#
# engine.jsonl (--engine-log-file):
# request_start: API server has tokenized the request and is handing it to the engine; not yet queued, scheduled or computed.
#
# core.jsonl (--core-log-file):
# step_start: engine core begins a step, before scheduling a batch and launching it on the GPU (async scheduling: some have no step_end).
# step_end: a batch's GPU results were turned into outputs, before they reach the API server (async scheduling: batch launched one step earlier).
#   executed_request_ids: requests computed in the batch (prefill chunk or decode), IDs suffixed "-<8 chars>" unlike request_start's.
#   kv_cache_usage: fraction of GPU KV cache blocks held by requests at step_end, after finished requests freed theirs.
#   finished_request_ids: requests that emitted output (usually one token) this step, finished or not; step_end ≈ their token time.
#   finish_reasons: per request above, "stop", "length", "abort", "error" or "repetition", or "None" if still running.
#   num_cached_tokens: per request above, prompt tokens loaded from prefix cache or LMCache instead of computed; first output only, else None.
#   out_token_count: number of requests with output this step (equals tokens emitted only without speculative decoding).

import json
import os
from datetime import datetime
from inspect import signature

ENGINE_LOG_ENV = "BMPD_ENGINE_LOG_FILE"
CORE_LOG_ENV = "BMPD_CORE_LOG_FILE"


def _write(path, event, **fields):
    """Append event with the current time and fields to the JSONL file at path."""
    with open(path, "a") as log_file:
        log_file.write(json.dumps({"event": event, "timestamp": datetime.now().timestamp(), **fields}) + "\n")


def register():
    """Add the two log flags to vLLM and write request_start, step_start and step_end events to the files they name."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core import EngineCore

    add_cli_args = EngineArgs.add_cli_args
    from_cli_args = EngineArgs.from_cli_args.__func__
    generate = AsyncLLM.generate
    generate_signature = signature(generate)
    step = EngineCore.step
    step_with_batch_queue = EngineCore.step_with_batch_queue
    update_from_output = Scheduler.update_from_output

    def add_log_file_args(parser):
        """vLLM's engine arguments plus --engine-log-file and --core-log-file."""
        parser = add_cli_args(parser)
        parser.add_argument("--engine-log-file", help="Append a request_start event for every request to this JSONL file.")
        parser.add_argument("--core-log-file", help="Write step_start and step_end events for every engine step to this JSONL file, replacing it at launch.")
        return parser

    def from_cli_args_with_logs(cls, args):
        """Engine arguments from args, with the log paths passed to the engine processes through the environment and the core log emptied."""
        if args.engine_log_file:
            os.environ[ENGINE_LOG_ENV] = args.engine_log_file
        if args.core_log_file:
            os.environ[CORE_LOG_ENV] = args.core_log_file
            os.makedirs(os.path.dirname(os.path.abspath(args.core_log_file)), exist_ok=True)
            if os.path.exists(args.core_log_file):
                os.remove(args.core_log_file)
        return from_cli_args(cls, args)

    async def generate_with_request_start(self, *args, **kwargs):
        """AsyncLLM.generate, first writing request_start before the request enters the engine."""
        engine_log_file = os.environ.get(ENGINE_LOG_ENV)
        if engine_log_file:
            os.makedirs(os.path.dirname(os.path.abspath(engine_log_file)), exist_ok=True)
            _write(engine_log_file, "request_start", request_id=generate_signature.bind(self, *args, **kwargs).arguments["request_id"])
        outputs = generate(self, *args, **kwargs)
        try:
            async for output in outputs:
                yield output
        finally:
            await outputs.aclose()

    def step_with_start(self):
        """EngineCore.step, first writing step_start if there is anything to schedule."""
        core_log_file = os.environ.get(CORE_LOG_ENV)
        if core_log_file and self.scheduler.has_requests():
            _write(core_log_file, "step_start")
        return step(self)

    def step_with_batch_queue_and_start(self):
        """EngineCore.step_with_batch_queue, first writing step_start."""
        core_log_file = os.environ.get(CORE_LOG_ENV)
        if core_log_file:
            _write(core_log_file, "step_start")
        return step_with_batch_queue(self)

    def update_from_output_with_step_end(self, scheduler_output, model_runner_output):
        """Scheduler.update_from_output, then writing step_end with the step's scheduled requests, KV cache usage and outputs."""
        engine_core_outputs = update_from_output(self, scheduler_output, model_runner_output)
        core_log_file = os.environ.get(CORE_LOG_ENV)
        if core_log_file:
            outputs = [output for core_outputs in engine_core_outputs.values() for output in core_outputs.outputs]
            _write(
                core_log_file,
                "step_end",
                executed_request_ids=list(scheduler_output.num_scheduled_tokens),
                kv_cache_usage=self.kv_cache_manager.usage,
                # kv_cache_usage=Scheduler.get_kv_cache_usage(),
                finished_request_ids=[output.request_id for output in outputs],
                finish_reasons=[str(output.finish_reason) for output in outputs],
                num_cached_tokens=[output.prefill_stats.num_cached_tokens if output.prefill_stats else None for output in outputs],
                # num_cached_tokens=[output.prefill_stats.num_cached_tokens for output in outputs],
                # num_cached_tokens=[output.num_cached_tokens for output in outputs],
                # num_cache_creation_tokens=[output.num_cache_creation_tokens for output in outputs],
                out_count=len(outputs),
            )
        return engine_core_outputs

    EngineArgs.add_cli_args = staticmethod(add_log_file_args)
    EngineArgs.from_cli_args = classmethod(from_cli_args_with_logs)
    AsyncLLM.generate = generate_with_request_start
    EngineCore.step = step_with_start
    EngineCore.step_with_batch_queue = step_with_batch_queue_and_start
    Scheduler.update_from_output = update_from_output_with_step_end
