# vLLM modifications

BMPD runs on the stock vLLM 0.23.0+cu129 wheel. Everything it changes in vLLM lives here and is loaded at runtime, so the changes are tracked in git and survive a vLLM reinstall.

Install once into the env that runs vLLM:

    pip install -e modifications/vllm --no-deps --no-build-isolation

This registers `vllm_tracing` as a vLLM general plugin and makes both modules importable in every vLLM process.

## vllm_tracing.py — request and engine-step event logs

A vLLM general plugin (entry point group `vllm.general_plugins`). vLLM runs its `register()` in every process it starts: the API server while it builds its argument parser, the engine core and the workers. It does nothing unless the flags below are given.

It adds two `vllm serve` flags:

- `--engine-log-file PATH`: the API server appends a `request_start` event for every request, written just before the request is handed to the engine.
- `--core-log-file PATH`: the engine core replaces the file at launch, then writes a `step_start` event before every scheduler step that has requests, and a `step_end` event after the step's outputs are processed.

The paths reach the engine core through the `BMPD_ENGINE_LOG_FILE` and `BMPD_CORE_LOG_FILE` environment variables, set when the flags are parsed.

Events, one JSON object per line, all with `event` and an ISO `timestamp`:

| event | extra fields |
|---|---|
| `request_start` | `request_id` |
| `step_start` | none |
| `step_end` | `executed_request_ids` (requests scheduled this step), `kv_cache_usage` (fraction of the GPU KV pool in use), `finished_request_ids` (every request with output this step, not only finished ones), `finish_reasons`, `num_cached_tokens`, `out_token_count` |

BMPD reads them in `interference_system_prototype/scripts/output/metric.py` (request arrival times from `request_start`; engine steps, the requests in each and KV usage from `step_start` and `step_end`), `interference_system_prototype/scripts/output/plot.py` and `profiling/experiments/interference/interference_metrics.py` (token emission times from `step_end`).

It wraps `EngineArgs.add_cli_args`, `EngineArgs.from_cli_args`, `AsyncLLM.generate`, `EngineCore.step`, `EngineCore.step_with_batch_queue` and `Scheduler.update_from_output`, so it is tied to vLLM 0.23.0's internals.

## in_order_scheduler.py — start waiting requests in priority order

`InOrderScheduler` extends vLLM's `AsyncScheduler`. Each step, stock vLLM skips past a waiting request that isn't ready (its LMCache lookup hasn't returned, or its KV is still loading) and starts the next one, so requests start in whatever order their lookups and loads finish. `InOrderScheduler` instead holds every later request for that step, while still letting them send lookups and start loading KV. With `--scheduling-policy priority` and each request's priority set to its send index, requests start in send order.

Loaded with `vllm serve --scheduler-cls in_order_scheduler.InOrderScheduler`, which the interference experiments pass through `launch_engine(scheduler_cls=...)` in `interference_system_prototype/scripts/src/system.py`.

It overrides `schedule` and `_try_promote_blocked_waiting_request` and wraps its connector's `get_num_new_matched_tokens`, so it is tied to vLLM 0.23.0's internals.

## Removed in-place edits

Until 2026-10-21 the env's vLLM carried conserve's in-place source edits. BMPD only used their tracing, now `vllm_tracing.py`. The full diff, including the parts BMPD dropped (fixed-batch `LLM.generate`, `LMCacheConnectorV1` `SupportsHMA`, `/tmp/decoder_forward_start_time.csv`), is kept in the conserve repo at `patches/vllm-0.23.0-inplace.diff`.
