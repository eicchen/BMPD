"""LMCache MP connector with per-request hold and store flags from kv_transfer_params."""

import time

from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.utils import init_logger

MARKER = "conserve_pd"
HOLD_TIMEOUT = 5.0
LOOKUP_INTERVAL = 0.02

logger = init_logger(__name__)


class LMCacheAdaptiveConnector(LMCacheMPConnector):
    """Stock LMCacheMPConnector plus hold/store flags, unconsumed-lookup lock release and two worker report fixes."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        """Create the stock connector and empty per-request flag state."""
        super().__init__(vllm_config, role, kv_cache_config)
        self._holds = {}
        self._released = set()
        self._no_store = set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Hold a flagged request until LMCache covers its prompt's full chunks, then run the stock lookup."""
        rid = request.request_id
        tracker = self._get_or_create_request_tracker(request)
        flags = (request.kv_transfer_params or {}).get(MARKER, {})
        if not flags.get("store", True):
            self._no_store.add(rid)
        if not flags.get("hold") or rid in self._released:
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        adapter = self.scheduler_adapter
        chunk = adapter.lmcache_tokens_per_chunk
        target = (request.num_prompt_tokens - 1) // chunk * chunk
        now = time.monotonic()
        hold = self._holds.setdefault(rid, [now, now - LOOKUP_INTERVAL])
        if num_computed_tokens >= target:
            self._end_hold(rid, "released", target, now)
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        if not adapter.is_healthy:
            self._end_hold(rid, "unhealthy", target, now)
            return 0, False
        hit = adapter.check_lookup_result(rid)
        if hit is None:
            return None, False
        if hit >= target:
            self._end_hold(rid, "released", target, now)
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        if hit:
            self._free_locks(rid, hit)
        adapter.cleanup_lookup_result(rid)
        if now - hold[0] >= HOLD_TIMEOUT:
            self._end_hold(rid, "timeout", target, now)
            return 0, False
        if now - hold[1] >= LOOKUP_INTERVAL:
            hold[1] = now
            adapter.maybe_submit_lookup_request(
                rid,
                token_ids=tracker.get_token_ids(),
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )
        return None, False

    def _end_hold(self, rid, outcome, target, now):
        """Log how rid's hold ended and stop holding it."""
        logger.info("[conserve_pd] hold %s req=%s target=%d waited=%.3fs", outcome, rid, target, now - self._holds.pop(rid)[0])
        self._released.add(rid)

    def _free_locks(self, rid, end):
        """Release the read locks rid's lookup holds on tokens [0, end)."""
        tracker = self._get_request_tracker(rid)
        self.scheduler_adapter.free_lookup_locks(
            token_ids=tracker.get_token_ids(),
            start=0,
            end=end,
            request_id=rid,
            cache_salt=tracker.cache_salt,
            request_configs=tracker.request_configs,
        )

    def build_connector_meta(self, scheduler_output):
        """Build the stock metadata without STORE entries of store: false requests."""
        metadata = super().build_connector_meta(scheduler_output)
        metadata.requests = [m for m in metadata.requests if m.direction != "STORE" or m.request_id not in self._no_store]
        return metadata

    def request_finished(self, request, block_ids):
        """Release the locks of an unconsumed lookup, then finish the request the stock way."""
        rid = request.request_id
        self._holds.pop(rid, None)
        self._released.discard(rid)
        self._no_store.discard(rid)
        while (hit := self.scheduler_adapter.check_lookup_result(rid)) is None:
            time.sleep(0.001)
        if hit:
            self._free_locks(rid, hit)
        return super().request_finished(request, block_ids)

    def wait_for_save(self):
        """Forget earlier finished stores of requests storing this step, then submit the stock stores."""
        self.worker_adapter.finished_stores.difference_update(
            {m.request_id for m in self._get_connector_metadata().requests if m.direction == "STORE"},
        )
        super().wait_for_save()

    def get_finished(self, finished_req_ids):
        """Stock finished sets, with requests whose load is reported now or still in flight dropped from sending."""
        sending, recving = super().get_finished(finished_req_ids)
        return sending.difference(recving, self.worker_adapter.retrieve_futures), recving
