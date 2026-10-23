"""vLLM scheduler that starts waiting requests strictly in priority order, loaded with --scheduler-cls in_order_scheduler.InOrderScheduler."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class InOrderScheduler(AsyncScheduler):
    """AsyncScheduler that holds every later waiting request in a step once one isn't ready, instead of skipping past it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._held = False
        get_num_new_matched_tokens = self.connector.get_num_new_matched_tokens

        def in_order_matched_tokens(request, num_computed_tokens):
            """The connector's matched tokens, or None to hold request if it would start computing behind one that isn't ready."""
            ext_tokens, load_kv_async = get_num_new_matched_tokens(request, num_computed_tokens)
            if ext_tokens is None:
                self._held = True
            elif self._held and not load_kv_async:
                return None, False
            return ext_tokens, load_kv_async

        self.connector.get_num_new_matched_tokens = in_order_matched_tokens

    def schedule(self):
        """Schedule a step with no request held yet."""
        self._held = False
        return super().schedule()

    def _try_promote_blocked_waiting_request(self, request):
        """Promote request once its KV has loaded, unless a request ahead of it in this step isn't ready."""
        if self._held:
            return False
        promoted = super()._try_promote_blocked_waiting_request(request)
        self._held = not promoted
        return promoted
