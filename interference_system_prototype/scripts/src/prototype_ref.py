"""Conversation prototype over any set of kv_producer and kv_consumer GPUs sharing one LMCache server."""

import asyncio
from collections import deque

from system import client, complete, converse, kv_capacity


class _Ledger:
    """Free KV tokens of each GPU, granted to waiters oldest first, each to the roomiest GPU of its set that fits."""

    def __init__(self, room):
        self.room = room
        self.queue = []

    def ask(self, need, gpus):
        """Future of the GPU in gpus that reserves need tokens for the caller."""
        fut = asyncio.get_running_loop().create_future()
        self.queue.append((need, gpus, fut))
        self._grant()
        return fut

    def release(self, gpu, need):
        """Return need tokens to gpu."""
        self.room[gpu] += need
        self._grant()

    def _grant(self):
        """Grant waiters oldest first, each that does not fit blocking its GPUs for later ones, dropping cancelled ones."""
        blocked, queue = set(), []
        for need, gpus, fut in self.queue:
            if fut.cancelled():
                continue
            gpu = max((g for g in gpus if g not in blocked), key=self.room.get, default=None)
            if gpu is not None and self.room[gpu] >= need:
                self.room[gpu] -= need
                fut.set_result(gpu)
            else:
                blocked.update(gpus)
                queue.append((need, gpus, fut))
        self.queue = queue

async def _conversation(clients, producers, consumers, ledger, turns):
    """Serve a conversation's turns on a producer, each moving to a consumer once one has KV room, and every turn after a move whole on that consumer; return their rows."""
    gpu = None
    # TEMP per-conversation S2 room begin: the conversation's peak context, held on its consumer from its move to its end
    peak = sum((r["in_token_size"] if isinstance(r["prompt"], str) else len(r["ids"])) + r["max_tokens"] for r in turns)
    # TEMP per-conversation S2 room end

    async def serve(rec):
        nonlocal gpu
        need = rec["prompt_size"] + rec["max_tokens"]
        if gpu is not None:
            # TEMP per-conversation S2 room begin: already held
            # await ledger.ask(need, [gpu])
            gpu_request = await complete(clients[gpu], rec, params={"store": False}, token_ids=True)
            # ledger.release(gpu, need)
            # TEMP per-conversation S2 room end
            ids, text = gpu_request.pop("token_ids"), gpu_request.pop("text")
            return {"id": rec["id"], "gpu_requests": [{"gpu": gpu, **gpu_request}]}, ids, text
        producer = await ledger.ask(need, producers)
        grant = None

        def ask():
            nonlocal grant
            # TEMP per-conversation S2 room begin
            # grant = ledger.ask(need, consumers)
            grant = ledger.ask(peak, consumers)
            # TEMP per-conversation S2 room end
            return grant

        first = await complete(clients[producer], rec, token_ids=True, handoff=ask)
        ids, text = first.pop("token_ids"), first.pop("text")
        gpu_requests = [{"gpu": producer, **first}]
        k = first["completion_tokens"]
        moved = grant.done() and k < rec["max_tokens"]
        if not grant.done():
            grant.cancel()
        elif not moved:
            # TEMP per-conversation S2 room begin
            # ledger.release(grant.result(), need)
            ledger.release(grant.result(), peak)
            # TEMP per-conversation S2 room end
        ledger.release(producer, need)
        if moved:
            gpu = grant.result()
            second = await complete(
                clients[gpu],
                {**rec, "prompt": ids},
                max_tokens=rec["max_tokens"] - k,
                params={"hold": True, "store": False},
                token_ids=True,
            )
            # TEMP per-conversation S2 room begin: held until the conversation ends
            # ledger.release(gpu, need)
            # TEMP per-conversation S2 room end
            ids = second.pop("token_ids")
            text += second.pop("text")
            gpu_requests.append({"gpu": gpu, **second})
        return {"id": rec["id"], "gpu_requests": gpu_requests}, ids, text

    # TEMP per-conversation S2 room begin: released when the conversation ends
    # return await converse(turns, serve)
    rows = await converse(turns, serve)
    if gpu is not None:
        ledger.release(gpu, peak)
    return rows
    # TEMP per-conversation S2 room end


async def replay(recs, engines):
    """Serve every conversation concurrently in file order, its turns in order, under one KV ledger over every GPU; return their rows."""
    producers = [gpu for gpu, role in engines.items() if role == "kv_producer"]
    consumers = [gpu for gpu, role in engines.items() if role == "kv_consumer"]
    ledger = _Ledger({gpu: kv_capacity(gpu) for gpu in engines})
    clients = {gpu: client(gpu) for gpu in engines}
    try:
        convs = await asyncio.gather(
            *(_conversation(clients, producers, consumers, ledger, turns) for turns in recs),
        )
        return [row for rows in convs for row in rows]
    finally:
        for c in clients.values():
            await c.close()
