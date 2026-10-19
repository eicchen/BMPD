"""Conversation variants of the disaggregated and colocated baselines: each conversation's turns in order on GPUs pinned to it, each prompt carrying its history."""

import asyncio
from functools import partial

from aiohttp import web

from system import TELEMETRY_PORT, client, complete, converse


class Disagg:
    """Disaggregated conversation replay: each turn one token on the conversation's kv_producer, then, once LMCache reports its KV stored, whole on its kv_consumer from that KV."""

    @staticmethod
    async def serve(clients, producer, consumer, stored, rec):
        first = await complete(clients[producer], rec, max_tokens=1)
        await stored[f"cmpl-{rec['id']}-0"].wait()
        second = await complete(clients[consumer], rec, token_ids=True)
        ids, text = second.pop("token_ids"), second.pop("text")
        return {
            "id": rec["id"],
            "gpu_requests": [
                {"gpu": producer, **first},
                {"gpu": consumer, **second},
            ],
        }, ids, text

    @staticmethod
    async def replay(recs, engines):
        producers = [gpu for gpu, role in engines.items() if role == "kv_producer"]
        consumers = [gpu for gpu, role in engines.items() if role == "kv_consumer"]
        stored = {f"cmpl-{rec['id']}-0": asyncio.Event() for turns in recs for rec in turns}

        async def telemetry(request):
            for rid in (await request.json())["request_ids_set"]:
                stored[rid.rsplit("-", 1)[0]].set()
            return web.json_response({})

        app = web.Application()
        app.router.add_post("/api/v1/telemetry", telemetry)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "localhost", TELEMETRY_PORT).start()
        clients = {gpu: client(gpu) for gpu in engines}
        try:
            tasks = []
            for i, turns in enumerate(recs):
                producer = producers[i % len(producers)]
                consumer = consumers[i % len(consumers)]
                tasks.append(converse(turns, partial(Disagg.serve, clients, producer, consumer, stored)))
            results = await asyncio.gather(*tasks)
            return [row for rows in results for row in rows]
        finally:
            for c in clients.values():
                await c.close()
            await runner.cleanup()


class Colocated:
    """Colocated conversation replay: each turn whole on the conversation's GPU."""

    @staticmethod
    async def serve(clients, gpu, rec):
        gpu_request = await complete(clients[gpu], rec, token_ids=True)
        ids, text = gpu_request.pop("token_ids"), gpu_request.pop("text")
        return {"id": rec["id"], "gpu_requests": [{"gpu": gpu, **gpu_request}]}, ids, text

    @staticmethod
    async def replay(recs, engines):
        gpus = list(engines)
        clients = {gpu: client(gpu) for gpu in gpus}
        try:
            tasks = []
            for i, turns in enumerate(recs):
                gpu = gpus[i % len(gpus)]
                tasks.append(converse(turns, partial(Colocated.serve, clients, gpu)))
            results = await asyncio.gather(*tasks)
            return [row for rows in results for row in rows]
        finally:
            for c in clients.values():
                await c.close()
