import asyncio
from collections import deque, defaultdict

from system import client, complete, converse, kv_capacity

"""
Handles the routing and pinning of conversations to gpus based on kv usage
WIP, to be finished later
Still needs KV monitoring, conversation pinning, convresation routing
"""
class _Router:
    """
    Handles queuing, pinning of conversation and routing of subsequent turns
    """
    def __init__(self,gpu_mem_dict):
        self.gpu_mem_dict = gpu_mem_dict #dict of gpu:kv_capacity
        self.input_queue = deque()
        self.conv_gpu_pin = defaultdict(list) # formatted as {gpu:[conv_id]}
        
    #? queue can be list of conversation or conv_id
    def _insert_queue(self, conv2queue):
        if not isinstance(conv2queue, (list, tuple)):
            conv2queue = list(conv2queue)
        self.input_queue.extend(conv2queue)
        
    def _pin_conv(self, gpu, conv_id):
        self.conv_gpu_pin[gpu].append(conv_id)
        
    def _unpin_conv(self, gpu, conv_id):
        self.conv_gpu_pin[gpu].remove(conv_id)
        
        
async def replay(recs, engines):
    router = _Router({gpu: kv_capacity(gpu) for gpu in engines})
    
    
async def test(recs, fake_gpu):
    router = _Router(fake_gpu)
    router._insert_queue([0,1,2,3])
    print(router.input_queue)
    router.input_queue.popleft()
    print(router.input_queue)
    
    router._pin_conv(0, 2)
    router._pin_conv(0, 1)
    
    print(router.conv_gpu_pin)
    
    router._unpin_conv(0,1)
    print(router.conv_gpu_pin)

if __name__ == "__main__":
    from pathlib import Path

    import sys
    for d in ("src", "output", "helpers"):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / d))
        
    from load import load
    
    fake_gpu = {0:200000, 1:200000}
    # asyncio.run(test(load(convs=5, turns=3, tokenize=False), fake_gpu))
    asyncio.run(test("test", fake_gpu))