"""Async KV Cache Offload Leader — scheduler-side logic for prefix matching and save/load orchestration."""

import logging
from typing import Dict, List, Tuple

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    KvCacheConnectorScheduler,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from .block_pool import CpuBlockPool
from .worker import _shared_state

logger = logging.getLogger(__name__)


class AsyncOffloadLeader(KvCacheConnectorScheduler):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        self.tokens_per_block = llm_args.kv_cache_config.tokens_per_block
        self.block_pool: CpuBlockPool = None

        self.request_hashes: Dict[int, List[int]] = {}
        self.request_gpu_blocks: Dict[int, List[int]] = {}

        self.pending_saves: list = []
        self.pending_loads: Dict[int, dict] = {}

    def wait_for_initialization(self):
        num_cpu_blocks = _shared_state.get('num_cpu_blocks', 0)
        if num_cpu_blocks == 0:
            raise RuntimeError(
                "AsyncOffloadWorker must be initialized before leader!"
            )

        self.block_pool = CpuBlockPool(num_cpu_blocks)
        logger.info(
            f"AsyncOffloadLeader: tokens_per_block={self.tokens_per_block}, "
            f"cpu_blocks={num_cpu_blocks}"
        )

    def _compute_block_hashes(self, tokens) -> List[int]:
        """Compute position-dependent hash chain for token blocks."""
        hashes = []
        prev_hash = 0
        tpb = self.tokens_per_block
        for i in range(0, len(tokens), tpb):
            block_tokens = tuple(tokens[i:i + tpb])
            if len(block_tokens) < tpb:
                break
            h = hash((prev_hash, block_tokens))
            hashes.append(h)
            prev_hash = h
        return hashes

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        if self.block_pool is None:
            return (0, False)

        tokens = list(request.get_tokens(0))
        block_hashes = self._compute_block_hashes(tokens)

        self.request_hashes[request.request_id] = block_hashes

        tpb = self.tokens_per_block
        if num_computed_tokens % tpb != 0:
            start_block = num_computed_tokens // tpb + 1
        else:
            start_block = num_computed_tokens // tpb

        matched = self.block_pool.find_prefix_match(block_hashes, start_block)

        if matched > 0:
            matched_tokens = matched * tpb
            self.pending_loads[request.request_id] = {
                'start_block': start_block,
                'num_blocks': matched,
                'block_hashes': block_hashes[start_block:start_block + matched],
            }
            return (matched_tokens, True)

        return (0, False)

    def update_state_after_alloc(
        self, request: LlmRequest, block_ids: List[int]
    ):
        self.request_gpu_blocks[request.request_id] = list(block_ids)

        if request.request_id in self.pending_loads:
            load_info = self.pending_loads[request.request_id]
            start = load_info['start_block']
            count = load_info['num_blocks']
            if start + count <= len(block_ids):
                load_info['gpu_block_ids'] = block_ids[start:start + count]
            else:
                logger.warning(
                    f"Not enough GPU blocks for request {request.request_id}: "
                    f"need [{start}:{start+count}], have {len(block_ids)}"
                )
                del self.pending_loads[request.request_id]

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        meta = {'loads': [], 'stores': []}

        for req_id, load_info in list(self.pending_loads.items()):
            if 'gpu_block_ids' not in load_info:
                continue
            gpu_bids = load_info['gpu_block_ids']
            cpu_bids = []
            valid = True
            for h in load_info['block_hashes']:
                cpu_bid = self.block_pool.lookup(h)
                if cpu_bid is not None:
                    cpu_bids.append(cpu_bid)
                else:
                    logger.warning(
                        f"CPU block evicted before load: req={req_id}"
                    )
                    valid = False
                    break
            if valid:
                meta['loads'].append((req_id, gpu_bids, cpu_bids))
            del self.pending_loads[req_id]

        for save_info in self.pending_saves:
            meta['stores'].append(save_info)
        self.pending_saves.clear()

        return meta

    def request_finished(
        self, request: LlmRequest, cache_block_ids: List[int]
    ) -> bool:
        req_id = request.request_id
        tokens = list(request.get_tokens(0))
        block_hashes = self._compute_block_hashes(tokens)

        gpu_blocks_to_save = []
        cpu_blocks_to_save = []

        num_full_blocks = min(len(block_hashes), len(cache_block_ids))
        for i in range(num_full_blocks):
            block_hash = block_hashes[i]
            existing = self.block_pool.lookup(block_hash)
            if existing is not None:
                self.block_pool.touch(block_hash)
            else:
                cpu_bid = self.block_pool.allocate(block_hash)
                gpu_blocks_to_save.append(cache_block_ids[i])
                cpu_blocks_to_save.append(cpu_bid)

        self.request_hashes.pop(req_id, None)
        self.request_gpu_blocks.pop(req_id, None)

        if gpu_blocks_to_save:
            self.pending_saves.append(
                (req_id, gpu_blocks_to_save, cpu_blocks_to_save)
            )
            return True

        return False
