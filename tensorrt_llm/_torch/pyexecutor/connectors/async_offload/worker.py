"""Async KV Cache Offload Worker — handles actual D2H/H2D transfers on dedicated CUDA streams."""

import logging
import os
from typing import Dict, List, Set, Tuple

import torch
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

logger = logging.getLogger(__name__)

_shared_state: dict = {}


class AsyncOffloadWorker(KvCacheConnectorWorker):

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        cpu_cache_gb = int(os.environ.get('ASYNC_OFFLOAD_CPU_GB', '200'))
        self.cpu_cache_bytes = cpu_cache_gb * (1 << 30)

        self.gpu_kv_cache: torch.Tensor = None
        self.cpu_kv_cache: torch.Tensor = None
        self.d2h_stream: torch.cuda.Stream = None
        self.h2d_stream: torch.cuda.Stream = None

        self.saving_events: Dict[int, torch.cuda.Event] = {}
        self.loading_events: Dict[int, torch.cuda.Event] = {}
        self.tracked_saving: Set[int] = set()
        self.tracked_loading: Set[int] = set()

        self.h2d_pending_event: torch.cuda.Event = None
        self.per_block_bytes: int = 0

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        self.gpu_kv_cache = kv_cache_tensor
        device = kv_cache_tensor.device

        num_gpu_blocks = kv_cache_tensor.shape[0]
        block_shape = list(kv_cache_tensor.shape[1:])
        self.per_block_bytes = kv_cache_tensor[0].numel() * kv_cache_tensor.element_size()

        num_cpu_blocks = self.cpu_cache_bytes // self.per_block_bytes

        logger.info(
            f"AsyncOffloadWorker: GPU blocks={num_gpu_blocks}, "
            f"CPU blocks={num_cpu_blocks}, "
            f"per_block={self.per_block_bytes} bytes, "
            f"shape={block_shape}, dtype={kv_cache_tensor.dtype}"
        )

        try:
            self.cpu_kv_cache = torch.empty(
                [num_cpu_blocks] + block_shape,
                dtype=kv_cache_tensor.dtype,
                device='cpu',
            ).pin_memory()
        except Exception:
            logger.warning("Failed to allocate pinned fp8 CPU memory, using uint8 view")
            total_elements = num_cpu_blocks
            for s in block_shape:
                total_elements *= s
            self.cpu_kv_cache = torch.empty(
                total_elements, dtype=torch.uint8, device='cpu'
            ).pin_memory().view([num_cpu_blocks] + block_shape)

        self.d2h_stream = torch.cuda.Stream(device=device)
        self.h2d_stream = torch.cuda.Stream(device=device)

        _shared_state['num_cpu_blocks'] = num_cpu_blocks
        _shared_state['per_block_bytes'] = self.per_block_bytes
        _shared_state['num_gpu_blocks'] = num_gpu_blocks
        _shared_state['worker_ready'] = True

        logger.info(
            f"AsyncOffloadWorker: CPU cache allocated "
            f"({num_cpu_blocks} blocks, "
            f"{num_cpu_blocks * self.per_block_bytes / (1 << 30):.1f} GB)"
        )

    def register_forward_pass_callable(self):
        return None

    def bind_connector_meta(self, metadata):
        super().bind_connector_meta(metadata)

    def start_load_kv(self, stream: torch.cuda.Stream):
        meta = self._metadata
        if meta is None:
            return

        stores = meta.get('stores', [])
        if stores:
            self.d2h_stream.wait_stream(stream)
            with torch.cuda.stream(self.d2h_stream):
                for req_id, gpu_bids, cpu_bids in stores:
                    for gpu_bid, cpu_bid in zip(gpu_bids, cpu_bids):
                        self.cpu_kv_cache[cpu_bid].copy_(
                            self.gpu_kv_cache[gpu_bid], non_blocking=True)
                event = self.d2h_stream.record_event()
            for req_id, _, _ in stores:
                self.saving_events[req_id] = event

        loads = meta.get('loads', [])
        if loads:
            with torch.cuda.stream(self.h2d_stream):
                for req_id, gpu_bids, cpu_bids in loads:
                    for gpu_bid, cpu_bid in zip(gpu_bids, cpu_bids):
                        self.gpu_kv_cache[gpu_bid].copy_(
                            self.cpu_kv_cache[cpu_bid], non_blocking=True)
                self.h2d_pending_event = self.h2d_stream.record_event()
            for req_id, _, _ in loads:
                self.loading_events[req_id] = self.h2d_pending_event

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        if layer_idx == 0 and self.h2d_pending_event is not None:
            stream.wait_event(self.h2d_pending_event)
            self.h2d_pending_event = None

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        pass

    def wait_for_save(self, stream: torch.cuda.Stream):
        pass

    def get_finished(
        self,
        finished_gen_req_ids: List[int],
        started_loading_req_ids: List[int],
    ) -> Tuple[List[int], List[int]]:
        for req_id in finished_gen_req_ids:
            self.tracked_saving.add(req_id)
        for req_id in started_loading_req_ids:
            self.tracked_loading.add(req_id)

        finished_saving = []
        finished_loading = []

        for req_id in list(self.tracked_saving):
            if req_id in self.saving_events:
                if self.saving_events[req_id].query():
                    finished_saving.append(req_id)
                    del self.saving_events[req_id]
                    self.tracked_saving.discard(req_id)

        for req_id in list(self.tracked_loading):
            if req_id in self.loading_events:
                if self.loading_events[req_id].query():
                    finished_loading.append(req_id)
                    del self.loading_events[req_id]
                    self.tracked_loading.discard(req_id)

        return finished_saving, finished_loading
