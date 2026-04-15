# Async KV Cache Offload Connector -- Replication Log

This log documents every step taken while replicating the async KV cache offload
connector from Docker image `trtllm-async-offload:v1.0` into this forked repo.

**Source**: Docker image `trtllm-async-offload:v1.0` (built on TRT-LLM 1.3.0rc11, NVIDIA PyTorch 26.02)
**Target**: Forked repo branch `feat/async-kv-offload-connector`, base commit `51f795617`

---

## Step 0: Branch creation and documentation init

- Created branch `feat/async-kv-offload-connector` from `main` at `51f795617`
- Created `REPLICATION_LOG.md` (this file) and `REPLICATION_NOTES.md`
- The Docker image adds 6 files under `/workspace/async_offload/` plus an `engine_config.yaml`.
  No upstream TRT-LLM source files were modified in the image.
- Key finding: the connector interface import path changed between the image's
  TRT-LLM version (`tensorrt_llm._torch.pyexecutor.kv_cache_connector`) and the
  current main branch (`tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector`).
  All imports will be adapted accordingly.
- Placement: files go under `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/`
  to match the repo's connector organization pattern.

## Step 1: Commit 1 -- Core CPU block pool module

- Created `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/__init__.py`
- Created `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/block_pool.py`
  Copied verbatim from Docker image. This module is a pure data structure (LRU
  eviction + hash-indexed lookup) with no TRT-LLM dependencies, so no import
  changes needed.

## Step 2: Commit 2 -- Async offload worker

- Created `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/worker.py`
- Adapted import path:
  `from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorWorker`
  changed to:
  `from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorWorker`
- Functionality unchanged: pinned memory allocation, dual CUDA stream (d2h/h2d),
  deferred D2H submission, non-blocking CUDA event polling for completion detection.

## Step 3: Commit 3 -- Leader + connector entry point

- Created `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/leader.py`
- Adapted import path:
  `from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (KvCacheConnectorScheduler, SchedulerOutput)`
  changed to:
  `from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (KvCacheConnectorScheduler, SchedulerOutput)`
- Also changed worker import in leader.py from `from .worker import _shared_state`
  (unchanged, relative import still valid).
- Created `tensorrt_llm/_torch/pyexecutor/connectors/async_offload/connector.py`
  as module entry point re-exporting AsyncOffloadLeader and AsyncOffloadWorker.

## Step 4: Commit 4 -- Registry integration + example config

- Added `"async_offload"` entry to `tensorrt_llm/_torch/pyexecutor/connectors/registry.py`
- Created `examples/apps/async_offload_config.yaml` as reference configuration

## Step 5: Push to remote

- Pushed branch `feat/async-kv-offload-connector` to origin
