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

## Step 6: Hardening fixes (v1.1)

Applied three fixes identified during code review (REPLICATION_NOTES.md issues 4, 5, 6):

1. **`tokens_per_block` default** (leader.py): Added `or 32` fallback for defensive coding
2. **Redundant `bind_connector_meta`** (worker.py): Removed dead override that only called super()
3. **Narrow exception catch** (worker.py): Changed `except Exception` to `except RuntimeError`, re-raise OOM

### Test Results — v1.0 Baseline vs v1.1 Regression (QPS 6, 10 rounds)

| Metric | v1.0 | v1.1 |
|--------|------|------|
| Rounds completed | 9/10 (R10 crashed) | **10/10** |
| Success rate | 100% (R1-R9) | **100% (all rounds)** |
| P50 TTFT (R1) | 6.12s | 3.49s |
| P50 TTFT (R5) | 12.85s | 9.53s |
| P50 TTFT (R9) | 18.27s | 19.50s |
| P50 E2E (R1) | 14.94s | 12.58s |
| P50 E2E (R5) | 22.69s | 21.50s |
| P50 E2E (R9) | 27.61s | 28.49s |
| P50 TPOT (R1) | 49.57ms | 50.67ms |
| P50 TPOT (R9) | 52.73ms | 50.00ms |

**Key findings:**
- v1.1 survived all 10 rounds vs v1.0 crashing at round 10
- Early-round latency is improved in v1.1 (lower TTFT)
- Late-round latency converges to similar values
- No functional regressions observed

Docker images:
- `trtllm-async-offload:v1.0` — original
- `trtllm-async-offload:v1.1` — with hardening fixes
