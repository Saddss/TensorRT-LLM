# Async KV Cache Offload Connector -- Replication Notes

Observations and potential issues found during the replication process.
These are recorded for future review and are NOT addressed during replication
to avoid introducing unintended changes.

---

## 1. Import Path Change (adapted during replication)

**Original** (Docker image, TRT-LLM 1.3.0rc11):
```python
from tensorrt_llm._torch.pyexecutor.kv_cache_connector import KvCacheConnectorScheduler
```

**Adapted** (current main branch):
```python
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import KvCacheConnectorScheduler
```

The connector interface was reorganized into a `connectors/` subdirectory between
the version in the Docker image and the current main branch. This is a straightforward
rename with no API changes observed in the base classes.

## 2. `_shared_state` Global Dict for Leader-Worker Communication

`worker.py` defines a module-level `_shared_state: dict = {}` that the Worker
populates with `num_cpu_blocks`, `per_block_bytes`, etc. after `register_kv_caches`.
The Leader imports this dict and reads from it in `wait_for_initialization()`.

**Concern**: This is a process-global mutable dict. If multiple connector instances
were created (e.g., in a multi-model scenario), they would silently overwrite each
other's state. The existing kvbm connector in TRT-LLM uses a different IPC mechanism.

**Risk level**: Low for single-model deployments (the intended use case), but would
need refactoring for multi-model.

## 3. `register_forward_pass_callable` Returns None

The Worker's `register_forward_pass_callable` returns `None`, meaning no callback
is registered into the forward pass CUDA stream. This is technically valid per the
interface contract, but it means D2H (GPU->CPU save) operations are not triggered
at the end of the forward pass. Instead, they are submitted at the beginning of the
*next* iteration's `start_load_kv` call.

**Implication**: There is one iteration of latency before D2H transfers begin. In
practice, this is acceptable because the transfers are fully async on a dedicated
stream, and the deferred design avoids blocking the current iteration's forward pass.

## 4. No `tokens_per_block` Default Handling

In `leader.py`:
```python
self.tokens_per_block = llm_args.kv_cache_config.tokens_per_block
```

If `tokens_per_block` is not explicitly set in the KV cache config, this could be
`None`, which would cause a `TypeError` when used as a divisor/range step in
`_compute_block_hashes`. The TRT-LLM default is typically 64, but the connector
does not fall back to this default.

## 5. Redundant `bind_connector_meta` Override

In `worker.py`:
```python
def bind_connector_meta(self, metadata):
    super().bind_connector_meta(metadata)
```

This override does nothing beyond calling the parent implementation. It could be
removed without any behavioral change. Kept as-is during replication to match the
original code exactly.

## 6. fp8 Pinned Memory Fallback via uint8 View

In `worker.py`, when allocating pinned CPU memory for fp8 dtypes:
```python
except Exception:
    logger.warning("Failed to allocate pinned fp8 CPU memory, using uint8 view")
    total_elements = num_cpu_blocks
    for s in block_shape:
        total_elements *= s
    self.cpu_kv_cache = torch.empty(
        total_elements, dtype=torch.uint8, device='cpu'
    ).pin_memory().view([num_cpu_blocks] + block_shape)
```

This catches *any* exception (not just dtype-related ones, e.g., OOM would also
be caught and silently retried). The uint8 view also assumes byte-level compatibility
between fp8 and uint8 representations, which works for raw memory copies but could
be fragile if tensor operations (rather than raw copies) are performed on the CPU cache.

---

## Resolution Status (v1.1)

| Issue | Status | Action |
|-------|--------|--------|
| 1. Import path | Already adapted | No change needed |
| 2. `_shared_state` global | Accepted risk | Low risk for single-model use case |
| 3. `register_forward_pass_callable` | Design choice | Intentional one-iteration delay |
| 4. `tokens_per_block` default | **Fixed in v1.1** | Added `or 32` fallback |
| 5. Redundant override | **Fixed in v1.1** | Removed dead code |
| 6. Broad exception catch | **Fixed in v1.1** | Narrowed to `RuntimeError`, re-raise OOM |

Verified via QPS 6 regression testing: v1.1 completed 10/10 rounds at 100% success
vs v1.0's 9/10 (crashed at round 10).
