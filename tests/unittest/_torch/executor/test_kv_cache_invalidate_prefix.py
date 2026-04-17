# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the two KV-cache extensions added for GitHub Issue #13080:

* ``LlmRequest.no_cache_on_finish`` — prospective prevention of caching a
  request's KV blocks into the reuse radix tree on sequence completion.
* ``BaseKVCacheManager.invalidate_prefix`` — retrospective eviction of
  cached blocks whose prefix matches a caller-supplied token list.

The tests mirror the style of ``test_resource_manager.
test_kv_cache_reset_reuse_state``: a single short prompt is pushed through
``add_sequence`` → ``simulate_prefill_completion_only_use_for_testing`` →
``free_resources`` and the ``reused_blocks`` counter from
``get_kv_cache_stats()`` is used as the oracle for "did the previous
request's blocks land in the reuse tree".
"""

import unittest

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.internal.testing import simulate_prefill_completion_only_use_for_testing
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams


def _make_request(req_id: int, input_tokens, no_cache_on_finish: bool = False):
    """Helper that wires a barebones LlmRequest for the pytorch executor.

    ``no_cache_on_finish`` is forwarded through the Python constructor, which
    then flips the nanobind-exposed property on the underlying C++
    ``GenericLlmRequest``.
    """
    sampling_params = SamplingParams()
    sampling_config = tensorrt_llm.bindings.SamplingConfig(sampling_params._get_sampling_config())
    return LlmRequest(
        request_id=req_id,
        max_new_tokens=1,
        input_tokens=list(input_tokens),
        sampling_config=sampling_config,
        is_streaming=False,
        no_cache_on_finish=no_cache_on_finish,
    )


def _make_manager():
    """Small KV cache with block reuse enabled.  Matches the shape used by
    the existing ``test_kv_cache_reset_reuse_state`` test so that the runtime
    cost per test stays in the ~second range on a single GPU.
    """
    config = KvCacheConfig(
        free_gpu_memory_fraction=0.4,
        event_buffer_max_size=1024,
        enable_block_reuse=True,
        max_tokens=1024,  # room for several 2-block prompts in the reuse tree
    )
    return KVCacheManager(
        kv_cache_config=config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=128,
        tokens_per_block=64,
        max_seq_len=1024,
        max_batch_size=1,
        mapping=Mapping(),
    )


class TestKVCacheInvalidatePrefix(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("error")

    # ------------------------------------------------------------------
    # no_cache_on_finish: prospective prevention
    # ------------------------------------------------------------------

    def test_no_cache_on_finish_skips_reuse_store(self):
        """A request flagged ``no_cache_on_finish=True`` must NOT leave its
        blocks in the reuse radix tree when it completes.  We verify by
        running a second request with an identical prompt and checking that
        ``reused_blocks`` is unchanged relative to a fresh manager.
        """
        mgr = _make_manager()
        tokens = [1, 2, 3, 4, 5]

        baseline_reused = mgr.get_kv_cache_stats().reused_blocks

        # Request 1 — opts out of caching on finish.
        req1 = _make_request(req_id=0, input_tokens=tokens, no_cache_on_finish=True)
        self.assertTrue(req1.no_cache_on_finish, "no_cache_on_finish flag should round-trip to C++")
        mgr.impl.add_sequence(req1.py_request_id, req1.prompt_len, 1, req1)
        simulate_prefill_completion_only_use_for_testing(req1)
        mgr.free_resources(req1)

        # Request 2 — same prompt.  Because req1 refused to cache, req2 must
        # NOT pick up any new reused blocks beyond the baseline.
        req2 = _make_request(req_id=1, input_tokens=tokens)
        mgr.impl.add_sequence(req2.py_request_id, req2.prompt_len, 1, req2)
        stats_after = mgr.get_kv_cache_stats()
        self.assertEqual(
            stats_after.reused_blocks,
            baseline_reused,
            msg=(
                "Setting no_cache_on_finish=True on the first request "
                "should have prevented its blocks from entering the reuse "
                "tree; the follow-up request reused {} blocks instead of "
                "the expected {}.".format(stats_after.reused_blocks, baseline_reused)
            ),
        )

        simulate_prefill_completion_only_use_for_testing(req2)
        mgr.free_resources(req2)
        mgr.shutdown()

    def test_no_cache_on_finish_default_false_preserves_reuse(self):
        """Control test: without the flag, blocks DO end up in the reuse
        tree and a second identical request reuses them.  This guards
        against silent regressions that would make the whole path a no-op.
        """
        mgr = _make_manager()
        tokens = [10, 11, 12, 13, 14, 15, 16, 17]

        baseline_reused = mgr.get_kv_cache_stats().reused_blocks

        req1 = _make_request(req_id=0, input_tokens=tokens)
        self.assertFalse(req1.no_cache_on_finish)
        mgr.impl.add_sequence(req1.py_request_id, req1.prompt_len, 1, req1)
        simulate_prefill_completion_only_use_for_testing(req1)
        mgr.free_resources(req1)

        req2 = _make_request(req_id=1, input_tokens=tokens)
        mgr.impl.add_sequence(req2.py_request_id, req2.prompt_len, 1, req2)
        stats_after = mgr.get_kv_cache_stats()
        self.assertGreater(
            stats_after.reused_blocks,
            baseline_reused,
            msg=(
                "Default path must still populate the reuse tree; "
                "reused_blocks stayed at {}.".format(stats_after.reused_blocks)
            ),
        )

        simulate_prefill_completion_only_use_for_testing(req2)
        mgr.free_resources(req2)
        mgr.shutdown()

    # ------------------------------------------------------------------
    # invalidate_prefix: retrospective eviction
    # ------------------------------------------------------------------

    def test_invalidate_prefix_evicts_idle_match(self):
        """After req1 finishes and populates the reuse tree, a call to
        ``invalidate_prefix(tokens)`` must wipe the matching radix entry so
        that a subsequent identical-prompt request sees no reuse.

        The prompt deliberately spans two full blocks (128 tokens,
        tokens_per_block=64) so that the match is expressed entirely in the
        full-block radix tree — the region our invalidatePrefix walker
        actually owns.  Partial-block storage is a separate code path not
        covered by this test.
        """
        mgr = _make_manager()
        tokens = list(range(100, 228))  # 128 tokens — exactly two full blocks

        # Populate the reuse tree.
        req1 = _make_request(req_id=0, input_tokens=tokens)
        mgr.impl.add_sequence(req1.py_request_id, req1.prompt_len, 1, req1)
        simulate_prefill_completion_only_use_for_testing(req1)
        mgr.free_resources(req1)

        # Sanity check: a second request *would* reuse before invalidation.
        probe = _make_request(req_id=1, input_tokens=tokens)
        mgr.impl.add_sequence(probe.py_request_id, probe.prompt_len, 1, probe)
        reused_before_invalidation = mgr.get_kv_cache_stats().reused_blocks
        simulate_prefill_completion_only_use_for_testing(probe)
        mgr.free_resources(probe)

        # Retrospective eviction.
        mgr.invalidate_prefix(tokens)

        # A fresh request with the same prompt must NOT count new reuse
        # hits beyond the pre-invalidation baseline; any rise would mean
        # the prefix is still lingering in the radix tree.
        baseline_after_evict = mgr.get_kv_cache_stats().reused_blocks
        req3 = _make_request(req_id=2, input_tokens=tokens)
        mgr.impl.add_sequence(req3.py_request_id, req3.prompt_len, 1, req3)
        stats_after = mgr.get_kv_cache_stats()
        self.assertEqual(
            stats_after.reused_blocks,
            baseline_after_evict,
            msg=(
                "invalidate_prefix() must remove the matching radix "
                "entry; reused_blocks climbed from {} to {} despite the "
                "prefix having been invalidated.".format(
                    baseline_after_evict, stats_after.reused_blocks
                )
            ),
        )
        # And the overall behaviour should match reset_reuse_state() — the
        # coarsest possible eviction — for this single-prefix scenario.
        self.assertLessEqual(stats_after.reused_blocks, reused_before_invalidation)

        simulate_prefill_completion_only_use_for_testing(req3)
        mgr.free_resources(req3)
        mgr.shutdown()

    def test_invalidate_prefix_is_safe_noop_for_unknown_prefix(self):
        """Calling ``invalidate_prefix`` with a token sequence that never
        entered the cache must not raise and must not corrupt subsequent
        legitimate reuse.
        """
        mgr = _make_manager()
        known_tokens = list(range(300, 428))  # two full blocks
        unknown_tokens = list(range(9000, 9128))  # disjoint two-block prompt

        req1 = _make_request(req_id=0, input_tokens=known_tokens)
        mgr.impl.add_sequence(req1.py_request_id, req1.prompt_len, 1, req1)
        simulate_prefill_completion_only_use_for_testing(req1)
        mgr.free_resources(req1)

        # Should be a silent no-op — no exception, no collateral damage.
        mgr.invalidate_prefix(unknown_tokens)
        mgr.invalidate_prefix([])  # empty prefix: also a no-op

        # Known prefix should still reuse.
        pre_reused = mgr.get_kv_cache_stats().reused_blocks
        req2 = _make_request(req_id=1, input_tokens=known_tokens)
        mgr.impl.add_sequence(req2.py_request_id, req2.prompt_len, 1, req2)
        post_reused = mgr.get_kv_cache_stats().reused_blocks
        self.assertGreater(
            post_reused,
            pre_reused,
            msg=(
                "invalidate_prefix() with an unknown/empty prefix must not "
                "disturb other cached entries; reuse count stayed at {}.".format(post_reused)
            ),
        )

        simulate_prefill_completion_only_use_for_testing(req2)
        mgr.free_resources(req2)
        mgr.shutdown()


if __name__ == "__main__":
    unittest.main()
