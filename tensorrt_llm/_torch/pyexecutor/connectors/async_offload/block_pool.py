"""CPU Block Pool with LRU eviction and hash-based prefix lookup."""

from collections import OrderedDict
from typing import List, Optional


class CpuBlockPool:
    """Manages a pool of CPU block IDs with LRU eviction and hash-indexed lookup.

    This class only manages block ID allocation and hash-to-block mapping.
    Actual memory management is handled by the worker.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.block_to_hash: dict[int, int] = {}
        self.lru: OrderedDict[int, bool] = OrderedDict()
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))

    def lookup(self, block_hash: int) -> Optional[int]:
        if block_hash in self.hash_to_block:
            block_id = self.hash_to_block[block_hash]
            self.lru.move_to_end(block_id)
            return block_id
        return None

    def find_prefix_match(self, block_hashes: List[int],
                          start_idx: int = 0) -> int:
        """Count consecutive blocks starting from start_idx that exist in the pool."""
        matched = 0
        for i in range(start_idx, len(block_hashes)):
            if block_hashes[i] in self.hash_to_block:
                matched += 1
            else:
                break
        return matched

    def allocate(self, block_hash: int) -> int:
        """Allocate a CPU block for the given hash, evicting LRU if needed."""
        if self.free_blocks:
            block_id = self.free_blocks.pop()
        else:
            block_id, _ = self.lru.popitem(last=False)
            old_hash = self.block_to_hash.pop(block_id)
            del self.hash_to_block[old_hash]

        self.hash_to_block[block_hash] = block_id
        self.block_to_hash[block_id] = block_hash
        self.lru[block_id] = True
        return block_id

    def touch(self, block_hash: int):
        """Update LRU for an existing block."""
        if block_hash in self.hash_to_block:
            block_id = self.hash_to_block[block_hash]
            self.lru.move_to_end(block_id)
