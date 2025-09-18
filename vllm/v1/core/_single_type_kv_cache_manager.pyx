# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cpython cimport Py_ssize_t

import itertools

from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import (ChunkedLocalAttentionSpec,
                                        CrossAttentionSpec, FullAttentionSpec,
                                        KVCacheSpec, MambaSpec,
                                        SlidingWindowSpec)
from vllm.v1.request import Request


cdef class SingleTypeKVCacheManager:
    """Cython-accelerated base manager for a single KV cache type."""

    cdef public Py_ssize_t block_size
    cdef public int dcp_world_size
    cdef public object kv_cache_spec
    cdef public object block_pool
    cdef dict req_to_blocks
    cdef dict num_cached_block
    cdef public int kv_cache_group_id
    cdef public object _null_block

    def __init__(
        self,
        object kv_cache_spec,
        object block_pool,
        int kv_cache_group_id,
        int dcp_world_size = 1,
    ) -> None:
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        if self.dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool

        self.req_to_blocks = {}
        self.num_cached_block = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    cdef list _ensure_request_blocks(self, str request_id):
        cdef list req_blocks = self.req_to_blocks.get(request_id)
        if req_blocks is None:
            req_blocks = []
            self.req_to_blocks[request_id] = req_blocks
        return req_blocks

    cpdef int get_num_blocks_to_allocate(self, str request_id, int num_tokens,
                                         list new_computed_blocks):
        cdef int num_required_blocks = cdiv(num_tokens, self.block_size)
        cdef list existing_blocks = self.req_to_blocks.get(request_id)
        cdef int num_existing_blocks = len(
            existing_blocks) if existing_blocks is not None else 0
        cdef int num_new_blocks = num_required_blocks - len(
            new_computed_blocks) - num_existing_blocks
        if num_new_blocks < 0:
            num_new_blocks = 0

        cdef int num_evictable_computed_blocks = 0
        cdef KVCacheBlock blk
        for blk in new_computed_blocks:
            if blk.ref_cnt == 0 and not blk.is_null:
                num_evictable_computed_blocks += 1
        return num_new_blocks + num_evictable_computed_blocks

    cpdef void save_new_computed_blocks(self, str request_id,
                                        list new_computed_blocks):
        cdef list req_blocks = self.req_to_blocks.get(request_id)
        if req_blocks is None:
            req_blocks = []
            self.req_to_blocks[request_id] = req_blocks
            if new_computed_blocks:
                req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = len(new_computed_blocks)
        else:
            assert len(new_computed_blocks) == 0

    cpdef list allocate_new_blocks(self, str request_id, int num_tokens):
        cdef list req_blocks = self._ensure_request_blocks(request_id)
        cdef int num_required_blocks = cdiv(num_tokens, self.block_size)
        cdef int num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        cdef list new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    cpdef void cache_blocks(self, Request request, int num_tokens):
        cdef int num_cached_blocks = self.num_cached_block.get(
            request.request_id, 0)
        cdef int num_full_blocks = num_tokens // self.block_size
        cdef list blocks = self._ensure_request_blocks(request.request_id)

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    cpdef void free(self, str request_id):
        cdef list req_blocks = self.req_to_blocks.pop(request_id, [])
        if req_blocks:
            self.block_pool.free_blocks(req_blocks[::-1])
        else:
            self.block_pool.free_blocks([])
        self.num_cached_block.pop(request_id, None)

    cpdef int get_num_common_prefix_blocks(self, str request_id,
                                           int num_running_requests):
        raise NotImplementedError

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        int dcp_world_size = 1,
    ):
        raise NotImplementedError

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        raise NotImplementedError


cdef class FullAttentionManager(SingleTypeKVCacheManager):

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        int dcp_world_size = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, (FullAttentionSpec, ChunkedLocalAttentionSpec)
        ), "FullAttentionManager can only be used for full attention " \
            "and chunked local attention groups"
        cdef tuple computed_blocks = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        cdef int block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            block_size *= dcp_world_size
        cdef int max_num_blocks = max_length // block_size
        cdef BlockHash block_hash
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            cached_block = block_pool.get_cached_block(block_hash,
                                                       kv_cache_group_ids)
            if cached_block:
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        pass

    cpdef int get_num_common_prefix_blocks(self, str request_id,
                                           int num_running_requests):
        cdef list blocks = self.req_to_blocks.get(request_id, [])
        cdef int num_common_blocks = 0
        cdef KVCacheBlock block
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


cdef class SlidingWindowManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool,
                 **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        int dcp_world_size = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups")
        assert dcp_world_size == 1, "DCP not support sliding window attn now."

        cdef int sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size)
        if use_eagle:
            sliding_window_contiguous_blocks += 1

        cdef int max_num_blocks = max_length // kv_cache_spec.block_size
        cdef tuple computed_blocks = tuple([
            block_pool.null_block
        ] * max_num_blocks for _ in range(len(kv_cache_group_ids)))
        cdef int num_contiguous_blocks = 0
        cdef bint match_found = False
        cdef int i
        for i in range(max_num_blocks - 1, -1, -1):
            cached_block = block_pool.get_cached_block(block_hashes[i],
                                                       kv_cache_group_ids)
            if cached_block:
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks:]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
        if use_eagle and computed_blocks and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        cdef int last_useful_token = num_computed_tokens - self.sliding_window + 1
        cdef int last_useful_block = last_useful_token // self.block_size
        cdef list blocks = self.req_to_blocks.get(request_id, [])
        cdef list removed_blocks = []
        cdef int i
        if last_useful_block <= 0:
            return
        for i in range(last_useful_block - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        if removed_blocks:
            self.block_pool.free_blocks(removed_blocks)

    cpdef int get_num_common_prefix_blocks(self, str request_id,
                                           int num_running_requests):
        return 0


cdef class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec,
                 block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        int dcp_world_size = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups")
        assert use_eagle is False, ("Hybrid KV cache is not supported for "
                                    "eagle + chunked local attention.")
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        cdef int max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (max_length //
                                         kv_cache_spec.attention_chunk_size *
                                         kv_cache_spec.attention_chunk_size)
        else:
            local_attention_start_idx = 0
        cdef int local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size)
        cdef tuple computed_blocks = tuple([
            block_pool.null_block
        ] * local_attention_start_block_idx
                                          for _ in range(len(
                                              kv_cache_group_ids)))
        cdef int i
        cdef BlockHash block_hash
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            cached_block = block_pool.get_cached_block(block_hash,
                                                       kv_cache_group_ids)
            if cached_block:
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        cdef int num_cached_block = self.num_cached_block.get(request_id, 0)
        cdef int local_attention_start_idx = (
            num_computed_tokens // self.attention_chunk_size *
            self.attention_chunk_size)
        cdef int first_useful_block_idx = local_attention_start_idx // self.block_size
        if num_cached_block > 0:
            if first_useful_block_idx <= num_cached_block:
                return
        cdef list blocks = self.req_to_blocks.get(request_id, [])
        cdef list removed_blocks = []
        cdef int i
        for i in range(first_useful_block_idx - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        if removed_blocks:
            self.block_pool.free_blocks(removed_blocks)

    cpdef int get_num_common_prefix_blocks(self, str request_id,
                                           int num_running_requests):
        return 0


cdef class MambaManager(SingleTypeKVCacheManager):

    cpdef int get_num_blocks_to_allocate(self, str request_id, int num_tokens,
                                         list new_computed_blocks):
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (self.kv_cache_spec.block_size *
                           self.kv_cache_spec.num_speculative_blocks)
        cdef int num_required_blocks = cdiv(num_tokens, self.block_size)
        cdef list existing_blocks = self.req_to_blocks.get(request_id)
        cdef int num_existing_blocks = len(
            existing_blocks) if existing_blocks is not None else 0
        cdef int num_new_blocks = num_required_blocks - len(
            new_computed_blocks) - num_existing_blocks
        if num_new_blocks < 0:
            num_new_blocks = 0
        cdef int num_evictable_computed_blocks = 0
        cdef KVCacheBlock blk
        for blk in new_computed_blocks:
            if blk.ref_cnt == 0 and not blk.is_null:
                num_evictable_computed_blocks += 1
        return num_new_blocks + num_evictable_computed_blocks

    cpdef list allocate_new_blocks(self, str request_id, int num_tokens):
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (self.kv_cache_spec.block_size *
                           self.kv_cache_spec.num_speculative_blocks)
        return SingleTypeKVCacheManager.allocate_new_blocks(
            self, request_id, num_tokens)


cdef class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    cpdef void save_new_computed_blocks(self, str request_id,
                                        list new_computed_blocks):
        assert len(new_computed_blocks) == 0

    cpdef void cache_blocks(self, Request request, int num_tokens):
        raise ValueError("Should not be called as prefix caching is disabled.")

    cpdef int get_num_common_prefix_blocks(self, str request_id,
                                           int num_running_requests):
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        int dcp_world_size = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        raise NotImplementedError(
            "CrossAttentionManager does not support caching")

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        pass


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
}


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    return manager_class(kv_cache_spec, **kwargs)
