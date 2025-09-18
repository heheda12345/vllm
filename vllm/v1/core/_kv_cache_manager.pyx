# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cpython cimport Py_ssize_t

from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


cdef inline tuple _make_empty_group_tuple(int num_groups):
    """Create a tuple of empty python lists for KV cache groups."""
    cdef list groups = []
    cdef int i
    for i in range(num_groups):
        groups.append([])
    return tuple(groups)


cdef inline bint _all_groups_empty(tuple groups):
    """Check whether all groups are empty."""
    cdef Py_ssize_t i
    for i in range(len(groups)):
        if len(groups[i]) != 0:
            return False
    return True


cdef class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    cdef tuple _blocks

    def __cinit__(self, tuple blocks):
        self._blocks = blocks

    @property
    def blocks(self):
        return self._blocks

    def __add__(self, KVCacheBlocks other) -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        cdef Py_ssize_t i, num_groups
        cdef list combined
        cdef list result = []
        cdef tuple self_blocks = self._blocks
        cdef tuple other_blocks = other._blocks
        num_groups = len(self_blocks)
        for i in range(num_groups):
            combined = list(self_blocks[i])
            combined.extend(other_blocks[i])
            result.append(combined)
        return KVCacheBlocks(tuple(result))

    def get_block_ids(
        self,
        allow_none: bool = False,
    ):
        """Converts the KVCacheBlocks instance to block_ids."""
        cdef tuple groups = self._blocks
        cdef Py_ssize_t i
        cdef list block_id_group
        cdef list result = []
        cdef list group

        if allow_none and _all_groups_empty(groups):
            return None

        for i in range(len(groups)):
            group = groups[i]
            block_id_group = []
            for block in group:
                block_id_group.append(block.block_id)
            result.append(block_id_group)
        return tuple(result)

    def get_unhashed_block_ids(self):
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        cdef tuple groups = self._blocks
        cdef list result = []
        cdef object block
        assert len(groups) == 1, "Only one group is supported"
        for block in groups[0]:
            if block.block_hash is None:
                result.append(block.block_id)
        return result

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(_make_empty_group_tuple(len(self._blocks)))


cdef class KVCacheManager:

    cdef public int max_model_len
    cdef bint enable_caching
    cdef bint use_eagle
    cdef bint log_stats
    cdef object prefix_cache_stats
    cdef object block_size
    cdef object coordinator
    cdef int num_kv_cache_groups
    cdef object block_pool
    cdef object kv_cache_config

    def __cinit__(
        self,
        KVCacheConfig kv_cache_config,
        int max_model_len,
        bint enable_caching = True,
        bint use_eagle = False,
        bint log_stats = False,
        bint enable_kv_cache_events = False,
        int dcp_world_size = 1,
    ) -> None:
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

            if dcp_world_size > 1:
                assert len(kv_cache_config.kv_cache_groups) == 1
                self.block_size *= dcp_world_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

    @property
    def usage(self) -> float:
        """Get the KV cache usage."""
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats."""
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    cpdef tuple get_computed_blocks(self, Request request):
        """Get the computed (cached) blocks for the request."""
        cdef int num_new_computed_tokens
        cdef tuple computed_blocks

        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        cdef int max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    cpdef object allocate_slots(
        self,
        Request request,
        int num_new_tokens,
        int num_new_computed_tokens = 0,
        KVCacheBlocks new_computed_blocks = None,
        int num_lookahead_tokens = 0,
        bint delay_cache_blocks = False,
        int num_encoder_tokens = 0,
    ):
        """Add slots for a request with new tokens to append."""
        cdef tuple new_computed_block_list
        cdef int num_computed_tokens
        cdef int num_tokens_need_slot
        cdef int num_blocks_to_allocate
        cdef tuple new_blocks
        cdef int num_tokens_to_cache

        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = _make_empty_group_tuple(
                len(self.kv_cache_config.kv_cache_groups))

        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None

        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens)

        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)

    cpdef void free(self, Request request):
        """Free the blocks allocated for the request."""
        self.coordinator.free(request.request_id)

    cpdef bint reset_prefix_cache(self):
        """Reset prefix cache."""
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    cpdef list get_num_common_prefix_blocks(
        self,
        Request request,
        int num_running_requests,
    ):
        """Calculate the number of common prefix blocks shared by all requests."""
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    cpdef list take_events(self):
        """Take the KV cache events from the block pool."""
        return self.block_pool.take_events()

    cpdef KVCacheBlocks get_blocks(self, str request_id):
        """Get the blocks of a request."""
        return KVCacheBlocks(self.coordinator.get_blocks(request_id))

    cpdef tuple get_block_ids(self, str request_id):
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    cpdef void cache_blocks(self, Request request, int num_computed_tokens):
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    cpdef KVCacheBlocks create_empty_block_list(self):
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(_make_empty_group_tuple(self.num_kv_cache_groups))


__all__ = ["KVCacheBlocks", "KVCacheManager"]
