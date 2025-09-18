# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cpython cimport Py_ssize_t
from typing import Optional

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager, FullAttentionManager, get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.request import Request


cdef class KVCacheCoordinator:
    """Cython-accelerated coordinator covering all KV cache groups."""

    cdef public object kv_cache_config
    cdef public int max_model_len
    cdef public bint enable_caching
    cdef public object block_pool
    cdef public bint use_eagle
    cdef tuple single_type_managers

    def __init__(
        self,
        KVCacheConfig kv_cache_config,
        int max_model_len,
        bint use_eagle,
        bint enable_caching,
        bint enable_kv_cache_events,
        int dcp_world_size,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(kv_cache_config.num_blocks, enable_caching,
                                    enable_kv_cache_events)

        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
            ) for i, kv_cache_group in enumerate(
                self.kv_cache_config.kv_cache_groups))

    cpdef int get_num_blocks_to_allocate(
        self,
        str request_id,
        int num_tokens,
        tuple new_computed_blocks,
        int num_encoder_tokens,
    ):
        cdef int num_blocks_to_allocate = 0
        cdef Py_ssize_t i
        cdef object manager
        cdef list blocks_for_group
        for i in range(len(self.single_type_managers)):
            manager = self.single_type_managers[i]
            if isinstance(manager, CrossAttentionManager):
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [])
            else:
                blocks_for_group = new_computed_blocks[
                    i] if i < len(new_computed_blocks) else []
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_tokens, blocks_for_group)
        return num_blocks_to_allocate

    cpdef void save_new_computed_blocks(
        self,
        str request_id,
        tuple new_computed_blocks,
    ):
        cdef Py_ssize_t i
        cdef object manager
        for i in range(len(self.single_type_managers)):
            manager = self.single_type_managers[i]
            manager.save_new_computed_blocks(request_id,
                                             new_computed_blocks[i])

    cpdef tuple allocate_new_blocks(
        self,
        str request_id,
        int num_tokens,
        int num_encoder_tokens = 0,
    ):
        cdef list result = []
        cdef object manager
        for manager in self.single_type_managers:
            if isinstance(manager, CrossAttentionManager):
                result.append(
                    manager.allocate_new_blocks(request_id, num_encoder_tokens))
            else:
                result.append(manager.allocate_new_blocks(request_id,
                                                          num_tokens))
        return tuple(result)

    cpdef void cache_blocks(self, Request request, int num_computed_tokens):
        cdef object manager
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    cpdef void free(self, str request_id):
        cdef object manager
        for manager in self.single_type_managers:
            manager.free(request_id)

    cpdef list get_num_common_prefix_blocks(self, str request_id,
                                            int num_running_requests):
        cdef list num_blocks_per_group = []
        cdef object manager
        for manager in self.single_type_managers:
            num_blocks_per_group.append(
                manager.get_num_common_prefix_blocks(request_id,
                                                     num_running_requests))
        return num_blocks_per_group

    cpdef void remove_skipped_blocks(self, str request_id,
                                     int num_computed_tokens):
        cdef object manager
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, num_computed_tokens)

    cpdef tuple get_blocks(self, str request_id):
        cdef list blocks = []
        cdef object manager
        cdef list manager_blocks
        for manager in self.single_type_managers:
            manager_blocks = manager.req_to_blocks.get(request_id)
            if manager_blocks is None:
                manager_blocks = []
            blocks.append(manager_blocks)
        return tuple(blocks)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        int max_cache_hit_length,
    ):
        raise NotImplementedError


cdef class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """Coordinator when prefix caching is disabled."""

    def __init__(self, KVCacheConfig kv_cache_config, int max_model_len,
                 bint use_eagle, bint enable_kv_cache_events,
                 int dcp_world_size):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         False,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        self.num_single_type_manager = len(self.single_type_managers)

    cpdef list get_num_common_prefix_blocks(self, str request_id,
                                            int num_running_requests):
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        int max_cache_hit_length,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        cdef tuple blocks = tuple([]
                                  for _ in range(self.num_single_type_manager))
        return blocks, 0


cdef class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """Coordinator for a single KV cache group."""

    def __init__(self, KVCacheConfig kv_cache_config, int max_model_len,
                 bint use_eagle, bint enable_caching,
                 bint enable_kv_cache_events, int dcp_world_size):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[
            0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        int max_cache_hit_length,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        cdef tuple hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            dcp_world_size=self.dcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


cdef class HybridKVCacheCoordinator(KVCacheCoordinator):
    """Coordinator for models with multiple KV cache types."""

    def __init__(self, KVCacheConfig kv_cache_config, int max_model_len,
                 bint use_eagle, bint enable_caching,
                 bint enable_kv_cache_events, int dcp_world_size):
        super().__init__(kv_cache_config,
                         max_model_len,
                         use_eagle,
                         enable_caching,
                         enable_kv_cache_events,
                         dcp_world_size=dcp_world_size)
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    cpdef void verify_and_split_kv_cache_groups(self):
        cdef Optional[FullAttentionSpec] full_attention_spec = None
        cdef Optional[KVCacheSpec] other_spec = None
        self.full_attention_group_ids = []
        self.other_group_ids = []
        cdef Py_ssize_t i
        cdef object g
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_spec is None:
                    full_attention_spec = g.kv_cache_spec
                else:
                    assert full_attention_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes exactly one type of "
                        "full attention groups now.")
                self.full_attention_group_ids.append(i)
            else:
                if other_spec is None:
                    other_spec = g.kv_cache_spec
                else:
                    assert other_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes exactly one other "
                        "type of groups now.")
                self.other_group_ids.append(i)

        assert full_attention_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of full "
            "attention groups now.")
        assert other_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of other "
            "groups now.")

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[
            self.other_group_ids[0]].__class__
        self.full_attention_spec = full_attention_spec
        self.other_spec = other_spec
        self.full_attention_block_size = self.full_attention_spec.block_size
        self.other_block_size = self.other_spec.block_size

        if self.enable_caching:
            assert self.other_block_size % self.full_attention_block_size == 0, (
                "KVCacheCoordinator assumes the block_size of full attention "
                "layers is divisible by other layers now.")

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridKVCacheCoordinator assumes the full attention group ids "
                "and other attention group ids do not interleave.")

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        int max_cache_hit_length,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks_full_attn = (
            self.full_attention_manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=self.full_attention_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.full_attention_spec,
                use_eagle=self.use_eagle,
            ))
        cdef int hit_length = len(
            hit_blocks_full_attn[0]) * self.full_attention_block_size

        hit_blocks_other_attn = (
            self.other_attention_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=hit_length,
                kv_cache_group_ids=self.other_group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=self.other_spec,
                use_eagle=self.use_eagle,
            ))
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size

        assert hit_length % self.full_attention_block_size == 0

        cdef list group_hit_blocks
        for group_hit_blocks in hit_blocks_full_attn:
            del group_hit_blocks[hit_length // self.full_attention_block_size:]

        if self.full_attn_first:
            hit_blocks = hit_blocks_full_attn + hit_blocks_other_attn
        else:
            hit_blocks = hit_blocks_other_attn + hit_blocks_full_attn
        return hit_blocks, hit_length


def get_kv_cache_coordinator(kv_cache_config: KVCacheConfig,
                             int max_model_len, bint use_eagle,
                             bint enable_caching,
                             bint enable_kv_cache_events,
                             int dcp_world_size) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(kv_cache_config,
                                               max_model_len,
                                               use_eagle,
                                               enable_kv_cache_events,
                                               dcp_world_size=dcp_world_size)
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(kv_cache_config,
                                         max_model_len,
                                         use_eagle,
                                         enable_caching,
                                         enable_kv_cache_events,
                                         dcp_world_size=dcp_world_size)
    return HybridKVCacheCoordinator(kv_cache_config,
                                    max_model_len,
                                    use_eagle,
                                    enable_caching,
                                    enable_kv_cache_events,
                                    dcp_world_size=dcp_world_size)
