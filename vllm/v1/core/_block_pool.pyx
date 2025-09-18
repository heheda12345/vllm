# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cpython cimport Py_ssize_t

from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.distributed.kv_events import (MEDIUM_GPU, AllBlocksCleared,
                                        BlockRemoved, BlockStored,
                                        KVCacheEvent)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHash, BlockHashWithGroupId,
                                         ExternalBlockHash,
                                         FreeKVCacheBlockQueue, KVCacheBlock,
                                         get_block_hash,
                                         make_block_hash_with_group_id,
                                         maybe_convert_block_hash)
from vllm.v1.request import Request

logger = init_logger(__name__)


cdef class BlockPool:

    cdef public int num_gpu_blocks
    cdef bint enable_caching
    cdef bint enable_kv_cache_events
    cdef list blocks
    cdef object free_block_queue
    cdef object cached_block_hash_to_block
    cdef object null_block
    cdef list kv_event_queue

    def __cinit__(
        self,
        int num_gpu_blocks,
        bint enable_caching,
        bint enable_kv_cache_events = False,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events

        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block = defaultdict(dict)

        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        if enable_kv_cache_events:
            self.kv_event_queue = []
        else:
            self.kv_event_queue = []

    cpdef object get_cached_block(
        self,
        BlockHash block_hash,
        list kv_cache_group_ids,
    ):
        cdef list cached_blocks = []
        cdef Py_ssize_t i
        cdef int group_id
        cdef object block_hash_with_group_id
        cdef object cached_blocks_one_group

        for i in range(len(kv_cache_group_ids)):
            group_id = kv_cache_group_ids[i]
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            cached_blocks_one_group = self.cached_block_hash_to_block.get(
                block_hash_with_group_id)
            if not cached_blocks_one_group:
                return None
            cached_blocks.append(next(iter(cached_blocks_one_group.values())))
        return cached_blocks

    cpdef void cache_full_blocks(
        self,
        Request request,
        list blocks,
        int num_cached_blocks,
        int num_full_blocks,
        int block_size,
        int kv_cache_group_id,
    ):
        if num_cached_blocks == num_full_blocks:
            return

        cdef list new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        cdef list new_block_hashes = request.block_hashes[num_cached_blocks:]

        cdef object new_hashes = [] if self.enable_kv_cache_events else None
        cdef Py_ssize_t i
        cdef KVCacheBlock blk
        cdef object block_hash
        cdef object block_hash_with_group_id
        cdef KVCacheBlock parent_block
        cdef object parent_block_hash

        for i in range(len(new_full_blocks)):
            blk = new_full_blocks[i]
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id)
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block[block_hash_with_group_id][
                blk.block_id] = blk
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash))

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size:num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=(request.lora_request.id
                             if request.lora_request else None),
                    medium=MEDIUM_GPU,
                ))

    cpdef list get_new_blocks(self, int num_blocks):
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        cdef list ret = self.free_block_queue.popleft_n(num_blocks)
        cdef KVCacheBlock block

        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    cdef bint _maybe_evict_cached_block(self, KVCacheBlock block):
        cdef object block_hash = block.block_hash
        if block_hash is None:
            return False
        cdef object blocks_by_id = self.cached_block_hash_to_block.get(
            block_hash)
        if blocks_by_id is None:
            return False
        block.reset_hash()
        blocks_by_id.pop(block.block_id, None)
        if len(blocks_by_id) == 0:
            del self.cached_block_hash_to_block[block_hash]

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(block_hashes=[
                    maybe_convert_block_hash(get_block_hash(block_hash))
                ], medium=MEDIUM_GPU))
        return True

    cpdef void touch(self, tuple blocks):
        cdef Py_ssize_t i, j
        cdef list blocks_per_group
        cdef KVCacheBlock block

        for i in range(len(blocks)):
            blocks_per_group = blocks[i]
            for j in range(len(blocks_per_group)):
                block = blocks_per_group[j]
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    cpdef void free_blocks(self, Iterable ordered_blocks):
        cdef list blocks_list = list(ordered_blocks)
        cdef KVCacheBlock block

        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])

    cpdef bint reset_prefix_cache(self):
        cdef int num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:
            logger.warning(
                "Failed to reset prefix cache because some blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        self.cached_block_hash_to_block = defaultdict(dict)

        cdef KVCacheBlock block
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    cpdef int get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    cpdef float get_usage(self):
        cdef int total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    cpdef list take_events(self):
        if not self.enable_kv_cache_events:
            return []
        cdef list events = self.kv_event_queue
        self.kv_event_queue = []
        return events


__all__ = ["BlockPool"]
