# Import methods in this file to configure per-layer block table
from abc import abstractmethod
import math
from typing import Protocol, Dict, Any, Type, TYPE_CHECKING
from torch import nn
from typing_extensions import TypeVar
from vllm.config import KVCacheConfig, ModelConfig, ParallelConfig
from vllm.core.block.block_table import BlockTable
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, Device, cdiv, chunk_list, get_dtype_size
from vllm.logger import init_logger


# TODO: should we call BlockTable's function here?
def require_kv_config_init(func):

    def wrapper(self, *args, **kwargs):
        assert self.initialized, "KV cache config is not initialized"
        return func(self, *args, **kwargs)

    return wrapper


class AppAwareManager:

    def __init__(self):
        self.kv_cache_config = KVCacheConfig(block_size_bytes=-1,
                                             num_logic_layers=-1)
        self.block_size = -1
        self.initialized = False

    def init_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        assert not self.initialized, "KV cache config is already initialized"
        self.kv_cache_config = kv_cache_config
        self.initialized = True

    @abstractmethod
    def get_page_size(self, block_size: int):
        pass

    @abstractmethod
    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        # FIXME(heheda12345): When implementing this interface,  we assume that
        # all sequences in the group share the same prompt. This is the same as
        # BlockSpaceManagerV2.
        pass

    @abstractmethod
    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        pass

    @abstractmethod
    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(
            self, seq: Sequence, block_table: BlockTable,
            num_lookahead_slots: int) -> int:
        pass


AppAwareAttnMetadataBuilder = AppAwareManager


def get_token_size_default(model_config: ModelConfig,
                           parallel_config: ParallelConfig, cache_dtype: str):
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = 1

    key_cache_block = num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total


class SelfAttentionManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str):
        super().__init__()
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       cache_dtype)
        self.block_size = -1

    def init_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        super().init_kv_cache_config(kv_cache_config)
        assert kv_cache_config.block_size_bytes % self.memory_per_token == 0
        self.block_size = kv_cache_config.block_size_bytes // self.memory_per_token

    def get_page_size(self, block_size: int):
        return block_size * self.memory_per_token

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_tokens = len(seq.get_token_ids())
        return cdiv(num_tokens, self.block_size) + num_lookahead_slots

    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        block_table = BlockTable(block_size=self.block_size,
                                 block_allocator=block_allocator,
                                 max_block_sliding_window=None)
        block_table.allocate(seq.get_token_ids())

        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(self, seq: Sequence,
                                               block_table: BlockTable,
                                               num_lookahead_slots: int):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        num_token_ids = len(unseen_token_ids) + num_lookahead_slots
        first_chunk_size = self.block_size - (block_table._num_full_slots %
                                              self.block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self.block_size))
        return num_token_blocks

    @require_kv_config_init
    def append_token_ids(self, seq: Sequence, block_table: BlockTable,
                         num_lookahead_slots: int):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        block_table.ensure_num_empty_slots(
            num_empty_slots=len(unseen_token_ids) + num_lookahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = block_table._num_full_slots // self.block_size
        token_blocks = block_table._chunk_token_blocks_for_append(
            unseen_token_ids)

        for i, token_block in enumerate(token_blocks):
            block_table._blocks.append_token_ids(first_block_idx + i,
                                                 token_block)

        block_table._num_full_slots += len(unseen_token_ids)


class EncoderDecoderManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str):
        super().__init__()
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       cache_dtype)
        self.block_size = -1

    def init_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        super().init_kv_cache_config(kv_cache_config)
        assert kv_cache_config.block_size_bytes % self.memory_per_token == 0
        self.block_size = kv_cache_config.block_size_bytes // self.memory_per_token

    def get_page_size(self, block_size: int):
        return block_size * self.memory_per_token

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                _num_lookahead_slots: int = 0) -> int:
        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_tokens = len(encoder_seq.get_token_ids())
            return cdiv(num_tokens, self.block_size)
        else:
            return 0

    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        encoder_seq = seq_group.get_encoder_seq()
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=block_allocator,
            max_block_sliding_window=None,
        )
        encoder_seq_token_ids = encoder_seq.get_token_ids()
        if encoder_seq_token_ids:
            block_table.allocate(encoder_seq_token_ids)
        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(self, seq, block_table,
                                               num_lookahead_slots):
        # Encoder-decoder KV cache size is not changed during decoding
        return 0

    @require_kv_config_init
    def append_token_ids(self, seq, block_table, num_lookahead_slots):
        # Encoder-decoder KV cache size is not changed during decoding
        pass


class SlidingWindowManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str,
                 sliding_window_size: int):
        super().__init__()
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       cache_dtype)
        assert sliding_window_size > 0
        self.sliding_window_size = sliding_window_size
        self.block_size = -1
        self.max_block_sliding_window = -1

    def init_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        super().init_kv_cache_config(kv_cache_config)
        assert kv_cache_config.block_size_bytes % self.memory_per_token == 0
        self.block_size = kv_cache_config.block_size_bytes // self.memory_per_token

        # +1 here because // rounds down
        num_blocks = (self.sliding_window_size - 1) // self.block_size + 1
        # +1 here because the last block may not be full,
        # and so the sequence stretches one more block at the beginning
        # For example, if sliding_window is 3 and block_size is 4,
        # we may need 2 blocks when the second block only holds 1 token.
        self.max_block_sliding_window = num_blocks + 1

    def get_page_size(self, block_size: int):
        return block_size * self.memory_per_token

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_tokens = len(seq.get_token_ids())
        num_required_blocks = cdiv(num_tokens,
                                   self.block_size) + num_lookahead_slots
        num_required_blocks = min(num_required_blocks,
                                  self.max_block_sliding_window)
        return num_required_blocks

    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # TODO: handle sliding window in this manager, so that we can remove the
        # sliding window logic in other parts of the code
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=block_allocator,
            max_block_sliding_window=self.max_block_sliding_window)
        block_table.allocate(seq.get_token_ids())

        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(self, seq: Sequence,
                                               block_table: BlockTable,
                                               num_lookahead_slots: int):
        # TODO: this function may add redundant blocks to the block table
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        num_token_ids = len(unseen_token_ids) + num_lookahead_slots
        first_chunk_size = self.block_size - (block_table._num_full_slots %
                                              self.block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self.block_size))
        return num_token_blocks

    @require_kv_config_init
    def append_token_ids(self, seq: Sequence, block_table: BlockTable,
                         num_lookahead_slots: int):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())
        num_computed_slots = seq.data.get_num_computed_tokens()

        null_block = block_table._allocator.allocate_or_get_null_block()
        assert num_computed_slots is not None
        end_block_idx = (num_computed_slots //
                         self.block_size) - self.max_block_sliding_window
        for idx in range(0, end_block_idx):
            b = block_table._blocks[idx]
            if b is not null_block:
                block_table._allocator.free(b)
                block_table._blocks[idx] = null_block

        block_table.ensure_num_empty_slots(
            num_empty_slots=len(unseen_token_ids) + num_lookahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = block_table._num_full_slots // self.block_size
        token_blocks = block_table._chunk_token_blocks_for_append(
            unseen_token_ids)

        for i, token_block in enumerate(token_blocks):
            block_table._blocks.append_token_ids(first_block_idx + i,
                                                 token_block)

        block_table._num_full_slots += len(unseen_token_ids)
