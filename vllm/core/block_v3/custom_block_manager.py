import math
from typing import List, Optional, Dict, NewType

from vllm.config import CacheConfig, KVCacheConfig, ModelConfig, ParallelConfig
from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import (
    Block,
    DeviceAwareBlockAllocator,
)
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.sequence import Sequence, SequenceGroup
from vllm.core.block_v3.registry import BLOCK_MANAGER_REGISTRY
from vllm.core.block_v3.custom_block import AppAwareManager
from vllm.logger import init_logger
from vllm.core.block.block_table import BlockTable

logger = init_logger(__name__)

CUSTOM_BLOCK_TABLE = Dict[int, BlockTable]


class CustomBlockManager:

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_config: CacheConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self._is_finalized = False
        # modifying _app_aware_managers is not allowed after is_finalized=True
        self._app_aware_managers: Dict[int, AppAwareManager] = {}
        # reading _kv_cache_config is not allowed before is_finalized=True
        self._kv_cache_config = KVCacheConfig(block_size_bytes=-1,
                                              num_logic_layers=-1)

    def compile(self):
        if self._is_finalized:
            return
        self._kv_cache_config = self._get_kv_cache_config()
        self._is_finalized = True

    @property
    def kv_cache_config(self):
        assert self._is_finalized
        return self._kv_cache_config

    def _get_cache_block_size(self):
        # TODO: determine the block size based on app_aware_managers
        head_size = self.model_config.get_head_size()
        num_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_attention_layers = 1

        key_cache_block = self.cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if self.cache_config.cache_dtype == "auto":
            dtype = self.model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    def _get_kv_cache_config(self):
        block_size_bytes = self._get_cache_block_size()
        return KVCacheConfig(block_size_bytes=block_size_bytes,
                             num_logic_layers=1)

    def add_app_aware_managers(self, managers: Dict[int, AppAwareManager]):
        assert not self._is_finalized
        assert managers.keys() & self._app_aware_managers.keys() == set()
        self._app_aware_managers.update(managers)

    def add_block_managers_of_model(self, model: "ModelConfig"):
        BLOCK_MANAGER_REGISTRY.add_managers_of_model(model, self)

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        assert self._is_finalized
        total_blocks = 0
        for id, manager in self._app_aware_managers.items():
            # TODO: the self.cache_config.block_size should be replaced with
            # a number determined by the app_aware_managers
            num_blocks = manager.get_num_required_blocks(
                seq_group, self.cache_config.block_size, num_lookahead_slots)
            print(f"block id: {id}, num_blocks: {num_blocks}")
            total_blocks += num_blocks
        return total_blocks

    def allocate_sequence(
            self, seq_group: SequenceGroup,
            allocator: DeviceAwareBlockAllocator) -> CUSTOM_BLOCK_TABLE:
        assert self._is_finalized
        block_table: CUSTOM_BLOCK_TABLE = {}
        for id, manager in self._app_aware_managers.items():
            block = manager.allocate_sequence(seq_group,
                                              self.cache_config.block_size,
                                              allocator)
            if block is not None:
                block_table[id] = block
        return block_table
