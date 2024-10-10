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


def require_kv_config_init(func):

    def wrapper(self, *args, **kwargs):
        assert self._initialized, "KV cache config is not initialized"
        return func(self, *args, **kwargs)

    return wrapper


def require_kv_config_not_init(func):

    def wrapper(self, *args, **kwargs):
        assert not self._initialized, "KV cache config is already initialized"
        return func(self, *args, **kwargs)

    return wrapper


class CustomBlockManager:

    def __init__(self, parallel_config: ParallelConfig,
                 cache_config: CacheConfig):
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self._initialized = False
        # modifying _app_aware_managers is not allowed after is_finalized=True
        self._app_aware_managers: Dict[int, AppAwareManager] = {}
        # reading _kv_cache_config is not allowed before is_finalized=True
        self._kv_cache_config = KVCacheConfig(block_size_bytes=-1,
                                              num_logic_layers=-1)

    @require_kv_config_not_init
    def compile(self):
        self._kv_cache_config = self._get_kv_cache_config()
        for manager in self._app_aware_managers.values():
            manager.init_kv_cache_config(self._kv_cache_config)
        self._initialized = True

    @property
    @require_kv_config_init
    def kv_cache_config(self):
        assert self._initialized
        return self._kv_cache_config

    def _get_cache_block_size(self):
        page_sizes = [
            manager.get_page_size(self.cache_config.block_size)
            for manager in self._app_aware_managers.values()
        ]
        # We assume all components use the same page size now.
        assert all(page_size == page_sizes[0] for page_size in page_sizes)
        return page_sizes[0]

    def _get_kv_cache_config(self):
        block_size_bytes = self._get_cache_block_size()
        return KVCacheConfig(block_size_bytes=block_size_bytes,
                             num_logic_layers=1)

    @require_kv_config_not_init
    def add_app_aware_managers(self, managers: Dict[int, AppAwareManager]):
        assert managers.keys() & self._app_aware_managers.keys() == set()
        self._app_aware_managers.update(managers)

    @require_kv_config_not_init
    def add_block_managers_of_model(self, model: "ModelConfig"):
        managers = BLOCK_MANAGER_REGISTRY.get_managers_of_model(
            model, self.cache_config, self.parallel_config)
        self.add_app_aware_managers(managers)

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        total_blocks = 0
        for id, manager in self._app_aware_managers.items():
            num_blocks = manager.get_num_required_blocks(
                seq_group, num_lookahead_slots)
            total_blocks += num_blocks
        return total_blocks

    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            allocator: DeviceAwareBlockAllocator) -> CUSTOM_BLOCK_TABLE:
        block_table: CUSTOM_BLOCK_TABLE = {}
        for layer_id, manager in self._app_aware_managers.items():
            block = manager.allocate_sequence(seq_group, allocator)
            if block is not None:
                block_table[layer_id] = block
        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            num_lookahead_slots: int) -> int:
        total_blocks = 0
        for layer_id, manager in self._app_aware_managers.items():
            assert layer_id in block_table
            num_blocks = manager.get_num_blocks_touched_by_append_slots(
                seq, block_table[layer_id], num_lookahead_slots)
            total_blocks += num_blocks
        return total_blocks

    @require_kv_config_init
    def append_token_ids(self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
                         num_lookahead_slots: int) -> int:
        for layer_id, manager in self._app_aware_managers.items():
            assert layer_id in block_table
            manager.append_token_ids(seq, block_table[layer_id],
                                     num_lookahead_slots)
