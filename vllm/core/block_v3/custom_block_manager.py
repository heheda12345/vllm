import math
from typing import List, Optional, Dict, NewType

from vllm.config import ModelConfig
from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import (
    Block,
    DeviceAwareBlockAllocator,
)
from vllm.utils import Device, cdiv, chunk_list
from vllm.sequence import Sequence, SequenceGroup
from vllm.core.block_v3.registry import BLOCK_MANAGER_REGISTRY
from vllm.core.block_v3.custom_block import AppAwareManager
from vllm.logger import init_logger
from vllm.core.block.block_table import BlockTable

logger = init_logger(__name__)

CUSTOM_BLOCK_TABLE = Dict[int, BlockTable]


class CustomBlockManager:

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        self._app_aware_managers: Dict[int, AppAwareManager] = {}

    def add_app_aware_managers(self, managers: Dict[int, AppAwareManager]):
        assert managers.keys() & self._app_aware_managers.keys() == set()
        self._app_aware_managers.update(managers)

    def add_block_tables_of_model(self, model: "ModelConfig"):
        BLOCK_MANAGER_REGISTRY.add_managers_of_model(model, self)

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        total_blocks = 0
        for id, manager in self._app_aware_managers.items():
            num_blocks = manager.get_num_required_blocks(
                seq_group, block_size, num_lookahead_slots)
            print(f"block id: {id}, num_blocks: {num_blocks}")
            total_blocks += num_blocks
        return total_blocks

    def allocate_sequence(self,
                          seq_group: SequenceGroup) -> CUSTOM_BLOCK_TABLE:
        block_table: CUSTOM_BLOCK_TABLE = {}
        for id, manager in self._app_aware_managers.items():
            block = manager.allocate_sequence(seq_group, self._block_size,
                                              self._allocator)
            if block is not None:
                block_table[id] = block
        return block_table
