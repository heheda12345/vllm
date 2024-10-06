import math
from typing import List, Optional, Dict

from vllm.config import ModelConfig
from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.utils import Device, cdiv, chunk_list
from vllm.sequence import Sequence, SequenceGroup
from vllm.core.block_v3.registry import BLOCK_TABLE_REGISTRY
from vllm.core.block_v3.custom_block import CustomBlock
from vllm.logger import init_logger

logger = init_logger(__name__)


class CustomBlockManager:

    def __init__(
        self,
        block_size: int,
        block_allocator: DeviceAwareBlockAllocator,
    ):
        self._block_size = block_size
        self._allocator = block_allocator
        self._custom_blocks: Dict[int, CustomBlock] = {}

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        total_blocks = 0
        for id, block in self._custom_blocks.items():
            num_blocks = block.get_num_required_blocks(seq_group, block_size,
                                                       num_lookahead_slots)
            print(f"block id: {id}, num_blocks: {num_blocks}")
            total_blocks += num_blocks
        return total_blocks

    def add_block_tables(self, block_tables: Dict[int, CustomBlock]):
        assert block_tables.keys() & self._custom_blocks.keys() == set()
        self._custom_blocks.update(block_tables)

    def add_block_tables_of_model(self, model: "ModelConfig"):
        BLOCK_TABLE_REGISTRY.add_to_block_manager(model, self)
