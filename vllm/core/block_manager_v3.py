from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.config import ModelConfig
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block_v3.custom_block_manager import CustomBlockManager
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.logger import init_logger

logger = init_logger(__name__)


class BlockSpaceManagerV3(BlockSpaceManager):

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported")
        if enable_caching:
            raise NotImplementedError("Prefix caching is not supported")

        self.global_block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.custom_block_manager = CustomBlockManager(
            block_size=block_size, block_allocator=self.global_block_allocator)

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        logger.info(
            "############### create BlockSpaceManagerV3, block_size: {}".
            format(block_size))

    def add_model(self, model: ModelConfig):
        self.custom_block_manager.add_block_tables_of_model(model)

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # import pdb
        # pdb.set_trace()
        assert len(seq_group.seqs) == 1
        num_required_blocks = self.custom_block_manager.get_num_required_blocks(
            seq_group,
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )
        print("num_required_blocks: ", num_required_blocks)

        num_free_gpu_blocks = self.global_block_allocator.get_num_free_blocks(
            device=Device.GPU)

        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.allocate")

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_append_slots")

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.append_slots")

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError("not implemented: BlockSpaceManagerV3.fork")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_swap_in")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.swap_in")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_swap_out")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.swap_out")

    def free(self, seq: Sequence) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError("not implemented: BlockSpaceManagerV3.free")

    def get_block_table(self, seq: Sequence) -> List[int]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_block_table")

    def get_num_free_gpu_blocks(self) -> int:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_num_free_gpu_blocks")

    def get_num_free_cpu_blocks(self) -> int:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_num_free_cpu_blocks")

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.access_all_blocks_in_seq")

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_common_computed_block_ids"
        )

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.mark_blocks_as_computed")

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_prefix_cache_hit_rate")
