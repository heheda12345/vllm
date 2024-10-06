# Import methods in this file to configure per-layer block table
from abc import abstractmethod
from typing import Protocol, Dict, Any, Type, TYPE_CHECKING
from torch import nn
from typing_extensions import TypeVar
from vllm.core.block.block_table import BlockTable
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device, cdiv, chunk_list
from vllm.logger import init_logger


class AppAwareManager:

    @abstractmethod
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        # FIXME(heheda12345): When implementing this interface,  we assume that
        # all sequences in the group share the same prompt. This is the same as
        # BlockSpaceManagerV3
        pass

    @abstractmethod
    def allocate_sequence(
            self, seq_group: SequenceGroup, block_size: int,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        pass


class SelfAttentionManager(AppAwareManager):

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_tokens = len(seq.get_token_ids())
        return cdiv(num_tokens, block_size) + num_lookahead_slots

    def allocate_sequence(
            self, seq_group: SequenceGroup, block_size: int,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        block_table = BlockTable(block_size=block_size,
                                 block_allocator=block_allocator,
                                 max_block_sliding_window=None)
        block_table.allocate(seq.get_token_ids())

        return block_table


class EncoderDecoderManager(AppAwareManager):

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                _num_lookahead_slots: int = 0) -> int:
        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_tokens = len(encoder_seq.get_token_ids())
            return cdiv(num_tokens, block_size)
        else:
            return 0

    def allocate_sequence(
            self, seq_group: SequenceGroup, block_size: int,
            block_allocator: DeviceAwareBlockAllocator) -> BlockTable:
        encoder_seq = seq_group.get_encoder_seq()
        block_table = BlockTable(
            block_size=block_size,
            block_allocator=block_allocator,
            max_block_sliding_window=None,
        )
        block_table.allocate(encoder_seq.get_token_ids())

        return block_table
