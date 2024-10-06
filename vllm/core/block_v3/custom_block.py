# Import methods in this file to configure per-layer block table
from abc import abstractmethod
from typing import Protocol, Dict, Any, Type, TYPE_CHECKING
from torch import nn
from typing_extensions import TypeVar
from vllm.core.block.block_table import BlockTable
from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import Device, cdiv, chunk_list
from vllm.logger import init_logger
if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.core.block_v3.custom_block_manager import CustomBlockManager


class CustomBlock:

    @abstractmethod
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        pass


class SelfAttentionBlock(CustomBlock):

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                num_lookahead_slots: int = 0) -> int:
        num_tokens = len(seq_group.seqs[0].get_token_ids())
        return cdiv(num_tokens, block_size) + num_lookahead_slots


class EncoderDecoderBlock(CustomBlock):

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                block_size: int,
                                _num_lookahead_slots: int = 0) -> int:
        if seq_group.encoder_prompt_token_ids is None:
            return 0
        num_tokens = len(seq_group.encoder_prompt_token_ids)
        return cdiv(num_tokens, block_size)
