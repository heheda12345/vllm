"""Compare the outputs of HF and vLLM when using greedy sampling.

This tests bigger models and use half precision.

Run `pytest tests/models/test_big_models.py`.
"""
import pytest

from ...models.utils import check_outputs_equal

MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "facebook/opt-125m"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("vllm_engine_args",
                         [{
                             'use_v2_block_manager': True,
                             'use_per_layer_block_manager': False,
                         }, {
                             'use_v2_block_manager': False,
                             'use_per_layer_block_manager': True,
                         }])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    max_tokens: int,
    vllm_engine_args,
) -> None:
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(model, **vllm_engine_args) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
