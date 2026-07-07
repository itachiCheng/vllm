# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


def _dflash_speculators_config(**overrides):
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        },
        "draft_vocab_size": 128,
        "target_hidden_size": 16,
        "aux_hidden_state_layer_ids": [2, 18, 33, 36],
        "mask_token_id": 127,
        "block_size": 16,
        "micro_block_size": 3,
        "anchor_len": 1,
        "speculators_config": {
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": 15}
            ],
            "verifier": {"name_or_path": "Qwen/Qwen3-8B"},
        },
    }
    config.update(overrides)
    return config


def test_dflash_speculators_config_preserves_micro_block_fields():
    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _dflash_speculators_config()
    )

    assert hf_config["architectures"] == ["DFlashDraftModel"]
    assert hf_config["draft_vocab_size"] == 128
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [2, 18, 33, 36]
    assert hf_config["dflash_config"] == {
        "mask_token_id": 127,
        "target_layer_ids": [2, 18, 33, 36],
        "block_size": 16,
        "micro_block_size": 3,
        "anchor_len": 1,
    }


def test_dflash_speculators_config_defaults_anchor_len():
    config = _dflash_speculators_config()
    del config["anchor_len"]

    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        config
    )

    assert hf_config["dflash_config"]["anchor_len"] == 1
