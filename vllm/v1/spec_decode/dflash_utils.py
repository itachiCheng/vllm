# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def validate_dflash_micro_block_layout(
    block_size: int,
    micro_block_size: int | None,
    anchor_len: int,
) -> None:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if anchor_len < 0 or anchor_len >= block_size:
        raise ValueError(
            "anchor_len must be in [0, block_size), got "
            f"anchor_len={anchor_len}, block_size={block_size}."
        )
    if micro_block_size is None:
        return
    if micro_block_size <= 0:
        raise ValueError(
            f"micro_block_size must be positive, got {micro_block_size}."
        )
    spec_len = block_size - anchor_len
    if spec_len % micro_block_size != 0:
        raise ValueError(
            "block_size - anchor_len must be divisible by micro_block_size, "
            f"got block_size={block_size}, anchor_len={anchor_len}, "
            f"micro_block_size={micro_block_size}."
        )


def build_dflash_micro_block_query_kv_mask(
    *,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    num_actual_tokens: int,
    max_seq_len: int,
    block_size: int,
    micro_block_size: int,
    anchor_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense DFlash micro-block mask for backends with mask tensors.

    The returned int8 mask uses 1 for masked positions and 0 for visible
    positions, matching vLLM-Ascend's split-fuse attention mask convention.
    Rows are packed query tokens and columns are logical KV positions.
    """
    validate_dflash_micro_block_layout(
        block_size=block_size,
        micro_block_size=micro_block_size,
        anchor_len=anchor_len,
    )

    mask = torch.ones(
        (num_actual_tokens, max_seq_len),
        dtype=torch.int8,
        device=device,
    )
    q_offsets = torch.arange(block_size, device=device)
    kv_offsets = torch.arange(block_size, device=device)

    q_is_anchor = q_offsets[:, None] < anchor_len
    kv_is_anchor = kv_offsets[None, :] < anchor_len
    anchor_visible = q_is_anchor & kv_is_anchor
    spec_anchor_visible = (~q_is_anchor) & kv_is_anchor

    q_micro = torch.div(
        q_offsets[:, None] - anchor_len,
        micro_block_size,
        rounding_mode="floor",
    )
    kv_micro = torch.div(
        kv_offsets[None, :] - anchor_len,
        micro_block_size,
        rounding_mode="floor",
    )
    spec_visible = (~q_is_anchor) & (~kv_is_anchor) & (kv_micro <= q_micro)
    query_block_visible = anchor_visible | spec_anchor_visible | spec_visible

    for req_idx in range(num_reqs):
        q_start = int(query_start_loc[req_idx].item())
        q_end = int(query_start_loc[req_idx + 1].item())
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        if q_len != block_size:
            raise ValueError(
                "DFlash micro-block attention expects one full draft block "
                f"per request, got q_len={q_len}, block_size={block_size}."
            )
        seq_len = int(seq_lens[req_idx].item())
        context_len = seq_len - q_len
        if context_len < 0:
            raise ValueError(
                f"Invalid DFlash seq_len={seq_len} for q_len={q_len}."
            )
        mask[q_start:q_end, :context_len] = 0
        mask[q_start:q_end, context_len : context_len + q_len] = torch.where(
            query_block_visible,
            torch.zeros((), dtype=torch.int8, device=device),
            torch.ones((), dtype=torch.int8, device=device),
        )

    return mask
