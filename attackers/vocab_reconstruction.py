"""Shared vocab-presence utilities for BCE-style inversion attackers
(LinearProbeAttacker, MLPAttacker): binary token-presence targets, mean
corpus token position, and top-K token reconstruction ordered by that
position. Both attackers use this identically -- only their head
architecture (linear vs. MLP) differs.
"""

from __future__ import annotations

import torch

VOCAB_SIZE = 30522
TOKENIZER_NAME = "bert-base-uncased"


def texts_to_token_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


def build_presence_targets(token_id_lists: list[list[int]], vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    targets = torch.zeros(len(token_id_lists), vocab_size)
    for row, token_ids in enumerate(token_id_lists):
        if token_ids:
            targets[row, token_ids] = 1.0
    return targets


def mean_token_position(token_id_lists: list[list[int]], vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Mean corpus position per token id, used to reorder an unordered
    top-K token prediction into a pseudo-passage at reconstruction time.
    Tokens never seen in the corpus get +inf, so they sort to the end.
    """
    position_sum = torch.zeros(vocab_size)
    position_count = torch.zeros(vocab_size)
    for token_ids in token_id_lists:
        for position, token_id in enumerate(token_ids):
            position_sum[token_id] += position
            position_count[token_id] += 1

    token_order = torch.full((vocab_size,), float("inf"))
    seen = position_count > 0
    token_order[seen] = position_sum[seen] / position_count[seen]
    return token_order


def reconstruct_from_logits(
    logits: torch.Tensor, tokenizer, token_order: torch.Tensor, top_k: int
) -> list[str]:
    """Take the top-K tokens by logit score per row, reorder them by mean
    corpus token position, and join into a pseudo-passage.
    """
    top_ids = torch.topk(logits, k=top_k, dim=1).indices.cpu()
    reconstructions: list[str] = []
    for row in top_ids:
        token_ids = row.tolist()
        token_ids.sort(key=lambda tid: token_order[tid].item())
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        reconstructions.append(tokenizer.convert_tokens_to_string(tokens))
    return reconstructions
