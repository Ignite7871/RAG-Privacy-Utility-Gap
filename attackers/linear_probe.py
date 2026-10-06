"""Linear multi-label BCE probe for embedding inversion.

A linear form of the bag-of-words inversion setting of Song and Raghunathan,
"Information Leakage in Embedding Models" (CCS 2020): a linear head predicts, for every
vocabulary token, whether it appears in the source passage, and reconstruction re-orders
the top-scoring tokens by their mean position across the training corpus.

This is not the ALGEN method of Chen/Xu/Bjerva (ACL 2025, github.com/siebeniris/ALGEN);
that attack is run through algen.py.
"""

from __future__ import annotations

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
from transformers import AutoTokenizer

from attackers.base import InversionAttacker
from attackers.vocab_reconstruction import (
    TOKENIZER_NAME,
    VOCAB_SIZE,
    build_presence_targets,
    mean_token_position,
    reconstruct_from_logits,
    texts_to_token_ids,
)
import torch


class LinearProbeAttacker(InversionAttacker):
    def __init__(
        self,
        embedding_dim: int,
        top_k: int = 16,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        device: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.W = torch.nn.Linear(embedding_dim, VOCAB_SIZE).to(self.device)

        # Mean corpus position per token id, learned in fit(); tokens never seen in
        # the alignment corpus are pushed to the end when reordering a reconstruction.
        self._token_order = torch.full((VOCAB_SIZE,), float("inf"))

    def fit(self, alignment_embeddings: torch.Tensor, alignment_texts: list[str]) -> None:
        token_id_lists = texts_to_token_ids(self.tokenizer, alignment_texts)
        self._token_order = mean_token_position(token_id_lists)

        embeddings = alignment_embeddings.to(self.device)
        targets = build_presence_targets(token_id_lists).to(self.device)

        optimizer = torch.optim.Adam(self.W.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        n = embeddings.shape[0]
        self.W.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self.W(embeddings[idx])
                loss = loss_fn(logits, targets[idx])
                loss.backward()
                optimizer.step()

    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        self.W.eval()
        reconstructions: list[str] = []
        with torch.no_grad():
            for start in range(0, test_embeddings.shape[0], self.batch_size):
                batch = test_embeddings[start : start + self.batch_size].to(self.device)
                logits = self.W(batch)
                reconstructions += reconstruct_from_logits(
                    logits, self.tokenizer, self._token_order, self.top_k
                )
        return reconstructions


if __name__ == "__main__":
    torch.manual_seed(0)

    embedding_dim = 16
    texts = [
        "the cat sat on the mat",
        "a dog ran across the yard",
        "the sun rose over the mountains",
        "she read a book by the fire",
    ]
    embeddings = torch.randn(len(texts), embedding_dim)

    attacker = LinearProbeAttacker(embedding_dim=embedding_dim, epochs=5, batch_size=2)
    attacker.fit(embeddings, texts)

    test_embeddings = torch.randn(2, embedding_dim)
    reconstructions = attacker.attack(test_embeddings)

    assert len(reconstructions) == test_embeddings.shape[0]
    assert all(isinstance(r, str) for r in reconstructions)
    for r in reconstructions:
        print(repr(r))
    print("smoke test passed")
