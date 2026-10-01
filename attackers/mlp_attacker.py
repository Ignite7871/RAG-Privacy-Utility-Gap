"""MLP multi-label BCE probe for embedding inversion.

Same attack family as LinearProbeAttacker (Morris et al.-style: predict, for every
vocabulary token, whether it appears in the source passage; reconstruct by re-ordering
the top-scoring tokens by their mean position across the training corpus) but with the
deeper head described in the paper's Section VI-C: d -> 1024 -> 1024 -> V, GELU
activations, LayerNorm after each hidden layer, dropout=0.1 by default (configurable --
see "Regularization" note below). Everything else -- BCE loss over binary token-presence
targets, top-K=16 reconstruction, mean-corpus-position reordering -- is identical to
LinearProbeAttacker; only the head architecture and epoch count differ.

Regularization: validation on MS MARCO/MiniLM found the paper's stated config
(dropout=0.1, no weight decay, 50 epochs) drives training BCE loss to ~0 well before
epoch 50, i.e. the deeper head overfits the alignment set rather than being
undertrained (see results/mlp_loss_curve_minilm.png,
results/mlp_regularization_sweep_minilm.png). dropout and weight_decay are exposed as
constructor arguments so this can be tuned per encoder; defaults are unchanged from the
paper's stated config.
"""

from __future__ import annotations

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
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


class MLPAttacker(InversionAttacker):
    HIDDEN_DIM = 1024

    def __init__(
        self,
        embedding_dim: int,
        top_k: int = 16,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.top_k = top_k
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.HIDDEN_DIM),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.HIDDEN_DIM, VOCAB_SIZE),
        ).to(self.device)

        # Mean corpus position per token id, learned in fit(); tokens never seen in
        # the alignment corpus are pushed to the end when reordering a reconstruction.
        self._token_order = torch.full((VOCAB_SIZE,), float("inf"))

    def fit(
        self,
        alignment_embeddings: torch.Tensor,
        alignment_texts: list[str],
        return_history: bool = False,
        held_out: tuple[torch.Tensor, list[str]] | None = None,
    ) -> dict[str, list[float]] | None:
        """held_out, if given, is an (embeddings, texts) pair not used for training;
        its mean BCE loss is logged once per epoch alongside the training loss, purely
        for diagnosing over/underfitting. Only tracked when return_history=True.
        """
        token_id_lists = texts_to_token_ids(self.tokenizer, alignment_texts)
        self._token_order = mean_token_position(token_id_lists)

        embeddings = alignment_embeddings.to(self.device)
        targets = build_presence_targets(token_id_lists).to(self.device)

        if held_out is not None:
            held_out_embeddings, held_out_texts = held_out
            held_out_embeddings = held_out_embeddings.to(self.device)
            held_out_targets = build_presence_targets(
                texts_to_token_ids(self.tokenizer, held_out_texts)
            ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        n = embeddings.shape[0]
        train_loss_history: list[float] = []
        held_out_loss_history: list[float] = []
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(embeddings[idx])
                loss = loss_fn(logits, targets[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if return_history:
                train_loss_history.append(epoch_loss / num_batches)

            if return_history and held_out is not None:
                self.model.eval()
                with torch.no_grad():
                    held_out_logits = self.model(held_out_embeddings)
                    held_out_loss_history.append(loss_fn(held_out_logits, held_out_targets).item())

        if not return_history:
            return None
        history = {"train_loss": train_loss_history}
        if held_out is not None:
            history["held_out_loss"] = held_out_loss_history
        return history

    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        self.model.eval()
        reconstructions: list[str] = []
        with torch.no_grad():
            for start in range(0, test_embeddings.shape[0], self.batch_size):
                batch = test_embeddings[start : start + self.batch_size].to(self.device)
                logits = self.model(batch)
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

    attacker = MLPAttacker(embedding_dim=embedding_dim, epochs=5, batch_size=2)
    attacker.fit(embeddings, texts)

    test_embeddings = torch.randn(2, embedding_dim)
    reconstructions = attacker.attack(test_embeddings)

    assert len(reconstructions) == test_embeddings.shape[0]
    assert all(isinstance(r, str) for r in reconstructions)
    for r in reconstructions:
        print(repr(r))
    print("smoke test passed")
