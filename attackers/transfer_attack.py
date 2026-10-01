"""TransferAttacker: query-free, transferable embedding inversion attack, per
Huang, Tsai, Hsiao, Lin, Lin (2024), "Transferable Embedding Inversion
Attack: Uncovering Privacy Risks in Text Embeddings without Model Queries"
(arXiv:2406.10280).

Unlike GEIAAttacker/AlgenAttacker/Vec2Text, this attacker never queries the
victim encoder at attack time, and only needs a *small* leaked dataset of
(surrogate_embedding, victim_embedding) pairs for the same texts at fit()
time -- not live access to the victim model. The attack has two stages:

  1. Adapter training: a public "surrogate" encoder's embeddings (this
     codebase uses the already-cached gtr-t5-base embeddings for the leaked
     alignment texts as the surrogate, matching the reference paper's use of
     GTR-base as a generic surrogate) are mapped into the victim's embedding
     space by a trainable MLP adapter, trained on the small leaked pair set
     with two losses: an intra-consistency term (MSE between adapted
     surrogate embeddings and true victim embeddings) and an
     inter-consistency term (matching the pairwise cosine-similarity
     structure between the two spaces), following Section 4.2 of the paper.
  2. Decoder training: a GEIA-style generative decoder (reusing
     GEIAAttacker's training loop) is trained to reconstruct text from the
     *adapted* surrogate embeddings, not the true victim embeddings.

At attack time, the trained decoder is applied directly to real victim
embeddings it was never trained on -- the "transfer" step -- since the
adapter's training objective pulled the adapted-surrogate distribution
toward the victim embedding distribution.

Deviations from the reference implementation
----------------------------------------------
1. The reference paper adds a third, adversarial-training term (a
   discriminator forcing adapted-surrogate embeddings to be indistinguishable
   from true victim embeddings) to further improve transfer quality. This is
   NOT implemented here -- only the intra- and inter-consistency losses are
   used. This likely understates the reference paper's reported transfer
   quality.
2. The reference paper trains its own surrogate encoder end-to-end (a public
   encoder plus adapter, fine-tuned together); here the surrogate encoder
   (gtr-t5-base) is used as a frozen, already-cached embedding source, and
   only the adapter on top of it is trained. This matches their "linear
   adapter on a frozen surrogate" ablation, not their strongest full
   fine-tuning configuration.
3. The decoder stage reuses GEIAAttacker's training loop verbatim (GPT-2,
   single-linear-projection prefix conditioning) rather than the reference
   paper's own DialoGPT-based decoder, for consistency with this codebase's
   other generative baseline.
"""

from __future__ import annotations

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.base import InversionAttacker
from attackers.geia import GEIAAttacker
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferAttacker(InversionAttacker):
    def __init__(
        self,
        surrogate_dim: int,
        victim_dim: int,
        decoder_model_name: str = "gpt2",
        adapter_hidden: int = 512,
        max_length: int = 32,
        decoder_lr: float = 5e-5,
        decoder_batch_size: int = 16,
        decoder_epochs: int = 15,
        adapter_lr: float = 1e-3,
        adapter_epochs: int = 300,
        adapter_batch_size: int = 128,
        inter_consistency_weight: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adapter_lr = adapter_lr
        self.adapter_epochs = adapter_epochs
        self.adapter_batch_size = adapter_batch_size
        self.inter_consistency_weight = inter_consistency_weight

        self.adapter = nn.Sequential(
            nn.Linear(surrogate_dim, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, victim_dim),
        ).to(self.device)

        self._decoder = GEIAAttacker(
            embedding_dim=victim_dim,
            decoder_model_name=decoder_model_name,
            max_length=max_length,
            lr=decoder_lr,
            batch_size=decoder_batch_size,
            epochs=decoder_epochs,
            device=str(self.device),
        )

    def _fit_adapter(self, surrogate_embeddings: torch.Tensor, victim_embeddings: torch.Tensor) -> None:
        surrogate = surrogate_embeddings.to(self.device)
        victim = victim_embeddings.to(self.device)
        n = surrogate.shape[0]

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.adapter_lr)
        self.adapter.train()
        for epoch in range(self.adapter_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, self.adapter_batch_size):
                idx = perm[start : start + self.adapter_batch_size]
                batch_surrogate = surrogate[idx]
                batch_victim = victim[idx]

                optimizer.zero_grad()
                adapted = self.adapter(batch_surrogate)

                # Intra-consistency: adapted embedding should match the true
                # victim embedding for the same text.
                intra_loss = F.mse_loss(adapted, batch_victim)

                # Inter-consistency: the pairwise cosine-similarity structure
                # within a batch should match between the adapted-surrogate
                # space and the true victim space, even where individual
                # vectors don't align exactly.
                adapted_norm = F.normalize(adapted, p=2, dim=1)
                victim_norm = F.normalize(batch_victim, p=2, dim=1)
                adapted_sim = adapted_norm @ adapted_norm.T
                victim_sim = victim_norm @ victim_norm.T
                inter_loss = F.mse_loss(adapted_sim, victim_sim)

                loss = intra_loss + self.inter_consistency_weight * inter_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 50 == 0 or epoch == self.adapter_epochs - 1:
                print(f"  [TransferAttack/adapter] epoch {epoch + 1}/{self.adapter_epochs}  loss={epoch_loss / num_batches:.4f}")

    def fit(
        self,
        alignment_embeddings: torch.Tensor,
        alignment_texts: list[str],
        surrogate_embeddings: torch.Tensor,
    ) -> None:
        """alignment_embeddings are the leaked TRUE victim embeddings for the
        same (small) set of texts as surrogate_embeddings -- both required to
        train the adapter. See module docstring: surrogate_embeddings should
        come from a public encoder (this codebase uses cached gtr-t5-base
        embeddings), not the victim encoder itself.
        """
        if surrogate_embeddings.shape[0] != alignment_embeddings.shape[0]:
            raise ValueError(
                f"surrogate_embeddings ({surrogate_embeddings.shape[0]} rows) and "
                f"alignment_embeddings ({alignment_embeddings.shape[0]} rows) must "
                "be paired 1:1 for the same leaked texts"
            )

        self._fit_adapter(surrogate_embeddings, alignment_embeddings)

        self.adapter.eval()
        with torch.no_grad():
            adapted = self.adapter(surrogate_embeddings.to(self.device)).cpu()

        print("  [TransferAttack] adapter trained; training decoder on adapted surrogate embeddings...")
        self._decoder.fit(adapted, alignment_texts)

    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        """test_embeddings are REAL victim embeddings the attacker never saw
        at fit() time -- no adapter is applied here, since the decoder was
        trained to expect vectors already living in the victim embedding
        space (the adapter's target space). This is the "transfer" step.
        """
        return self._decoder.attack(test_embeddings)


if __name__ == "__main__":
    torch.manual_seed(0)

    surrogate_dim, victim_dim = 24, 16
    texts = [
        "the cat sat on the mat",
        "a dog ran across the yard",
        "the sun rose over the mountains",
        "she read a book by the fire",
    ]
    surrogate_embeddings = torch.randn(len(texts), surrogate_dim)
    victim_embeddings = torch.randn(len(texts), victim_dim)

    attacker = TransferAttacker(
        surrogate_dim=surrogate_dim,
        victim_dim=victim_dim,
        decoder_epochs=2,
        decoder_batch_size=2,
        adapter_epochs=5,
        adapter_batch_size=2,
    )
    attacker.fit(victim_embeddings, texts, surrogate_embeddings)

    test_embeddings = torch.randn(2, victim_dim)
    reconstructions = attacker.attack(test_embeddings)

    assert len(reconstructions) == test_embeddings.shape[0]
    assert all(isinstance(r, str) for r in reconstructions)
    for r in reconstructions:
        print(repr(r))
    print("smoke test passed")
