from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class InversionAttacker(ABC):
    """Shared interface for embedding-inversion attackers.

    Implementations: linear probe, ALGEN, MLP. Experiment scripts loop over
    attackers through this interface without special-casing each one.
    """

    @abstractmethod
    def fit(self, alignment_embeddings: torch.Tensor, alignment_texts: list[str]) -> None:
        """Train the attacker on (embedding, text) pairs from the alignment split."""
        ...

    @abstractmethod
    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        """Reconstruct one text per row of test_embeddings."""
        ...
