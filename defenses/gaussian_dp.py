"""GaussianDPDefense: the Gaussian mechanism of differential privacy, applied to the
coordinates of stored embeddings (an alternative to adversarial encoder training).

sigma = sqrt(2*ln(1.25/delta)) / epsilon (Dwork et al. 2006 Gaussian mechanism, delta=1e-5
fixed). This sigma corresponds to an L2 sensitivity of 1, the norm bound of a unit-norm
embedding; treating neighbouring records as arbitrary unit vectors gives sensitivity up to
2, which would double sigma at a given epsilon. epsilon=inf maps to sigma=0 (no noise, the
vanilla baseline). Noise is i.i.d. per coordinate, then the embedding is L2-renormalised,
matching the L2-normalised-embedding threat model. At large epsilon the formal guarantee is
weak, so protection measured there is empirical, not a formal privacy bound.
"""

from __future__ import annotations

import math

import torch


class GaussianDPDefense:
    def __init__(self, delta: float = 1e-5, seed: int = 42) -> None:
        self.delta = delta
        self.seed = seed

    @staticmethod
    def sigma_from_epsilon(epsilon: float, delta: float = 1e-5) -> float:
        if math.isinf(epsilon):
            return 0.0
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def apply(
        self,
        embeddings: torch.Tensor,
        epsilon: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Adds i.i.d. N(0, sigma^2) noise per coordinate, then L2-renormalises.

        `generator` lets a caller draw independent noise across multiple calls (e.g.
        corpus vs. queries) by reusing one advancing generator, rather than every call
        re-seeding to the same state. If omitted, a fresh generator seeded by
        `self.seed` is used (deterministic single-call convenience/tests).
        """
        sigma = self.sigma_from_epsilon(epsilon, self.delta)
        if sigma == 0.0:
            return embeddings / embeddings.norm(p=2, dim=1, keepdim=True)

        if generator is None:
            generator = torch.Generator().manual_seed(self.seed)
        noise = torch.randn(embeddings.shape, generator=generator) * sigma
        noised = embeddings + noise
        return noised / noised.norm(p=2, dim=1, keepdim=True)


if __name__ == "__main__":
    torch.manual_seed(0)

    defense = GaussianDPDefense()
    embeddings = torch.randn(10, 384)
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)

    # ---- epsilon=inf is a no-op (sigma=0), returns L2-normalised input unchanged ----
    unchanged = defense.apply(embeddings, epsilon=float("inf"))
    assert torch.allclose(embeddings, unchanged, atol=1e-6), "epsilon=inf should be a no-op"
    print("[PASS] epsilon=inf is a no-op on already-normalised input")

    # ---- output is always unit-norm regardless of epsilon ----
    for eps in [1, 10, 100]:
        noised = defense.apply(embeddings, epsilon=eps)
        norms = noised.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"eps={eps} not renormalised"
    print("[PASS] output is L2-normalised at all tested epsilon")

    # ---- sigma formula sanity: smaller epsilon -> larger sigma -> more distortion ----
    sigmas = [GaussianDPDefense.sigma_from_epsilon(eps) for eps in [1, 10, 100]]
    assert sigmas[0] > sigmas[1] > sigmas[2], "sigma should decrease as epsilon increases"
    print(f"[PASS] sigma decreases monotonically with epsilon: {sigmas}")

    # ---- explicit formula check at delta=1e-5 ----
    expected_sigma_eps1 = math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
    assert abs(sigmas[0] - expected_sigma_eps1) < 1e-9
    print(f"[PASS] sigma(epsilon=1, delta=1e-5) = {sigmas[0]:.4f}")

    print("\nall correctness tests passed")
