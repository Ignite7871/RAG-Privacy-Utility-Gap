"""RotatingEncoderDefense: a keyed random orthogonal rotation applied post-encoding,
rotated on a schedule (Option A).

Design rationale (see conversation context, not reproduced here): DP noise and
AdvEnc's gradient-based adversarial training both degrade the geometry retrieval
depends on as a side effect of resisting inversion -- there's no guarantee the two
objectives can be separated when the defense is learned. An orthogonal transformation
sidesteps that tradeoff by construction: for any orthogonal R, <Rx, Ry> = <x, y>
exactly (not approximately), so Recall@k/NDCG are mathematically identical to
vanilla, regardless of R. Inversion resistance instead comes from *rotating* R over
time: an attacker who collects alignment pairs during rotation window k learns a
linear head calibrated to R_k's coordinate system; once the defender advances to
R_{k+1}, that head is stale and the attacker needs a fresh O(d) alignment samples to
re-adapt (Theorem 1). The corollary: rotate before an adversary can plausibly submit
O(d) queries (empirically n*~200 for MiniLM, d=384) to keep an adaptive attacker
perpetually below saturation.

This file is ONLY the rotation primitive + its correctness tests (retrieval-neutrality,
determinism, orthogonality). The attacker-side evaluation (query-budget-per-window
sweep, the FM3-style bounded-saturation curve) is a separate, later piece.
"""

from __future__ import annotations

import torch


class RotatingEncoderDefense:
    """Generates deterministic, keyed d x d orthogonal rotation matrices, one per
    rotation "epoch" (window index), and applies them to embeddings.
    """

    def __init__(self, dim: int, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._cache: dict[int, torch.Tensor] = {}

    def _seed_for_epoch(self, epoch: int) -> int:
        # Large prime multiplier keeps (seed, epoch) pairs well separated across a
        # wide range of epochs -- avoids the accidental aliasing a plain `seed +
        # epoch` could produce between e.g. (seed=42, epoch=1) and (seed=43, epoch=0)
        # if two defense instances with different seeds are ever compared.
        return (self.seed * 1_000_003 + epoch) % (2**31 - 1)

    def get_rotation(self, epoch: int) -> torch.Tensor:
        """Deterministically generates (and caches) a dim x dim orthogonal matrix for
        the given epoch, via QR decomposition of a seed-and-epoch-derived Gaussian
        random matrix. Repeated calls for the same epoch return the identical cached
        tensor without recomputing.
        """
        if epoch not in self._cache:
            generator = torch.Generator().manual_seed(self._seed_for_epoch(epoch))
            gaussian = torch.randn(self.dim, self.dim, generator=generator)
            q, r = torch.linalg.qr(gaussian)

            # QR is only unique up to the signs of R's diagonal; fix them so the
            # result is a deterministic, Haar-uniformly-distributed orthogonal
            # matrix (Mezzadri, "How to generate random matrices from the classical
            # compact groups", 2007) rather than an artifact of whatever sign
            # convention the LAPACK backend happens to emit.
            diag_sign = torch.diagonal(r).sign()
            diag_sign[diag_sign == 0] = 1.0
            q = q * diag_sign.unsqueeze(0)

            self._cache[epoch] = q

        return self._cache[epoch]

    def transform(self, embeddings: torch.Tensor, epoch: int) -> torch.Tensor:
        """Applies get_rotation(epoch) to a batch of embeddings: embeddings @ R_epoch."""
        rotation = self.get_rotation(epoch).to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings @ rotation


if __name__ == "__main__":
    torch.manual_seed(0)

    DIM = 384
    defense = RotatingEncoderDefense(dim=DIM, seed=42)

    # ---- retrieval-neutrality: pairwise inner products preserved exactly ----
    vectors = torch.randn(100, DIM)
    vectors = vectors / vectors.norm(p=2, dim=1, keepdim=True)

    gram_before = vectors @ vectors.T
    rotated = defense.transform(vectors, epoch=0)
    gram_after = rotated @ rotated.T

    max_diff = (gram_before - gram_after).abs().max().item()
    assert torch.allclose(gram_before, gram_after, atol=1e-5), (
        f"pairwise inner products NOT preserved under rotation, max abs diff={max_diff}"
    )
    print(f"[PASS] pairwise inner products preserved under rotation (max abs diff={max_diff:.2e})")

    # ---- determinism / caching ----
    r0_first = defense.get_rotation(0)
    r0_second = defense.get_rotation(0)
    assert torch.equal(r0_first, r0_second), "get_rotation(0) is not deterministic across calls"
    print("[PASS] get_rotation(0) called twice returns the identical matrix")

    r1 = defense.get_rotation(1)
    assert not torch.allclose(r0_first, r1), "get_rotation(0) and get_rotation(1) produced the same matrix"
    max_abs = (r0_first - r1).abs().max().item()
    print(f"[PASS] get_rotation(0) and get_rotation(1) are different matrices (max abs diff={max_abs:.4f})")

    # ---- orthogonality ----
    for epoch, r in [(0, r0_first), (1, r1)]:
        identity_error = (r @ r.T - torch.eye(DIM)).abs().max().item()
        assert identity_error < 1e-4, f"epoch {epoch} rotation is not orthogonal, max |R R^T - I|={identity_error}"
        print(f"[PASS] epoch {epoch} rotation is orthogonal (max |R R^T - I|={identity_error:.2e})")

    print("\nall correctness tests passed")
