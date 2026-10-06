"""RotatingEncoderDefense: a keyed random orthogonal rotation applied to stored and query
embeddings, with the key advancing per rotation window ("epoch").

Row-vector convention: transform(X, k) = X @ R_k, where R_k is a d x d orthogonal matrix.
This is the setting of the paper's key-rotation section, whose two propositions the code
relies on.

What the rotation guarantees
  * Retrieval neutrality (Proposition 2). For any orthogonal R, <Rx, Ry> = <x, y> in exact
    arithmetic, so inner-product rankings, Recall@k and NDCG@k are identical to the
    unrotated ones whenever the corpus and the queries are rotated under the same R_k. In
    float32 the Gram matrix agrees to about 1e-6 (the self-test below prints the value).
  * Equivariance of ridge inversion (Proposition 3). W(XR, T) = R^T W(X, T), so an attacker
    whose alignment pairs and test embeddings come from the same window does exactly as
    well as against the unrotated encoder (approximately for the Adam-trained
    LinearProbe, whose initialisation is not rotation-invariant). Rotation does not make
    inversion harder in any window the attacker can observe.

What it protects against
  * Stale material only. A probe fit under R_j and applied under R_k != R_j sees inputs
    rotated by the independent Haar-random matrix R_k R_j^T and is left with little beyond
    the token prior. The measured effect is in results/rotation_both_metrics.csv.

Scope (the paper's assumptions)
  (i)   the key is secret and windows use independent Haar-random rotations;
  (ii)  stored and query embeddings are rotated under the same key;
  (iii) the attacker holds no row-aligned snapshots of the index from two windows: an
        orthogonal Procrustes fit on m >= d matched rows recovers R_j^T R_k and undoes the
        rotation (results/rotation_snapshot_linking.csv);
  (iv)  the attacker's pairs per window are capped by the defender.
Nothing is claimed against an attacker who violates (iii) or exceeds the cap in (iv). Treat
rotation as a way to invalidate leaked material, not as a defense against an attacker who
can query the current encoder.

Implementation caveat: R_k is drawn from torch's non-cryptographic generator seeded with a
31-bit function of (seed, epoch), so assumption (i) is idealised here. A deployment would
derive R_k from a keyed cryptographic generator with a full-length secret.

This file is ONLY the rotation primitive and its correctness tests (retrieval neutrality,
determinism, orthogonality). The attacker-side evaluation is in
experiments/rotation_eval_both_metrics.py (same window vs. next window) and
experiments/prior_baseline_and_snapshot_linking.py (embedding-free floor, snapshot linking).
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
