"""Measures ALGEN's alignment quality directly: how well a closed-form ridge
map can project each victim encoder's embedding space into the flan-t5-small
generator's own embedding space, and how similar the two spaces already are
before any alignment (via linear CKA). No beam search / generation involved,
so this runs in seconds per encoder -- unlike experiments/run_algen.py's full
attack sweep.

MUST be run under .venv-algen-legacy, not .venv312 -- it instantiates
AlgenAttacker, which loads the ALGEN generator checkpoint via transformers.
See claude.md: "Dual-venv setup". The checkpoint is incompatible with
transformers>=5.x and produces silently wrong *embeddings* there too, not
just wrong generations -- so R^2/CKA computed under the main venv would be
untrustworthy the same way attack() is.

CKA methodology: standard linear CKA (Kornblith et al. 2019, "Similarity of
Neural Network Representations Revisited"), via the efficient feature-space
formula. This repo doesn't have the paper's Section V text available to
consult directly -- if Section V used a different variant (kernel/RBF CKA,
minibatch/unbiased HSIC, etc.), these numbers won't match it exactly and
this should be revisited against the actual paper text.
"""

from __future__ import annotations

import csv
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.algen import AlgenAttacker  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT = REPO_ROOT / "checkpoints" / "algen_generator" / "checkpoint_epoch_99.pt"
OUT_CSV = REPO_ROOT / "results" / "algen" / "alignment_quality.csv"
N_ALIGN = 2000

DATASETS = ["msmarco", "nq"]
ENCODERS = {"minilm": 384, "mpnet": 768, "gtr": 768, "bge": 1024}


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Standard linear CKA (Kornblith et al. 2019), efficient feature-space form:
    CKA(X,Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    where Xc, Yc are X, Y column-centered (n_samples x n_features each).
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    numerator = (Xc.T @ Yc).norm(p="fro") ** 2
    denominator = (Xc.T @ Xc).norm(p="fro") * (Yc.T @ Yc).norm(p="fro")
    return (numerator / denominator).item()


def ridge_r2(X: torch.Tensor, Y: torch.Tensor, T: torch.Tensor) -> float:
    """In-sample R^2 of the closed-form ridge fit: how much of the variance in
    the generator's own embedding space Y is explained by the linear
    projection X @ T of the victim encoder's embeddings. Evaluated on the same
    alignment set used to fit T (this measures alignability of the two
    representation spaces, not generalization to held-out data).
    """
    Y_hat = X @ T
    ss_res = ((Y - Y_hat) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot


def main() -> None:
    attacker = AlgenAttacker(
        generator_checkpoint=str(CHECKPOINT), generator_model_name="google/flan-t5-small"
    )

    rows = []
    for dataset, encoder in product(DATASETS, ENCODERS):
        align_path = CACHE_DIR / f"{dataset}_{encoder}_align.pt"
        if not align_path.exists():
            print(f"[skip] {dataset}/{encoder}: {align_path} not found")
            continue

        t0 = time.perf_counter()
        align_data = torch.load(align_path, weights_only=False)
        X = align_data["embeddings"][:N_ALIGN].to(attacker.device)
        texts = align_data["text"][:N_ALIGN]

        # Reuses AlgenAttacker's real fit() -- the same closed-form ridge
        # solve used by the full attack pipeline -- then reaches into its
        # private _target_embeddings() to get the same Y it aligned against,
        # for the R^2/CKA diagnostics below.
        attacker.fit(X, texts)
        Y = attacker._target_embeddings(texts)

        r2 = ridge_r2(X, Y, attacker.T)
        cka = linear_cka(X, Y)
        elapsed = time.perf_counter() - t0

        rows.append({
            "dataset": dataset, "encoder": encoder, "dim": ENCODERS[encoder],
            "ridge_R2": r2, "CKA_to_generator_space": cka,
        })
        print(f"{dataset}/{encoder} (d={ENCODERS[encoder]}): R^2={r2:.4f} CKA={cka:.4f} ({elapsed:.2f}s)")

    print("\n" + "=" * 62)
    print(f"{'dataset':<10} {'encoder':<8} {'dim':>5} {'ridge_R2':>10} {'CKA_to_generator_space':>24}")
    for row in rows:
        print(
            f"{row['dataset']:<10} {row['encoder']:<8} {row['dim']:>5} "
            f"{row['ridge_R2']:>10.4f} {row['CKA_to_generator_space']:>24.4f}"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "encoder", "dim", "ridge_R2", "CKA_to_generator_space"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
