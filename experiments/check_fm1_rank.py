"""Bounded follow-up to check_fm1_collapse.py: does AdvEnc collapse actually reduce
embedding dimensionality (fewer usable directions for a linear head to separate
documents on), or something else? Computes the singular value spectrum (via SVD, on
mean-centered embeddings -- i.e. PCA-style) of the same 400-passage embedding matrices
used in the collapse check, for vanilla vs v1-defended vs v2-defended.

Effective rank at 95% variance and participation ratio are two standard ways to
summarize "how many directions actually carry signal" from a singular value spectrum;
top_singular_value_fraction (largest singular value squared / total variance) is a
simpler single-number collapse indicator -- a low-rank collapse concentrates variance
into very few directions, which should show up as a rising top_singular_value_fraction
and a falling effective rank as lambda_priv increases (vanilla -> v1 -> v2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on
# Windows). Empirically `datasets` must be imported before `sentence_transformers`
# specifically, or the process crashes with an access violation -- see
# experiments/advenc_cpu_scale_check.py.
import datasets  # noqa: E402, F401
from sentence_transformers import SentenceTransformer  # noqa: E402, F401

from experiments.check_fm1_collapse import (  # noqa: E402
    CHECKPOINT_DIR,
    N_PASSAGES,
    encode,
    load_encoder,
)

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "fm1_rank_check.json"
VARIANCE_THRESHOLD = 0.95


def rank_stats(embeddings: torch.Tensor) -> dict[str, float]:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values**2
    total_variance = variance.sum()

    cumulative_fraction = torch.cumsum(variance, dim=0) / total_variance
    effective_rank_95 = int((cumulative_fraction < VARIANCE_THRESHOLD).sum().item()) + 1

    participation_ratio = (variance.sum() ** 2 / (variance**2).sum()).item()
    top_singular_value_fraction = (variance[0] / total_variance).item()

    return {
        "effective_rank_95": effective_rank_95,
        "participation_ratio": participation_ratio,
        "top_singular_value_fraction": top_singular_value_fraction,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    texts = test_data["text"][:N_PASSAGES]
    print(f"loaded {len(texts)} passages (msmarco/minilm test split, first {N_PASSAGES})")

    configs = [
        ("vanilla", None),
        ("v1_defended", CHECKPOINT_DIR / "advenc_v1_cpu_minilm.pt"),
        ("v2_defended", CHECKPOINT_DIR / "advenc_v2_cpu_minilm.pt"),
    ]

    rows: list[dict] = []
    for name, checkpoint_path in configs:
        print(f"\n=== {name} ===", flush=True)
        model = load_encoder(checkpoint_path, device)
        embeddings = encode(model, texts).cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stats = rank_stats(embeddings)
        rows.append({"encoder": name, **stats})
        print(
            f"{name}: effective_rank_95={stats['effective_rank_95']} "
            f"participation_ratio={stats['participation_ratio']:.2f} "
            f"top_singular_value_fraction={stats['top_singular_value_fraction']:.4f}"
        )

    print("\n" + "=" * 70)
    print(f"{'encoder':<14} {'effective_rank_95':>18} {'top_singular_value_fraction':>28}")
    for row in rows:
        print(f"{row['encoder']:<14} {row['effective_rank_95']:>18} {row['top_singular_value_fraction']:>28.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
