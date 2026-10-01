"""Direct check of the paper's FM1 contrastive-collapse signature (Fig 1): does
AdvEnc-defended encoder training actually compress passage embeddings toward each
other (rising mean pairwise cosine similarity), independent of what the downstream
attack numbers in results/advenc/cpu_scale_downstream_check.json show?

This is a direct measurement, no attacker involved: 400 passages (matching the paper's
Fig 1 n=400), all pairwise cosine similarities (400 choose 2 = 79,800 pairs) for the
vanilla encoder and each CPU-scale defended checkpoint, plus linear CKA (Kornblith et
al. 2019, same formula as experiments/algen_alignment_quality.py) between vanilla and
each defended encoder's embeddings on the same passages.

Paper's Fig 1 numbers for reference: vanilla mean=0.024/std=0.118, v2-equivalent
defended mean=0.107/std=0.084.
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
import datasets  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "fm1_collapse_check.json"

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
N_PASSAGES = 400


def load_encoder(checkpoint_path: Path | None, device: torch.device) -> SentenceTransformer:
    model = SentenceTransformer(BASE_MODEL_NAME).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    embeddings = model.encode(texts, batch_size=128, convert_to_tensor=True, show_progress_bar=False)
    return F.normalize(embeddings, p=2, dim=1)


def pairwise_cosine_stats(embeddings: torch.Tensor) -> tuple[float, float]:
    sim = embeddings @ embeddings.T
    n = sim.shape[0]
    row_idx, col_idx = torch.triu_indices(n, n, offset=1)
    upper = sim[row_idx, col_idx]
    return upper.mean().item(), upper.std().item()


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Standard linear CKA (Kornblith et al. 2019), efficient feature-space form:
    CKA(X,Y) = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    where Xc, Yc are X, Y column-centered (n_samples x n_features each). Matches
    experiments/algen_alignment_quality.py's implementation.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    numerator = (Xc.T @ Yc).norm(p="fro") ** 2
    denominator = (Xc.T @ Xc).norm(p="fro") * (Yc.T @ Yc).norm(p="fro")
    return (numerator / denominator).item()


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

    embeddings_by_name: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    for name, checkpoint_path in configs:
        print(f"\n=== {name} ===", flush=True)
        model = load_encoder(checkpoint_path, device)
        embeddings = encode(model, texts).cpu()
        embeddings_by_name[name] = embeddings
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_cos, std_cos = pairwise_cosine_stats(embeddings)
        cka_vs_vanilla = None if name == "vanilla" else linear_cka(embeddings_by_name["vanilla"], embeddings)

        rows.append(
            {
                "encoder": name,
                "mean_cosine_sim": mean_cos,
                "std_cosine_sim": std_cos,
                "CKA_vs_vanilla": cka_vs_vanilla,
            }
        )
        cka_str = f"{cka_vs_vanilla:.4f}" if cka_vs_vanilla is not None else "--"
        print(f"{name}: mean_cos={mean_cos:.4f} std_cos={std_cos:.4f} CKA_vs_vanilla={cka_str}")

    print("\n" + "=" * 70)
    print(f"{'encoder':<14} {'mean_cosine_sim':>16} {'std_cosine_sim':>16} {'CKA_vs_vanilla':>16}")
    for row in rows:
        cka_str = f"{row['CKA_vs_vanilla']:.4f}" if row["CKA_vs_vanilla"] is not None else "--"
        print(f"{row['encoder']:<14} {row['mean_cosine_sim']:>16.4f} {row['std_cosine_sim']:>16.4f} {cka_str:>16}")

    print("\npaper's Fig 1 reference: vanilla mean=0.024/std=0.118, v2-equivalent defended mean=0.107/std=0.084")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
