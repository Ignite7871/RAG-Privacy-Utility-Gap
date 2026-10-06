"""Follow-up to check_fm1_collapse.py and check_fm1_rank.py: does v2's raw
pairwise-cosine-similarity collapse come from a shared-direction/centroid shift rather
than reduced effective rank? If so, this would also explain the non-adaptive vs. adaptive
gap seen in check_fm1_collapse.py and advenc_cpu_scale_check.py: a non-adaptive attacker
trained on vanilla embedding statistics would be thrown off by a shifted centroid, while an
adaptive attacker retraining on the defended distribution does not care where the centroid
sits.

Uses the same 400 passages / three encoders (vanilla, v1_defended, v2_defended) as the
other two checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on
# Windows). Empirically `datasets` must be imported before `sentence_transformers`
# specifically, or the process crashes with an access violation -- see
# experiments/advenc_cpu_scale_check.py.
import datasets  # noqa: E402, F401
from sentence_transformers import SentenceTransformer  # noqa: E402, F401

from experiments.check_fm1_collapse import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    N_PASSAGES,
    encode,
    load_encoder,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT_PATH = REPO_ROOT / "results" / "advenc" / "fm1_centroid_shift_check.json"


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

    centroids: dict[str, torch.Tensor] = {}
    for name, checkpoint_path in configs:
        print(f"\n=== {name} ===", flush=True)
        model = load_encoder(checkpoint_path, device)
        embeddings = encode(model, texts).cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        centroid = embeddings.mean(dim=0)
        centroids[name] = centroid
        print(f"{name}: centroid_norm={centroid.norm(p=2).item():.4f}")

    results = {}
    for name in ["vanilla", "v1_defended", "v2_defended"]:
        results[name] = {"centroid_norm": centroids[name].norm(p=2).item()}

    for name in ["v1_defended", "v2_defended"]:
        cos_sim = F.cosine_similarity(centroids["vanilla"].unsqueeze(0), centroids[name].unsqueeze(0)).item()
        l2_dist = (centroids["vanilla"] - centroids[name]).norm(p=2).item()
        results[name]["cosine_sim_vs_vanilla_centroid"] = cos_sim
        results[name]["l2_distance_vs_vanilla_centroid"] = l2_dist

    print("\n" + "=" * 70)
    print(f"{'encoder':<14} {'centroid_norm':>14} {'cos_sim_vs_vanilla':>20} {'l2_dist_vs_vanilla':>20}")
    for name in ["vanilla", "v1_defended", "v2_defended"]:
        row = results[name]
        cos_sim = row.get("cosine_sim_vs_vanilla_centroid")
        l2_dist = row.get("l2_distance_vs_vanilla_centroid")
        cos_str = f"{cos_sim:.4f}" if cos_sim is not None else "--"
        l2_str = f"{l2_dist:.4f}" if l2_dist is not None else "--"
        print(f"{name:<14} {row['centroid_norm']:>14.4f} {cos_str:>20} {l2_str:>20}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
