"""Sanity check (before accepting Recall@5=0 from
experiments/advenc_retrieval_eval.py as real): is the query-passage relationship in
the GPU-scale defended encoder genuinely destroyed (true-positive similarity
indistinguishable from random negatives), or does a real but weak signal survive that
a ranking metric could hide entirely (true-positive systematically above random, but
still losing to closer negatives)?

An exact zero can come from plumbing rather than from the encoder: the real-ALGEN
checkpoint silently produces empty-string decodes under the wrong transformers version (see
README.md, "Environments"). Recall@5=0/200 has the same shape, so this check rules out
that kind of artifact.

Reuses the exact same 200 queries, ground truth, and corpus/encoder loading path as
advenc_retrieval_eval.py for a direct apples-to-apples comparison -- no ranking
involved this time, just raw similarity scores plus an embedding-norm check to rule
out a query-vs-passage normalization inconsistency in the retrieval harness itself.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on
# Windows). Empirically `datasets` must be imported before `sentence_transformers`
# specifically, or the process crashes with an access violation -- see
# experiments/advenc_cpu_scale_check.py. Must stay the first import in this file.
import datasets  # noqa: E402, F401

from experiments.advenc_cpu_scale_check import encode_texts, load_defended_encoder  # noqa: E402
from experiments.advenc_retrieval_eval import CHECKPOINT_PATH, N_QUERIES, sample_queries_with_ground_truth  # noqa: E402
from experiments.check_fm1_collapse import encode, load_encoder  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "gpu50k_similarity_sanity_check.json"

N_RANDOM_NEGATIVES = 20
SEED = 42


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)

    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    align_texts, test_texts = align_data["text"], test_data["text"]
    corpus_texts = align_texts + test_texts
    n_corpus = len(corpus_texts)

    corpus_index = {text: i for i, text in enumerate(corpus_texts)}
    assert len(corpus_index) == len(corpus_texts), "duplicate passages in the 50k corpus pool"

    print(f"sampling {N_QUERIES} queries with ground truth (same method as the retrieval eval)...", flush=True)
    query_pairs = sample_queries_with_ground_truth(corpus_index, N_QUERIES)
    query_texts = [q for q, _ in query_pairs]
    relevant_indices = [idx for _, idx in query_pairs]

    # Only cached models/checkpoints needed from here -- no more Hub streaming.
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"

    print("\nencoding corpus + queries through the gpu50k defended encoder...", flush=True)
    defended_model = load_defended_encoder(CHECKPOINT_PATH, device)
    defended_align_emb = encode_texts(defended_model, align_texts)
    defended_test_emb = encode_texts(defended_model, test_texts)
    defended_corpus_emb = torch.cat([defended_align_emb, defended_test_emb], dim=0)
    defended_query_emb = encode_texts(defended_model, query_texts)
    del defended_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- embedding norm check: rule out a query-vs-passage normalization bug ----
    print("\n=== embedding L2 norms (should be ~1.0) ===")
    print("10 defended query norms:", [round(defended_query_emb[i].norm(p=2).item(), 4) for i in range(10)])
    print("10 defended passage norms:", [round(defended_corpus_emb[i].norm(p=2).item(), 4) for i in range(10)])

    # ---- direct similarity comparison: true positive vs random negatives ----
    print("\n=== true-positive vs random-negative similarity ===", flush=True)
    true_positive_sims: list[float] = []
    random_negative_means: list[float] = []
    wins = 0

    for i, rel_idx in enumerate(relevant_indices):
        tp_sim = (defended_query_emb[i] @ defended_corpus_emb[rel_idx]).item()
        true_positive_sims.append(tp_sim)

        neg_indices = rng.sample(range(n_corpus), N_RANDOM_NEGATIVES + 1)
        neg_indices = [idx for idx in neg_indices if idx != rel_idx][:N_RANDOM_NEGATIVES]
        neg_sims = (defended_query_emb[i] @ defended_corpus_emb[neg_indices].T).tolist()
        neg_mean = sum(neg_sims) / len(neg_sims)
        random_negative_means.append(neg_mean)

        if tp_sim > neg_mean:
            wins += 1

    mean_tp = sum(true_positive_sims) / len(true_positive_sims)
    mean_neg = sum(random_negative_means) / len(random_negative_means)

    print(f"mean true-positive similarity:   {mean_tp:.4f}")
    print(f"mean random-negative similarity: {mean_neg:.4f}")
    print(f"queries where true-positive > random-negative mean: {wins}/{N_QUERIES}")

    results = {
        "mean_true_positive_similarity": mean_tp,
        "mean_random_negative_similarity": mean_neg,
        "true_positive_wins": wins,
        "n_queries": N_QUERIES,
        "n_random_negatives_per_query": N_RANDOM_NEGATIVES,
        "sample_query_norms": [defended_query_emb[i].norm(p=2).item() for i in range(10)],
        "sample_passage_norms": [defended_corpus_emb[i].norm(p=2).item() for i in range(10)],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
