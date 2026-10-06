"""Retrieval-quality evaluation (Recall@5/10, NDCG@10) for the GPU-scale AdvEnc-v2
encoder; the query-sampling and scoring helpers are reused by the other retrieval scripts.

Protocol (symmetric, deployment-style): 200 MS MARCO queries are encoded once with the
vanilla encoder and once with the defended encoder; each is evaluated against a corpus
index built with the SAME encoder (vanilla queries vs vanilla-encoded corpus; defended
queries vs defended-encoded corpus). This matches a deployment that switches to the
defended encoder end to end, rather than mixing clean queries with a perturbed index.

Corpus: the 50,000-passage MS MARCO pool used throughout the repository
(data/cache/msmarco_minilm_{align,test}.pt, 40k+10k). Queries and their ground-truth
relevant passage are sampled by re-streaming MS MARCO v2.1 and keeping rows whose
is_selected passage text is a member of that same 50k-passage pool. The pool was built
from the passage_text of that stream's early rows (data/encode.py), so nearly every early
row's own selected passage is already in it and little of the stream needs to be scanned.
"""

from __future__ import annotations

import json
import math
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
from experiments.check_fm1_collapse import encode, load_encoder  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "advenc_v2_gpu50k_minilm.pt"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "gpu50k_retrieval_eval.json"

N_QUERIES = 200
TOP_K = 10


def sample_queries_with_ground_truth(
    corpus_index: dict[str, int], n_queries: int
) -> list[tuple[str, int]]:
    ds = datasets.load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    pairs: list[tuple[str, int]] = []
    for row in ds:
        for text, is_selected in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]):
            if is_selected and text in corpus_index:
                pairs.append((row["query"], corpus_index[text]))
                break
        if len(pairs) >= n_queries:
            break
    if len(pairs) < n_queries:
        raise RuntimeError(f"only found {len(pairs)} queries with ground truth in the corpus, needed {n_queries}")
    return pairs[:n_queries]


def recall_and_ndcg(
    query_emb: torch.Tensor, corpus_emb: torch.Tensor, relevant_indices: list[int], k: int = TOP_K
) -> dict[str, float]:
    sims = query_emb @ corpus_emb.T
    top_k_indices = torch.topk(sims, k=k, dim=1).indices.tolist()

    n = len(relevant_indices)
    recall_5 = 0
    recall_10 = 0
    ndcg_10_total = 0.0
    for row, rel_idx in zip(top_k_indices, relevant_indices):
        if rel_idx in row[:5]:
            recall_5 += 1
        if rel_idx in row:
            recall_10 += 1
            rank = row.index(rel_idx) + 1
            ndcg_10_total += 1.0 / math.log2(rank + 1)

    return {
        "recall@5": recall_5 / n,
        "recall@10": recall_10 / n,
        "ndcg@10": ndcg_10_total / n,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    vanilla_align_emb, align_texts = align_data["embeddings"], align_data["text"]
    vanilla_test_emb, test_texts = test_data["embeddings"], test_data["text"]

    corpus_texts = align_texts + test_texts
    vanilla_corpus_emb = torch.cat([vanilla_align_emb, vanilla_test_emb], dim=0)
    print(f"corpus: {len(corpus_texts)} passages (vanilla embeddings already cached)")

    corpus_index = {text: i for i, text in enumerate(corpus_texts)}
    assert len(corpus_index) == len(corpus_texts), "duplicate passages in the 50k corpus pool"

    print(f"sampling {N_QUERIES} queries with ground truth in the corpus...", flush=True)
    query_pairs = sample_queries_with_ground_truth(corpus_index, N_QUERIES)
    query_texts = [q for q, _ in query_pairs]
    relevant_indices = [idx for _, idx in query_pairs]

    # Everything from here only loads already-cached models/checkpoints -- no more
    # Hub streaming needed. Force offline mode to avoid a recurring flaky HEAD
    # request this environment makes when checking for PEFT adapter configs on every
    # fresh SentenceTransformer(...) construction (see experiments/
    # advenc_gpu50k_diagnostics.py, which hit this twice before this fix).
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"

    # ---- vanilla condition: vanilla queries vs vanilla-encoded corpus ----
    print("\n=== vanilla: encoding queries ===", flush=True)
    vanilla_model = load_encoder(None, device)
    vanilla_query_emb = encode(vanilla_model, query_texts).cpu()
    del vanilla_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vanilla_metrics = recall_and_ndcg(vanilla_query_emb, vanilla_corpus_emb, relevant_indices)
    print(f"vanilla: {vanilla_metrics}")

    # ---- defended condition: defended queries vs defended-encoded corpus ----
    print("\n=== defended: encoding corpus + queries through gpu50k defended encoder ===", flush=True)
    defended_model = load_defended_encoder(CHECKPOINT_PATH, device)
    defended_align_emb = encode_texts(defended_model, align_texts)
    defended_test_emb = encode_texts(defended_model, test_texts)
    defended_corpus_emb = torch.cat([defended_align_emb, defended_test_emb], dim=0)
    defended_query_emb = encode_texts(defended_model, query_texts)
    del defended_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    defended_metrics = recall_and_ndcg(defended_query_emb, defended_corpus_emb, relevant_indices)
    print(f"defended: {defended_metrics}")

    print("\n" + "=" * 60)
    print(f"{'metric':<12} {'vanilla':>12} {'gpu50k-defended':>18}")
    for key in ["recall@5", "recall@10", "ndcg@10"]:
        print(f"{key:<12} {vanilla_metrics[key]:>12.4f} {defended_metrics[key]:>18.4f}")

    results = {
        "n_queries": N_QUERIES,
        "n_corpus": len(corpus_texts),
        "vanilla": vanilla_metrics,
        "gpu50k_defended": defended_metrics,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
