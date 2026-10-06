"""Gaussian DP defense sweep: retrieval utility and non-adaptive inversion over epsilon.

Method: Gaussian mechanism (defenses/gaussian_dp.py), sigma = sqrt(2*ln(1.25/delta))/eps,
delta=1e-5, applied as i.i.d. per-coordinate noise then L2-renormalised.

Per epsilon in {1, 2, 5, 10, 20, 50, 100, 200, 500, inf}, plus 25, 30 and 40 (appended at
the end of EPSILONS so the seeds of the earlier points do not change):
  1. Non-adaptive inversion: LinearProbeAttacker fit ONCE on VANILLA alignment embeddings
     (the eavesdropper does not know about the defense), attacked against DP-noised
     test-split embeddings. Plain and content-word ROUGE-L precision are reported over the
     full n_test=10,000 split (attack() is a cheap forward pass with no training).
  2. Retrieval: reuses experiments/advenc_retrieval_eval.py's Recall@5/10/NDCG@10 harness
     (sample_queries_with_ground_truth, recall_and_ndcg), but with CLEAN (unnoised) queries
     against a DP-NOISED corpus index, because the noise is added to STORED embeddings:
     queries are embedded fresh at query time and are not part of what is persisted, so
     they are not DP-noised. This deliberately differs from advenc_retrieval_eval.py's
     symmetric protocol (defended queries + defended corpus), because AdvEnc replaces the
     encoder end to end while DP only perturbs what is written to disk.

epsilon=inf (sigma=0) is the vanilla baseline row. The attacker is fit on the full
n_align=40,000 alignment set. The fit happens once, since it depends only on vanilla
embeddings, and takes about 316s on the RTX 4060 Laptop GPU.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on
# Windows). Must stay the first import in this file (see advenc_retrieval_eval.py).
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from defenses.gaussian_dp import GaussianDPDefense  # noqa: E402
from experiments.advenc_retrieval_eval import (  # noqa: E402
    N_QUERIES,
    recall_and_ndcg,
    sample_queries_with_ground_truth,
)
from experiments.check_fm1_collapse import encode, load_encoder  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_CSV = REPO_ROOT / "results" / "dp_defense" / "dp_sweep_full.csv"

EMBEDDING_DIM = 384
# New points are appended (not inserted) so the noise seed, which depends on list position, is unchanged for existing points.
EPSILONS: list[float] = [1, 2, 5, 10, 20, 50, 100, 200, 500, float("inf"), 25, 30, 40]
DELTA = 1e-5
SEED = 42


def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)

    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    vanilla_align_emb, align_texts = align_data["embeddings"], align_data["text"]
    vanilla_test_emb, test_texts = test_data["embeddings"], test_data["text"]

    corpus_texts = align_texts + test_texts
    vanilla_corpus_emb = torch.cat([vanilla_align_emb, vanilla_test_emb], dim=0)
    corpus_index = {text: i for i, text in enumerate(corpus_texts)}
    assert len(corpus_index) == len(corpus_texts), "duplicate passages in the 50k corpus pool"

    print(f"sampling {N_QUERIES} queries with ground truth in the corpus...", flush=True)
    query_pairs = sample_queries_with_ground_truth(corpus_index, N_QUERIES)
    query_texts = [q for q, _ in query_pairs]
    relevant_indices = [idx for _, idx in query_pairs]

    # Everything from here only needs cached embeddings -- no more Hub streaming, so
    # force offline mode (avoids a recurring flaky HEAD request noted in
    # advenc_gpu50k_diagnostics.py).
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"

    # Queries are embedded fresh at query time and never DP-noised (see module
    # docstring) -- encode them once with the vanilla encoder, reused for every epsilon.
    print("encoding 200 queries with the vanilla encoder (clean, not DP-noised)...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vanilla_model = load_encoder(None, device)
    vanilla_query_emb = encode(vanilla_model, query_texts).cpu()
    del vanilla_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Full n_align=40,000 alignment set. Fit ONCE on vanilla embeddings -- the
    # non-adaptive eavesdropper never sees the defended distribution.
    fit_texts = align_texts
    fit_embeddings = vanilla_align_emb
    n_align_used = len(fit_texts)

    print(f"fitting non-adaptive LinearProbeAttacker on n_align={n_align_used} vanilla pairs...", flush=True)
    t_fit = time.time()
    attacker = LinearProbeAttacker(embedding_dim=EMBEDDING_DIM)
    attacker.fit(fit_embeddings, fit_texts)
    print(f"  fit done in {time.time() - t_fit:.1f}s", flush=True)

    defense = GaussianDPDefense(delta=DELTA, seed=SEED)
    rows: list[dict] = []

    for i, epsilon in enumerate(EPSILONS):
        eps_label = "inf" if epsilon == float("inf") else str(epsilon)
        print(f"\n=== epsilon={eps_label} ===", flush=True)
        t_eps = time.time()

        # One generator per epsilon, advanced sequentially across the two noise draws
        # below (test-split, then corpus) so the draws are independent, not identical.
        gen = torch.Generator().manual_seed(SEED * 1_000_003 + i)

        # ---- 1. non-adaptive inversion precision ----
        noised_test_emb = defense.apply(vanilla_test_emb, epsilon, generator=gen)
        predictions = attacker.attack(noised_test_emb)
        scores = full_scores(predictions, test_texts)
        rouge = {"precision": scores["rouge_l_precision"]}

        # ---- 2. retrieval: clean queries vs. DP-noised corpus index ----
        noised_corpus_emb = defense.apply(vanilla_corpus_emb, epsilon, generator=gen)
        metrics = recall_and_ndcg(vanilla_query_emb, noised_corpus_emb, relevant_indices)

        sigma = GaussianDPDefense.sigma_from_epsilon(epsilon, DELTA)
        rows.append(
            {
                "epsilon": eps_label,
                "sigma": sigma,
                "rouge_l_precision": rouge["precision"],
                "content_precision": scores["content_precision"],
                "recall_at_5": metrics["recall@5"],
                "recall_at_10": metrics["recall@10"],
                "ndcg_at_10": metrics["ndcg@10"],
            }
        )
        print(
            f"  sigma={sigma:.4f} rouge_l_precision={rouge['precision']:.4f} "
            f"recall@5={metrics['recall@5']:.4f} recall@10={metrics['recall@10']:.4f} "
            f"ndcg@10={metrics['ndcg@10']:.4f} ({time.time() - t_eps:.1f}s)",
            flush=True,
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epsilon", "sigma", "rouge_l_precision", "content_precision", "recall_at_5", "recall_at_10", "ndcg_at_10"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")

    total_time = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"{'epsilon':>8} {'sigma':>10} {'rouge_l_prec':>13} {'recall@5':>10} {'recall@10':>10} {'ndcg@10':>10}")
    for row in rows:
        print(
            f"{row['epsilon']:>8} {row['sigma']:>10.4f} {row['rouge_l_precision']:>13.4f} "
            f"{row['recall_at_5']:>10.4f} {row['recall_at_10']:>10.4f} {row['ndcg_at_10']:>10.4f}"
        )
    print(f"\ntotal wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
