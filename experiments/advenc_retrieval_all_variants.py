"""Retrieval utility of every AdvEnc variant (vanilla, v1 CPU, v2 CPU, v2 GPU/50k).

Extends experiments/advenc_retrieval_eval.py (200 queries, GPU-scale v2 only) to all defended
checkpoints and 1,000 queries, with percentile-bootstrap 95% CIs. This includes AdvEnc-v1
(trained on real query-passage pairs). Protocol is unchanged: queries and the 50k corpus are
both encoded by the SAME encoder.

Results -> results/advenc/retrieval_all_variants.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# `datasets` must be imported first, process-wide (see experiments/advenc_cpu_scale_check.py).
import datasets  # noqa: E402, F401

from experiments.advenc_cpu_scale_check import encode_texts, load_defended_encoder  # noqa: E402
from experiments.advenc_retrieval_eval import sample_queries_with_ground_truth  # noqa: E402
from experiments.check_fm1_collapse import encode, load_encoder  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
CKPT = REPO_ROOT / "checkpoints"
OUT = REPO_ROOT / "results" / "advenc" / "retrieval_all_variants.json"
N_QUERIES = 1000
N_BOOT = 2000
VARIANTS = {
    "vanilla": None,
    "advenc_v1_cpu": "advenc_v1_cpu_minilm.pt",
    "advenc_v2_cpu": "advenc_v2_cpu_minilm.pt",
    "advenc_v2_gpu50k": "advenc_v2_gpu50k_minilm.pt",
    "advenc_v1_gpu50k": "advenc_v1_gpu50k_minilm.pt",
}


def per_query_scores(q: torch.Tensor, corpus: torch.Tensor, rel: list[int], k: int = 10) -> dict[str, torch.Tensor]:
    top = torch.topk(q @ corpus.T, k=k, dim=1).indices
    rel_t = torch.tensor(rel).unsqueeze(1)
    hit = top == rel_t
    rank = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.zeros(len(rel), dtype=torch.long))
    ndcg = torch.where(rank > 0, 1.0 / torch.log2(rank.float() + 1), torch.zeros(len(rel)))
    return {"recall@5": ((rank > 0) & (rank <= 5)).float(), "recall@10": (rank > 0).float(), "ndcg@10": ndcg}


def summarise(scores: dict[str, torch.Tensor]) -> dict:
    g = torch.Generator().manual_seed(0)
    n = len(next(iter(scores.values())))
    idx = torch.randint(0, n, (N_BOOT, n), generator=g)
    out = {}
    for name, v in scores.items():
        boots = v[idx].mean(1)
        out[name] = {"mean": float(v.mean()), "ci95": [float(boots.quantile(0.025)), float(boots.quantile(0.975))]}
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align = torch.load(CACHE / "msmarco_minilm_align.pt", weights_only=False)
    test = torch.load(CACHE / "msmarco_minilm_test.pt", weights_only=False)
    corpus_texts = align["text"] + test["text"]
    index = {t: i for i, t in enumerate(corpus_texts)}
    pairs = sample_queries_with_ground_truth(index, N_QUERIES)
    q_texts, rel = [q for q, _ in pairs], [i for _, i in pairs]
    os.environ["HF_HUB_OFFLINE"] = "1"

    results = {"n_queries": N_QUERIES, "n_corpus": len(corpus_texts), "variants": {}}
    for name, ckpt in VARIANTS.items():
        if ckpt is None:
            corpus = torch.cat([align["embeddings"], test["embeddings"]])
            model = load_encoder(None, device)
            q = encode(model, q_texts).cpu()
        else:
            model = load_defended_encoder(CKPT / ckpt, device)
            corpus = torch.cat([encode_texts(model, align["text"]), encode_texts(model, test["text"])])
            q = encode_texts(model, q_texts)
        del model
        torch.cuda.empty_cache()
        results["variants"][name] = summarise(per_query_scores(q, corpus, rel))
        print(name, {k: round(v["mean"], 4) for k, v in results["variants"][name].items()}, flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
