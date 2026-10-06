"""Run GEIAAttacker on cached embeddings. Main-venv (.venv312) entry point --
GEIA only needs GPT-2 + current transformers, no legacy-venv subprocess
required (unlike run_algen.py).

Reads cached embeddings produced by data/encode.py, runs
GEIAAttacker.fit()+attack(), and writes results to
results/geia/{dataset}_{encoder}_n{n_align}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
from attackers.geia import GEIAAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus, token_f1_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results" / "geia"


def run(
    dataset: str,
    encoder: str,
    n_align: int,
    n_test: int,
    epochs: int,
    batch_size: int,
) -> dict:
    align_path = CACHE_DIR / f"{dataset}_{encoder}_align.pt"
    test_path = CACHE_DIR / f"{dataset}_{encoder}_test.pt"

    align_data = torch.load(align_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    align_embeddings = align_data["embeddings"][:n_align]
    align_texts = align_data["text"][:n_align]
    test_embeddings = test_data["embeddings"][:n_test]
    test_texts = test_data["text"][:n_test]

    embedding_dim = align_embeddings.shape[1]

    t0 = time.perf_counter()
    attacker = GEIAAttacker(embedding_dim=embedding_dim, epochs=epochs, batch_size=batch_size)
    init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    attacker.fit(align_embeddings, align_texts)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructions = attacker.attack(test_embeddings)
    attack_time = time.perf_counter() - t0

    rouge = rouge_l_corpus(reconstructions, test_texts)
    f1 = token_f1_corpus(reconstructions, test_texts)

    return {
        "attacker": "geia",
        "dataset": dataset,
        "encoder": encoder,
        "n_align": n_align,
        "n_test": n_test,
        "epochs": epochs,
        "batch_size": batch_size,
        "timing": {"init": init_time, "fit": fit_time, "attack": attack_time},
        "rouge_l": rouge,
        "token_f1": f1,
        "reconstructions": [
            {"source": src, "reconstruction": recon}
            for src, recon in zip(test_texts, reconstructions)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEIAAttacker on cached embeddings.")
    parser.add_argument("--dataset", required=True, help='e.g. "msmarco" or "nq"')
    parser.add_argument("--encoder", required=True, help='e.g. "minilm", "mpnet", "gtr", "bge"')
    parser.add_argument("--n_align", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    result = run(
        dataset=args.dataset,
        encoder=args.encoder,
        n_align=args.n_align,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}_{args.encoder}_n{args.n_align}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"ROUGE-L: {result['rouge_l']}")
    print(f"Token F1: {result['token_f1']}")
    print(f"wrote {out_path}")
