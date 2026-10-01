"""Single-combination confirmation check: does the regularization-sweep winner
(dropout=0.3, weight_decay=0, see results/mlp_regularization_sweep_minilm.png) actually
improve real attack precision on MS MARCO/MiniLM, before rerunning the full
4-combination validation from results/mlp_vs_linear_validation.json? Baseline MLP on
MiniLM there was rougeL_precision=0.3761 (paper: 0.536).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.mlp_attacker import MLPAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "mlp_higher_dropout_check_minilm.json"

EMBEDDING_DIM = 384


def main() -> None:
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    align_embeddings, align_texts = align_data["embeddings"], align_data["text"]
    test_embeddings, test_texts = test_data["embeddings"], test_data["text"]
    print(f"align={len(align_texts)}, test={len(test_texts)} (msmarco/minilm)")

    t0 = time.perf_counter()
    attacker = MLPAttacker(embedding_dim=EMBEDDING_DIM, dropout=0.3, weight_decay=0.0)
    attacker.fit(align_embeddings, align_texts)
    predictions = attacker.attack(test_embeddings)
    elapsed = time.perf_counter() - t0

    rouge = rouge_l_corpus(predictions, test_texts)
    result = {
        "attacker": "MLP",
        "encoder": "minilm",
        "dim": EMBEDDING_DIM,
        "dropout": 0.3,
        "weight_decay": 0.0,
        "rougeL_precision": rouge["precision"],
        "rougeL_recall": rouge["recall"],
        "rougeL_f1": rouge["f1"],
        "wall_time_seconds": elapsed,
        "n_align": len(align_texts),
        "n_test": len(test_texts),
    }
    print(f"\nMLP_minilm (dropout=0.3): rougeL_precision={rouge['precision']:.4f} ({elapsed:.1f}s)")
    print("baseline (dropout=0.1) was: rougeL_precision=0.3761")
    print("paper's reported value:    rougeL_precision=0.536")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
