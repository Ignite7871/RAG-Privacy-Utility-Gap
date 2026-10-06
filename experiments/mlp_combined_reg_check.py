"""Regularization variant for the MLP-vs-LinearProbe gap (see
results/mlp_regularization_sweep_minilm.png, results/mlp_higher_dropout_check_minilm.json):
combines the two partial improvements -- higher_dropout (0.3) and a much lighter
weight_decay (1e-3, ten times lighter than the 1e-2 that underfit badly) -- in one config,
on MS MARCO/MiniLM only. It checks whether combining the two regularization knobs beats
higher_dropout alone (0.4046) meaningfully.

Two phases:
  1. Same 38k/2k held-out split as the regularization sweep, to report train_loss and
     held_out_loss for direct comparison against the other three configs.
  2. Full 40k align set + real attack() on the 10k test set, for ROUGE-L precision.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
from attackers.mlp_attacker import MLPAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "mlp_combined_reg_check_minilm.json"

EMBEDDING_DIM = 384
N_HELD_OUT = 2000
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3


def main() -> None:
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    all_embeddings, all_texts = align_data["embeddings"], align_data["text"]
    test_embeddings, test_texts = test_data["embeddings"], test_data["text"]

    # Phase 1: held-out loss, same split as the regularization sweep.
    train_embeddings = all_embeddings[:-N_HELD_OUT]
    train_texts = all_texts[:-N_HELD_OUT]
    held_out_embeddings = all_embeddings[-N_HELD_OUT:]
    held_out_texts = all_texts[-N_HELD_OUT:]

    print(f"=== phase 1: held-out loss (dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}) ===", flush=True)
    torch.manual_seed(0)
    diag_attacker = MLPAttacker(embedding_dim=EMBEDDING_DIM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY)
    history = diag_attacker.fit(
        train_embeddings, train_texts, return_history=True, held_out=(held_out_embeddings, held_out_texts)
    )
    final_train_loss = history["train_loss"][-1]
    final_held_out_loss = history["held_out_loss"][-1]
    print(f"final train_loss={final_train_loss:.4f}  final held_out_loss={final_held_out_loss:.4f}")

    # Phase 2: full align set, real attack() on the real test set.
    print("\n=== phase 2: full align + attack() on real test set ===", flush=True)
    t0 = time.perf_counter()
    attacker = MLPAttacker(embedding_dim=EMBEDDING_DIM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY)
    attacker.fit(all_embeddings, all_texts)
    predictions = attacker.attack(test_embeddings)
    elapsed = time.perf_counter() - t0

    rouge = rouge_l_corpus(predictions, test_texts)

    print("\n" + "=" * 60)
    print(f"combined (dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}):")
    print(f"  final train_loss      = {final_train_loss:.4f}")
    print(f"  final held_out_loss   = {final_held_out_loss:.4f}")
    print(f"  rougeL_precision      = {rouge['precision']:.4f}  ({elapsed:.1f}s)")
    print("\nfor comparison:")
    print("  baseline       (dropout=0.1, wd=0)     : held_out_loss=0.0078, rougeL_precision=0.3761")
    print("  higher_dropout (dropout=0.3, wd=0)     : held_out_loss=0.0062, rougeL_precision=0.4046")
    print("  weight_decay   (dropout=0.1, wd=1e-2)  : held_out_loss=0.0800")
    print("  LinearProbe on minilm                  : rougeL_precision=0.4199")

    result = {
        "attacker": "MLP",
        "encoder": "minilm",
        "dim": EMBEDDING_DIM,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "final_train_loss": final_train_loss,
        "final_held_out_loss": final_held_out_loss,
        "rougeL_precision": rouge["precision"],
        "rougeL_recall": rouge["recall"],
        "rougeL_f1": rouge["f1"],
        "wall_time_seconds": elapsed,
        "n_align": len(all_texts),
        "n_test": len(test_texts),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
