"""Diagnostic checkpoint for the MLP-vs-LinearProbe validation gap (see
results/mlp_vs_linear_validation.json): trains MLPAttacker on MS MARCO/MiniLM only,
at the paper's stated 50 epochs and at an extended 150 epochs, and plots both loss
curves. No attack()/beam search/BGE-large/LinearProbe involved -- this isolates whether
the MLP head is underfitting (loss still declining at epoch 50) or plateaued
(architecture/lr mismatch), per the two candidate explanations for why MLP lost to
LinearProbe on MiniLM despite matching the paper's expected order on BGE-large.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.mlp_attacker import MLPAttacker  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "mlp_loss_curve_minilm.png"

EMBEDDING_DIM = 384
CONDITIONS = [("50 epochs (current config)", 50), ("150 epochs (extended)", 150)]


def main() -> None:
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    embeddings, texts = align_data["embeddings"], align_data["text"]
    print(f"loaded {len(texts)} alignment passages (msmarco/minilm)")

    histories = {}
    for label, epochs in CONDITIONS:
        print(f"\n=== training MLPAttacker: {label} ===", flush=True)
        torch.manual_seed(0)
        attacker = MLPAttacker(embedding_dim=EMBEDDING_DIM, epochs=epochs, lr=1e-3)
        history = attacker.fit(embeddings, texts, return_history=True)["train_loss"]
        histories[label] = history
        print(f"{label}: final-epoch loss = {history[-1]:.4f}")

    fifty_epoch_loss = histories["50 epochs (current config)"][-1]
    extended_history = histories["150 epochs (extended)"]
    loss_at_epoch_50_in_extended_run = extended_history[49]
    final_extended_loss = extended_history[-1]

    print("\n" + "=" * 60)
    print(f"50-epoch run,  final (epoch 50)  loss: {fifty_epoch_loss:.4f}")
    print(f"150-epoch run, loss AT epoch 50        : {loss_at_epoch_50_in_extended_run:.4f}")
    print(f"150-epoch run, final (epoch 150) loss: {final_extended_loss:.4f}")

    plt.figure(figsize=(8, 5))
    for label, epochs in CONDITIONS:
        history = histories[label]
        plt.plot(range(1, epochs + 1), history, label=label)
    plt.axvline(x=50, color="gray", linestyle="--", linewidth=1, label="epoch 50")
    plt.xlabel("epoch")
    plt.ylabel("mean BCE loss")
    plt.title("MLPAttacker training loss -- MS MARCO / MiniLM")
    plt.legend()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
