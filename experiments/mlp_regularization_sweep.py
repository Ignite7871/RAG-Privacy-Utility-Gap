"""Regularization sweep for the MLP-vs-LinearProbe overfitting finding (see
results/mlp_loss_curve_minilm.png: MLPAttacker's training BCE loss on MS MARCO/MiniLM
collapses to ~0 well before epoch 50, i.e. it memorizes the alignment set rather than
being undertrained). This script gets direct confirmation via held-out loss and checks
whether weight decay or higher dropout prevents the divergence.

Carves the last 2,000 of MS MARCO/MiniLM's 40,000 alignment passages off as a held-out
split (never seen during training) and trains three MLPAttacker configs for 50 epochs
each, logging train-loss and held-out-loss every epoch:
  1. baseline       -- dropout=0.1, weight_decay=0    (paper's stated config)
  2. weight_decay    -- dropout=0.1, weight_decay=1e-2
  3. higher_dropout  -- dropout=0.3, weight_decay=0

No attack()/beam search/reconstruction here -- purely training dynamics. Whichever
config ends with the lowest held-out loss (not lowest train loss) is the one worth
testing against the real MiniLM test set next.
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
OUT_PATH = REPO_ROOT / "results" / "mlp_regularization_sweep_minilm.png"

EMBEDDING_DIM = 384
EPOCHS = 50
N_HELD_OUT = 2000

CONFIGS = [
    ("baseline", {"dropout": 0.1, "weight_decay": 0.0}),
    ("weight_decay", {"dropout": 0.1, "weight_decay": 1e-2}),
    ("higher_dropout", {"dropout": 0.3, "weight_decay": 0.0}),
]


def main() -> None:
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    all_embeddings, all_texts = align_data["embeddings"], align_data["text"]

    train_embeddings = all_embeddings[:-N_HELD_OUT]
    train_texts = all_texts[:-N_HELD_OUT]
    held_out_embeddings = all_embeddings[-N_HELD_OUT:]
    held_out_texts = all_texts[-N_HELD_OUT:]
    print(
        f"train={len(train_texts)}, held_out={len(held_out_texts)} "
        f"(msmarco/minilm, last {N_HELD_OUT} carved off)"
    )

    histories = {}
    for label, kwargs in CONFIGS:
        print(f"\n=== training MLPAttacker: {label} ({kwargs}) ===", flush=True)
        torch.manual_seed(0)
        attacker = MLPAttacker(embedding_dim=EMBEDDING_DIM, epochs=EPOCHS, lr=1e-3, **kwargs)
        history = attacker.fit(
            train_embeddings,
            train_texts,
            return_history=True,
            held_out=(held_out_embeddings, held_out_texts),
        )
        histories[label] = history
        print(
            f"{label}: final train_loss={history['train_loss'][-1]:.4f}  "
            f"final held_out_loss={history['held_out_loss'][-1]:.4f}"
        )

    print("\n" + "=" * 60)
    print(f"{'config':<16} {'final train_loss':>18} {'final held_out_loss':>22}")
    best_label, best_held_out_loss = None, float("inf")
    for label, _ in CONFIGS:
        h = histories[label]
        final_held_out = h["held_out_loss"][-1]
        print(f"{label:<16} {h['train_loss'][-1]:>18.4f} {final_held_out:>22.4f}")
        if final_held_out < best_held_out_loss:
            best_label, best_held_out_loss = label, final_held_out
    print(f"\nlowest final held_out_loss: {best_label} ({best_held_out_loss:.4f})")

    plt.figure(figsize=(9, 6))
    colors = {"baseline": "tab:blue", "weight_decay": "tab:orange", "higher_dropout": "tab:green"}
    for label, _ in CONFIGS:
        h = histories[label]
        epochs_axis = range(1, EPOCHS + 1)
        plt.plot(epochs_axis, h["train_loss"], color=colors[label], linestyle="-", label=f"{label} (train)")
        plt.plot(
            epochs_axis, h["held_out_loss"], color=colors[label], linestyle="--", label=f"{label} (held-out)"
        )
    plt.xlabel("epoch")
    plt.ylabel("mean BCE loss")
    plt.title("MLPAttacker regularization sweep -- MS MARCO / MiniLM (train vs. held-out loss)")
    plt.legend()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
