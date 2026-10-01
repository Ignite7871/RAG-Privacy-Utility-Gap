"""Isolates 'amount of training' from 'scale of data' for the CPU-scale AdvEnc-v2
decoder-vs-trivial-baseline finding (see cpu_scale_downstream_check.json's Part 1):
was decoder_loss=0.0067 vs trivial=0.0072 (barely beating the trivial baseline) an
artifact of only training for 5 epochs, or does the internal decoder stay near the
trivial floor regardless of how long it trains?

Single continuous 100-epoch training run on the SAME 1600-pair self-pair pool as the
original CPU-scale v2 run (build_self_pairs(1600) is deterministic streaming, so this
reproduces the identical pool), reading decoder_loss off the per-epoch trainlog at
epochs [5, 20, 50, 100] rather than restarting training four times from scratch --
the same efficient checkpoint-extraction pattern used in
experiments/mlp_loss_diagnostics.py (cheaper, and avoids reintroducing seed-related
variance across independent restarts).

Trivial marginal-frequency baseline for this exact 1600-pair pool was already computed
in experiments/advenc_cpu_scale_check.py: 0.007221239618957043. Reused verbatim per
the brief -- it depends only on the pair pool's token distribution, not epoch count.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from defenses.adv_encoder import AdvEncTrainer, build_self_pairs  # noqa: E402

import torch  # noqa: E402

OUT_PATH = REPO_ROOT / "results" / "advenc" / "v2_cpu_epoch_sweep.json"

N_PAIRS = 1600
MAX_EPOCHS = 100
CHECKPOINT_EPOCHS = [5, 20, 50, 100]
LAMBDA_PRIV = 2.0
SEED = 42

# Already computed for this exact 1600-pair pool (experiments/advenc_cpu_scale_check.py).
TRIVIAL_BASELINE = 0.007221239618957043


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"building {N_PAIRS} self-pairs (same pool as the original CPU-scale v2 run)...", flush=True)
    pairs = build_self_pairs(N_PAIRS)

    trainer = AdvEncTrainer(encoder_model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=16)
    print(f"training for {MAX_EPOCHS} epochs, reading off decoder_loss at {CHECKPOINT_EPOCHS}...", flush=True)
    log = trainer.train(pairs, epochs=MAX_EPOCHS, lambda_priv=LAMBDA_PRIV, progress_every=50)

    rows = []
    for epochs in CHECKPOINT_EPOCHS:
        decoder_loss = log["decoder_loss"][epochs - 1]
        below_trivial = decoder_loss < TRIVIAL_BASELINE
        rows.append(
            {
                "epochs": epochs,
                "decoder_loss": decoder_loss,
                "trivial_baseline": TRIVIAL_BASELINE,
                "decoder_loss_below_trivial": below_trivial,
            }
        )

    print("\n" + "=" * 70)
    print(f"{'epochs':>8} {'decoder_loss':>14} {'trivial_baseline':>18} {'below_trivial':>15}")
    for row in rows:
        print(
            f"{row['epochs']:>8} {row['decoder_loss']:>14.4f} {row['trivial_baseline']:>18.4f} "
            f"{str(row['decoder_loss_below_trivial']):>15}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "full_decoder_loss_trajectory": log["decoder_loss"]}, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
