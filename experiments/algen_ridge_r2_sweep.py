"""Follow-up to algen_alignment_quality.py: tracks ridge_R2 across alignment
set sizes for msmarco/mpnet and msmarco/gtr specifically -- the two cases
whose full-attack ROUGE-L curves diverged (gtr plateaus 1000->2000, mpnet
keeps accelerating; see results/algen/algen_sweep_summary.csv). Checks
whether that divergence is already visible in the linear alignment fit
itself, before generation.

MUST be run under .venv-algen-legacy -- same reason as algen_alignment_quality.py
(see README.md, "Environments").
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
from attackers.algen import AlgenAttacker  # noqa: E402
from experiments.algen_alignment_quality import ridge_r2  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT = REPO_ROOT / "checkpoints" / "algen_generator" / "checkpoint_epoch_99.pt"
OUT_CSV = REPO_ROOT / "results" / "algen" / "ridge_r2_sweep.csv"

COMBOS = [("msmarco", "mpnet"), ("msmarco", "gtr")]
N_ALIGN_VALUES = [50, 200, 1000, 2000]


def main() -> None:
    attacker = AlgenAttacker(
        generator_checkpoint=str(CHECKPOINT), generator_model_name="google/flan-t5-small"
    )

    rows = []
    for dataset, encoder in COMBOS:
        align_data = torch.load(CACHE_DIR / f"{dataset}_{encoder}_align.pt", weights_only=False)

        for n_align in N_ALIGN_VALUES:
            t0 = time.perf_counter()
            X = align_data["embeddings"][:n_align].to(attacker.device)
            texts = align_data["text"][:n_align]

            attacker.fit(X, texts)
            Y = attacker._target_embeddings(texts)
            r2 = ridge_r2(X, Y, attacker.T)
            elapsed = time.perf_counter() - t0

            rows.append({"dataset": dataset, "encoder": encoder, "n_align": n_align, "ridge_R2": r2})
            print(f"{dataset}/{encoder} n_align={n_align}: R^2={r2:.4f} ({elapsed:.2f}s)")

    print("\n" + "=" * 45)
    print(f"{'dataset':<10} {'encoder':<8} {'n_align':>8} {'ridge_R2':>10}")
    for row in rows:
        print(f"{row['dataset']:<10} {row['encoder']:<8} {row['n_align']:>8} {row['ridge_R2']:>10.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "encoder", "n_align", "ridge_R2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
