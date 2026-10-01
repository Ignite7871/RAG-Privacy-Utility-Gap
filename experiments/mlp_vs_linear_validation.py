"""One-off validation checkpoint: reproduce the paper's headline MLP-vs-LinearProbe
comparison (Section VI-C) on the full 40k/10k MS MARCO split for MiniLM and BGE-large,
to sanity-check MLPAttacker/LinearProbeAttacker against the original crossover pattern
(MLP wins at low dimension, LinearProbe wins at high dimension) before trusting them for
the full sweep. Not part of the regular experiment pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.mlp_attacker import MLPAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_PATH = REPO_ROOT / "results" / "mlp_vs_linear_validation.json"

ENCODERS = {"minilm": 384, "bge": 1024}
ATTACKERS = {"MLP": MLPAttacker, "LinearProbe": LinearProbeAttacker}


def load_split(dataset: str, encoder: str, split: str) -> tuple[torch.Tensor, list[str]]:
    data = torch.load(CACHE_DIR / f"{dataset}_{encoder}_{split}.pt", weights_only=False)
    return data["embeddings"], data["text"]


def main() -> None:
    results = {}
    for encoder, dim in ENCODERS.items():
        align_embeddings, align_texts = load_split("msmarco", encoder, "align")
        test_embeddings, test_texts = load_split("msmarco", encoder, "test")

        for attacker_name, attacker_cls in ATTACKERS.items():
            key = f"{attacker_name}_{encoder}"
            print(f"=== {key} (align={len(align_texts)}, test={len(test_texts)}) ===", flush=True)
            t0 = time.perf_counter()

            attacker = attacker_cls(embedding_dim=dim)
            attacker.fit(align_embeddings, align_texts)
            predictions = attacker.attack(test_embeddings)

            elapsed = time.perf_counter() - t0
            rouge = rouge_l_corpus(predictions, test_texts)

            results[key] = {
                "attacker": attacker_name,
                "encoder": encoder,
                "dim": dim,
                "rougeL_precision": rouge["precision"],
                "rougeL_recall": rouge["recall"],
                "rougeL_f1": rouge["f1"],
                "wall_time_seconds": elapsed,
                "n_align": len(align_texts),
                "n_test": len(test_texts),
            }
            print(
                f"{key}: rougeL_precision={rouge['precision']:.4f} "
                f"rougeL_f1={rouge['f1']:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(f"{'attacker':<12} {'encoder':<8} {'rougeL_P':>10} {'wall_time_s':>12}")
    for row in results.values():
        print(
            f"{row['attacker']:<12} {row['encoder']:<8} "
            f"{row['rougeL_precision']:>10.4f} {row['wall_time_seconds']:>12.1f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
