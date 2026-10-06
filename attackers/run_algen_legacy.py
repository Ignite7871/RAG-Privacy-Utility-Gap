"""Standalone ALGEN attack runner, meant to be invoked under .venv-algen-legacy
(transformers==4.52.4) -- NOT under the main .venv312 (see README.md, "Environments"). Reads cached embeddings produced by the main venv's data/encode.py,
runs AlgenAttacker fit()+attack(), and writes results to
results/algen/{dataset}_{encoder}_n{n_align}.json.

The main experiment runner (experiments/run_algen.py, .venv312) invokes this
script via subprocess using .venv-algen-legacy's python executable by path --
it does not import attackers.algen directly, so nothing in the main pipeline
needs to handle the transformers version split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# This file lives inside the attackers/ package itself, so running it directly
# (`python attackers/run_algen_legacy.py`) needs the repo root on sys.path
# before `attackers.algen` is importable as a package.
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
from attackers.algen import AlgenAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus, token_f1_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results" / "algen"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "algen_generator" / "checkpoint_epoch_99.pt"


def run(
    dataset: str,
    encoder: str,
    n_align: int,
    n_test: int,
    checkpoint: Path,
    generator_model_name: str,
) -> dict:
    align_path = CACHE_DIR / f"{dataset}_{encoder}_align.pt"
    test_path = CACHE_DIR / f"{dataset}_{encoder}_test.pt"

    align_data = torch.load(align_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)

    align_embeddings = align_data["embeddings"][:n_align]
    align_texts = align_data["text"][:n_align]
    test_embeddings = test_data["embeddings"][:n_test]
    test_texts = test_data["text"][:n_test]

    t0 = time.perf_counter()
    attacker = AlgenAttacker(generator_checkpoint=str(checkpoint), generator_model_name=generator_model_name)
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
        "dataset": dataset,
        "encoder": encoder,
        "n_align": n_align,
        "n_test": n_test,
        "checkpoint": str(checkpoint),
        "generator_model_name": generator_model_name,
        "timing": {"init": init_time, "fit": fit_time, "attack": attack_time},
        "rouge_l": rouge,
        "token_f1": f1,
        "reconstructions": [
            {"source": src, "reconstruction": recon}
            for src, recon in zip(test_texts, reconstructions)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AlgenAttacker on cached embeddings (.venv-algen-legacy only)."
    )
    parser.add_argument("--dataset", required=True, help='e.g. "msmarco" or "nq"')
    parser.add_argument("--encoder", required=True, help='e.g. "minilm", "mpnet", "gtr", "bge"')
    parser.add_argument("--n_align", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--generator_model_name", default="google/flan-t5-small")
    parser.add_argument("--out_dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    result = run(
        dataset=args.dataset,
        encoder=args.encoder,
        n_align=args.n_align,
        n_test=args.n_test,
        checkpoint=args.checkpoint,
        generator_model_name=args.generator_model_name,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.dataset}_{args.encoder}_n{args.n_align}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"ROUGE-L: {result['rouge_l']}")
    print(f"Token F1: {result['token_f1']}")
    print(f"wrote {out_path}")
