"""Run TransferAttacker (Huang et al. query-free/surrogate attack) on cached
embeddings. Main-venv (.venv312) entry point.

Uses cached gtr-t5-base embeddings as the "surrogate" encoder output (a
public, generic model the attacker can freely run) and cached
{encoder}-embeddings as the "leaked" true victim embeddings for the same
alignment texts. Writes results to
results/transfer_attack/{dataset}_{victim_encoder}_n{n_align}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
from attackers.transfer_attack import TransferAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus, token_f1_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
RESULTS_DIR = REPO_ROOT / "results" / "transfer_attack"

SURROGATE_ENCODER = "gtr"


def run(
    dataset: str,
    victim_encoder: str,
    n_align: int,
    n_test: int,
    epochs: int,
    batch_size: int,
    adapter_epochs: int,
) -> dict:
    if victim_encoder == SURROGATE_ENCODER:
        raise ValueError(
            f"victim_encoder cannot equal the surrogate encoder ({SURROGATE_ENCODER}) -- "
            "the attack's premise is that the attacker does NOT have query access to the "
            "victim, only to a different, public surrogate model."
        )

    victim_align_path = CACHE_DIR / f"{dataset}_{victim_encoder}_align.pt"
    victim_test_path = CACHE_DIR / f"{dataset}_{victim_encoder}_test.pt"
    surrogate_align_path = CACHE_DIR / f"{dataset}_{SURROGATE_ENCODER}_align.pt"

    victim_align = torch.load(victim_align_path, weights_only=False)
    victim_test = torch.load(victim_test_path, weights_only=False)
    surrogate_align = torch.load(surrogate_align_path, weights_only=False)

    # Sanity check: same texts in the same order across encoder caches (see
    # main pipeline's data/encode.py -- all encoders are run over the same
    # passage list), so row i is the same passage in both files.
    assert victim_align["text"][:5] == surrogate_align["text"][:5], (
        "victim/surrogate alignment caches are not row-aligned by passage -- "
        "cannot pair them 1:1"
    )

    victim_embeddings = victim_align["embeddings"][:n_align]
    surrogate_embeddings = surrogate_align["embeddings"][:n_align]
    align_texts = victim_align["text"][:n_align]
    test_embeddings = victim_test["embeddings"][:n_test]
    test_texts = victim_test["text"][:n_test]

    victim_dim = victim_embeddings.shape[1]
    surrogate_dim = surrogate_embeddings.shape[1]

    t0 = time.perf_counter()
    attacker = TransferAttacker(
        surrogate_dim=surrogate_dim,
        victim_dim=victim_dim,
        decoder_epochs=epochs,
        decoder_batch_size=batch_size,
        adapter_epochs=adapter_epochs,
    )
    init_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    attacker.fit(victim_embeddings, align_texts, surrogate_embeddings)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstructions = attacker.attack(test_embeddings)
    attack_time = time.perf_counter() - t0

    rouge = rouge_l_corpus(reconstructions, test_texts)
    f1 = token_f1_corpus(reconstructions, test_texts)

    return {
        "attacker": "transfer_attack",
        "dataset": dataset,
        "victim_encoder": victim_encoder,
        "surrogate_encoder": SURROGATE_ENCODER,
        "n_align": n_align,
        "n_test": n_test,
        "epochs": epochs,
        "batch_size": batch_size,
        "adapter_epochs": adapter_epochs,
        "timing": {"init": init_time, "fit": fit_time, "attack": attack_time},
        "rouge_l": rouge,
        "token_f1": f1,
        "reconstructions": [
            {"source": src, "reconstruction": recon}
            for src, recon in zip(test_texts, reconstructions)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TransferAttacker on cached embeddings.")
    parser.add_argument("--dataset", required=True, help='e.g. "msmarco" or "nq"')
    parser.add_argument("--victim_encoder", required=True, help='e.g. "minilm", "mpnet", "bge" (not "gtr" -- that is the surrogate)')
    parser.add_argument("--n_align", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--adapter_epochs", type=int, default=300)
    args = parser.parse_args()

    result = run(
        dataset=args.dataset,
        victim_encoder=args.victim_encoder,
        n_align=args.n_align,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        adapter_epochs=args.adapter_epochs,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.dataset}_{args.victim_encoder}_n{args.n_align}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"ROUGE-L: {result['rouge_l']}")
    print(f"Token F1: {result['token_f1']}")
    print(f"wrote {out_path}")
