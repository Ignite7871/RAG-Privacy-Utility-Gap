"""Cross-domain alignment: does an attacker need alignment data from the victim's own domain?

For each encoder, a LinearProbeAttacker is fit on n alignment pairs from a SOURCE corpus
(MS MARCO or NQ) and scored on test embeddings from a TARGET corpus. The four (source,
target) cells separate the in-domain case (attacker has same-domain leaked data) from the
cross-domain case (attacker only has a public corpus of a different style).

Alignment subsets are the first n items of a seeded permutation of the 40k align pool; test
is a fixed 2,000-passage subset of each corpus's 10k test split, identical across seeds.
Results -> results/cross_domain/cross_domain.csv (one row per encoder/source/target/n/seed).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see CLAUDE.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
OUT = REPO_ROOT / "results" / "cross_domain" / "cross_domain.csv"
ENCODERS = ["minilm", "mpnet", "gtr", "bge"]
CORPORA = ["msmarco", "nq"]
N_ALIGN = [200, 1000, 5000]
SEEDS = [0, 1, 2]
N_TEST = 2000
FIELDS = ["encoder", "source", "target", "n_align", "seed", "n_test", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1", "content_precision", "content_recall", "content_f1", "fit_seconds"]


def load(corpus: str, enc: str, split: str) -> dict:
    return torch.load(CACHE / f"{corpus}_{enc}_{split}.pt", weights_only=False)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if OUT.exists():
        with OUT.open(newline="") as f:
            done = {(r["encoder"], r["source"], r["target"], int(r["n_align"]), int(r["seed"])) for r in csv.DictReader(f)}
    new_file = not OUT.exists()
    with OUT.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for enc in ENCODERS:
            data = {c: {"align": load(c, enc, "align"), "test": load(c, enc, "test")} for c in CORPORA}
            dim = data["msmarco"]["align"]["embeddings"].shape[1]
            test_idx = torch.randperm(len(data["msmarco"]["test"]["text"]), generator=torch.Generator().manual_seed(123))[:N_TEST]
            for source in CORPORA:
                for target in CORPORA:
                    tgt = data[target]["test"]
                    test_emb = tgt["embeddings"][test_idx]
                    test_txt = [tgt["text"][i] for i in test_idx.tolist()]
                    for n in N_ALIGN:
                        for seed in SEEDS:
                            key = (enc, source, target, n, seed)
                            if key in done:
                                continue
                            torch.manual_seed(seed)
                            src = data[source]["align"]
                            perm = torch.randperm(len(src["text"]), generator=torch.Generator().manual_seed(seed))[:n]
                            t0 = time.time()
                            atk = LinearProbeAttacker(embedding_dim=dim)
                            atk.fit(src["embeddings"][perm], [src["text"][i] for i in perm.tolist()])
                            fit_s = time.time() - t0
                            r = full_scores(atk.attack(test_emb), test_txt)
                            writer.writerow({"encoder": enc, "source": source, "target": target, "n_align": n, "seed": seed,
                                             "n_test": N_TEST, **r, "fit_seconds": round(fit_s, 1)})
                            f.flush()
                            print(f"{enc} {source}->{target} n={n} seed={seed}: P={r['rouge_l_precision']:.4f} C={r['content_precision']:.4f} ({fit_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
