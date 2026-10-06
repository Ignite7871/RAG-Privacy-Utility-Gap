"""Adaptive attacker against Gaussian-DP-noised embeddings.

experiments/dp_defense_sweep.py fits LinearProbe ONCE on vanilla embeddings and attacks the
noised ones (a non-adaptive attacker, A1). Here the attacker knows sigma and refits on
noised alignment embeddings (independent noise per embedding, then L2 renormalisation, exactly as
GaussianDPDefense.apply), then attacks noised test embeddings (A2). MiniLM-L6 / MS MARCO.

Results -> results/dp_defense/dp_adaptive.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from defenses.gaussian_dp import GaussianDPDefense  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
OUT = REPO_ROOT / "results" / "dp_defense" / "dp_adaptive.csv"
EPSILONS = [1, 2, 5, 10, 20, 25, 30, 40, 50, 100, 200]
N_ALIGN = [1000, 10000]
N_TEST = 2000
FIELDS = ["epsilon", "sigma", "n_align", "n_test", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1", "content_precision", "content_recall", "content_f1", "fit_seconds"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    align = torch.load(CACHE / "msmarco_minilm_align.pt", weights_only=False)
    test = torch.load(CACHE / "msmarco_minilm_test.pt", weights_only=False)
    t_idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(123))[:N_TEST]
    test_emb, test_txt = test["embeddings"][t_idx], [test["text"][i] for i in t_idx.tolist()]
    a_perm = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(0))
    dp = GaussianDPDefense()

    done: set[tuple] = set()
    if OUT.exists():
        with OUT.open(newline="") as f:
            done = {(float(r["epsilon"]), int(r["n_align"])) for r in csv.DictReader(f)}
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for eps in EPSILONS:
            for n in N_ALIGN:
                if (float(eps), n) in done:
                    continue
                gen = torch.Generator().manual_seed(1000 + eps)
                idx = a_perm[:n]
                a_emb = dp.apply(align["embeddings"][idx], eps, generator=gen)
                t_emb = dp.apply(test_emb, eps, generator=gen)
                torch.manual_seed(0)
                t0 = time.time()
                atk = LinearProbeAttacker(embedding_dim=a_emb.shape[1])
                atk.fit(a_emb, [align["text"][i] for i in idx.tolist()])
                fit_s = time.time() - t0
                r = full_scores(atk.attack(t_emb), test_txt)
                w.writerow({"epsilon": eps, "sigma": GaussianDPDefense.sigma_from_epsilon(eps), "n_align": n, "n_test": N_TEST,
                            **r, "fit_seconds": round(fit_s, 1)})
                f.flush()
                print(f"eps={eps} n={n}: P={r['rouge_l_precision']:.4f} C={r['content_precision']:.4f} ({fit_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
