"""Adaptive attackers vs. the adversarially trained encoders, across alignment budgets.

For each defended MiniLM checkpoint (AdvEnc-v1 CPU, AdvEnc-v2 CPU, AdvEnc-v2 GPU/50k) and
for the vanilla encoder, fit LinearProbe (3 seeds) and the MLP attacker
(1 seed) on n in {200, 1000, 5000, 10000, 40000} alignment pairs encoded by THAT encoder, and score
on the same 2,000-passage test subset. Defended embeddings are encoded once and cached in
data/cache/advenc_*.pt (same format as data/encode.py).

Results -> results/advenc/adaptive_budget_sweep.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# `datasets` must be imported first, process-wide (see experiments/advenc_cpu_scale_check.py).
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from attackers.mlp_attacker import MLPAttacker  # noqa: E402
from attackers.vocab_reconstruction import build_presence_targets, texts_to_token_ids  # noqa: E402
from experiments.advenc_cpu_scale_check import encode_texts, load_defended_encoder  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
CKPT = REPO_ROOT / "checkpoints"
OUT = REPO_ROOT / "results" / "advenc" / "adaptive_budget_sweep.csv"
ENCODERS = {
    "vanilla": None,
    "advenc_v1_cpu": "advenc_v1_cpu_minilm.pt",
    "advenc_v2_cpu": "advenc_v2_cpu_minilm.pt",
    "advenc_v2_gpu50k": "advenc_v2_gpu50k_minilm.pt",
    "advenc_v1_gpu50k": "advenc_v1_gpu50k_minilm.pt",
}
BUDGETS = [200, 1000, 5000, 10000, 40000]
LP_SEEDS = [0, 1, 2]
MLP_SEED = 0
N_TEST = 2000
FIELDS = ["encoder", "attacker", "n_align", "seed", "n_test", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1", "content_precision", "content_recall", "content_f1", "heldout_bce", "fit_seconds"]


def embeddings_for(name: str) -> tuple[dict, dict]:
    align = torch.load(CACHE / "msmarco_minilm_align.pt", weights_only=False)
    test = torch.load(CACHE / "msmarco_minilm_test.pt", weights_only=False)
    if name == "vanilla":
        return align, test
    paths = {s: CACHE / f"{name}_minilm_{s}.pt" for s in ("align", "test")}
    if not all(p.exists() for p in paths.values()):
        os.environ["HF_HUB_OFFLINE"] = "1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = load_defended_encoder(CKPT / ENCODERS[name], device)
        for split, src in (("align", align), ("test", test)):
            torch.save({"text": src["text"], "embeddings": encode_texts(enc, src["text"])}, paths[split])
        del enc
        torch.cuda.empty_cache()
    return torch.load(paths["align"], weights_only=False), torch.load(paths["test"], weights_only=False)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if OUT.exists():
        with OUT.open(newline="") as f:
            done = {(r["encoder"], r["attacker"], int(r["n_align"]), int(r["seed"])) for r in csv.DictReader(f)}
    new_file = not OUT.exists()
    test_idx = torch.randperm(10_000, generator=torch.Generator().manual_seed(123))[:N_TEST]
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for name in ENCODERS:
            align, test = embeddings_for(name)
            test_emb = test["embeddings"][test_idx]
            test_txt = [test["text"][i] for i in test_idx.tolist()]
            for n in BUDGETS:
                perm_all = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(0))
                jobs = [("linearprobe", s) for s in LP_SEEDS if n < 40000 or s == 0] + [("mlp", MLP_SEED)]
                for attacker, seed in jobs:
                    if (name, attacker, n, seed) in done:
                        continue
                    torch.manual_seed(seed)
                    perm = perm_all if n == 40000 else torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(seed))
                    idx = perm[:n]
                    emb, txt = align["embeddings"][idx], [align["text"][i] for i in idx.tolist()]
                    cls = LinearProbeAttacker if attacker == "linearprobe" else MLPAttacker
                    t0 = time.time()
                    atk = cls(embedding_dim=emb.shape[1])
                    atk.fit(emb, txt)
                    fit_s = time.time() - t0
                    r = full_scores(atk.attack(test_emb), test_txt)
                    bce = ""
                    if attacker == "linearprobe":
                        # Held-out multi-label BCE of the fitted linear head (is the defended
                        # embedding space easier for a linear head to fit than the vanilla one?).
                        with torch.no_grad():
                            tgt = build_presence_targets(texts_to_token_ids(atk.tokenizer, test_txt)).to(atk.device)
                            bce = float(torch.nn.functional.binary_cross_entropy_with_logits(atk.W(test_emb.to(atk.device)), tgt))
                    w.writerow({"encoder": name, "attacker": attacker, "n_align": n, "seed": seed, "n_test": N_TEST,
                                **r, "heldout_bce": bce,
                                "fit_seconds": round(fit_s, 1)})
                    f.flush()
                    print(f"{name} {attacker} n={n} seed={seed}: P={r['rouge_l_precision']:.4f} C={r['content_precision']:.4f} ({fit_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
