"""LinearProbe at full scale (n_align=40,000, n_test=10,000) on both corpora and all four encoders,
recording standard and content-word ROUGE-L, plus the context-free (embedding-free) floor on the
same test passages. Replaces the standard-metric-only table for the paper's main attack comparison.

Results -> results/linearprobe_full_scale_both_metrics.csv (resumable)
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
OUT = REPO_ROOT / "results" / "linearprobe_full_scale_both_metrics.csv"
FIELDS = ["corpus", "encoder", "n_align", "n_test", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
          "content_precision", "content_recall", "content_f1", "fit_seconds"]


def main() -> None:
    done = set()
    if OUT.exists():
        with OUT.open(newline="") as f:
            done = {(r["corpus"], r["encoder"]) for r in csv.DictReader(f)}
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for corpus in ("msmarco", "nq"):
            for enc in ("minilm", "mpnet", "gtr", "bge"):
                if (corpus, enc) in done:
                    continue
                a = torch.load(CACHE / f"{corpus}_{enc}_align.pt", weights_only=False)
                t = torch.load(CACHE / f"{corpus}_{enc}_test.pt", weights_only=False)
                torch.manual_seed(42)
                t0 = time.time()
                atk = LinearProbeAttacker(embedding_dim=a["embeddings"].shape[1])
                atk.fit(a["embeddings"], a["text"])
                fit_s = time.time() - t0
                r = full_scores(atk.attack(t["embeddings"]), t["text"])
                w.writerow({"corpus": corpus, "encoder": enc, "n_align": len(a["text"]), "n_test": len(t["text"]), **r,
                            "fit_seconds": round(fit_s, 1)})
                f.flush()
                print(corpus, enc, {k: round(v, 4) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
