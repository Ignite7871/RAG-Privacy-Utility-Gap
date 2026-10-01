"""LinearProbe and embedding-free floors in ALGEN's native 32-token setting (the msmarco32/nq32 caches),
same 500 test prefixes and same alignment prefixes as attackers/run_algen_legacy.py uses ([:n_align]).
-> results/algen_native/baselines.csv"""
from __future__ import annotations
import csv, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: F401
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from attackers.linear_probe import LinearProbeAttacker
from attackers.metrics import full_scores
from attackers.vocab_reconstruction import TOKENIZER_NAME, VOCAB_SIZE, mean_token_position, reconstruct_from_logits, texts_to_token_ids
import torch

C = REPO_ROOT / "data" / "cache"
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
stop = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
for w in ENGLISH_STOP_WORDS:
    for i in tok(w, add_special_tokens=False)["input_ids"]: stop[i] = True
rows = []
for corpus in ("msmarco32", "nq32"):
    for enc in ("minilm", "mpnet", "gtr", "bge"):
        a = torch.load(C / f"{corpus}_{enc}_align.pt", weights_only=False)
        t = torch.load(C / f"{corpus}_{enc}_test.pt", weights_only=False)
        for n in (50, 200, 1000, 2000):
            torch.manual_seed(0)
            lp = LinearProbeAttacker(embedding_dim=a["embeddings"].shape[1])
            lp.fit(a["embeddings"][:n], a["text"][:n])
            rows.append({"corpus": corpus, "encoder": enc, "n_align": n, "attacker": "linearprobe", **full_scores(lp.attack(t["embeddings"]), t["text"])})
        ids = texts_to_token_ids(tok, a["text"])
        df = torch.zeros(VOCAB_SIZE)
        for l in ids: df[list(set(l))] += 1
        order = mean_token_position(ids)
        for name, mask in (("floor_frequent_tokens", None), ("floor_frequent_content_words", stop)):
            d = df.clone()
            if mask is not None: d[mask] = 0; d[:1000] = 0
            rows.append({"corpus": corpus, "encoder": enc, "n_align": len(a["text"]), "attacker": name,
                         **full_scores(reconstruct_from_logits(d.log1p().unsqueeze(0).repeat(len(t["text"]), 1), tok, order, 16), t["text"])})
        print(corpus, enc, flush=True)
(REPO_ROOT / "results/algen_native").mkdir(exist_ok=True)
with (REPO_ROOT / "results/algen_native/baselines.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
