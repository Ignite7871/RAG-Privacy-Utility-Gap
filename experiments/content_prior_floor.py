"""Stronger embedding-free baseline: emit the K most document-frequent NON-stopword tokens.
The plain floor (top tokens by df) is all stopwords, so its content precision is 0 by construction;
this baseline shows what a content-aware but embedding-free guesser achieves. -> results/content_prior_floor.csv"""
from __future__ import annotations
import csv, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: F401
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from attackers.metrics import full_scores
from attackers.vocab_reconstruction import TOKENIZER_NAME, VOCAB_SIZE, mean_token_position, reconstruct_from_logits, texts_to_token_ids
import torch

tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
stop_ids = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
for w in ENGLISH_STOP_WORDS:
    for i in tok(w, add_special_tokens=False)["input_ids"]:
        stop_ids[i] = True
rows = []
for corpus in ("msmarco", "nq"):
    a = torch.load(REPO_ROOT / "data/cache" / f"{corpus}_minilm_align.pt", weights_only=False)
    t = torch.load(REPO_ROOT / "data/cache" / f"{corpus}_minilm_test.pt", weights_only=False)
    idx = torch.randperm(len(t["text"]), generator=torch.Generator().manual_seed(123))[:2000]
    txt = [t["text"][i] for i in idx.tolist()]
    for n in (1000, 40000):
        ids = texts_to_token_ids(tok, a["text"][:n])
        df = torch.zeros(VOCAB_SIZE)
        for l in ids: df[list(set(l))] += 1
        df[stop_ids] = 0
        df[:1000] = 0  # [PAD]/[unused]/punctuation region of the BERT vocab
        for k in (16,):
            preds = reconstruct_from_logits(df.log1p().unsqueeze(0).repeat(len(txt), 1), tok, mean_token_position(ids), k)
            r = full_scores(preds, txt)
            rows.append({"corpus": corpus, "n_align": n, "top_k": k, **r})
            print(rows[-1], flush=True)
with (REPO_ROOT / "results/content_prior_floor.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
