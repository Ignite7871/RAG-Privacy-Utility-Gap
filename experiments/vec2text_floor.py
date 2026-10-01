"""Embedding-free floor in Vec2Text's 32-token setting (same 200 test prefixes as run_vec2text.py)."""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: F401
from transformers import AutoTokenizer
from attackers.metrics import full_scores
from attackers.vocab_reconstruction import TOKENIZER_NAME, VOCAB_SIZE, mean_token_position, reconstruct_from_logits, texts_to_token_ids
import torch

SEED, N_TEST, N_ALIGN = 42, 200, 5000
cache = REPO_ROOT / "data" / "cache"
t5 = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
bert = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
def prefixes(texts):
    ids = t5(texts, truncation=True, max_length=32, add_special_tokens=False)["input_ids"]
    return [t5.decode(i) for i in ids]
test = torch.load(cache / "msmarco_gtr_test.pt", weights_only=False)
align = torch.load(cache / "msmarco_gtr_align.pt", weights_only=False)
t_idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(SEED))[:N_TEST]
test_pref = prefixes([test["text"][i] for i in t_idx.tolist()])
a_idx = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(SEED))[:N_ALIGN]
a_pref = prefixes([align["text"][i] for i in a_idx.tolist()])
ids = texts_to_token_ids(bert, a_pref)
df = torch.zeros(VOCAB_SIZE)
for l in ids: df[list(set(l))] += 1
logits = df.log1p().unsqueeze(0).repeat(N_TEST, 1)
preds = reconstruct_from_logits(logits, bert, mean_token_position(ids), 16)
r = full_scores(preds, test_pref)
print(json.dumps(r, indent=1))
Path(REPO_ROOT / "results" / "vec2text" / "floor_32tok.json").write_text(json.dumps(r, indent=2))
