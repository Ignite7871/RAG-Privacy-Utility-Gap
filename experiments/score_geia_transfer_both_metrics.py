"""Re-score the saved GEIA / transfer-attack reconstructions with the content metric and add a
matched LinearProbe row (first 2,000 align pairs, first 200 test passages, MiniLM / MS MARCO) and the
embedding-free floors on the same 200 passages. -> results/geia_transfer_both_metrics.json"""
from __future__ import annotations
import json, sys
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

R = REPO_ROOT / "results"
out = {}
for name, path in (("geia", R / "geia/msmarco_minilm_n2000.json"), ("transfer", R / "transfer_attack/msmarco_minilm_n2000.json")):
    d = json.load(open(path, encoding="utf-8"))
    out[name] = full_scores([x["reconstruction"] for x in d["reconstructions"]], [x["source"] for x in d["reconstructions"]])
a = torch.load(REPO_ROOT / "data/cache/msmarco_minilm_align.pt", weights_only=False)
t = torch.load(REPO_ROOT / "data/cache/msmarco_minilm_test.pt", weights_only=False)
txt = t["text"][:200]
torch.manual_seed(0)
lp = LinearProbeAttacker(embedding_dim=384)
lp.fit(a["embeddings"][:2000], a["text"][:2000])
out["linearprobe_n2000"] = full_scores(lp.attack(t["embeddings"][:200]), txt)
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
ids = texts_to_token_ids(tok, a["text"][:2000])
df = torch.zeros(VOCAB_SIZE)
for l in ids: df[list(set(l))] += 1
order = mean_token_position(ids)
out["floor_top16"] = full_scores(reconstruct_from_logits(df.log1p().unsqueeze(0).repeat(200, 1), tok, order, 16), txt)
stop = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
for w in ENGLISH_STOP_WORDS:
    for i in tok(w, add_special_tokens=False)["input_ids"]: stop[i] = True
df[stop] = 0; df[:1000] = 0
out["floor_content_top16"] = full_scores(reconstruct_from_logits(df.log1p().unsqueeze(0).repeat(200, 1), tok, order, 16), txt)
(R / "geia_transfer_both_metrics.json").write_text(json.dumps(out, indent=2))
for k, v in out.items(): print(k, {m: round(x, 3) for m, x in v.items() if m in ("rouge_l_precision", "content_precision", "content_recall")})
