"""Build 32-token-prefix embedding caches for ALGEN's native setting.

The authors' ALGEN pipeline aligns and evaluates on texts truncated to 32 tokens: victim embeddings X are
computed from that same short text, and the generator's target embeddings Y come from the identical text.
Our main experiments embed full-length passages, so X (full passage) and Y (32-token prefix) describe
different texts. This script writes caches in the same format as data/encode.py, with dataset names
`msmarco32` / `nq32`, whose texts are the first 32 flan-t5 tokens of the first 2,000 alignment and 500 test
passages of the corresponding corpus, embedded by each victim encoder. They can be attacked with
attackers/run_algen_legacy.py unchanged (--dataset msmarco32 ...).

Run under .venv312 (main environment).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace imports must come before torch (see CLAUDE.md)
import datasets  # noqa: E402, F401
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from data.encode import ENCODERS, cache_path  # noqa: E402

import torch  # noqa: E402

N_ALIGN, N_TEST, MAX_TOKENS = 2000, 500, 32


def main() -> None:
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")

    def prefixes(texts: list[str]) -> list[str]:
        ids = tok(texts, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
        return [tok.decode(i, skip_special_tokens=True) for i in ids]

    for corpus in ("msmarco", "nq"):
        for enc, (model_id, _dim) in ENCODERS.items():
            src = {s: torch.load(cache_path(corpus, enc, s), weights_only=False)["text"] for s in ("align", "test")}
            texts = {"align": prefixes(src["align"][:N_ALIGN]), "test": prefixes(src["test"][:N_TEST])}
            model = SentenceTransformer(model_id)
            for split, t in texts.items():
                emb = model.encode(t, batch_size=128, convert_to_tensor=True, show_progress_bar=False).float()
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu()
                out = cache_path(f"{corpus}32", enc, split)
                torch.save({"text": t, "embeddings": emb}, out)
                print("saved", out.name, tuple(emb.shape), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
