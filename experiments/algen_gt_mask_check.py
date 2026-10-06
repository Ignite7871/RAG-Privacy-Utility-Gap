"""Effect of the reference implementation's ground-truth attention mask on ALGEN.

The authors' attack passes the tokenized TRUE text's attention mask to the generator (attacker_gt.py), which
leaks each passage's length. Our AlgenAttacker.attack() only sees embeddings and uses an all-ones mask. This
script fits ALGEN once per setting and decodes the same test embeddings both ways, on the 32-token native-setting
caches (msmarco32/nq32), and reports plain and content ROUGE-L. Run under .venv-algen-legacy (transformers 4.52.4).
-> results/algen_native/gt_mask_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md)
from attackers.algen import AlgenAttacker, _tokenize_with_punctuation  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
CKPT = REPO_ROOT / "checkpoints" / "algen_generator" / "checkpoint_epoch_99.pt"
N_ALIGN, N_TEST = 1000, 500


def main() -> None:
    out = {}
    for corpus, enc in (("msmarco32", "minilm"), ("msmarco32", "gtr"), ("nq32", "minilm")):
        a = torch.load(CACHE / f"{corpus}_{enc}_align.pt", weights_only=False)
        t = torch.load(CACHE / f"{corpus}_{enc}_test.pt", weights_only=False)
        atk = AlgenAttacker(generator_checkpoint=str(CKPT))
        atk.fit(a["embeddings"][:N_ALIGN], a["text"][:N_ALIGN])
        test_emb, test_txt = t["embeddings"][:N_TEST], t["text"][:N_TEST]
        blind = atk.attack(test_emb)
        aligned = test_emb.to(atk.device) @ atk.T
        mask = _tokenize_with_punctuation(test_txt, atk.generator.tokenizer, atk.max_length, atk.device)["attention_mask"]
        with torch.no_grad():
            ids = atk.generator.generate({"hidden_states": aligned, "attention_mask": mask})
        leaked = [x.strip() for x in atk.generator.tokenizer.batch_decode(ids, skip_special_tokens=True)]
        out[f"{corpus}/{enc}"] = {"all_ones_mask": full_scores(blind, test_txt), "ground_truth_mask": full_scores(leaked, test_txt)}
        for k, v in out[f"{corpus}/{enc}"].items():
            print(corpus, enc, k, "P=%.3f C=%.3f Crec=%.3f" % (v["rouge_l_precision"], v["content_precision"], v["content_recall"]), flush=True)
    (REPO_ROOT / "results" / "algen_native" / "gt_mask_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
