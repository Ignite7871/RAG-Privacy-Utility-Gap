"""AlgenAttacker built on the ALGEN authors' own code, imported rather than copied.

ALGEN: Chen, Xu and Bjerva, "ALGEN: Few-shot Inversion Attacks on Textual Embeddings via Cross-Model Alignment and
Generation", ACL 2025 (https://aclanthology.org/2025.acl-long.1185/); code: https://github.com/siebeniris/ALGEN.

That repository carries no license, so this file contains none of its code. It imports the generator model
(`decoder_finetune.DecoderFinetuneModel`) and two helpers (`inversion_utils.add_punctuation_token_ids`,
`inversion_utils.mean_pool`) from a local clone, and adds only what is ours: the ridge-regularised closed-form alignment
(the normal equation), the fit/attack interface of `attackers.base.InversionAttacker`, and the all-ones attention mask.

Setup (once):
    git clone https://github.com/siebeniris/ALGEN third_party/ALGEN
    git -C third_party/ALGEN checkout 08b892391075de841991a03c447800ec2356925
or point the environment variable ALGEN_SRC at the clone's `src` directory. Run only in an environment with
`transformers==4.52.4` (the released checkpoint decodes empty or degenerate text under transformers 5.x without raising).

The released generator checkpoint (Zenodo record 15639971, `mmarco_english`) is downloaded separately and passed as
`generator_checkpoint`.

Difference from the reference evaluation: at attack time we pass an all-ones attention mask instead of the true text's
mask, because our interface receives embeddings only; on 32-token texts the two are identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PINNED_COMMIT = "08b892391075de841991a03c447800ec2356925"
ALGEN_SRC = Path(os.environ.get("ALGEN_SRC", Path(__file__).resolve().parent.parent / "third_party" / "ALGEN" / "src"))
if not (ALGEN_SRC / "decoder_finetune.py").exists():
    raise ImportError(
        f"ALGEN source not found at {ALGEN_SRC}. Clone https://github.com/siebeniris/ALGEN and check out {PINNED_COMMIT}, "
        "or set ALGEN_SRC to its src/ directory (see this file's docstring)."
    )
sys.path.insert(0, str(ALGEN_SRC))

# HuggingFace import must precede torch import (CUDA DLL conflicts on Windows)
from transformers import AutoTokenizer  # noqa: E402,F401

from attackers.base import InversionAttacker  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from decoder_finetune import DecoderFinetuneModel  # noqa: E402  (authors' code)
from inversion_utils import add_punctuation_token_ids, mean_pool  # noqa: E402  (authors' code)


# Kept for scripts written against the earlier interface (same signature as the authors' helper).
_tokenize_with_punctuation = add_punctuation_token_ids

class AlgenAttacker(InversionAttacker):
    def __init__(
        self,
        generator_checkpoint: str,
        generator_model_name: str = "google/flan-t5-small",
        max_length: int = 32,
        reg_lambda: float | None = 1.0,
        device: str | None = None,
    ) -> None:
        self.max_length = max_length
        self.reg_lambda = reg_lambda
        self.generator = DecoderFinetuneModel(generator_model_name, max_length)
        self.device = self.generator.device
        # weights_only=False: the checkpoint's metadata contains numpy scalars; trusted source only.
        checkpoint = torch.load(generator_checkpoint, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint["model_state_dict"])
        self.generator.eval()
        self.T: torch.Tensor | None = None

    def _target_embeddings(self, texts: list[str]) -> torch.Tensor:
        tokens = add_punctuation_token_ids(texts, self.generator.tokenizer, self.max_length, self.device)
        with torch.no_grad():
            hidden = self.generator.encoder_decoder.encoder(**tokens).last_hidden_state
        return F.normalize(mean_pool(hidden, tokens["attention_mask"]), p=2, dim=1)

    def fit(self, alignment_embeddings: torch.Tensor, alignment_texts: list[str]) -> None:
        X = alignment_embeddings.to(self.device)
        Y = self._target_embeddings(alignment_texts)
        # Closed-form ridge alignment of the victim space onto the generator's encoder space (the "normal equation").
        lhs = X.T @ X
        if self.reg_lambda:
            lhs = lhs + self.reg_lambda * torch.eye(lhs.shape[0], dtype=X.dtype, device=X.device)
        self.T = torch.linalg.pinv(lhs) @ X.T @ Y

    def attack(self, test_embeddings: torch.Tensor) -> list[str]:
        if self.T is None:
            raise RuntimeError("AlgenAttacker.attack() called before fit()")
        aligned = test_embeddings.to(self.device) @ self.T
        mask = torch.ones(aligned.shape[0], self.max_length, dtype=torch.long, device=self.device)
        self.generator.eval()
        with torch.no_grad():
            ids = self.generator.generate({"hidden_states": aligned, "attention_mask": mask})
        return [t.strip() for t in self.generator.tokenizer.batch_decode(ids, skip_special_tokens=True)]
