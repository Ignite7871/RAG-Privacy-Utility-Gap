"""Document how the released ALGEN generator behaves under transformers 5.x (run in the MAIN environment, .venv312).

Stage 1: the authors' unmodified DecoderFinetuneModel.generate() -> records the exception raised.
Stage 2: after moving max_length from the model config to generation_config (what the error message suggests), decode the
first 8 test embeddings and record how many outputs are empty. Uses the 32-token msmarco32/minilm caches, n_align=1000.
Needs ALGEN_SRC (see attackers/algen_adapter.py). -> results/algen_native/transformers5_check.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import transformers  # noqa: E402
from attackers.algen_adapter import AlgenAttacker  # noqa: E402
import torch  # noqa: E402

C = REPO_ROOT / "data" / "cache"
a = torch.load(C / "msmarco32_minilm_align.pt", weights_only=False)
t = torch.load(C / "msmarco32_minilm_test.pt", weights_only=False)
atk = AlgenAttacker(generator_checkpoint=str(REPO_ROOT / "checkpoints/algen_generator/checkpoint_epoch_99.pt"))
atk.fit(a["embeddings"][:1000], a["text"][:1000])
report = {"transformers": transformers.__version__}
try:
    atk.attack(t["embeddings"][:8])
    report["stage1"] = "no exception"
except Exception as e:  # noqa: BLE001
    report["stage1"] = {"exception": type(e).__name__, "message": str(e)[:240]}
ed = atk.generator.encoder_decoder
ed.config.max_length = None
ed.generation_config.max_length = 32
out = atk.attack(t["embeddings"][:8])
report["stage2"] = {"n": len(out), "n_empty": sum(1 for o in out if not o.strip()), "outputs": out}
(REPO_ROOT / "results/algen_native/transformers5_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "stage2"}, indent=1)[:400]); print("n_empty", report["stage2"]["n_empty"], "of", report["stage2"]["n"])
