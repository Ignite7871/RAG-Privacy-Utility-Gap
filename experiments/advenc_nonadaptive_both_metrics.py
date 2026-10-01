"""Non-adaptive attacker (A1) against each AdvEnc variant, both metrics.
A LinearProbe is fit ONCE on n=40,000 vanilla alignment pairs and applied to the vanilla and defended
test embeddings (same 2,000-passage subset as the adaptive sweep). -> results/advenc/nonadaptive_both_metrics.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: E402, F401
from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
import torch  # noqa: E402

C = REPO_ROOT / "data" / "cache"
a = torch.load(C / "msmarco_minilm_align.pt", weights_only=False)
t = torch.load(C / "msmarco_minilm_test.pt", weights_only=False)
idx = torch.randperm(10_000, generator=torch.Generator().manual_seed(123))[:2000]
txt = [t["text"][i] for i in idx.tolist()]
torch.manual_seed(0)
atk = LinearProbeAttacker(embedding_dim=384)
atk.fit(a["embeddings"], a["text"])
out = {"vanilla": full_scores(atk.attack(t["embeddings"][idx]), txt)}
for name in ("advenc_v1_cpu", "advenc_v2_cpu", "advenc_v2_gpu50k", "advenc_v1_gpu50k"):
    d = torch.load(C / f"{name}_minilm_test.pt", weights_only=False)
    out[name] = full_scores(atk.attack(d["embeddings"][idx]), txt)
(REPO_ROOT / "results/advenc/nonadaptive_both_metrics.json").write_text(json.dumps(out, indent=2))
for k, v in out.items(): print(k, "P=%.3f C=%.3f" % (v["rouge_l_precision"], v["content_precision"]))
