"""Vec2Text (Morris et al., 2024) vs LinearProbe, evaluated in Vec2Text's native setting.

The released jxm/gtr__nq__32 inverter+corrector does NOT operate on this repo's cached GTR
embeddings: its embedder is a bare mean-pooled gtr-t5-base encoder over at most 32 tokens,
whereas data/encode.py stores SentenceTransformer outputs (extra Dense projection + L2
normalisation, full-length passages). Feeding cached embeddings to Vec2Text silently yields
unrelated text (verified in a smoke test). We therefore embed MS MARCO passages, truncated
to their first 32 embedder tokens, with Vec2Text's own embedder, and fit LinearProbe on
embeddings from the same embedder and truncation, so both attackers see identical victim
embeddings. Scores are ROUGE-L against the 32-token prefix (native setting) and, secondarily,
against the full passage.

Run under .venv-vec2text (transformers==4.52.4): vec2text 0.0.13 is incompatible with
transformers>=5. Predictions are flushed per batch (resumable).
Outputs -> results/vec2text/{summary.json, preds.json}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# vec2text unconditionally imports the Unix-only `resource` module; stub it (never called).
_resource_stub = types.ModuleType("resource")
_resource_stub.RLIMIT_CORE = 0
_resource_stub.RLIM_INFINITY = -1
_resource_stub.setrlimit = lambda *a, **k: None
sys.modules["resource"] = _resource_stub

# HuggingFace import must precede torch import (see CLAUDE.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores, token_f1_corpus  # noqa: E402

import torch  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
OUT_DIR = REPO_ROOT / "results" / "vec2text"
SEED = 42
N_TEST = 200
N_ALIGN_LIST = [800, 5000]
MAX_TOKENS = 32  # jxm/gtr__nq__32 sequence length


def prefixes(tokenizer, texts: list[str]) -> list[str]:
    ids = tokenizer(texts, truncation=True, max_length=MAX_TOKENS, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(i) for i in ids]


@torch.no_grad()
def embed(corrector, texts: list[str], device: str, bs: int = 64) -> torch.Tensor:
    out = []
    for s in range(0, len(texts), bs):
        enc = corrector.embedder_tokenizer(
            texts[s : s + bs], return_tensors="pt", max_length=MAX_TOKENS, truncation=True, padding="max_length"
        ).to(device)
        out.append(corrector.inversion_trainer.call_embedding_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).float().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N test passages (smoke test)")
    args = ap.parse_args()

    import vec2text

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrector = vec2text.load_pretrained_corrector("gtr-base")

    test = torch.load(CACHE_DIR / "msmarco_gtr_test.pt", weights_only=False)
    align = torch.load(CACHE_DIR / "msmarco_gtr_align.pt", weights_only=False)
    t_idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(SEED))[: args.limit or N_TEST]
    test_full = [test["text"][i] for i in t_idx.tolist()]
    tok = corrector.embedder_tokenizer
    test_pref = prefixes(tok, test_full)
    test_emb = embed(corrector, test_pref, device)

    tag = f"steps{args.num_steps}_beam{args.beam}" + ("_smoke" if args.limit else "")
    preds_path = OUT_DIR / f"preds_{tag}.json"
    preds: list[str] = json.loads(preds_path.read_text())["predictions"] if preds_path.exists() else []
    t0 = time.time()
    for s in range(len(preds), len(test_pref), args.batch_size):
        preds += vec2text.invert_embeddings(
            embeddings=test_emb[s : s + args.batch_size].to(device),
            corrector=corrector, num_steps=args.num_steps, sequence_beam_width=args.beam,
        )
        preds_path.write_text(json.dumps({"predictions": preds}))
        print(f"  vec2text {len(preds)}/{len(test_pref)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    result = {
        "n_test": len(test_pref),
        "setting": "MS MARCO passages truncated to 32 gtr-t5 tokens; victim embeddings from vec2text's own gtr-base embedder",
        "vec2text_settings": {"checkpoint": "jxm/gtr__nq__32", "num_steps": args.num_steps, "sequence_beam_width": args.beam},
        "vec2text": {"vs_prefix": full_scores(preds, test_pref), "vs_full": full_scores(preds, test_full),
                     "token_f1_vs_prefix": token_f1_corpus(preds, test_pref)},
        "linearprobe": {},
        "examples": [],
    }
    a_perm = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(SEED))
    lp_preds_for_examples = None
    for n in N_ALIGN_LIST:
        a_full = [align["text"][i] for i in a_perm[:n].tolist()]
        a_emb = embed(corrector, prefixes(tok, a_full), device)
        torch.manual_seed(SEED)
        lp = LinearProbeAttacker(embedding_dim=a_emb.shape[1])
        lp.fit(a_emb, prefixes(tok, a_full))
        p = lp.attack(test_emb)
        result["linearprobe"][str(n)] = {"vs_prefix": full_scores(p, test_pref), "vs_full": full_scores(p, test_full),
                                         "token_f1_vs_prefix": token_f1_corpus(p, test_pref)}
        lp_preds_for_examples = p if n == N_ALIGN_LIST[0] else lp_preds_for_examples
    result["examples"] = [{"source_prefix": s, "vec2text": v, "linearprobe_n800": l}
                          for s, v, l in list(zip(test_pref, preds, lp_preds_for_examples))[:10]]

    out = OUT_DIR / f"summary_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    r = result["vec2text"]["vs_prefix"]
    print(f"vec2text vs prefix: P={r['rouge_l_precision']:.4f} contentP={r['content_precision']:.4f}")
    for n, v in result["linearprobe"].items():
        r = v["vs_prefix"]
        print(f"linearprobe n={n} vs prefix: P={r['rouge_l_precision']:.4f} contentP={r['content_precision']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
