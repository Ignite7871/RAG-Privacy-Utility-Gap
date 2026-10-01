"""CPU-scale AdvEnc sanity checks (before any GPU-scale/50k run):

1. Trivial-baseline BCE check: is the internal adversarial decoder's final training
   loss (see results/advenc/{v1,v2}_cpu_minilm_trainlog.json) actually better than a
   decoder that ignores the embedding entirely and just predicts each vocab token's
   marginal presence rate (the information-theoretic floor for an input-independent
   predictor)? If the trained loss is close to that floor, the decoder learned little
   beyond class-imbalance exploitation, not real reconstruction.

   Two trivial baselines are reported: (a) uniform logit=0 (p=0.5 everywhere), a
   maximum-entropy no-information baseline -- exactly ln(2)=0.693 regardless of target
   distribution, included because it's the literal "constant zero logit" baseline;
   and (b) the per-token marginal-frequency baseline (each of the 30522 outputs
   predicts that specific token's empirical presence rate across the pair set,
   ignoring the input) -- this is the real "cheap trivial solution" a bias-only fit
   could find almost immediately given severe class imbalance, and is the meaningful
   comparison point for the observed ~0.007-0.08 decoder losses.

2. The actual downstream attack eval (the Table IV/VII-equivalent result): freeze each
   CPU-scale defended encoder checkpoint, fit LinearProbeAttacker two ways --
   non-adaptive (fit on VANILLA embeddings, attack the DEFENDED encoder's test
   embeddings) and adaptive (fit AND attack on the DEFENDED encoder's embeddings) --
   and report ROUGE-L precision against the vanilla-encoder baseline. This is distinct
   from and does not follow from the min-max trainlog's internal decoder_loss.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows).
# Empirically this specifically requires `datasets` to be imported before
# `sentence_transformers` -- importing sentence_transformers first crashes the
# process with an access violation (STATUS_ACCESS_VIOLATION) on this machine, not a
# catchable Python exception. See data/encode.py for the same ordering.
import datasets  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus  # noqa: E402
from attackers.vocab_reconstruction import (  # noqa: E402
    TOKENIZER_NAME,
    build_presence_targets,
    texts_to_token_ids,
)
from defenses.adv_encoder import build_query_passage_pairs, build_self_pairs  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
TRAINLOG_DIR = REPO_ROOT / "results" / "advenc"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "cpu_scale_downstream_check.json"

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def trivial_baseline_bce(texts: list[str]) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    targets = build_presence_targets(texts_to_token_ids(tokenizer, texts))

    zero_logits = torch.zeros_like(targets)
    uniform_baseline = F.binary_cross_entropy_with_logits(zero_logits, targets).item()

    p_token = targets.mean(dim=0).clamp(1e-7, 1 - 1e-7)
    marginal_entropy = -(p_token * torch.log(p_token) + (1 - p_token) * torch.log(1 - p_token))
    marginal_baseline = marginal_entropy.mean().item()

    return {"uniform_logit0_baseline": uniform_baseline, "marginal_frequency_baseline": marginal_baseline}


def load_defended_encoder(checkpoint_path: Path, device: torch.device) -> SentenceTransformer:
    model = SentenceTransformer(BASE_MODEL_NAME).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode_texts(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    embeddings = model.encode(texts, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    return F.normalize(embeddings, p=2, dim=1).cpu()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    # ---- Part 1: trivial baseline BCE check ----
    print("=== Part 1: trivial baseline BCE check ===", flush=True)
    v1_pairs = build_query_passage_pairs(1000)
    v2_pairs = build_self_pairs(1600)
    v1_passages = [p for _, p in v1_pairs]
    v2_passages = [p for _, p in v2_pairs]

    v1_trivial = trivial_baseline_bce(v1_passages)
    v2_trivial = trivial_baseline_bce(v2_passages)

    with open(TRAINLOG_DIR / "v1_cpu_minilm_trainlog.json", encoding="utf-8") as f:
        v1_final_decoder_loss = json.load(f)["decoder_loss"][-1]
    with open(TRAINLOG_DIR / "v2_cpu_minilm_trainlog.json", encoding="utf-8") as f:
        v2_final_decoder_loss = json.load(f)["decoder_loss"][-1]

    print(f"v1 (1000 pairs): trivial={v1_trivial}  actual_final_decoder_loss={v1_final_decoder_loss:.4f}")
    print(f"v2 (1600 pairs): trivial={v2_trivial}  actual_final_decoder_loss={v2_final_decoder_loss:.4f}")
    results["trivial_baseline"] = {
        "v1": {**v1_trivial, "actual_final_decoder_loss": v1_final_decoder_loss},
        "v2": {**v2_trivial, "actual_final_decoder_loss": v2_final_decoder_loss},
    }

    # ---- Part 2: downstream attack eval ----
    print("\n=== Part 2: downstream attack eval (LinearProbeAttacker vs AdvEnc) ===", flush=True)
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    vanilla_align_emb, align_texts = align_data["embeddings"], align_data["text"]
    vanilla_test_emb, test_texts = test_data["embeddings"], test_data["text"]

    torch.manual_seed(42)
    print("fitting vanilla-baseline LinearProbeAttacker...", flush=True)
    t0 = time.perf_counter()
    vanilla_attacker = LinearProbeAttacker(embedding_dim=EMBEDDING_DIM)
    vanilla_attacker.fit(vanilla_align_emb, align_texts)
    vanilla_preds = vanilla_attacker.attack(vanilla_test_emb)
    vanilla_rouge = rouge_l_corpus(vanilla_preds, test_texts)
    print(f"vanilla baseline: rougeL_precision={vanilla_rouge['precision']:.4f} ({time.perf_counter() - t0:.1f}s)")
    results["vanilla_baseline"] = {"rougeL_precision": vanilla_rouge["precision"]}

    for variant in ["v1", "v2"]:
        checkpoint_path = CHECKPOINT_DIR / f"advenc_{variant}_cpu_minilm.pt"
        print(f"\n--- {variant} ---", flush=True)
        defended_encoder = load_defended_encoder(checkpoint_path, device)

        print(f"encoding align+test texts through {variant} defended encoder...", flush=True)
        defended_align_emb = encode_texts(defended_encoder, align_texts)
        defended_test_emb = encode_texts(defended_encoder, test_texts)
        del defended_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Non-adaptive: reuse the vanilla-fitted attacker, attack the defended test embeddings.
        nonadaptive_preds = vanilla_attacker.attack(defended_test_emb)
        nonadaptive_rouge = rouge_l_corpus(nonadaptive_preds, test_texts)
        print(f"{variant} non-adaptive: rougeL_precision={nonadaptive_rouge['precision']:.4f}")

        # Adaptive: fresh attacker, fit and attack directly on the defended embeddings.
        torch.manual_seed(42)
        t0 = time.perf_counter()
        adaptive_attacker = LinearProbeAttacker(embedding_dim=EMBEDDING_DIM)
        adaptive_attacker.fit(defended_align_emb, align_texts)
        adaptive_preds = adaptive_attacker.attack(defended_test_emb)
        adaptive_rouge = rouge_l_corpus(adaptive_preds, test_texts)
        print(
            f"{variant} adaptive: rougeL_precision={adaptive_rouge['precision']:.4f} "
            f"({time.perf_counter() - t0:.1f}s)"
        )

        results[variant] = {
            "nonadaptive_rougeL_precision": nonadaptive_rouge["precision"],
            "adaptive_rougeL_precision": adaptive_rouge["precision"],
        }

    print("\n" + "=" * 60)
    print(f"{'condition':<30} {'rougeL_precision':>18}")
    print(f"{'vanilla baseline':<30} {results['vanilla_baseline']['rougeL_precision']:>18.4f}")
    for variant in ["v1", "v2"]:
        print(f"{variant + ' non-adaptive':<30} {results[variant]['nonadaptive_rougeL_precision']:>18.4f}")
        print(f"{variant + ' adaptive':<30} {results[variant]['adaptive_rougeL_precision']:>18.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
