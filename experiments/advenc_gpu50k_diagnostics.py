"""Diagnostics on the GPU-scale checkpoint (checkpoints/advenc_v2_gpu50k_minilm.pt),
mirroring the four CPU-scale checks exactly (trivial baseline, downstream attack eval,
collapse/CKA, effective rank + centroid shift) for direct comparability -- same methods,
same 400-passage sample (first 400 of msmarco_minilm_test.pt), same vanilla baseline.

For the scale comparison (50,000 vs 1,600 pairs, 20 vs 5 epochs) it records: whether the
internal decoder's loss converges near the trivial marginal-frequency baseline, whether the
encoder shifts the embedding distribution toward a new centroid (effective rank and
participation ratio), and how much of the non-adaptive precision loss an adaptive
attacker recovers.

Results -> results/advenc/gpu50k_diagnostics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on
# Windows). Empirically `datasets` must be imported before `sentence_transformers`
# specifically, or the process crashes with an access violation -- see
# experiments/advenc_cpu_scale_check.py. This must stay the first import in this file;
# the modules imported below each do their own correct internal ordering, but only
# once `datasets` has already been loaded once, process-wide.
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import rouge_l_corpus  # noqa: E402
from defenses.adv_encoder import build_self_pairs  # noqa: E402
from experiments.advenc_cpu_scale_check import (  # noqa: E402
    encode_texts,
    load_defended_encoder,
    trivial_baseline_bce,
)
from experiments.check_fm1_collapse import (  # noqa: E402
    N_PASSAGES,
    encode,
    linear_cka,
    load_encoder,
    pairwise_cosine_stats,
)
from experiments.check_fm1_rank import rank_stats  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "advenc_v2_gpu50k_minilm.pt"
TRAINLOG_PATH = REPO_ROOT / "results" / "advenc" / "v2_gpu50k_minilm_trainlog.json"
OUT_PATH = REPO_ROOT / "results" / "advenc" / "gpu50k_diagnostics.json"

EMBEDDING_DIM = 384

# CPU-scale v2 reference numbers, for the side-by-side table.
CPU_SCALE_V2 = {
    "trivial_baseline": 0.007221239618957043,
    "final_decoder_loss": 0.0067,
    "nonadaptive_rougeL_precision": 0.3870,
    "adaptive_rougeL_precision": 0.4072,
    "mean_cosine_sim": 0.1351,
    "effective_rank_95": 191,
    "centroid_norm": 0.3705,
}
VANILLA_ROUGEL_PRECISION = 0.4200


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    # ---- Part 1: trivial baseline check ----
    print("=== Part 1: trivial baseline check (50,000-pair GPU-scale pool) ===", flush=True)
    pairs = build_self_pairs(50_000)
    passages = [p for _, p in pairs]
    trivial = trivial_baseline_bce(passages)

    with open(TRAINLOG_PATH, encoding="utf-8") as f:
        trainlog = json.load(f)
    final_decoder_loss = trainlog["decoder_loss"][-1]

    print(f"trivial={trivial}  actual_final_decoder_loss={final_decoder_loss:.4f}")
    results["trivial_baseline"] = {**trivial, "actual_final_decoder_loss": final_decoder_loss}

    # Everything from here on only loads already-cached models/checkpoints (MiniLM,
    # bert-base-uncased tokenizer, our own checkpoint file) -- no new Hub downloads
    # needed. Force offline mode to avoid a recurring flaky HEAD request this
    # environment makes when checking for PEFT adapter configs on every fresh
    # SentenceTransformer(...) construction, which has twice killed this run with a
    # DNS/closed-client error even though the model is fully local already.
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"

    # ---- Part 2: downstream attack eval ----
    print("\n=== Part 2: downstream attack eval ===", flush=True)
    align_data = torch.load(CACHE_DIR / "msmarco_minilm_align.pt", weights_only=False)
    test_data = torch.load(CACHE_DIR / "msmarco_minilm_test.pt", weights_only=False)
    vanilla_align_emb, align_texts = align_data["embeddings"], align_data["text"]
    vanilla_test_emb, test_texts = test_data["embeddings"], test_data["text"]

    torch.manual_seed(42)
    print("fitting vanilla-baseline LinearProbeAttacker...", flush=True)
    vanilla_attacker = LinearProbeAttacker(embedding_dim=EMBEDDING_DIM)
    vanilla_attacker.fit(vanilla_align_emb, align_texts)

    defended_encoder = load_defended_encoder(CHECKPOINT_PATH, device)
    print("encoding align+test texts through gpu50k defended encoder...", flush=True)
    defended_align_emb = encode_texts(defended_encoder, align_texts)
    defended_test_emb = encode_texts(defended_encoder, test_texts)
    del defended_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nonadaptive_preds = vanilla_attacker.attack(defended_test_emb)
    nonadaptive_rouge = rouge_l_corpus(nonadaptive_preds, test_texts)
    print(f"non-adaptive: rougeL_precision={nonadaptive_rouge['precision']:.4f}")

    torch.manual_seed(42)
    adaptive_attacker = LinearProbeAttacker(embedding_dim=EMBEDDING_DIM)
    adaptive_attacker.fit(defended_align_emb, align_texts)
    adaptive_preds = adaptive_attacker.attack(defended_test_emb)
    adaptive_rouge = rouge_l_corpus(adaptive_preds, test_texts)
    print(f"adaptive: rougeL_precision={adaptive_rouge['precision']:.4f}")

    results["downstream_attack"] = {
        "nonadaptive_rougeL_precision": nonadaptive_rouge["precision"],
        "adaptive_rougeL_precision": adaptive_rouge["precision"],
    }

    # ---- Part 3+4: collapse/CKA, effective rank, centroid shift (same 400-passage sample) ----
    print(
        "\n=== Part 3+4: collapse/CKA, effective rank, centroid shift "
        "(same 400-passage sample as CPU-scale check) ===",
        flush=True,
    )
    sample_texts = test_texts[:N_PASSAGES]

    vanilla_model = load_encoder(None, device)
    vanilla_emb_400 = encode(vanilla_model, sample_texts).cpu()
    del vanilla_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    defended_model = load_encoder(CHECKPOINT_PATH, device)
    defended_emb_400 = encode(defended_model, sample_texts).cpu()
    del defended_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_cos, std_cos = pairwise_cosine_stats(defended_emb_400)
    cka = linear_cka(vanilla_emb_400, defended_emb_400)
    rank = rank_stats(defended_emb_400)

    vanilla_centroid = vanilla_emb_400.mean(dim=0)
    defended_centroid = defended_emb_400.mean(dim=0)
    centroid_norm = defended_centroid.norm(p=2).item()
    centroid_cos_sim = F.cosine_similarity(
        vanilla_centroid.unsqueeze(0), defended_centroid.unsqueeze(0)
    ).item()
    centroid_l2_dist = (vanilla_centroid - defended_centroid).norm(p=2).item()

    print(f"mean_cos={mean_cos:.4f} std_cos={std_cos:.4f} CKA_vs_vanilla={cka:.4f}")
    print(
        f"effective_rank_95={rank['effective_rank_95']} "
        f"participation_ratio={rank['participation_ratio']:.2f} "
        f"top_singular_value_fraction={rank['top_singular_value_fraction']:.4f}"
    )
    print(
        f"centroid_norm={centroid_norm:.4f} cos_sim_vs_vanilla={centroid_cos_sim:.4f} "
        f"l2_dist_vs_vanilla={centroid_l2_dist:.4f}"
    )

    results["collapse_rank_centroid"] = {
        "mean_cosine_sim": mean_cos,
        "std_cosine_sim": std_cos,
        "CKA_vs_vanilla": cka,
        "effective_rank_95": rank["effective_rank_95"],
        "participation_ratio": rank["participation_ratio"],
        "top_singular_value_fraction": rank["top_singular_value_fraction"],
        "centroid_norm": centroid_norm,
        "centroid_cos_sim_vs_vanilla": centroid_cos_sim,
        "centroid_l2_dist_vs_vanilla": centroid_l2_dist,
    }

    # ---- summary table: CPU-scale v2 vs gpu50k v2, side by side ----
    print("\n" + "=" * 78)
    print(f"{'metric':<32} {'cpu-scale v2':>18} {'gpu50k v2':>18}")
    rows = [
        ("trivial_baseline", CPU_SCALE_V2["trivial_baseline"], trivial["marginal_frequency_baseline"]),
        ("final_decoder_loss", CPU_SCALE_V2["final_decoder_loss"], final_decoder_loss),
        (
            "nonadaptive_rougeL_precision",
            CPU_SCALE_V2["nonadaptive_rougeL_precision"],
            nonadaptive_rouge["precision"],
        ),
        ("adaptive_rougeL_precision", CPU_SCALE_V2["adaptive_rougeL_precision"], adaptive_rouge["precision"]),
        ("mean_cosine_sim", CPU_SCALE_V2["mean_cosine_sim"], mean_cos),
        ("effective_rank_95", CPU_SCALE_V2["effective_rank_95"], rank["effective_rank_95"]),
        ("centroid_norm", CPU_SCALE_V2["centroid_norm"], centroid_norm),
    ]
    for name, cpu_val, gpu_val in rows:
        print(f"{name:<32} {float(cpu_val):>18.4f} {float(gpu_val):>18.4f}")
    print(f"{'vanilla_rougeL_precision (ref)':<32} {VANILLA_ROUGEL_PRECISION:>18.4f} {'--':>18}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
