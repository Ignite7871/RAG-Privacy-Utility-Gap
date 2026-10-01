"""Second-encoder check: AdvEnc at a reduced training scale (gpu10k) on MiniLM-L6 and MPNet.

For each encoder, the vanilla encoder and the two defended variants (v1 query-passage, v2 self-pair; 10,000 pairs,
10 epochs, batch 16; defenses/adv_encoder.py --scale gpu10k) are evaluated with
  * an adaptive LinearProbe (n_align in {200, 1,000, 5,000, 10,000} x 3 seeds, and 40,000 x 1 seed),
  * the non-adaptive probe (fit once on 40,000 vanilla pairs, applied to the defended test embeddings),
  * retrieval (1,000 MS MARCO queries, 50,000-passage index, queries and passages encoded by the same encoder).
Test subset and protocol match experiments/advenc_adaptive_budget_sweep.py (2,000 test passages).
Outputs: results/advenc/reduced_scale_eval.csv, results/advenc/reduced_scale_retrieval.json. Resumable.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# `datasets` must be imported first, process-wide (see experiments/advenc_cpu_scale_check.py).
import datasets  # noqa: E402, F401
from sentence_transformers import SentenceTransformer  # noqa: E402

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from data.encode import ENCODERS as ENCODER_IDS  # noqa: E402
from experiments.advenc_cpu_scale_check import encode_texts  # noqa: E402
from experiments.advenc_retrieval_all_variants import per_query_scores, summarise  # noqa: E402
from experiments.advenc_retrieval_eval import sample_queries_with_ground_truth  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
CKPT = REPO_ROOT / "checkpoints"
OUT_CSV = REPO_ROOT / "results" / "advenc" / "reduced_scale_eval.csv"
OUT_RET = REPO_ROOT / "results" / "advenc" / "reduced_scale_retrieval.json"
SCALE = "gpu10k"
ENCODERS = ["minilm", "mpnet"]
# v1s / v2w are the pair-type vs. lambda_priv ablations (MiniLM only); missing checkpoints are skipped.
# A label "v1@b8" means the same variant trained at batch size 8 (scale gpu10k_b8), used for MPNet because its self-pair
# variant does not fit in 8 GB at batch 16.
VARIANTS_BY_ENCODER = {"minilm": ["v1", "v2", "v1s", "v2w"], "mpnet": ["v1", "v2", "v1@b8", "v2@b8", "v1@b8bf16", "v2@b8bf16"]}
BUDGETS = [200, 1000, 5000, 10000, 40000]
SEEDS = [0, 1, 2]
N_QUERIES = 1000
FIELDS = ["encoder", "variant", "attacker", "n_align", "seed", "n_test", "rouge_l_precision", "rouge_l_recall", "rouge_l_f1",
          "content_precision", "content_recall", "content_f1"]


def load_vanilla(enc):
    return (torch.load(CACHE / f"msmarco_{enc}_align.pt", weights_only=False), torch.load(CACHE / f"msmarco_{enc}_test.pt", weights_only=False))


def split_label(label):
    variant, _, tag = label.partition("@")
    return variant, {"b8": "gpu10k_b8", "b8bf16": "gpu10k_b8_bf16"}.get(tag, SCALE)


def load_defended_model(enc, variant, device):
    variant, scale = split_label(variant)
    path = CKPT / f"advenc_{variant}_{scale}_{enc}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    model = SentenceTransformer(ckpt["encoder_name"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def defended_embeddings(enc, variant, vanilla_align, vanilla_test, device):
    v, scale = split_label(variant)
    name = f"advenc_{v}_{scale}"
    paths = {s: CACHE / f"{name}_{enc}_{s}.pt" for s in ("align", "test")}
    if not all(p.exists() for p in paths.values()):
        model = load_defended_model(enc, variant, device)
        if model is None:
            return None
        for split, src in (("align", vanilla_align), ("test", vanilla_test)):
            torch.save({"text": src["text"], "embeddings": encode_texts(model, src["text"])}, paths[split])
        del model
        torch.cuda.empty_cache()
    return torch.load(paths["align"], weights_only=False), torch.load(paths["test"], weights_only=False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_CSV.exists():
        with OUT_CSV.open(newline="") as f:
            done = {(r["encoder"], r["variant"], r["attacker"], int(r["n_align"]), int(r["seed"])) for r in csv.DictReader(f)}
    new_file = not OUT_CSV.exists()
    test_idx = torch.randperm(10_000, generator=torch.Generator().manual_seed(123))[:2000]

    # queries shared by every encoder (same corpus pool)
    a0, t0 = load_vanilla("minilm")
    corpus_texts = a0["text"] + t0["text"]
    index = {t: i for i, t in enumerate(corpus_texts)}
    pairs = sample_queries_with_ground_truth(index, N_QUERIES)
    q_texts, rel = [q for q, _ in pairs], [i for _, i in pairs]
    retrieval = json.loads(OUT_RET.read_text()) if OUT_RET.exists() else {}

    with OUT_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for enc in ENCODERS:
            van_align, van_test = load_vanilla(enc)
            test_txt = [van_test["text"][i] for i in test_idx.tolist()]
            dim = van_align["embeddings"].shape[1]
            perm0 = torch.randperm(len(van_align["text"]), generator=torch.Generator().manual_seed(0))
            nonadaptive_probe = None
            variants = [("vanilla", (van_align, van_test))]
            for v in VARIANTS_BY_ENCODER[enc]:
                emb = defended_embeddings(enc, v, van_align, van_test, device)
                if emb is None:
                    print(f"[skip] no checkpoint for {enc} {v} {SCALE}", flush=True)
                    continue
                variants.append((v, emb))
            for variant, (align, test) in variants:
                test_emb = test["embeddings"][test_idx]
                for n in BUDGETS:
                    for seed in ([0] if n == 40000 else SEEDS):
                        key = (enc, variant, "adaptive", n, seed)
                        need_probe = variant == "vanilla" and n == 40000
                        if key in done and not need_probe:
                            continue
                        torch.manual_seed(seed)
                        perm = perm0 if n == 40000 else torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(seed))
                        idx = perm[:n]
                        atk = LinearProbeAttacker(embedding_dim=dim)
                        atk.fit(align["embeddings"][idx], [align["text"][i] for i in idx.tolist()])
                        if need_probe:
                            nonadaptive_probe = atk
                        if key in done:
                            continue
                        r = full_scores(atk.attack(test_emb), test_txt)
                        w.writerow({"encoder": enc, "variant": variant, "attacker": "adaptive", "n_align": n, "seed": seed, "n_test": 2000, **r})
                        fh.flush()
                        print(enc, variant, "adaptive", n, seed, "P=%.3f C=%.3f" % (r["rouge_l_precision"], r["content_precision"]), flush=True)
                if variant != "vanilla" and nonadaptive_probe is not None and (enc, variant, "nonadaptive", 40000, 0) not in done:
                    r = full_scores(nonadaptive_probe.attack(test_emb), test_txt)
                    w.writerow({"encoder": enc, "variant": variant, "attacker": "nonadaptive", "n_align": 40000, "seed": 0, "n_test": 2000, **r})
                    fh.flush()
                    print(enc, variant, "nonadaptive", "P=%.3f C=%.3f" % (r["rouge_l_precision"], r["content_precision"]), flush=True)

                # retrieval: queries and corpus through the same (possibly defended) encoder
                rkey = f"{enc}/{variant}"
                if rkey not in retrieval:
                    if variant == "vanilla":
                        model = SentenceTransformer(ENCODER_IDS[enc][0]).to(device)
                        corpus = torch.cat([van_align["embeddings"], van_test["embeddings"]])
                    else:
                        model = load_defended_model(enc, variant, device)
                        corpus = torch.cat([align["embeddings"], test["embeddings"]])
                    q = encode_texts(model, q_texts)
                    del model
                    torch.cuda.empty_cache()
                    retrieval[rkey] = summarise(per_query_scores(q, corpus, rel))
                    OUT_RET.write_text(json.dumps({"n_queries": N_QUERIES, **retrieval}, indent=2))
                    print(rkey, {k: round(v["mean"], 4) for k, v in retrieval[rkey].items() if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
