"""Two controls for interpreting the key-rotation results.

(1) Context-free floor: ROUGE-L precision of an attacker that ignores the embedding entirely and emits
    the corpus's most frequent tokens (a LinearProbeAttacker whose weights are zero, i.e. bias only).
    Every stale-attacker number should be read against this floor, not against zero.

(2) Snapshot-linking attack on rotation: an attacker holding a probe fit under rotation window 0
    (stale) plus two ROW-ALIGNED snapshots of the stored index, before (X R0) and after (X R1) a
    rotation, estimates R0^T R1 by orthogonal Procrustes from m matched rows (no text needed),
    maps the new snapshot back to window-0 coordinates and applies the stale probe.

Results -> results/rotation_snapshot_linking.csv, results/prior_floor.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see CLAUDE.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402, F401

from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from attackers.vocab_reconstruction import TOKENIZER_NAME, VOCAB_SIZE, mean_token_position, reconstruct_from_logits, texts_to_token_ids  # noqa: E402
from defenses.rotating_encoder import RotatingEncoderDefense  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
N_TEST = 2000
N_ALIGN = 1000
M_LIST = [10, 50, 100, 200, 384, 500, 1000, 5000]
SEEDS = [0, 1, 2]


def main() -> None:
    # ---- (1) context-free floor, all corpus/encoder pairs use the same text pools ----
    rows = []
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    for corpus in ("msmarco", "nq"):
        for enc, dim in (("minilm", 384), ("bge", 1024)):
            align = torch.load(CACHE / f"{corpus}_{enc}_align.pt", weights_only=False)
            test = torch.load(CACHE / f"{corpus}_{enc}_test.pt", weights_only=False)
            idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(123))[:N_TEST]
            t_emb, t_txt = test["embeddings"][idx], [test["text"][i] for i in idx.tolist()]
            for n in (1000, 40000):
                # Context-free attacker: emit the top-K tokens by document frequency in the alignment
                # texts, ordered by mean corpus position (exactly LinearProbe's reconstruction rule, with
                # the embedding's contribution removed). No training, no GPU.
                ids = texts_to_token_ids(tok, align["text"][:n])
                df = torch.zeros(VOCAB_SIZE)
                for lst in ids:
                    df[list(set(lst))] += 1
                order = mean_token_position(ids)
                logits = df.log1p().unsqueeze(0).repeat(len(t_txt), 1)
                r = full_scores(reconstruct_from_logits(logits, tok, order, 16), t_txt)
                rows.append({"corpus": corpus, "encoder": enc, "n_align": n, **r})
                print("floor", rows[-1], flush=True)
    with (REPO_ROOT / "results" / "prior_floor.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- (2) snapshot-linking attack (MiniLM / MS MARCO) ----
    align = torch.load(CACHE / "msmarco_minilm_align.pt", weights_only=False)
    test = torch.load(CACHE / "msmarco_minilm_test.pt", weights_only=False)
    idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(123))[:N_TEST]
    t_txt = [test["text"][i] for i in idx.tolist()]
    store = test["embeddings"]  # the 10k stored vectors; snapshots are row-aligned
    dim = store.shape[1]
    out = []
    for seed in SEEDS:
        defense = RotatingEncoderDefense(dim=dim, seed=42 + seed)
        s0, s1 = defense.transform(store, 0), defense.transform(store, 1)
        perm = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(seed))[:N_ALIGN]
        torch.manual_seed(seed)
        probe = LinearProbeAttacker(embedding_dim=dim)
        probe.fit(defense.transform(align["embeddings"][perm], 0), [align["text"][i] for i in perm.tolist()])
        stale = full_scores(probe.attack(s1[idx]), t_txt)
        synced = full_scores(probe.attack(s0[idx]), t_txt)
        for m in M_LIST:
            rows_m = torch.randperm(len(store), generator=torch.Generator().manual_seed(seed + 7))[:m]
            u, _, vh = torch.linalg.svd(s1[rows_m].T @ s0[rows_m])  # Procrustes: find Q with s1 Q ~ s0
            q = u @ vh
            mapped = s1[idx] @ q
            p = full_scores(probe.attack(mapped), t_txt)
            out.append({"seed": seed, "m_matched_rows": m,
                        "linked_P": p["rouge_l_precision"], "stale_P": stale["rouge_l_precision"], "synced_P": synced["rouge_l_precision"],
                        "linked_C": p["content_precision"], "stale_C": stale["content_precision"], "synced_C": synced["content_precision"]})
            print(out[-1], flush=True)
    with (REPO_ROOT / "results" / "rotation_snapshot_linking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
