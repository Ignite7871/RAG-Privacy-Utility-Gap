"""Key rotation vs LinearProbe, both metrics, 3 seeds (MiniLM / MS MARCO, n_test=2000).

For each n_align, a probe is fit on alignment pairs rotated under window 0 and scored on test
embeddings rotated under (a) window 0 (same_epoch), (b) window 1 (next_epoch); an unrotated probe
fit on unrotated pairs and scored on unrotated test embeddings is the no-defense reference.
Results -> results/rotation_both_metrics.csv
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: E402, F401
from attackers.linear_probe import LinearProbeAttacker  # noqa: E402
from attackers.metrics import full_scores  # noqa: E402
from defenses.rotating_encoder import RotatingEncoderDefense  # noqa: E402
import torch  # noqa: E402

CACHE = REPO_ROOT / "data" / "cache"
OUT = REPO_ROOT / "results" / "rotation_both_metrics.csv"
NS, SEEDS, N_TEST = [50, 100, 200, 500, 1000], [0, 1, 2], 2000


def main() -> None:
    align = torch.load(CACHE / "msmarco_minilm_align.pt", weights_only=False)
    test = torch.load(CACHE / "msmarco_minilm_test.pt", weights_only=False)
    idx = torch.randperm(len(test["text"]), generator=torch.Generator().manual_seed(123))[:N_TEST]
    t_emb, t_txt = test["embeddings"][idx], [test["text"][i] for i in idx.tolist()]
    rows = []
    for seed in SEEDS:
        defense = RotatingEncoderDefense(dim=t_emb.shape[1], seed=42 + seed)
        for n in NS:
            perm = torch.randperm(len(align["text"]), generator=torch.Generator().manual_seed(seed))[:n]
            a_emb, a_txt = align["embeddings"][perm], [align["text"][i] for i in perm.tolist()]
            for cond, train, tst in (("unrotated", a_emb, t_emb),
                                     ("same_epoch", defense.transform(a_emb, 0), defense.transform(t_emb, 0)),
                                     ("next_epoch", defense.transform(a_emb, 0), defense.transform(t_emb, 1))):
                torch.manual_seed(seed)
                atk = LinearProbeAttacker(embedding_dim=t_emb.shape[1])
                atk.fit(train, a_txt)
                rows.append({"seed": seed, "n_align": n, "condition": cond, **full_scores(atk.attack(tst), t_txt)})
                print(rows[-1]["condition"], n, seed, round(rows[-1]["rouge_l_precision"], 3), round(rows[-1]["content_precision"], 3), flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
