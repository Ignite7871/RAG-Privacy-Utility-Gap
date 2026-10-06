"""AdvEnc: adversarial encoder training defense.

Min-max training between a frozen-per-step decoder Dphi (the "attacker" during
training) and a sentence-transformers encoder Etheta being fine-tuned to resist it,
while still producing useful retrieval embeddings:

    decoder step:  L_dec = BCE( Dphi(sg[Etheta(x)]), t )
    encoder step:  L_enc = lambda_ret * L_InfoNCE  -  lambda_priv * L_dec

sg[.] is stop-gradient. The decoder step trains Dphi to invert the encoder's CURRENT
(frozen) embeddings; the encoder step then fine-tunes Etheta to keep retrieval quality
(InfoNCE) while maximizing the (now-frozen) decoder's reconstruction loss, i.e.
degrading its own invertibility against the decoder it just trained. This alternates
every batch: decoder step first (train the attacker against the current encoder), then
encoder step (train the encoder against the just-updated attacker). The comments in
train() show how stop-gradient and the frozen-but-differentiable decoder are implemented;
the implementation notes at the bottom of this file list the fixed hyperparameters and
design choices.

x is always the passage/document side of a pair (the content actually stored in a RAG
vector DB, and so the thing an inversion attacker would target) -- not the query.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from attackers.vocab_reconstruction import (  # noqa: E402
    TOKENIZER_NAME,
    VOCAB_SIZE,
    build_presence_targets,
    texts_to_token_ids,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# AdvEnc-v1 (query-passage pairs) and AdvEnc-v2 (self-pairs). lambda_ret, the learning
# rates and the InfoNCE temperature are fixed implementation defaults, not tuned (see the
# implementation notes at the bottom of this file).
LAMBDA_RET = 1.0
ADV_ENC_CONFIGS: dict[str, dict] = {
    "v1": {"lambda_priv": 0.3, "epochs": 3, "pair_mode": "query_passage"},
    "v2": {"lambda_priv": 2.0, "epochs": 5, "pair_mode": "self"},
    # Ablation variants (gpu10k only): swap one factor at a time relative to v1/v2 to separate pair type from lambda_priv.
    "v1s": {"lambda_priv": 2.0, "epochs": 5, "pair_mode": "query_passage"},
    "v2w": {"lambda_priv": 0.3, "epochs": 3, "pair_mode": "self"},
}

# Training pairs per scale: cpu is the small-scale configuration (1,000 and 1,600 pairs);
# gpu50k is the GPU-scale configuration, 50,000 pairs for both variants.
SCALE_N_PAIRS: dict[str, dict[str, int]] = {
    "cpu": {"v1": 1000, "v2": 1600},
    "gpu50k": {"v1": 50_000, "v2": 50_000},
    # Reduced scale for the second-encoder check (10,000 pairs; see SCALE_TRAINING_OVERRIDES).
    "gpu10k": {"v1": 10_000, "v2": 10_000, "v1s": 10_000, "v2w": 10_000},
    # Same data as gpu10k at batch size 8: MPNet self-pair training (v2) exceeds 8 GB at batch 16.
    "gpu10k_b8": {"v1": 10_000, "v2": 10_000},
    # Same as gpu10k_b8 with bf16 autocast, used only if the float32 batch-8 run does not fit in 8 GB.
    "gpu10k_b8_bf16": {"v1": 10_000, "v2": 10_000},
}

# GPU-scale training runs 20 epochs, which overrides the per-variant epoch counts above;
# CPU scale keeps those.
#
# batch_size is 32 at gpu50k, set by the memory of the card, not by tuning: profiling
# (experiments/profile_advenc_step.py) showed batch=128 pushing peak_reserved CUDA memory
# to ~14.7GB against an 8.2GB RTX 4060 Laptop card -- an overflow into the slow Windows
# shared-memory fallback.
SCALE_TRAINING_OVERRIDES: dict[str, dict] = {
    "gpu50k": {"epochs": 20, "batch_size": 32},
    # Reduced scale: 10 epochs, batch 16 so the larger second encoder (mpnet) fits in 8 GB.
    "gpu10k": {"epochs": 10, "batch_size": 16},
    "gpu10k_b8": {"epochs": 10, "batch_size": 8},
    "gpu10k_b8_bf16": {"epochs": 10, "batch_size": 8, "bf16": True},
}


class _Decoder(nn.Module):
    """Dphi: 2-layer MLP, d -> hidden -> V, LayerNorm + GELU. No dropout."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 512, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    """In-batch-negatives InfoNCE: for each anchor_i, positive_i is the target class
    among all positives in the batch (standard SimCSE/SBERT-style contrastive loss).
    """
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    logits = anchor @ positive.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


class AdvEncTrainer:
    def __init__(
        self,
        encoder_model_name: str,
        decoder_hidden_dim: int = 512,
        lr_encoder: float = 2e-5,
        lr_decoder: float = 1e-3,
        temperature: float = 0.05,
        batch_size: int = 16,
        device: str | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        # amp_dtype=torch.bfloat16 runs the encoder forward/backward under autocast (same batch, same in-batch negatives,
        # same decoder/encoder alternation); the decoder and all losses stay in float32. Default None = plain float32.
        self.amp_dtype = amp_dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(encoder_model_name).to(self.device)
        embedding_dim = self.encoder.get_embedding_dimension()

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.decoder = _Decoder(embedding_dim, decoder_hidden_dim, VOCAB_SIZE).to(self.device)

        self.lr_encoder = lr_encoder
        self.lr_decoder = lr_decoder
        self.temperature = temperature
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Grad-enabled forward pass through the SentenceTransformer pipeline,
        bypassing .encode() (which runs no_grad/eval internally). Assumes
        self.encoder is in train() mode -- see train() below. Two separate calls to
        this method on the *same* text (AdvEnc-v2's self-pairs) give two
        independently-dropout-masked embeddings, which is exactly the "different
        dropout" positive-pair trick; if the underlying model has no active dropout,
        both calls are deterministic and the pair degenerates to a literal duplicate
        automatically, with no special-casing.
        """
        features = self.encoder.preprocess(texts)
        # preprocess() (sentence-transformers >=5.x) includes non-tensor bookkeeping
        # keys alongside the input tensors (e.g. "modality": "text" for its
        # multimodal dispatch) -- only move actual tensors to device.
        features = {k: v.to(self.device) for k, v in features.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            out = self.encoder(features)["sentence_embedding"]
        return out.float()

    def train(
        self,
        pairs: list[tuple[str, str]],
        epochs: int,
        lambda_ret: float = LAMBDA_RET,
        lambda_priv: float = 1.0,
        progress_every: int = 0,
    ) -> dict[str, list[float]]:
        """pairs: (anchor, positive) text pairs. positive is always the passage/document
        side -- the decoder reconstructs it, and it's the second InfoNCE argument.
        For v1 (query_passage), anchor=query, positive=relevant passage. For v2
        (self), anchor=positive=the same passage (see _encode's docstring for how
        that still yields two distinct embeddings).

        progress_every: if >0, print elapsed time and running-average seconds/step
        every this many batches, within each epoch (train() otherwise only prints
        once per epoch, which for large pair counts can leave a long silent gap with
        no way to tell slow-but-progressing apart from stalled).
        """
        self.encoder.train()
        self.decoder.train()

        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr_encoder)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr_decoder)
        bce_loss = nn.BCEWithLogitsLoss()

        log: dict[str, list[float]] = {
            "encoder_loss": [], "decoder_loss": [], "retrieval_loss": [], "adv_decoder_loss": [],
        }

        n = len(pairs)
        total_batches_per_epoch = (n + self.batch_size - 1) // self.batch_size
        train_t0 = time.perf_counter()
        for epoch in range(epochs):
            random.shuffle(pairs)
            totals = {"encoder_loss": 0.0, "decoder_loss": 0.0, "retrieval_loss": 0.0, "adv_decoder_loss": 0.0}
            num_batches = 0
            epoch_t0 = time.perf_counter()

            for start in range(0, n, self.batch_size):
                batch = pairs[start : start + self.batch_size]
                anchor_texts = [a for a, _ in batch]
                positive_texts = [p for _, p in batch]

                target = build_presence_targets(
                    texts_to_token_ids(self.decoder_tokenizer, positive_texts)
                ).to(self.device)

                # ---- decoder step: L_dec = BCE(Dphi(sg[Etheta(x)]), t) ----
                # A dedicated no_grad forward pass computes sg[Etheta(x)], rather than
                # reusing a tensor from one grad-enabled forward shared by both steps.
                # Sharing it (positive_emb.detach()) would save a forward pass but keep
                # the decoder step's and the encoder step's backward graphs alive at the
                # same time, roughly doubling peak activation memory for the step; at
                # batch=128 peak_reserved reached ~14.7GB on an 8.2GB card, where Windows
                # spills into slow shared system memory and per-step throughput becomes
                # erratic (experiments/profile_advenc_step.py). Recomputing here frees
                # this block's activations (no_grad -> nothing retained) before the
                # encoder step's forward passes begin, at the cost of one extra forward
                # pass per training step.
                decoder_optimizer.zero_grad()
                with torch.no_grad():
                    positive_emb_for_decoder = self._encode(positive_texts)
                dec_logits = self.decoder(positive_emb_for_decoder)
                dec_loss = bce_loss(dec_logits, target)
                dec_loss.backward()
                decoder_optimizer.step()
                del positive_emb_for_decoder

                # ---- encoder step: L_enc = lambda_ret * InfoNCE - lambda_priv * L_dec ----
                # Freeze the decoder's weights (no .grad accumulates on them, and
                # autograd skips building their backward nodes) but leave it in the
                # graph, so gradient still flows *through* it into positive_emb and
                # on into the encoder. Must be restored to requires_grad=True before
                # the next iteration's decoder step, or that step's backward() would
                # silently produce no gradient for the decoder and decoder_optimizer
                # .step() would become a no-op for the rest of training.
                for p in self.decoder.parameters():
                    p.requires_grad_(False)

                encoder_optimizer.zero_grad()
                anchor_emb = self._encode(anchor_texts)
                positive_emb = self._encode(positive_texts)
                retrieval_loss = info_nce_loss(anchor_emb, positive_emb, self.temperature)
                adv_dec_logits = self.decoder(positive_emb)
                adv_dec_loss = bce_loss(adv_dec_logits, target)
                encoder_loss = lambda_ret * retrieval_loss - lambda_priv * adv_dec_loss
                encoder_loss.backward()
                encoder_optimizer.step()

                for p in self.decoder.parameters():
                    p.requires_grad_(True)

                totals["encoder_loss"] += encoder_loss.item()
                totals["decoder_loss"] += dec_loss.item()
                totals["retrieval_loss"] += retrieval_loss.item()
                totals["adv_decoder_loss"] += adv_dec_loss.item()
                num_batches += 1

                if progress_every and num_batches % progress_every == 0:
                    elapsed = time.perf_counter() - epoch_t0
                    print(
                        f"  epoch {epoch + 1}/{epochs} step {num_batches}/{total_batches_per_epoch} "
                        f"({elapsed:.1f}s elapsed, {elapsed / num_batches:.3f}s/step, "
                        f"~{elapsed / num_batches * total_batches_per_epoch:.0f}s/epoch projected)",
                        flush=True,
                    )

            for key in log:
                log[key].append(totals[key] / num_batches)
            epoch_elapsed = time.perf_counter() - epoch_t0
            total_elapsed = time.perf_counter() - train_t0
            print(
                f"epoch {epoch + 1}/{epochs}: "
                f"encoder_loss={log['encoder_loss'][-1]:.4f} "
                f"decoder_loss={log['decoder_loss'][-1]:.4f} "
                f"retrieval_loss={log['retrieval_loss'][-1]:.4f} "
                f"adv_decoder_loss={log['adv_decoder_loss'][-1]:.4f} "
                f"| epoch_time={epoch_elapsed:.1f}s total_elapsed={total_elapsed:.1f}s "
                f"({total_elapsed / 60:.1f} min)",
                flush=True,
            )

        return log

    def save_checkpoint(self, path: Path, variant: str, encoder_name: str, config: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.encoder.state_dict(),
                "variant": variant,
                "encoder_name": encoder_name,
                "config": config,
            },
            path,
        )


def build_query_passage_pairs(n_pairs: int) -> list[tuple[str, str]]:
    """(query, relevant passage) pairs from microsoft/ms_marco v2.1, using each row's
    is_selected flags. One relevant passage per query (some rows have more than one
    selected passage; only the first is used).
    """
    ds = datasets.load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    pairs: list[tuple[str, str]] = []
    for row in ds:
        for text, is_selected in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]):
            if is_selected:
                pairs.append((row["query"], text))
                break
        if len(pairs) >= n_pairs:
            break
    if len(pairs) < n_pairs:
        raise RuntimeError(f"only collected {len(pairs)} query-passage pairs, needed {n_pairs}")
    return pairs[:n_pairs]


def build_self_pairs(n_pairs: int) -> list[tuple[str, str]]:
    """(passage, passage) pairs -- same passage twice; see AdvEncTrainer._encode for
    how this still produces two distinct embeddings via dropout.
    """
    ds = datasets.load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    seen: set[str] = set()
    passages: list[str] = []
    for row in ds:
        for text in row["passages"]["passage_text"]:
            if text not in seen:
                seen.add(text)
                passages.append(text)
            if len(passages) >= n_pairs:
                break
        if len(passages) >= n_pairs:
            break
    if len(passages) < n_pairs:
        raise RuntimeError(f"only collected {len(passages)} passages, needed {n_pairs}")
    return [(p, p) for p in passages[:n_pairs]]


if __name__ == "__main__":
    import argparse

    from data.encode import ENCODERS

    parser = argparse.ArgumentParser(description="Train an AdvEnc-defended sentence encoder.")
    parser.add_argument("--variant", choices=list(ADV_ENC_CONFIGS), required=True)
    parser.add_argument(
        "--scale", choices=list(SCALE_N_PAIRS), required=True,
        help="Training-data scale: cpu (small-scale configuration, 1,000-1,600 pairs), "
             "gpu50k (GPU-scale configuration, 50,000 pairs) or the reduced-scale gpu10k* "
             "variants (10,000 pairs).",
    )
    parser.add_argument(
        "--encoder", choices=list(ENCODERS), required=True,
        help="Base encoder to adversarially fine-tune (see data/encode.py:ENCODERS).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help="Print elapsed time / projected epoch duration every N batches (0 = off, "
             "only print once per epoch). Recommended for gpu50k scale.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = ADV_ENC_CONFIGS[args.variant]
    n_pairs = SCALE_N_PAIRS[args.scale][args.variant]
    overrides = SCALE_TRAINING_OVERRIDES.get(args.scale, {})
    epochs = overrides.get("epochs", config["epochs"])
    batch_size = overrides.get("batch_size", 16)  # AdvEncTrainer's own default
    model_id, _dim = ENCODERS[args.encoder]

    print(
        f"building {n_pairs} {config['pair_mode']} pairs for variant={args.variant} scale={args.scale} "
        f"(epochs={epochs}, batch_size={batch_size})...",
        flush=True,
    )
    pair_build_t0 = time.perf_counter()
    if config["pair_mode"] == "query_passage":
        pairs = build_query_passage_pairs(n_pairs)
    else:
        pairs = build_self_pairs(n_pairs)
    print(f"built {len(pairs)} pairs in {time.perf_counter() - pair_build_t0:.1f}s", flush=True)

    trainer = AdvEncTrainer(
        encoder_model_name=model_id, batch_size=batch_size,
        amp_dtype=torch.bfloat16 if overrides.get("bf16") else None,
    )

    t0 = time.perf_counter()
    log = trainer.train(
        pairs, epochs=epochs, lambda_ret=LAMBDA_RET, lambda_priv=config["lambda_priv"],
        progress_every=args.progress_every,
    )
    wall_time_seconds = time.perf_counter() - t0
    print(f"total wall time: {wall_time_seconds:.1f}s ({wall_time_seconds / 60:.1f} min)")

    run_config = {
        **config,
        "n_pairs": n_pairs,
        "scale": args.scale,
        "epochs": epochs,
        "batch_size": batch_size,
        "wall_time_seconds": wall_time_seconds,
    }
    checkpoint_path = REPO_ROOT / "checkpoints" / f"advenc_{args.variant}_{args.scale}_{args.encoder}.pt"
    trainer.save_checkpoint(checkpoint_path, variant=args.variant, encoder_name=model_id, config=run_config)
    print(f"wrote {checkpoint_path}")

    log_path = REPO_ROOT / "results" / "advenc" / f"{args.variant}_{args.scale}_{args.encoder}_trainlog.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_with_meta = {**log, "wall_time_seconds": wall_time_seconds, "epochs": epochs, "batch_size": batch_size}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_with_meta, f, indent=2)
    print(f"wrote {log_path}")


# Implementation notes
# - Alternation: strict 1:1, decoder step first and then encoder step, every batch.
#   decoder_loss is logged per epoch. Compare it with the trivial marginal-frequency
#   baseline (experiments/advenc_cpu_scale_check.py,
#   results/advenc/cpu_scale_downstream_check.json) before reading a fast-dropping
#   decoder_loss as a sign of a strong inverter: a loss near that baseline means the
#   decoder learned little beyond class-imbalance exploitation.
# - x is the passage/document side. Passage embeddings are what an attacker who obtained
#   the vector store would target, not query embeddings.
# - lambda_ret (1.0), the InfoNCE temperature (0.05), lr_encoder (2e-5, a standard
#   transformer fine-tuning rate) and lr_decoder (1e-3, as for LinearProbeAttacker and
#   MLPAttacker) are fixed implementation defaults at every scale, not tuned. Epochs and
#   batch size per scale are in SCALE_TRAINING_OVERRIDES; cpu scale uses the per-variant
#   epochs and batch_size=16 (AdvEncTrainer's default).
# - {scale} in file names is the training-data scale (cpu 1,000-1,600 pairs, gpu50k 50,000
#   pairs, gpu10k* 10,000 pairs), not the base encoder. Names are
#   advenc_{variant}_{scale}_{encoder}.pt and {variant}_{scale}_{encoder}_trainlog.json,
#   with the encoder as its own component.
# - Dropout: self.encoder.train() is set once at the start of train() and never switched
#   to eval() (v2's positive pairs need dropout). The decoder-step target embeddings are
#   therefore also computed with dropout, for both v1 and v2.
# - is_selected: MS MARCO v2.1 rows can have no selected passage (skipped by
#   build_query_passage_pairs) or several (only the first is used), the natural reading of
#   "relevant passage" (singular).
