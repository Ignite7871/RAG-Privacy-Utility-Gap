"""AdvEnc: adversarial encoder training defense, paper Section V-A.

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
encoder step (train the encoder against the just-updated attacker) -- see "Min-max
loop mechanics" below for exactly how stop-gradient / frozen-but-differentiable are
implemented, and the "Ambiguities flagged for review" section at the bottom of this
file for everything the paper excerpt underspecifies.

x is always the passage/document side of a pair (the content actually stored in a RAG
vector DB, and so the thing an inversion attacker would target) -- not the query. See
"x = query or passage?" below.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see claude.md: CUDA DLL conflicts on Windows)
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

# AdvEnc-v1 (q-p) and AdvEnc-v2 (self), per the paper's stated configs. lambda_ret and
# per-run hyperparameters (lr, batch size, temperature) are NOT given in the paper
# excerpt this was implemented against -- see "Unspecified hyperparameters" below.
LAMBDA_RET = 1.0
ADV_ENC_CONFIGS: dict[str, dict] = {
    "v1": {"lambda_priv": 0.3, "epochs": 3, "pair_mode": "query_passage"},
    "v2": {"lambda_priv": 2.0, "epochs": 5, "pair_mode": "self"},
    # Ablation variants (gpu10k only): swap one factor at a time relative to v1/v2 to separate pair type from lambda_priv.
    "v1s": {"lambda_priv": 2.0, "epochs": 5, "pair_mode": "query_passage"},
    "v2w": {"lambda_priv": 0.3, "epochs": 3, "pair_mode": "self"},
}

# CPU scale matches the paper's stated small-scale config (Table IV); gpu50k is the
# GPU-scale re-evaluation (Table VII), 50,000 pairs for both variants.
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

# GPU-scale epochs are paper-sourced (Section V-H: 20 epochs); overrides the
# per-variant epoch counts above. CPU scale keeps those defaults -- this was
# previously an open question (see "Unspecified hyperparameters" below) and is now
# resolved for gpu50k specifically.
#
# batch_size is 32, NOT the paper's stated 128: profiling (experiments/
# profile_advenc_step.py) showed batch=128 pushing peak_reserved CUDA memory to
# ~14.7GB against this machine's 8.2GB RTX 4060 Laptop card -- a hard overflow into
# slow Windows shared-memory fallback, not a tunable inefficiency. This is a
# documented hardware-driven deviation from Section V-H, not a methodology choice.
SCALE_TRAINING_OVERRIDES: dict[str, dict] = {
    "gpu50k": {"epochs": 20, "batch_size": 32},
    # Reduced scale: 10 epochs, batch 16 so the larger second encoder (mpnet) fits in 8 GB.
    "gpu10k": {"epochs": 10, "batch_size": 16},
    "gpu10k_b8": {"epochs": 10, "batch_size": 8},
    "gpu10k_b8_bf16": {"epochs": 10, "batch_size": 8, "bf16": True},
}


class _Decoder(nn.Module):
    """Dphi: 2-layer MLP, d -> hidden -> V, LayerNorm + GELU. No dropout (none
    mentioned in the paper's description of this component)."""

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
        automatically, matching the paper's stated fallback with no special-casing.
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
                # A dedicated no_grad forward pass for sg[Etheta(x)], NOT a reused
                # tensor from a grad-enabled forward computed once for both steps.
                # An earlier version computed positive_emb once (with grad) and used
                # positive_emb.detach() here to save a forward pass -- but that kept
                # both the decoder step's and encoder step's backward graphs alive
                # simultaneously, effectively doubling peak activation memory for the
                # whole step. Profiling at batch=128 (experiments/profile_advenc_step.py)
                # showed peak_reserved hitting ~14.7GB against an 8.2GB card -- Windows
                # silently spilling into slow shared system memory, not a crash, but
                # devastating and erratic per-step throughput. Recomputing here means
                # this block's activations are freed (no_grad -> nothing retained)
                # before the encoder step's forward passes even begin, at the cost of
                # one extra forward pass per training step.
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

    parser = argparse.ArgumentParser(description="Train an AdvEnc-defended sentence encoder (paper Section V-A).")
    parser.add_argument("--variant", choices=list(ADV_ENC_CONFIGS), required=True)
    parser.add_argument(
        "--scale", choices=list(SCALE_N_PAIRS), required=True,
        help="Training-data scale: cpu (paper's small-scale config, Table IV) or "
             "gpu50k (GPU-scale re-evaluation, 50,000 pairs, Table VII).",
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


# Resolved (was flagged, now settled by user review):
# - Decoder:encoder step ratio stays 1:1 -- matches the paper, don't diverge from what
#   we're reproducing. decoder_loss is logged per-epoch regardless. NOTE: an earlier
#   note here treated a fast-dropping decoder_loss as diagnostic of "FM2 (active
#   facilitation)" -- that read did NOT survive the CPU-scale trivial-baseline check
#   (results/advenc/cpu_scale_downstream_check.json): decoder_loss near the
#   marginal-frequency trivial baseline means the decoder learned little beyond
#   class-imbalance exploitation, not that it became a strong/facilitated inverter.
#   Don't re-read decoder_loss alone as an FM signal without that baseline comparison.
# - x = passage/document side: confirmed correct. The paper's threat model (Section
#   II-A, eavesdropper attacker) is about someone who intercepted the vector store --
#   passage embeddings, not query embeddings.
# - Unspecified hyperparameters (lambda_ret, temperature, lr_encoder, lr_decoder): keep
#   as configurable defaults; disclose the chosen values explicitly in the paper's
#   methodology section as a reproducibility note. epochs is paper-sourced for gpu50k
#   scale (20, Section V-H) -- see SCALE_TRAINING_OVERRIDES. batch_size for gpu50k is
#   32, a documented deviation from the paper's stated 128 due to this machine's 8GB
#   VRAM ceiling (see SCALE_TRAINING_OVERRIDES' comment and the memory-profiling note
#   in train() above) -- disclose this as a hardware limitation, not silently. cpu
#   scale still uses the epochs-per-variant/batch_size=16 defaults below.
# - {scale} in filenames: means training-data scale (cpu ~1000-1600 pairs vs. gpu50k
#   50,000 pairs, matching Table IV vs. Table VII), NOT the base encoder. Filenames are
#   now advenc_{variant}_{scale}_{encoder}.pt / {variant}_{scale}_{encoder}_trainlog.json,
#   with encoder tracked as its own component so this can generalize beyond MiniLM
#   later even though the original FM1-FM3 ablations only used MiniLM.
# - .train() throughout / is_selected handling: no change needed.
# - Naming note for the paper (not a code change): the CPU-scale findings (collapse
#   traced to a centroid/distributional shift rather than rank reduction -- see
#   results/advenc/fm1_*.json) contradict the specific "rank collapse -> easier
#   inversion" mechanism the original draft's "FM1" label was attached to. Don't reuse
#   "FM1" for this in the revised taxonomy -- it's a different mechanism and needs its
#   own name (e.g. "FM1' -- Distributional Drift") so the methodology and results
#   sections don't silently mean different things by the same label.
#
# Still open:
# 1. Decoder/encoder step ratio and ordering: implemented as strict 1:1 alternation,
#    decoder step first then encoder step, every batch (see "Resolved" above -- kept
#    intentionally). Still worth watching decoder_loss in the trainlog for signs the
#    decoder never got competent enough to make L_priv meaningful -- but compare it
#    against a trivial baseline before drawing conclusions (see resolved note above).
# 2. (resolved, see above)
# 3. Unspecified hyperparameters: lambda_ret (defaulted to 1.0), InfoNCE temperature
#    (0.05), lr_encoder (2e-5, standard transformer fine-tuning rate), lr_decoder
#    (1e-3, matching LinearProbeAttacker/MLPAttacker elsewhere in this repo) remain
#    implementation defaults, not paper-sourced values, at both scales. batch_size and
#    epochs are resolved for gpu50k (paper-sourced, see above); cpu scale's batch_size
#    =16 is still just a reasonable-throughput default, not paper-sourced.
# 4. (resolved, see above)
# 5. Dropout/eval mode during training: self.encoder.train() is set once at the start
#    of train() and never toggled to eval() (needed for v2's dropout-based positive
#    pairs, and standard for fine-tuning generally). This means decoder-step target
#    embeddings (positive_emb) are also computed with dropout noise, for both v1 and
#    v2 -- the paper doesn't say whether the decoder's training target should be a
#    deterministic (eval-mode) embedding instead. Likely immaterial but noting it.
# 6. is_selected passages: MS MARCO v2.1 rows occasionally have zero selected
#    passages (unanswerable queries) or more than one; build_query_passage_pairs
#    skips rows with none and takes only the first when there's more than one. Not
#    stated as a requirement, just the natural reading of "relevant passage" (singular).
