"""Memory/timing diagnostic (not a training run): profile exactly 5 AdvEnc training steps
at batch=32 with the decoupled forward pass (see defenses/adv_encoder.py's train()), to
check that peak_reserved CUDA memory stays under the RTX 4060 Laptop's 8188MB card and
that per-step time stabilizes, before the full 50k-pair/20-epoch gpu50k run.

For reference, batch=128 with the anchor/positive embeddings shared across both steps
reached peak_reserved=14690MB (1.8x the physical card) and erratic 3.8-6.5s/step timing
after step 1. The profiled logic mirrors train(): the decoder step recomputes positive_emb
via a fresh no_grad forward pass (freed before the encoder step begins) instead of reusing
a graph-attached tensor.

Every phase is bracketed by torch.cuda.synchronize() + time.perf_counter(), since
unsynchronized CUDA calls return before the GPU work is actually done and would
misattribute time. Phases map to train()'s two blocks: decoder_step (no_grad positive
encode + decoder forward + backward + optimizer.step) and encoder_step (grad-enabled
anchor+positive encode + InfoNCE + decoder-frozen forward + backward + optimizer.step);
each step does its own encoding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on
# Windows). Empirically `datasets` must be imported before `sentence_transformers`
# specifically, or the process crashes with an access violation -- see
# experiments/advenc_cpu_scale_check.py.
import datasets  # noqa: E402, F401

from attackers.vocab_reconstruction import build_presence_targets, texts_to_token_ids  # noqa: E402
from defenses.adv_encoder import AdvEncTrainer, build_self_pairs, info_nce_loss  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

N_STEPS = 5
BATCH_SIZE = 32


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def print_memory_stats(label: str) -> None:
    stats = torch.cuda.memory_stats()
    allocated_mb = stats["allocated_bytes.all.current"] / 1024**2
    reserved_mb = stats["reserved_bytes.all.current"] / 1024**2
    peak_allocated_mb = stats["allocated_bytes.all.peak"] / 1024**2
    peak_reserved_mb = stats["reserved_bytes.all.peak"] / 1024**2
    print(f"\n=== memory stats: {label} ===")
    print(f"allocated={allocated_mb:.0f}MB  reserved={reserved_mb:.0f}MB  (reserved-allocated gap indicates fragmentation)")
    print(f"peak_allocated={peak_allocated_mb:.0f}MB  peak_reserved={peak_reserved_mb:.0f}MB")
    print(f"num_alloc_retries={stats.get('num_alloc_retries')}  num_ooms={stats.get('num_ooms')}")
    print(f"cumulative allocation calls={stats.get('allocation.all.allocated')}  free calls={stats.get('allocation.all.freed')}")
    print(f"\n--- full torch.cuda.memory_summary() ({label}) ---")
    print(torch.cuda.memory_summary(abbreviated=False))


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available -- this diagnostic is only meaningful on GPU.")
        return

    print(f"building {N_STEPS * BATCH_SIZE} self-pairs for the profiling run...", flush=True)
    pairs = build_self_pairs(N_STEPS * BATCH_SIZE)

    trainer = AdvEncTrainer(
        encoder_model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=BATCH_SIZE
    )
    trainer.encoder.train()
    trainer.decoder.train()

    encoder_optimizer = torch.optim.Adam(trainer.encoder.parameters(), lr=trainer.lr_encoder)
    decoder_optimizer = torch.optim.Adam(trainer.decoder.parameters(), lr=trainer.lr_decoder)
    bce_loss = nn.BCEWithLogitsLoss()

    torch.cuda.reset_peak_memory_stats()

    step_timings = []
    for step in range(N_STEPS):
        batch = pairs[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        anchor_texts = [a for a, _ in batch]
        positive_texts = [p for _, p in batch]

        t_start = sync_time()

        target = build_presence_targets(
            texts_to_token_ids(trainer.decoder_tokenizer, positive_texts)
        ).to(trainer.device)

        # ---- decoder step (now includes its own no_grad encode) ----
        decoder_optimizer.zero_grad()
        with torch.no_grad():
            positive_emb_for_decoder = trainer._encode(positive_texts)
        dec_logits = trainer.decoder(positive_emb_for_decoder)
        dec_loss = bce_loss(dec_logits, target)
        dec_loss.backward()
        decoder_optimizer.step()
        del positive_emb_for_decoder
        t_decoder_step = sync_time()

        # ---- encoder step (now includes its own grad-enabled encode) ----
        for p in trainer.decoder.parameters():
            p.requires_grad_(False)

        encoder_optimizer.zero_grad()
        anchor_emb = trainer._encode(anchor_texts)
        positive_emb = trainer._encode(positive_texts)
        retrieval_loss = info_nce_loss(anchor_emb, positive_emb, trainer.temperature)
        adv_dec_logits = trainer.decoder(positive_emb)
        adv_dec_loss = bce_loss(adv_dec_logits, target)
        encoder_loss = 1.0 * retrieval_loss - 1.0 * adv_dec_loss
        encoder_loss.backward()
        encoder_optimizer.step()

        for p in trainer.decoder.parameters():
            p.requires_grad_(True)
        t_encoder_step = sync_time()

        timings = {
            "step": step + 1,
            "decoder_step_s": t_decoder_step - t_start,
            "encoder_step_s": t_encoder_step - t_decoder_step,
            "total_s": t_encoder_step - t_start,
        }
        step_timings.append(timings)
        print(
            f"step {step + 1}/{N_STEPS}: "
            f"decoder_step={timings['decoder_step_s']:.3f}s "
            f"encoder_step={timings['encoder_step_s']:.3f}s "
            f"TOTAL={timings['total_s']:.3f}s",
            flush=True,
        )

        if step == 0:
            print_memory_stats("after step 1")

    print_memory_stats("after step 5")

    print("\n" + "=" * 60)
    print(f"{'step':>5} {'decoder_step_s':>16} {'encoder_step_s':>16} {'total_s':>10}")
    for t in step_timings:
        print(f"{t['step']:>5} {t['decoder_step_s']:>16.3f} {t['encoder_step_s']:>16.3f} {t['total_s']:>10.3f}")

    steps_2_to_5 = step_timings[1:]
    avg_total = sum(t["total_s"] for t in steps_2_to_5) / len(steps_2_to_5)
    print(f"\naverage total_s over steps 2-5: {avg_total:.3f}s/step")


if __name__ == "__main__":
    main()
