"""Feasibility check ONLY -- does not run a full evaluation.

Checks whether the `vec2text` package (pip install vec2text, now in .venv312) can load
the jxm/gtr__nq__32 checkpoint the paper cites and run inference (no training) against
5 real GTR-base test embeddings. Reports: does it load, do reconstructions look like
real text (same failure signature check as the ALGEN checkpoint smoke test -- silent
empty/degenerate output, not an exception), and per-passage inference time to project a
full-eval time budget.

Windows compatibility note: vec2text 0.0.13 unconditionally does `import resource` at
module scope in vec2text/experiments.py (used only inside one function, for setting
core-dump limits via resource.setrlimit -- never called here). `resource` is Unix-only
and doesn't exist on Windows, so the bare import crashes vec2text's own __init__.py
before any of our code runs. Worked around with an in-process stub module (harmless:
we never call the function that would use it) rather than patching the installed
package.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- Windows `resource` module stub, must happen before `import vec2text` ---
_resource_stub = types.ModuleType("resource")
_resource_stub.RLIMIT_CORE = 0
_resource_stub.RLIM_INFINITY = -1
_resource_stub.setrlimit = lambda *a, **k: None
sys.modules["resource"] = _resource_stub

# HuggingFace import must precede torch import (see README.md: CUDA DLL conflicts on Windows)
import datasets  # noqa: E402, F401

import torch  # noqa: E402

N_SAMPLES = 5


def main() -> None:
    t0 = time.time()
    print("importing vec2text...", flush=True)
    import vec2text  # noqa: E402

    print(f"  imported in {time.time() - t0:.1f}s", flush=True)

    print("\nloading jxm/gtr__nq__32 (+ __correct) checkpoint...", flush=True)
    t_load = time.time()
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    print(f"  loaded in {time.time() - t_load:.1f}s", flush=True)

    print("\nloading 5 GTR-base test embeddings from data/cache/msmarco_gtr_test.pt...", flush=True)
    test_data = torch.load(REPO_ROOT / "data" / "cache" / "msmarco_gtr_test.pt", weights_only=False)
    embeddings = test_data["embeddings"][:N_SAMPLES]
    texts = test_data["text"][:N_SAMPLES]

    print(f"\nrunning inference (num_steps=20, sequence_beam_width=4, matching paper's setup)...", flush=True)
    t_inf = time.time()
    predictions = vec2text.invert_embeddings(
        embeddings=embeddings.to("cuda" if torch.cuda.is_available() else "cpu"),
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4,
    )
    inf_time = time.time() - t_inf
    per_passage = inf_time / N_SAMPLES

    print(f"\n  inference done in {inf_time:.1f}s total, {per_passage:.2f}s/passage")
    print("\n" + "=" * 70)
    for i, (src, pred) in enumerate(zip(texts, predictions)):
        print(f"\n[{i}] SOURCE:  {src[:150]}")
        print(f"[{i}] RECON:   {pred[:150]!r}")
        print(f"[{i}] len(recon)={len(pred)} chars, non-empty={bool(pred.strip())}")

    print("\n" + "=" * 70)
    print(f"per-passage inference time: {per_passage:.2f}s")
    for n_test, label in [(50, "50-passage subset"), (200, "200-passage (n_test=1,000 GTR-base setup)")]:
        projected = per_passage * n_test
        print(f"projected time for {label}: {projected:.0f}s ({projected / 60:.1f} min)")

    print(f"\ntotal script time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
