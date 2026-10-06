"""Main-venv (.venv312) entry point for ALGEN experiments.

Runs entirely under .venv312 -- it never imports attackers.algen or
transformers itself. Each (dataset, encoder, n_align) combination is
dispatched as a subprocess to .venv-algen-legacy's python executable running
attackers/run_algen_legacy.py, which is pinned to transformers==4.52.4 for
the ALGEN checkpoint (see README.md, "Environments"). This keeps the
transformers version split entirely at the process boundary.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_PYTHON = REPO_ROOT / ".venv-algen-legacy" / "Scripts" / "python.exe"
RUN_SCRIPT = REPO_ROOT / "attackers" / "run_algen_legacy.py"

DEFAULT_N_ALIGN_SWEEP = [50, 200, 1000, 2000]


def run_one(dataset: str, encoder: str, n_align: int, n_test: int) -> None:
    if not LEGACY_PYTHON.exists():
        raise FileNotFoundError(
            f"{LEGACY_PYTHON} not found -- create .venv-algen-legacy first (see README.md)"
        )

    print(f"=== {dataset}/{encoder} n_align={n_align} n_test={n_test} ===")
    subprocess.run(
        [
            str(LEGACY_PYTHON),
            str(RUN_SCRIPT),
            "--dataset", dataset,
            "--encoder", encoder,
            "--n_align", str(n_align),
            "--n_test", str(n_test),
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALGEN experiments via the .venv-algen-legacy subprocess.")
    parser.add_argument("--dataset", required=True, help='e.g. "msmarco" or "nq"')
    parser.add_argument("--encoder", required=True, help='e.g. "minilm", "mpnet", "gtr", "bge"')
    parser.add_argument(
        "--n_align", type=int, nargs="+", default=DEFAULT_N_ALIGN_SWEEP,
        help=f"one or more alignment-set sizes to sweep (default: {DEFAULT_N_ALIGN_SWEEP})",
    )
    parser.add_argument("--n_test", type=int, default=50)
    args = parser.parse_args()

    for n_align in args.n_align:
        run_one(args.dataset, args.encoder, n_align, args.n_test)
