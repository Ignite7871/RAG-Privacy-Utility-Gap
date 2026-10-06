# Privacy-utility trade-offs of embedding-inversion defenses in RAG

Code, derived results and figure scripts for evaluating embedding-inversion attacks and defenses
(adversarial encoder training, Gaussian noise, keyed rotation) on dense retrieval embeddings.
See `REPRODUCE.md` for how each result in the paper is produced.

## What is in this repository

| Path | Contents |
|---|---|
| `data/encode.py` | Builds the 50,000-passage MS MARCO and NQ pools, splits them (40,000 alignment / 10,000 test, seed 42) and caches encoder embeddings |
| `attackers/` | LinearProbe, MLP head, ALGEN, GEIA, transfer attack, and the metrics (plain and content-word ROUGE-L) |
| `defenses/` | Adversarial encoder training (AdvEnc), Gaussian DP, keyed rotation |
| `experiments/` | One script per experiment, plus resumable queue scripts (`run_revision_queue*.sh`) |
| `results/` | Derived results (CSV/JSON) and plotting scripts |
| `paper/` | Not part of this repository (the manuscript is under review); the figure script writes its output here |

Not included: raw MS MARCO / NQ passages and cached embeddings (`data/cache/`), model checkpoints, and result files that
contain passage text (only text-free `*.metrics.json` versions are kept; regenerate the originals by rerunning the
experiments).

**ALGEN.** `attackers/algen.py` is a thin adapter that imports the ALGEN authors' own code from a local clone; no ALGEN code is
redistributed here (their repository carries no license and states the code is for research purposes). Set it up once:

```
git clone https://github.com/siebeniris/ALGEN third_party/ALGEN
git -C third_party/ALGEN checkout 08b892391075de841991a03c447800ec2356925
```

(or point `ALGEN_SRC` at the clone's `src/` directory), download the authors' `mmarco_english` generator checkpoint from
Zenodo record 15639971 into `checkpoints/algen_generator/`, and run ALGEN only in the legacy environment below.
Please cite Chen, Xu and Bjerva, ACL 2025, when using it.

## Environments

Three virtual environments are needed because two attacks depend on older library versions.

| Environment | Used for | Requirements file |
|---|---|---|
| `.venv312` (Python 3.12, `torch==2.6.0+cu124`) | Everything except the two below | `requirements.txt` |
| `.venv-algen-legacy` (`transformers==4.52.4`) | ALGEN only (`attackers/run_algen_legacy.py`, `experiments/algen_gt_mask_check.py`) | `requirements-algen-legacy.txt` |
| `.venv-vec2text` (`transformers==4.52.4`, `vec2text==0.0.13`) | Vec2Text only (`experiments/run_vec2text.py`) | `requirements-vec2text.txt` |

Why: the released ALGEN checkpoint and `vec2text` 0.0.13 are incompatible with `transformers` 5.x, and the failure is
**silent** (empty or unrelated output, no exception). Do not run either attack in `.venv312`.

Two further notes for Windows: import HuggingFace libraries (`datasets`, `transformers`, `sentence_transformers`) before
`torch`, or CUDA DLL conflicts can occur; and Vec2Text does not run on the cached SentenceTransformer embeddings (it
needs its own embedder and 32-token text, see `experiments/run_vec2text.py`).

### PyTorch

All three requirements files pin `torch==2.6.0+cu124`, the CUDA 12.4 build that the results were produced with. That build is
hosted on the PyTorch package index, **not on PyPI**: with PyPI alone, pip stops with
`No matching distribution found for torch==2.6.0+cu124`. Each requirements file therefore starts with
`--extra-index-url https://download.pytorch.org/whl/cu124`, so `pip install -r requirements.txt` works as is. The wheel bundles
its CUDA 12.4 runtime, so no separate CUDA toolkit is needed, only an NVIDIA driver recent enough for CUDA 12.4. The wheels are
built for 64-bit Python 3.12 (the `cp312` tag).

If you prefer to install PyTorch separately (the version spec `2.6.0` resolves to `2.6.0+cu124` on that index):

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check the install with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, which should print
`2.6.0+cu124 True`. On a machine without a CUDA GPU the same wheel runs on the CPU (`False`), which is enough for the CPU-scale
AdvEnc runs but not practical for the GPU-scale ones. Another CUDA build, or the PyPI CPU wheel (`torch==2.6.0`), should work
if you change the pin and the index line, but the results here were produced and checked only with `2.6.0+cu124`.

## Quick start

```
python -m venv .venv312 && .venv312\Scripts\pip install -r requirements.txt     # also installs torch 2.6.0+cu124 (see PyTorch above)
.venv312\Scripts\python data\encode.py            # downloads MS MARCO and BeIR/NQ, encodes with four encoders
.venv312\Scripts\python experiments\linearprobe_full_scale_both_metrics.py
```

Hardware used: one RTX 4060 Laptop GPU (8 GB). The GPU-scale AdvEnc run needs batch size 32 at this memory size.

## Data and licenses

MS MARCO and Natural Questions/BeIR are downloaded from their official sources by `data/encode.py`; their own terms apply
and no passage text is redistributed here. The code in this repository is released under the MIT License (see `LICENSE`).
