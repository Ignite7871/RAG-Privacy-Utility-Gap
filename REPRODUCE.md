# Reproducing the results

Run from the repository root. "main" is `.venv312`; the three environments are described in `README.md`.
Scripts are resumable where noted. Expected wall times are for one RTX 4060 Laptop GPU.

## 0. Data

| Step | Command (env) | Output |
|---|---|---|
| Encode pools (MS MARCO, NQ; four encoders) | `python data/encode.py` (main) | `data/cache/{corpus}_{encoder}_{align,test}.pt` |
| 32-token caches for ALGEN's native setting | `python experiments/algen_native_prefix_caches.py` (main) | `data/cache/{msmarco32,nq32}_*.pt` |

## 1. Baselines and metric calibration

| Result | Command | Output |
|---|---|---|
| Embedding-free frequent-token baseline | `python experiments/prior_baseline_and_snapshot_linking.py` | `results/prior_floor.csv` |
| Frequent content-word baseline | `python experiments/content_prior_floor.py` | `results/content_prior_floor.csv` |
| LinearProbe at full scale, both metrics | `python experiments/linearprobe_full_scale_both_metrics.py` (about 1.5 h) | `results/linearprobe_full_scale_both_metrics.csv` |
| Cross-domain alignment | `python experiments/cross_domain_alignment.py` (resumable, about 1.3 h) | `results/cross_domain/cross_domain.csv` |
| MLP vs. LinearProbe | `python experiments/mlp_vs_linear_validation.py`, `mlp_higher_dropout_check.py`, `mlp_combined_reg_check.py` | `results/mlp_*.json` |

## 2. Other attackers

The ALGEN rows first need the one-time setup in `README.md` (clone the authors' repository at the pinned commit and download their
checkpoint) and the legacy environment.

| Result | Command | Output |
|---|---|---|
| ALGEN, full passages, 500 test | `bash experiments/run_algen_large_test.sh` then `python experiments/aggregate_algen_n500.py` (legacy env for the first) | `results/algen_n500/summary.csv` |
| ALGEN, native 32-token setting | `bash experiments/run_algen_large_test.sh "msmarco32:minilm ... nq32:bge" "50 200 1000 2000" results/algen_native`, then `python experiments/aggregate_algen_n500.py algen_native` and `python experiments/algen_native_baselines.py` | `results/algen_native/{summary,baselines}.csv` |
| ALGEN attention-mask check | `python experiments/algen_gt_mask_check.py` (legacy env) | `results/algen_native/gt_mask_check.json` |
| ALGEN under transformers 5.x (error, then empty output) | `ALGEN_SRC=... python experiments/algen_transformers5_check.py` (main env) | `results/algen_native/transformers5_check.json` |
| Vec2Text vs. LinearProbe | `python experiments/run_vec2text.py` (vec2text env, about 1.3 h), `python experiments/vec2text_floor.py` (main) | `results/vec2text/summary_steps20_beam4.metrics.json`, `floor_32tok.json` |
| GEIA / transfer attack scoring | `python experiments/score_geia_transfer_both_metrics.py` (after `experiments/run_geia.py`, `run_transfer_attack.py`) | `results/geia_transfer_both_metrics.json` |

## 3. Adversarial encoder training (AdvEnc)

| Result | Command | Output |
|---|---|---|
| Train variants | `python defenses/adv_encoder.py --variant {v1,v2} --scale {cpu,gpu50k} --encoder minilm` (gpu50k: about 2 h) | `checkpoints/advenc_*.pt` (not distributed), `results/advenc/*_trainlog.json` |
| Adaptive attackers x budgets, held-out BCE | `python experiments/advenc_adaptive_budget_sweep.py` (resumable, about 30 min per encoder) | `results/advenc/adaptive_budget_sweep.csv` |
| Non-adaptive attacker | `python experiments/advenc_nonadaptive_both_metrics.py` | `results/advenc/nonadaptive_both_metrics.json` |
| Retrieval, all variants (1,000 queries, bootstrap CI) | `python experiments/advenc_retrieval_all_variants.py` | `results/advenc/retrieval_all_variants.json` |
| Reduced-scale ablation (pair type vs. privacy weight) and second encoder | `python defenses/adv_encoder.py --variant {v1,v2,v1s,v2w} --scale gpu10k --encoder minilm`; `--variant {v1,v2} --scale gpu10k_b8 --encoder mpnet` (plus `--variant v1 --scale gpu10k --encoder mpnet` for batch 16); then `python experiments/advenc_reduced_scale_eval.py` (resumable) | `results/advenc/reduced_scale_eval.csv`, `reduced_scale_retrieval.json` |
| Retrieval sanity check, geometry | `experiments/advenc_similarity_sanity_check.py`, `advenc_gpu50k_diagnostics.py`, `check_fm1_*.py` | `results/advenc/*.json` |

## 4. Gaussian DP and key rotation

| Result | Command | Output |
|---|---|---|
| DP: retrieval and non-adaptive attacker | `python experiments/dp_defense_sweep.py` | `results/dp_defense/dp_sweep_full.csv` |
| DP: adaptive attacker | `python experiments/dp_adaptive_sweep.py` (resumable) | `results/dp_defense/dp_adaptive.csv` |
| Key rotation, 3 seeds, both metrics | `python experiments/rotation_eval_both_metrics.py` | `results/rotation_both_metrics.csv` |
| Snapshot-linking attack | `python experiments/prior_baseline_and_snapshot_linking.py` | `results/rotation_snapshot_linking.csv` |

## 5. Figures and the manuscript

| Item | Command |
|---|---|
| Privacy vs. utility figure | `python results/plot_privacy_utility.py` -> `paper/fig_privacy_utility.{pdf,png}` |
| Text-free result copies for release | `python experiments/strip_text_from_results.py` |
| Derived-value check | `python experiments/audit_derived.py` recomputes the percentages and ranges quoted in the paper from the result files |

Queue scripts `experiments/run_revision_queue{,2,3}.sh` chain the long jobs in the order used for the revision.

## Reproducibility notes

**Batch size of the CPU-scale AdvEnc runs.** The checkpoints and trainlogs of `advenc_v1_cpu_minilm` and `advenc_v2_cpu_minilm`
(made on 2026-07-10) do not record the batch size, and the original command lines are not recoverable: the project was not under
git at the time, the shell history has no training commands, and earlier session transcripts were not kept. The paper therefore
states "most likely 16". The evidence: 16 is the default of `defenses/adv_encoder.py` (`AdvEncTrainer(batch_size=16)`, and the script
comment records that CPU scale uses it); `experiments/advenc_epoch_sweep.py`, run the next day on the same 1,600-pair self-pair pool,
sets `batch_size=16` explicitly; and its first five epochs reproduce the v2 CPU run's decoder-loss trajectory to within 0.74% per
epoch (`results/advenc/v2_cpu_epoch_sweep.json` against `v2_cpu_minilm_trainlog.json`), which is close but not identical. No such
rerun exists for the query-passage run (v1); its batch size rests on the same default. Later runs record their batch size
(`config.batch_size` in each checkpoint and `batch_size` in each trainlog).

**Precision.** Every AdvEnc checkpoint in `checkpoints/` stores float32 tensors only, and the trainer contains no mixed-precision
code path other than the optional `bf16` scale (`gpu10k_b8_bf16`), which is used only if the float32 batch-8 MPNet run does not fit
in 8 GB.

**MPNet training batch size.** The MPNet self-pair variant (v2) ran out of GPU memory at batch size 16 on the 8 GB card, so MPNet v1 and v2 were
trained at batch size 8 (scale `gpu10k_b8`, float32) to keep them comparable; an extra v1 run at batch 16 (`gpu10k`) is also reported. MiniLM runs
used batch 16, so MiniLM and MPNet results are compared in direction only. A bf16-autocast scale (`gpu10k_b8_bf16`) was prepared as a fallback and
was not needed. Gradient accumulation was deliberately not used as a substitute for a larger batch: the InfoNCE loss takes its negatives from
inside each batch, so it is not equivalent.

