# Pipeline: what ran, in what order, and what it produced

The project ran on Kaggle (2×T4). Each stage was a self-contained notebook cell that wrote the
scripts in `scripts/` to disk and executed them; those cells are not included here because they
embedded a verbatim copy of `src/` and `scripts/` and would only duplicate the code below.
Every stage appends to `results/numbers.json` through `record_numbers`, which also writes an
audit line to `results/numbers_history.jsonl` (timestamp, argv, git sha) — 204 entries in the
final snapshot.

## Behavioural arm — cancelled by pivot 4.2

| stage | script | produced |
|---|---|---|
| **B0 · Gate 0** | `scripts/gate0_real_model.py`, `gate0_fp16_checks.py`, `gate0_kaggle.py` | `gate0.*` — model load, dtype decision (fp16 accepted with a recorded deviation), throughput, and the extraction-validity checks: batched vs single, hooks vs `hidden_states`, bitwise reproducibility, and a naive-index negative control that returns cos 0.22 |
| **B1 · probe stage** | `scripts/probe_stage.py` | `prompt_records.jsonl` (384), `probe_acts_fp16.pt`, and the pre-registered validations `probe.v1`–`probe.v5` under the frozen layer/position rule |
| **B2 · pilot A** | `scripts/pilot.py` | `pilot.*` — behavioural pilot, first at T = 0.7, then re-run at T = 1.0 under the frozen temperature rule |
| **B3 · Gate 1** | `scripts/gate1_checks.py` | `pilot.decision.*` — the budget rule returned `fits = false` (14.19 h against 14.0) |
| **B3b · PC escalation** | `scripts/pilot.py` | the escalated positive control, dead in both environments |
| **B4, B5 · main generation** | `scripts/main_gen.py` | **never run** — cancelled by pivot 4.2 |

Two independent stop signals fired on the same day, the arm was closed under the recorded rule,
and the two behavioural confirmatory tests were declared void.

## Probe arm

| stage | script | produced |
|---|---|---|
| **G1 · new cells** | `scripts/g1_extract.py` | `new_prompt_records.jsonl` (137), `probe_acts_extra_fp16.pt`, frozen prompt artefact sha |
| **P2 · main analysis** | `scripts/p2_analysis.py` | `probe.exploratory.p2.*` (1549 keys): per-cell bands, read positions, layers, precision, LOIO, cross-environment transfer, the alternative-explanation checks (norm, anisotropy, length, NLL, structure) |
| **P3 · repair** | `scripts/p3_repair.py` | the leading fully-out-of-sample procedure — leave-one-decision-point-out, both direction and threshold recomputed on each of the 32 folds — for all 15 cells |
| **P4 · blocks B1–B9** | `scripts/p4_analysis.py` | story-matched contrasts, bands for the remaining cells, precision by origin of the negatives, LR/LDA comparison, item-level "ask the model", length and NLL |
| **P5 · alternative directions** | `scripts/p5_lda_and_b3.py` | shrinkage LDA, LinearSVC, PCA→LDA, and the blind-coding package |
| **P5b · direction stability** | `scripts/p5b_stability.py` | split-half stability of each direction, effective rank of the deployment covariance, and the firing sets under the different directions |
| **P6 · external set** | `scripts/p6_wild_scale.py` | **attempted and stopped**: the composition rule for the Dolly prompts was not preserved and could not be fully reconstructed (35 of 37 recovered); see `docs/DEVIATIONS.md` |
| **P7 · pooled positive class** | `scripts/p7_pooled_direction.py` | the direction refitted on all insertion wordings, and leave-one-insertion-out |
| **P8, P12 · figures** | `scripts/p8_figures.py`, `p12_extra_figures.py` | `results/figures/` |
| **P9 · qualitative sample** | `scripts/p9_qualitative.py` | `results/qualitative_prompts.md` — seeded random prompts, not hand-picked |
| **P10 · audit** | `scripts/p10_audit.py` | units, the two FPR procedures, exact-test floors |
| **P11 · steering** | `scripts/p11_steer.py`, `p11_reparse.py`, `p11_view.py` | the causal test with the shuffled-label and random controls |
| **export / dump** | `scripts/export_records.py`, `final_dump_en.py` | `records_full.csv`, `records_texts.jsonl`, `FINAL_DUMP_EN.md` |

## Reproducing without a GPU

Everything from P7 onward, plus the export and the dump, runs on CPU from `results/` in minutes.
Activation extraction (B1, G1) and steering (P11) need the model; the tensors themselves are
listed with their sha256 in `results/DATA_NOT_INCLUDED.md`.
