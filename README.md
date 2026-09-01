# A probe with held-out AUC 1.000 is not a monitor

Evaluation-awareness probing on **Qwen/Qwen3.5-9B** · layer 21 · read position `last` ·
dtype float16 · direction: difference of means · prefill only · 529 records.

A linear probe fitted on the standard minimal contrast — a prompt that announces an evaluation
against the same prompt without that line — separates its training classes perfectly, held-out
AUC 1.000. On held-out instances of those same prompts its false-positive rate is 0.000. On 37
external prompts it has never seen, its false-positive rate is 0.162 and its precision at 1 %
prevalence is 0.053: about eighteen false alarms for every true one.

Looking at *which* external prompts it fires on turned out to matter more than the rate. Five of
the six are the same semantic class — a professional asked to judge a specific person from measured
performance — and an authored prompt written for the "evaluation of humans" cell lands 0.0009 away
from the top external one, on texts that share no vocabulary and no subject.

Full write-up: submitted separately. Complete numeric dump for independent checking:
[`results/FINAL_DUMP_EN.md`](results/FINAL_DUMP_EN.md).

## Headline numbers

| | |
|---|---|
| held-out AUC (E1 vs deployment) | 1.000 |
| TPR at threshold (leave-one-out) | 0.906 — ≈0.95 **by construction**, not a result |
| FPR, in-distribution deployment (n = 32) | 0.000  [0.000; 0.107] |
| FPR, pre-registered topical (n = 8) | 0.250  [0.07; 0.59] |
| FPR, authored after the result (n = 22) | 0.182  [0.07; 0.39] |
| FPR, external Dolly (n = 37) | 0.162  [0.08; 0.31] |
| topical FPR, fully out-of-sample (n = 30) | 0.214 |
| precision at 1 % prevalence | 0.035–0.053 |
| false positives from one domain | 5 of 6; Fisher one-sided p = 0.00071 |
| records | 529 |
| keys in `results/numbers.json` | 4579 |

## Reproducing

Everything except activation extraction runs on CPU in minutes.

```bash
pip install numpy pandas matplotlib scikit-learn scipy
PYTHONPATH=. python3 scripts/export_records.py     --out results   # rebuilds records_full.csv
PYTHONPATH=. python3 scripts/p8_figures.py         --out results   # figures 1-7
PYTHONPATH=. python3 scripts/p12_extra_figures.py  --out results   # figures 8-15
PYTHONPATH=. python3 scripts/p10_audit.py          --out results   # units / procedures / p-floors
PYTHONPATH=. python3 scripts/final_dump_en.py      --out results --md results/FINAL_DUMP_EN.md
```

Every **cell-level** aggregate in the results section of the write-up is recomputable from
`results/records_full.csv` alone, and section 2 of the dump reconciles the stored values against
that recomputation for 15 of 15 cells. Three other families of number are not in that file and come
from elsewhere in `results/`: the steering experiment (`p11_reparsed.json`), the within-cell AUCs
against the model's own answers (`ask_model.jsonl`), and the leave-one-decision-point-out
false-positive rates (`numbers.json`).

The leading out-of-sample procedure exists in two independent implementations written at different
stages — `scripts/p2_analysis.py` (ten cells) and `scripts/p3_repair.py` (all fifteen) — and they
agree to five decimal places on every cell they share.

Stages that need a GPU (activation extraction, steering) are `scripts/probe_stage.py`,
`scripts/g1_extract.py` and `scripts/p11_steer.py`.

## What the dump does not cover

`results/FINAL_DUMP_EN.md` was generated before the last round of analysis and covers the
cell-level tables. Four quantities quoted in the write-up live only in `results/numbers.json`,
under the keys named here:

| quantity | key prefix |
|---|---|
| hard-negative refit (LOPO and leave-one-family-out) | `probe.exploratory.p2.a5_hard_negative.*` |
| random-direction band | `probe.exploratory.p2.bands.*.random_band` |
| split-half minimum cosines | `probe.exploratory.p5.direction_stability.*.cos_min` |
| threshold sweep, TPR 0.80–0.99 | `probe.exploratory.p2.objections.V2_thresholds` |

**One block in the dump prints empty, and it is a bug rather than an absence.** The section
"firing sets under the different directions" reads keys `fired_sets.*.wild_fired_ids` that
`scripts/p5b_stability.py` does not write; it writes `.ids`, `.n_fired`, `.n_total` and
`.tau_median_loo`. The values are all in `numbers.json` under
`probe.exploratory.p5.fired_sets.*` and `probe.exploratory.p5.fired_overlap.*`, and the write-up
quotes them: diff-of-means fires on six of the 37 external prompts, shrinkage LDA on four, and the
two sets do not overlap at all. The generator was not patched because the code was frozen before
this was found.

**Two results were computed after the code snapshot was frozen** and are therefore not produced by
anything in `scripts/`: the domain clustering of the external false positives, and the extraction
noise floor. Both read only `results/records_full.csv` and `results/records_texts.jsonl`, both are
described in the write-up in enough detail to redo in a few lines, and neither requires rerunning
the model. Five figures were regenerated at the same time by a standalone script for the same
reason; the numbers behind each are in the write-up and in `numbers.json`.

## Layout

```
src/          extraction, prompt contracts, scoring, pre-registration harness
scripts/      analysis stages: probe_stage, p2, p3, p5-p12, dump generators
configs/      conditions.yaml, analysis.yaml — the pre-registered configuration
data/         decision points and topic prompts
docs/         pre-registration addendum, external review, research log, deviations
tests/        unit tests for extraction, contracts, prompts, scoring
results/      numbers.json (every number), records_full.csv (per-record export),
              records_texts.jsonl (the prompt texts), reports, figures, MANIFEST.json
```

## What is not in this repository

Activation tensors and the raw Dolly dump are too large for git; see
[`results/DATA_NOT_INCLUDED.md`](results/DATA_NOT_INCLUDED.md) for sizes and sha256, so a reader
can verify they hold the same files. No cell-level aggregate depends on them.

## Status

Everything after the behavioural arm was closed by a pre-recorded stopping rule (pivot 4.2) is
**exploratory**; the only surviving confirmatory family is the `S − Dstr` pair with Holm
correction. Four of five pre-registered validations failed and the project continued down the
branch recorded in advance for that outcome.

Twenty deviations from the pre-registration are listed and ordered by impact in
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md). A twenty-first was found after that file was frozen and
is recorded in the write-up: the eight pre-registered topical prompts were split post-hoc into two
cells after the pooled 2/8 rate was known. It is material because the smaller half has FPR 0.500
and enters the leading pool; the pooled 0.250 that the write-up quotes does not depend on it.

The extraction checks include a negative control that is capable of failing: a deliberately wrong
token index gives cos 0.22 and rel_l2 3.61 against the true activations. The residual-stream hook
matches `hidden_states[L+1]` at `max_abs_diff = 0.0`, batched extraction matches per-record
extraction under both padding sides, and the cross-session anchor is rel_l2 = 0.002.

## A note on language

The write-up, this README, `results/FINAL_DUMP_EN.md`, `docs/DEVIATIONS.md` and the figure captions
are in English. Code comments, the pre-registration addendum, the research log and some
intermediate stage reports are in Russian, the language the project was carried out in. Nothing
required to check a number is Russian-only: the dump covers every cell-level quantity, and
`results/records_full.csv` lets any of them be recomputed independently.
