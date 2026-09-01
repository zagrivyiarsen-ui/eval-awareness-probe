# Evaluation-awareness probing: a linear probe with held-out AUC 1.000 is not a monitor

Model Qwen/Qwen3.5-9B · layer 21 ·
position `last` · dtype float16 · direction: difference of means

A linear probe fitted on the standard minimal contrast — a prompt that announces an evaluation
against the same prompt without that line — separates its training classes perfectly (held-out AUC
1.000) and is useless as a monitor: precision 0.035–0.053 at 1 % prevalence on negatives it never
saw, against a false-positive rate of 0.000 on the prompts it was calibrated on. Any short foreign
prompt sits at roughly 0.6 of the entire training shift regardless of content, and every negative
cell lies inside the band of directions refitted on shuffled labels — so the offset belongs to the
geometry of the subspace the contrast spans, not to what the direction learned. The threshold that
separates the training classes perfectly lands in the middle of everything else. The contrast the
direction was fitted on was never evaluation-versus-not.

Full write-up: submitted separately. Complete numeric dump for independent checking:
[`results/FINAL_DUMP_EN.md`](results/FINAL_DUMP_EN.md).

## Headline numbers

| | |
|---|---|
| held-out AUC (E1 vs deployment) | 1.000 |
| TPR at threshold (leave-one-out) | 0.906 — ≈0.95 **by construction** |
| FPR, in-distribution deployment (n = 32) | 0.000  [0.000; 0.107] |
| FPR, topical prompts (n = 30) | 0.200 in-sample · 0.214 fully out-of-sample |
| FPR, external Dolly (n = 37) | 0.162  [0.08; 0.31] |
| precision at 1 % prevalence | 0.035–0.053 |
| records | 529 |
| keys in `numbers.json` | 4579 |

## Reproducing the analysis

Everything except activation extraction runs on CPU in minutes:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
PYTHONPATH=. python3 scripts/export_records.py --out results   # rebuilds records_full.csv
PYTHONPATH=. python3 scripts/p8_figures.py     --out results   # figures 1-7
PYTHONPATH=. python3 scripts/p12_extra_figures.py --out results # figures 8-15
PYTHONPATH=. python3 scripts/p10_audit.py      --out results   # units / procedures / p-floors
PYTHONPATH=. python3 scripts/final_dump_en.py  --out results --md results/FINAL_DUMP_EN.md
```

Every **cell-level** aggregate in §4 of the write-up is recomputable from
`results/records_full.csv` alone, and the reconciliation of stored against recomputed values
is section 2 of the dump. The steering results, the within-cell AUCs against the model's own
answers and the leave-one-decision-point-out false-positive rates are not recomputable from
that file; they come from `results/p11_reparsed.json`, `results/ask_model.jsonl` and
`results/numbers.json`.

Stages that need a GPU (activation extraction, steering) are in `scripts/probe_stage.py`,
`scripts/g1_extract.py` and `scripts/p11_steer.py`.

## Layout

```
src/          extraction, prompt contracts, scoring, pre-registration harness
scripts/      analysis stages: probe_stage, p2, p7-p12, dump generators
configs/      conditions.yaml, analysis.yaml — the pre-registered configuration
data/         decision points and topic prompts
docs/         pre-registration addendum, external review, research log, sourced baselines
tests/        unit tests for extraction, contracts, prompts, scoring
results/      numbers.json (every number), records_full.csv (per-record export),
              reports, figures, MANIFEST.json (sha256 of every file)
```

## What is not in this repository

Activation tensors and the raw Dolly dump are too large for git; see
[`results/DATA_NOT_INCLUDED.md`](results/DATA_NOT_INCLUDED.md) for their sizes and sha256, so
that a reader can verify they have the same files. Nothing in the write-up depends on them:
every cell-level aggregate is recomputable from the per-record export.

## Status

Everything after the behavioural arm was closed (pivot 4.2) is **exploratory**; the only
surviving confirmatory family is the `S − Dstr` pair with Holm correction. Deviations from the
pre-registration are listed in the write-up and in `docs/`.

## A note on language

The write-up, this README, `results/FINAL_DUMP_EN.md`, `docs/DEVIATIONS.md` and the figure
captions are in English. Code comments, the pre-registration addendum, the research log and some
intermediate stage reports are in Russian, the language the project was carried out in. Nothing
required to check a number is Russian-only: `FINAL_DUMP_EN.md` covers every reported quantity,
and `results/records_full.csv` lets every cell-level aggregate be recomputed independently.
