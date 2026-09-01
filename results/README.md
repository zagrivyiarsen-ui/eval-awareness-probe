# `results/` — what each file is

Generated inventory. The one file to start from is `FINAL_DUMP_EN.md`; the one file that makes every aggregate checkable is `records_full.csv`.

## Primary data — everything else is derived from these

| file | KB | what it is |
|---|---|---|
| `ask_model.jsonl` | 161 | Raw answers to the three explicit questions asked of the model at T=0 with an empty think prefill. |
| `new_prompt_records.jsonl` | 156 | The 137 additional records. Note `input_ids` is null here — see docs/DEVIATIONS.md, item 9. |
| `nll.csv` | 20 | Per-record negative log-likelihood, used for the text-unfamiliarity control. Covers 209 of 529 records. |
| `numbers.json` | 305 | Every number the project produced, as flat dotted keys. 3744 keys. Namespaces: `gate0.*` extraction validity, `probe.v1`–`v5` the pre-registered validations, `probe.exploratory.*` everything after the pivot, `pilot.*` and `pivot.*` the cancelled behavioural arm. |
| `numbers_history.jsonl` | 942 | Audit trail: one line per write to numbers.json with timestamp, argv and git sha. 204 entries. Lets a reader see when each number was recorded and by which command. |
| `p11_reparsed.json` | 5 | The same generations re-classified with a wider rule that catches long-form negations. Quote yes-rates from here, not from numbers.json — see docs/DEVIATIONS.md, item 8. |
| `p11_samples.json` | 73 | All 512 generations from the steering experiment, one per prompt per configuration, so the yes/no classification can be re-checked by hand. |
| `prompt_records.jsonl` | 2520 | The 384 pre-registered prompt records with token ids, user spans, template and config hashes. |
| `records_full.csv` | 103 | Per-record export, 529 rows x 32 columns: scores under four direction constructions and three read positions, firing flags at two thresholds, lengths, NLL, last-token ids, the model's own answers. **Every aggregate in the write-up is recomputable from this file alone.** |
| `records_schema.json` | 2 | Column dictionary for records_full.csv: what each column is, which direction and threshold it refers to, and its caveats. |
| `records_texts.jsonl` | 983 | The prompt text of each record, keyed the same way. |

## Reports and dumps

| file | KB | what it is |
|---|---|---|
| `DATA_NOT_INCLUDED.md` | 1 | Artefacts too large for git — activations, the raw Dolly dump, pilot generations — with their sizes and sha256 so a reader can verify they hold the same files. |
| `FINAL_DUMP_EN.md` | 101 | The complete numeric dump for independent checking: field completeness, stored aggregates reconciled against recomputation from records_full.csv, every table, and an explicit list of numbers declared invalid and why. Start here if you want to verify a claim. |
| `MANIFEST.json` | 18 | sha256 of every file committed to this repository. |
| `p11_report.md` | 5 | Stage P11: the steering experiment with its two controls and the manipulation check. |
| `p2_report.md` | 6 | Stage P2: the main cell analysis. |
| `p4_report.md` | 2 | Stage P4: story-matched contrasts, precision by origin of the negatives, LR/LDA comparison, item-level 'ask the model'. |
| `p5_report.md` | 2 | Stage P5: alternative direction constructions and split-half stability. |
| `p7_report.md` | 4 | Stage P7: the direction refitted on a pooled positive class, and leave-one-insertion-out. |
| `probe_extra.md` | 1 | Short notes from the additional-cell extraction (G1). |
| `qualitative_prompts.json` | 1 | The seed and the keys of the prompts selected above, so the sample can be reproduced. |
| `qualitative_prompts.md` | 10 | A seeded random sample of raw prompts — the training contrast pair, a story triple, a firing and a non-firing external prompt. Selected by seed, not hand-picked. |
| `structure_density.md` | 1 | Markup density of the pre-registered conditions at matched lengths, for the structural control. |

## Frozen inputs and auxiliary tables

| file | KB | what it is |
|---|---|---|
| `dolly_for_blind_coding.md` | 17 | The blind-coding package for the external negatives: the prompts and the frozen coding rule. The labels were never collected, so the category they would test is not claimed. |
| `dolly_labels_TEMPLATE.json` | 1 | Empty answer template for the blind coding above. |
| `env_freeze.txt` | 18 | Frozen package versions of the session that produced the activations. |
| `lexicon_a2.json` | 2 | The registered evaluation lexicon that was removed from the a2 prompts, recorded so the substitution can be audited. |
| `p4_frozen.json` | 42 | Frozen artefact of stage P4: the story map linking the matched a1/a2/a_head prompts, and the legacy reclassification. Frozen so later stages cannot silently redefine the groups. |
| `precision_curve.csv` | 0 | Precision against prevalence, tabulated. |
| `precision_surface.csv` | 3 | Precision over the grid of prevalence and false-positive rate. |
| `prereg_renders.txt` | 20 | Rendered prompts for each pre-registered condition, captured before the runs, so the conditions can be inspected as the model saw them. |

## Remnants of the behavioural arm, cancelled by pivot 4.2

| file | KB | what it is |
|---|---|---|
| `pilot_batches.jsonl` | 7 | Batch composition of the behavioural pilot. |
| `pilot_batches_T07_archived.jsonl` | 5 | The same at T=0.7, archived when the frozen temperature rule fired and the pilot was re-run at T=1.0. |
| `pilot_s_parse.jsonl` | 4 | Parsed pilot outcomes. |
| `pilot_s_parse_T07_archived.jsonl` | 4 | The same for the archived T=0.7 run. |

## Figures

14 figures in `figures/`, each described in [`figures/INDEX.md`](figures/INDEX.md).
