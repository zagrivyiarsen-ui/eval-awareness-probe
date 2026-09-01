# Files not included in this repository

Too large for git. Listed with size and sha256 so that a reader can verify they hold the same artefacts. No number in the write-up depends on them: every aggregate is recomputable from `records_full.csv`, and section 2 of `FINAL_DUMP_EN.md` reconciles the stored aggregates against that recomputation.

| file | MB | sha256 (first 16) | what it is |
|---|---|---|---|
| `dolly-15k.jsonl` | 12.5 | `2df9083338b4abd6` | raw Databricks Dolly-15k (public, re-downloadable) |
| `gate0_acts.pt` | 27.0 | `41201cdf0083f5ac` | Gate 0 extraction-validity checks |
| `pilot_gens.jsonl` | 3.7 | `5286b717880d0068` | pilot generations, behavioural arm (closed by pivot 4.2) |
| `pilot_gens_T07_archived.jsonl` | 2.9 | `aeb4a9e9dabcff0f` | archived pilot generations at T=0.7 |
| `probe_acts_extra_fp16.pt` | 19.3 | `ccc04810256fbd16` | same for the 145 additional records |
| `probe_acts_fp16.pt` | 55.1 | `f6983b296e6438e2` | residual-stream activations, 384 pre-registered records, 3 layers x 3 positions |

Dolly-15k is public and re-downloadable; the activations can be regenerated with `scripts/probe_stage.py` and `scripts/g1_extract.py` on the model and revision recorded in `numbers.json`.